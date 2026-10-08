"""Unified pipeline tests use only synthetic rows and locally created weights."""
import importlib.util
import json
from dataclasses import asdict

import pytest
import torch

pytest.importorskip('trl')
pytest.importorskip('peft')
pytest.importorskip('datasets')

from datasets import Dataset
from transformers import (Gemma4TextConfig, Gemma4ForCausalLM, PreTrainedTokenizerFast,
                          BitsAndBytesConfig)
from tokenizers import Tokenizer, models, pre_tokenizers
from trl import SFTConfig, GRPOConfig

from gos_gemma import (attach_gos, assert_gos_only_trainable, frozen_features,
                       save_gos_adapter, load_gos_adapter, selected_logps)
from train_unified import (UnifiedConfig, GoSSFTTrainer, GoSGRPOTrainer,
                           sequential_rollout, exact_answer_reward, load_reward)
from unified_data import (assistant_tokens, mixed_examples, normalize_row, task_key, task_split,
                          token_windows, UnifiedCollator)


torch.set_num_threads(2)


def tokenizer():
    raw = Tokenizer(models.WordLevel({'[PAD]': 0, '[EOS]': 1, '[UNK]': 2,
        'user': 3, 'assistant': 4, 'system': 5, ':': 6, 'a': 7, 'b': 8, 'ok': 9, 'bad': 10}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    result.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return result


def tiny_base():
    config = Gemma4TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, vocab_size_per_layer_input=32, hidden_size_per_layer_input=8,
        layer_types=['sliding_attention', 'full_attention'], sliding_window=32,
        pad_token_id=0, eos_token_id=1, bos_token_id=2)
    return Gemma4ForCausalLM(config)


def test_config_guardrails():
    assert UnifiedConfig().validate().max_seq_length == 4096
    for changes in ({'max_seq_length': 4097}, {'max_prompt_tokens': 4096},
                    {'gradient_accumulation_steps': 3}, {'gpu_memory_fraction': 1.0}):
        with pytest.raises(ValueError):
            UnifiedConfig(**changes).validate()


def test_schema_adapters_and_cross_source_grouping():
    stratos = normalize_row({'system': 'Reason', 'conversations':
        [{'from': 'human', 'value': 'a'}, {'from': 'gpt', 'value': '<think>x</think>ok'}]})
    feedback = normalize_row({'query': ' a ', 'answer': 'ok'})
    fable = normalize_row({'messages': [{'role': 'user', 'content': 'a'},
        {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'x'}, {'type': 'text', 'text': 'ok'}]}]})
    assert task_key(stratos) == task_key(feedback) == task_key(fable)
    assert '<think>' in fable[-1]['content']
    assert task_split(task_key(stratos), 17) == task_split(task_key(feedback), 17)
    counts = {split: sum(task_split(str(index), 17) == split for index in range(10000))
              for split in ('train', 'validation', 'test')}
    assert 7800 < counts['train'] < 8200
    assert 850 < counts['test'] < 1150


def test_assistant_mask_and_explicit_friction():
    messages = normalize_row({'messages': [{'role': 'user', 'content': 'a'},
        {'role': 'assistant', 'content': 'bad'}, {'role': 'tool', 'content': 'error', 'is_error': True},
        {'role': 'assistant', 'content': 'ok'}]})
    ids, labels, friction = assistant_tokens(messages, tokenizer())
    assert labels[0] == -100
    assert friction.count(1.0) == 1
    assert sum(value >= 0 for value in friction) == 1
    assert 9 in labels and 10 in labels
    assert labels[ids.index(7)] == -100  # user task is not a target


def test_long_windows_4096_cap_and_no_double_loss():
    ids = list(range(10000))
    labels = ids.copy()
    windows = list(token_windows(ids, labels, [-100.0]*len(ids), 4096, 256))
    assert max(len(window['input_ids']) for window in windows) == 4096
    supervised = [value for window in windows for value in window['labels'] if value != -100]
    assert supervised == ids[1:]
    with pytest.raises(ValueError):
        list(token_windows(ids, labels, [-100.0]*len(ids), 4097))


def test_lazy_stream_mix_and_holdout(tmp_path):
    sources = []
    for name, weight in [('stratos', 2), ('fable', 1), ('feedback', 1)]:
        path = tmp_path / f'{name}.jsonl'
        path.write_text(''.join(json.dumps({'prompt': f'a {index}', 'answer': 'ok'}) + '\n' for index in range(200)))
        sources.append({'name': name, 'path': str(path), 'weight': weight})
    examples = list(mixed_examples(sources, tokenizer(), 'train', limit=40))
    assert [sum(row['source'] == name for row in examples) for name in ['stratos', 'fable', 'feedback']] == [20, 10, 10]
    assert all(task_split(row['task_key'], 17) == 'train' for row in examples)
    validation = list(mixed_examples(sources, tokenizer(), 'validation', limit=20))
    test = list(mixed_examples(sources, tokenizer(), 'test', limit=20))
    assert not {row['task_key'] for row in examples} & {row['task_key'] for row in validation + test}
    # Creating the generator never accesses a remote source.
    stream = mixed_examples([{'name':'feedback', 'weight':1, 'repo':'does-not-exist'}], tokenizer(), 'train')
    with pytest.raises(ValueError, match='Remote streaming is disabled'):
        next(stream)


def test_peft_freezing_causality_and_roundtrip(tmp_path):
    torch.manual_seed(3)
    base = tiny_base().eval()
    ids = torch.tensor([[7,8,9,1]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        before = base(ids).logits
    model = attach_gos(base, width=16, nodes=3, rounds=2).eval()
    assert assert_gos_only_trainable(model) > 0
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, before)
    hidden = frozen_features(model, ids, mask)
    logps, _ = selected_logps(model, hidden[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1), 2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    (-logps.mean()).backward()
    optimizer.step()
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if 'lora_' not in name)
    with torch.no_grad():
        original = model(ids).logits
        altered_ids = ids.clone(); altered_ids[0, -1] = 10
        changed = model(altered_ids).logits
        torch.testing.assert_close(original[:, :-1], changed[:, :-1])
    save_gos_adapter(model, tmp_path / 'adapter', asdict(UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=2)))
    from safetensors.torch import load_file
    assert all('lora_' in key for key in load_file(tmp_path / 'adapter' / 'adapter_model.safetensors'))
    fresh = tiny_base()
    # Preserve frozen tensors, removing the existing custom wrapper first.
    state = {key.replace('lm_head.base_layer.', 'lm_head.'): value for key, value in base.state_dict().items()
             if 'lora_' not in key}
    fresh.load_state_dict(state)
    restored = load_gos_adapter(fresh, tmp_path / 'adapter').eval()
    with torch.no_grad():
        torch.testing.assert_close(original, restored(ids).logits)


def test_chunked_logps_match_full_gradient():
    torch.manual_seed(4)
    model = attach_gos(tiny_base(), width=16, nodes=3, rounds=2)
    ids = torch.tensor([[7,8,9,1]])
    hidden = frozen_features(model, ids, torch.ones_like(ids))[:, :-1].reshape(-1, 32)
    targets = ids[:, 1:].reshape(-1)
    # Give graph weights a nonzero gradient path through the residual up-projection.
    torch.nn.init.normal_(model.get_output_embeddings().lora_B['default'].weight, std=0.01)
    chosen, entropy = selected_logps(model, hidden, targets, 1, entropy=True)
    loss = -chosen.mean() - entropy.mean() * 0.01
    loss.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    logits = model.get_output_embeddings()(hidden).float()
    logprob = logits.log_softmax(-1)
    full_loss = -logprob.gather(-1, targets[:, None]).mean() + (logprob.exp()*logprob).sum(-1).mean()*0.01
    full_loss.backward()
    torch.testing.assert_close(loss, full_loss)
    for name, p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad, expected[name], rtol=2e-4, atol=2e-6)


def fixture_dataset():
    tok = tokenizer()
    ids, labels, friction = assistant_tokens(normalize_row({'prompt':'a', 'answer':'ok'}), tok)
    data = Dataset.from_list([{'input_ids':ids, 'attention_mask':[1]*len(ids),
        'labels':labels, 'friction_targets':friction}]*4)
    return tok, data


def test_real_trl_sft_optimizer_and_checkpoint(tmp_path):
    tok, data = fixture_dataset()
    model = attach_gos(tiny_base(), width=16, nodes=3, rounds=2)
    config = UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=2, logits_chunk_size=2)
    args = SFTConfig(output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=0.01, use_cpu=True, bf16=False,
        report_to='none', dataset_kwargs={'skip_prepare_dataset':True}, remove_unused_columns=False,
        save_steps=2, prediction_loss_only=True, max_length=128)
    trainer = GoSSFTTrainer(model=model, args=args, train_dataset=data, eval_dataset=data,
        processing_class=tok, data_collator=UnifiedCollator(0), gos_config=config)
    trainer.train()
    assert trainer.state.global_step == 2
    assert (tmp_path / 'checkpoint-2' / 'adapter_model.safetensors').exists()
    assert math_is_finite(trainer.evaluate()['eval_loss'])
    assert model.get_output_embeddings().lora_B['default'].weight.abs().sum() > 0
    assert_gos_only_trainable(model)
    checkpoint = tmp_path / 'checkpoint-2'
    fresh_base = tiny_base()
    frozen_state = {key.replace('lm_head.base_layer.', 'lm_head.'): value
                    for key, value in model.get_base_model().state_dict().items() if 'lora_' not in key}
    fresh_base.load_state_dict(frozen_state)
    restored = load_gos_adapter(fresh_base, checkpoint, trainable=True)
    resumed_args = SFTConfig(output_dir=str(tmp_path), max_steps=3, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=0.01, use_cpu=True, bf16=False,
        report_to='none', dataset_kwargs={'skip_prepare_dataset':True}, remove_unused_columns=False,
        save_strategy='no', prediction_loss_only=True, max_length=128)
    resumed = GoSSFTTrainer(model=restored, args=resumed_args, train_dataset=data,
        processing_class=tok, data_collator=UnifiedCollator(0), gos_config=config)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3


def math_is_finite(value):
    import math
    return math.isfinite(value)


def test_real_trl_grpo_group_and_backward(tmp_path):
    tok = tokenizer(); tok.padding_side = 'left'
    model = attach_gos(tiny_base(), width=16, nodes=3, rounds=2)
    config = UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=2, logits_chunk_size=2,
        max_seq_length=64, max_prompt_tokens=48, max_completion_tokens=4, overlap=4,
        gradient_accumulation_steps=2, grpo_generations=2)
    from datasets import IterableDataset
    def local_examples():
        yield from [{'prompt':'user: a [EOS] assistant: ', 'answer':'ok'}]*4
    data = IterableDataset.from_generator(local_examples)
    # Synthetic rewards test group-relative policy gradients, not code correctness.
    def synthetic_reward(completions, **kwargs):
        return [float(index % 2) for index in range(len(completions))]
    args = GRPOConfig(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=1,
        per_device_eval_batch_size=2, gradient_accumulation_steps=2, generation_batch_size=2,
        num_generations=2, max_completion_length=4, beta=0.0, report_to='none', use_cpu=True,
        bf16=False, use_vllm=False, save_strategy='no', use_cache=False,
        generation_kwargs={'use_cache':False,'logits_to_keep':1})
    trainer = GoSGRPOTrainer(model=model, args=args, train_dataset=data, processing_class=tok,
        reward_funcs=synthetic_reward, rollout_func=sequential_rollout, gos_config=config)
    trainer.train()
    assert trainer.state.global_step == 1
    assert_gos_only_trainable(model)
    assert model.get_output_embeddings().lora_B['default'].weight.abs().sum() > 0
    evaluation = Dataset.from_list([{'prompt':'user: a [EOS] assistant: ', 'answer':'ok'}]*2)
    assert math_is_finite(trainer.evaluate(eval_dataset=evaluation)['eval_loss'])


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('bitsandbytes') is None,
                    reason='NF4 test needs CUDA and bitsandbytes')
def test_real_nf4_gemma_head_training(tmp_path):
    tiny_base().save_pretrained(tmp_path)
    base = Gemma4ForCausalLM.from_pretrained(tmp_path, local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={'':'cuda:0'})
    model = attach_gos(base, width=16, nodes=3, rounds=2)
    assert base.is_loaded_in_4bit
    ids = torch.tensor([[7,8,9,1]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = frozen_features(model, ids, torch.ones_like(ids))
        logps, _ = selected_logps(model, hidden[:, :-1].reshape(-1,32), ids[:,1:].reshape(-1), 2)
    (-logps.mean()).backward()
    assert assert_gos_only_trainable(model) > 0
    assert model.get_output_embeddings().lora_B['default'].weight.grad.abs().sum() > 0


def test_verified_reward_range():
    assert exact_answer_reward(['<think>x</think>42'], ['42']) == [1.0]
    reward = load_reward('train_unified:exact_answer_reward')
    assert reward(completions=['wrong'], answer=['42']) == [0.0]


def test_code_verifier_never_pulls_or_runs_on_host(monkeypatch):
    import types
    import docker_code_reward as verifier
    calls = []
    monkeypatch.setattr(verifier.shutil, 'which', lambda name: '/usr/bin/docker')
    def docker_mock(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=42 if command[1] == 'run' else 0)
    monkeypatch.setattr(verifier.subprocess, 'run', docker_mock)
    assert verifier.code_test_reward(['```python\ndef add(a,b): return a+b\n```'],
                                    tests=['assert add(1,2) == 3']) == [1.0]
    command = next(command for command in calls if command[1] == 'run')
    assert '--pull=never' in command and '--network=none' in command and '--read-only' in command
    assert not any(part in ('-v', '--volume', '--privileged') for part in command)
    with pytest.raises(ValueError, match='tests string'):
        verifier.code_test_reward(['anything'], tests=[None])
