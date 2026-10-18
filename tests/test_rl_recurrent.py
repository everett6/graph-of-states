import torch

from rl_recurrent import PassSchedule, RecurrentGRPOTrainer, chunked_logps
from test_recurrent import tiny
from test_unified import tokenizer


def recurrent_model():
    from experiment_support import configure
    return configure(tiny(), 'recurrent', recurrent={'start': 2, 'end': 4, 'rank': 4, 'lora_rank': 2, 'max_passes': 2})


def test_chunked_logps_match_full_logits():
    model = recurrent_model()
    ids = torch.randint(3, 48, (1, 12))
    with torch.no_grad():
        full = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), -1)
        expected = full.gather(-1, ids[0, 1:, None]).squeeze(-1)[-5:]
        chosen, entropy = chunked_logps(model, ids, 5, chunk=2, entropy=True)
    assert torch.allclose(chosen, expected, atol=1e-5)
    assert torch.allclose(entropy, -(full.exp() * full).sum(-1)[-5:], atol=1e-4)


def test_padded_batches_score_like_unpadded_rows():
    from types import SimpleNamespace
    model = recurrent_model()
    trainer = object.__new__(RecurrentGRPOTrainer)
    trainer.gos_config, trainer.temperature = SimpleNamespace(logits_chunk=3), 1.0
    a, b = torch.tensor([5, 6, 7, 8, 9]), torch.tensor([6, 7, 8, 9, 10, 11])
    # prompts left-padded to 3, completions right-padded to 4
    ids = torch.tensor([[0, 5, 6, 7, 8, 9, 0], [6, 7, 8, 9, 10, 11, 12]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        logps, _, _ = trainer._get_per_token_logps_and_entropies(model, ids, mask, 4)
        alone_a, _ = chunked_logps(model, a[None], 3)
        alone_b, _ = chunked_logps(model, torch.cat([b, torch.tensor([12])])[None], 4)
    assert torch.allclose(logps[0, :3], alone_a, atol=1e-5) and logps[0, 3] == 0
    assert torch.allclose(logps[1], alone_b, atol=1e-5)


def test_one_grpo_step_trains_the_recurrent_block(tmp_path):
    from types import SimpleNamespace
    from datasets import IterableDataset
    from trl import GRPOConfig
    from gos_recurrent import recurrence_parameters, text_model
    from train_unified import sequential_rollout
    tok = tokenizer(); tok.padding_side = 'left'
    model = recurrent_model()
    before = [p.detach().clone() for p in recurrence_parameters(model)]
    config = SimpleNamespace(max_prompt_tokens=48, max_completion_tokens=4, agent_max_tools=0, grpo_use_cache=False,
                             logits_chunk=2)
    data = IterableDataset.from_generator(lambda: iter([{'prompt': 'user: a [EOS] assistant: ', 'answer': 'ok'}] * 4))
    args = GRPOConfig(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=1, learning_rate=1e-2,
        gradient_accumulation_steps=2, generation_batch_size=2, num_generations=2, max_completion_length=4, beta=0.0,
        report_to='none', use_cpu=True, bf16=False, use_vllm=False, save_strategy='no', use_cache=False)
    log = []
    trainer = RecurrentGRPOTrainer(model=model, args=args, train_dataset=data, processing_class=tok,
        reward_funcs=lambda completions, **k: [float(i % 2) for i in range(len(completions))],
        rollout_func=sequential_rollout, gos_config=config, callbacks=[PassSchedule(model, 2, 0, None, log)])
    trainer.train()
    assert trainer.state.global_step == 1 and log and log[0]['passes'] in (1, 2)
    assert text_model(model).recurrence is not None
    assert any(not torch.equal(p, q) for p, q in zip(recurrence_parameters(model), before))


def test_rl_can_continue_from_a_saved_sft_adapter(tmp_path):
    from gos_recurrent import recurrence_parameters, save_recurrence
    from rl_recurrent import load_trainable
    model = recurrent_model()
    with torch.no_grad():
        for parameter in recurrence_parameters(model):
            parameter.normal_(0, 0.1)
    model.save_pretrained(tmp_path / 'adapter')
    save_recurrence(model, tmp_path / 'adapter')
    ids = torch.randint(3, 48, (1, 9))
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    resumed = load_trainable(tiny(), 'recurrent', tmp_path / 'adapter', 2).eval()
    with torch.no_grad():
        assert torch.allclose(resumed(input_ids=ids, use_cache=False).logits, expected, atol=1e-5)
    assert all(p.requires_grad for p in recurrence_parameters(resumed))
    assert any(p.requires_grad for n, p in resumed.named_parameters() if 'lora_' in n)
