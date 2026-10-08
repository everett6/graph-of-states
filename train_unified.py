"""Gemma NF4 + GoS-only SFT/GRPO using PEFT, TRL, and Accelerate.

No data/model sources are accessed by --check-config. Remote reads require
explicit flags. Test partition is opened only by --stage test.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
import importlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

from gos_gemma import (attach_gos, assert_gos_only_trainable, frozen_features, inner_lm,
                       load_gos_adapter, save_gos_adapter, selected_logps)
from unified_data import DEFAULT_SOURCES, UnifiedCollator, mixed_examples


@dataclass
class UnifiedConfig:
    model: str = 'google/gemma-4-12B-it-qat-q4_0-unquantized'
    model_revision: str | None = None
    output_dir: str = 'runs/gemma-gos'
    sources: list[dict] = field(default_factory=lambda: [dict(source) for source in DEFAULT_SOURCES])
    max_seq_length: int = 4096
    max_prompt_tokens: int = 3072
    max_completion_tokens: int = 1024
    overlap: int = 256
    graph_width: int = 128
    graph_nodes: int = 4
    graph_rounds: int = 4
    graph_enhancements: dict = field(default_factory=dict)
    inference_cache: str = 'none'
    kv_quantization: str = 'none'
    kv_kernel: str = 'torch'
    attention_backend: str = 'sdpa'
    checkpoint_backend: str = 'torch'
    checkpoint_cpu_offload: bool = False
    coherence_weight: float = 0.1
    tool_gate_weight: float = 0.1
    h2o_heavy_tokens: int = 256
    h2o_recent_tokens: int = 256
    logits_chunk_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 0.0002
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 100
    eval_samples: int = 128
    grpo_generations: int = 4
    critic_weight: float = 0.1
    gpu_memory_fraction: float = 0.90
    seed: int = 17
    reward_function: str = 'train_unified:exact_answer_reward'

    def validate(self):
        if not 2 <= self.max_seq_length <= 4096:
            raise ValueError('max_seq_length must be 2..4096')
        if min(self.max_prompt_tokens, self.max_completion_tokens) < 1 or self.max_prompt_tokens + self.max_completion_tokens > self.max_seq_length:
            raise ValueError('Prompt + completion budgets must fit inside max_seq_length <= 4096')
        if not 0 <= self.overlap < self.max_seq_length - 1:
            raise ValueError('Invalid overlap')
        if min(self.graph_width, self.graph_nodes, self.graph_rounds, self.logits_chunk_size,
               self.gradient_accumulation_steps, self.max_steps, self.eval_steps,
               self.save_steps, self.eval_samples) < 1 or self.learning_rate <= 0:
            raise ValueError('Training sizes and learning rate must be positive')
        if not 0 < self.gpu_memory_fraction <= 0.95:
            raise ValueError('gpu_memory_fraction must be in (0, 0.95]')
        if self.grpo_generations < 2 or self.gradient_accumulation_steps % self.grpo_generations:
            raise ValueError('Accumulation must be divisible by grpo_generations >= 2')
        if self.save_steps % self.eval_steps:
            raise ValueError('save_steps must be a multiple of eval_steps')
        if self.critic_weight < 0 or not self.sources:
            raise ValueError('Invalid critic weight or empty sources')
        if any(type(source.get('weight')) is not int or source['weight'] < 1 for source in self.sources):
            raise ValueError('Source weights must be positive integers')
        feedback = sum(source['weight'] for source in self.sources if source['name'] == 'feedback')
        reasoning = sum(source['weight'] for source in self.sources if source['name'] != 'feedback')
        if reasoning != 3 * feedback or not feedback:
            raise ValueError('Source weights must preserve 75% reasoning / 25% feedback')
        if len({source['name'] for source in self.sources}) != len(self.sources):
            raise ValueError('Source names must be unique')
        validate_enhancements(self.graph_enhancements, self.graph_rounds)
        if self.inference_cache not in ('none', 'h2o', 'sliding'):
            raise ValueError('inference_cache must be none, h2o or sliding')
        from gos_h2o import H2OConfig
        H2OConfig(self.h2o_heavy_tokens, self.h2o_recent_tokens, self.kv_quantization, self.kv_kernel).validate()
        if self.attention_backend not in ('sdpa', 'eager', 'flash_attention_3'):
            raise ValueError('Invalid attention backend')
        if self.checkpoint_backend not in ('torch', 'deepspeed'):
            raise ValueError('Invalid checkpoint backend')
        if self.checkpoint_cpu_offload and self.checkpoint_backend != 'deepspeed':
            raise ValueError('CPU activation offload requires DeepSpeed')
        if not all(math.isfinite(weight) and weight >= 0 for weight in (self.coherence_weight, self.tool_gate_weight)):
            raise ValueError('Auxiliary weights must be finite and nonnegative')
        return self


def validate_enhancements(options, rounds):
    defaults = dict(scratchpad=False, latent_slots=0, thermal=False, min_rounds=2,
        cross_modal=False, branches=False, async_branches=False, dual_friction=False, tool_gate=False,
        temperature=1.0, temperature_floor=0.5, entropy_tolerance=0.01, motion_tolerance=0.01)
    if not isinstance(options, dict) or set(options) - set(defaults):
        raise ValueError('Unknown graph_enhancements setting')
    settings = defaults | options
    if any(type(settings[key]) is not bool for key in ('scratchpad', 'thermal', 'cross_modal', 'branches', 'async_branches', 'dual_friction', 'tool_gate')):
        raise ValueError('Feature switches must be booleans')
    if settings['async_branches'] and not settings['branches']:
        raise ValueError('async_branches requires branches')
    if type(settings['latent_slots']) is not int or settings['latent_slots'] < 0:
        raise ValueError('latent_slots must be a nonnegative integer')
    if settings['thermal'] and (type(settings['min_rounds']) is not int or not 1 <= settings['min_rounds'] <= rounds):
        raise ValueError('min_rounds must fit graph_rounds')
    import math
    for key in ('temperature', 'temperature_floor', 'entropy_tolerance', 'motion_tolerance'):
        if not isinstance(settings[key], (float, int)) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if settings['temperature'] < settings['temperature_floor']:
        raise ValueError('temperature must be >= temperature_floor')


def exact_answer_reward(completions, answer, **kwargs):
    """A runnable exact-answer baseline; use a verifier plugin for semantic code tests."""
    import re
    def final(text):
        if isinstance(text, list):
            text = '\n'.join(message.get('content', '') for message in text)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        return text
    return [float(final(generated) == final(reference)) for generated, reference in zip(completions, answer)]


def load_reward(path):
    module, name = path.split(':', 1)
    verifier = getattr(importlib.import_module(module), name)
    def verified_reward(completions, **kwargs):
        outcomes = verifier(completions=completions, **kwargs)
        if len(outcomes) != len(completions) or any(not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 <= value <= 1 for value in outcomes):
            raise ValueError('Verifier must return one finite success score in [0, 1] per completion')
        # Measured friction = 1 - verified success. Rewarding its inverse is
        # equivalent to verified success; neural self-ratings are never rewards.
        return [1.0 - (1.0 - float(value)) for value in outcomes]
    return verified_reward


class AdapterSaveMixin:
    def _save(self, output_dir=None, state_dict=None):
        path = Path(output_dir or self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(path, state_dict=state_dict, save_embedding_layers=False)
        self.processing_class.save_pretrained(path)
        (path / 'gos_config.json').write_text(json.dumps(asdict(self.gos_config), indent=2) + '\n')
        torch.save(self.args, path / 'training_args.bin')


class GoSSFTTrainer(AdapterSaveMixin, SFTTrainer):
    """TRL owns optimization/resume; this overrides only the bounded loss path."""
    def __init__(self, *args, gos_config, **kwargs):
        self.gos_config = gos_config
        # TRL's built-in chunked loss reads the bare LM weight, bypassing a
        # custom head. Our overridden loss chunks the actual adapter forward.
        kwargs['args'].loss_type = 'nll'
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        config = self.gos_config
        hidden = frozen_features(model, inputs['input_ids'], inputs['attention_mask'], config.max_seq_length)
        targets = inputs['labels'][:, 1:]
        valid = targets.ne(-100)
        modalities, modality_mask = inputs.get('modal_states'), inputs.get('modal_mask')
        aligned = modalities[:, :-1][valid] if modalities is not None else None
        aligned_mask = modality_mask[:, :-1][valid] if modality_mask is not None else None
        logps, _ = selected_logps(model, hidden[:, :-1][valid], targets[valid], config.logits_chunk_size,
            modal_states=aligned, modal_mask=aligned_mask, checkpoint_backend=config.checkpoint_backend)
        loss = -logps.mean()
        friction = inputs.get('friction_targets')
        if friction is not None and friction.ge(0).any():
            known = friction.ge(0)
            head = inner_lm(model).get_output_embeddings()
            # Few explicit outcome positions, so this auxiliary graph is small.
            risk = head.risk_logits(hidden[known], stream='operational')
            loss = loss + config.critic_weight * F.binary_cross_entropy_with_logits(risk, friction[known])
        head = inner_lm(model).get_output_embeddings()
        tool_targets = inputs.get('tool_targets')
        if tool_targets is not None and tool_targets.ge(0).any():
            known = tool_targets.ge(0)
            loss = loss + config.tool_gate_weight * F.binary_cross_entropy_with_logits(head.tool_logits(hidden[known]), tool_targets[known])
        if aligned is not None and config.coherence_weight:
            graph = head.lora_A[head.active_adapters[0]]
            # Bound this auxiliary pass to the same projection chunk size.
            for start in range(0, len(aligned), config.logits_chunk_size):
                stop = start + config.logits_chunk_size
                from gos_runtime import checkpoint_call
                def coherence(features, states, masks):
                    return (graph.features_and_risk(features, True, states, masks)[2]['coherence_loss'],)
                arguments = (hidden[:, :-1][valid][start:stop], aligned[start:stop], aligned_mask[start:stop])
                value = checkpoint_call(coherence, arguments, config.checkpoint_backend)[0] if torch.is_grad_enabled() else coherence(*arguments)[0]
                weight = aligned_mask[start:stop].sum() / aligned_mask.sum().clamp_min(1)
                loss = loss + config.coherence_weight * value * weight
        return (loss, {'loss': loss}) if return_outputs else loss


class GoSGRPOTrainer(AdapterSaveMixin, GRPOTrainer):
    def __init__(self, *args, gos_config, **kwargs):
        self.gos_config = gos_config
        super().__init__(*args, **kwargs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if rewards.shape[1] != 1 or not torch.isfinite(rewards).all() or not ((rewards >= 0) & (rewards <= 1)).all():
            raise ValueError('GoS critic requires one verified reward function with outcomes in [0, 1]')
        self._verified_failures = 1.0 - rewards[:, 0].detach()
        return rewards

    def _generate_and_score_completions(self, inputs):
        batch = super()._generate_and_score_completions(inputs)
        batch['verified_failures'] = self._verified_failures
        return batch

    def _compute_loss(self, model, inputs):
        policy_loss = super()._compute_loss(model, inputs)
        head = inner_lm(model).get_output_embeddings()
        risk = head.risk_logits(self._risk_summary_features, stream='semantic')
        auxiliary = F.binary_cross_entropy_with_logits(risk, inputs['verified_failures'].to(risk))
        normalizer = self.current_gradient_accumulation_steps if model.training else 1.0
        return policy_loss + self.gos_config.critic_weight * auxiliary / normalizer

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                          logits_to_keep, batch_size=None, compute_entropy=False,
                                          compute_aux_loss=False, **kwargs):
        if compute_aux_loss or any(value is not None for value in kwargs.values()):
            raise ValueError('This memory-bounded GRPO implementation supports text-only inputs')
        logps, entropies, summaries = [], [], []
        # Always score one completion at a time, even when TRL holds a group.
        for start in range(input_ids.shape[0]):
            ids, mask = input_ids[start:start+1], attention_mask[start:start+1]
            hidden = frozen_features(model, ids, mask, self.gos_config.max_seq_length)
            last = (torch.arange(ids.shape[1], device=ids.device) * mask[0]).max()
            summaries.append(hidden[:, last])
            features = hidden[:, :-1][:, -logits_to_keep:]
            targets = ids[:, -logits_to_keep:]
            chosen, entropy = selected_logps(model, features.reshape(-1, features.shape[-1]),
                targets.reshape(-1), self.gos_config.logits_chunk_size, self.temperature, compute_entropy, checkpoint_backend=self.gos_config.checkpoint_backend)
            logps.append(chosen.reshape(1, -1))
            entropies.append(entropy.reshape(1, -1))
        self._risk_summary_features = torch.cat(summaries)
        return torch.cat(logps), torch.cat(entropies) if compute_entropy else None, None


@torch.no_grad()
def sequential_rollout(prompts, trainer):
    """TRL supplies repeated group prompts; generate serially without a KV cache."""
    config = trainer.gos_config
    model = trainer.accelerator.unwrap_model(trainer.model)
    tokenizer = trainer.processing_class
    prompt_ids, completion_ids, logprobs = [], [], []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError('Unified GRPO expects pre-rendered text prompts')
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > config.max_prompt_tokens:
            raise ValueError('GRPO prompt exceeds budget')
        inputs = torch.tensor([ids], device=trainer.accelerator.device)
        mask = torch.ones_like(inputs)
        with trainer.accelerator.autocast():
            generated = model.generate(input_ids=inputs, attention_mask=mask, use_cache=False,
                max_new_tokens=config.max_completion_tokens, do_sample=True,
                temperature=trainer.temperature, top_k=0, top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                logits_to_keep=1)
            completion = generated[:, inputs.shape[1]:]
            chosen, _, _ = trainer._get_per_token_logps_and_entropies(
                model, generated, torch.ones_like(generated), completion.shape[1])
        prompt_ids.append(ids)
        completion_ids.append(completion[0].tolist())
        logprobs.append(chosen[0].tolist())
    return {'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'logprobs': logprobs}


class GuardrailCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f'checkpoint-{state.global_step}'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'gos_config.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        assert_gos_only_trainable(model)


class EpochSeededStream:
    """Picklable lazy generator for Hugging Face IterableDataset."""
    def __init__(self, config, tokenizer, split, mode, allow_remote, limit=None):
        self.config, self.tokenizer, self.split = config, tokenizer, split
        self.mode, self.allow_remote, self.limit = mode, allow_remote, limit

    def __call__(self):
        yield from mixed_examples(self.config.sources, self.tokenizer, self.split, self.config.seed,
            self.mode, self.config.max_seq_length, self.config.overlap, self.config.max_prompt_tokens,
            self.limit, self.allow_remote)


def build_datasets(config, tokenizer, mode, allow_remote, include_train=True, split='validation'):
    from datasets import Dataset, IterableDataset
    train = IterableDataset.from_generator(EpochSeededStream(config, tokenizer, 'train', mode, allow_remote)) if include_train else None
    evaluation = list(EpochSeededStream(config, tokenizer, split, mode, allow_remote, config.eval_samples)())
    if not evaluation:
        raise ValueError(f'No usable {split} examples. Check schemas, source sizes, template and token limits.')
    return train, Dataset.from_list(evaluation)


def load_nf4(config, allow_model_download=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('Production NF4 training requires a CUDA-enabled process')
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.model_revision,
        local_files_only=not allow_model_download, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('Use a fast tokenizer with PAD/EOS and a compatible chat template')
    tokenizer.model_max_length = config.max_seq_length
    nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['lm_head'])
    from gos_runtime import validate_attention_backend
    validate_attention_backend(config.attention_backend)
    base = AutoModelForCausalLM.from_pretrained(config.model, revision=config.model_revision,
        local_files_only=not allow_model_download, quantization_config=nf4,
        dtype=torch.bfloat16, device_map={'': device}, attn_implementation=config.attention_backend)
    if not getattr(base, 'is_loaded_in_4bit', False):
        raise RuntimeError('NF4 loading was not applied')
    base.requires_grad_(False).eval()
    base.config.use_cache = False
    base.config.output_hidden_states = False
    base.config.output_attentions = False
    return base, tokenizer


@torch.no_grad()
def generated_test_metrics(trainer, config, tokenizer, allow_remote_data):
    """Use the untouched test partition for actual generated-answer rewards."""
    _, data = build_datasets(config, tokenizer, 'grpo', allow_remote_data,
                             include_train=False, split='test')
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    verifier = load_reward(config.reward_function)
    predictions = []
    for example in data:
        ids = tokenizer.encode(example['prompt'], add_special_tokens=False)
        inputs = torch.tensor([ids], device=trainer.accelerator.device)
        with trainer.accelerator.autocast():
            cache_stats = None
            if config.inference_cache in ('h2o', 'sliding'):
                from gos_h2o import h2o_generate
                outputs, cache_stats = h2o_generate(model, inputs, config.max_completion_tokens,
                    config.h2o_heavy_tokens, config.h2o_recent_tokens, max_seq_length=config.max_seq_length,
                    kv_quantization=config.kv_quantization, kernel=config.kv_kernel, policy=config.inference_cache)
            else:
                outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=config.max_completion_tokens, do_sample=False, use_cache=False,
                    logits_to_keep=1, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(outputs[0, len(ids):], skip_special_tokens=True)
        reward = verifier(completions=[completion], answer=[example['answer']],
                          tests=[example.get('tests')], prompts=[example['prompt']])[0]
        predictions.append({**example, 'completion': completion, 'verified_reward': reward, 'cache_stats': cache_stats})
    path = Path(config.output_dir) / 'test-predictions.jsonl'
    path.write_text(''.join(json.dumps(example, ensure_ascii=False) + '\n' for example in predictions))
    return {'test_generated_examples': len(predictions),
            'test_mean_reward': sum(example['verified_reward'] for example in predictions) / len(predictions),
            'test_reward_function': config.reward_function, 'test_cache_policy': config.inference_cache}


def run(config, stage, allow_remote_data=False, allow_model_download=False, adapter=None, resume=None):
    from accelerate import PartialState
    from transformers import set_seed
    config.validate()
    if resume:
        previous = json.loads((Path(resume) / 'gos_config.json').read_text())
        for key, value in asdict(config).items():
            if key not in ('output_dir', 'max_steps') and previous.get(key, getattr(UnifiedConfig(), key)) != value:
                raise ValueError(f'Resume configuration changed {key}; preserve source/split/optimizer settings')
    from gos_runtime import configure_checkpointing
    configure_checkpointing(config.checkpoint_backend, config.checkpoint_cpu_offload)
    state = PartialState()
    if state.num_processes != 1:
        raise ValueError('This recipe is restricted to one GPU; launch with --num_processes 1')
    # Reject remote sources before touching model files or constructing a stream.
    if not allow_remote_data and any(not source.get('path') for source in config.sources):
        raise ValueError('Provide local source paths or explicitly set --allow-remote-data at training time')
    output = Path(config.output_dir)
    if stage in ('sft', 'grpo') and output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Training output must be empty unless resuming a checkpoint')
    set_seed(config.seed)
    base, tokenizer = load_nf4(config, allow_model_download)
    if stage in ('grpo', 'test') and not adapter:
        raise ValueError('GRPO/test requires --adapter from an SFT training run')
    if adapter:
        previous = json.loads((Path(adapter) / 'gos_config.json').read_text())
        for key in ('graph_width', 'graph_nodes', 'graph_rounds', 'graph_enhancements', 'model', 'model_revision', 'seed', 'sources'):
            if previous.get(key, {} if key == 'graph_enhancements' else None) != getattr(config, key):
                raise ValueError(f'Adapter and configuration disagree on {key}')
    model = load_gos_adapter(base, adapter, trainable=stage != 'test') if adapter else attach_gos(
        base, config.graph_width, config.graph_nodes, config.graph_rounds, config.graph_enhancements)
    if stage == 'test':
        # Evaluate likelihood with the SFT objective, not GRPO reward loss.
        model.requires_grad_(False)
    mode = 'grpo' if stage == 'grpo' else 'sft'
    train_data, eval_data = build_datasets(config, tokenizer, mode, allow_remote_data,
                                          include_train=stage != 'test', split='test' if stage == 'test' else 'validation')
    common = dict(output_dir=config.output_dir, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_steps, learning_rate=config.learning_rate, bf16=True,
        # Backbone has no autograd graph. Head/projection checkpointing is
        # enforced by selected_logps rather than this backbone-only switch.
        gradient_checkpointing=False, optim='adamw_torch', report_to='none',
        remove_unused_columns=False, dataloader_num_workers=0, seed=config.seed,
        logging_steps=10, eval_strategy='steps' if stage != 'test' else 'no',
        eval_steps=config.eval_steps, save_steps=config.save_steps, save_total_limit=2,
        prediction_loss_only=True, accelerator_config={'dispatch_batches': False})
    callbacks = [GuardrailCallback(config)]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizers = (torch.optim.AdamW(parameters, lr=config.learning_rate), None) if parameters else (None, None)
    if stage in ('sft', 'test'):
        args = SFTConfig(**common, max_length=config.max_seq_length, packing=False,
            dataset_kwargs={'skip_prepare_dataset': True}, load_best_model_at_end=stage != 'test',
            metric_for_best_model='eval_loss', greater_is_better=False)
        trainer = GoSSFTTrainer(model=model, args=args, train_dataset=train_data if train_data is not None else eval_data,
            eval_dataset=eval_data, processing_class=tokenizer, data_collator=UnifiedCollator(tokenizer.pad_token_id),
            callbacks=callbacks, optimizers=optimizers, gos_config=config)
    else:
        tokenizer.padding_side = 'left'
        # TRL requires complete evaluation groups. Actual generation and policy
        # scoring remain serial inside the custom rollout/log-prob methods.
        common['per_device_eval_batch_size'] = config.grpo_generations
        args = GRPOConfig(**common, num_generations=config.grpo_generations,
            max_completion_length=config.max_completion_tokens, beta=0.0,
            use_vllm=False, use_cache=False, loss_type='grpo', temperature=1.0, top_k=0, top_p=1.0,
            generation_batch_size=config.grpo_generations,
            generation_kwargs={'use_cache': False, 'logits_to_keep': 1}, log_completions=False)
        trainer = GoSGRPOTrainer(model=model, args=args, reward_funcs=load_reward(config.reward_function),
            train_dataset=train_data, eval_dataset=eval_data, processing_class=tokenizer,
            callbacks=callbacks, optimizers=optimizers, rollout_func=sequential_rollout, gos_config=config)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'unified_config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    if stage != 'test':
        trainer.train(resume_from_checkpoint=resume)
        save_gos_adapter(model, output / 'adapter', asdict(config))
        tokenizer.save_pretrained(output / 'adapter')
    results = trainer.evaluate()
    if stage == 'test':
        results.update(generated_test_metrics(trainer, config, tokenizer, allow_remote_data))
    results['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    results['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (output / f'{stage}-metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/gemma5070.json')
    parser.add_argument('--stage', choices=['sft', 'grpo', 'test'], default='sft')
    parser.add_argument('--check-config', action='store_true', help='Print validated config without reading any data/models')
    parser.add_argument('--allow-remote-data', action='store_true', help='Permit streaming network reads when actually training')
    parser.add_argument('--allow-model-download', action='store_true')
    parser.add_argument('--adapter')
    parser.add_argument('--resume')
    args = parser.parse_args()
    config = UnifiedConfig(**json.loads(Path(args.config).read_text())).validate()
    if args.check_config:
        print(json.dumps({'config': asdict(config), 'base_quantization': 'NF4 + double quantization',
            'trainable': 'GoSReasoningLayer only', 'split': 'task-hash 80/10/10',
            'activation_checkpointing': 'graph and vocabulary-projection chunks',
            'remote_data_enabled': False, 'models_loaded': False}, indent=2))
        return
    from gos_runtime import cleanup_checkpointing
    try:
        run(config, args.stage, args.allow_remote_data, args.allow_model_download, args.adapter, args.resume)
    finally:
        cleanup_checkpointing()


if __name__ == '__main__':
    main()
