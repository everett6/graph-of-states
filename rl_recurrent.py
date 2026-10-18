"""GRPO with verified rewards for the recurrent-depth block (or its all-linear LoRA control).

train_unified.py trains only the GoS head (frozen-backbone features, risk critic). Here the trainable
parameters live inside the network (gos_recurrent block + block LoRA, or LoRA everywhere), so log-probs come
from a full forward with a chunked vocabulary projection. Each step samples a pass count (like the SFT
benchmark) and uses it for both generation and scoring. agent_max_tools > 0 uses agent_rl.agent_rollout.
init_adapter continues from an SFT adapter directory (e.g. runs/pilot-heldout-v1/<variant>/adapter).

    .venv-training/bin/python rl_recurrent.py --config configs/rl-recurrent-agent.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from agent_rl import agent_rollout
from gos_gemma import inner_lm

DEFAULTS = dict(variant='recurrent', recurrent={'start': 18, 'end': 24, 'rank': 64, 'lora_rank': 16, 'max_passes': 4},
                pass_weights=None, lora_rank=8, lora_targets='all-linear', steps=200, learning_rate=1e-5,
                grpo_generations=4, max_prompt_tokens=2048, max_completion_tokens=1024, agent_max_tools=0,
                reward_function='verified_rewards:grpo_verified_reward', seed=17, logits_chunk=64,
                grpo_use_cache=True, save_steps=50)


def chunked_logps(model, input_ids, logits_to_keep, temperature=1.0, chunk=64, entropy=False):
    """Log-probs of the last `logits_to_keep` tokens without a [tokens, vocab] tensor for the whole sequence."""
    lm = inner_lm(model)
    hidden = lm.base_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False,
                           return_dict=True).last_hidden_state
    features, targets = hidden[0, :-1][-logits_to_keep:], input_ids[0, -logits_to_keep:]
    config = lm.config.get_text_config() if hasattr(lm.config, 'get_text_config') else lm.config
    head, cap = lm.get_output_embeddings(), getattr(config, 'final_logit_softcapping', None)
    scale = getattr(config, 'logit_scale', None) or getattr(config, 'output_multiplier', None) or 1.0

    def project(x, y):
        logits = head(x) * scale
        if cap:
            logits = cap * torch.tanh(logits / cap)
        logp = torch.log_softmax(logits.float() / temperature, -1)
        chosen = logp.gather(-1, y[:, None]).squeeze(-1)
        return chosen, (-(logp.exp() * logp).sum(-1) if entropy else chosen * 0)

    parts = [checkpoint(project, features[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
             if torch.is_grad_enabled() else project(features[i:i + chunk], targets[i:i + chunk])
             for i in range(0, len(targets), chunk)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


class RecurrentGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, gos_config, **kwargs):
        self.gos_config = gos_config
        super().__init__(*args, **kwargs)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                          batch_size=None, compute_entropy=False, compute_aux_loss=False, **kwargs):
        if compute_aux_loss or any(value is not None for value in kwargs.values()):
            raise ValueError('Text-only GRPO')
        logps, entropies = [], []
        for row in range(input_ids.shape[0]):  # one completion at a time bounds memory
            # Drop the batch's left prompt padding and right completion padding so positions match the
            # unpadded single-sequence generation; padded completion slots get 0 (TRL masks them).
            prompt, completion = input_ids[row, :-logits_to_keep], input_ids[row, -logits_to_keep:]
            prompt = prompt[attention_mask[row, :-logits_to_keep].bool()]
            length = int(attention_mask[row, -logits_to_keep:].sum())
            if length == 0:
                zeros = input_ids.new_zeros(logits_to_keep, dtype=torch.float)
                logps.append(zeros)
                entropies.append(zeros)
                continue
            chosen, entropy = chunked_logps(model, torch.cat([prompt, completion[:length]])[None], length,
                                            self.temperature, self.gos_config.logits_chunk, compute_entropy)
            pad = logits_to_keep - length
            logps.append(torch.nn.functional.pad(chosen, (0, pad)))
            entropies.append(torch.nn.functional.pad(entropy, (0, pad)))
        return torch.stack(logps), (torch.stack(entropies) if compute_entropy else None), None


class PassSchedule(TrainerCallback):
    """Sample the recurrent pass count before each step (generation and scoring share it)."""
    def __init__(self, model, spec, seed, log):
        from gos_recurrent import RandomPasses
        self.sample, self.log = RandomPasses(model, spec, seed), log

    def on_step_begin(self, args, state, control, **kwargs):
        self.log.append({'step': state.global_step, 'passes': self.sample(state.global_step)})


def load_trainable(base, variant, path, passes):
    """Continue from an SFT adapter saved by benchmark_gos (PEFT LoRA, plus recurrence.pt for 'recurrent')."""
    from peft import PeftModel
    base.requires_grad_(False)
    if variant == 'recurrent':
        from gos_recurrent import load_recurrence, recurrence_parameters
        load_recurrence(base, path, passes)
    model = PeftModel.from_pretrained(base, path, is_trainable=True)
    if variant == 'recurrent':
        for parameter in recurrence_parameters(model):
            parameter.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    settings = {**DEFAULTS, **json.loads(Path(args.config).read_text())}
    for key in ('base_path', 'output_dir', 'sources'):
        if key not in settings:
            raise ValueError(f'Config needs {key}')
    if settings['variant'] not in ('recurrent', 'lora') or not 0 <= settings['agent_max_tools'] <= 8:
        raise ValueError('variant must be recurrent or lora; agent_max_tools in [0, 8]')
    if args.check_config:
        print(json.dumps(settings, indent=2))
        return
    from datasets import IterableDataset
    from experiment_support import configure
    from local_gemma import load_causal_lm
    from train_unified import load_reward, sequential_rollout
    from unified_data import mixed_examples
    torch.manual_seed(settings['seed'])
    base, tokenizer = load_causal_lm(settings['base_path'])
    tokenizer.padding_side = 'left'
    if settings.get('init_adapter'):
        model = load_trainable(base, settings['variant'], settings['init_adapter'],
                               settings['recurrent'].get('max_passes', 1))
    else:
        model = configure(base, settings['variant'], lora_rank=settings['lora_rank'],
                          lora_targets=settings['lora_targets'], recurrent=settings['recurrent'])
    config = SimpleNamespace(**{k: settings[k] for k in ('max_prompt_tokens', 'max_completion_tokens',
                                                        'agent_max_tools', 'grpo_use_cache', 'logits_chunk')})
    data = IterableDataset.from_generator(lambda: mixed_examples(
        settings['sources'], tokenizer, 'train', settings['seed'], 'grpo', prompt_limit=settings['max_prompt_tokens'],
        agent_tools=settings['agent_max_tools']))
    output = Path(settings['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rl_config.json').write_text(json.dumps(settings, indent=2) + '\n')
    groups = settings['grpo_generations']
    grpo = GRPOConfig(output_dir=str(output), max_steps=settings['steps'], learning_rate=settings['learning_rate'],
        per_device_train_batch_size=1, gradient_accumulation_steps=groups, generation_batch_size=groups,
        num_generations=groups, max_completion_length=settings['max_completion_tokens'], beta=0.0,
        loss_type='grpo', temperature=1.0, top_k=0, top_p=1.0, use_vllm=False, use_cache=False, bf16=True,
        gradient_checkpointing=False, save_steps=settings['save_steps'], logging_steps=1, report_to='none',
        seed=settings['seed'])
    passes_log = []
    callbacks = []
    if settings['variant'] == 'recurrent' and settings['recurrent'].get('max_passes', 1) > 1:
        spec = {'pass_weights': settings['pass_weights'], **settings['recurrent']}
        callbacks.append(PassSchedule(model, spec, settings['seed'], passes_log))
    trainer = RecurrentGRPOTrainer(model=model, args=grpo, reward_funcs=load_reward(settings['reward_function']),
        train_dataset=data, processing_class=tokenizer, callbacks=callbacks, gos_config=config,
        rollout_func=agent_rollout if settings['agent_max_tools'] else sequential_rollout)
    trainer.train()
    model.save_pretrained(output / 'adapter')
    if settings['variant'] == 'recurrent':
        from gos_recurrent import save_recurrence
        save_recurrence(model, output / 'adapter')
    (output / 'passes.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in passes_log))


if __name__ == '__main__':
    main()
