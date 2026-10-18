import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

import random

from gos_recurrent import (RandomPasses, attach_recurrence, block_lora_targets, load_recurrence, pass_trace,
                           recurrence_parameters, sample_passes, save_recurrence, scheduled_mean, set_passes,
                           text_model)


def tiny(layers=6, window=4):
    torch.manual_seed(0)
    types = ['sliding_attention', 'sliding_attention', 'full_attention'] * (layers // 3)
    config = Gemma4TextConfig(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8, max_position_embeddings=128,
        vocab_size_per_layer_input=0, hidden_size_per_layer_input=0, layer_types=types, sliding_window=window,
        pad_token_id=0, eos_token_id=1, bos_token_id=2, attention_k_eq_v=False)
    return Gemma4ForCausalLM(config).eval()


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def perturb(model):
    adapter = text_model(model).recurrence
    with torch.no_grad():
        adapter.gate.normal_(0.5, 0.2)
        adapter.inject_up.weight.normal_(0, 0.2)


def test_zero_init_matches_base_at_any_pass_count():
    model = tiny()
    ids = torch.randint(3, 48, (2, 10))
    reference = logits(model, ids)
    attach_recurrence(model, 2, 4, passes=1)
    for passes in (1, 2, 5):
        set_passes(model, passes)
        assert torch.allclose(logits(model, ids), reference, atol=1e-5)


def test_extra_passes_change_output_and_share_weights():
    model = tiny()
    count = sum(p.numel() for p in model.parameters())
    attach_recurrence(model, 2, 4, passes=3)
    perturb(model)
    decoder = text_model(model)
    assert len(decoder.layers) == 6 + 2 * 2 and decoder.config.num_hidden_layers == 10
    keys = list(model.state_dict())
    assert len(keys) == len(set(keys))
    added = sum(p.numel() for p in recurrence_parameters(model))
    assert sum(p.numel() for p in model.parameters()) == count + added
    ids = torch.randint(3, 48, (1, 8))
    three = logits(model, ids)
    set_passes(model, 1)
    assert not torch.allclose(three, logits(model, ids))


@pytest.mark.parametrize('passes', [1, 3])
def test_cached_generation_matches_full_recompute(passes):
    model = tiny(window=4)  # window smaller than the sequence exercises sliding caches
    attach_recurrence(model, 2, 4, passes=passes)
    perturb(model)
    ids = torch.randint(3, 48, (1, 5))
    with torch.no_grad():
        cached = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=True,
                                pad_token_id=0, eos_token_id=None)
        uncached = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False,
                                  pad_token_id=0, eos_token_id=None)
    assert torch.equal(cached, uncached)
    full = logits(model, cached)
    with torch.no_grad():
        step = model(input_ids=cached[:, :-1], use_cache=True)
        last = model(input_ids=cached[:, -1:], past_key_values=step.past_key_values, use_cache=True).logits
    assert torch.allclose(last[:, -1], full[:, -1], atol=1e-4)


def test_gradients_reach_adapter_and_block_lora(tmp_path):
    from peft import LoraConfig, get_peft_model
    model = tiny()
    attach_recurrence(model, 2, 4, passes=2)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=block_lora_targets(2, 4)))
    for parameter in recurrence_parameters(model):
        parameter.requires_grad_(True)
    names = [name for name, p in model.named_parameters() if p.requires_grad and 'lora_' in name]
    assert names and all('.layers.2.' in name or '.layers.3.' in name for name in names)
    perturb(model)
    ids = torch.randint(3, 48, (1, 9))
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    adapter = text_model(model).recurrence
    assert adapter.gate.grad is not None and adapter.gate.grad.abs().sum() > 0
    assert all(p.grad is not None for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad)
    save_recurrence(model, tmp_path)
    other = tiny()
    load_recurrence(other, tmp_path)
    assert torch.equal(text_model(other).recurrence.gate, adapter.gate.detach())


def test_random_passes_and_validation():
    model = tiny()
    attach_recurrence(model, 2, 4)
    sampler = RandomPasses(model, {'max_passes': 4, 'pass_weights': [2, 1, 1, 1]}, seed=1)
    seen = {sampler(step) for step in range(50)}
    assert seen == {1, 2, 3, 4}
    with pytest.raises(ValueError):
        set_passes(model, 0)
    with pytest.raises(ValueError):
        attach_recurrence(tiny(), 0, 2)


def lora_model(passes):
    from peft import LoraConfig, get_peft_model
    model = tiny()
    attach_recurrence(model, 2, 4, passes=passes)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=block_lora_targets(2, 4)))
    for parameter in recurrence_parameters(model):
        parameter.requires_grad_(True)
    perturb(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0, 0.1)
    return model


def grads(model, ids, passes, backprop):
    set_passes(model, passes, backprop)
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    return float(loss), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_truncated_backprop_keeps_forward_and_cuts_early_passes():
    model = lora_model(4)
    ids = torch.randint(3, 48, (1, 9))
    full_loss, full = grads(model, ids, 4, None)
    same_loss, same = grads(model, ids, 4, 4)
    cut_loss, cut = grads(model, ids, 4, 1)
    assert full_loss == pytest.approx(cut_loss, abs=1e-6) and full_loss == pytest.approx(same_loss, abs=1e-6)
    assert set(full) == set(same) == set(cut)  # last pass still trains gate, injection and block LoRA
    assert all(torch.allclose(full[n], same[n], atol=1e-6) for n in full)
    assert any(not torch.allclose(full[n], cut[n], atol=1e-6) for n in full)
    frozen = [slot.frozen for slot in text_model(model).layers]
    assert frozen == [False, False] + [True] * 6 + [False, False] + [False, False]
    with pytest.raises(ValueError):
        set_passes(model, 2, 0)


def test_truncated_window_matches_detached_state():
    """backprop=k gives the gradient of a run where the state entering the last k passes is detached."""
    model = lora_model(3)
    ids = torch.randint(3, 48, (1, 7))
    _, cut = grads(model, ids, 3, 1)
    decoder = text_model(model)
    entry = decoder.layers[2 + 2 * 2]  # first slot of pass 3
    original = entry.forward
    def detached(hidden_states, *args, **kwargs):
        return original(hidden_states.detach(), *args, **kwargs)
    set_passes(model, 3)
    decoder.layers[6].forward = detached
    model.zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    reference = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(reference) == set(cut)
    assert all(torch.allclose(reference[n], cut[n], atol=1e-6) for n in cut)


def test_categorical_sampler_replays_benchmark_schedule():
    spec = {'max_passes': 4, 'pass_weights': [3, 1, 1, 1]}
    for step in range(20):
        expected = random.Random(f'17:{step}').choices(range(1, 5), [3, 1, 1, 1])[0]
        assert sample_passes(17, step, spec) == expected


def test_poisson_lognormal_curriculum():
    assert scheduled_mean(0, 8, 100) == 1 and scheduled_mean(100, 8, 100) == 8
    assert scheduled_mean(50, 8, 100) == pytest.approx(4.5)
    assert scheduled_mean(50, 8, 100, '1-sqrt') == pytest.approx(1 + 7 * (1 - 0.5 ** 0.5))
    with pytest.raises(ValueError):
        scheduled_mean(5, 8, 100, 'cosine')
    spec = {'sampler': 'poisson_lognormal', 'max_passes': 32, 'mean_passes': 6, 'curriculum_steps': 1000}
    assert {sample_passes(0, step, spec) for step in range(5)} == {1}
    late = [sample_passes(0, step, spec) for step in range(1000, 5000)]
    assert 1 <= min(late) and max(late) <= 32
    assert sum(late) / len(late) == pytest.approx(6, rel=0.08)
    capped = {**spec, 'max_passes': 4, 'curriculum_steps': 0}
    assert max(sample_passes(0, step, capped) for step in range(300)) == 4
    assert sample_passes(3, 7, spec) == sample_passes(3, 7, spec)


def test_pass_trace_reports_each_extra_pass():
    model = tiny()
    attach_recurrence(model, 2, 4, passes=4)
    ids = torch.randint(3, 48, (1, 6))
    assert pass_trace(model, ids) == [0.0, 0.0, 0.0]  # zero gate: extra passes change nothing
    perturb(model)
    trace = pass_trace(model, ids)
    assert len(trace) == 3 and all(value > 0 for value in trace)
    assert text_model(model).recurrence.trace is None
