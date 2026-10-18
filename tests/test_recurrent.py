import pytest
import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from gos_recurrent import (RandomPasses, attach_recurrence, block_lora_targets, load_recurrence,
                           recurrence_parameters, save_recurrence, set_passes, text_model)


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
    sampler = RandomPasses(model, 4, seed=1, weights=[2, 1, 1, 1])
    seen = {sampler() for _ in range(50)}
    assert seen == {1, 2, 3, 4}
    with pytest.raises(ValueError):
        set_passes(model, 0)
    with pytest.raises(ValueError):
        attach_recurrence(tiny(), 0, 2)
