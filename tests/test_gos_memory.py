"""Memory/loop/cache tests use synthetic tensors and local random Gemma weights."""
import copy
from dataclasses import asdict

import pytest
import torch

from gos_memory import HolographicScratchpad, RecurrentLatentCompactor
from gos_gemma import (TokenGraph, attach_gos, selected_logps, frozen_features,
                      save_gos_adapter, load_gos_adapter)
from gos_h2o import H2OConfig, HeavyHitterKV, h2o_generate, h2o_session
from train_unified import UnifiedConfig
from test_unified import tiny_base


OPTIONS = dict(scratchpad=True, latent_slots=2, thermal=True, min_rounds=2,
               temperature=1.5, temperature_floor=0.75,
               entropy_tolerance=0.01, motion_tolerance=0.01)


@pytest.mark.parametrize('width', [15, 16])
def test_unitary_binding_retrieval_and_gradients(width):
    pad = HolographicScratchpad(width, 1)
    value = torch.randn(3, 1, width, requires_grad=True)
    recovered = pad.read(pad.bind(value).squeeze(1))
    torch.testing.assert_close(recovered, value, atol=1e-6, rtol=1e-5)
    recovered.square().sum().backward()
    assert torch.isfinite(value.grad).all()
    assert pad.phase.grad is not None


def test_recurrent_slots_are_bounded_and_differentiable():
    compactor = RecurrentLatentCompactor(16, 2)
    tokens = torch.randn(3, 4, 16, requires_grad=True)
    memory = torch.zeros(3, 2, 16)
    for index in range(12):
        memory = compactor(memory, tokens + index / 12)
        assert memory.shape == (3, 2, 16)
    memory.square().mean().backward()
    assert tokens.grad.abs().sum() > 0
    assert compactor.queries.grad.abs().sum() > 0


def test_thermal_halt_chunk_invariance_and_no_persistent_state():
    options = OPTIONS | dict(entropy_tolerance=2.0, motion_tolerance=100.0)
    graph = TokenGraph(32, 16, nodes=3, rounds=8, enhancements=options)
    hidden = torch.randn(5, 32)
    features, _, stats = graph.features_and_risk(hidden, diagnostics=True)
    assert stats['rounds'].tolist() == [2] * 5
    assert stats['latent_slots'] == 2
    split = torch.cat([graph(row[None]) for row in hidden])
    torch.testing.assert_close(split, features, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(graph(hidden), features)
    graph._body.enhancements['motion_tolerance'] = 1e-12
    _, _, stats = graph.features_and_risk(hidden, diagnostics=True)
    assert stats['rounds'].tolist() == [8] * 5


def test_heavy_hitters_keep_actual_attention_mass_and_recent_tokens():
    cache = HeavyHitterKV(H2OConfig(1, 2))
    for position in range(6):
        cache.append(torch.full((1, 2, 1, 4), float(position)), torch.ones(1, 2, 1, 4), position)
        # Head zero keeps the oldest heavy token; head one favors latest old token.
        scores = torch.zeros(1, 4, 1, cache.keys.shape[-2])
        scores[:, :2, :, 0] = 1
        scores[0, 2:, 0, cache.positions[1] == 1] = 1
        if position == 0:
            scores[:, 2:, :, 0] = 1
        cache.observe_and_evict(scores)
        assert cache.keys.shape[-2] <= 3
    assert cache.positions[0].tolist() == [0, 4, 5]
    assert cache.positions[1, -2:].tolist() == [4, 5]
    assert cache.peak_tokens == 4 and cache.evicted == 3
    assert not torch.equal(cache.positions[0], cache.positions[1])


def test_h2o_no_eviction_matches_causal_generation_and_restores_backend():
    torch.manual_seed(5)
    model = tiny_base().eval()
    ids = torch.tensor([[2, 7, 8, 7, 8, 9]])
    original = model.config._attn_implementation
    with torch.no_grad():
        exact = model.generate(ids, max_new_tokens=4, use_cache=False, do_sample=False, logits_to_keep=1)
        actual, stats = h2o_generate(model, ids, 4, heavy_tokens=16, recent_tokens=16)
        torch.testing.assert_close(actual, exact)
        assert stats['evicted_per_head_total'] == 0
        with pytest.raises(RuntimeError, match='sentinel'):
            with h2o_session(model, H2OConfig(1, 2)):
                raise RuntimeError('sentinel')
    assert model.config._attn_implementation == original
    assert not any(hasattr(module, '_gos_h2o_session') for module in model.modules())


def test_enhanced_adapter_causal_gradients_checkpoint_and_roundtrip(tmp_path):
    torch.manual_seed(8)
    base = tiny_base()
    original = copy.deepcopy(base)
    model = attach_gos(base, 16, 3, 4, OPTIONS)
    head = model.get_base_model().get_output_embeddings()
    torch.nn.init.normal_(head.lora_B['default'].weight, std=0.01)
    ids = torch.tensor([[2, 7, 8, 9, 7, 8]])
    mask = torch.ones_like(ids)
    hidden = frozen_features(model, ids, mask)
    features, targets = hidden[:, :-1].reshape(-1, 32), ids[:, 1:].reshape(-1)
    logps, _ = selected_logps(model, features, targets, chunk_size=2)
    critic = torch.nn.functional.binary_cross_entropy_with_logits(head.risk_logits(hidden[0, -1:]), torch.ones(1))
    (-logps.mean() + critic).backward()
    graph = head.lora_A['default']
    for name, parameter in zip(graph.parameter_names, graph.parameters()):
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not parameter.requires_grad)
    # Actual head forward remains causal when the suffix is changed.
    model.eval()
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
        changed = ids.clone(); changed[0, -1] = 10
        torch.testing.assert_close(model(changed, use_cache=False).logits[:, :-1], logits[:, :-1])
    cfg = asdict(UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=4, graph_enhancements=OPTIONS))
    save_gos_adapter(model, tmp_path, cfg)
    restored = load_gos_adapter(original, tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, logits)
        result, stats = h2o_generate(restored, ids, 4, heavy_tokens=1, recent_tokens=2)
    assert result.shape[1] > ids.shape[1]
    assert stats['retained_tokens_per_head'] <= 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_enhancements_bf16_gpu_and_bounded_decoding():
    from transformers import Gemma4ForCausalLM, BitsAndBytesConfig
    base = tiny_base()
    model = Gemma4ForCausalLM.from_pretrained(None, config=base.config, state_dict=base.state_dict(),
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['lm_head']),
        device_map={'': 0}, dtype=torch.bfloat16)
    model = attach_gos(model, 16, 3, 4, OPTIONS)
    # Match TRL's adapter dtype conversion on a quantized backbone.
    head = model.get_base_model().get_output_embeddings()
    head.lora_A.to(torch.bfloat16); head.lora_B.to(torch.bfloat16)
    torch.nn.init.normal_(head.lora_B['default'].weight, std=0.01)
    ids = torch.tensor([[2, 7, 8, 9, 7, 8, 9, 8]], device='cuda')
    hidden = frozen_features(model, ids, torch.ones_like(ids))
    logps, _ = selected_logps(model, hidden[0, :-1], ids[0, 1:], chunk_size=2)
    (-logps.mean()).backward()
    assert torch.isfinite(head.lora_B['default'].weight.grad).all()
    model.eval()
    output, stats = h2o_generate(model, ids, 4, heavy_tokens=1, recent_tokens=2)
    assert output.is_cuda and stats['peak_tokens_per_head'] == 4
