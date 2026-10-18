import pytest
import torch

from gos_gemma import attach_gos, deep_supervised_logps, frozen_features, selected_logps
from test_unified import tiny_base
from train_unified import validate_enhancements


def test_deep_supervision_final_pass_matches_normal_output_and_trains_every_pass():
    torch.manual_seed(0)
    options = {'deep_supervision': True, 'scratchpad': True, 'latent_slots': 2}
    validate_enhancements(options, 4)
    model = attach_gos(tiny_base(), 16, 3, 4, options)
    layer = model.get_base_model().get_output_embeddings()
    torch.nn.init.normal_(layer.lora_B['default'].weight, std=0.5)  # nonzero so passes differ
    ids = torch.randint(0, 32, (1, 9))
    hidden = frozen_features(model, ids, torch.ones_like(ids))[0, :-1]
    targets = ids[0, 1:]
    per_pass = deep_supervised_logps(model, hidden, targets, chunk_size=3)
    assert per_pass.shape == (4, 8)
    with torch.no_grad():
        normal, _ = selected_logps(model, hidden, targets, chunk_size=3)
    assert torch.allclose(per_pass[-1].detach(), normal, atol=1e-5)
    assert not torch.allclose(per_pass[0], per_pass[-1])
    (-per_pass[0].mean()).backward()  # the first pass alone receives gradient
    graph = layer.lora_A['default']
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in graph.parameters())


def test_deep_supervision_rejects_adaptive_training_and_plain_heads():
    with pytest.raises(ValueError):
        validate_enhancements({'deep_supervision': True, 'adaptive': True}, 4)
    model = attach_gos(tiny_base(), 16, 3, 2, {})
    with pytest.raises(ValueError):
        deep_supervised_logps(model, torch.zeros(2, 32), torch.zeros(2, dtype=torch.long))
