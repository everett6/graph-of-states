"""Behavioral checks for training, portability, padding, and exact resume."""
import json
from pathlib import Path

import pytest
import torch

from gos_engine_architecture import LRRConfig, LatentRecursiveReconstruction
from lrr_data import (ByteTokenizer, BuiltinFrozenBackbone, FeatureDataset, collate_features,
                      create_cache, read_records, split_records)
from train_lrr import build_parser, evaluate_model, load_checkpoint, train


torch.set_num_threads(2)


def records():
    return [{'prompt': f'Input {index}', 'answer': str(index % 2)} for index in range(6)]


def cache(path, examples=None):
    spec = {'kind': 'builtin', 'width': 16, 'seed': 19}
    create_cache(examples or records(), path, BuiltinFrozenBackbone(16, 19), spec)
    return FeatureDataset(path)


def args(root, features, epochs=2, resume=None, batch_size=2, accumulation_steps=1):
    command = ['train', '--features', str(features), '--output', str(root), '--device', 'cpu',
               '--width', '16', '--nodes', '3', '--rounds', '2', '--epochs', str(epochs),
               '--batch-size', str(batch_size), '--accumulation-steps', str(accumulation_steps)]
    if resume:
        command.extend(['--resume', str(resume)])
    return build_parser().parse_args(command)


def test_byte_tokenizer_unicode_and_eos():
    tokenizer = ByteTokenizer()
    text = 'print("π = 3")\n🙂'
    assert tokenizer.decode([1] + tokenizer.encode(text) + [2] + tokenizer.encode('ignored')) == text
    assert max(tokenizer.encode(text)) < tokenizer.vocab_size


def test_json_validation_rejection_and_deduplication(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text('\n'.join(json.dumps(item) for item in
        [{'prompt': 'A', 'answer': '1'}, {'prompt': 'A', 'answer': '1'},
         {'prompt': 'B', 'answer': 'wrong', 'passed': False}]))
    examples, stats = read_records(path)
    assert len(examples) == 1
    assert stats['failed_rejected'] == stats['duplicates_removed'] == 1
    path.write_text('{"prompt":"A","answer":"1","passed":1}')
    with pytest.raises(ValueError, match='passed must'):
        read_records(path)


def test_split_and_padding_do_not_change_predictions(tmp_path):
    train_records, validation = split_records(records(), 0.3, 4)
    assert not {x['prompt'] for x in train_records} & {x['prompt'] for x in validation}
    dataset = cache(tmp_path / 'features', [{'prompt': 'A', 'answer': '0'},
        {'prompt': 'Longer prompt', 'answer': '100'}])
    batch = collate_features([dataset[0], dataset[1]])
    assert batch['labels'][0, 2:].eq(-100).all()
    model = LatentRecursiveReconstruction(LRRConfig(backbone_dim=16, width=16, vocab_size=259, rounds=2)).eval()
    with torch.no_grad():
        joint, _ = model(batch['hidden'], batch['mask'], batch['inputs'])
        one = collate_features([dataset[0]])
        alone, _ = model(one['hidden'], one['mask'], one['inputs'])
        torch.testing.assert_close(joint[:1, :alone.shape[1]], alone, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='at least one'):
        model.encode(batch['hidden'], torch.zeros_like(batch['mask']))


def test_conditioned_training_and_graph_gradients():
    torch.manual_seed(3)
    model = LatentRecursiveReconstruction(LRRConfig(backbone_dim=8, width=24, vocab_size=8,
                                                   nodes=3, rounds=2))
    hidden = torch.randn(4, 3, 8, requires_grad=True)
    mask = torch.ones(4, 3, dtype=torch.bool)
    inputs = torch.ones(4, 1, dtype=torch.long)
    labels = torch.tensor([[3], [4], [5], [6]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(80):
        optimizer.zero_grad()
        logits, _ = model(hidden, mask, inputs)
        loss = torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 0])
        loss.backward()
        assert hidden.grad is None
        assert model.transition.edge[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert logits[:, 0].argmax(-1).tolist() == labels[:, 0].tolist()
    assert loss.item() < 0.05


def test_epoch_resume_matches_uninterrupted_training(tmp_path):
    dataset = cache(tmp_path / 'features')
    full_path = train(args(tmp_path / 'full', dataset.root, epochs=2))
    partial_path = train(args(tmp_path / 'resumed', dataset.root, epochs=1))
    resumed_path = train(args(tmp_path / 'resumed', dataset.root, epochs=2, resume=partial_path))
    full, full_checkpoint = load_checkpoint(full_path, 'cpu')
    resumed, checkpoint = load_checkpoint(resumed_path, 'cpu')
    assert checkpoint['global_step'] == full_checkpoint['global_step']
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    metrics = evaluate_model(resumed, dataset, 'cpu', batch_size=2, max_new_tokens=4,
                             prediction_path=tmp_path / 'predictions.jsonl')
    assert metrics['examples'] == len(dataset)
    assert 0 <= metrics['exact_match'] <= 1
    assert len((tmp_path / 'predictions.jsonl').read_text().splitlines()) == len(dataset)
    assert checkpoint['builtin_backbone_state']
    assert (tmp_path / 'resumed' / 'best.pt').exists()


def test_accumulation_matches_equivalent_batch(tmp_path):
    dataset = cache(tmp_path / 'features', records()[:4])
    large = train(args(tmp_path / 'large', dataset.root, epochs=1, batch_size=4))
    accumulated = train(args(tmp_path / 'small', dataset.root, epochs=1, batch_size=2, accumulation_steps=2))
    large_model, _ = load_checkpoint(large, 'cpu')
    small_model, _ = load_checkpoint(accumulated, 'cpu')
    for key, value in large_model.state_dict().items():
        torch.testing.assert_close(value, small_model.state_dict()[key], rtol=1e-4, atol=1e-5)


def test_rejects_cache_mismatch_and_validation_leakage(tmp_path):
    dataset = cache(tmp_path / 'features')
    other = cache(tmp_path / 'other', records()[:2])
    arguments = args(tmp_path / 'out', dataset.root)
    arguments.validation_features = str(other.root)
    with pytest.raises(ValueError, match='leakage'):
        train(arguments)


def test_checkpointed_gradients_and_fixed_depth_generation():
    torch.manual_seed(11)
    config = LRRConfig(backbone_dim=8, width=16, vocab_size=16, rounds=3,
                       checkpoint_rounds=True, halt_tolerance=100)
    model = LatentRecursiveReconstruction(config)
    hidden = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    logits, _ = model(hidden, mask, torch.ones(2, 2, dtype=torch.long))
    logits.square().sum().backward()
    assert model.transition.propose[0].weight.grad.abs().sum() > 0
    _, fixed = model.generate(hidden, mask, 1, 2, 3)
    _, halted = model.generate(hidden, mask, 1, 2, 3, dynamic_halt=True)
    assert fixed['steps'].tolist() == [3, 3]
    assert halted['steps'].tolist() == [2, 2]
    assert model.training


def test_local_huggingface_backbone(tmp_path):
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    from gos_engine_architecture import FrozenBackbone
    vocabulary = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2, 'hello': 3, 'world': 4}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token='[UNK]'))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw,
        pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path)
    transformers.GPT2Model(transformers.GPT2Config(vocab_size=5, n_embd=16, n_layer=1,
        n_head=2, n_positions=32)).save_pretrained(tmp_path)
    backbone = FrozenBackbone(str(tmp_path), device='cpu', local_files_only=True)
    hidden, mask = backbone.extract(['hello world', 'hello'])
    assert hidden.shape == (2, 2, 16)
    assert not hidden.requires_grad
    assert mask.tolist() == [[True, True], [True, False]]
    assert all(not parameter.requires_grad for parameter in backbone.model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not visible to this process')
@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
def test_cuda_training_and_generation(tmp_path, precision):
    dataset = cache(tmp_path / 'features', records()[:4])
    arguments = args(tmp_path / 'training', dataset.root, epochs=1)
    arguments.device = 'cuda'
    arguments.precision = precision
    arguments.checkpoint_rounds = True
    path = train(arguments)
    model, checkpoint = load_checkpoint(path, 'cuda')
    assert checkpoint['metrics']['peak_allocated_bytes'] > 0
    assert checkpoint['epoch'] == 1
    batch = collate_features([dataset[0], dataset[1]])
    ids, _ = model.generate(batch['hidden'].cuda(), batch['mask'].cuda(), 1, 2, 4)
    assert ids.shape[0] == 2
    assert not ids.eq(0).any() and not ids.eq(1).any()
