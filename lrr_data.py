"""Portable byte targets, frozen feature extraction, and validated JSONL caches."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


class ByteTokenizer:
    """UTF-8 bytes plus distinct PAD, BOS, EOS; no vocabulary downloads."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return [value + 3 for value in text.encode('utf-8')]

    def decode(self, ids) -> str:
        values = []
        for token in ids:
            token = int(token)
            if token == self.eos_token_id:
                break
            if 3 <= token < self.vocab_size:
                values.append(token - 3)
        return bytes(values).decode('utf-8', errors='replace')


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_records(path: str | Path) -> tuple[list[dict], dict]:
    records, seen = [], {}
    stats = {'lines': 0, 'failed_rejected': 0, 'duplicates_removed': 0}
    with open(path, encoding='utf-8') as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            stats['lines'] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{number}: invalid JSON') from error
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{number}: expected a JSON object')
            if 'passed' in record and type(record['passed']) is not bool:
                raise ValueError(f'{path}:{number}: passed must be true or false')
            if record.get('passed') is False:
                stats['failed_rejected'] += 1
                continue
            if not all(isinstance(record.get(key), str) and record[key].strip()
                       for key in ('prompt', 'answer')):
                raise ValueError(f'{path}:{number}: prompt and answer must be nonempty strings')
            prompt, answer = record['prompt'], record['answer']
            key = prompt.strip()
            if key in seen:
                if seen[key] != answer:
                    raise ValueError(f'{path}:{number}: conflicting answers for the same prompt')
                stats['duplicates_removed'] += 1
                continue
            seen[key] = answer
            records.append({'prompt': prompt, 'answer': answer})
    if not records:
        raise ValueError('No usable examples')
    return records, stats


def split_records(records: list[dict], fraction: float, seed: int):
    if not 0 <= fraction < 1:
        raise ValueError('Validation fraction must be in [0, 1)')
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    count = max(1, round(len(order) * fraction)) if fraction else 0
    if count >= len(records):
        raise ValueError('Need at least two examples for a train/validation split')
    validation = set(order[:count])
    return ([record for index, record in enumerate(records) if index not in validation],
            [record for index, record in enumerate(records) if index in validation])


class BuiltinFrozenBackbone(nn.Module):
    """Small random frozen contextual encoder for offline pipeline validation.

    This contains no pretrained world knowledge. Persist the exact weights so
    caches and inference share an encoder even across different PyTorch builds.
    """
    def __init__(self, width: int = 64, seed: int = 17):
        super().__init__()
        self.width = width
        self.seed = seed
        self.tokenizer = ByteTokenizer()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(259, width, padding_idx=0)
            layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0.0,
                                               batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def extract(self, prompts, max_length=256, truncate=False):
        sequences = [self.tokenizer.encode(prompt) for prompt in prompts]
        if any(not sequence for sequence in sequences):
            raise ValueError('Prompts must be nonempty')
        if not truncate and any(len(sequence) > max_length for sequence in sequences):
            raise ValueError('Prompt exceeds --max-prompt-tokens; raise limit or use --truncate-prompts')
        sequences = [sequence[:max_length] for sequence in sequences]
        device = self.embedding.weight.device
        ids = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[index, :len(sequence)] = True
        positions = torch.arange(ids.shape[1], device=device).float()[:, None]
        frequencies = torch.exp(torch.arange(0, self.width, 2, device=device).float()
                                * (-math.log(10000) / self.width))
        positional = torch.zeros(ids.shape[1], self.width, device=device)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        positional[:, 1::2] = torch.cos(positions * frequencies)
        hidden = self.encoder(self.embedding(ids) + positional, src_key_padding_mask=~mask)
        return hidden.detach(), mask


def make_backbone(spec: dict, device: str, builtin_state=None):
    if spec['kind'] == 'builtin':
        backbone = BuiltinFrozenBackbone(spec['width'], spec['seed'])
        if builtin_state is not None:
            backbone.load_state_dict(builtin_state)
        return backbone.to(device)
    if spec['kind'] != 'huggingface':
        raise ValueError('Unknown backbone kind')
    from gos_engine_architecture import FrozenBackbone
    return FrozenBackbone(spec['model_id'], spec.get('quantize_4bit', False), device,
                          revision=spec.get('revision'), local_files_only=spec.get('local_files_only', False))


def create_cache(records, output, backbone, spec, max_prompt_tokens=256,
                 max_answer_tokens=256, truncate=False):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Cache directory must be empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    # Include content in dataset identity; feature identity additionally covers
    # exact builtin weights or resolved Hugging Face model revision.
    metadata = {'schema_version': 2, 'backbone': spec, 'backbone_dim': backbone.width,
                'vocab_size': tokenizer.vocab_size, 'target_tokenizer': 'utf8-bytes-v1',
                'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
                'max_prompt_tokens': max_prompt_tokens, 'max_answer_tokens': max_answer_tokens,
                'truncate_prompts': truncate, 'examples': len(records),
                'records_digest': fingerprint(records)}
    if isinstance(backbone, BuiltinFrozenBackbone):
        state = {key: value.cpu() for key, value in backbone.state_dict().items()}
        torch.save(state, root / 'backbone.pt')
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        metadata['backbone_weights_digest'] = digest.hexdigest()
    for index, record in enumerate(records):
        answer = tokenizer.encode(record['answer']) + [tokenizer.eos_token_id]
        if len(answer) > max_answer_tokens:
            raise ValueError(f'Answer {index} exceeds --max-answer-tokens; raise the limit')
        hidden, mask = backbone.extract([record['prompt']], max_prompt_tokens, truncate=truncate)
        torch.save({'hidden': hidden.cpu().to(torch.float16), 'mask': mask.cpu(),
                    'inputs': torch.tensor([[tokenizer.bos_token_id] + answer[:-1]]),
                    'labels': torch.tensor([answer]), **record}, root / f'{index:08d}.pt')
    metadata['feature_signature'] = fingerprint({key: metadata.get(key) for key in
        ('backbone', 'backbone_dim', 'backbone_weights_digest', 'target_tokenizer',
         'max_prompt_tokens', 'truncate_prompts')})
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        if self.metadata.get('schema_version') != 2:
            raise ValueError('Recreate the cache using the current prepare command (schema v2)')
        self.files = sorted(self.root.glob('[0-9]*.pt'))
        if not self.files or len(self.files) != self.metadata['examples']:
            raise ValueError('Empty or incomplete feature cache')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = torch.load(self.files[index], map_location='cpu', weights_only=True)
        item['hidden'] = item['hidden'].float().squeeze(0)
        for key in ('mask', 'inputs', 'labels'):
            item[key] = item[key].squeeze(0)
        return item


def collate_features(items):
    batch, prompt_length, answer_length = len(items), max(len(x['mask']) for x in items), max(len(x['labels']) for x in items)
    hidden = torch.zeros(batch, prompt_length, items[0]['hidden'].shape[-1])
    mask = torch.zeros(batch, prompt_length, dtype=torch.bool)
    inputs = torch.zeros(batch, answer_length, dtype=torch.long)
    labels = torch.full((batch, answer_length), -100, dtype=torch.long)
    for index, item in enumerate(items):
        length, target_length = len(item['mask']), len(item['labels'])
        hidden[index, :length] = item['hidden']
        mask[index, :length] = item['mask']
        inputs[index, :target_length] = item['inputs']
        labels[index, :target_length] = item['labels']
    return {'hidden': hidden, 'mask': mask, 'inputs': inputs, 'labels': labels,
            'prompts': [x['prompt'] for x in items], 'answers': [x['answer'] for x in items]}
