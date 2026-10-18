"""Recurrent depth inside a pretrained decoder: re-pass a middle block of the same layers.

prelude (layers [0, start)) -> block (layers [start, end)) run `passes` times -> coda (rest).
Pass 1 is the original computation. Each extra pass r reuses the same block weights on

    x_r = s_{r-1} + inject(e)                       # e = block input, s_{r-1} = previous pass output
    s_r = s_{r-1} + gate * (block(x_r) - s_{r-1})   # gate and inject are zero-initialized

so a freshly attached model is exactly the base model at any pass count, and training decides how
much the extra passes contribute. Weights are shared across passes, so any pass count can be run
at inference (trained with a random count per step). Every pass has its own KV-cache slots, so
cached generation is exact. Trainable: the gate, the low-rank injection and optional LoRA on the
block layers (shared by all passes). Related work: Huginn (Geiping et al. 2025), Ouro/LoopLM,
retrofitted recurrence (McLeish et al. 2025).
"""
import json
from pathlib import Path

import torch
from torch import nn


class RecurrenceAdapter(nn.Module):
    def __init__(self, hidden_size, rank=64):
        super().__init__()
        self.inject_down = nn.Linear(hidden_size, rank, bias=False)
        self.inject_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.inject_up.weight)
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.norm_eps = 1e-6
        self._anchor = None
        self._previous = None

    def inject(self, state, anchor):
        normed = anchor * torch.rsqrt(anchor.float().pow(2).mean(-1, keepdim=True) + self.norm_eps).to(anchor.dtype)
        delta = self.inject_up(self.inject_down(normed.to(self.inject_down.weight.dtype)))
        return state + delta.to(state.dtype)

    def mix(self, previous, output):
        return previous + (self.gate.to(output.dtype) * (output - previous))


class PassLayer(nn.Module):
    """One slot of the expanded layer list: a shared decoder layer at a given pass and cache index."""
    def __init__(self, layer, cache_index, adapter=None, role=None, owner=False):
        super().__init__()
        if owner:
            self.layer = layer  # registered once so parameters/state_dict keys stay unique
        else:
            object.__setattr__(self, 'layer', layer)
        object.__setattr__(self, 'adapter', adapter)
        self.cache_index = cache_index
        self.role = role  # None | 'anchor' (block start, pass 1) | 'begin' | 'end' | 'begin_end'

    def forward(self, hidden_states, *args, **kwargs):
        adapter, role = self.adapter, self.role
        if role == 'anchor':
            adapter._anchor = hidden_states
        if role in ('begin', 'begin_end'):
            adapter._previous = hidden_states
            hidden_states = adapter.inject(hidden_states, adapter._anchor)
        attention = self.layer.self_attn
        original = attention.layer_idx
        attention.layer_idx = self.cache_index
        try:
            output = self.layer(hidden_states, *args, **kwargs)
        finally:
            attention.layer_idx = original
        if role in ('end', 'begin_end'):
            output = adapter.mix(adapter._previous, output)
        return output


def text_model(model):
    """The Gemma4TextModel inside a CausalLM / PEFT wrapper."""
    if hasattr(model, 'get_base_model'):
        model = model.get_base_model()  # PEFT wrapper
    for candidate in (model, getattr(model, 'model', None)):
        if candidate is not None and hasattr(candidate, 'layers') and hasattr(candidate, 'embed_tokens'):
            return candidate
    raise ValueError('Could not find the decoder layer stack')


def attach_recurrence(model, start, end, passes=1, rank=64):
    """Install shared-weight re-passing of layers [start, end). Returns the RecurrenceAdapter."""
    decoder = text_model(model)
    if hasattr(decoder, 'recurrence'):
        raise ValueError('Recurrence already attached')
    layers = list(decoder.layers)
    config = decoder.config
    if not 0 < start < end <= len(layers):
        raise ValueError('Need 0 < start < end <= num_hidden_layers')
    if getattr(config, 'num_kv_shared_layers', 0) or getattr(config, 'hidden_size_per_layer_input', 0):
        raise ValueError('KV-shared or per-layer-embedding stacks are not supported')
    device = layers[0].input_layernorm.weight.device
    adapter = RecurrenceAdapter(config.hidden_size, rank).to(device=device, dtype=torch.float32)
    decoder.recurrence = adapter
    decoder._recurrence_spec = {'start': start, 'end': end, 'rank': rank, 'base_layers': layers,
                                'base_types': list(config.layer_types)}
    set_passes(model, passes)
    return adapter


def set_passes(model, passes):
    """Rebuild the expanded layer list for `passes` block passes (any count; weights are shared)."""
    decoder = text_model(model)
    spec = decoder._recurrence_spec
    if type(passes) is not int or passes < 1:
        raise ValueError('passes must be a positive integer')
    start, end, base, types = spec['start'], spec['end'], spec['base_layers'], spec['base_types']
    slots, slot_types = [], []
    def add(layer, kind, role=None, owner=False):
        slots.append(PassLayer(layer, len(slots), decoder.recurrence, role, owner))
        slot_types.append(kind)
    for index in range(start):
        add(base[index], types[index], owner=True)
    for index in range(start, end):
        add(base[index], types[index], 'anchor' if index == start else None, owner=True)
    for _ in range(passes - 1):
        for index in range(start, end):
            role = 'begin_end' if end - start == 1 else 'begin' if index == start else 'end' if index == end - 1 else None
            add(base[index], types[index], role)
    for index in range(end, len(base)):
        add(base[index], types[index], owner=True)
    decoder.layers = nn.ModuleList(slots)
    decoder.config.layer_types = slot_types
    decoder.config.num_hidden_layers = len(slots)
    spec['passes'] = passes
    return passes


def recurrence_parameters(model):
    return list(text_model(model).recurrence.parameters())


def save_recurrence(model, path):
    decoder = text_model(model)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.recurrence.state_dict(), path / 'recurrence.pt')
    spec = {key: decoder._recurrence_spec[key] for key in ('start', 'end', 'rank', 'passes')}
    (path / 'recurrence.json').write_text(json.dumps(spec, indent=2) + '\n')


def load_recurrence(model, path, passes=None):
    path = Path(path)
    spec = json.loads((path / 'recurrence.json').read_text())
    adapter = attach_recurrence(model, spec['start'], spec['end'], passes or spec['passes'], spec['rank'])
    adapter.load_state_dict(torch.load(path / 'recurrence.pt', map_location='cpu', weights_only=True))
    return adapter


def block_lora_targets(start, end):
    """PEFT target regex: attention and MLP projections of the re-passed block only."""
    indices = '|'.join(str(index) for index in range(start, end))
    return rf'.*layers\.({indices})\.layer\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'


class RandomPasses:
    """Sample a pass count per training step (heavier mass on 1 keeps the base behaviour intact)."""
    def __init__(self, model, maximum, seed=0, weights=None):
        import random
        self.model, self.maximum = model, maximum
        self.rng = random.Random(seed)
        self.weights = weights or [1.0] * maximum

    def __call__(self):
        passes = self.rng.choices(range(1, self.maximum + 1), self.weights)[0]
        return set_passes(self.model, passes)
