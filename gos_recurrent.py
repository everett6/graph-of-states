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

Training recipe from retrofitted recurrence (arXiv 2511.07384): pass counts drawn from a Poisson-lognormal
whose mean is raised over a curriculum (`sample_passes`), and backprop only through the last k passes
(`set_passes(..., backprop=k)`; earlier passes run without autograd, so memory stays flat as passes grow).
"""
import json
import math
from pathlib import Path
import random

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
        self.trace = None  # set to a list to record each extra pass's relative state change

    def inject(self, state, anchor):
        normed = anchor * torch.rsqrt(anchor.float().pow(2).mean(-1, keepdim=True) + self.norm_eps).to(anchor.dtype)
        delta = self.inject_up(self.inject_down(normed.to(self.inject_down.weight.dtype)))
        return state + delta.to(state.dtype)

    def mix(self, previous, output):
        mixed = previous + (self.gate.to(output.dtype) * (output - previous))
        if self.trace is not None:
            with torch.no_grad():
                change = (mixed - previous).float().norm() / previous.float().norm().clamp_min(1e-12)
            self.trace.append(float(change))
        return mixed


class PassLayer(nn.Module):
    """One slot of the expanded layer list: a shared decoder layer at a given pass and cache index."""
    def __init__(self, layer, cache_index, adapter=None, role=None, owner=False, frozen=False):
        super().__init__()
        if owner:
            self.layer = layer  # registered once so parameters/state_dict keys stay unique
        else:
            object.__setattr__(self, 'layer', layer)
        object.__setattr__(self, 'adapter', adapter)
        self.cache_index = cache_index
        self.role = role  # None | 'anchor' (block start, pass 1) | 'begin' | 'end' | 'begin_end'
        self.frozen = frozen  # pass outside the backprop window: run without autograd

    def forward(self, hidden_states, *args, **kwargs):
        if self.frozen and torch.is_grad_enabled():
            # The anchor is this pass's input tensor, so it keeps its graph for later injections.
            with torch.no_grad():
                return self._forward(hidden_states, *args, **kwargs)
        return self._forward(hidden_states, *args, **kwargs)

    def _forward(self, hidden_states, *args, **kwargs):
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


def set_passes(model, passes, backprop=None):
    """Rebuild the expanded layer list for `passes` block passes (any count; weights are shared).
    backprop=k keeps autograd for the last k passes only (truncated backprop); None keeps all."""
    decoder = text_model(model)
    spec = decoder._recurrence_spec
    if type(passes) is not int or passes < 1:
        raise ValueError('passes must be a positive integer')
    if backprop is not None and (type(backprop) is not int or backprop < 1):
        raise ValueError('backprop must be a positive integer or None')
    cut = passes - backprop if backprop else 0  # passes 1..cut run without autograd
    start, end, base, types = spec['start'], spec['end'], spec['base_layers'], spec['base_types']
    slots, slot_types = [], []
    def add(layer, kind, role=None, owner=False, frozen=False):
        slots.append(PassLayer(layer, len(slots), decoder.recurrence, role, owner, frozen))
        slot_types.append(kind)
    for index in range(start):
        add(base[index], types[index], owner=True)
    for index in range(start, end):
        add(base[index], types[index], 'anchor' if index == start else None, owner=True, frozen=cut >= 1)
    for number in range(2, passes + 1):
        for index in range(start, end):
            role = 'begin_end' if end - start == 1 else 'begin' if index == start else 'end' if index == end - 1 else None
            add(base[index], types[index], role, frozen=number <= cut)
    for index in range(end, len(base)):
        add(base[index], types[index], owner=True)
    decoder.layers = nn.ModuleList(slots)
    decoder.config.layer_types = slot_types
    decoder.config.num_hidden_layers = len(slots)
    spec['passes'], spec['backprop'] = passes, backprop
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
    """Set a sampled pass count before each training step (see sample_passes) with the spec's
    truncated-backprop window (`backprop_passes`)."""
    def __init__(self, model, spec, seed=0):
        self.model, self.spec, self.seed = model, spec, seed

    def __call__(self, step):
        passes = sample_passes(self.seed, step, self.spec)
        return set_passes(self.model, passes, self.spec.get('backprop_passes'))

def scheduled_mean(step, mean, curriculum_steps=0, shape='linear'):
    """Mean pass count at `step`: raised from 1 to `mean` over the first curriculum_steps steps
    ('linear' or '1-sqrt', the two schedules compared in 2511.07384), then held."""
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return float(mean)
    progress = step / curriculum_steps
    if shape == '1-sqrt':
        progress = 1 - math.sqrt(1 - progress)
    elif shape != 'linear':
        raise ValueError("curriculum shape must be 'linear' or '1-sqrt'")
    return 1 + (mean - 1) * progress


def poisson(rng, rate):
    """Poisson draw (Knuth for small rates, normal approximation above 30)."""
    if rate > 30:
        return max(0, round(rng.gauss(rate, math.sqrt(rate))))
    limit, count, product = math.exp(-rate), 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def sample_passes(seed, step, spec):
    """Pass count for a training step, keyed on (seed, step) so a resumed run replays the schedule.
    spec: max_passes, and either pass_weights (uniform-style categorical, the default) or
    sampler='poisson_lognormal' with mean_passes, sigma (0.5), curriculum_steps, curriculum.
    Poisson-lognormal (Huginn / 2511.07384): tau ~ N(log(m - 1) - sigma^2 / 2, sigma), passes = 1 + Poisson(e^tau),
    so the mean is m; draws are clipped to max_passes."""
    rng = random.Random(f'{seed}:{step}')
    top = spec.get('max_passes', 4)
    if spec.get('sampler', 'categorical') == 'categorical':
        return rng.choices(range(1, top + 1), spec.get('pass_weights') or [1] * top)[0]
    if spec['sampler'] != 'poisson_lognormal':
        raise ValueError("sampler must be 'categorical' or 'poisson_lognormal'")
    mean = scheduled_mean(step, spec.get('mean_passes', top), spec.get('curriculum_steps', 0),
                          spec.get('curriculum', 'linear'))
    if mean <= 1:
        return 1
    sigma = spec.get('sigma', 0.5)
    tau = rng.gauss(math.log(mean - 1) - sigma ** 2 / 2, sigma)
    return min(top, 1 + poisson(rng, math.exp(tau)))


def pass_trace(model, ids):
    """Relative state change of each extra pass (||s_r - s_{r-1}|| / ||s_{r-1}||) on one forward.
    Shrinking values mean the block converges toward a fixed point; growing ones mean it diverges."""
    adapter = text_model(model).recurrence
    adapter.trace = []
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        return list(adapter.trace)
    finally:
        adapter.trace = None
