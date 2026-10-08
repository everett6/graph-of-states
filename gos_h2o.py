"""Attention-score H2O-style bounded decoding for Transformers Gemma 4 text.

Primary method: https://arxiv.org/abs/2306.14048 . This implementation uses
per-KV-head scores averaged over grouped query heads, plus a recent window.
It owns K/V tensors inside a scoped attention backend (not HF DynamicCache).
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AttentionInterface

from gos_gemma import inner_lm


@dataclass
class H2OConfig:
    heavy_tokens: int = 256
    recent_tokens: int = 256
    quantization: str = "none"
    kernel: str = "torch"

    def validate(self):
        if type(self.heavy_tokens) is not int or type(self.recent_tokens) is not int or self.heavy_tokens < 0 or self.recent_tokens < 1:
            raise ValueError('H2O needs heavy_tokens >= 0 and recent_tokens >= 1')
        if self.quantization not in ('none', 'int4') or self.kernel not in ('torch', 'triton'):
            raise ValueError('Invalid KV quantization/kernel')
        return self


class HeavyHitterKV:
    """One layer, batch size one; separate retained positions per KV head."""
    def __init__(self, config):
        self.config = config.validate()
        self._keys = self._values = self.scores = self.positions = None
        self.evicted = 0
        self.peak_tokens = 0

    @property
    def keys(self):
        return self._keys.unpack() if self.config.quantization == 'int4' else self._keys

    @property
    def values(self):
        return self._values.unpack() if self.config.quantization == 'int4' else self._values

    def storage_bytes(self):
        def size(tensor):
            return tensor.storage_bytes() if self.config.quantization == 'int4' else tensor.numel() * tensor.element_size()
        return size(self._keys) + size(self._values)

    def append(self, keys, values, position):
        if keys.shape[0] != 1 or keys.shape[-2] != 1:
            raise ValueError('H2O streams exactly one unpadded token at a time')
        positions = torch.full(keys.shape[1:3], position, device=keys.device, dtype=torch.long)
        scores = torch.zeros(keys.shape[1:3], device=keys.device, dtype=torch.float32)
        if self.config.quantization == 'int4':
            from gos_int4 import PackedInt4
            keys, values = (PackedInt4.pack(tensor, kernel=self.config.kernel) for tensor in (keys, values))
        if self._keys is None:
            self._keys, self._values, self.positions, self.scores = keys, values, positions, scores
        else:
            if self.config.quantization == 'int4':
                self._keys, self._values = self._keys.append(keys), self._values.append(values)
            else:
                self._keys = torch.cat((self._keys, keys), -2)
                self._values = torch.cat((self._values, values), -2)
            self.positions = torch.cat((self.positions, positions), -1)
            self.scores = torch.cat((self.scores, scores), -1)
        self.peak_tokens = max(self.peak_tokens, self._keys.shape[-2])

    def observe_and_evict(self, weights):
        # [1, KV heads, grouped query heads, query=1, cached tokens]
        heads = self._keys.shape[1]
        grouped = weights.float().reshape(1, heads, -1, 1, weights.shape[-1])
        self.scores += grouped.mean(2).sum((0, 2))
        count = self._keys.shape[-2]
        budget = self.config.heavy_tokens + self.config.recent_tokens
        if count <= budget:
            return
        recent_start = count - self.config.recent_tokens
        heavy = self.scores[:, :recent_start].topk(self.config.heavy_tokens, dim=-1).indices
        recent = torch.arange(recent_start, count, device=self.scores.device).expand(heads, -1)
        selected = torch.cat((heavy, recent), -1).sort(-1).values
        if self.config.quantization == 'int4':
            self._keys, self._values = self._keys.gather_tokens(selected), self._values.gather_tokens(selected)
        else:
            self._keys = self._keys.gather(2, selected[None, :, :, None].expand(1, heads, budget, self._keys.shape[-1])).contiguous()
            self._values = self._values.gather(2, selected[None, :, :, None].expand(1, heads, budget, self._values.shape[-1])).contiguous()
        self.positions = self.positions.gather(1, selected)
        self.scores = self.scores.gather(1, selected)
        self.evicted += count - budget


def _h2o_attention(module, query, key, value, attention_mask, scaling=None,
                   dropout=0.0, softcap=None, sliding_window=None, **kwargs):
    session = module._gos_h2o_session
    cache = session['layers'].get(module.layer_idx)
    if cache is None:
        cache = session['layers'][module.layer_idx] = HeavyHitterKV(session['config'])
    position = session['position']
    cache.append(key, value, position)
    groups = query.shape[1] // key.shape[1]
    keys = cache.keys.repeat_interleave(groups, dim=1)
    values = cache.values.repeat_interleave(groups, dim=1)
    logits = (query @ keys.transpose(-2, -1)) * (scaling if scaling is not None else query.shape[-1] ** -0.5)
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    positions = cache.positions.repeat_interleave(groups, dim=0)
    allowed = positions <= position
    if sliding_window is not None:
        allowed &= positions > position - sliding_window
    logits = logits.masked_fill(~allowed[None, :, None], torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits.float(), -1).to(query.dtype)
    output = (weights @ values).transpose(1, 2).contiguous()
    cache.observe_and_evict(weights)
    return output, None


AttentionInterface.register('gos_h2o', _h2o_attention)


@contextmanager
def h2o_session(model, config):
    """Install locally scoped state; restore backend and release all KV on exit.

    Not thread-safe on the same model. No training, beam search, batch padding,
    multimodal inputs or generic model support is silently implied.
    """
    lm = inner_lm(model)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    modules = [module for module in lm.modules() if isinstance(module, Gemma4TextAttention)]
    if not modules or getattr(lm.config, 'model_type', '') != 'gemma4_text':
        raise ValueError('H2O backend currently supports Gemma4ForCausalLM text only')
    if model.training or torch.is_grad_enabled():
        raise ValueError('H2O is inference-only: call eval() under torch.no_grad()')
    if any(hasattr(module, '_gos_h2o_session') for module in modules):
        raise RuntimeError('H2O session already active on this model')
    if lm.config.use_bidirectional_attention == 'all':
        raise ValueError('H2O requires causal text attention')
    previous = lm.config._attn_implementation
    session = {'config': config.validate(), 'layers': {}, 'position': 0}
    try:
        for module in modules:
            module._gos_h2o_session = session
        lm.set_attn_implementation('gos_h2o')
        yield session
    finally:
        lm.set_attn_implementation(previous)
        for module in modules:
            del module._gos_h2o_session
        session['layers'].clear()


@torch.no_grad()
def h2o_generate(model, input_ids, max_new_tokens=128, heavy_tokens=256, recent_tokens=256,
                 eos_token_id=None, max_seq_length=4096, temperature=0.0, kv_quantization="none", kernel="torch", policy="h2o"):
    """Stream prompt and completion, preserving absolute RoPE token positions.

    Returns (prompt+completion IDs, cache statistics). Prompt prefill is also
    streamed and evicted; attention workspace peaks at budget+1 per head.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError('Provide one nonempty, unpadded prompt')
    if max_new_tokens < 1 or max_seq_length > 4096 or input_ids.shape[1] + max_new_tokens > max_seq_length:
        raise ValueError('Prompt + completion must fit max_seq_length <= 4096')
    if temperature < 0:
        raise ValueError('Temperature must be nonnegative')
    if policy not in ('h2o', 'sliding'):
        raise ValueError('Cache policy must be h2o or sliding')
    if policy == 'sliding':
        heavy_tokens = 0
    lm = inner_lm(model)
    if eos_token_id is None:
        eos_token_id = lm.generation_config.eos_token_id
    eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    result = input_ids.clone()
    with h2o_session(model, H2OConfig(heavy_tokens, recent_tokens, kv_quantization, kernel)) as session:
        def step(token, position):
            session['position'] = position
            output = model(input_ids=token, position_ids=torch.tensor([[position]], device=token.device),
                attention_mask={'full_attention': None, 'sliding_attention': None},
                use_cache=False, logits_to_keep=1, return_dict=True)
            return output.logits[:, -1].float()
        for position in range(input_ids.shape[1]):
            logits = step(input_ids[:, position:position+1], position)
        for index in range(max_new_tokens):
            token = logits.argmax(-1, keepdim=True) if temperature == 0 else torch.multinomial(torch.softmax(logits / temperature, -1), 1)
            result = torch.cat((result, token), -1)
            if token.item() in eos or index + 1 == max_new_tokens:
                break
            logits = step(token, input_ids.shape[1] + index)
        stats = {'cache_policy': policy, 'kv_quantization': kv_quantization, 'kernel': kernel, 'layers': len(session['layers']),
                 'evicted_per_head_total': sum(cache.evicted for cache in session['layers'].values()),
                 'peak_tokens_per_head': max(cache.peak_tokens for cache in session['layers'].values()),
                 'retained_tokens_per_head': max(cache.keys.shape[-2] for cache in session['layers'].values()),
                 'budget_per_head': heavy_tokens + recent_tokens,
                 'resident_kv_storage_bytes': sum(cache.storage_bytes() for cache in session['layers'].values())}
    return result, stats
