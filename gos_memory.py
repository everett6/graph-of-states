"""Pure, per-token recursive memory: HRR binding, bounded slots and thermal halting."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class HolographicScratchpad(nn.Module):
    """Superpose role-bound vectors using real unitary circular convolution.

    Memory belongs to one forward invocation, never to the module. Unbinding
    one entry is exact; superposed entries incur the usual HRR interference.
    """
    def __init__(self, width, roles, decay=0.8):
        super().__init__()
        self.width, self.decay = width, decay
        self.phase = nn.Parameter(torch.empty(roles, width // 2 + 1).uniform_(-math.pi, math.pi))

    def spectrum(self):
        # Real DC/Nyquist bins are required for a real, unitary convolution.
        mask = torch.ones_like(self.phase, dtype=torch.float32)
        mask[:, 0] = 0
        if self.width % 2 == 0:
            mask[:, -1] = 0
        phase = self.phase.float() * mask
        return torch.polar(torch.ones_like(phase), phase)

    def bind(self, values):
        return torch.fft.irfft(torch.fft.rfft(values.float(), dim=-1) * self.spectrum(), n=self.width, dim=-1).to(values.dtype)

    def read(self, memory):
        return torch.fft.irfft(torch.fft.rfft(memory.float(), dim=-1)[:, None] * self.spectrum().conj(), n=self.width, dim=-1).to(memory.dtype)

    def write(self, memory, values):
        return self.decay * memory + (1 - self.decay) * self.bind(values).mean(1)


class RecurrentLatentCompactor(nn.Module):
    """Fold successive graph-state tokens into a fixed number of latent slots.

    Every round combines the existing slots with new state tokens. Slot count
    stays constant rather than retaining rounds*nodes historical tokens.
    This is reasoning-history compression, not transformer token/KV merging.
    """
    def __init__(self, width, slots):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, memory, tokens):
        candidates = torch.cat((memory, tokens), 1)
        weights = torch.softmax(self.queries @ self.key(candidates).transpose(-1, -2) / math.sqrt(tokens.shape[-1]), -1)
        proposal = weights @ self.value(candidates)
        gate = torch.sigmoid(self.gate(torch.cat((memory, proposal), -1)))
        return self.norm(memory + gate * (proposal - memory))


def thermal_statistics(routes, previous, state, previous_entropy, initial_temperature, floor):
    """Temperature cools with entropy change; halt requires entropy AND motion."""
    probabilities = routes.float().clamp_min(1e-8)
    normalizer = math.log(routes.shape[-1]) if routes.shape[-1] > 1 else 1.0
    entropy = -(probabilities * probabilities.log()).sum(-1).mean(-1) / normalizer
    entropy_change = (entropy - previous_entropy).abs()
    motion = (state.float() - previous.float()).square().mean((1, 2)).sqrt()
    temperature = floor + (initial_temperature - floor) * (entropy + entropy_change).clamp(0, 1)
    return entropy, entropy_change, motion, temperature
