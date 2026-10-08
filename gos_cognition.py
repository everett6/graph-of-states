"""Paired-state coherence and independent matrix branches for the GoS head."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossModalCoherence(nn.Module):
    """Fuse causal, pre-aligned modality features; does not supply an image encoder."""
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(2 * width, width)

    def forward(self, text, modalities, mask):
        # Include text as a valid fallback, even for missing modality evidence.
        candidates = torch.cat((text[:, None], modalities), 1)
        valid = torch.cat((torch.ones_like(mask[:, :1]), mask), 1)
        scores = (self.query(text)[:, None] * self.key(candidates)).sum(-1) / math.sqrt(text.shape[-1])
        weights = scores.masked_fill(~valid, float('-inf')).softmax(-1)
        context = (weights[..., None] * self.value(candidates)).sum(1)
        gate = self.gate(torch.cat((text, context), -1)).sigmoid()
        fused = torch.where(mask.any(-1, keepdim=True), text + gate * context, text)
        alignment = 1 - F.cosine_similarity(text[:, None].float(), modalities.float(), dim=-1)
        loss = (alignment * mask).sum() / mask.sum().clamp_min(1)
        return fused, loss


class AsynchronousMatrixBranches(nn.Module):
    """Independent node matrices; optional CUDA-stream dispatch during inference.

    A barrier precedes graph fusion. Training uses the same branch equations
    sequentially, preserving checkpoint/autograd semantics. Stream dispatch is
    an experiment: tiny branches may be faster on a single stream.
    """
    def __init__(self, width, nodes, asynchronous=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nodes, width, width) / math.sqrt(width))
        self.bias = nn.Parameter(torch.zeros(nodes, width))
        self.asynchronous = asynchronous
        self._streams = {}

    def forward(self, states, allow_async=False):
        if not allow_async or not self.asynchronous or not states.is_cuda or torch.is_grad_enabled():
            return torch.tanh(torch.einsum('bnd,ndh->bnh', states, self.weight) + self.bias)
        device = states.device
        if device not in self._streams:
            self._streams[device] = [torch.cuda.Stream(device=device) for _ in range(states.shape[1])]
        current = torch.cuda.current_stream(device)
        outputs = []
        for index, stream in enumerate(self._streams[device]):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                states.record_stream(stream)
                self.weight.record_stream(stream)
                self.bias.record_stream(stream)
                outputs.append(torch.tanh(states[:, index] @ self.weight[index] + self.bias[index]))
        for stream, output in zip(self._streams[device], outputs):
            current.wait_stream(stream)
            output.record_stream(current)
        return torch.stack(outputs, 1)
