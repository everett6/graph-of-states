"""Optional Triton INT4 kernels. No dependency on Triton in the CPU path."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, Q, S, WIDTH: tl.constexpr, GROUPS: tl.constexpr, G: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * G + tl.arange(0, G)
    values = tl.load(X + row * WIDTH + offsets, offsets < WIDTH, other=0).to(tl.float32)
    # Match PyTorch's float32 reciprocal multiplication, including tie cases.
    scale = tl.maximum(tl.max(tl.abs(values)), 1.0e-8) * (1.0 / 7.0)
    tl.store(S + row * GROUPS + group, scale)
    pairs = group * G + 2 * tl.arange(0, G // 2)
    lo = tl.load(X + row * WIDTH + pairs, pairs < WIDTH, other=0).to(tl.float32)
    hi = tl.load(X + row * WIDTH + pairs + 1, pairs + 1 < WIDTH, other=0).to(tl.float32)
    lo = (tl.minimum(tl.maximum(tl.floor(tl.div_rn(lo, scale) + 0.5), -7), 7) + 8).to(tl.uint8)
    hi = (tl.minimum(tl.maximum(tl.floor(tl.div_rn(hi, scale) + 0.5), -7), 7) + 8).to(tl.uint8)
    tl.store(Q + row * GROUPS * (G // 2) + pairs // 2, lo | (hi << 4))


@triton.jit
def _unpack(Q, S, Y, TOTAL: tl.constexpr, WIDTH: tl.constexpr, GROUPS: tl.constexpr,
            G: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // WIDTH
    col = offsets % WIDTH
    packed = tl.load(Q + row * GROUPS * (G // 2) + col // 2, offsets < TOTAL, other=0)
    scale = tl.load(S + row * GROUPS + col // G, offsets < TOTAL, other=0)
    value = ((packed >> ((col % 2) * 4)) & 15).to(tl.float32) - 8
    tl.store(Y + offsets, value * scale, offsets < TOTAL)


def pack_int4(tensor, group_size):
    if not tensor.is_cuda:
        raise ValueError('Triton INT4 kernels require CUDA')
    groups = triton.cdiv(tensor.shape[-1], group_size)
    data = torch.empty((*tensor.shape[:-1], groups * group_size // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((*tensor.shape[:-1], groups), device=tensor.device, dtype=torch.float32)
    _pack[(tensor.numel() // tensor.shape[-1], groups)](tensor, data, scales, tensor.shape[-1], groups, group_size, enable_fp_fusion=False)
    return data, scales


def unpack_int4(packed):
    result = torch.empty(packed.shape, device=packed.data.device, dtype=packed.dtype)
    _unpack[(triton.cdiv(result.numel(), 256),)](packed.data, packed.scales, result, result.numel(),
        packed.shape[-1], packed.scales.shape[-1], packed.group_size, 256)
    return result
