"""Packed symmetric INT4 KV storage; optional Triton packing/dequantization."""
import math
import torch


class PackedInt4:
    def __init__(self, data, scales, shape, group_size=32, kernel='torch', dtype=torch.bfloat16):
        self.data, self.scales, self.shape = data, scales, shape
        self.group_size, self.kernel, self.dtype = group_size, kernel, dtype

    @classmethod
    def pack(cls, tensor, group_size=32, kernel='torch'):
        if group_size < 2 or group_size % 2 or group_size & (group_size - 1):
            raise ValueError('INT4 group_size must be an even power of two')
        if kernel not in ('torch', 'triton'):
            raise ValueError('Unknown INT4 kernel')
        shape = tuple(tensor.shape)
        groups = math.ceil(shape[-1] / group_size)
        if kernel == 'triton':
            from gos_kernels import pack_int4
            data, scales = pack_int4(tensor.contiguous(), group_size)
        else:
            padded = torch.nn.functional.pad(tensor.float(), (0, groups * group_size - shape[-1]))
            grouped = padded.reshape(*shape[:-1], groups, group_size)
            scales = grouped.abs().amax(-1).clamp_min(1e-8) / 7
            quantized = (torch.floor(grouped / scales[..., None] + 0.5).clamp(-7, 7) + 8).to(torch.uint8).flatten(-2)
            data = quantized[..., 0::2] | (quantized[..., 1::2] << 4)
        return cls(data, scales, shape, group_size, kernel, tensor.dtype)

    def unpack(self):
        if self.kernel == 'triton':
            from gos_kernels import unpack_int4
            return unpack_int4(self)
        quantized = torch.stack((self.data & 15, self.data >> 4), -1).flatten(-2).float() - 8
        grouped = quantized.reshape(*self.shape[:-1], self.scales.shape[-1], self.group_size)
        return (grouped * self.scales[..., None]).flatten(-2)[..., :self.shape[-1]].to(self.dtype)

    def append(self, other):
        shape = self.shape[:-2] + (self.shape[-2] + other.shape[-2], self.shape[-1])
        return PackedInt4(torch.cat((self.data, other.data), -2), torch.cat((self.scales, other.scales), -2),
                          shape, self.group_size, self.kernel, self.dtype)

    def gather_tokens(self, indices):
        # Per-KV-head selection, batch size one.
        def gather(tensor):
            return tensor.gather(2, indices[None, :, :, None].expand(1, tensor.shape[1], indices.shape[-1], tensor.shape[-1])).contiguous()
        shape = self.shape[:-2] + (indices.shape[-1], self.shape[-1])
        return PackedInt4(gather(self.data), gather(self.scales), shape, self.group_size, self.kernel, self.dtype)

    def storage_bytes(self):
        return self.data.numel() * self.data.element_size() + self.scales.numel() * self.scales.element_size()
