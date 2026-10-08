"""Offline GPU checks and measurements, without model/dataset downloads."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F

from gos_int4 import PackedInt4


def diagnose():
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    report = {'gpu': props.name, 'compute_capability': list(torch.cuda.get_device_capability(device)),
              'total_vram_bytes': props.total_memory, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'external_downloads': False}
    q = torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.bfloat16)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        report['pytorch_flash_sdpa'] = bool(torch.isfinite(output).all())
    except RuntimeError as error:
        report['pytorch_flash_sdpa'] = str(error)
    source = torch.randn(1, 4, 512, 128, device='cuda', dtype=torch.bfloat16)
    results = {}
    for kernel in ('torch', 'triton'):
        # Warm the kernels before measuring; include dequantization workspace.
        packed = PackedInt4.pack(source, kernel=kernel)
        packed.unpack(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(10):
            packed = PackedInt4.pack(source, kernel=kernel)
            recovered = packed.unpack()
        torch.cuda.synchronize()
        results[kernel] = {'pack_unpack_ms': (time.perf_counter()-start)*100,
            'persistent_int4_bytes': packed.storage_bytes(),
            'dense_bf16_bytes': source.numel()*source.element_size(),
            'max_absolute_error': (recovered-source).abs().max().item(),
            'peak_allocated_bytes_including_workspace': torch.cuda.max_memory_allocated()}
    report['int4'] = results
    from gos_cognition import AsynchronousMatrixBranches
    branch = AsynchronousMatrixBranches(128, 4, asynchronous=True).cuda().eval()
    states = torch.randn(32, 4, 128, device='cuda')
    timings = {}
    with torch.no_grad():
        for mode in (False, True):
            for _ in range(3):
                branch(states, allow_async=mode)
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(20):
                branch(states, allow_async=mode)
            torch.cuda.synchronize()
            timings['cuda_streams' if mode else 'batched_matrix'] = (time.perf_counter()-start)*50
    report['branch_ms'] = timings
    report['flash_attention_3'] = 'Hopper SM90 only; disabled on SM120'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/gpu-diagnostics.json')
    args = parser.parse_args()
    report = diagnose()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
