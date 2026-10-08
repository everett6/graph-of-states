"""Hardware checks and interchangeable activation checkpoint backends."""
import importlib.util
import torch
from torch.utils.checkpoint import checkpoint

_owns_process_group = False


def validate_attention_backend(name):
    if name not in ('sdpa', 'eager', 'flash_attention_3'):
        raise ValueError('attention_backend must be sdpa, eager or flash_attention_3')
    if name == 'flash_attention_3':
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise ValueError('FlashAttention-3 requires Hopper (SM90); RTX 5070 uses SM120. Select sdpa.')
        if importlib.util.find_spec('flash_attn_interface') is None:
            raise RuntimeError('Install the Hopper flash-attention package before selecting FlashAttention-3')


def configure_checkpointing(backend, cpu_offload=False):
    if backend == 'torch':
        if cpu_offload:
            raise ValueError('checkpoint_cpu_offload requires the deepspeed backend')
        return
    if backend != 'deepspeed':
        raise ValueError('checkpoint_backend must be torch or deepspeed')
    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError('Install the optional deepspeed extra to use its checkpoint backend') from error
    if not torch.cuda.is_available():
        raise ValueError('This DeepSpeed checkpoint integration requires CUDA')
    global _owns_process_group
    if not deepspeed.comm.is_initialized():
        if torch.distributed.is_initialized():
            deepspeed.init_distributed(auto_mpi_discovery=False)
        else:
            # Single-process rendezvous without requiring user MPI/environment.
            import tempfile
            from pathlib import Path
            rendezvous = tempfile.NamedTemporaryFile(prefix='gos-ds-', delete=False)
            rendezvous.close()
            try:
                deepspeed.init_distributed(dist_backend='gloo', auto_mpi_discovery=False,
                    init_method=Path(rendezvous.name).as_uri(), rank=0, world_size=1)
                _owns_process_group = True
            finally:
                Path(rendezvous.name).unlink(missing_ok=True)
    # No engine/ZeRO wrapping of the frozen quantized backbone. The checkpoint
    # API operates only on small graph/vocabulary chunks, one GPU, no partitions.
    deepspeed.checkpointing.configure(mpu_=None, partition_activations=False,
        contiguous_checkpointing=False, checkpoint_in_cpu=cpu_offload,
        synchronize=False, profile=False)


def checkpoint_call(function, arguments, backend='torch'):
    if backend == 'torch':
        return checkpoint(function, *arguments, use_reentrant=False)
    if backend != 'deepspeed':
        raise ValueError('Unknown checkpoint backend')
    import deepspeed
    # Non-reentrant checkpointing supports detached inputs with trainable
    # parameters captured by the graph/projection closure.
    values = deepspeed.checkpointing.non_reentrant_checkpoint(function, *arguments)
    return (values,) if isinstance(values, torch.Tensor) else values


def cleanup_checkpointing():
    global _owns_process_group
    if _owns_process_group and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    _owns_process_group = False
