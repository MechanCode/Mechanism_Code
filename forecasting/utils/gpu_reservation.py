"""Keep a visible CUDA memory reservation for a process lifetime."""

from __future__ import annotations

import gc

import torch


def reserve_gpu_memory(target_mb: int, safety_mb: int = 1024, chunk_mb: int = 256) -> int:
    """Grow PyTorch's CUDA cache to target_mb and return reserved MiB."""
    if target_mb <= 0:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("GPU memory reservation requested, but CUDA is unavailable")

    torch.cuda.set_device(0)
    torch.cuda.init()
    mib = 1024 * 1024
    target_bytes = target_mb * mib
    reserved_bytes = torch.cuda.memory_reserved(0)
    needed_bytes = max(0, target_bytes - reserved_bytes)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    safety_bytes = safety_mb * mib

    if needed_bytes > max(0, free_bytes - safety_bytes):
        raise RuntimeError(
            "Cannot establish GPU reservation: "
            f"target={target_mb} MiB, currently_reserved={reserved_bytes // mib} MiB, "
            f"free={free_bytes // mib} MiB, total={total_bytes // mib} MiB, "
            f"required_safety_margin={safety_mb} MiB"
        )

    blocks: list[torch.Tensor] = []
    remaining = needed_bytes
    chunk_bytes = max(1, chunk_mb) * mib
    try:
        while remaining > 0:
            size = min(chunk_bytes, remaining)
            blocks.append(torch.empty(size, dtype=torch.uint8, device="cuda:0"))
            remaining -= size
    except Exception:
        del blocks
        gc.collect()
        torch.cuda.empty_cache()
        raise

    del blocks
    gc.collect()
    reserved_mb = torch.cuda.memory_reserved(0) // mib
    print(
        f"[GPU reservation] PyTorch CUDA cache holds {reserved_mb} MiB "
        f"(requested {target_mb} MiB); cache will remain until process exit."
    )
    return int(reserved_mb)
