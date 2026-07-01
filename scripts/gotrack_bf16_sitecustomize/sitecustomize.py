"""Patch xformers attention for GoTrack subprocesses on Blackwell GPUs.

This module is imported automatically by Python when its directory is prepended
to PYTHONPATH. The batch wrapper enables it only for the GoTrack tracking
subprocess, not for overlay rendering or anchor-bank generation.
"""
from __future__ import annotations

import os


if (
    os.environ.get("AUTODEX_GOTRACK_BF16_XFORMERS") == "1"
    or os.environ.get("AUTODEX_GOTRACK_BF16_AUTOCAST") == "1"
):
    try:
        import torch
        import xformers.ops as xops

        _autodex_original_mea = xops.memory_efficient_attention

        def _autodex_bf16_memory_efficient_attention(query, key, value, *args, **kwargs):
            if (
                isinstance(query, torch.Tensor)
                and query.is_cuda
                and query.dtype == torch.float32
            ):
                out = _autodex_original_mea(
                    query.to(torch.bfloat16),
                    key.to(torch.bfloat16),
                    value.to(torch.bfloat16),
                    *args,
                    **kwargs,
                )
                return out.to(torch.float32)
            return _autodex_original_mea(query, key, value, *args, **kwargs)

        xops.memory_efficient_attention = _autodex_bf16_memory_efficient_attention
        try:
            import xformers.ops.fmha as fmha

            fmha.memory_efficient_attention = _autodex_bf16_memory_efficient_attention
        except Exception:
            pass
        print("[autodex-bf16] patched xformers memory_efficient_attention fp32->bf16", flush=True)
    except Exception as exc:
        print(f"[autodex-bf16] failed to patch xformers attention: {exc!r}", flush=True)
        if os.environ.get("AUTODEX_GOTRACK_BF16_REQUIRED", "1") == "1":
            raise
