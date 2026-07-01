"""Enable BF16 autocast for GoTrack subprocesses on Blackwell GPUs.

This module is imported automatically by Python when its directory is prepended
to PYTHONPATH. The batch wrapper enables it only for the GoTrack tracking
subprocess, not for overlay rendering or anchor-bank generation.
"""
from __future__ import annotations

import os


if os.environ.get("AUTODEX_GOTRACK_BF16_AUTOCAST") == "1":
    try:
        import torch

        if torch.cuda.is_available():
            _autodex_bf16_autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            _autodex_bf16_autocast.__enter__()
            print("[autodex-bf16] enabled torch.autocast(cuda, bfloat16)", flush=True)
        else:
            print("[autodex-bf16] CUDA unavailable; autocast not enabled", flush=True)
    except Exception as exc:
        print(f"[autodex-bf16] failed to enable autocast: {exc!r}", flush=True)
        if os.environ.get("AUTODEX_GOTRACK_BF16_REQUIRED", "1") == "1":
            raise
