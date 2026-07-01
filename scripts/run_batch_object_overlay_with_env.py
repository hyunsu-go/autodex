#!/usr/bin/env python3
"""Run batch_object_overlay with local conda path auto-detection.

The original batch script is shared with older machines and keeps its
interpreter paths hard-coded. This wrapper lets the episode scheduler use the
same script without editing it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPT = REPO_ROOT / "src" / "process" / "batch_object_overlay.py"
BF16_SITECUSTOMIZE = REPO_ROOT / "scripts" / "gotrack_bf16_sitecustomize"


def _first_existing(env_keys: tuple[str, ...], candidates: tuple[Path, ...]) -> Path:
    for key in env_keys:
        value = os.environ.get(key)
        if value and Path(value).expanduser().exists():
            return Path(value).expanduser()
    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()
    missing = ", ".join(str(p.expanduser()) for p in candidates)
    raise FileNotFoundError(f"none of the candidate interpreters exist: {missing}")


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("autodex_batch_object_overlay", BATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepend_ld_library_path(*dirs: Path) -> None:
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    prepend = [str(d) for d in dirs if d.is_dir() and str(d) not in existing]
    if prepend:
        os.environ["LD_LIBRARY_PATH"] = ":".join(prepend + existing)


def _nccl_dirs_for_python(python_bin: Path) -> list[Path]:
    env_root = python_bin.parent.parent
    return [
        path
        for path in env_root.glob("lib/python*/site-packages/nvidia/nccl/lib")
        if (path / "libnccl.so.2").exists()
    ]


def _is_gotrack_tracking_command(cmd) -> bool:
    return any(str(part).endswith("run_multiview_gotrack_anchor_online.py") for part in list(cmd))


class _BatchSubprocessProxy:
    def __init__(self, module, gotrack_forward_precision: str):
        self._module = module
        self._gotrack_forward_precision = gotrack_forward_precision

    def __getattr__(self, name):
        return getattr(self._module, name)

    def Popen(self, cmd, *args, **kwargs):
        if self._gotrack_forward_precision == "bf16" and _is_gotrack_tracking_command(cmd):
            env = dict(os.environ)
            env.update(kwargs.get("env") or {})
            existing = env.get("PYTHONPATH", "")
            paths = [str(BF16_SITECUSTOMIZE)]
            if existing:
                paths.append(existing)
            env["PYTHONPATH"] = os.pathsep.join(paths)
            env["AUTODEX_GOTRACK_BF16_XFORMERS"] = "1"
            env.setdefault("AUTODEX_GOTRACK_BF16_REQUIRED", "1")
            kwargs["env"] = env
            print("[env] enabled GoTrack BF16 xformers shim", flush=True)
        return self._module.Popen(cmd, *args, **kwargs)


def main() -> int:
    home = Path.home()
    batch = _load_batch_module()
    batch.GOTRACK_PY = _first_existing(
        ("AUTODEX_GOTRACK_PY", "GOTRACK_PY"),
        (
            batch.GOTRACK_PY,
            home / "anaconda3" / "envs" / "gotrack_cu128" / "bin" / "python",
            home / "anaconda3" / "envs" / "gotrack" / "bin" / "python",
            home / "miniconda3" / "envs" / "gotrack" / "bin" / "python",
        ),
    )
    batch.FPOSE_PY = _first_existing(
        ("AUTODEX_FPOSE_PY", "FPOSE_PY"),
        (
            home / "anaconda3" / "envs" / "planner" / "bin" / "python",
            home / "anaconda3" / "envs" / "paradex" / "bin" / "python",
            home / "anaconda3" / "envs" / "gotrack_cu128" / "bin" / "python",
            home / "anaconda3" / "envs" / "foundationpose" / "bin" / "python",
            batch.FPOSE_PY,
            home / "miniconda3" / "envs" / "foundationpose" / "bin" / "python",
        ),
    )
    _prepend_ld_library_path(*_nccl_dirs_for_python(batch.GOTRACK_PY), *_nccl_dirs_for_python(batch.FPOSE_PY))
    gotrack_forward_precision = os.environ.get("AUTODEX_GOTRACK_FORWARD_PRECISION", "bf16").strip().lower()
    batch.subprocess = _BatchSubprocessProxy(batch.subprocess, gotrack_forward_precision)
    print(f"[env] GOTRACK_PY={batch.GOTRACK_PY}", flush=True)
    print(f"[env] FPOSE_PY={batch.FPOSE_PY}", flush=True)
    print(f"[env] AUTODEX_GOTRACK_FORWARD_PRECISION={gotrack_forward_precision}", flush=True)
    print(f"[env] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
    batch.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
