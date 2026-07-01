"""Run visual overlay checks for GoTrack pose records."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from autodex.tracking.progress import atomic_write_json


def default_overlay_output_dir(trial_dir: Path) -> Path:
    return Path(trial_dir).expanduser() / "object_tracking" / "overlay_check"


def default_records_path(trial_dir: Path) -> Path:
    return Path(trial_dir).expanduser() / "object_tracking" / "gotrack_output" / "world_pose_records.json"


def default_cam_param_dir(trial_dir: Path) -> Path:
    return Path(trial_dir).expanduser() / "cam_param"


def discover_videos_dir(trial_dir: Path) -> Optional[Path]:
    trial_dir = Path(trial_dir).expanduser()
    candidates = [
        trial_dir / "videos",
        trial_dir / "raw" / "exec" / "videos",
        trial_dir / "raw" / "exec",
        trial_dir / "raw" / "place" / "videos",
        trial_dir / "raw" / "place",
        trial_dir / "raw" / "videos",
        trial_dir / "raw",
    ]
    for p in candidates:
        if p.is_dir() and any(p.glob("*.avi")):
            return p
    return None


def resolve_overlay_python(overlay_python: Optional[Path] = None) -> Path:
    if overlay_python is not None:
        return Path(overlay_python).expanduser()
    for env_key in ("AUTODEX_OVERLAY_PYTHON", "FPOSE_PY"):
        val = os.environ.get(env_key)
        if val:
            p = Path(val).expanduser()
            if p.exists():
                return p
    candidates = [
        Path.home() / "miniconda3/envs/foundationpose/bin/python",
        Path.home() / "anaconda3/envs/foundationpose/bin/python",
        Path(sys.executable),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(sys.executable)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_cam_param(src_dir: Path, out_dir: Path) -> Path:
    """Write a cam_param copy with keys expected by overlay_object_video_single.py."""
    src_dir = Path(src_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    intr = json.loads((src_dir / "intrinsics.json").read_text(encoding="utf-8"))
    extr = json.loads((src_dir / "extrinsics.json").read_text(encoding="utf-8"))
    norm: Dict[str, Dict[str, Any]] = {}
    for serial, data in intr.items():
        k_undist = (
            data.get("intrinsics_undistort")
            or data.get("K_undist")
            or data.get("K")
            or data.get("original_intrinsics")
        )
        if k_undist is None:
            raise KeyError(f"{serial}: no intrinsics_undistort/K_undist/K/original_intrinsics")
        item = dict(data)
        item["intrinsics_undistort"] = k_undist
        item.setdefault("original_intrinsics", data.get("K_orig", k_undist))
        item.setdefault("dist_params", data.get("dist_params", []))
        norm[serial] = item
    atomic_write_json(out_dir / "intrinsics.json", norm)
    atomic_write_json(out_dir / "extrinsics.json", extr)
    return out_dir


def run_overlay_check(
    trial_dir: Path,
    obj_name: str,
    mesh_path: Path,
    records_path: Optional[Path] = None,
    cam_param_dir: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    overlay_python: Optional[Path] = None,
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """Run the existing object overlay renderer and return a status dict."""
    trial_dir = Path(trial_dir).expanduser()
    output_dir = Path(output_dir).expanduser() if output_dir else default_overlay_output_dir(trial_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "overlay_status.json"
    log_path = output_dir / "overlay_check.log"

    records_path = Path(records_path).expanduser() if records_path else default_records_path(trial_dir)
    cam_param_dir = Path(cam_param_dir).expanduser() if cam_param_dir else default_cam_param_dir(trial_dir)
    videos_dir = Path(videos_dir).expanduser() if videos_dir else discover_videos_dir(trial_dir)
    overlay_python = resolve_overlay_python(overlay_python)

    started_at = time.time()
    base = {
        "status": "running",
        "started_at": started_at,
        "trial_dir": str(trial_dir),
        "obj": obj_name,
        "records_path": str(records_path),
        "cam_param_dir": str(cam_param_dir),
        "videos_dir": str(videos_dir) if videos_dir else None,
        "output_dir": str(output_dir),
        "overlay_python": str(overlay_python),
        "log_path": str(log_path),
    }
    atomic_write_json(status_path, base)

    if videos_dir is None:
        result = dict(base, status="failed", reason="videos_dir_not_found", finished_at=time.time())
        atomic_write_json(status_path, result)
        return result
    missing = [p for p in (records_path, cam_param_dir, videos_dir, Path(mesh_path).expanduser()) if not Path(p).exists()]
    if missing:
        result = dict(base, status="failed", reason="missing_inputs", missing=[str(p) for p in missing], finished_at=time.time())
        atomic_write_json(status_path, result)
        return result

    overlay_cam_param = _normalize_cam_param(cam_param_dir, output_dir / "_cam_param_overlay")
    script = _repo_root() / "src/visualization/overlay_object_video_single.py"
    cmd = [
        str(overlay_python),
        str(script),
        "--videos_dir",
        str(videos_dir),
        "--cam_param_dir",
        str(overlay_cam_param),
        "--gotrack_records",
        str(records_path),
        "--mesh",
        str(Path(mesh_path).expanduser()),
        "--output_dir",
        str(output_dir),
        "--alpha",
        str(float(alpha)),
    ]
    atomic_write_json(status_path, dict(base, status="running", command=cmd))
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    overlays = sorted(str(p) for p in output_dir.glob("overlay_*.mp4"))
    status = "ok" if proc.returncode == 0 and overlays else "failed"
    result = dict(
        base,
        status=status,
        returncode=int(proc.returncode),
        command=cmd,
        overlays=overlays,
        n_overlays=len(overlays),
        finished_at=time.time(),
        runtime_sec=time.time() - started_at,
    )
    if status != "ok":
        result["reason"] = "overlay_command_failed" if proc.returncode else "no_overlay_outputs"
    atomic_write_json(status_path, result)
    return result
