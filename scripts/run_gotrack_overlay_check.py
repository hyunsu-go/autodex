#!/usr/bin/env python3
"""Render overlay videos from an existing GoTrack tracking result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.tracking.overlay_check import run_overlay_check  # noqa: E402
from autodex.tracking.session import default_mesh_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trial-dir", required=True)
    p.add_argument("--obj-name", "--obj", dest="obj_name", required=True)
    p.add_argument("--mesh-path", default=None)
    p.add_argument("--records-path", default=None)
    p.add_argument("--cam-param-dir", default=None)
    p.add_argument("--videos-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--overlay-python", default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    mesh = Path(args.mesh_path).expanduser() if args.mesh_path else default_mesh_path(args.obj_name)
    result = run_overlay_check(
        trial_dir=Path(args.trial_dir),
        obj_name=args.obj_name,
        mesh_path=mesh,
        records_path=Path(args.records_path).expanduser() if args.records_path else None,
        cam_param_dir=Path(args.cam_param_dir).expanduser() if args.cam_param_dir else None,
        videos_dir=Path(args.videos_dir).expanduser() if args.videos_dir else None,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
        overlay_python=Path(args.overlay_python).expanduser() if args.overlay_python else None,
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
