#!/usr/bin/env python3
"""Run one live distributed GoTrack tracking session.

This is a standalone wrapper around the existing GoTrack daemon/tracker path.
It writes durable progress under:

    <trial_dir>/object_tracking/gotrack_output/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.tracking.session import DEFAULT_PCS, run_from_args  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial-dir", required=True,
                   help="Trial/episode directory containing pose_world.npy and cam_param/.")
    p.add_argument("--obj-name", "--obj", dest="obj_name", required=True,
                   help="AutoDex object name.")
    p.add_argument("--mesh-path", default=None,
                   help="Default: ~/shared_data/AutoDex/object/paradex/<obj>/raw_mesh/<obj>.obj")
    p.add_argument("--anchor-bank-path", default=None,
                   help="Default: AutoDex MV-GoTrack anchor_banks/<obj>.npz")
    p.add_argument("--init-pose-npy", default=None,
                   help="Default: <trial-dir>/pose_world.npy, fallback init_pose_world.npy")
    p.add_argument("--cam-param-dir", default=None,
                   help="Default: <trial-dir>/cam_param")
    p.add_argument("--output-dir", default=None,
                   help="Default: <trial-dir>/object_tracking/gotrack_output")

    p.add_argument("--pc-list", nargs="+", default=DEFAULT_PCS,
                   help="Capture PC names.")
    p.add_argument("--capture-ips", nargs="+", default=None,
                   help="Capture PC IPs, same order as --pc-list. If omitted, paradex get_pc_ip is used.")
    p.add_argument("--active-serials", nargs="+", default=None,
                   help="Optional camera serial subset. Default: all calibrated cameras.")

    p.add_argument("--port-obs", type=int, default=1235)
    p.add_argument("--port-prior", type=int, default=1236)
    p.add_argument("--port-cmd-track", type=int, default=6892)
    p.add_argument("--min-cams-per-frame", type=int, default=6)

    p.add_argument("--max-frames", type=int, default=-1,
                   help="-1 means unlimited until interrupted or --max-seconds.")
    p.add_argument("--max-seconds", type=float, default=-1.0,
                   help="-1 means unlimited.")
    p.add_argument("--progress-interval-s", type=float, default=0.5)
    p.add_argument("--stale-after-s", type=float, default=1.5,
                   help="Per-PC status becomes stale after this many seconds without obs.")
    p.add_argument("--run-index-dir", default=None,
                   help="Default: ~/shared_data/AutoDex/object_tracking/gotrack_runs")
    p.add_argument("--web-port", type=int, default=0,
                   help="Start existing in-process Flask dashboard on this port. 0 disables.")

    p.add_argument("--force", action="store_true",
                   help="Run even if world_pose_records.json already has pose records.")
    p.add_argument("--skip-daemon-check", action="store_true",
                   help="Do not SSH-check gotrack_daemon process count before running.")
    p.add_argument("--allow-missing-daemons", action="store_true",
                   help="Continue even if daemon check reports missing PCs.")
    p.add_argument("--daemon-check-timeout-s", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and write manifest/state without sending daemon commands.")
    p.add_argument("--run-overlay-check", action="store_true",
                   help="After tracking, render overlay videos for visual validation.")
    p.add_argument("--overlay-videos-dir", default=None,
                   help="Default: auto-detect <trial>/videos or raw/* video folders.")
    p.add_argument("--overlay-output-dir", default=None,
                   help="Default: <trial>/object_tracking/overlay_check")
    p.add_argument("--overlay-python", default=None,
                   help="Default: AUTODEX_OVERLAY_PYTHON/FPOSE_PY/foundationpose env/current Python.")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    p.add_argument("--overlay-strict", action="store_true",
                   help="Return failure if overlay rendering fails.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    result = run_from_args(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in ("ok", "skipped_done", "dry_run_done") else 1


if __name__ == "__main__":
    raise SystemExit(main())
