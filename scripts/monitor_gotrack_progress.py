#!/usr/bin/env python3
"""Monitor durable GoTrack tracking progress files.

Usage:
    python scripts/monitor_gotrack_progress.py --trial-dir <trial>
    python scripts/monitor_gotrack_progress.py --output-dir <trial>/object_tracking/gotrack_output --watch
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def _load(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": repr(exc), "_path": str(path)}


def _out_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    if args.trial_dir:
        return Path(args.trial_dir).expanduser() / "object_tracking/gotrack_output"
    raise SystemExit("Provide --trial-dir or --output-dir")


def _fmt_age(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}s"
    except Exception:
        return str(value)


def _status_rank(status: str) -> int:
    order = {
        "failed": 0,
        "daemon_missing": 1,
        "stale": 2,
        "start_sent": 3,
        "init_sent": 4,
        "running": 5,
        "stopping": 6,
        "complete": 7,
        "not_started": 8,
    }
    return order.get(status, 9)


def render(out_dir: Path) -> str:
    state = _load(out_dir / "state.json", {})
    pc_status = _load(out_dir / "capture_pc_status.json", {"pcs": {}})
    summary = _load(out_dir / "summary.json", {})

    lines = []
    now_str = time.strftime("%H:%M:%S")
    phase = state.get("phase", summary.get("status", "unknown"))
    obj = state.get("obj") or summary.get("obj") or "-"
    trial = Path(state.get("trial_dir", "")).name if state.get("trial_dir") else "-"
    fps = float(state.get("fps", 0.0) or 0.0)
    ok = int(state.get("frames_success", summary.get("frames_success", 0)) or 0)
    received = int(state.get("frames_received", ok) or 0)
    failed = int(state.get("frames_failed", max(0, received - ok)) or 0)
    last_fid = state.get("last_frame_id", summary.get("last_frame_index", "-"))

    lines.append(f"GoTrack tracking  {now_str}")
    lines.append(
        f"{obj}/{trial}  [{phase}]  frames ok={ok} fail={failed} "
        f"received={received} last={last_fid} fps={fps:.2f}"
    )

    pcs: Dict[str, Dict[str, Any]] = pc_status.get("pcs", {}) if isinstance(pc_status, dict) else {}
    if not pcs:
        lines.append("capture PCs: no status yet")
    else:
        for pc, st in sorted(pcs.items(), key=lambda kv: (_status_rank(str(kv[1].get("status", ""))), kv[0])):
            status = str(st.get("status", st.get("phase", "-")))
            frame = st.get("last_frame_id")
            frames = st.get("frames_received", 0)
            age = _fmt_age(st.get("last_obs_age_s"))
            daemon = st.get("daemon_count")
            daemon_s = "-" if daemon is None else str(daemon)
            line = (
                f"{pc:<8} {status:<15} frame={str(frame):>8} "
                f"age={age:>7} frames={str(frames):>5} daemon={daemon_s}"
            )
            if st.get("last_command_result", {}).get("error"):
                line += f" error={st['last_command_result']['error']}"
            lines.append(line)

    fail_by = state.get("fail_by_reason") or {}
    if fail_by:
        fail_s = ", ".join(f"{k}={v}" for k, v in sorted(fail_by.items()))
        lines.append(f"fail reasons: {fail_s}")

    files = state.get("output_files") or {}
    poses_final = files.get("poses_final", str(out_dir / "world_pose_records.json"))
    poses_live = files.get("poses_live", str(out_dir / "world_pose_records.jsonl"))
    if Path(poses_final).exists():
        lines.append(f"output: {poses_final}")
    else:
        lines.append(f"output: {poses_live}")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trial-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    out_dir = _out_dir(args)

    if args.watch:
        try:
            while True:
                os.system("clear")
                print(render(out_dir), flush=True)
                time.sleep(max(0.2, args.interval))
        except KeyboardInterrupt:
            return 130
    print(render(out_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
