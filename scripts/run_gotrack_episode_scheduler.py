#!/usr/bin/env python3
"""Episode-level dynamic scheduler for offline GoTrack/overlay processing.

Typical flow:
  1. init a shared queue under ~/shared_data/AutoDex/object_tracking/episode_scheduler
  2. launch one worker on each capture PC via ssh
  3. watch the queue with scripts/gotrack_episode_dashboard.py
"""
from __future__ import annotations

import argparse
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.tracking.episode_queue import (  # noqa: E402
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_OVERLAY_ROOT,
    DEFAULT_PCS,
    DEFAULT_SCHEDULE_ROOT,
    EpisodeScheduleStore,
    discover_episodes,
    run_episode_task,
)


def _schedule_dir(args: argparse.Namespace) -> Path:
    if args.schedule_dir:
        return Path(args.schedule_dir).expanduser()
    if args.schedule_id:
        return Path(args.schedule_root).expanduser() / args.schedule_id
    raise SystemExit("--schedule-dir or --schedule-id is required for this mode")


def _init(args: argparse.Namespace) -> int:
    tasks = discover_episodes(
        experiment_root=Path(args.experiment_root),
        hand=args.hand,
        objects=args.obj,
        episodes=args.ep,
    )
    if args.dry_run:
        print(f"would create schedule for {len(tasks)} episodes")
        for task in tasks[:20]:
            print(f"  {task['hand']}/{task['obj']}/{task['episode']}  cams={len(task['serials'])}")
        if len(tasks) > 20:
            print(f"  ... {len(tasks) - 20} more")
        return 0
    store = EpisodeScheduleStore.create(
        schedule_root=Path(args.schedule_root),
        experiment_root=Path(args.experiment_root),
        overlay_root=Path(args.overlay_root),
        hand=args.hand,
        tasks=tasks,
        stages=args.stages,
        schedule_id=args.schedule_id,
        force=args.force,
    )
    print(store.schedule_dir)
    print(f"created {len(tasks)} episode tasks")
    return 0


def _worker(args: argparse.Namespace) -> int:
    store = EpisodeScheduleStore.open(_schedule_dir(args))
    worker_id = args.worker_id or socket.gethostname()
    repo_dir = Path(args.repo_dir).expanduser()
    store.update_worker(worker_id, status="starting", host=socket.gethostname(), pid=__import__("os").getpid())
    store.event("worker_started", worker_id=worker_id, host=socket.gethostname())
    n_done = n_failed = 0
    try:
        while True:
            task = store.claim_next(worker_id=worker_id, retry_failed=args.retry_failed)
            if task is None:
                store.update_worker(worker_id, status="idle", reason="no_claimable_tasks")
                break
            result = run_episode_task(
                store=store,
                task=task,
                worker_id=worker_id,
                repo_dir=repo_dir,
                python_bin=args.python,
                track_cams=args.track_cams,
                dry_run=args.dry_run,
            )
            if result.get("status") in {"done", "skipped_done", "dry_run_done"}:
                n_done += 1
            else:
                n_failed += 1
            if args.once:
                break
    finally:
        store.update_worker(worker_id, status="stopped", done=n_done, failed=n_failed)
        store.event("worker_stopped", worker_id=worker_id, done=n_done, failed=n_failed)
    print(f"worker {worker_id}: done={n_done} failed={n_failed}")
    return 0 if n_failed == 0 else 2


def _launch(args: argparse.Namespace) -> int:
    schedule_dir = _schedule_dir(args)
    repo_dir = Path(args.repo_dir).expanduser()
    pcs: List[str] = list(args.pcs or DEFAULT_PCS)
    log_dir = schedule_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for pc in pcs:
        worker_cmd = [
            args.python,
            str(repo_dir / "scripts" / "run_gotrack_episode_scheduler.py"),
            "--mode", "worker",
            "--schedule-dir", str(schedule_dir),
            "--repo-dir", str(repo_dir),
            "--python", args.python,
            "--worker-id", pc,
        ]
        if args.retry_failed:
            worker_cmd.append("--retry-failed")
        if args.worker_dry_run:
            worker_cmd.append("--dry-run")
        if args.track_cams:
            worker_cmd.extend(["--track-cams", *args.track_cams])
        quoted = " ".join(shlex.quote(x) for x in worker_cmd)
        remote_log = log_dir / f"{pc}.log"
        remote = f"cd {shlex.quote(str(repo_dir))} && nohup {quoted} > {shlex.quote(str(remote_log))} 2>&1 &"
        ssh_cmd = ["ssh", pc, remote]
        print("+ " + " ".join(shlex.quote(x) for x in ssh_cmd))
        if not args.dry_run:
            subprocess.run(ssh_cmd, check=True)
            time.sleep(0.2)
    return 0


def _status(args: argparse.Namespace) -> int:
    from autodex.tracking.episode_queue import summarize_schedule

    store = EpisodeScheduleStore.open(_schedule_dir(args))
    summary = summarize_schedule(store)
    counts = summary["counts"]
    print(f"schedule: {summary['schedule_dir']}")
    print(f"tasks: {summary['done_like']}/{summary['n_tasks']} done  remaining={summary['remaining']}")
    print("counts:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"throughput: {summary['throughput_eps_per_hour']:.2f} ep/hour")
    eta = summary.get("eta_sec")
    print(f"eta_sec: {eta:.0f}" if eta is not None else "eta_sec: -")
    for w in summary["workers"]:
        print(f"worker {w.get('worker_id')}: {w.get('status')} task={w.get('task_id')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["init", "worker", "launch", "status"], required=True)
    p.add_argument("--hand", default=None, choices=["allegro", "inspire"])
    p.add_argument("--obj", nargs="+", default=None)
    p.add_argument("--ep", nargs="+", default=None)
    p.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    p.add_argument("--overlay-root", default=str(DEFAULT_OVERLAY_ROOT))
    p.add_argument("--schedule-root", default=str(DEFAULT_SCHEDULE_ROOT))
    p.add_argument("--schedule-id", default=None)
    p.add_argument("--schedule-dir", default=None)
    p.add_argument("--stages", choices=["both", "gotrack", "overlay"], default="both")
    p.add_argument("--repo-dir", default=str(REPO_ROOT))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pcs", nargs="+", default=DEFAULT_PCS)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--track-cams", nargs="+", default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--worker-dry-run", action="store_true",
                   help="With --mode launch, make launched workers claim tasks without running GoTrack.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.mode == "init":
        if not args.hand:
            raise SystemExit("--hand is required for --mode init")
        return _init(args)
    if args.mode == "worker":
        return _worker(args)
    if args.mode == "launch":
        return _launch(args)
    if args.mode == "status":
        return _status(args)
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
