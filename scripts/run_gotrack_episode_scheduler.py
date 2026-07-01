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
from typing import List, Optional

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


def _remote_schedule_dir(local_schedule_dir: Path, override: Optional[str] = None) -> str:
    if override:
        return override
    local_shared = Path.home().expanduser() / "shared_data"
    try:
        rel = local_schedule_dir.expanduser().resolve().relative_to(local_shared.resolve())
        return "${HOME}/shared_data/" + rel.as_posix()
    except ValueError:
        return str(local_schedule_dir)


def _remote_arg(value: str) -> str:
    if value in {"$REPO", "$SCHEDULE"}:
        return f'"{value}"'
    return shlex.quote(value)


def _init(args: argparse.Namespace) -> int:
    tasks = discover_episodes(
        experiment_root=Path(args.experiment_root),
        hand=args.hand,
        objects=args.obj,
        episodes=args.ep,
        recursive=args.recursive or args.hand == "all",
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
    pcs: List[str] = list(args.pcs or DEFAULT_PCS)
    ssh_tty_pcs = set(args.ssh_tty_pcs or [])
    log_dir = schedule_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    remote_schedule = _remote_schedule_dir(schedule_dir, args.remote_schedule_dir)
    remote_repo = args.remote_repo_dir
    if args.retry_failed:
        store = EpisodeScheduleStore.open(schedule_dir)
        n_reset = store.reset_failed_claims()
        if n_reset:
            print(f"reset {n_reset} failed claim lock(s)")
    for pc in pcs:
        worker_cmd = [
            args.python,
            "scripts/run_gotrack_episode_scheduler.py",
            "--mode", "worker",
            "--schedule-dir", "$SCHEDULE",
            "--repo-dir", "$REPO",
            "--python", args.python,
            "--worker-id", pc,
        ]
        if args.retry_failed:
            worker_cmd.append("--retry-failed")
        if args.worker_dry_run:
            worker_cmd.append("--dry-run")
        if args.track_cams:
            worker_cmd.extend(["--track-cams", *args.track_cams])
        quoted = " ".join(_remote_arg(x) for x in worker_cmd)
        remote_log_name = f"{pc}.log"
        remote_parts = [
            "set -eu",
            f"REPO={remote_repo}",
            f"SCHEDULE={remote_schedule}",
            'export PATH="$HOME/anaconda3/bin:$HOME/miniconda3/bin:$PATH"',
            'mkdir -p "$SCHEDULE/launcher_logs"',
            'cd "$REPO"',
        ]
        if args.sync_git:
            remote_parts.extend([
                f"git fetch {shlex.quote(args.remote_url)} {shlex.quote(args.branch)}",
                f"git checkout -B {shlex.quote(args.branch)} FETCH_HEAD",
            ])
        remote_parts.append(f'nohup {quoted} > "$SCHEDULE/launcher_logs/{remote_log_name}" 2>&1 &')
        remote = "; ".join(remote_parts)
        ssh_cmd = ["ssh"]
        if pc in ssh_tty_pcs:
            ssh_cmd.append("-tt")
        ssh_cmd.extend([pc, remote])
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
    p.add_argument("--hand", default=None, choices=["allegro", "inspire", "all"])
    p.add_argument("--obj", nargs="+", default=None)
    p.add_argument("--ep", nargs="+", default=None)
    p.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    p.add_argument("--overlay-root", default=str(DEFAULT_OVERLAY_ROOT))
    p.add_argument("--schedule-root", default=str(DEFAULT_SCHEDULE_ROOT))
    p.add_argument("--schedule-id", default=None)
    p.add_argument("--schedule-dir", default=None)
    p.add_argument("--recursive", action="store_true",
                   help="With --mode init, discover all valid episodes recursively under --experiment-root.")
    p.add_argument("--stages", choices=["both", "gotrack", "overlay"], default="both")
    p.add_argument("--repo-dir", default=str(REPO_ROOT))
    p.add_argument("--remote-repo-dir", default="${HOME}/AutoDex",
                   help="With --mode launch, repo path expression on capture PCs. Default: ${HOME}/AutoDex")
    p.add_argument("--remote-schedule-dir", default=None,
                   help="With --mode launch, schedule path expression on capture PCs. Default maps ~/shared_data.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pcs", nargs="+", default=DEFAULT_PCS)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--ssh-tty-pcs", nargs="+", default=None,
                   help="With --mode launch, add -tt for PCs that require a TTY for SSH commands.")
    p.add_argument("--sync-git", action="store_true",
                   help="With --mode launch, fetch and checkout --branch on each capture PC before starting worker.")
    p.add_argument("--branch", default="tracking-session-progress")
    p.add_argument("--remote-url", default="https://github.com/hyunsu-go/AutoDex.git")
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
