"""Episode-level dynamic scheduling for offline GoTrack processing.

This module is intentionally file-based so multiple capture PCs can coordinate
through the shared AutoDex result directory without a long-lived queue server.
Workers claim episodes by atomically creating a lock directory.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from autodex.tracking.progress import append_jsonl, atomic_write_json, utc_ts


DEFAULT_PCS = ["capture1", "capture2", "capture3", "capture5", "capture6"]
DEFAULT_EXPERIMENT_ROOT = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
DEFAULT_OVERLAY_ROOT = Path.home() / "shared_data" / "AutoDex" / "object_overlay_video"
DEFAULT_SCHEDULE_ROOT = Path.home() / "shared_data" / "AutoDex" / "object_tracking" / "episode_scheduler"
GOTRACK_REL = Path("object_tracking") / "gotrack_output"

GOTRACK_PROGRESS_RE = re.compile(r"gotrack_anchor_mv:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")
OVERLAY_PROGRESS_RE = re.compile(r"\[overlay_progress\]\s+(\d+)/(\d+)")
TQDM_FRAME_PROGRESS_RE = re.compile(r"frames:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


def safe_task_id(hand: str, obj: str, episode: str) -> str:
    raw = f"{hand}__{obj}__{episode}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def safe_task_id_from_parts(parts: Iterable[str]) -> str:
    raw = "__".join(str(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def gotrack_done(episode_dir: Path) -> bool:
    rec = Path(episode_dir) / GOTRACK_REL / "world_pose_records.json"
    if not rec.exists():
        return False
    try:
        records = json.loads(rec.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(records, list) and any(
        isinstance(r, dict) and r.get("pose_world") is not None for r in records
    )


def overlay_files(overlay_dir: Path) -> List[str]:
    if not Path(overlay_dir).is_dir():
        return []
    return sorted(str(p.resolve()) for p in Path(overlay_dir).glob("overlay_*.mp4"))


def overlay_done(overlay_dir: Path, serials: Iterable[str]) -> bool:
    serials = list(serials)
    if not serials:
        return False
    return all((Path(overlay_dir) / f"overlay_{s}.mp4").exists() for s in serials)


def discover_episodes(
    experiment_root: Path,
    hand: str,
    objects: Optional[Iterable[str]] = None,
    episodes: Optional[Iterable[str]] = None,
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    """Discover saved video episodes under ``<root>/<hand>/<obj>/<episode>``."""
    root = Path(experiment_root).expanduser()
    object_filter = set(objects or [])
    episode_filter = set(episodes or [])
    if recursive or hand in {"all", "*"}:
        tasks: List[Dict[str, Any]] = []
        hand_names = {"allegro", "inspire"}
        for videos_dir in sorted(root.rglob("videos")):
            ep = videos_dir.parent
            if not ep.is_dir():
                continue
            rel_parts = ep.relative_to(root).parts
            if len(rel_parts) < 3:
                continue
            episode = ep.name
            if episode_filter and episode not in episode_filter:
                continue
            hand_index = None
            for idx in range(len(rel_parts) - 2):
                if rel_parts[idx] in hand_names:
                    hand_index = idx
                    break
            if hand_index is None or hand_index + 2 >= len(rel_parts):
                continue
            task_hand = rel_parts[hand_index]
            obj = rel_parts[hand_index + 1]
            if object_filter and obj not in object_filter:
                continue
            pose_world = ep / "pose_world.npy"
            cam_param = ep / "cam_param" / "intrinsics.json"
            if not pose_world.exists() or not cam_param.exists():
                continue
            serials = sorted(p.stem for p in videos_dir.glob("*.avi"))
            if not serials:
                continue
            tasks.append({
                "task_id": safe_task_id_from_parts(rel_parts),
                "hand": task_hand,
                "obj": obj,
                "episode": episode,
                "episode_dir": str(ep),
                "videos_dir": str(videos_dir),
                "serials": serials,
                "experiment_rel": str(Path(*rel_parts)),
                "overlay_rel": str(Path(*rel_parts)),
            })
        return tasks

    hand_dir = root / hand
    object_names = list(objects or [])
    if not object_names and hand_dir.is_dir():
        object_names = sorted(p.name for p in hand_dir.iterdir() if p.is_dir())

    tasks: List[Dict[str, Any]] = []
    for obj in sorted(object_names):
        obj_dir = hand_dir / obj
        if not obj_dir.is_dir():
            continue
        for ep in sorted(obj_dir.iterdir()):
            if not ep.is_dir():
                continue
            if episode_filter and ep.name not in episode_filter:
                continue
            videos_dir = ep / "videos"
            pose_world = ep / "pose_world.npy"
            cam_param = ep / "cam_param" / "intrinsics.json"
            if not videos_dir.is_dir() or not pose_world.exists() or not cam_param.exists():
                continue
            serials = sorted(p.stem for p in videos_dir.glob("*.avi"))
            if not serials:
                continue
            task_id = safe_task_id(hand, obj, ep.name)
            tasks.append({
                "task_id": task_id,
                "hand": hand,
                "obj": obj,
                "episode": ep.name,
                "episode_dir": str(ep),
                "videos_dir": str(videos_dir),
                "serials": serials,
                "experiment_rel": str(Path(hand) / obj / ep.name),
                "overlay_rel": str(Path(hand) / obj / ep.name),
            })
    return tasks


@dataclass
class EpisodeScheduleStore:
    schedule_dir: Path
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT
    overlay_root: Path = DEFAULT_OVERLAY_ROOT
    stages: str = "both"
    stale_after_s: float = 3600.0
    schedule_id: str = field(default_factory=lambda: time.strftime("gotrack_ep_%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        self.schedule_dir = Path(self.schedule_dir).expanduser()
        self.experiment_root = Path(self.experiment_root).expanduser()
        self.overlay_root = Path(self.overlay_root).expanduser()
        self.tasks_dir = self.schedule_dir / "tasks"
        self.claims_dir = self.schedule_dir / "claims"
        self.workers_dir = self.schedule_dir / "workers"
        self.logs_dir = self.schedule_dir / "logs"
        self.events_path = self.schedule_dir / "events.jsonl"
        self.manifest_path = self.schedule_dir / "manifest.json"
        for d in (self.tasks_dir, self.claims_dir, self.workers_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        schedule_root: Path,
        experiment_root: Path,
        overlay_root: Path,
        hand: str,
        tasks: List[Dict[str, Any]],
        stages: str = "both",
        schedule_id: Optional[str] = None,
        force: bool = False,
    ) -> "EpisodeScheduleStore":
        schedule_id = schedule_id or time.strftime("gotrack_ep_%Y%m%d_%H%M%S")
        schedule_dir = Path(schedule_root).expanduser() / schedule_id
        if schedule_dir.exists() and not force:
            raise FileExistsError(f"schedule already exists: {schedule_dir}")
        if schedule_dir.exists() and force:
            shutil.rmtree(schedule_dir)
        store = cls(
            schedule_dir=schedule_dir,
            experiment_root=experiment_root,
            overlay_root=overlay_root,
            stages=stages,
            schedule_id=schedule_id,
        )
        manifest = {
            "schedule_id": schedule_id,
            "created_at": utc_ts(),
            "hand": hand,
            "stages": stages,
            "experiment_root": str(Path(experiment_root).expanduser()),
            "overlay_root": str(Path(overlay_root).expanduser()),
            "schedule_dir": str(schedule_dir),
            "n_tasks": len(tasks),
        }
        atomic_write_json(store.manifest_path, manifest)
        for task in tasks:
            store.write_task(store._initial_task(task))
        store.event("schedule_created", **manifest)
        return store

    @classmethod
    def open(cls, schedule_dir: Path) -> "EpisodeScheduleStore":
        manifest = load_json(Path(schedule_dir).expanduser() / "manifest.json", {})
        return cls(
            schedule_dir=Path(schedule_dir).expanduser(),
            experiment_root=Path(os.environ.get(
                "AUTODEX_EXPERIMENT_ROOT",
                manifest.get("experiment_root", DEFAULT_EXPERIMENT_ROOT),
            )),
            overlay_root=Path(os.environ.get(
                "AUTODEX_OVERLAY_ROOT",
                manifest.get("overlay_root", DEFAULT_OVERLAY_ROOT),
            )),
            stages=str(manifest.get("stages", "both")),
            schedule_id=str(manifest.get("schedule_id", Path(schedule_dir).name)),
        )

    def overlay_dir_for(self, task: Dict[str, Any]) -> Path:
        if task.get("overlay_rel"):
            return self.overlay_root / str(task["overlay_rel"])
        return self.overlay_root / str(task["hand"]) / str(task["obj"]) / str(task["episode"])

    def task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def claim_dir(self, task_id: str) -> Path:
        return self.claims_dir / f"{task_id}.lock"

    def _initial_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(task)
        overlay_dir = self.overlay_dir_for(out)
        gt_ok = gotrack_done(Path(out["episode_dir"]))
        ov_files = overlay_files(overlay_dir)
        ov_ok = overlay_done(overlay_dir, out.get("serials", []))
        if self.stages == "gotrack":
            status = "skipped_done" if gt_ok else "pending"
        elif self.stages == "overlay":
            status = "skipped_done" if ov_ok else "pending"
        else:
            status = "skipped_done" if gt_ok and ov_ok else "pending"
        out.update({
            "status": status,
            "phase": "complete" if status == "skipped_done" else "pending",
            "gotrack_done": gt_ok,
            "overlay_done": ov_ok,
            "overlay_output_dir": str(overlay_dir),
            "overlay_files": ov_files,
            "attempts": 0,
            "created_at": utc_ts(),
            "updated_at": utc_ts(),
        })
        return out

    def write_task(self, task: Dict[str, Any]) -> None:
        data = dict(task)
        data["updated_at"] = utc_ts()
        atomic_write_json(self.task_path(str(data["task_id"])), data)

    def load_task(self, task_id: str) -> Dict[str, Any]:
        return load_json(self.task_path(task_id), {})

    def tasks(self) -> List[Dict[str, Any]]:
        return [
            load_json(path, {})
            for path in sorted(self.tasks_dir.glob("*.json"))
            if path.is_file()
        ]

    def event(self, event: str, **fields: Any) -> None:
        item = {"ts": utc_ts(), "event": str(event)}
        item.update(fields)
        append_jsonl(self.events_path, item)

    def update_worker(self, worker_id: str, **fields: Any) -> None:
        data = load_json(self.workers_dir / f"{worker_id}.json", {})
        data.update(fields)
        data.setdefault("worker_id", worker_id)
        data["updated_at"] = utc_ts()
        atomic_write_json(self.workers_dir / f"{worker_id}.json", data)

    def workers(self) -> List[Dict[str, Any]]:
        return [
            load_json(path, {})
            for path in sorted(self.workers_dir.glob("*.json"))
            if path.is_file()
        ]

    def claim_next(self, worker_id: str, retry_failed: bool = False) -> Optional[Dict[str, Any]]:
        for task in self.tasks():
            task_id = str(task.get("task_id", ""))
            status = str(task.get("status", "pending"))
            if not task_id:
                continue
            if status in {"done", "skipped_done", "running"}:
                continue
            if status == "failed" and not retry_failed:
                continue
            lock_dir = self.claim_dir(task_id)
            try:
                os.mkdir(lock_dir)
            except FileExistsError:
                continue
            except OSError:
                continue
            now = utc_ts()
            claim = {
                "task_id": task_id,
                "worker_id": worker_id,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "claimed_at": now,
            }
            atomic_write_json(lock_dir / "claim.json", claim)
            task["status"] = "running"
            task["phase"] = "claimed"
            task["worker_id"] = worker_id
            task["claimed_at"] = now
            task["attempts"] = int(task.get("attempts", 0)) + 1
            self.write_task(task)
            self.event("task_claimed", task_id=task_id, worker_id=worker_id,
                       obj=task.get("obj"), episode=task.get("episode"))
            return task
        return None

    def reset_failed_claims(self) -> int:
        n_reset = 0
        for task in self.tasks():
            if str(task.get("status", "")) != "failed":
                continue
            task_id = str(task.get("task_id", ""))
            if not task_id:
                continue
            lock_dir = self.claim_dir(task_id)
            if lock_dir.exists():
                shutil.rmtree(lock_dir, ignore_errors=True)
                n_reset += 1
        if n_reset:
            self.event("failed_claims_reset", count=n_reset)
        return n_reset

    def refresh_outputs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        overlay_dir = self.overlay_dir_for(task)
        task = dict(task)
        task["gotrack_done"] = gotrack_done(Path(task["episode_dir"]))
        task["overlay_done"] = overlay_done(overlay_dir, task.get("serials", []))
        task["overlay_output_dir"] = str(overlay_dir)
        task["overlay_files"] = overlay_files(overlay_dir)
        return task


def build_episode_command(
    repo_dir: Path,
    hand: str,
    obj: str,
    episode: str,
    episode_dir: Optional[str] = None,
    overlay_output_dir: Optional[str] = None,
    task_id: Optional[str] = None,
    python_bin: str = sys.executable,
    track_cams: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[str]:
    batch_script = Path(repo_dir).expanduser() / "scripts" / "run_batch_object_overlay_with_env.py"
    if not batch_script.exists():
        batch_script = Path(repo_dir).expanduser() / "src" / "process" / "batch_object_overlay.py"
    if batch_script.name == "run_batch_object_overlay_with_env.py" and episode_dir:
        cmd = [
            str(python_bin),
            str(batch_script),
            "--episode-dir", str(episode_dir),
            "--hand", hand,
            "--obj", obj,
            "--ep", episode,
        ]
        if overlay_output_dir:
            cmd.extend(["--overlay-output-dir", str(overlay_output_dir)])
        if task_id:
            cmd.extend(["--cache-key", str(task_id)])
    else:
        cmd = [
            str(python_bin),
            str(batch_script),
            "--hand", hand,
            "--obj", obj,
            "--ep", episode,
        ]
    if track_cams:
        cmd.extend(["--track-cams", *track_cams])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_episode_task(
    store: EpisodeScheduleStore,
    task: Dict[str, Any],
    worker_id: str,
    repo_dir: Path,
    python_bin: str = sys.executable,
    track_cams: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    task_id = str(task["task_id"])
    log_path = store.logs_dir / f"{task_id}.{worker_id}.log"
    cmd = build_episode_command(
        repo_dir=repo_dir,
        hand=str(task["hand"]),
        obj=str(task["obj"]),
        episode=str(task["episode"]),
        episode_dir=str(task.get("episode_dir", "")),
        overlay_output_dir=str(store.overlay_dir_for(task)),
        task_id=task_id,
        python_bin=python_bin,
        track_cams=track_cams,
        dry_run=dry_run,
    )
    started = utc_ts()
    task.update({
        "status": "running",
        "phase": "dry_run" if dry_run else "gotrack_or_overlay",
        "started_at": started,
        "command": cmd,
        "log_path": str(log_path),
    })
    store.write_task(task)
    store.update_worker(worker_id, status="running", task_id=task_id,
                        obj=task.get("obj"), episode=task.get("episode"))
    store.event("task_started", task_id=task_id, worker_id=worker_id, command=cmd)

    if dry_run:
        time.sleep(0.05)
        result = dict(task, status="dry_run_done", phase="complete", returncode=0,
                      finished_at=utc_ts(), runtime_sec=utc_ts() - started)
        store.write_task(store.refresh_outputs(result))
        store.event("task_done", task_id=task_id, worker_id=worker_id, status="dry_run_done")
        return result

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    last_phase = None
    last_progress = 0
    failure_hint = None
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(repo_dir).expanduser()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            stripped = line.strip()
            phase = None
            done = total = None
            m = GOTRACK_PROGRESS_RE.search(stripped)
            if m:
                phase = "gotrack"
                done, total = int(m.group(1)), int(m.group(2))
            else:
                m = OVERLAY_PROGRESS_RE.search(stripped)
                if m:
                    phase = "overlay"
                    done, total = int(m.group(1)), int(m.group(2))
                else:
                    m = TQDM_FRAME_PROGRESS_RE.search(stripped)
                    if m:
                        if " overlay" in stripped:
                            phase = "overlay"
                        elif " gotrack" in stripped:
                            phase = "gotrack"
                        else:
                            phase = last_phase or "running"
                        done, total = int(m.group(1)), int(m.group(2))
            if phase:
                last_phase = phase
                last_progress = done or 0
                task.update({
                    "phase": phase,
                    "frame_done": done,
                    "frame_total": total,
                    "progress_ratio": (float(done) / float(total)) if total else None,
                    "last_line": stripped[-300:],
                })
                store.write_task(task)
            elif stripped:
                task["last_line"] = stripped[-300:]
                if "[fail]" in stripped or "Traceback" in stripped or stripped.startswith("ModuleNotFoundError"):
                    failure_hint = stripped[-300:]
                    task["failure_hint"] = failure_hint
                if last_phase:
                    task["phase"] = last_phase
                store.write_task(task)
        proc.wait()

    finished = utc_ts()
    task = store.refresh_outputs(task)
    if proc.returncode == 0:
        required_ok = (
            task.get("gotrack_done") if store.stages == "gotrack"
            else task.get("overlay_done") if store.stages == "overlay"
            else (task.get("gotrack_done") and task.get("overlay_done"))
        )
        status = "done" if required_ok else "failed"
        reason = None if required_ok else "command_succeeded_but_outputs_missing"
    else:
        status = "failed"
        reason = f"returncode={proc.returncode}"

    if status == "failed" and reason == "command_succeeded_but_outputs_missing" and failure_hint:
        reason = f"{reason}: {failure_hint}"

    task.update({
        "status": status,
        "phase": "complete" if status == "done" else "failed",
        "returncode": int(proc.returncode),
        "reason": reason,
        "failure_hint": failure_hint,
        "finished_at": finished,
        "runtime_sec": finished - started,
        "last_progress": last_progress,
    })
    store.write_task(task)
    store.update_worker(worker_id, status="idle", last_task_id=task_id,
                        last_status=status, task_id=None)
    store.event("task_done" if status == "done" else "task_failed",
                task_id=task_id, worker_id=worker_id, status=status, reason=reason)
    return task


def summarize_schedule(store: EpisodeScheduleStore) -> Dict[str, Any]:
    tasks = store.tasks()
    workers = store.workers()
    counts: Dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    done_like = counts.get("done", 0) + counts.get("skipped_done", 0) + counts.get("dry_run_done", 0)
    started = [
        float(t.get("started_at", t.get("created_at", 0)))
        for t in tasks
        if t.get("started_at") or t.get("created_at")
    ]
    finished = [
        float(t.get("finished_at", 0))
        for t in tasks
        if t.get("finished_at")
    ]
    now = utc_ts()
    t0 = min(started) if started else now
    elapsed = max(now - t0, 1e-6)
    rate = done_like / elapsed
    remaining = max(0, len(tasks) - done_like - counts.get("failed", 0))
    eta = remaining / rate if rate > 0 else None
    return {
        "now": now,
        "schedule_dir": str(store.schedule_dir),
        "manifest": load_json(store.manifest_path, {}),
        "counts": counts,
        "n_tasks": len(tasks),
        "done_like": done_like,
        "remaining": remaining,
        "elapsed_sec": elapsed,
        "throughput_eps_per_hour": rate * 3600.0,
        "eta_sec": eta,
        "tasks": tasks,
        "workers": workers,
        "last_finished_at": max(finished) if finished else None,
    }
