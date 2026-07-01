"""Durable progress files for live GoTrack tracking sessions.

The tracking loop should not depend on a terminal or an in-memory dashboard.
This module writes append-only events plus small atomic JSON snapshots so a
separate monitor can report progress even after a process restart.
"""
from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


GOTRACK_REL = Path("object_tracking") / "gotrack_output"
RUN_INDEX_REL = Path("object_tracking") / "gotrack_runs"


def utc_ts() -> float:
    return time.time()


def atomic_write_json(path: Path, data: Dict[str, Any] | List[Any]) -> None:
    """Write JSON via os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True, default=str))
        f.write("\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"status": "corrupt_jsonl_line", "raw": line})
    return out


def output_dir_for_trial(trial_dir: Path) -> Path:
    return Path(trial_dir).expanduser() / GOTRACK_REL


def default_run_index_dir() -> Path:
    return Path.home() / "shared_data/AutoDex" / RUN_INDEX_REL


def make_run_id(obj_name: str, trial_dir: Path | str) -> str:
    trial = str(Path(trial_dir).expanduser())
    digest = hashlib.sha1(trial.encode("utf-8")).hexdigest()[:12]
    return f"{obj_name}_{Path(trial).name}_{digest}"


def _load_latest_runs(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"updated_at": utc_ts(), "runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"updated_at": utc_ts(), "runs": {}}
        data.setdefault("runs", {})
        return data
    except Exception:
        return {"updated_at": utc_ts(), "runs": {}}


def upsert_run_index(record: Dict[str, Any], index_dir: Optional[Path] = None) -> None:
    """Append a run-state event and update the compact latest-run index.

    The append-only file answers "what happened over time"; the latest JSON
    answers "what is the current status of every run".
    """
    root = Path(index_dir).expanduser() if index_dir else default_run_index_dir()
    root.mkdir(parents=True, exist_ok=True)
    now = utc_ts()
    rec = dict(record)
    rec.setdefault("updated_at", now)
    rec.setdefault("run_id", make_run_id(str(rec.get("obj", "object")), str(rec.get("trial_dir", ""))))

    append_jsonl(root / "runs.jsonl", rec)

    latest_path = root / "runs_latest.json"
    latest = _load_latest_runs(latest_path)
    runs = latest.setdefault("runs", {})
    prev = runs.get(rec["run_id"], {}) if isinstance(runs.get(rec["run_id"]), dict) else {}
    merged = dict(prev)
    merged.update(rec)
    runs[rec["run_id"]] = merged
    latest["updated_at"] = now
    latest["index_dir"] = str(root)
    atomic_write_json(latest_path, latest)


def load_run_index(index_dir: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(index_dir).expanduser() if index_dir else default_run_index_dir()
    return _load_latest_runs(root / "runs_latest.json")


def world_pose_records_done(out_dir: Path) -> bool:
    """True if final GoTrack records exist and contain at least one pose."""
    rec_path = Path(out_dir) / "world_pose_records.json"
    if not rec_path.exists():
        return False
    try:
        records = json.loads(rec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(records, list):
        return False
    return any(isinstance(r, dict) and r.get("pose_world") is not None for r in records)


def normalize_pose_record(
    frame_id: int,
    pose_world: Any,
    info: Optional[Dict[str, Any]] = None,
    wall_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Record format consumed by existing overlay code.

    Existing overlay code indexes records by ``frame_index`` and reads
    ``pose_world``. Keep those fields stable and add extra tracking stats.
    """
    info = info or {}
    rec = {
        "frame_index": int(frame_id),
        "frame_id": int(frame_id),
        "wall_ts": float(wall_ts if wall_ts is not None else utc_ts()),
        "pose_world": pose_world.tolist() if hasattr(pose_world, "tolist") else pose_world,
        "status": "ok",
    }
    if "n_inliers" in info:
        rec["num_inlier_anchors"] = int(info.get("n_inliers", 0))
    if "n_triangulated" in info:
        rec["num_triangulated_anchors"] = int(info.get("n_triangulated", 0))
    if "mean_residual_mm" in info:
        rec["mean_anchor_fit_residual_mm"] = float(info.get("mean_residual_mm", -1.0))
    return rec


class TrackingProgressStore:
    """Owns all durable files for one tracking run."""

    def __init__(self, out_dir: Path, manifest: Optional[Dict[str, Any]] = None):
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        self.manifest_path = self.out_dir / "run_manifest.json"
        self.events_path = self.out_dir / "events.jsonl"
        self.state_path = self.out_dir / "state.json"
        self.capture_status_path = self.out_dir / "capture_pc_status.json"
        self.pose_jsonl_path = self.out_dir / "world_pose_records.jsonl"
        self.pose_json_path = self.out_dir / "world_pose_records.json"
        self.summary_path = self.out_dir / "summary.json"
        self._state: Dict[str, Any] = {}
        self._capture_status: Dict[str, Any] = {"updated_at": utc_ts(), "pcs": {}}
        if manifest is not None:
            self.write_manifest(manifest)

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        data = dict(manifest)
        data.setdefault("created_at", utc_ts())
        data.setdefault("output_dir", str(self.out_dir))
        data.setdefault("files", self.files())
        atomic_write_json(self.manifest_path, data)

    def files(self) -> Dict[str, str]:
        return {
            "manifest": str(self.manifest_path),
            "events": str(self.events_path),
            "state": str(self.state_path),
            "capture_pc_status": str(self.capture_status_path),
            "poses_live": str(self.pose_jsonl_path),
            "poses_final": str(self.pose_json_path),
            "summary": str(self.summary_path),
        }

    def event(self, event: str, **fields: Any) -> None:
        item = {"ts": utc_ts(), "event": str(event)}
        item.update(fields)
        append_jsonl(self.events_path, item)

    def update_state(self, **fields: Any) -> Dict[str, Any]:
        now = utc_ts()
        self._state.update(fields)
        self._state["updated_at"] = now
        self._state.setdefault("started_at", now)
        self._state.setdefault("output_files", self.files())
        atomic_write_json(self.state_path, self._state)
        return dict(self._state)

    def set_capture_pcs(self, pc_status: Dict[str, Dict[str, Any]]) -> None:
        self._capture_status = {"updated_at": utc_ts(), "pcs": pc_status}
        atomic_write_json(self.capture_status_path, self._capture_status)

    def update_capture_pc(self, pc_name: str, **fields: Any) -> None:
        pcs = dict(self._capture_status.get("pcs", {}))
        entry = dict(pcs.get(pc_name, {}))
        entry.update(fields)
        pcs[pc_name] = entry
        self.set_capture_pcs(pcs)

    def append_pose(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.pose_jsonl_path, record)

    def finalize_pose_records(self) -> List[Dict[str, Any]]:
        records = [
            r for r in load_jsonl(self.pose_jsonl_path)
            if isinstance(r, dict) and r.get("pose_world") is not None
        ]
        records.sort(key=lambda r: int(r.get("frame_index", r.get("frame_id", 0))))
        atomic_write_json(self.pose_json_path, records)
        return records

    def write_summary(self, summary: Dict[str, Any]) -> None:
        data = dict(summary)
        data.setdefault("finished_at", utc_ts())
        data.setdefault("output_dir", str(self.out_dir))
        data.setdefault("files", self.files())
        atomic_write_json(self.summary_path, data)


def summarize_records(records: Iterable[Dict[str, Any]], started_at: float) -> Dict[str, Any]:
    records = list(records)
    now = utc_ts()
    n = len(records)
    first = int(records[0]["frame_index"]) if records else None
    last = int(records[-1]["frame_index"]) if records else None
    elapsed = max(now - started_at, 1e-6)
    residuals = [
        float(r["mean_anchor_fit_residual_mm"])
        for r in records
        if r.get("mean_anchor_fit_residual_mm") is not None
    ]
    return {
        "status": "ok" if n > 0 else "no_pose_records",
        "frames_success": n,
        "first_frame_index": first,
        "last_frame_index": last,
        "runtime_sec": elapsed,
        "throughput_success_fps": n / elapsed,
        "mean_residual_mm": (sum(residuals) / len(residuals)) if residuals else None,
    }
