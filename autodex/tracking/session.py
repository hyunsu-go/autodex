"""Standalone live GoTrack tracking session orchestration.

This module intentionally does not modify ``GoTrackTracker`` or
``gotrack_daemon``. It wraps them with trial-directory outputs, durable
progress files, and per-capture-PC status snapshots.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq

from autodex.tracking.progress import (
    TrackingProgressStore,
    make_run_id,
    normalize_pose_record,
    output_dir_for_trial,
    summarize_records,
    upsert_run_index,
    world_pose_records_done,
)


logger = logging.getLogger(__name__)


DEFAULT_PCS = ["capture1", "capture2", "capture3", "capture5", "capture6"]
DEFAULT_PORT_OBS = 1235
DEFAULT_PORT_PRIOR = 1236
DEFAULT_PORT_CMD_TRACK = 6892


def _to_home_relative(path: Path | str) -> str:
    p = str(Path(path).expanduser())
    home = str(Path.home())
    if p.startswith(home + "/"):
        return "~/" + p[len(home) + 1:]
    return p


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_mesh_path(obj_name: str) -> Path:
    return Path.home() / "shared_data/AutoDex/object/paradex" / obj_name / "raw_mesh" / f"{obj_name}.obj"


def default_anchor_bank_path(obj_name: str) -> Path:
    return (
        _repo_root()
        / "autodex/perception/thirdparty/MV-GoTrack/anchor_banks"
        / f"{obj_name}.npz"
    )


def default_init_pose_path(trial_dir: Path) -> Path:
    trial_dir = Path(trial_dir).expanduser()
    primary = trial_dir / "pose_world.npy"
    if primary.exists():
        return primary
    return trial_dir / "init_pose_world.npy"


def default_cam_param_dir(trial_dir: Path) -> Path:
    return Path(trial_dir).expanduser() / "cam_param"


def resolve_capture_ips(pc_list: List[str], capture_ips: Optional[List[str]]) -> List[str]:
    if capture_ips:
        if len(capture_ips) != len(pc_list):
            raise ValueError("--capture-ips length must match --pc-list length")
        return list(capture_ips)
    try:
        from paradex.utils.system import get_pc_ip
    except Exception as exc:
        raise RuntimeError(
            "--capture-ips is required because paradex.utils.system.get_pc_ip "
            f"could not be imported: {exc}"
        ) from exc
    return [str(get_pc_ip(pc)) for pc in pc_list]


def _as_4x4(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 12:
        return np.vstack([arr.reshape(3, 4), [0, 0, 0, 1]])
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Expected 12 or 16 extrinsic values, got {arr.size}")


def load_calibration(cam_param_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray], int, int]:
    """Load AutoDex camera calibration in either system or trial-side layout."""
    cam_param_dir = Path(cam_param_dir).expanduser()
    intr_path = cam_param_dir / "intrinsics.json"
    extr_path = cam_param_dir / "extrinsics.json"
    if not intr_path.exists() or not extr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics/extrinsics under {cam_param_dir}")

    with open(intr_path, encoding="utf-8") as f:
        intr_raw = json.load(f)
    with open(extr_path, encoding="utf-8") as f:
        extr_raw = json.load(f)

    intrinsics_full: Dict[str, Dict[str, Any]] = {}
    for serial, d in intr_raw.items():
        if "K_undist" in d:
            k_undist = d["K_undist"]
        elif "intrinsics_undistort" in d:
            k_undist = d["intrinsics_undistort"]
        elif "K" in d:
            k_undist = d["K"]
        else:
            raise KeyError(f"{serial}: no K_undist/intrinsics_undistort/K in intrinsics")

        if "K_orig" in d:
            k_orig = d["K_orig"]
        elif "original_intrinsics" in d:
            k_orig = d["original_intrinsics"]
        else:
            k_orig = k_undist

        intrinsics_full[str(serial)] = {
            "K_orig": np.asarray(k_orig, dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(k_undist, dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d.get("dist_params", []), dtype=np.float64).reshape(-1),
            "width": int(d["width"]),
            "height": int(d["height"]),
        }

    extrinsics_full = {
        str(serial): _as_4x4(ext)
        for serial, ext in extr_raw.items()
    }

    sample = next(iter(intrinsics_full.values()))
    return intrinsics_full, extrinsics_full, int(sample["height"]), int(sample["width"])


def filter_calibration_to_serials(
    intrinsics_full: Dict[str, Dict[str, Any]],
    extrinsics_full: Dict[str, np.ndarray],
    serials: Optional[List[str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    if not serials:
        common = sorted(set(intrinsics_full) & set(extrinsics_full))
    else:
        wanted = {str(s) for s in serials}
        common = sorted(wanted & set(intrinsics_full) & set(extrinsics_full))
    return (
        {s: intrinsics_full[s] for s in common},
        {s: extrinsics_full[s] for s in common},
    )


def build_daemon_init_payload(
    obj_name: str,
    mesh_path: Path,
    anchor_bank_path: Path,
    intrinsics_full: Dict[str, Dict[str, Any]],
    extrinsics_full: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    intrinsics_payload = {
        s: {
            "K": np.asarray(v["K_undist"], dtype=np.float64).reshape(3, 3).tolist(),
            "K_orig": np.asarray(v["K_orig"], dtype=np.float64).reshape(3, 3).tolist(),
            "dist_params": np.asarray(v["dist_params"], dtype=np.float64).reshape(-1).tolist(),
            "width": int(v["width"]),
            "height": int(v["height"]),
        }
        for s, v in intrinsics_full.items()
    }
    extrinsics_payload = {
        s: np.asarray(v, dtype=np.float64).reshape(4, 4).tolist()
        for s, v in extrinsics_full.items()
    }
    return {
        "mesh_path": _to_home_relative(mesh_path),
        "anchor_bank_path": _to_home_relative(anchor_bank_path),
        "object_id": 1,
        "object_name": obj_name,
        "intrinsics": intrinsics_payload,
        "extrinsics": extrinsics_payload,
        "mesh_scale": 1.0,
        "unit_scale_mode": "auto",
        "num_iters": 1,
        "first_frame_num_iters": 5,
    }


def check_gotrack_daemon(pc_name: str, timeout_s: float = 3.0) -> Dict[str, Any]:
    cmd = [
        "ssh",
        "-o",
        f"ConnectTimeout={max(1, int(timeout_s))}",
        pc_name,
        "pgrep -fc 'python.*gotrack_daemon'",
    ]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s + 2.0,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "count": 0, "error": repr(exc)}
    raw = (proc.stdout or "").strip()
    try:
        count = int(raw.splitlines()[-1]) if raw else 0
    except ValueError:
        count = 0
    return {
        "ok": proc.returncode == 0 and count > 0,
        "count": count,
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "").strip(),
    }


class CaptureCommandClient:
    """REQ/REP client with per-PC response reporting.

    This mirrors paradex CommandSender's wire format but returns structured
    results so durable progress can identify which PC accepted a command.
    """

    def __init__(self, pc_list: List[str], capture_ips: List[str], port: int, timeout_ms: int = 60000):
        if len(pc_list) != len(capture_ips):
            raise ValueError("pc_list and capture_ips must have the same length")
        self.pc_list = list(pc_list)
        self.capture_ips = list(capture_ips)
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)

    def _send_one(self, pc: str, ip: str, cmd: str, wait: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        try:
            sock.connect(f"tcp://{ip}:{self.port}")
            sock.send_json({"command": cmd, "is_wait": bool(wait), "info": info})
            response = sock.recv_json()
            ok = response.get("state") != "error"
            return {"ok": bool(ok), "pc": pc, "ip": ip, "response": response}
        except Exception as exc:
            return {"ok": False, "pc": pc, "ip": ip, "error": repr(exc)}
        finally:
            sock.close()

    def send(self, cmd: str, wait: bool = False, info: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        info = info or {}
        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(self.pc_list) or 1) as ex:
            futures = {
                ex.submit(self._send_one, pc, ip, cmd, wait, info): pc
                for pc, ip in zip(self.pc_list, self.capture_ips)
            }
            for fut in as_completed(futures):
                pc = futures[fut]
                results[pc] = fut.result()
        return results


@dataclass
class TrackingSessionConfig:
    trial_dir: Path
    obj_name: str
    mesh_path: Optional[Path] = None
    anchor_bank_path: Optional[Path] = None
    init_pose_npy: Optional[Path] = None
    cam_param_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    pc_list: List[str] = field(default_factory=lambda: list(DEFAULT_PCS))
    capture_ips: Optional[List[str]] = None
    active_serials: Optional[List[str]] = None
    port_obs: int = DEFAULT_PORT_OBS
    port_prior: int = DEFAULT_PORT_PRIOR
    port_cmd_track: int = DEFAULT_PORT_CMD_TRACK
    min_cams_per_frame: int = 6
    max_frames: int = -1
    max_seconds: float = -1.0
    progress_interval_s: float = 0.5
    stale_after_s: float = 1.5
    run_index_dir: Optional[Path] = None
    force: bool = False
    check_daemons: bool = True
    allow_missing_daemons: bool = False
    daemon_check_timeout_s: float = 3.0
    web_port: int = 0
    dry_run: bool = False
    run_overlay_check: bool = False
    overlay_videos_dir: Optional[Path] = None
    overlay_output_dir: Optional[Path] = None
    overlay_python: Optional[Path] = None
    overlay_alpha: float = 0.5
    overlay_strict: bool = False

    def normalized(self) -> "TrackingSessionConfig":
        self.trial_dir = Path(self.trial_dir).expanduser()
        self.mesh_path = Path(self.mesh_path).expanduser() if self.mesh_path else default_mesh_path(self.obj_name)
        self.anchor_bank_path = (
            Path(self.anchor_bank_path).expanduser()
            if self.anchor_bank_path
            else default_anchor_bank_path(self.obj_name)
        )
        self.init_pose_npy = (
            Path(self.init_pose_npy).expanduser()
            if self.init_pose_npy
            else default_init_pose_path(self.trial_dir)
        )
        self.cam_param_dir = (
            Path(self.cam_param_dir).expanduser()
            if self.cam_param_dir
            else default_cam_param_dir(self.trial_dir)
        )
        self.output_dir = (
            Path(self.output_dir).expanduser()
            if self.output_dir
            else output_dir_for_trial(self.trial_dir)
        )
        self.run_index_dir = Path(self.run_index_dir).expanduser() if self.run_index_dir else None
        self.overlay_videos_dir = Path(self.overlay_videos_dir).expanduser() if self.overlay_videos_dir else None
        self.overlay_output_dir = Path(self.overlay_output_dir).expanduser() if self.overlay_output_dir else None
        self.overlay_python = Path(self.overlay_python).expanduser() if self.overlay_python else None
        return self


class GoTrackTrackingSession:
    def __init__(self, config: TrackingSessionConfig):
        self.cfg = config.normalized()
        self.capture_ips = resolve_capture_ips(self.cfg.pc_list, self.cfg.capture_ips)
        self.pc_by_ip = {ip: pc for pc, ip in zip(self.cfg.pc_list, self.capture_ips)}
        self.ip_by_pc = {pc: ip for pc, ip in zip(self.cfg.pc_list, self.capture_ips)}
        self.store: Optional[TrackingProgressStore] = None
        self.run_id = make_run_id(self.cfg.obj_name, self.cfg.trial_dir)
        self._pc_status: Dict[str, Dict[str, Any]] = {}
        self._pc_seen_fid: Dict[str, int] = {}
        self._pc_seen_count: Dict[str, int] = {}
        self._stop_progress = threading.Event()
        self._started_at = time.time()

    def _initial_pc_status(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        return {
            pc: {
                "ip": self.ip_by_pc[pc],
                "phase": "not_started",
                "status": "not_started",
                "daemon_seen": None,
                "daemon_count": None,
                "init_sent": False,
                "start_sent": False,
                "stop_sent": False,
                "first_obs_ts": None,
                "last_obs_ts": None,
                "last_obs_age_s": None,
                "last_frame_id": None,
                "frames_received": 0,
                "updated_at": now,
            }
            for pc in self.cfg.pc_list
        }

    def _write_pc_status(self) -> None:
        if self.store is not None:
            self.store.set_capture_pcs(self._pc_status)

    def _index_update(self, status: str, **fields: Any) -> None:
        pc_counts: Dict[str, int] = {}
        for st in self._pc_status.values():
            key = str(st.get("status", st.get("phase", "unknown")))
            pc_counts[key] = pc_counts.get(key, 0) + 1
        rec = {
            "run_id": self.run_id,
            "status": status,
            "phase": fields.pop("phase", status),
            "obj": self.cfg.obj_name,
            "trial_dir": str(self.cfg.trial_dir),
            "trial_name": self.cfg.trial_dir.name,
            "output_dir": str(self.cfg.output_dir),
            "pc_list": list(self.cfg.pc_list),
            "pc_status_counts": pc_counts,
        }
        rec.update(fields)
        try:
            upsert_run_index(rec, index_dir=self.cfg.run_index_dir)
        except Exception as exc:
            logger.warning(f"[run_index] update failed: {exc}")

    def _set_pc(self, pc: str, **fields: Any) -> None:
        entry = dict(self._pc_status.get(pc, {}))
        entry.update(fields)
        entry["updated_at"] = time.time()
        self._pc_status[pc] = entry

    def _preflight(self) -> None:
        required = {
            "trial_dir": self.cfg.trial_dir,
            "mesh_path": self.cfg.mesh_path,
            "anchor_bank_path": self.cfg.anchor_bank_path,
            "init_pose_npy": self.cfg.init_pose_npy,
            "cam_param_dir": self.cfg.cam_param_dir,
        }
        missing = [f"{k}: {v}" for k, v in required.items() if v is None or not Path(v).exists()]
        if missing:
            raise FileNotFoundError("Missing required tracking inputs:\n" + "\n".join(missing))

        if self.cfg.check_daemons:
            with ThreadPoolExecutor(max_workers=len(self.cfg.pc_list)) as ex:
                futures = {
                    ex.submit(check_gotrack_daemon, pc, self.cfg.daemon_check_timeout_s): pc
                    for pc in self.cfg.pc_list
                }
                for fut in as_completed(futures):
                    pc = futures[fut]
                    res = fut.result()
                    ok = bool(res.get("ok"))
                    self._set_pc(
                        pc,
                        daemon_seen=ok,
                        daemon_count=res.get("count"),
                        daemon_check=res,
                        status="not_started" if ok else "daemon_missing",
                        phase="not_started" if ok else "daemon_missing",
                    )
            self._write_pc_status()
            missing_daemons = [
                pc for pc, st in self._pc_status.items()
                if not st.get("daemon_seen")
            ]
            if missing_daemons and not self.cfg.allow_missing_daemons:
                raise RuntimeError(
                    "GoTrack daemon missing on: "
                    + ", ".join(missing_daemons)
                    + " (use --allow-missing-daemons to continue anyway)"
                )

    def _command(self, client: CaptureCommandClient, cmd: str, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        results = client.send(cmd, wait=False, info=info)
        for pc, res in results.items():
            ok = bool(res.get("ok"))
            if cmd == "init":
                self._set_pc(pc, init_sent=ok, phase="init_sent" if ok else "failed",
                             status="init_sent" if ok else "failed", last_command_result=res)
            elif cmd == "start":
                self._set_pc(pc, start_sent=ok, phase="start_sent" if ok else "failed",
                             status="start_sent" if ok else "failed", last_command_result=res)
            elif cmd == "stop":
                self._set_pc(pc, stop_sent=ok, phase="stopping" if ok else "failed",
                             status="stopping" if ok else "failed", last_command_result=res)
            else:
                self._set_pc(pc, last_command_result=res)
        self._write_pc_status()
        if self.store is not None:
            n_ok = sum(1 for r in results.values() if r.get("ok"))
            self.store.event(f"daemon_{cmd}_sent", n_ok=n_ok, n_total=len(results), results=results)
        return results

    def _progress_loop(self, tracker: Any) -> None:
        while not self._stop_progress.is_set():
            self._snapshot_tracker_status(tracker)
            time.sleep(max(0.1, float(self.cfg.progress_interval_s)))
        self._snapshot_tracker_status(tracker)

    def _snapshot_tracker_status(self, tracker: Any) -> None:
        now = time.time()
        with tracker._status_lock:
            status = dict(tracker.status)
            per_pc = dict(status.get("per_pc_last_frame", {}))
            counts = dict(status.get("counts", {}))

        for ip, info in per_pc.items():
            pc = self.pc_by_ip.get(ip, ip)
            frame_id = int(info.get("frame_id", -1))
            ts = float(info.get("ts", now))
            if self._pc_seen_fid.get(pc) != frame_id:
                self._pc_seen_fid[pc] = frame_id
                self._pc_seen_count[pc] = self._pc_seen_count.get(pc, 0) + 1
            age = max(0.0, now - ts)
            first_obs_ts = self._pc_status.get(pc, {}).get("first_obs_ts") or ts
            stale = age > self.cfg.stale_after_s
            self._set_pc(
                pc,
                phase="stale" if stale else "running",
                status="stale" if stale else "running",
                first_obs_ts=first_obs_ts,
                last_obs_ts=ts,
                last_obs_age_s=age,
                last_frame_id=frame_id,
                frames_received=self._pc_seen_count.get(pc, 0),
            )

        self._write_pc_status()

        if self.store is not None:
            self.store.update_state(
                phase="tracking",
                obj=self.cfg.obj_name,
                trial_dir=str(self.cfg.trial_dir),
                expected_pcs=list(self.cfg.pc_list),
                capture_ips=list(self.capture_ips),
                last_frame_id=int(status.get("frame_id", -1)),
                fps=float(status.get("fps", 0.0)),
                frames_received=int(counts.get("received", 0)),
                frames_success=int(counts.get("success", 0)),
                frames_failed=max(0, int(counts.get("received", 0)) - int(counts.get("success", 0))),
                fail_by_reason=dict(counts.get("fail_by_reason", {})),
                last_fit_ok=status.get("last_fit_ok"),
                fail_reason=status.get("fail_reason"),
                per_pc_last_frame={
                    self.pc_by_ip.get(ip, ip): {
                        "frame_id": int(v.get("frame_id", -1)),
                        "age_s": max(0.0, now - float(v.get("ts", now))),
                    }
                    for ip, v in per_pc.items()
                },
            )

    def _mark_complete(self, ok: bool) -> None:
        for pc, st in list(self._pc_status.items()):
            if st.get("status") == "failed":
                continue
            if ok:
                self._set_pc(pc, phase="complete", status="complete")
            else:
                self._set_pc(pc, phase=st.get("phase", "failed"), status=st.get("status", "failed"))
        self._write_pc_status()

    def _run_overlay_check(self) -> Dict[str, Any]:
        from autodex.tracking.overlay_check import run_overlay_check
        result = run_overlay_check(
            trial_dir=self.cfg.trial_dir,
            obj_name=self.cfg.obj_name,
            mesh_path=self.cfg.mesh_path,
            records_path=Path(self.cfg.output_dir) / "world_pose_records.json",
            cam_param_dir=self.cfg.cam_param_dir,
            videos_dir=self.cfg.overlay_videos_dir,
            output_dir=self.cfg.overlay_output_dir,
            overlay_python=self.cfg.overlay_python,
            alpha=self.cfg.overlay_alpha,
        )
        if self.store is not None:
            self.store.event("overlay_check_done", status=result.get("status"),
                             output_dir=result.get("output_dir"),
                             n_overlays=result.get("n_overlays", 0))
            self.store.update_state(overlay_check=result)
        self._index_update(
            str(result.get("status", "unknown")),
            phase="overlay_check",
            overlay_status=result.get("status"),
            overlay_output_dir=result.get("output_dir"),
            overlay_n_outputs=result.get("n_overlays", 0),
            overlay_files=result.get("overlays", []),
        )
        if self.cfg.overlay_strict and result.get("status") != "ok":
            raise RuntimeError(f"overlay check failed: {result}")
        return result

    def run(self) -> Dict[str, Any]:
        out_dir = Path(self.cfg.output_dir)
        if world_pose_records_done(out_dir) and not self.cfg.force:
            overlay_result = self._run_overlay_check() if self.cfg.run_overlay_check else None
            summary_path = out_dir / "summary.json"
            frames_success = None
            if summary_path.exists():
                try:
                    frames_success = json.loads(summary_path.read_text(encoding="utf-8")).get("frames_success")
                except Exception:
                    frames_success = None
            self._index_update(
                "skipped_done",
                phase="done",
                frames_success=frames_success,
                reason="world_pose_records.json already has pose records",
                overlay_status=(overlay_result or {}).get("status"),
                overlay_output_dir=(overlay_result or {}).get("output_dir"),
                overlay_n_outputs=(overlay_result or {}).get("n_overlays"),
                overlay_files=(overlay_result or {}).get("overlays", []),
            )
            return {
                "status": "skipped_done",
                "reason": "world_pose_records.json already has pose records",
                "output_dir": str(out_dir),
                "overlay_check": overlay_result,
            }

        manifest = {
            "run_id": self.run_id,
            "obj": self.cfg.obj_name,
            "trial_dir": str(self.cfg.trial_dir),
            "pc_list": list(self.cfg.pc_list),
            "capture_ips": list(self.capture_ips),
            "ports": {
                "obs": self.cfg.port_obs,
                "prior": self.cfg.port_prior,
                "cmd_track": self.cfg.port_cmd_track,
            },
            "mesh_path": str(self.cfg.mesh_path),
            "anchor_bank_path": str(self.cfg.anchor_bank_path),
            "init_pose_npy": str(self.cfg.init_pose_npy),
            "cam_param_dir": str(self.cfg.cam_param_dir),
            "limits": {
                "min_cams_per_frame": int(self.cfg.min_cams_per_frame),
                "max_frames": int(self.cfg.max_frames),
                "max_seconds": float(self.cfg.max_seconds),
                "progress_interval_s": float(self.cfg.progress_interval_s),
                "stale_after_s": float(self.cfg.stale_after_s),
            },
        }
        self.store = TrackingProgressStore(out_dir, manifest=manifest)
        self._pc_status = self._initial_pc_status()
        self._write_pc_status()
        self.store.event("session_created", **manifest)
        self._index_update("created", phase="preflight", started_at=self._started_at)
        self.store.update_state(
            phase="preflight",
            obj=self.cfg.obj_name,
            trial_dir=str(self.cfg.trial_dir),
            expected_pcs=list(self.cfg.pc_list),
            capture_ips=list(self.capture_ips),
            limits=dict(manifest["limits"]),
            upload={"status": "not_started"},
        )

        self._preflight()
        self.store.event("preflight_done")
        self._index_update("preflight_done", phase="preflight_done")

        intrinsics_full, extrinsics_full, _, _ = load_calibration(self.cfg.cam_param_dir)
        intrinsics_full, extrinsics_full = filter_calibration_to_serials(
            intrinsics_full, extrinsics_full, self.cfg.active_serials
        )
        if not intrinsics_full:
            raise RuntimeError("No active cameras after calibration filtering")

        init_pose = np.load(self.cfg.init_pose_npy)
        if init_pose.shape != (4, 4):
            raise ValueError(f"init pose must be 4x4, got {init_pose.shape}")

        payload = build_daemon_init_payload(
            self.cfg.obj_name,
            self.cfg.mesh_path,
            self.cfg.anchor_bank_path,
            intrinsics_full,
            extrinsics_full,
        )
        if self.cfg.dry_run:
            self.store.update_state(phase="dry_run_done", n_cameras=len(intrinsics_full))
            self.store.write_summary({"status": "dry_run_done", "n_cameras": len(intrinsics_full)})
            self._index_update("dry_run_done", phase="dry_run_done", frames_success=0)
            return {"status": "dry_run_done", "output_dir": str(out_dir)}

        client = CaptureCommandClient(
            pc_list=self.cfg.pc_list,
            capture_ips=self.capture_ips,
            port=self.cfg.port_cmd_track,
        )

        self.store.update_state(phase="daemon_init", n_cameras=len(intrinsics_full))
        self._index_update("daemon_init", phase="daemon_init", n_cameras=len(intrinsics_full))
        self._command(client, "init", payload)

        from autodex.perception.gotrack_tracker import GoTrackTracker

        tracker = GoTrackTracker(
            capture_pc_ips=self.capture_ips,
            port_obs=self.cfg.port_obs,
            port_prior=self.cfg.port_prior,
            min_cams_per_frame=self.cfg.min_cams_per_frame,
        )
        with tracker._status_lock:
            tracker.status["obj_name"] = self.cfg.obj_name
        if self.cfg.web_port > 0:
            tracker.start_dashboard(self.cfg.web_port)

        progress_thread = threading.Thread(target=self._progress_loop, args=(tracker,), daemon=True)
        progress_thread.start()

        if self.cfg.max_seconds and self.cfg.max_seconds > 0:
            def _timer() -> None:
                time.sleep(float(self.cfg.max_seconds))
                tracker._stop.set()
                self.store.event("max_seconds_reached", max_seconds=self.cfg.max_seconds)
            threading.Thread(target=_timer, daemon=True).start()

        records: List[Dict[str, Any]] = []
        ok = False
        error: Optional[str] = None
        self._started_at = time.time()
        try:
            self.store.update_state(phase="daemon_start")
            self._index_update("daemon_start", phase="daemon_start")
            self._command(client, "start", {"trial_ts": self.cfg.trial_dir.name})
            self.store.update_state(phase="tracking")
            self.store.event("tracking_started")
            self._index_update("running", phase="tracking", frames_success=0, frames_received=0)

            for frame_id, pose_world, info in tracker.track(init_pose):
                rec = normalize_pose_record(frame_id, pose_world, info=info)
                self.store.append_pose(rec)
                records.append(rec)
                if len(records) == 1:
                    self.store.event("first_pose_written", frame_id=int(frame_id))
                elif len(records) % 30 == 0:
                    self.store.event("pose_progress", frames_success=len(records), frame_id=int(frame_id))
                    self._index_update(
                        "running",
                        phase="tracking",
                        frames_success=len(records),
                        last_frame_id=int(frame_id),
                    )
                if self.cfg.max_frames > 0 and len(records) >= self.cfg.max_frames:
                    self.store.event("max_frames_reached", max_frames=self.cfg.max_frames)
                    tracker._stop.set()
                    break
            ok = len(records) > 0
        except KeyboardInterrupt:
            error = "KeyboardInterrupt"
            self.store.event("interrupted")
            tracker._stop.set()
        except Exception as exc:
            error = repr(exc)
            self.store.event("session_error", error=error)
            tracker._stop.set()
            raise
        finally:
            try:
                self.store.update_state(phase="stopping")
                self._command(client, "stop", {})
            except Exception as exc:
                self.store.event("daemon_stop_error", error=repr(exc))
            self._stop_progress.set()
            progress_thread.join(timeout=2.0)
            tracker.close()

            final_records = self.store.finalize_pose_records()
            summary = summarize_records(final_records, self._started_at)
            if error:
                summary["status"] = "failed"
                summary["error"] = error
            elif not ok:
                summary["status"] = "failed_no_pose_records"
            summary["frames_failed"] = max(
                0,
                int(self.store._state.get("frames_received", 0)) - int(summary.get("frames_success", 0)),
            )
            summary["pc_status"] = self._pc_status
            self.store.write_summary(summary)
            self._mark_complete(ok=summary.get("status") == "ok")
            self.store.update_state(
                phase="done" if summary.get("status") == "ok" else "failed",
                summary=summary,
            )
            self.store.event("session_done", status=summary.get("status"), frames_success=len(final_records))
            overlay_result = None
            if self.cfg.run_overlay_check and summary.get("status") == "ok":
                try:
                    overlay_result = self._run_overlay_check()
                except Exception as exc:
                    overlay_result = {"status": "failed", "error": repr(exc)}
                    self.store.event("overlay_check_error", error=repr(exc))
                    if self.cfg.overlay_strict:
                        raise
                summary["overlay_check"] = overlay_result
                self.store.write_summary(summary)
            self._index_update(
                "done" if summary.get("status") == "ok" else "failed",
                phase="done" if summary.get("status") == "ok" else "failed",
                frames_success=len(final_records),
                frames_failed=summary.get("frames_failed"),
                finished_at=summary.get("finished_at"),
                summary_status=summary.get("status"),
                overlay_status=(overlay_result or {}).get("status"),
                overlay_output_dir=(overlay_result or {}).get("output_dir"),
                overlay_n_outputs=(overlay_result or {}).get("n_overlays"),
                overlay_files=(overlay_result or {}).get("overlays", []),
            )

        return {
            "status": "ok" if ok else "failed_no_pose_records",
            "output_dir": str(out_dir),
            "frames_success": len(records),
        }


def make_config_from_args(args: Any) -> TrackingSessionConfig:
    obj_name = getattr(args, "obj_name", None) or getattr(args, "obj", None)
    if not obj_name:
        raise ValueError("--obj-name is required")
    return TrackingSessionConfig(
        trial_dir=Path(args.trial_dir),
        obj_name=str(obj_name),
        mesh_path=Path(args.mesh_path).expanduser() if getattr(args, "mesh_path", None) else None,
        anchor_bank_path=Path(args.anchor_bank_path).expanduser() if getattr(args, "anchor_bank_path", None) else None,
        init_pose_npy=Path(args.init_pose_npy).expanduser() if getattr(args, "init_pose_npy", None) else None,
        cam_param_dir=Path(args.cam_param_dir).expanduser() if getattr(args, "cam_param_dir", None) else None,
        output_dir=Path(args.output_dir).expanduser() if getattr(args, "output_dir", None) else None,
        pc_list=list(getattr(args, "pc_list", DEFAULT_PCS)),
        capture_ips=list(getattr(args, "capture_ips", None) or []) or None,
        active_serials=list(getattr(args, "active_serials", None) or []) or None,
        port_obs=int(getattr(args, "port_obs", DEFAULT_PORT_OBS)),
        port_prior=int(getattr(args, "port_prior", DEFAULT_PORT_PRIOR)),
        port_cmd_track=int(getattr(args, "port_cmd_track", DEFAULT_PORT_CMD_TRACK)),
        min_cams_per_frame=int(getattr(args, "min_cams_per_frame", 6)),
        max_frames=int(getattr(args, "max_frames", -1)),
        max_seconds=float(getattr(args, "max_seconds", -1.0)),
        progress_interval_s=float(getattr(args, "progress_interval_s", 0.5)),
        stale_after_s=float(getattr(args, "stale_after_s", 1.5)),
        run_index_dir=Path(args.run_index_dir).expanduser() if getattr(args, "run_index_dir", None) else None,
        force=bool(getattr(args, "force", False)),
        check_daemons=not bool(getattr(args, "skip_daemon_check", False)),
        allow_missing_daemons=bool(getattr(args, "allow_missing_daemons", False)),
        daemon_check_timeout_s=float(getattr(args, "daemon_check_timeout_s", 3.0)),
        web_port=int(getattr(args, "web_port", 0)),
        dry_run=bool(getattr(args, "dry_run", False)),
        run_overlay_check=bool(getattr(args, "run_overlay_check", False)),
        overlay_videos_dir=Path(args.overlay_videos_dir).expanduser() if getattr(args, "overlay_videos_dir", None) else None,
        overlay_output_dir=Path(args.overlay_output_dir).expanduser() if getattr(args, "overlay_output_dir", None) else None,
        overlay_python=Path(args.overlay_python).expanduser() if getattr(args, "overlay_python", None) else None,
        overlay_alpha=float(getattr(args, "overlay_alpha", 0.5)),
        overlay_strict=bool(getattr(args, "overlay_strict", False)),
    )


def run_from_args(args: Any) -> Dict[str, Any]:
    cfg = make_config_from_args(args)
    session = GoTrackTrackingSession(cfg)
    return session.run()


if __name__ == "__main__":
    raise SystemExit("Use src/execution/run_gotrack_session.py")
