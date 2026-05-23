#!/usr/bin/env python3
"""Reorient + drop policy.

Per-cycle (structure mirrors naive_drop.py — DO NOT diverge):
    perception -> classify tabletop pose (before) -> plan
    -> execute (grasp + lift 25cm)
    -> charuco check (is the object actually lifted?)
        * fail  -> reset_fallback (open hand, sequential retract)  -> next cycle
        * pass  -> reorient wrist mid-air so the held object matches the
                   user-specified target tabletop rotation (z-aligned to the
                   current object yaw so motion is minimal)
                -> cartesian descent so the held object hovers ~15cm release
                   height above the table (object mesh-bottom must stay above
                   TABLE_SURFACE_Z — the ~10cm descent is bounded so it never
                   crashes the object into the table even if mesh is tall)
                -> release (reverse squeeze -> grasp -> pregrasp)
                -> reset_fallback (open hand -> sequential retract)
                -> post-drop perception -> classify tabletop pose (after)

The target tabletop pose is selected ONCE at script start via
``--target_tabletop`` (filename stem, e.g. ``002`` for ``002.npy``) and
shared across all cycles.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/reorient_drop.py --obj brown_ramen \\
        --target_tabletop 002 --auto
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import glob
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.execution.label import auto_label_charuco
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


# ── Hardcoded defaults (rarely changed across runs) ──────────────────────────
GRASP_VERSION = "table_only"
LIFT_HEIGHT_M = 0.25         # +25cm above grasp pose
RELEASE_HEIGHT_M = 0.15      # release while object hovers ~15cm above grasp z
EXP_NAME = "reset_test/reorient_drop"
SCENE = "table"
SUCCESS_ONLY = False
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
# Snapshot daemon ports (scripts/snapshot_daemons.sh start)
PORT_SNAP = 5009
PORT_SNAP_CMD = 6894
PROMPT = "object on the checkerboard"
SIL_ITERS = 100
SIL_LR = 0.002
INIT_TIMEOUT_S = 120.0
POST_INIT_TIMEOUT_S = 60.0
STREAM_FPS = 30
STREAM_WARMUP_S = 2.0
VIDEO_FPS = 30
CYCLE_SLEEP_S = 2.0
POST_DROP_SETTLE_S = 1.0
CHARUCO_BOARD = "1"


class _SoftSkip(Exception):
    """Perception/plan/lift/charuco failure: log the cycle and continue."""


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stop_video(rcc, sync_generator, timestamp_monitor):
    """Stop video recording. Order matters — see naive_drop.py for rationale."""
    rcc.stop()
    time.sleep(0.3)
    sync_generator.stop()
    timestamp_monitor.stop()


def _load_target_tabletop(obj_name: str, key: str) -> tuple[str, int, np.ndarray]:
    """Return (filename, sorted_idx, 4x4 pose) for the requested tabletop.

    ``key`` is the FILENAME stem (e.g. ``"002"`` for ``002.npy``) — matches
    what the user sees on disk. ``sorted_idx`` is the 0-based position in the
    sorted glob, identical to what ``classify_tabletop_pose`` returns, so we
    can compare against ``tb_after['idx']`` for hit-target accounting.
    """
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data",
                                "info", "tabletop")
    files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not files:
        sys.exit(f"no tabletop poses found at {tabletop_dir}")
    fname = f"{key}.npy"
    matched_idx = next((i for i, f in enumerate(files)
                        if os.path.basename(f) == fname), None)
    if matched_idx is None:
        avail = ", ".join(os.path.basename(f).replace(".npy", "") for f in files)
        sys.exit(f"tabletop {fname!r} not found in {tabletop_dir} — "
                 f"available: [{avail}]")
    pose = np.load(files[matched_idx])
    if pose.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = pose
        pose = T
    return fname, matched_idx, pose


def _z_align_yaw(R_est: np.ndarray, R_tab: np.ndarray) -> np.ndarray:
    """Return R_z(theta) @ R_tab where theta is the optimal yaw aligning R_tab
    to R_est. Mirrors _z_aligned_geodesic_deg in tabletop_pose.py — the
    tabletop class is yaw-invariant, so we pick the yaw that makes the
    reorient motion minimal w.r.t. the currently-grasped object orientation.
    """
    M = R_est @ R_tab.T
    theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_tab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_tabletop", type=str, required=True,
                        help="Filename stem of the desired tabletop pose "
                             "(e.g. '002' for 002.npy in "
                             "{obj}/processed_data/info/tabletop/).")
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port_viser", type=int, default=8080)
    args = parser.parse_args()

    # Asset sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Target tabletop pose (robot-frame, fixed for whole run).
    tt_filename, tt_sorted_idx, target_tabletop_robot = _load_target_tabletop(
        args.obj, args.target_tabletop)
    R_target_robot = target_tabletop_robot[:3, :3]
    print(f"[target] tabletop file={tt_filename} "
          f"(sorted_idx={tt_sorted_idx} — matches classify_tabletop_pose)")

    # Calibration.
    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in PC_LIST]
    pc_serials = {p: get_camera_list(p) for p in PC_LIST}
    active = {s for pc in PC_LIST for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active across {len(PC_LIST)} PCs ({H}x{W})")

    # Hardware stream.
    client_name = f"reorient_drop_{os.getpid()}"
    rcc = remote_camera_controller(client_name, pc_list=PC_LIST)
    print(f"[stream] starting on {len(PC_LIST)} PCs @ {STREAM_FPS} FPS "
          f"(client={client_name})...")
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
    print(f"[video] @ {VIDEO_FPS} FPS")

    sub = args.hand
    obj_root = Path(project_dir) / "experiment" / EXP_NAME / sub / args.obj
    obj_root.mkdir(parents=True, exist_ok=True)

    # Init orchestrator (1x).
    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
    # Snapshot orchestrator for charuco lift-check (no video stop needed).
    snap_orch = SnapshotOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_snap=PORT_SNAP, port_cmd=PORT_SNAP_CMD,
    )
    n_cams_total = sum(len(pc_serials[p]) for p in PC_LIST)
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print("[planner] warming up curobo...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0
    vis = None

    # ── Guaranteed cleanup (atexit + signal handlers) ────────────────────
    _cleanup_done = [False]

    def _do_cleanup():
        if _cleanup_done[0]:
            return
        _cleanup_done[0] = True
        print("\n[cleanup] tearing down resources...")
        nonlocal vis

        import threading
        def _call_with_timeout(label, fn, timeout=5):
            done = threading.Event()
            err_holder: list = []
            def _wrap():
                try:
                    fn()
                except Exception as ce:
                    err_holder.append(ce)
                finally:
                    done.set()
            t = threading.Thread(target=_wrap, daemon=True)
            t.start()
            done.wait(timeout=timeout)
            if not done.is_set():
                print(f"[cleanup] {label} TIMED OUT after {timeout}s — skipping")
                return False
            if err_holder:
                print(f"[cleanup] {label} failed: {err_holder[0]!r}")
            return True

        for label, fn in (
            ("vis.stop_viewer", (lambda: vis.stop_viewer()) if vis else None),
            ("executor.stop_recording", executor.stop_recording),
            ("executor.shutdown", executor.shutdown),
            ("orch.close", orch.close),
            ("rcc.stop", rcc.stop),
            ("sync_generator.stop", sync_generator.stop),
            ("timestamp_monitor.stop", timestamp_monitor.stop),
            ("sync_generator.end", sync_generator.end),
            ("timestamp_monitor.end", timestamp_monitor.end),
            ("rcc.end", rcc.end),
        ):
            if fn is None:
                continue
            _call_with_timeout(label, fn, timeout=5)
        print("[cleanup] done — forcing exit")
        os._exit(0)

    def _signal_handler(signum, frame):
        print(f"\n[signal] received {signal.Signals(signum).name}, cleaning up...")
        _do_cleanup()
        sys.exit(128 + signum)

    atexit.register(_do_cleanup)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while True:
            print(f"\n{'#'*60}\n# Cycle {cycle}\n{'#'*60}")
            chime.info()
            if not args.auto:
                try:
                    cmd = input(f"[cycle {cycle}] Enter=start, q=quit: ").strip().lower()
                except KeyboardInterrupt:
                    print("\n[loop] KeyboardInterrupt, stopping.")
                    break
                if cmd == "q":
                    break

            trial_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cdir = obj_root / trial_ts
            (cdir / "plan").mkdir(parents=True, exist_ok=True)
            save_current_C2R(str(cdir))
            save_current_camparam(str(cdir))

            rec: dict = {
                "cycle": cycle,
                "trial_ts": trial_ts,
                "start": _now(),
                "obj": args.obj, "hand": args.hand, "scene": SCENE,
                "target_tabletop": {
                    "key": args.target_tabletop,
                    "filename": tt_filename,
                    "sorted_idx": tt_sorted_idx,
                },
                "status": "started",
                "progress": {
                    "perception": None, "tabletop_before": None, "plan": None,
                    "execute": None, "charuco": None, "reorient": None,
                    "descent": None, "release": None, "retract": None,
                    "post_perception": None, "tabletop_after": None,
                },
                "timing": {},
                "tabletop_before": None,
                "tabletop_after": None,
                "tabletop_changed": None,
                "drop_quality": None,
                "files": {
                    "pose_world": "pose_world.npy",
                    "pose_world_after_drop": "pose_world_after_drop.npy",
                    "perception_images": "init_capture/",
                    "label_images": "label_at_lift/",
                    "post_drop_images": "post_drop_capture/",
                    "plan_dir": "plan/",
                    "actual_robot": "raw/",
                    "cam_param": "cam_param/",
                    "raw_video": "raw/",
                    "timestamps": "raw/timestamps",
                },
            }

            result = None
            scene_cfg = None
            planned_obj_pose = None
            T_obj_in_wrist = None
            video_started = False

            try:
                # 1. Perception (always — no lock_pose).
                print(f"[cycle {cycle}] Perception...")
                t0 = time.time()
                pose_world, perc_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "init_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=INIT_TIMEOUT_S,
                )
                rec["timing"]["perception_s"] = round(time.time() - t0, 2)
                if perc_timing:
                    rec["timing"]["perception_detail"] = perc_timing
                if pose_world is None:
                    reason = (perc_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["perception"] = f"failed: {reason}"
                    rec["status"] = "perception_failed"
                    rec["reason"] = reason
                    print(f"    perception FAILED ({reason}) — skipping cycle")
                    raise _SoftSkip
                rec["progress"]["perception"] = "ok"
                np.save(cdir / "pose_world.npy", pose_world)

                # 2. Tabletop classification (before).
                c2r = load_c2r(str(cdir))
                pose_robot_before = np.linalg.inv(c2r) @ pose_world
                tb_before = classify_tabletop_pose(pose_robot_before, args.obj)
                rec["tabletop_before"] = tb_before
                rec["progress"]["tabletop_before"] = (
                    f"idx={tb_before['idx']} ({tb_before['rot_err_deg']:.1f}deg)"
                    if tb_before else "no_tabletop_data"
                )
                if tb_before:
                    print(f"    [tabletop before] idx={tb_before['idx']} "
                          f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}deg")

                # 3. Plan.
                print(f"[cycle {cycle}] Planning (scene={SCENE})...")
                t0 = time.time()
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
                scene_cfg = add_obstacles(scene_cfg, SCENE)
                result = planner.plan(
                    scene_cfg, args.obj, GRASP_VERSION,
                    skip_done=False,
                    success_only=SUCCESS_ONLY, hand=args.hand,
                )
                rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                print(f"    plan: {rec['timing']['plan_s']}s  success={result.success}")
                _write_json(cdir / "scene_cfg.json", scene_cfg)
                if not result.success:
                    rec["progress"]["plan"] = "failed"
                    rec["status"] = "plan_failed"
                    if args.viz:
                        if vis is not None:
                            try:
                                vis.stop_viewer()
                            except Exception as ve:
                                print(f"[viz] previous viewer stop failed: {ve!r}")
                        try:
                            cand_wrist, _, cand_grasp, cand_filtered = (
                                planner.get_candidates(
                                    scene_cfg, args.obj, GRASP_VERSION,
                                    success_only=SUCCESS_ONLY,
                                    skip_done=False, hand=args.hand,
                                )
                            )
                            vis = ScenePlanVisualizer(scene_cfg, None,
                                                       port=args.port_viser,
                                                       hand=args.hand)
                            vis.add_candidates(cand_wrist, cand_grasp,
                                               cand_filtered)
                            vis.start_viewer(use_thread=True)
                            print(f"[viz] plan FAILED — candidates viewer at "
                                  f"http://localhost:{args.port_viser}")
                        except Exception as cve:
                            print(f"[viz] failed-plan viewer setup failed: {cve!r}")
                    raise _SoftSkip
                rec["progress"]["plan"] = "ok"

                np.save(cdir / "plan" / "traj.npy", result.traj)
                np.save(cdir / "plan" / "wrist_se3.npy", result.wrist_se3)
                np.save(cdir / "plan" / "pregrasp_pose.npy", result.pregrasp_pose)
                np.save(cdir / "plan" / "grasp_pose.npy", result.grasp_pose)
                if result.timing:
                    _write_json(cdir / "plan" / "timing.json", result.timing)
                rec["scene_info"] = result.scene_info
                rec["candidate_idx"] = (result.timing.get("candidate_idx")
                                        if result.timing else None)

                if args.viz:
                    if vis is not None:
                        try:
                            vis.stop_viewer()
                        except Exception as ve:
                            print(f"[viz] previous viewer stop failed: {ve!r}")
                    vis = ScenePlanVisualizer(scene_cfg, result,
                                              port=args.port_viser,
                                              hand=args.hand)
                    vis.start_viewer(use_thread=True)
                    print(f"[viz] http://localhost:{args.port_viser}")

                # 4. Recording start (arm/hand + video).
                raw_dir = str(cdir / "raw")
                rcc.stop()
                video_rel = os.path.join(
                    "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "raw",
                )
                rcc.start("full", True, video_rel)   # "full" = video AVI + SHM stream (for snapshot_daemon)
                timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                sync_generator.start(fps=VIDEO_FPS)
                video_started = True
                executor.start_recording(raw_dir)

                # 5. Execute (grasp + 25cm lift).
                print(f"[cycle {cycle}] Execute (grasp + lift {LIFT_HEIGHT_M*100:.0f}cm)...")
                t0 = time.time()
                try:
                    s_hand = executor.execute(result, lift_height=LIFT_HEIGHT_M)
                except Exception as e:
                    rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                    states = getattr(executor, "state_timestamps", []) or []
                    phase = states[-1]["state"] if states else "unknown"
                    rec["progress"]["execute"] = f"{phase}_failed: {e!r}"
                    rec["progress"]["charuco"] = "skipped"
                    rec["progress"]["reorient"] = "skipped"
                    rec["progress"]["descent"] = "skipped"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = f"{phase}_failed"
                    rec["error"] = repr(e)
                    rec["fail_phase"] = phase
                    rec["fail_primitive"] = getattr(e, "where", None)
                    print(f"[cycle {cycle}] {phase.upper()} FAILED: {e!r} "
                          f"— reset_fallback only")

                    cleanup_errs: list = []
                    try:
                        executor.stop_recording()
                    except Exception as ce:
                        cleanup_errs.append(f"stop_recording: {ce!r}")
                    if video_started:
                        try:
                            _stop_video(rcc, sync_generator, timestamp_monitor)
                            video_started = False
                            rcc.start("stream", False, fps=STREAM_FPS)
                        except Exception as ce:
                            cleanup_errs.append(f"video_stop_or_stream: {ce!r}")
                    if cleanup_errs:
                        rec["cleanup_errors"] = cleanup_errs

                    try:
                        fb_log = executor.reset_fallback(result)
                        rec["reset"] = fb_log
                        rec["progress"]["retract"] = "fallback_after_lift_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip
                rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                rec["progress"]["execute"] = "ok"
                if s_hand is not None:
                    np.save(cdir / "squeeze_hand.npy", s_hand)

                # Snapshot the constant object-in-wrist transform at grasp.
                T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp

                # 6. Charuco lift-check via snapshot_daemon — pull one JPEG per
                #    camera over ZMQ from the SHM ring buffer. Video keeps
                #    recording the whole time (rcc mode "full" populates both
                #    AVI and SHM in parallel — paradex camera.py line 224-225),
                #    so we don't touch rcc/sync_generator at all here.
                print(f"[cycle {cycle}] Charuco lift-check (snapshot_daemon)...")
                t0 = time.time()
                label_abs = str(cdir / "label_at_lift" / "raw" / "images")
                _, snap_timing = snap_orch.snap(
                    n_expected=n_cams_total, timeout_s=3.0,
                    save_dir_local=label_abs,
                )
                n_jpg = len(glob.glob(os.path.join(label_abs, "*.jpg")))
                print(f"    [charuco] {n_jpg}/{n_cams_total} JPGs collected in "
                      f"{snap_timing['dispatch_to_collected_s']:.2f}s "
                      f"-> {label_abs}")
                auto_succ, label_info = auto_label_charuco(
                    label_abs, required_board=CHARUCO_BOARD)
                rec["charuco_snap"] = snap_timing
                rec["timing"]["charuco_s"] = round(time.time() - t0, 2)
                rec["charuco"] = label_info
                rec["charuco_success"] = bool(auto_succ)
                if label_info.get("reason"):
                    rec["progress"]["charuco"] = f"failed: {label_info['reason']}"
                    print(f"    [charuco] FAILED ({label_info['reason']})")
                else:
                    rec["progress"]["charuco"] = (
                        "pass" if auto_succ else
                        f"fail (covered {label_info['covered']}/{label_info['expected']})"
                    )
                    print(f"    [charuco] success={auto_succ}  "
                          f"covered {label_info['covered']}/{label_info['expected']}")

                if not auto_succ:
                    # Lift did not clear the table — drop the (failed) grasp
                    # and reset. No reorient / release / post-perception.
                    rec["progress"]["reorient"] = "skipped"
                    rec["progress"]["descent"] = "skipped"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = "charuco_fail"
                    # Tear down THIS trial's recording (video + sync + ts).
                    try:
                        executor.stop_recording()
                    except Exception:
                        pass
                    if video_started:
                        try:
                            _stop_video(rcc, sync_generator, timestamp_monitor)
                            video_started = False
                        except Exception:
                            pass
                    rcc.start("stream", False, fps=STREAM_FPS)
                    try:
                        fb_log = executor.reset_fallback(result)
                        rec["reset"] = fb_log
                        rec["progress"]["retract"] = "fallback_after_charuco_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip

                # 7. Reorient via cuRobo: yaw-search IK + plan_single_js. Picks
                #    the IK solution closest to current qpos so joint 6 doesn't
                #    wrap into a limit. Object is held with squeeze (s_hand).
                print(f"[cycle {cycle}] Reorient (target tabletop "
                      f"file={tt_filename})...")
                t0 = time.time()
                T_link6_now = executor.arm.get_data()["position"].copy()
                T_wrist_now = T_link6_now @ executor._link6_to_wrist
                T_obj_now_world = T_wrist_now @ T_obj_in_wrist

                # Build target wrist pose (position held at lifted z, rotation
                # from target tabletop in world frame).
                R_target_world = c2r[:3, :3] @ R_target_robot
                T_obj_target_world = T_obj_now_world.copy()
                T_obj_target_world[:3, :3] = R_target_world
                T_wrist_target = T_obj_target_world @ np.linalg.inv(T_obj_in_wrist)

                # Current 22-DOF qpos. Hand is at squeeze (controller-converted
                # s_hand) — but the planner needs cuRobo URDF order. Use the
                # plan's grasp_pose as the held hand config (same convention as
                # plan_js_to_init's start_hand_qpos).
                cur_arm = executor.arm.get_data()["qpos"]
                current_qpos = np.concatenate([cur_arm, result.grasp_pose])

                reorient_traj, reorient_info = planner.plan_wrist_reorient(
                    scene_cfg, current_qpos, T_wrist_target,
                    hold_hand_qpos=result.grasp_pose, n_yaw=8,
                )
                rec["timing"]["reorient_plan_s"] = round(time.time() - t0, 2)
                rec["reorient_info"] = reorient_info
                rec["reorient"] = {
                    "T_wrist_before": T_wrist_now.tolist(),
                    "T_wrist_target": T_wrist_target.tolist(),
                    "T_obj_target_world": T_obj_target_world.tolist(),
                    "R_target_robot": R_target_robot.tolist(),
                }

                # Execute reorient. Try cuRobo first; if it fails, fall back to
                # _move_cartesian — under NO circumstances skip reorient and
                # drop the object as-is. The whole point of this script is to
                # reorient before release.
                if reorient_traj is not None:
                    t1 = time.time()
                    executor._log_state("reorient")
                    arm_traj = reorient_traj[:, :6]
                    hand_traj = np.array([executor._convert(reorient_traj[i, 6:])
                                          for i in range(len(reorient_traj))])
                    executor._move_joints(arm_traj, hand_traj)
                    rec["timing"]["reorient_exec_s"] = round(time.time() - t1, 2)
                    rec["progress"]["reorient"] = (
                        f"planner_ok (yaw_idx={reorient_info['best_yaw_idx']}, "
                        f"arm_dist={reorient_info['best_arm_dist_rad']:.3f}rad)"
                    )
                    print(f"    [reorient] planner OK  "
                          f"yaw_idx={reorient_info['best_yaw_idx']}/"
                          f"{reorient_info['n_yaw']}  "
                          f"arm_dist={reorient_info['best_arm_dist_rad']:.3f}rad  "
                          f"plan={reorient_info.get('plan_s')}s  "
                          f"exec={rec['timing']['reorient_exec_s']}s")
                else:
                    # Cartesian-servo fallback: at least try to rotate the
                    # wrist toward target. May not fully reach if joint limits
                    # bite, but never skip reorient entirely.
                    t1 = time.time()
                    T_link6_target = (T_wrist_target
                                      @ np.linalg.inv(executor._link6_to_wrist))
                    executor._log_state("reorient")
                    executor._move_cartesian(T_link6_target, vel_scale=1/1.5)
                    rec["timing"]["reorient_exec_s"] = round(time.time() - t1, 2)
                    # Measure how much we actually reoriented.
                    T_wrist_actual = (executor.arm.get_data()["position"]
                                      @ executor._link6_to_wrist)
                    T_obj_actual = T_wrist_actual @ T_obj_in_wrist
                    R_err = T_obj_actual[:3, :3].T @ R_target_world
                    cos = (np.trace(R_err) - 1.0) / 2.0
                    rot_err_deg = float(np.degrees(np.arccos(
                        float(np.clip(cos, -1.0, 1.0)))))
                    rec["progress"]["reorient"] = (
                        f"cartesian_fallback (planner_reason="
                        f"{reorient_info.get('reason')}, "
                        f"residual_err={rot_err_deg:.1f}deg)"
                    )
                    print(f"    [reorient] planner FAILED "
                          f"({reorient_info.get('reason')}) — cartesian "
                          f"fallback used, residual_err={rot_err_deg:.1f}deg")

                # 9. Cartesian descent: lower link6 so the object hovers at
                #    RELEASE_HEIGHT_M above the original grasp z. We lifted by
                #    LIFT_HEIGHT_M, so descend (LIFT_HEIGHT_M - RELEASE_HEIGHT_M).
                #    Descent is bounded — the object cannot reach the table
                #    because RELEASE_HEIGHT_M (15cm) > any reasonable mesh
                #    half-height for the manipulable objects here.
                descend = LIFT_HEIGHT_M - RELEASE_HEIGHT_M
                print(f"[cycle {cycle}] Descent ({descend*100:.0f}cm)...")
                t0 = time.time()
                T_link6_after_reorient = executor.arm.get_data()["position"].copy()
                T_link6_release = T_link6_after_reorient.copy()
                T_link6_release[2, 3] -= descend
                executor._log_state("descent")
                executor._move_cartesian(T_link6_release, vel_scale=1/1.5)
                rec["timing"]["descent_s"] = round(time.time() - t0, 2)
                rec["progress"]["descent"] = "ok"

                # Snapshot planned object pose at the moment of release.
                T_wrist_release = (executor.arm.get_data()["position"]
                                   @ executor._link6_to_wrist)
                planned_obj_pose = T_wrist_release @ T_obj_in_wrist

                # 10. Release (squeeze -> grasp -> pregrasp).
                print(f"[cycle {cycle}] Release...")
                t0 = time.time()
                try:
                    executor.release(result)
                except Exception as re_e:
                    rec["progress"]["release"] = f"exception: {re_e!r}"
                    rec["release_error"] = repr(re_e)
                    print(f"    release FAILED: {re_e!r}")
                else:
                    rec["timing"]["release_s"] = round(time.time() - t0, 2)
                    rec["progress"]["release"] = "ok"

                # 11. Reset_fallback: open hand to hand_init + sequential arm retract.
                t1 = time.time()
                try:
                    fb_log = executor.reset_fallback(result)
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["reset"] = fb_log
                    rec["progress"]["retract"] = "fallback_sequential"
                    rec["states"] = executor.state_timestamps
                    print(f"    release={rec['timing'].get('release_s', '?')}s  "
                          f"retract={rec['timing']['retract_s']}s  "
                          f"final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
                except Exception as fb_e:
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["progress"]["retract"] = f"fallback_failed: {fb_e!r}"
                    rec["fallback_error"] = repr(fb_e)
                    rec["status"] = "reset_failed"
                    rec["progress"]["post_perception"] = "skipped"
                    print(f"    reset_fallback FAILED: {fb_e!r}")
                    raise _SoftSkip

                # 12. Stop recordings → video stop → image snap → stream.
                executor.stop_recording()
                _stop_video(rcc, sync_generator, timestamp_monitor)
                video_started = False
                time.sleep(0.5)

                image_rel = os.path.join(
                    "shared_data", "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "post_drop_snap",
                )
                rcc.start("image", False, image_rel)
                rcc.stop()
                time.sleep(0.5)
                rcc.start("stream", False, fps=STREAM_FPS)

                if POST_DROP_SETTLE_S > 0:
                    time.sleep(POST_DROP_SETTLE_S)

                # 13. Post-drop perception + tabletop_after + drop_quality.
                print(f"[cycle {cycle}] Post-drop perception...")
                t0 = time.time()
                pose_world_after, post_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "post_drop_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=POST_INIT_TIMEOUT_S,
                )
                rec["timing"]["post_perception_s"] = round(time.time() - t0, 2)
                if post_timing:
                    rec["timing"]["post_perception_detail"] = post_timing

                if pose_world_after is None:
                    reason = (post_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["post_perception"] = f"failed: {reason}"
                    rec["status"] = "ok_post_perception_fail"
                else:
                    np.save(cdir / "pose_world_after_drop.npy", pose_world_after)
                    rec["progress"]["post_perception"] = "ok"
                    rec["status"] = "ok"

                    pose_robot_after = np.linalg.inv(c2r) @ pose_world_after
                    tb_after = classify_tabletop_pose(pose_robot_after, args.obj)
                    rec["tabletop_after"] = tb_after
                    rec["progress"]["tabletop_after"] = (
                        f"idx={tb_after['idx']} ({tb_after['rot_err_deg']:.1f}deg)"
                        if tb_after else "no_tabletop_data"
                    )
                    if tb_after:
                        rec["tabletop_hit_target"] = bool(
                            tb_after["idx"] == tt_sorted_idx
                        )
                        if tb_before:
                            rec["tabletop_changed"] = bool(
                                tb_after["idx"] != tb_before["idx"]
                            )
                        print(f"    [tabletop after]  idx={tb_after['idx']} "
                              f"({tb_after['filename']}) err={tb_after['rot_err_deg']:.1f}deg "
                              f"target={tt_filename} "
                              f"hit={rec['tabletop_hit_target']}")

                    R_a = planned_obj_pose[:3, :3]
                    R_b = pose_robot_after[:3, :3]
                    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
                    cos = float(np.clip(cos, -1.0, 1.0))
                    rot_err = float(np.degrees(np.arccos(cos)))
                    trans_err = float(np.linalg.norm(
                        planned_obj_pose[:3, 3] - pose_robot_after[:3, 3]
                    ))
                    z_drop = float(planned_obj_pose[2, 3] - pose_robot_after[2, 3])
                    rec["drop_quality"] = {
                        "planned_obj_pose_robot": planned_obj_pose.tolist(),
                        "post_drop_obj_pose_robot": pose_robot_after.tolist(),
                        "trans_err_m": trans_err,
                        "rot_err_deg": rot_err,
                        "z_drop_m": z_drop,
                    }
                    print(f"    drop_quality: trans={trans_err*1000:.1f}mm  "
                          f"rot={rot_err:.1f}deg  z={z_drop*1000:.1f}mm")

            except _SoftSkip:
                pass
            except Exception as e:
                rec["status"] = "aborted"
                rec["error"] = repr(e)
                cleanup_errs: list = []
                try:
                    executor.stop_recording()
                except Exception as ce:
                    cleanup_errs.append(f"stop_recording: {ce!r}")
                if video_started:
                    try:
                        _stop_video(rcc, sync_generator, timestamp_monitor)
                        video_started = False
                    except Exception as ce:
                        cleanup_errs.append(f"_stop_video: {ce!r}")
                if cleanup_errs:
                    rec["cleanup_errors"] = cleanup_errs
                rec["end"] = _now()
                _write_json(cdir / "result.json", rec)
                trials.append(rec)
                _write_json(summary_path, trials)
                print(f"[cycle {cycle}] ABORTED: {e!r}")
                raise

            # End-of-cycle safety cleanup.
            cycle_cleanup_errs: list = []
            if video_started:
                try:
                    _stop_video(rcc, sync_generator, timestamp_monitor)
                    video_started = False
                    rcc.start("stream", False, fps=STREAM_FPS)
                except Exception as ce:
                    cycle_cleanup_errs.append(f"video_cleanup: {ce!r}")
            if cycle_cleanup_errs:
                rec["cleanup_errors"] = (rec.get("cleanup_errors") or []) + cycle_cleanup_errs

            arm_data = executor.arm.get_data()
            rec["final_qpos"] = arm_data["qpos"].tolist()
            rec["final_arm_pose"] = arm_data["position"].tolist()

            rec["end"] = _now()
            _write_json(cdir / "result.json", rec)
            trials.append(rec)
            _write_json(summary_path, trials)

            n_ok = sum(1 for c in trials if c.get("status") == "ok")
            n_charuco_fail = sum(1 for c in trials if c.get("status") == "charuco_fail")
            n_hit = sum(1 for c in trials if c.get("tabletop_hit_target") is True)
            print(f"    cycle {cycle} done — ok: {n_ok}/{len(trials)}  "
                  f"charuco_fail: {n_charuco_fail}  hit_target: {n_hit}")
            cycle += 1
            time.sleep(CYCLE_SLEEP_S)

    finally:
        print(f"\n{'='*60}\nSUMMARY: {args.obj} × {len(trials)} trials  "
              f"target_tabletop={tt_filename}")
        for c in trials:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            dq = c.get("drop_quality")
            if dq:
                extra += f"  drop t={dq['trans_err_m']*1000:.0f}mm r={dq['rot_err_deg']:.0f}d"
            if c.get("tabletop_hit_target") is not None:
                extra += f"  hit={c['tabletop_hit_target']}"
            print(f"  {c.get('trial_ts', '?')}: {tag}{extra}")
        _write_json(summary_path, trials)
        print(f"  summary -> {summary_path}")
        _do_cleanup()


if __name__ == "__main__":
    main()
