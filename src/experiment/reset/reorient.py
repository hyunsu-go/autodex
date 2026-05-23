#!/usr/bin/env python3
"""Reorient policy: perception → 8-phase put-down plan (i → target_j) →
post-perception. Replaces naive_drop's free-fall release with a deliberate
``approach → grasp_close → lift → rotate → place → release → depart → retract``
trajectory computed on-the-fly by ``plan_reset.plan_one_cell()``.

Per cycle:
    perception -> classify tabletop_before (i)
    -> if i == target_j: log "already_at_target" and continue
    -> plan_one_cell(i, target_j, T_obj_start, T_obj_end, h_cm)
    -> if plan fails: log + reset_fallback + continue
    -> phase-by-phase execute via executor._move_joints
    -> post-perception -> classify tabletop_after
    -> reorient_success = (tabletop_after_idx == target_j_idx)

h_cm controls release height (``reorient_4`` = 4 cm above table at release).
With h_cm > 0 the trajectory stops after ``place`` (hand keeps holding).

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/reorient.py --obj attached_container \
        --target_j 16 --h_cm 0 --auto
"""
from __future__ import annotations

import argparse
import atexit
import datetime
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

from autodex.utils.path import project_dir
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.experiment.reset.tabletop_pose import classify_tabletop_pose
from src.grasp_generation.reorient import plan_reset

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


# ── Hardcoded defaults (rarely changed across runs) ──────────────────────────
EXP_NAME = "reset_test/reorient"
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
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


class _SoftSkip(Exception):
    """Perception/plan/execute failure: log the cycle and continue to the next."""


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _pose_idx_from_filename(filename: str) -> int:
    """Extract integer pose index from ``classify_tabletop_pose`` filename
    (e.g. ``"016.npy"`` → ``16``). plan_reset / candidates use ints as keys."""
    return int(filename.replace(".npy", ""))


def _stop_video(rcc, sync_generator, timestamp_monitor):
    rcc.stop()
    time.sleep(0.3)
    sync_generator.stop()
    timestamp_monitor.stop()


def _retract_to_init(executor: RealExecutor):
    """Sequential return to XARM_INIT (mirrors RealExecutor.execute() step 1).
    plan_one_cell builds the ``approach`` phase from INIT_STATE, so the real
    arm must be at INIT before phase execution starts."""
    order = [1, 2, 5, 0, 3, 4]
    if executor.arm.get_data()["qpos"][1] < executor._xarm_init[1]:
        order = [2, 1, 5, 0, 3, 4]
    executor._move_joint_sequential(executor._xarm_init[:6], order, threshold=0.06)
    err = float(np.linalg.norm(
        executor.arm.get_data()["qpos"] - executor._xarm_init[:6]
    ))
    if err > 0.1:
        raise RuntimeError(
            f"_retract_to_init: final_qpos_err={err:.3f} > 0.1 — arm not at XARM_INIT"
        )


def _open_and_retract(executor: RealExecutor):
    """Failure-path reset: open hand to init, then sequential clear_view retract.
    Inlined version of executor.reset_fallback() without needing a PlanResult."""
    init_hand = executor._convert(executor._hand_init)
    executor._log_state("hand_init")
    executor._move_hand(init_hand)
    time.sleep(0.5)
    executor._log_state("clear_view")
    clear_view = executor._xarm_init.copy()
    clear_view[0] -= np.deg2rad(60.0)
    order = [1, 2, 5, 0, 3, 4]
    if executor.arm.get_data()["qpos"][1] < executor._xarm_init[1]:
        order = [2, 1, 5, 0, 3, 4]
    executor._move_joint_sequential(clear_view[:6], order, threshold=0.06)
    executor._log_state("reset_done")


def _phase_names(h_cm: int) -> list:
    if h_cm == 0:
        return list(plan_reset.PHASE_NAMES_FULL)
    return list(plan_reset.PHASE_NAMES_HOLD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_j", type=int, required=True,
                        help="Target tabletop pose filename int (e.g. 16 for 016.npy).")
    parser.add_argument("--h_cm", type=int, default=0,
                        help="Selects candidates/.../reorient_{h_cm}; >0 = hold "
                             "(no release).")
    parser.add_argument("--place_x", type=float, default=0.55)
    parser.add_argument("--place_y", type=float, default=0.0)
    parser.add_argument("--place_tz_deg", type=float, default=None,
                        help="Target yaw (deg) relative to T_j canonical. "
                             "If omitted, plan_one_cell searches 0..330° per seed.")
    parser.add_argument("--max_seeds", type=int, default=20)
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    args = parser.parse_args()

    # Asset sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

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

    client_name = f"reorient_{os.getpid()}"
    rcc = remote_camera_controller(client_name, pc_list=PC_LIST)
    print(f"[stream] starting on {len(PC_LIST)} PCs @ {STREAM_FPS} FPS "
          f"(client={client_name})...")
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
    print(f"[video] @ {VIDEO_FPS} FPS (always on during reorient motion)")

    sub = args.hand
    obj_root = Path(project_dir) / "experiment" / EXP_NAME / sub / args.obj
    obj_root.mkdir(parents=True, exist_ok=True)

    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print(f"[planner] warming up curobo + ik_solver for hand={args.hand}...")
    planner, base_world = plan_reset.init_planner(args.hand)
    # Post-hoc carry-arc check (1x load).
    try:
        urdf_fk, ee_link = plan_reset.load_fk_urdf(args.hand)
    except Exception as e:
        print(f"[planner] load_fk_urdf failed ({e!r}) — carry-arc check disabled")
        urdf_fk, ee_link = None, "base_link"
    try:
        obj_verts = plan_reset.load_object_vertices(args.obj)
    except Exception as e:
        print(f"[planner] load_object_vertices failed ({e!r}) — carry-arc check disabled")
        obj_verts = None

    Tj_can = plan_reset.load_tabletop_pose(args.obj, args.target_j)
    h_m = args.h_cm / 100.0
    if args.place_tz_deg is None:
        T_obj_end_fixed = None
        place_search_tzs = list(np.arange(0.0, 360.0, 30.0))
    else:
        T_obj_end_fixed = plan_reset.make_obj_pose(
            Tj_can,
            np.array([args.place_x, args.place_y, Tj_can[2, 3] + h_m]),
            args.place_tz_deg,
        )
        place_search_tzs = None

    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0

    # ── Guaranteed cleanup ───────────────────────────────────────────────
    _cleanup_done = [False]

    def _do_cleanup():
        if _cleanup_done[0]:
            return
        _cleanup_done[0] = True
        print("\n[cleanup] tearing down resources...")

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
            print(f"\n{'#'*60}\n# Cycle {cycle}  (target_j={args.target_j}, h_cm={args.h_cm})\n{'#'*60}")
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
                "obj": args.obj, "hand": args.hand,
                "target_j": args.target_j, "h_cm": args.h_cm,
                "place_x": args.place_x, "place_y": args.place_y,
                "place_tz_deg": args.place_tz_deg,
                "status": "started",
                "progress": {
                    "perception": None, "tabletop_before": None, "plan": None,
                    "phases": {}, "post_perception": None, "tabletop_after": None,
                },
                "timing": {},
                "tabletop_before": None,
                "tabletop_after": None,
                "reorient_success": None,
                "files": {
                    "pose_world": "pose_world.npy",
                    "pose_world_after": "pose_world_after.npy",
                    "perception_images": "init_capture/",
                    "post_drop_images": "post_drop_capture/",
                    "plan_dir": "plan/",
                    "actual_robot": "raw/",
                    "cam_param": "cam_param/",
                    "raw_video": "raw/",
                    "timestamps": "raw/timestamps",
                },
            }

            video_started = False
            plan_ok = False
            T_obj_end_used = None

            try:
                # 1. Perception.
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

                # 2. tabletop_before.
                c2r = load_c2r(str(cdir))
                pose_robot_before = np.linalg.inv(c2r) @ pose_world
                tb_before = classify_tabletop_pose(pose_robot_before, args.obj)
                rec["tabletop_before"] = tb_before
                if not tb_before:
                    rec["progress"]["tabletop_before"] = "no_tabletop_data"
                    rec["status"] = "no_tabletop_data"
                    print("    no tabletop_data for object — skipping cycle")
                    raise _SoftSkip
                i_idx = _pose_idx_from_filename(tb_before["filename"])
                rec["i_idx"] = i_idx
                rec["progress"]["tabletop_before"] = (
                    f"i={i_idx} (file={tb_before['filename']}, err={tb_before['rot_err_deg']:.1f}°)"
                )
                print(f"    [tabletop before] i={i_idx} ({tb_before['filename']}) "
                      f"err={tb_before['rot_err_deg']:.1f}°  target_j={args.target_j}")

                if i_idx == args.target_j:
                    rec["status"] = "already_at_target"
                    rec["progress"]["plan"] = "skipped (i == target_j)"
                    print(f"    already at target (i={i_idx} == target_j) — skipping cycle")
                    raise _SoftSkip

                # 3. Plan (8-phase reorient).
                print(f"[cycle {cycle}] Planning reorient i={i_idx} → j={args.target_j} "
                      f"(h_cm={args.h_cm})...")
                T_obj_start = pose_robot_before.copy()
                t0 = time.time()
                try:
                    plan_result = plan_reset.plan_one_cell(
                        planner, obj_name=args.obj, hand=args.hand,
                        h_cm=args.h_cm, i=i_idx, j=args.target_j,
                        T_obj_start=T_obj_start,
                        T_obj_end=T_obj_end_fixed,
                        place_xy=(args.place_x, args.place_y),
                        place_search_tzs=place_search_tzs,
                        base_world=base_world,
                        max_seeds=args.max_seeds, verbose=True,
                        urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
                    )
                except FileNotFoundError as fe:
                    rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                    rec["progress"]["plan"] = "no_candidates"
                    rec["status"] = "no_candidates"
                    rec["error"] = repr(fe)
                    print(f"    no candidates for cell {i_idx}_{args.target_j}: {fe!r}")
                    raise _SoftSkip
                rec["timing"]["plan_s"] = round(time.time() - t0, 2)

                if plan_result.get("status") != "ok":
                    fail_counts = plan_result.get("fail_counts", {})
                    dom = plan_reset.dominant_fail_category(fail_counts)
                    rec["progress"]["plan"] = f"failed (dominant={dom})"
                    rec["plan_fail_counts"] = fail_counts
                    rec["status"] = "plan_failed"
                    print(f"    plan FAILED (dominant={dom}) {fail_counts}")
                    raise _SoftSkip

                plan_ok = True
                trajs = plan_result["trajs"]
                T_obj_end_used = plan_result["T_obj_end"]
                rec["progress"]["plan"] = "ok"
                rec["seed_id"] = plan_result["seed_id"]
                rec["place_tz_used"] = plan_result.get("place_tz_used")
                rec["T_obj_start"] = T_obj_start.tolist()
                rec["T_obj_end"] = T_obj_end_used.tolist()
                rec["T_obj_apex_i"] = plan_result["T_obj_apex_i"].tolist()
                rec["T_obj_apex_j"] = plan_result["T_obj_apex_j"].tolist()

                trajs_np = {k: v.astype(np.float32) for k, v in trajs.items()}
                np.savez(cdir / "plan" / "trajectory.npz", **trajs_np)
                _write_json(cdir / "plan" / "meta.json", {
                    "seed_id": plan_result["seed_id"],
                    "i": i_idx, "j": args.target_j, "h_cm": args.h_cm,
                    "T_obj_start": T_obj_start.tolist(),
                    "T_obj_end": T_obj_end_used.tolist(),
                    "T_obj_apex_i": plan_result["T_obj_apex_i"].tolist(),
                    "T_obj_apex_j": plan_result["T_obj_apex_j"].tolist(),
                    "place_tz_used": plan_result.get("place_tz_used"),
                    "fail_counts_before_ok": plan_result.get("fail_counts", {}),
                })
                print(f"    plan ok (seed={plan_result['seed_id']}, "
                      f"{rec['timing']['plan_s']}s)")

                # 4. Ensure arm at INIT (approach trajectory starts there).
                print(f"[cycle {cycle}] Retracting to INIT before approach...")
                _retract_to_init(executor)

                # 5. Start recording.
                raw_dir = str(cdir / "raw")
                rcc.stop()
                video_rel = os.path.join(
                    "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "raw",
                )
                rcc.start("video", True, video_rel)
                timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                sync_generator.start(fps=VIDEO_FPS)
                video_started = True
                executor.start_recording(raw_dir)

                # 6. Phase-by-phase execute.
                executor.state_timestamps = []
                phase_names = _phase_names(args.h_cm)
                print(f"[cycle {cycle}] Executing phases: {phase_names}")
                t_exec0 = time.time()
                try:
                    for phase in phase_names:
                        traj = trajs[phase]                          # (T, 6+J)
                        if traj.ndim != 2 or traj.shape[1] < 7:
                            raise RuntimeError(
                                f"phase {phase}: unexpected traj shape {traj.shape}"
                            )
                        arm_traj = traj[:, :6]
                        hand_traj = np.array(
                            [executor._convert(traj[t, 6:]) for t in range(len(traj))]
                        )
                        executor._log_state(phase)
                        t_p0 = time.time()
                        executor._move_joints(arm_traj, hand_traj)
                        rec["progress"]["phases"][phase] = {
                            "n_waypoints": int(len(traj)),
                            "elapsed_s": round(time.time() - t_p0, 2),
                        }
                        time.sleep(0.05)   # small settle between phases
                    executor._log_state("reorient_done")
                except Exception as e:
                    rec["timing"]["execute_s"] = round(time.time() - t_exec0, 2)
                    states = getattr(executor, "state_timestamps", []) or []
                    phase = states[-1]["state"] if states else "unknown"
                    rec["progress"]["phases"][phase] = f"failed: {e!r}"
                    rec["status"] = f"{phase}_failed"
                    rec["error"] = repr(e)
                    rec["fail_phase"] = phase
                    print(f"[cycle {cycle}] {phase.upper()} FAILED: {e!r} — reset_fallback only")

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
                        _open_and_retract(executor)
                        rec["progress"]["retract"] = "fallback_after_phase_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip
                rec["timing"]["execute_s"] = round(time.time() - t_exec0, 2)
                rec["states"] = executor.state_timestamps

                # 7. Stop recordings → video stop → image snapshot → stream on.
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

                # 8. Post-perception (only for full put-down).
                if args.h_cm > 0:
                    rec["progress"]["post_perception"] = "skipped (h_cm>0, hand still holding)"
                    rec["status"] = "ok_hold"
                else:
                    print(f"[cycle {cycle}] Post-perception...")
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
                        np.save(cdir / "pose_world_after.npy", pose_world_after)
                        rec["progress"]["post_perception"] = "ok"
                        pose_robot_after = np.linalg.inv(c2r) @ pose_world_after
                        tb_after = classify_tabletop_pose(pose_robot_after, args.obj)
                        rec["tabletop_after"] = tb_after
                        if tb_after:
                            j_idx_after = _pose_idx_from_filename(tb_after["filename"])
                            rec["j_idx_after"] = j_idx_after
                            rec["progress"]["tabletop_after"] = (
                                f"j={j_idx_after} (file={tb_after['filename']}, "
                                f"err={tb_after['rot_err_deg']:.1f}°)"
                            )
                            rec["reorient_success"] = bool(j_idx_after == args.target_j)
                            R_a = T_obj_end_used[:3, :3]
                            R_b = pose_robot_after[:3, :3]
                            cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
                            cos = float(np.clip(cos, -1.0, 1.0))
                            rot_err = float(np.degrees(np.arccos(cos)))
                            trans_err = float(np.linalg.norm(
                                T_obj_end_used[:3, 3] - pose_robot_after[:3, 3]
                            ))
                            rec["reorient_quality"] = {
                                "target_obj_pose_robot": T_obj_end_used.tolist(),
                                "post_obj_pose_robot": pose_robot_after.tolist(),
                                "trans_err_m": trans_err,
                                "rot_err_deg": rot_err,
                            }
                            rec["status"] = ("ok" if rec["reorient_success"]
                                             else "ok_wrong_class")
                            print(f"    [tabletop after]  j_after={j_idx_after} "
                                  f"target={args.target_j}  "
                                  f"success={rec['reorient_success']}  "
                                  f"trans={trans_err*1000:.1f}mm  rot={rot_err:.1f}°")
                        else:
                            rec["progress"]["tabletop_after"] = "no_tabletop_data"
                            rec["status"] = "ok_no_tabletop_data"

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
            n_wrong = sum(1 for c in trials if c.get("status") == "ok_wrong_class")
            n_plan_fail = sum(1 for c in trials if c.get("status") == "plan_failed")
            print(f"    cycle {cycle} done — ok: {n_ok}/{len(trials)}  "
                  f"wrong_class: {n_wrong}  plan_fail: {n_plan_fail}")
            cycle += 1
            time.sleep(CYCLE_SLEEP_S)

    finally:
        print(f"\n{'='*60}\nSUMMARY: {args.obj} target_j={args.target_j} "
              f"× {len(trials)} trials")
        for c in trials:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            rq = c.get("reorient_quality")
            if rq:
                extra += (f"  reorient t={rq['trans_err_m']*1000:.0f}mm "
                          f"r={rq['rot_err_deg']:.0f}°")
            if c.get("reorient_success") is not None:
                extra += f"  success={c['reorient_success']}"
            print(f"  {c.get('trial_ts', '?')}: {tag}{extra}")
        _write_json(summary_path, trials)
        print(f"  summary -> {summary_path}")
        _do_cleanup()


if __name__ == "__main__":
    main()
