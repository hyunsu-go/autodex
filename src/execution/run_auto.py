#!/usr/bin/env python3
"""Automated mode: distributed FoundPose init -> planning -> execute -> label.

Replaces the legacy SAM3+FPose perception pipeline used by
`src/execution_prev/run_auto.py`. Per trial we run one FoundPose init across
the capture PCs and treat that pose as the object pose for planning.

Prerequisites:
    bash scripts/init_daemons.sh start    # init_daemon on capture1-3, 5, 6
    bash scripts/init_daemons.sh status   # 1 daemon per PC

Usage:
    python src/execution/run_auto.py --obj attached_container
    python src/execution/run_auto.py --obj brown_ramen --scene wall --wall_angle 0
    python src/execution/run_auto.py --obj brown_ramen --success_only --viz
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller import get_arm, get_hand  # noqa: F401  (warm import)
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import get_label

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)

DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


# ── calibration ──────────────────────────────────────────────────────────────

def _load_calib(calib_dir: Path):
    """Read intrinsics.json + extrinsics.json into the dict shape InitOrchestrator wants."""
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a

    first = next(iter(intrinsics_full.values()))
    return intrinsics_full, extrinsics_full, int(first["height"]), int(first["width"])


# ── single trial ─────────────────────────────────────────────────────────────

_active_vis: Optional[ScenePlanVisualizer] = None


def run_single_trial(
    args,
    *,
    scene_prefix: str,
    orch: InitOrchestrator,
    planner: GraspPlanner,
    executor: RealExecutor,
    rcc,
    sync_generator,
    timestamp_monitor,
) -> dict:
    global _active_vis
    if _active_vis is not None:
        try:
            _active_vis.server.stop()
        except Exception:
            pass
        _active_vis = None

    obj = args.obj
    hand = args.hand
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = f"{scene_prefix}/{hand}" if scene_prefix else hand
    img_dir = os.path.join(project_dir, "experiment", args.exp_name, sub, obj, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    timing: dict = {}

    def _ts() -> str:
        return datetime.datetime.now().isoformat()

    # ── 1. save calib (raw images come from init pipeline itself) ────────────
    print(f"\n{'='*60}")
    print(f"[1/6] Trial dir -> {dir_idx}")
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    # ── 2. Distributed FoundPose init ───────────────────────────────────────
    print(f"[2/6] Init pipeline (FoundPose distributed)...")
    timing["perception_start"] = _ts()
    t0 = time.time()
    save_capture_dir = os.path.join(img_dir, "init_capture")
    pose_world, perc_timing = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=save_capture_dir,
        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        timeout_s=args.init_timeout_s,
    )
    timing["perception_s"] = round(time.time() - t0, 2)
    if perc_timing:
        timing["perception_detail"] = perc_timing

    if pose_world is None:
        reason = (perc_timing or {}).get("reason", "perception_failed")
        print(f"    Perception FAILED ({reason})")
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": reason, "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return fail

    print(f"    Perception: {timing['perception_s']}s")
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 3. scene_cfg + plan ──────────────────────────────────────────────────
    print(f"[3/6] Planning (version={args.grasp_version}, scene={args.scene})...")
    timing["planning_start"] = _ts()
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj)
    scene_cfg = add_obstacles(
        scene_cfg, args.scene,
        wall_gap=args.wall_gap, wall_angle=args.wall_angle,
        seed=args.clutter_seed,
        clutter_min_dist=args.clutter_min_dist,
        clutter_max_dist=args.clutter_max_dist,
        clutter_n=args.clutter_n,
        shelf_width=args.shelf_width, shelf_depth=args.shelf_depth,
        shelf_height=args.shelf_height, shelf_gap=args.shelf_gap,
        shelf_back=not args.no_shelf_back,
        shelf_sides=not args.no_shelf_sides,
        shelf_top=not args.no_shelf_top,
    )
    result = planner.plan(
        scene_cfg, obj, args.grasp_version,
        skip_done=(args.scene == "table"),
        success_only=args.success_only, hand=hand,
    )
    timing["plan_s"] = round(time.time() - t0, 2)
    print(f"    Plan: {timing['plan_s']}s  success={result.success}")

    if not result.success:
        print("    Planning FAILED — launching visualizer to inspect...")
        wrist_se3, _, grasp_pose, filtered = planner.get_candidates(
            scene_cfg, obj, args.grasp_version,
            success_only=args.success_only,
            skip_done=(args.scene == "table"), hand=hand,
        )
        fv = ScenePlanVisualizer(scene_cfg, None, port=8080, hand=hand)
        fv.add_candidates(wrist_se3, grasp_pose, filtered)
        fv.start_viewer(use_thread=True)
        _active_vis = fv
        chime.error()
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": "planning_failed", "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return fail

    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    if result.timing:
        with open(os.path.join(plan_dir, "timing.json"), "w") as f:
            json.dump(result.timing, f, indent=2)
    print(f"    Scene info: {result.scene_info}")

    if args.viz:
        print("    Launching visualizer (http://localhost:8080)...")
        sv = ScenePlanVisualizer(scene_cfg, result, port=8080, hand=hand)
        sv.start_viewer(use_thread=True)
        _active_vis = sv

    # ── 4. Execute (stream off, video on) ───────────────────────────────────
    print(f"[4/6] Executing on robot...")
    timing["execution_start"] = _ts()
    # Pause the always-on stream so video can take its place.
    try:
        rcc.stop()
    except Exception:
        pass

    raw_rel = os.path.join("AutoDex", "experiment", args.exp_name, sub, obj, dir_idx, "raw")
    raw_dir = os.path.join(img_dir, "raw")
    rcc.start("video", True, raw_rel)
    timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    sync_generator.start(fps=30)

    t0 = time.time()
    s_hand = executor.execute(result)
    timing["execute_s"] = round(time.time() - t0, 2)
    timing["execution_states"] = executor.state_timestamps

    rcc.stop()
    timestamp_monitor.stop()
    sync_generator.stop()

    # ── 5. Label ─────────────────────────────────────────────────────────────
    timing["label_start"] = _ts()
    print(f"[5/6] Label the result")
    label_rel = os.path.join("shared_data", "AutoDex", "experiment", args.exp_name,
                             sub, obj, dir_idx, "label", "raw")
    rcc.start("image", False, label_rel)
    rcc.stop()
    try:
        succ, note = get_label()
    except KeyboardInterrupt:
        print("\n[interrupted] Releasing and cleaning up...")
        executor.release(result)
        executor.stop_recording()
        raise

    # ── 6. Release & save ────────────────────────────────────────────────────
    print(f"[6/6] Releasing...")
    executor.release(result)
    executor.stop_recording()

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

    trial_result = {
        "dir_idx": dir_idx,
        "scene_type": args.scene,
        "success": succ,
        "scene_info": result.scene_info,
        "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
        "timing": timing,
    }
    if note is not None:
        trial_result["note"] = note
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(trial_result, f, indent=2, default=str)

    # Persist result back to the candidate dir for the table scene only.
    if succ is not None and result.scene_info is not None and args.scene == "table":
        from autodex.utils.path import get_candidate_path
        sei = result.scene_info
        cand_result_path = os.path.join(
            get_candidate_path(hand), args.grasp_version, obj,
            sei[0], sei[1], sei[2], "result.json",
        )
        with open(cand_result_path, "w") as f:
            json.dump({"success": succ, "dir_idx": dir_idx}, f)

    status = "SUCCESS" if succ else ("ISSUE" if succ is None else "FAIL")
    print(f"    Result: {status}  saved to {img_dir}/result.json")

    # Resume the stream so the next trial's init has live SHM frames.
    rcc.start("stream", False, fps=args.stream_fps)

    return trial_result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default=None, help="Defaults to grasp_version")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--viz", action="store_true")

    # scene-specific args (pass-through to autodex.planner.obstacles.add_obstacles)
    parser.add_argument("--wall_gap", type=float, default=0.04)
    parser.add_argument("--wall_angle", type=float, default=0.0)
    parser.add_argument("--clutter_seed", type=int, default=42)
    parser.add_argument("--clutter_min_dist", type=float, default=0.12)
    parser.add_argument("--clutter_max_dist", type=float, default=0.20)
    parser.add_argument("--clutter_n", type=int, default=4)
    parser.add_argument("--shelf_width", type=float, default=0.30)
    parser.add_argument("--shelf_depth", type=float, default=0.30)
    parser.add_argument("--shelf_height", type=float, default=0.30)
    parser.add_argument("--shelf_gap", type=float, default=0.02)
    parser.add_argument("--no_shelf_back", action="store_true")
    parser.add_argument("--no_shelf_sides", action="store_true")
    parser.add_argument("--no_shelf_top", action="store_true")

    # init pipeline args
    parser.add_argument("--pc_list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port_mask", type=int, default=5006)
    parser.add_argument("--port_pose", type=int, default=5007)
    parser.add_argument("--port_cmd", type=int, default=6893)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=120.0)
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="Camera calib dir. Default: latest under ~/shared_data/cam_param/.")
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.grasp_version

    # scene_prefix: '' (table), 'wall', 'shelf', 'cluttered', plus '_success_only' suffix.
    scene_prefix = args.scene if args.scene != "table" else ""
    if args.success_only:
        scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"

    # Mesh / FoundPose assets sanity check.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj} (expected under {assets_root})")

    # Calibration.
    if args.calib_dir:
        calib_dir = Path(args.calib_dir).expanduser()
    else:
        calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs  ({H}x{W})")

    # Hardware init.
    rcc = remote_camera_controller("run_auto", pc_list=args.pc_list)
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    rcc.start("stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Init orchestrator (FoundPose distributed).
    print(f"[orch] initializing for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print("[planner] warming up...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    def _cleanup():
        print("\n[cleanup] Stopping hardware...")
        for fn in (rcc.stop, timestamp_monitor.stop, sync_generator.stop,
                   executor.stop_recording):
            try:
                fn()
            except Exception:
                pass

    results: List[dict] = []
    trial = 0
    try:
        while True:
            trial += 1
            print(f"\n{'#'*60}\n# Trial {trial}\n{'#'*60}")
            chime.info()
            try:
                cmd = input("Press Enter to start trial, 'q' to quit: ").strip().lower()
            except KeyboardInterrupt:
                _cleanup()
                break
            if cmd == "q":
                break

            tr = run_single_trial(
                args, scene_prefix=scene_prefix,
                orch=orch, planner=planner, executor=executor,
                rcc=rcc, sync_generator=sync_generator,
                timestamp_monitor=timestamp_monitor,
            )
            results.append(tr)
            n_succ = sum(1 for r in results if r.get("success"))
            print(f"\n    Running total: {n_succ}/{len(results)} success")
    finally:
        # Summary + cleanup.
        print(f"\n{'='*60}\nSUMMARY: {args.obj} x {len(results)} trials")
        n_succ = sum(1 for r in results if r.get("success"))
        if results:
            print(f"  Success: {n_succ}/{len(results)} ({100*n_succ/len(results):.0f}%)")
        for r in results:
            status = "OK" if r.get("success") else r.get("reason", "FAIL")
            print(f"  {r['dir_idx']}: {status}")

        sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
        summary_path = os.path.join(project_dir, "experiment", args.exp_name, sub,
                                    args.obj, "summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            orch.close()
        except Exception:
            pass
        for fn in (timestamp_monitor.end, sync_generator.end, rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
