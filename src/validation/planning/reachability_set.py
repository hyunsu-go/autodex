"""
Reachability Set via IK Solving

Grid search over (x_offset, z_rotation) to find where the robot can reach
grasp candidates using IK only (no trajectory planning).

Each grid point runs N trials with different seeds to check IK consistency.
Saves qpos and object poses for later visualization.

Usage:
    # Single object
    python src/validation/planning/reachability_set.py --obj attached_container --version selected_100

    # All objects
    python src/validation/planning/reachability_set.py --version selected_100

    # Custom grid
    python src/validation/planning/reachability_set.py --obj attached_container --version selected_100 \
        --x_min 0.2 --x_max 0.5 --x_step 0.05 --z_step 30 --n_trials 10
"""

import os
import sys
import time
import argparse
import json
import logging
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner import GraspPlanner
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart


def load_tabletop_scene(obj_name, pose_idx="000", r=0.4, theta=0.0):
    """Load object pose at polar (r, theta) around the xarm base.

    Position    : Rz(theta) @ (tabletop_t + [r, 0, 0])
    Orientation : tabletop_pose_rotation (NOT rotated — every object faces the
                  same global direction; we only orbit the position).
    """
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(os.listdir(pose_dir))
        raise FileNotFoundError(f"Pose {pose_idx} not found. Available: {available}")

    obj_pose = np.load(pose_file)

    obj_pose[0, 3] += r
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    obj_pose[:3, 3] = Rz @ obj_pose[:3, 3]

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    scene_cfg = {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": mesh_path,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            },
            # Floor everywhere outside the table — prevents arm from reaching below z=0.
            "floor": {
                "dims": [10.0, 10.0, 0.05],
                "pose": [0.0, 0.0, -0.025, 1, 0, 0, 0],
            },
        },
    }
    return scene_cfg, obj_pose


def get_tabletop_poses(obj_name):
    """Get all available tabletop pose indices."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted([f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy")])


def get_all_objects():
    """Find all objects that have tabletop poses."""
    objects = []
    for obj_name in sorted(os.listdir(obj_path)):
        tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
        if os.path.isdir(tabletop_dir) and len(os.listdir(tabletop_dir)) > 0:
            objects.append(obj_name)
    return objects


def run_reachability(obj_name, grasp_version, n_trials,
                     radii, thetas_deg, save_dir=None, hand="allegro"):
    """IK reachability polar grid search for one object."""
    pose_indices = get_tabletop_poses(obj_name)
    if not pose_indices:
        print(f"  No tabletop poses found for {obj_name}")
        return None

    print(f"Object: {obj_name}")
    print(f"Hand: {hand}")
    print(f"Grasp version: {grasp_version}")
    print(f"Poses: {pose_indices}")
    print(f"Radii (m): {radii}")
    print(f"Thetas (deg): {thetas_deg}")
    print(f"Trials per grid point: {n_trials}")
    total_points = len(pose_indices) * len(radii) * len(thetas_deg)
    print(f"Total grid points: {total_points}")
    print("=" * 60)

    # Suppress cuRobo warnings during batch IK
    logging.getLogger("curobo").setLevel(logging.ERROR)

    planner = GraspPlanner(hand=hand)

    grid_results = []
    viz_data = []

    # Build grid points
    grid_points = [(p, r, th) for p in pose_indices for r in radii for th in thetas_deg]

    pbar = tqdm(grid_points, desc=f"{obj_name}", unit="pt")
    n_fail = 0
    n_warn = 0

    for pose_idx, r_val, theta_deg in pbar:
        theta_rad = np.radians(theta_deg)
        scene_cfg, obj_pose = load_tabletop_scene(
            obj_name, pose_idx, r=r_val, theta=theta_rad)

        trial_ik_counts = []
        trial_timings = []
        # Per-grasp success across trials. Key = "scene_type/scene_id/grasp_idx".
        per_grasp_succ = {}
        per_grasp_qpos = {}  # key -> qpos list (first successful trial's qpos)

        for trial in range(n_trials):
            result = planner.solve_ik(
                scene_cfg, obj_name=obj_name,
                grasp_version=grasp_version, seed=trial, hand=hand,
            )
            n_ik = result["n_ik_success"]
            trial_ik_counts.append(n_ik)
            trial_timings.append(result["timing"])

            scene_info_trial = result["scene_info"]
            succ_mask_trial = result["ik_success"]
            ik_qpos_trial = result["ik_qpos"]
            for ci, ok in enumerate(succ_mask_trial):
                if not ok:
                    continue
                st, sid, gid = scene_info_trial[ci]
                key = f"{st}/{sid}/{gid}"
                per_grasp_succ[key] = per_grasp_succ.get(key, 0) + 1
                if key not in per_grasp_qpos:
                    per_grasp_qpos[key] = ik_qpos_trial[ci].tolist()

        ik_counts = np.array(trial_ik_counts)
        avg_timing = {}
        for key in ["load_candidates_s", "world_setup_s", "filter_s", "ik_solve_s"]:
            vals = [t.get(key, 0) for t in trial_timings]
            avg_timing[key] = round(float(np.mean(vals)), 3)

        n_trials_with_ik = int((ik_counts > 0).sum())

        if n_trials_with_ik == 0:
            n_fail += 1
        elif n_trials_with_ik < n_trials:
            n_warn += 1

        entry = {
            "pose_idx": pose_idx,
            "r": r_val,
            "theta_deg": theta_deg,
            "n_trials": n_trials,
            "trials_with_ik": n_trials_with_ik,
            "ik_counts": ik_counts.tolist(),
            "ik_mean": round(float(ik_counts.mean()), 1),
            "ik_min": int(ik_counts.min()),
            "ik_max": int(ik_counts.max()),
            "n_total": result["n_total"],
            "n_backward": result["n_backward"],
            "n_valid": result["n_valid"],
            "avg_timing": avg_timing,
            "per_grasp_succ": per_grasp_succ,
        }
        grid_results.append(entry)

        if per_grasp_qpos:
            viz_data.append({
                "pose_idx": pose_idx,
                "r": r_val,
                "theta_deg": theta_deg,
                "obj_pose": obj_pose.tolist(),
                "per_grasp_qpos": per_grasp_qpos,
            })

        pbar.set_postfix(fail=n_fail, warn=n_warn,
                         ik=f"{ik_counts.mean():.0f}/{result['n_valid']}")

    # Summary
    print(f"\n{'=' * 60}")
    print("REACHABILITY SUMMARY")
    print(f"{'=' * 60}")

    n_reachable = sum(1 for r in grid_results if r["trials_with_ik"] == r["n_trials"])
    n_partial = sum(1 for r in grid_results if 0 < r["trials_with_ik"] < r["n_trials"])
    n_unreachable = sum(1 for r in grid_results if r["trials_with_ik"] == 0)

    print(f"Grid points: {len(grid_results)}")
    print(f"  Always reachable: {n_reachable}")
    print(f"  Sometimes reachable: {n_partial}")
    print(f"  Never reachable: {n_unreachable}")

    ik_means = [r["ik_mean"] for r in grid_results]
    print(f"IK solutions per grid point: mean={np.mean(ik_means):.1f}  "
          f"min={np.min(ik_means):.0f}  max={np.max(ik_means):.0f}")

    # Per-stage timing
    print(f"\nPer-stage timing (mean across grid):")
    for key in ["load_candidates_s", "world_setup_s", "collision_check_s", "ik_solve_s"]:
        vals = [r["avg_timing"].get(key, 0) for r in grid_results]
        print(f"  {key}: mean={np.mean(vals):.3f}s")

    # Show unreachable/partial points
    problem_points = [r for r in grid_results if r["trials_with_ik"] < r["n_trials"]]
    if problem_points:
        print(f"\nProblem points ({len(problem_points)}):")
        for r in sorted(problem_points, key=lambda x: x["trials_with_ik"]):
            print(f"  pose={r['pose_idx']} r={r['r']:.2f} θ={r['theta_deg']:3.0f}°  "
                  f"ik_success={r['trials_with_ik']}/{r['n_trials']}  "
                  f"ik_count={r['ik_mean']:.1f}±{np.std(r['ik_counts']):.1f}")

    # Save
    if save_dir is None:
        save_dir = os.path.join("outputs", "reachability", obj_name)
    os.makedirs(save_dir, exist_ok=True)

    summary = {
        "obj_name": obj_name,
        "grasp_version": grasp_version,
        "n_trials": n_trials,
        "radii": radii,
        "thetas_deg": thetas_deg,
        "pose_indices": pose_indices,
        "n_reachable": n_reachable,
        "n_partial": n_partial,
        "n_unreachable": n_unreachable,
        "grid": grid_results,
    }

    out_path = os.path.join(save_dir, f"reachability_{grasp_version}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nGrid results saved to: {out_path}")

    # Save viz data (qpos + obj pose) separately as npz for easy loading
    if viz_data:
        viz_path = os.path.join(save_dir, f"reachability_{grasp_version}_viz.json")
        with open(viz_path, "w") as f:
            json.dump(viz_data, f, indent=2)
        print(f"Viz data saved to: {viz_path} ({len(viz_data)} reachable grid points)")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IK reachability grid search")
    parser.add_argument("--obj", type=str, default=None, help="Object name (omit for all)")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"],
                        help="Hand type (selects candidates + robot config)")
    parser.add_argument("--version", type=str, required=True, help="Grasp candidate version")
    parser.add_argument("--n_trials", type=int, default=10, help="Trials per grid point")
    parser.add_argument("--r_min", type=float, default=0.20, help="Min radial distance (m)")
    parser.add_argument("--r_max", type=float, default=0.60, help="Max radial distance (m)")
    parser.add_argument("--r_step", type=float, default=0.10, help="Radial step (m)")
    parser.add_argument("--theta_step", type=float, default=30, help="Theta step (deg)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    radii = np.arange(args.r_min, args.r_max + 1e-6, args.r_step).round(3).tolist()
    thetas_deg = np.arange(0, 360, args.theta_step).tolist()

    if args.obj is not None:
        objects = [args.obj]
    else:
        # Restrict to objects that have candidates for the requested hand
        from autodex.utils.path import get_candidate_path
        cand_root = os.path.join(get_candidate_path(args.hand), args.version)
        if not os.path.isdir(cand_root):
            raise FileNotFoundError(f"No candidates dir: {cand_root}")
        objects = sorted(os.listdir(cand_root))
        print(f"Found {len(objects)} objects for hand={args.hand} version={args.version}: {objects}\n")

    all_summaries = {}
    for obj_name in objects:
        print(f"\n{'#' * 60}")
        print(f"# {obj_name}")
        print(f"{'#' * 60}")
        try:
            obj_save_dir = args.save_dir
            if obj_save_dir is None and args.hand != "allegro":
                obj_save_dir = os.path.join("outputs", "reachability", args.hand, obj_name)
            summary = run_reachability(
                obj_name=obj_name,
                grasp_version=args.version,
                n_trials=args.n_trials,
                radii=radii,
                thetas_deg=thetas_deg,
                save_dir=obj_save_dir,
                hand=args.hand,
            )
            if summary:
                all_summaries[obj_name] = summary
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback
            traceback.print_exc()

    if len(all_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("ALL OBJECTS SUMMARY")
        print(f"{'=' * 60}")
        for obj_name, s in all_summaries.items():
            total = s["n_reachable"] + s["n_partial"] + s["n_unreachable"]
            print(f"  {obj_name}: reachable={s['n_reachable']}/{total}  "
                  f"partial={s['n_partial']}  unreachable={s['n_unreachable']}")
