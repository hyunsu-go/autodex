"""
Retreat Test

For each tabletop pose, find grasps that can retreat 5cm without colliding
with table or object mesh. Uses pregrasp finger config (open) for the sweep.

12 directions per grasp:
- 6 world axes: ±x, ±y, ±z
- 6 wrist axes: ±x, ±y, ±z (rotated to world per grasp)

Usage:
    python src/validation/planning/retreat_test.py --obj attached_container
"""

import os
import sys
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path, load_candidate


SWEEP_DIST = 0.05
SWEEP_STEPS = 6  # t = 0, 0.01, 0.02, 0.03, 0.04, 0.05

WORLD_AXES = np.array([
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
], dtype=np.float64)

DIR_NAMES = [
    "world +x", "world -x", "world +y", "world -y", "world +z", "world -z",
    "wrist +x", "wrist -x", "wrist +y", "wrist -y", "wrist +z", "wrist -z",
]


def get_tabletop_poses(obj_name):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted([f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy")])


def load_tabletop_pose(obj_name, pose_idx, x_offset=0.4, z_rotation=0.0):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    obj_pose = np.load(os.path.join(pose_dir, f"{pose_idx}.npy"))
    obj_pose[0, 3] += x_offset
    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]
    return obj_pose


def build_world_cfg(obj_name, obj_pose):
    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    quat_xyzw = Rot.from_matrix(obj_pose[:3, :3]).as_quat()
    pose7 = [
        float(obj_pose[0, 3]), float(obj_pose[1, 3]), float(obj_pose[2, 3]),
        float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]),
    ]
    return {
        "cuboid": {
            "table": {
                "dims": [2, 2, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
                "color": [0.5, 0.5, 0.5, 1.0],
            },
        },
        "mesh": {
            "object": {
                "file_path": mesh_path,
                "pose": pose7,
            },
        },
    }


def build_sweep_batch(wrist_world, pregrasp, t_values):
    """Returns flattened (N*K*T, 4, 4) poses and (N*K*T, J) qpos."""
    N = wrist_world.shape[0]
    K = 12
    T = len(t_values)
    J = pregrasp.shape[1]

    dirs_world = np.broadcast_to(WORLD_AXES[None, :, :], (N, 6, 3))
    dirs_local = (wrist_world[:, :3, :3] @ WORLD_AXES.T).transpose(0, 2, 1)
    dirs = np.concatenate([dirs_world, dirs_local], axis=1)  # (N, 12, 3)

    deltas = dirs[:, :, None, :] * t_values[None, None, :, None]  # (N, K, T, 3)

    poses = np.broadcast_to(wrist_world[:, None, None, :, :], (N, K, T, 4, 4)).copy()
    poses[:, :, :, :3, 3] = wrist_world[:, None, None, :3, 3] + deltas

    qpos = np.broadcast_to(pregrasp[:, None, None, :], (N, K, T, J)).copy()
    return poses.reshape(-1, 4, 4), qpos.reshape(-1, J)


def get_all_objects(hand, version):
    """Objects that have both tabletop poses and candidates for (hand, version)."""
    from autodex.utils.path import get_candidate_path
    cand_root = os.path.join(get_candidate_path(hand), version)
    if not os.path.isdir(cand_root):
        return []
    cand_objs = set(os.listdir(cand_root))
    objs = []
    for obj_name in sorted(os.listdir(obj_path)):
        if obj_name not in cand_objs:
            continue
        tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
        if os.path.isdir(tabletop_dir) and len(os.listdir(tabletop_dir)) > 0:
            objs.append(obj_name)
    return objs


def run_object(obj_name, args, planner, t_values, z_rad):
    pose_indices = get_tabletop_poses(obj_name)
    if not pose_indices:
        print(f"  [skip] no tabletop poses")
        return None

    identity = np.eye(4)
    wrist_obj, pregrasp, grasp, scene_info = load_candidate(
        obj_name, identity, args.version, shuffle=False, skip_done=False, hand=args.hand
    )
    N = len(wrist_obj)
    if N == 0:
        print(f"  [skip] no candidates")
        return None

    per_pose_result = {}
    for pose_idx in pose_indices:
        obj_pose = load_tabletop_pose(obj_name, pose_idx, args.x_offset, z_rad)
        world_cfg = build_world_cfg(obj_name, obj_pose)
        wrist_world = obj_pose @ wrist_obj

        poses_flat, qpos_flat = build_sweep_batch(wrist_world, pregrasp, t_values)
        coll_flat = planner._check_collision(world_cfg, poses_flat, qpos_flat)
        coll = coll_flat.reshape(N, 12, SWEEP_STEPS)

        retreat_safe = ~coll.any(axis=2)
        any_retreat = retreat_safe.any(axis=1)
        per_dir_count = retreat_safe.sum(axis=0)

        per_pose_result[pose_idx] = {
            "n_any_retreat": int(any_retreat.sum()),
            "any_retreat_indices": np.where(any_retreat)[0].tolist(),
            "per_dir_count": per_dir_count.tolist(),
            "retreat_safe": retreat_safe.tolist(),
        }

        print(f"  Pose {pose_idx}: any_retreat={any_retreat.sum()}/{N}")

    universal = np.ones(N, dtype=bool)
    for r in per_pose_result.values():
        m = np.zeros(N, dtype=bool)
        m[r["any_retreat_indices"]] = True
        universal &= m
    universal_idx = np.where(universal)[0].tolist()
    print(f"  Universal (retreat in ALL {len(pose_indices)} poses): {len(universal_idx)}/{N}")

    save_dir = os.path.join("outputs", "retreat_test", obj_name)
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "obj_name": obj_name,
        "version": args.version,
        "hand": args.hand,
        "x_offset": args.x_offset,
        "z_rotation_deg": args.z_rotation,
        "sweep_dist": SWEEP_DIST,
        "sweep_steps": SWEEP_STEPS,
        "dir_names": DIR_NAMES,
        "pose_indices": pose_indices,
        "n_candidates": N,
        "scene_info": scene_info,
        "per_pose": per_pose_result,
        "universal_grasps": universal_idx,
    }
    out_path = os.path.join(save_dir, f"retreat_{args.version}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default=None,
                        help="Single object name. If omitted, iterate all available.")
    parser.add_argument("--version", type=str, default="selected_100")
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--z_rotation", type=float, default=0.0)
    args = parser.parse_args()

    z_rad = np.radians(args.z_rotation)
    t_values = np.linspace(0.0, SWEEP_DIST, SWEEP_STEPS)

    if args.obj:
        objs = [args.obj]
    else:
        objs = get_all_objects(args.hand, args.version)
        print(f"Running over {len(objs)} objects.")

    if not objs:
        print("No objects to process.")
        return

    print(f"Hand: {args.hand} | Version: {args.version}")
    print(f"Sweep: {SWEEP_STEPS} steps up to {SWEEP_DIST*100:.0f}cm")
    print("=" * 70)

    planner = GraspPlanner(hand=args.hand)

    for i, obj_name in enumerate(objs):
        print(f"\n[{i+1}/{len(objs)}] {obj_name}")
        try:
            run_object(obj_name, args, planner, t_values, z_rad)
        except Exception as ex:
            print(f"  [error] {ex}")


if __name__ == "__main__":
    main()
