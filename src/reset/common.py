"""Shared helpers for reset compute/view scripts."""
import os
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot

import autodex.utils.path as _autopath
from autodex.utils.conversion import cart2se3


SWEEP_DIST = 0.10
SWEEP_STEPS = 11
SWEEP_STEP = SWEEP_DIST / (SWEEP_STEPS - 1)
BBOX_EXPAND = 0.02

WORLD_AXES = np.array([
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
], dtype=np.float64)

DIR_NAMES = [
    "world +x", "world -x", "world +y", "world -y", "world +z", "world -z",
    "wrist +x", "wrist -x", "wrist +y", "wrist -y", "wrist +z", "wrist -z",
]


def load_scene_pose(obj_name, scene_type, scene_id, obj_root):
    p = os.path.join(obj_root, obj_name, "scene", scene_type, f"{scene_id}.json")
    with open(p) as f:
        j = json.load(f)
    return cart2se3(np.array(j["scene"]["mesh"]["target"]["pose"]))


def load_obb(obj_name, obj_root):
    p = os.path.join(obj_root, obj_name, "processed_data", "info", "simplified.json")
    with open(p) as f:
        info = json.load(f)
    return np.array(info["obb_transform"]), np.array(info["obb"]) / 2.0


def apply_xz_perturbation(pose, x_offset, z_rad):
    out = pose.copy()
    out[0, 3] += x_offset
    if z_rad != 0.0:
        c, s = np.cos(z_rad), np.sin(z_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[:3, :3] = Rz @ out[:3, :3]
    return out


def bbox_world_frame(obj_pose, obb_tf):
    return obj_pose @ obb_tf


def points_inside_bbox(points_world, bbox_world, half_ext_expanded):
    R = bbox_world[:3, :3]
    t = bbox_world[:3, 3]
    rel = points_world - t
    local = rel @ R
    return np.all(np.abs(local) <= half_ext_expanded, axis=-1)


def build_world_cfg(obj_name, obj_pose, obj_root):
    mesh_path = os.path.join(obj_root, obj_name, "processed_data", "mesh", "simplified.obj")
    quat_xyzw = Rot.from_matrix(obj_pose[:3, :3]).as_quat()
    pose7 = [
        float(obj_pose[0, 3]), float(obj_pose[1, 3]), float(obj_pose[2, 3]),
        float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]),
    ]
    cfg = {
        "cuboid": {
            "table": {
                "dims": [2, 2, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0],
                "color": [0.5, 0.5, 0.5, 1.0],
            },
        },
    }
    if os.path.exists(mesh_path):
        cfg["mesh"] = {
            "object": {"file_path": mesh_path, "pose": pose7},
        }
    return cfg


def list_tabletop_poses(obj_name, obj_root):
    """List enumerated canonical tabletop poses from info/tabletop/*.npy."""
    d = os.path.join(obj_root, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(d):
        return []
    return sorted(f.replace(".npy", "") for f in os.listdir(d) if f.endswith(".npy"))


def load_tabletop_pose(obj_name, pose_idx, obj_root, x_offset=0.0, z_rad=0.0):
    """Load a canonical tabletop pose and apply x_offset / z-rotation perturbation."""
    p = os.path.join(obj_root, obj_name, "processed_data", "info", "tabletop", f"{pose_idx}.npy")
    pose = np.load(p)
    out = pose.copy()
    out[0, 3] += x_offset
    if z_rad != 0.0:
        c, s = np.cos(z_rad), np.sin(z_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[:3, :3] = Rz @ out[:3, :3]
    return out


def get_native_poses(scene_info, obj_name, x_offset, z_rad, obj_root):
    """For each grasp, return its scene-native pose (with x_offset/z_rot applied)."""
    N = len(scene_info)
    poses = np.zeros((N, 4, 4))
    scene_pose_cache = {}
    for i, (st, sid, _) in enumerate(scene_info):
        key = (st, sid)
        if key not in scene_pose_cache:
            scene_pose_cache[key] = load_scene_pose(obj_name, st, sid, obj_root)
        poses[i] = apply_xz_perturbation(scene_pose_cache[key], x_offset, z_rad)
    return poses


def compute_sweep(planner, obj_name, wrist_obj, pregrasp, scene_info,
                  x_offset, z_rad, obj_root, qpos_override=None,
                  progress_cb=None):
    """12-direction × SWEEP_STEPS sweep, per-grasp using its native scene pose.

    qpos_override: (N, J) finger qpos to use (default = pregrasp).
    progress_cb(scene_idx, total, scene_type, scene_id, n_grasps): optional callback.

    Returns dict with all cached arrays.
    """
    N = len(wrist_obj)
    K = 12
    T = SWEEP_STEPS
    t_values = np.linspace(0.0, SWEEP_DIST, T)

    qpos_per_grasp = pregrasp if qpos_override is None else qpos_override

    obb_tf, half_ext = load_obb(obj_name, obj_root)
    half_ext_expanded = half_ext + BBOX_EXPAND

    native_poses = get_native_poses(scene_info, obj_name, x_offset, z_rad, obj_root)
    wrist_world = np.einsum('nij,njk->nik', native_poses, wrist_obj)

    dirs_world = np.broadcast_to(WORLD_AXES[None, :, :], (N, 6, 3))
    dirs_local = (wrist_world[:, :3, :3] @ WORLD_AXES.T).transpose(0, 2, 1)
    dirs = np.concatenate([dirs_world, dirs_local], axis=1)

    deltas = dirs[:, :, None, :] * t_values[None, None, :, None]
    poses = np.broadcast_to(wrist_world[:, None, None, :, :], (N, K, T, 4, 4)).copy()
    poses[:, :, :, :3, 3] = wrist_world[:, None, None, :3, 3] + deltas

    qpos_b = np.broadcast_to(
        qpos_per_grasp[:, None, None, :], (N, K, T, qpos_per_grasp.shape[1])
    ).copy()

    scene_groups = defaultdict(list)
    for i, (st, sid, _) in enumerate(scene_info):
        scene_groups[(st, sid)].append(i)

    centers_all = None
    collide_all = None
    radii_all = None
    bbox_worlds = np.zeros((N, 4, 4))

    for gi_scene, ((st, sid), grasp_idx_list) in enumerate(scene_groups.items()):
        if progress_cb:
            progress_cb(gi_scene, len(scene_groups), st, sid, len(grasp_idx_list))
        grasp_idx_arr = np.array(grasp_idx_list, dtype=int)
        scene_pose = native_poses[grasp_idx_arr[0]]
        world_cfg = build_world_cfg(obj_name, scene_pose, obj_root)

        poses_sub = poses[grasp_idx_arr].reshape(-1, 4, 4)
        qpos_sub = qpos_b[grasp_idx_arr].reshape(-1, qpos_b.shape[-1])
        centers_sub, radii, collide_sub = planner.check_collision_per_sphere(
            world_cfg, poses_sub, qpos_sub
        )
        M_sub = len(grasp_idx_arr)
        N_s = radii.shape[0]
        if centers_all is None:
            centers_all = np.zeros((N, K, T, N_s, 3))
            collide_all = np.zeros((N, K, T, N_s), dtype=bool)
            radii_all = radii
        centers_all[grasp_idx_arr] = centers_sub.reshape(M_sub, K, T, N_s, 3)
        collide_all[grasp_idx_arr] = collide_sub.reshape(M_sub, K, T, N_s)
        bbox_world_scene = bbox_world_frame(scene_pose, obb_tf)
        for gi in grasp_idx_list:
            bbox_worlds[gi] = bbox_world_scene

    in_bbox_all = np.zeros_like(collide_all)
    for gi in range(N):
        in_bbox_all[gi] = points_inside_bbox(
            centers_all[gi], bbox_worlds[gi], half_ext_expanded
        )

    coll_during = collide_all.any(axis=(2, 3))
    all_outside_final = ~in_bbox_all[:, :, -1, :].any(axis=2)
    dir_collision_free = ~coll_during
    dir_escape = all_outside_final
    dir_safe = dir_collision_free & dir_escape

    return {
        "centers": centers_all,
        "radii": radii_all,
        "collide": collide_all,
        "in_bbox": in_bbox_all,
        "native_poses": native_poses,
        "wrist_world": wrist_world,
        "dirs": dirs,
        "t_values": t_values,
        "bbox_worlds": bbox_worlds,
        "half_ext_expanded": half_ext_expanded,
        "dir_collision_free": dir_collision_free,
        "dir_escape": dir_escape,
        "dir_safe": dir_safe,
    }


def get_all_objects(hand, version, obj_root):
    # late binding so the monkey-patched get_candidate_path is respected
    cand_root = os.path.join(_autopath.get_candidate_path(hand), version)
    if not os.path.isdir(cand_root):
        return []
    cand_objs = set(os.listdir(cand_root))
    objs = []
    for obj_name in sorted(os.listdir(obj_root)):
        if obj_name not in cand_objs:
            continue
        d = os.path.join(obj_root, obj_name, "processed_data", "info", "tabletop")
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            objs.append(obj_name)
    return objs


def find_open_qpos(planner, obj_name, scene_info, native_poses, wrist_world,
                   pregrasp, obj_root, open_steps=11):
    """Stage 1: linearly interp pregrasp → zeros at the native wrist pose. For
    each (tabletop-pose, grasp) pair, pick α* that MAXIMIZES the minimum
    sphere-to-obstacle clearance (table + object mesh), among α values that
    keep all spheres collision-free."""
    N, J = pregrasp.shape
    K = open_steps
    alphas_grid = np.linspace(0.0, 1.0, K)
    open_target = np.zeros(J, dtype=pregrasp.dtype)

    q_interp = (
        pregrasp[:, None, :] * (1 - alphas_grid[None, :, None])
        + open_target[None, None, :] * alphas_grid[None, :, None]
    )  # (N, K, J)
    wrist_b = np.broadcast_to(wrist_world[:, None, :, :], (N, K, 4, 4)).copy()

    scene_groups = defaultdict(list)
    for i, (st, sid, _) in enumerate(scene_info):
        scene_groups[(st, sid)].append(i)

    # Per-(grasp, α) min clearance = min over spheres of (-ESDF - radius).
    # Positive = safe by that distance; negative = sphere overlaps obstacle.
    min_clearance = np.full((N, K), -np.inf, dtype=np.float64)
    for (st, sid), grasp_idx_list in scene_groups.items():
        grasp_idx_arr = np.array(grasp_idx_list, dtype=int)
        scene_pose = native_poses[grasp_idx_arr[0]]
        world_cfg = build_world_cfg(obj_name, scene_pose, obj_root)
        poses_sub = wrist_b[grasp_idx_arr].reshape(-1, 4, 4)
        qpos_sub = q_interp[grasp_idx_arr].reshape(-1, J)
        _, radii, esdf = planner.check_collision_per_sphere(
            world_cfg, poses_sub, qpos_sub, compute_esdf=True,
        )
        # esdf: (B, N_s), negative outside; clearance = -esdf - radius
        clearance = (-esdf) - radii[None, :]
        m = clearance.min(axis=1)  # bottleneck per state
        min_clearance[grasp_idx_arr] = m.reshape(len(grasp_idx_arr), K)

    # Pick α* per grasp: argmax clearance over the *contiguous safe range*
    # starting from α=0. This guarantees the open→close path is safe.
    safe_mask = min_clearance > 0  # (N, K)
    chosen_alphas = np.zeros(N, dtype=np.float64)
    for i in range(N):
        # Find longest contiguous-safe prefix from α=0.
        end = 0
        while end < K and safe_mask[i, end]:
            end += 1
        if end == 0:
            chosen_alphas[i] = 0.0
            continue
        prefix = slice(0, end)
        best_local = np.argmax(min_clearance[i, prefix])
        chosen_alphas[i] = alphas_grid[best_local]

    q_open = (
        pregrasp * (1 - chosen_alphas[:, None])
        + open_target[None, :] * chosen_alphas[:, None]
    )
    return q_open, chosen_alphas


def save_npz(out_path, result, scene_info, pregrasp, method, args_dict, obj_name,
             extra_arrays=None):
    payload = {
        "centers": result["centers"].astype(np.float32),
        "radii": result["radii"].astype(np.float32),
        "collide": result["collide"],
        "in_bbox": result["in_bbox"],
        "native_poses": result["native_poses"].astype(np.float32),
        "wrist_world": result["wrist_world"].astype(np.float32),
        "dirs": result["dirs"].astype(np.float32),
        "t_values": result["t_values"].astype(np.float32),
        "bbox_worlds": result["bbox_worlds"].astype(np.float32),
        "half_ext_expanded": result["half_ext_expanded"].astype(np.float32),
        "dir_collision_free": result["dir_collision_free"],
        "dir_escape": result["dir_escape"],
        "dir_safe": result["dir_safe"],
        "pregrasp": pregrasp.astype(np.float32),
        "scene_info": np.array(scene_info, dtype=object),
        "dir_names": np.array(DIR_NAMES),
        "meta": np.array([{
            "obj": obj_name,
            "method": method,
            "sweep_dist": SWEEP_DIST,
            "sweep_steps": SWEEP_STEPS,
            "bbox_expand": BBOX_EXPAND,
            **args_dict,
        }], dtype=object),
    }
    if extra_arrays:
        for k, v in extra_arrays.items():
            payload[k] = v
    np.savez_compressed(out_path, **payload)
