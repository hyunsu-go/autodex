"""Method 2: open then gradient

Stage 1: open fingers (same as open_translate).
Stage 2: gradient descent on wrist xyz (rotation fixed). Loss = Σ ESDF of hand
         spheres w.r.t. world geometry (table + scene's object mesh). Negative
         ESDF = outside obstacle; we want it more negative (away from obstacles).
         Step lr default ~0.005, max iter 30. No projection.

After gradient: re-evaluate sphere positions / per-sphere collide / per-sphere
in-bbox along the trajectory for visualization. Save npz with K=1 (no
direction axis) and T=n_iter+1 (iteration axis).

Usage:
    python src/reset/compute_open_gradient.py --hand inspire_left --version v3 \\
        --obj_root /home/mingi/shared_data/AutoDex/object/robothome
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT, load_candidate
from autodex.utils.conversion import se32action, se32cart

from common import (
    BBOX_EXPAND,
    get_all_objects, save_npz,
    get_native_poses, find_open_qpos,
    load_obb, bbox_world_frame, points_inside_bbox, build_world_cfg,
)

from curobo.geom.types import WorldConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer


N_ITER = 30
LR = 0.005
OPEN_STEPS = 11


def gradient_descent_wrist_per_scene(planner, world_cfg, wrist_se3, qpos, n_iter, lr):
    """Per-scene batched gradient descent on wrist xyz.

    wrist_se3: (M, 4, 4)  — initial wrist poses
    qpos:      (M, J)     — fixed finger configs
    Returns trajectory (M, n_iter+1, 4, 4).
    """
    M = wrist_se3.shape[0]
    J = qpos.shape[1]
    device = planner._tensor_args.device

    rw_config = RobotWorldConfig.load_from_config(
        planner._hand_cfg,
        WorldConfig.from_dict(world_cfg),
        collision_activation_distance=0.0,
        tensor_args=planner._tensor_args,
    )
    rw = RobotWorld(rw_config)
    n_dof = rw.kinematics.get_dof()

    if n_dof == J + 6:
        q_init_np = np.array([se32action(w, g) for w, g in zip(wrist_se3, qpos)])
    else:
        q_init_np = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, qpos)])
        if q_init_np.shape[1] != n_dof:
            q_init_np = qpos

    q_init = torch.tensor(q_init_np, dtype=torch.float32, device=device)
    delta = torch.zeros((M, 3), dtype=torch.float32, device=device, requires_grad=True)

    weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    act = torch.tensor([0.0], dtype=torch.float32, device=device)

    trajectory = np.zeros((M, n_iter + 1, 4, 4))
    trajectory[:, 0] = wrist_se3

    buffer = None
    n_spheres = None

    for t_iter in range(n_iter):
        # Construct q with current delta on the xyz columns; keep rotation/finger fixed.
        q = torch.cat([q_init[:, 0:3] + delta, q_init[:, 3:]], dim=1)

        state = rw.get_kinematics(q)
        spheres = state.link_spheres_tensor.unsqueeze(1)  # (M, 1, N_s, 4)

        if buffer is None:
            buffer = CollisionQueryBuffer.initialize_from_shape(
                spheres.shape, planner._tensor_args, rw.world_model.collision_types
            )
            n_spheres = spheres.shape[2]
            import inspect
            _sig = inspect.signature(rw.world_model.get_sphere_distance)
            _needs_contact_dist = "contact_distance" in _sig.parameters

        kwargs = {"sum_collisions": False, "compute_esdf": True}
        if _needs_contact_dist:
            kwargs["contact_distance"] = torch.zeros(
                n_spheres, dtype=torch.float32, device=device,
            )
        d = rw.world_model.get_sphere_distance(spheres, buffer, weight, act, **kwargs)
        # d (M, 1, N_s): <0 outside, >0 inside

        loss = d.sum()
        loss.backward()

        with torch.no_grad():
            delta -= lr * delta.grad
            delta.grad.zero_()

        new_wrist = wrist_se3.copy()
        new_wrist[:, :3, 3] = wrist_se3[:, :3, 3] + delta.detach().cpu().numpy()
        trajectory[:, t_iter + 1] = new_wrist

    return trajectory


def evaluate_trajectory(planner, obj_name, scene_info, native_poses, trajectories,
                       q_open, obj_root, half_ext_expanded, obb_tf):
    """Per-step sphere positions, world collision, and bbox membership."""
    N, T = trajectories.shape[0], trajectories.shape[1]

    scene_groups = defaultdict(list)
    for i, (st, sid, _) in enumerate(scene_info):
        scene_groups[(st, sid)].append(i)

    centers_all = None
    collide_all = None
    radii = None
    bbox_worlds = np.zeros((N, 4, 4))

    for (st, sid), grasp_idx_list in scene_groups.items():
        grasp_idx_arr = np.array(grasp_idx_list, dtype=int)
        scene_pose = native_poses[grasp_idx_arr[0]]
        world_cfg = build_world_cfg(obj_name, scene_pose, obj_root)

        M = len(grasp_idx_arr)
        poses_sub = trajectories[grasp_idx_arr].reshape(M * T, 4, 4)
        qpos_sub = np.repeat(q_open[grasp_idx_arr], T, axis=0)

        centers_sub, radii_, collide_sub = planner.check_collision_per_sphere(
            world_cfg, poses_sub, qpos_sub
        )
        N_s = radii_.shape[0]
        if centers_all is None:
            centers_all = np.zeros((N, T, N_s, 3))
            collide_all = np.zeros((N, T, N_s), dtype=bool)
            radii = radii_
        centers_all[grasp_idx_arr] = centers_sub.reshape(M, T, N_s, 3)
        collide_all[grasp_idx_arr] = collide_sub.reshape(M, T, N_s)

        bbox_world_scene = bbox_world_frame(scene_pose, obb_tf)
        for gi in grasp_idx_list:
            bbox_worlds[gi] = bbox_world_scene

    in_bbox = np.zeros_like(collide_all)
    for gi in range(N):
        in_bbox[gi] = points_inside_bbox(centers_all[gi], bbox_worlds[gi], half_ext_expanded)

    return centers_all, radii, collide_all, in_bbox, bbox_worlds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default=None)
    parser.add_argument("--version", default="v3")
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--z_rotation", type=float, default=0.0)
    parser.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    parser.add_argument("--candidates_root",
                        default=os.path.join(os.path.expanduser("~"), "AutoDex", "candidates"),
                        help="Root directory containing {hand}/{version}/{obj}/.")
    parser.add_argument("--n_iter", type=int, default=N_ITER)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    import autodex.utils.path as _autopath
    _autopath.get_candidate_path = lambda hand: os.path.join(args.candidates_root, hand)

    z_rad = np.radians(args.z_rotation)

    objs = [args.obj] if args.obj else get_all_objects(args.hand, args.version, args.obj_root)
    if not objs:
        print("No objects to process.")
        return

    print(f"[open_gradient] hand={args.hand} ver={args.version} N_obj={len(objs)}  "
          f"n_iter={args.n_iter} lr={args.lr}")
    print("=" * 70)

    planner = GraspPlanner(hand=args.hand)

    for oi, obj_name in enumerate(objs):
        print(f"\n[{oi+1}/{len(objs)}] {obj_name}")
        try:
            wrist_obj, pregrasp, _, scene_info = load_candidate(
                obj_name, np.eye(4), args.version,
                shuffle=False, skip_done=False, hand=args.hand,
            )
            if len(wrist_obj) == 0:
                print("  [skip] no candidates")
                continue

            N = len(wrist_obj)
            native_poses = get_native_poses(scene_info, obj_name, args.x_offset, z_rad, args.obj_root)
            wrist_world = np.einsum('nij,njk->nik', native_poses, wrist_obj)

            print("  stage 1: open fingers...")
            q_open, alphas = find_open_qpos(
                planner, obj_name, scene_info, native_poses, wrist_world,
                pregrasp, args.obj_root, open_steps=OPEN_STEPS,
            )
            print(f"    avg α = {alphas.mean():.2f}  (range {alphas.min():.2f}-{alphas.max():.2f})")

            print(f"  stage 2: gradient descent (n_iter={args.n_iter}, lr={args.lr})...")
            scene_groups = defaultdict(list)
            for i, (st, sid, _) in enumerate(scene_info):
                scene_groups[(st, sid)].append(i)

            trajectories = np.zeros((N, args.n_iter + 1, 4, 4))
            for gi_scene, ((st, sid), grasp_idx_list) in enumerate(scene_groups.items()):
                print(f"    scene {gi_scene+1}/{len(scene_groups)}  {st}/{sid}  "
                      f"({len(grasp_idx_list)} grasps)")
                grasp_idx_arr = np.array(grasp_idx_list, dtype=int)
                scene_pose = native_poses[grasp_idx_arr[0]]
                world_cfg = build_world_cfg(obj_name, scene_pose, args.obj_root)
                traj_sub = gradient_descent_wrist_per_scene(
                    planner, world_cfg,
                    wrist_world[grasp_idx_arr], q_open[grasp_idx_arr],
                    n_iter=args.n_iter, lr=args.lr,
                )
                trajectories[grasp_idx_arr] = traj_sub

            obb_tf, half_ext = load_obb(obj_name, args.obj_root)
            half_ext_expanded = half_ext + BBOX_EXPAND

            print("  evaluating trajectory...")
            centers, radii, collide, in_bbox, bbox_worlds = evaluate_trajectory(
                planner, obj_name, scene_info, native_poses, trajectories,
                q_open, args.obj_root, half_ext_expanded, obb_tf,
            )
            T = trajectories.shape[1]

            # Reshape to (N, K=1, T, N_s, ...) to share schema with translate npz
            centers = centers[:, None]
            collide = collide[:, None]
            in_bbox = in_bbox[:, None]

            # dirs: initial→final wrist direction (for retreat-line viz)
            dirs_final = trajectories[:, -1, :3, 3] - trajectories[:, 0, :3, 3]
            norm = np.linalg.norm(dirs_final, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1.0, norm)
            dirs = (dirs_final / norm)[:, None, :]  # (N, 1, 3)

            coll_during = collide.any(axis=(2, 3))               # (N, 1)
            all_outside_final = ~in_bbox[:, :, -1, :].any(axis=2) # (N, 1)
            dir_collision_free = ~coll_during
            dir_escape = all_outside_final
            dir_safe = dir_collision_free & dir_escape

            result = {
                "centers": centers,
                "radii": radii,
                "collide": collide,
                "in_bbox": in_bbox,
                "native_poses": native_poses,
                "wrist_world": wrist_world,
                "dirs": dirs,
                "t_values": np.linspace(0.0, 1.0, T),
                "bbox_worlds": bbox_worlds,
                "half_ext_expanded": half_ext_expanded,
                "dir_collision_free": dir_collision_free,
                "dir_escape": dir_escape,
                "dir_safe": dir_safe,
            }

            n_safe = int(dir_safe.any(axis=1).sum())
            print(f"  → {n_safe}/{N} grasps have a safe gradient endpoint")

            save_dir = os.path.join("outputs", "reset", obj_name)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"open_gradient_{args.version}.npz")
            save_npz(
                out_path, result, scene_info, pregrasp,
                method="open_gradient",
                args_dict={
                    "hand": args.hand, "version": args.version,
                    "x_offset": args.x_offset, "z_rotation": args.z_rotation,
                    "obj_root": args.obj_root,
                    "open_steps": OPEN_STEPS,
                    "n_iter": args.n_iter, "lr": args.lr,
                },
                obj_name=obj_name,
                extra_arrays={
                    "trajectories": trajectories.astype(np.float32),
                    "q_open": q_open.astype(np.float32),
                    "alphas": alphas.astype(np.float32),
                    "dir_names": np.array(["grad"]),
                },
            )
            print(f"  saved: {out_path}")
        except Exception as ex:
            print(f"  [error] {ex}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
