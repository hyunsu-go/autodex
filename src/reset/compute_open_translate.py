"""Method 2: open then translate

Stage 1: linearly interpolate finger qpos from pregrasp toward fully-open
         (zeros) at the grasp wrist pose. Pick the most-open α* per grasp
         that still has no collision against table + object mesh.
Stage 2: with the resulting q_open[i], sweep wrist 10cm along 12 directions
         (same as Method 1).

Saves outputs/reset/{obj}/open_translate_{version}.npz (includes q_open + alphas).

Usage:
    python src/reset/compute_open_translate.py --hand inspire_left --version v3 --obj_root /home/mingi/shared_data/AutoDex/object/robothome
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT, load_candidate

from common import (
    SWEEP_DIST, SWEEP_STEPS,
    get_all_objects, compute_sweep, save_npz,
    get_native_poses, find_open_qpos,
)


OPEN_STEPS = 11


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
    args = parser.parse_args()

    # Override the candidate root used by load_candidate/get_all_objects.
    import autodex.utils.path as _autopath
    _autopath.get_candidate_path = lambda hand: os.path.join(args.candidates_root, hand)

    z_rad = np.radians(args.z_rotation)

    objs = [args.obj] if args.obj else get_all_objects(args.hand, args.version, args.obj_root)
    if not objs:
        print("No objects to process.")
        return

    print(f"[open_translate] hand={args.hand} ver={args.version} N_obj={len(objs)} "
          f"open_steps={OPEN_STEPS} sweep={SWEEP_DIST*100:.0f}cm steps={SWEEP_STEPS}")
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

            native_poses = get_native_poses(
                scene_info, obj_name, args.x_offset, z_rad, args.obj_root
            )
            wrist_world = np.einsum('nij,njk->nik', native_poses, wrist_obj)

            print("  stage 1: open fingers...")
            q_open, alphas = find_open_qpos(
                planner, obj_name, scene_info, native_poses, wrist_world,
                pregrasp, args.obj_root, open_steps=OPEN_STEPS,
            )
            print(f"    avg α = {alphas.mean():.2f}  (range {alphas.min():.2f}-{alphas.max():.2f})")

            print("  stage 2: translate sweep with q_open...")
            def cb(gi, total, st, sid, n):
                print(f"    scene {gi+1}/{total}  {st}/{sid}  ({n} grasps)")

            result = compute_sweep(
                planner, obj_name, wrist_obj, pregrasp, scene_info,
                x_offset=args.x_offset, z_rad=z_rad, obj_root=args.obj_root,
                qpos_override=q_open, progress_cb=cb,
            )

            n_safe = int(result["dir_safe"].any(axis=1).sum())
            print(f"  → {n_safe}/{len(wrist_obj)} grasps have at least one safe direction")

            save_dir = os.path.join("outputs", "reset", obj_name)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"open_translate_{args.version}.npz")
            save_npz(
                out_path, result, scene_info, pregrasp,
                method="open_translate",
                args_dict={
                    "hand": args.hand, "version": args.version,
                    "x_offset": args.x_offset, "z_rotation": args.z_rotation,
                    "obj_root": args.obj_root, "open_steps": OPEN_STEPS,
                },
                obj_name=obj_name,
                extra_arrays={
                    "q_open": q_open.astype(np.float32),
                    "alphas": alphas.astype(np.float32),
                },
            )
            print(f"  saved: {out_path}")
        except Exception as ex:
            print(f"  [error] {ex}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
