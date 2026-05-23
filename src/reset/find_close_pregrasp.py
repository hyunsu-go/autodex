"""Rank pregrasp candidates by min sphere-to-object distance.

Per grasp: build world with object only (no table), wrist at wrist_se3 (object
frame), forward kinematics on pregrasp qpos → per-sphere ESDF → clearance =
-esdf - radius. Take min over spheres = bottleneck per grasp. Sort ascending
to find pregrasps whose closest hand sphere is nearest to the object surface.

Usage:
    python src/reset/find_close_pregrasp.py --hand inspire_left --version table_only \
        --obj attached_container --top 20 \
        --candidates_root /home/mingi/shared_data/AutoDex/candidates
"""
import os
import sys
import argparse
import numpy as np

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT, load_candidate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import get_all_objects


def object_only_world_cfg(obj_name, obj_root):
    mesh_path = os.path.join(obj_root, obj_name, "processed_data", "mesh", "simplified.obj")
    return {"mesh": {"object": {"file_path": mesh_path,
                                "pose": [0, 0, 0, 1, 0, 0, 0]}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None)
    ap.add_argument("--version", default="table_only")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--threshold_mm", type=float, default=None,
                    help="optional: only print grasps with min clearance below this (mm)")
    ap.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    ap.add_argument("--candidates_root",
                    default=os.path.join(os.path.expanduser("~"), "AutoDex", "candidates"))
    args = ap.parse_args()

    import autodex.utils.path as _ap
    _ap.get_candidate_path = lambda hand: os.path.join(args.candidates_root, hand)

    objs = [args.obj] if args.obj else get_all_objects(args.hand, args.version, args.obj_root)
    if not objs:
        print("No objects to process."); return

    planner = GraspPlanner(hand=args.hand)

    for obj_name in objs:
        wrist_obj, pregrasp, _, scene_info = load_candidate(
            obj_name, np.eye(4), args.version,
            shuffle=False, skip_done=False, hand=args.hand,
        )
        N = len(wrist_obj)
        if N == 0:
            print(f"[{obj_name}] no candidates"); continue

        world_cfg = object_only_world_cfg(obj_name, args.obj_root)
        if not os.path.exists(world_cfg["mesh"]["object"]["file_path"]):
            print(f"[{obj_name}] no simplified mesh; skip"); continue

        _, radii, esdf = planner.check_collision_per_sphere(
            world_cfg, wrist_obj, pregrasp, compute_esdf=True,
        )
        # esdf: (N, N_s) signed dist, negative outside. clearance = -esdf - radius.
        clearance = (-esdf) - radii[None, :]                  # (N, N_s)
        per_grasp_min = clearance.min(axis=1)                 # (N,)  ← min sphere clearance per grasp

        order = np.argsort(per_grasp_min)
        if args.threshold_mm is not None:
            order = order[per_grasp_min[order] < args.threshold_mm / 1000.0]

        K = min(args.top, len(order))
        print(f"\n[{obj_name}] N={N}  showing top {K} closest pregrasps "
              f"(min sphere clearance, mm):")
        print(f"  {'rank':>4}  {'scene':<18} {'grasp':<6} {'min_mm':>8}  collide_n")
        for r in range(K):
            i = int(order[r])
            st, sid, gi = scene_info[i]
            n_coll = int((clearance[i] < 0).sum())
            print(f"  {r:>4}  {st+'/'+str(sid):<18} {gi:<6} "
                  f"{per_grasp_min[i] * 1000:>8.2f}  {n_coll}")

        out_dir = os.path.join("outputs", "close_pregrasp", obj_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.version.replace('/', '_')}.npz")
        np.savez_compressed(
            out_path,
            scene_info=np.array(scene_info, dtype=object),
            per_grasp_min=per_grasp_min.astype(np.float32),
            clearance=clearance.astype(np.float32),
            radii=radii.astype(np.float32),
            order=order.astype(np.int64),
        )
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
