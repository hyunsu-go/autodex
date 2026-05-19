"""Compare an object's tabletop poses after factoring out gravity-axis (yaw) rotation.

For each pose pair (i, j), find the yaw rotation about world z that best aligns
pose_j to pose_i. Report the residual angular difference. Also visualize
selected pair side-by-side in viser.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH


def best_yaw_align(R_i: np.ndarray, R_j: np.ndarray):
    """Find yaw angle alpha and residual angular distance.

    Goal: R_z(alpha) @ R_j ≈ R_i, find alpha and residual.
    Let M = R_i @ R_j.T. We seek alpha maximizing trace(M @ R_z(-alpha)).
    Closed-form: alpha = atan2(M10 - M01, M00 + M11).
    Residual angle = acos((max_trace - 1) / 2).
    """
    M = R_i @ R_j.T
    A = M[0, 0] + M[1, 1]
    B = M[1, 0] - M[0, 1]
    C = M[2, 2]
    alpha = float(np.arctan2(B, A))
    max_trace = float(np.hypot(A, B) + C)
    cos_res = (max_trace - 1) / 2
    cos_res = max(-1.0, min(1.0, cos_res))
    residual_rad = float(np.arccos(cos_res))
    return alpha, residual_rad


def Rz(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def residual_table(obj_name: str, obj_root: Path):
    td = obj_root / obj_name / "processed_data" / "info" / "tabletop"
    pids = sorted([int(p.stem) for p in td.glob("*.npy")])
    Ts = {i: np.load(td / f"{i:03d}.npy") for i in pids}
    raw = np.zeros((len(pids), len(pids)))
    res = np.zeros((len(pids), len(pids)))
    yaw = np.zeros((len(pids), len(pids)))
    for a, i in enumerate(pids):
        for b, j in enumerate(pids):
            if i == j:
                continue
            dR = Ts[i][:3, :3] @ Ts[j][:3, :3].T
            raw_ang = np.rad2deg(R.from_matrix(dR).magnitude())
            alpha, residual = best_yaw_align(Ts[i][:3, :3], Ts[j][:3, :3])
            raw[a, b] = raw_ang
            res[a, b] = np.rad2deg(residual)
            yaw[a, b] = np.rad2deg(alpha)
    return pids, raw, res, yaw


def print_tables(obj_name, pids, raw, res, yaw):
    n = len(pids)
    print(f"\n[{obj_name}] poses: {pids}")
    print("\nraw angle (deg) — no alignment:")
    print("     " + " ".join(f"{j:>6}" for j in pids))
    for a, i in enumerate(pids):
        print(f"{i:>3}: " + " ".join(f"{raw[a,b]:>6.1f}" if a != b else "    -" for b in range(n)))
    print("\nresidual after best yaw align (deg) — small => duplicate up to yaw:")
    print("     " + " ".join(f"{j:>6}" for j in pids))
    for a, i in enumerate(pids):
        print(f"{i:>3}: " + " ".join(f"{res[a,b]:>6.1f}" if a != b else "    -" for b in range(n)))


def viz_compare(obj_name, obj_root: Path, pose_i: int, pose_j: int, port: int):
    td = obj_root / obj_name / "processed_data" / "info" / "tabletop"
    T_i = np.load(td / f"{pose_i:03d}.npy")
    T_j = np.load(td / f"{pose_j:03d}.npy")
    alpha, residual = best_yaw_align(T_i[:3, :3], T_j[:3, :3])
    R_yaw = Rz(alpha)

    mesh_path = obj_root / obj_name / "processed_data" / "mesh" / "simplified.obj"
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")

    server = viser.ViserServer(port=port)
    server.scene.world_axes.visible = True

    def add(name, R_mat, t, color):
        m = mesh.copy()
        m.visual.face_colors = np.tile(np.array(list(color) + [200], dtype=np.uint8),
                                        (len(m.faces), 1))
        q_xyzw = R.from_matrix(R_mat).as_quat()
        wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        server.scene.add_mesh_trimesh(name, m, position=np.asarray(t, dtype=np.float64), wxyz=wxyz)

    # pose i (red), pose j original at +x offset (blue),
    # pose j with yaw alignment at +x*2 offset (green)
    # overlay all three at the same position so differences are visible directly
    center = T_i[:3, 3]
    add("/pose_i", T_i[:3, :3], center, (220, 60, 60))
    add("/pose_j_raw", T_j[:3, :3], center, (60, 80, 220))
    add("/pose_j_aligned", R_yaw @ T_j[:3, :3], center, (60, 200, 80))

    raw_ang = np.rad2deg(R.from_matrix(T_i[:3, :3] @ T_j[:3, :3].T).magnitude())
    print(f"\npose {pose_i} vs {pose_j}:")
    print(f"  raw angle:           {raw_ang:.2f} deg")
    print(f"  best yaw rotation:   {np.rad2deg(alpha):.2f} deg")
    print(f"  residual after yaw:  {np.rad2deg(residual):.2f} deg")
    print(f"\n  red    = pose {pose_i}")
    print(f"  blue   = pose {pose_j} (original)")
    print(f"  green  = pose {pose_j} after yaw rotation (≈ pose {pose_i} if residual small)")
    print(f"\nviser on http://localhost:{port}")

    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--obj_root", type=Path, default=Path(DEFAULT_OBJ_PATH))
    parser.add_argument("--table_only", action="store_true",
                        help="print tables only, no viewer")
    parser.add_argument("--i", type=int, help="pose i for viewer")
    parser.add_argument("--j", type=int, help="pose j for viewer")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    pids, raw, res, yaw = residual_table(args.obj, args.obj_root)
    print_tables(args.obj, pids, raw, res, yaw)

    if args.table_only:
        return
    if args.i is None or args.j is None:
        # default to first off-diagonal pair
        args.i, args.j = pids[0], pids[1]
        print(f"\n(no --i/--j given, using {args.i} vs {args.j})")
    viz_compare(args.obj, args.obj_root, args.i, args.j, args.port)


if __name__ == "__main__":
    main()
