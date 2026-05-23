"""Interactive: adjust arm_to_hand (link6 -> wrist) with sliders while watching
the hand-mesh overlay across all camera views (grid).

Sliders: tx/ty/tz (mm), rx/ry/rz (deg).
Keys: 's' = print current values, 'r' = reset, 'q' = quit (save if --out given).

Usage:
    python src/validation/adjust_arm_to_hand.py \
        --hand inspire_left --obj donut --date 20260523_032544_drop \
        --exp_name reset_test/naive_drop --frame 300
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))
sys.path.insert(0, str(REPO))

from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule

from autodex.utils.sync import convert_inspire_raw, ALLEGRO_POS_TO_URDF


def _import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_obj_mod = _import_from_path(
    "overlay_object_video_single",
    REPO / "src" / "visualization" / "overlay_object_video_single.py",
)
_rob_mod = _import_from_path(
    "overlay_robot_video",
    REPO / "src" / "visualization" / "overlay_robot_video.py",
)
ObjectOverlayRenderer = _obj_mod.ObjectOverlayRenderer
RobotOverlayRenderer = _rob_mod.RobotOverlayRenderer
_label_for_link = _rob_mod._label_for_link


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
URDF_BASE = REPO / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"


def _rpy_to_R(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_se3(tx, ty, tz, rx, ry, rz):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def _sync_one_frame(exp_dir, hand_type, t_target,
                    arm_time_offset=0.03, hand_time_offset=0.03):
    arm_dir = exp_dir / "raw" / "arm"
    hand_dir = exp_dir / "raw" / "hand"

    arm_t = np.load(arm_dir / "time.npy") + arm_time_offset
    arm_p = np.load(arm_dir / "position.npy")
    n = min(len(arm_t), len(arm_p))
    arm_t, arm_p = arm_t[:n], arm_p[:n]
    arm_row = arm_p[int(np.argmin(np.abs(arm_t - t_target)))]

    has_hand = (hand_dir / "time.npy").exists() and (hand_dir / "position.npy").exists()
    if has_hand:
        hand_t = np.load(hand_dir / "time.npy") + hand_time_offset
        hand_p = np.load(hand_dir / "position.npy")
        n = min(len(hand_t), len(hand_p))
        hand_t, hand_p = hand_t[:n], hand_p[:n]
        hidx = int(np.argmin(np.abs(hand_t - t_target)))
        hand_row = hand_p[hidx:hidx + 1]
    else:
        sq_path = exp_dir / "squeeze_hand.npy"
        hand_row = np.load(sq_path).reshape(1, -1)

    if hand_type in ("inspire", "inspire_left"):
        hand_row = convert_inspire_raw(hand_row)
    elif hand_type == "allegro":
        hand_row = hand_row[:, ALLEGRO_POS_TO_URDF]
    return arm_row, hand_row[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--exp_name", default="selected_100")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--serial", nargs="+", default=None,
                    help="restrict to specific camera serial(s)")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--with_arm", action="store_true",
                    help="also overlay arm mesh (default: hand only)")
    ap.add_argument("--with_object", action="store_true")
    ap.add_argument("--tile_w", type=int, default=480)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--crop_pad", type=int, default=40,
                    help="padding (px) added around the hand-joints bbox before tiling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    exp_dir = EXP_BASE / args.exp_name / args.hand / args.obj / args.date
    if not exp_dir.is_dir():
        raise FileNotFoundError(exp_dir)

    # Camera params
    intrinsic, extrinsic_cw = load_camparam(str(exp_dir))
    c2r = np.load(exp_dir / "C2R.npy")

    videos_dir = exp_dir / "videos"
    serials = sorted(s for s in intrinsic if (videos_dir / f"{s}.avi").exists())
    if args.serial:
        wanted = set(args.serial)
        serials = [s for s in serials if s in wanted]
    if not serials:
        raise RuntimeError("no serials found")

    # Open captures, get H,W and frame counts
    caps = {}
    frame_counts = []
    for s in serials:
        cap = cv2.VideoCapture(str(videos_dir / f"{s}.avi"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            continue
        caps[s] = cap
        frame_counts.append(n)
    serials = [s for s in serials if s in caps]
    if not serials:
        raise RuntimeError("no usable video captures")
    max_frame = min(frame_counts) - 1

    def read_frame(s, f):
        caps[s].set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = caps[s].read()
        return fr if ok else None

    probe = read_frame(serials[0], args.frame)
    if probe is None:
        raise RuntimeError(f"cannot read frame {args.frame}")
    H, W = probe.shape[:2]
    print(f"[info] {len(serials)} cameras, frame range [0,{max_frame}], {W}x{H}")
    # state holder; will be refreshed inside render_and_show when frame slider changes
    current_frame = [args.frame]
    raw_frames = {s: read_frame(s, args.frame) for s in serials}

    # qpos loader (per frame)
    video_times = np.load(exp_dir / "raw" / "timestamps" / "timestamp.npy")
    have_synced = ((exp_dir / "arm" / "state.npy").exists() and
                   (exp_dir / "hand" / "state.npy").exists())
    if have_synced:
        arm_synced = np.load(exp_dir / "arm" / "state.npy")
        hand_synced = np.load(exp_dir / "hand" / "state.npy")
    def get_qpos(f):
        if have_synced:
            return np.concatenate([arm_synced[f], hand_synced[f]])
        t_target = float(video_times[f])
        arm_row, hand_row = _sync_one_frame(exp_dir, args.hand, t_target)
        return np.concatenate([arm_row, hand_row])
    qpos = get_qpos(args.frame)

    # URDF + FK
    urdf_path = Path(args.urdf) if args.urdf \
        else URDF_BASE / f"{args.hand}_description" / f"xarm_{args.hand}.urdf"
    print(f"[info] urdf: {urdf_path}")
    robot = RobotModule(str(urdf_path))
    dof = robot.get_num_joints()
    robot.update_cfg(qpos[:dof])
    scene = robot.scene
    all_link_names = list(scene.geometry.keys())

    link_pose_default = {ln: scene.graph.get(ln)[0] for ln in all_link_names}
    link6_in_robot = scene.graph.get("link6")[0]
    try:
        anchor_in_robot = scene.graph.get("base_link")[0]
    except Exception:
        anchor_in_robot = None
        for ln in all_link_names:
            if "base_link" in ln.lower():
                anchor_in_robot = link_pose_default[ln]
                break
        if anchor_in_robot is None:
            raise RuntimeError("no base_link anchor found")

    def is_hand_side(name):
        s = name.lower()
        if any(k in s for k in ["base_link", "thumb", "index", "middle", "ring", "little", "pinky", "wrist"]):
            return True
        if re.match(r"link\d+\.", s) or s == "base.obj":
            return False
        return True

    # Pick which links to render
    if args.with_arm:
        render_names = all_link_names
    else:
        render_names = [ln for ln in all_link_names if is_hand_side(ln)]
    render_meshes = [scene.geometry[ln] for ln in render_names]
    render_labels = {ln: _label_for_link(ln) for ln in render_names}
    print(f"[info] rendering {len(render_names)} links (arm_included={args.with_arm})")

    # Renderer over all selected serials
    intr_subset = {s: intrinsic[s] for s in serials}
    extr_world = {s: extrinsic_cw[s][:3, :] for s in serials}
    robot_renderer = RobotOverlayRenderer(
        render_meshes, render_names, render_labels, intr_subset, extr_world, H, W,
    )
    ordered = robot_renderer.serials

    obj_renderer = None
    pose_world = None
    if args.with_object and (exp_dir / "pose_world.npy").exists():
        mesh_path = OBJ_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        intr_K = {s: intrinsic[s]["intrinsics_undistort"] for s in serials}
        extr_cw_dict = {s: extrinsic_cw[s] for s in serials}
        obj_renderer = ObjectOverlayRenderer(mesh, intr_K, extr_cw_dict, H, W)
        pose_world = np.load(exp_dir / "pose_world.npy")

    # URDF defaults for arm_to_hand
    txt = urdf_path.read_text()
    m = re.search(r'<joint name="arm_to_hand"[^>]*>.*?<origin\s+([^/]*?)/>', txt, re.S)
    body = m.group(1)
    def_xyz = [float(x) for x in re.search(r'xyz="([^"]+)"', body).group(1).split()]
    def_rpy = [float(x) for x in re.search(r'rpy="([^"]+)"', body).group(1).split()]
    init_tx_mm = def_xyz[0] * 1000
    init_ty_mm = def_xyz[1] * 1000
    init_tz_mm = def_xyz[2] * 1000
    init_rx_deg = float(np.degrees(def_rpy[0]))
    init_ry_deg = float(np.degrees(def_rpy[1]))
    init_rz_deg = float(np.degrees(def_rpy[2]))
    print(f"[info] default xyz_mm=({init_tx_mm:+.2f},{init_ty_mm:+.2f},{init_tz_mm:+.2f})  "
          f"rpy_deg=({init_rx_deg:+.2f},{init_ry_deg:+.2f},{init_rz_deg:+.2f})")

    # Slider scaling
    T_RANGE = 200.0
    R_RANGE = 180.0
    def t_to_slider(mm): return int(np.clip((mm + T_RANGE) * 10, 0, T_RANGE * 20))
    def t_from_slider(s): return s / 10.0 - T_RANGE
    def r_to_slider(deg): return int(np.clip((deg + R_RANGE) * 10, 0, R_RANGE * 20))
    def r_from_slider(s): return s / 10.0 - R_RANGE

    # Per-camera K (3x3) and world-from-cam extrinsic for projection
    K_per_cam = {s: np.asarray(intrinsic[s]["intrinsics_undistort"], dtype=np.float64) for s in serials}
    ext_per_cam = {s: np.asarray(extrinsic_cw[s], dtype=np.float64) for s in serials}

    def project_point(s, p_world):
        K3 = K_per_cam[s]
        E = ext_per_cam[s]
        Rcw = E[:3, :3]; tcw = E[:3, 3]
        p_cam = Rcw @ p_world + tcw
        if p_cam[2] <= 1e-6:
            return None
        u = (K3 @ p_cam) / p_cam[2]
        return float(u[0]), float(u[1])

    # Grid layout — per-cam bbox crop around projected hand joints, resize to tile_w
    tw = args.tile_w
    th = tw
    cols = args.cols
    rows = int(np.ceil(len(ordered) / cols))
    gw, gh = tw * cols, th * rows
    pad = args.crop_pad
    print(f"[info] grid {cols}x{rows} tile={tw}x{th} canvas={gw}x{gh}  pad={pad}px")

    # Names of hand-side links (used for bbox projection each frame)
    hand_side_names = [ln for ln in render_names if is_hand_side(ln)]

    nonlocal_state = {"link6": link6_in_robot, "anchor": anchor_in_robot}

    win = "overlay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_grid = [None]
    last_frame_drawn = [args.frame]

    def render_and_show():
        # frame slider
        f = cv2.getTrackbarPos("frame", win)
        f = max(0, min(f, max_frame))
        if f != last_frame_drawn[0]:
            # reload frames + qpos + FK + default link poses for new frame
            for s in serials:
                fr = read_frame(s, f)
                if fr is not None:
                    raw_frames[s] = fr
            current_frame[0] = f
            qp = get_qpos(f)
            robot.update_cfg(qp[:dof])
            for ln in render_names:
                link_pose_default[ln] = scene.graph.get(ln)[0]
            # re-bind link6 and anchor pose for this frame
            nonlocal_state["link6"] = scene.graph.get("link6")[0]
            try:
                nonlocal_state["anchor"] = scene.graph.get("base_link")[0]
            except Exception:
                pass
            last_frame_drawn[0] = f

        tx_mm = t_from_slider(cv2.getTrackbarPos("tx mm", win))
        ty_mm = t_from_slider(cv2.getTrackbarPos("ty mm", win))
        tz_mm = t_from_slider(cv2.getTrackbarPos("tz mm", win))
        rx_deg = r_from_slider(cv2.getTrackbarPos("rx deg", win))
        ry_deg = r_from_slider(cv2.getTrackbarPos("ry deg", win))
        rz_deg = r_from_slider(cv2.getTrackbarPos("rz deg", win))

        arm_to_hand = _make_se3(
            tx_mm / 1000, ty_mm / 1000, tz_mm / 1000,
            np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg),
        )
        new_anchor = nonlocal_state["link6"] @ arm_to_hand

        link_poses_robot = []
        for ln in render_names:
            T_def = link_pose_default[ln]
            if is_hand_side(ln):
                rel = np.linalg.inv(nonlocal_state["anchor"]) @ T_def
                link_poses_robot.append(new_anchor @ rel)
            else:
                link_poses_robot.append(T_def)
        link_poses_world = [c2r @ T for T in link_poses_robot]

        frames_list = [raw_frames[s] for s in ordered]
        if obj_renderer is not None and pose_world is not None:
            frames_list = obj_renderer.render(pose_world, frames_list)
        final = robot_renderer.render(link_poses_world, frames_list)

        # Collect all hand-side link world positions for bbox projection
        hand_pts_world = []
        for idx, ln in enumerate(render_names):
            if ln not in set(hand_side_names):
                continue
            T_world = link_poses_world[idx]
            hand_pts_world.append(T_world[:3, 3])
        hand_pts_world = np.array(hand_pts_world)

        grid = np.zeros((gh, gw, 3), dtype=np.uint8)
        for k, img in enumerate(final):
            r, c = k // cols, k % cols
            s = ordered[k]
            # project all hand joints to this camera
            us, vs = [], []
            for p in hand_pts_world:
                uv = project_point(s, p)
                if uv is not None:
                    us.append(uv[0]); vs.append(uv[1])
            if not us:
                tile_full = cv2.resize(img, (tw, th))
            else:
                x0 = int(round(min(us) - pad)); y0 = int(round(min(vs) - pad))
                x1 = int(round(max(us) + pad)); y1 = int(round(max(vs) + pad))
                # make square (use longer side) so resize keeps aspect
                bw, bh = x1 - x0, y1 - y0
                side = max(bw, bh)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                x0 = cx - side // 2; y0 = cy - side // 2
                x1 = x0 + side;      y1 = y0 + side
                tile_full = np.zeros((side, side, 3), dtype=np.uint8)
                sx0, sy0 = max(0, x0), max(0, y0)
                sx1, sy1 = min(W, x1), min(H, y1)
                if sx1 > sx0 and sy1 > sy0:
                    dx0, dy0 = sx0 - x0, sy0 - y0
                    tile_full[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
                tile_full = cv2.resize(tile_full, (tw, th))
            tile = tile_full
            cv2.putText(tile, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, s, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = tile

        txt_line = (f"xyz_mm=({tx_mm:+.1f},{ty_mm:+.1f},{tz_mm:+.1f})  "
                    f"rpy_deg=({rx_deg:+.1f},{ry_deg:+.1f},{rz_deg:+.1f})")
        cv2.putText(grid, txt_line, (10, gh - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, grid)
        last_grid[0] = grid
        # stdout (live, overwritten each change)
        import sys as _sys
        _sys.stdout.write(
            f'\r URDF arm_to_hand  xyz="{tx_mm/1000:.5f} {ty_mm/1000:.5f} {tz_mm/1000:.5f}"  '
            f'rpy="{np.radians(rx_deg):.6f} {np.radians(ry_deg):.6f} {np.radians(rz_deg):.6f}"  '
        )
        _sys.stdout.flush()

    def on_change(_v):
        render_and_show()

    cv2.createTrackbar("frame", win, args.frame, max_frame, on_change)
    cv2.createTrackbar("tx mm", win, t_to_slider(init_tx_mm), int(T_RANGE * 20), on_change)
    cv2.createTrackbar("ty mm", win, t_to_slider(init_ty_mm), int(T_RANGE * 20), on_change)
    cv2.createTrackbar("tz mm", win, t_to_slider(init_tz_mm), int(T_RANGE * 20), on_change)
    cv2.createTrackbar("rx deg", win, r_to_slider(init_rx_deg), int(R_RANGE * 20), on_change)
    cv2.createTrackbar("ry deg", win, r_to_slider(init_ry_deg), int(R_RANGE * 20), on_change)
    cv2.createTrackbar("rz deg", win, r_to_slider(init_rz_deg), int(R_RANGE * 20), on_change)

    render_and_show()
    print("\nKeys: q=quit (saves last grid if --out), s=print values, r=reset\n")

    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            tx_mm = t_from_slider(cv2.getTrackbarPos("tx mm", win))
            ty_mm = t_from_slider(cv2.getTrackbarPos("ty mm", win))
            tz_mm = t_from_slider(cv2.getTrackbarPos("tz mm", win))
            rx_deg = r_from_slider(cv2.getTrackbarPos("rx deg", win))
            ry_deg = r_from_slider(cv2.getTrackbarPos("ry deg", win))
            rz_deg = r_from_slider(cv2.getTrackbarPos("rz deg", win))
            print(f'URDF arm_to_hand:\n  xyz="{tx_mm/1000:.5f} {ty_mm/1000:.5f} {tz_mm/1000:.5f}"\n  '
                  f'rpy="{np.radians(rx_deg):.6f} {np.radians(ry_deg):.6f} {np.radians(rz_deg):.6f}"')
        elif k == ord('r'):
            cv2.setTrackbarPos("tx mm", win, t_to_slider(init_tx_mm))
            cv2.setTrackbarPos("ty mm", win, t_to_slider(init_ty_mm))
            cv2.setTrackbarPos("tz mm", win, t_to_slider(init_tz_mm))
            cv2.setTrackbarPos("rx deg", win, r_to_slider(init_rx_deg))
            cv2.setTrackbarPos("ry deg", win, r_to_slider(init_ry_deg))
            cv2.setTrackbarPos("rz deg", win, r_to_slider(init_rz_deg))

    cv2.destroyAllWindows()
    if args.out and last_grid[0] is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, last_grid[0])
        print(f"[done] saved {args.out}")


if __name__ == "__main__":
    main()
