#!/usr/bin/env python3
"""Generate reorient experiment figures.

For each trial under
    ~/shared_data/AutoDex/experiment/reset_test/reorient/{hand}/{obj}/{ts}/

produce a single PNG combining:
  (1) Headless Open3D render of scene+grasp
      (table + perceived object + inspire hand at the chosen reset grasp)
  (2) 4x5 grid of all 20 camera frames at the "grasp" state
      (state="grasp" — finger closure start)
  (3) 4x5 grid of all 20 camera frames at the "descent" state
      (state="descent" — start of lay-down descent after reorient)

Camera serials are stamped on every grid cell.
Output: ~/CORL_2026_latex/figure/reset/{obj}_{ts}.png

Usage:
    python src/visualization/reset_figure.py                          # all trials
    python src/visualization/reset_figure.py --obj donut              # one object
    python src/visualization/reset_figure.py --obj donut --ts 20260523_033519  # one trial
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation as R

# Make src/ importable so we can reuse the BODex reset-scene generator
sys.path.insert(0, str(Path.home() / "AutoDex"))
from src.grasp_generation.reorient.gen_scene import gen_reorient_scene  # noqa: E402

from paradex.visualization.robot import RobotModule  # noqa: E402


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "reset_test" / "reorient"
OUTPUT_DIR = Path.home() / "CORL_2026_latex" / "figure" / "reset"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"

HAND_URDF = {
    "inspire_left":  str(URDF_ROOT / "inspire_description" / "inspire_hand_left.urdf"),
    "inspire":       str(URDF_ROOT / "inspire_description" / "inspire_hand_right.urdf"),
    "allegro":       str(URDF_ROOT / "allegro_description" / "allegro_hand_description_right.urdf"),
}

# Which local axis of the wrist frame points OUT of the palm (palm front normal,
# i.e. the direction the palm "faces"). Camera will be placed along this axis.
# Empirically tuned per hand — flip the sign if the view ends up showing the
# back of the hand.
PALM_NORMAL_AXIS = {
    "inspire_left":  np.array([0.0, -1.0, 0.0]),
    "inspire":       np.array([0.0, -1.0, 0.0]),
    "allegro":       np.array([0.0,  0.0, 1.0]),
}

COLOR_OBJECT   = np.array([0.0, 100/255, 0.0])
COLOR_TABLE_I  = np.array([ 90, 155, 200]) / 255.0   # saturated sky blue — bottom (pose i)
COLOR_TABLE_J  = np.array([235, 145, 100]) / 255.0   # saturated coral   — top (pose j)
COLOR_PILLAR   = np.array([200, 175, 100]) / 255.0   # mustard           — virtual pillars
COLOR_ROBOT    = np.array([153, 128, 224]) / 255.0


# ───────────────────────── helpers ─────────────────────────

def trimesh_to_o3d(mesh, color=None):
    o = o3d.geometry.TriangleMesh()
    o.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o.compute_vertex_normals()
    if color is not None:
        o.paint_uniform_color(color)
    else:
        vc = None
        try:
            vc_arr = np.asarray(mesh.visual.vertex_colors)
            if vc_arr.ndim == 2 and vc_arr.shape[1] >= 3 and vc_arr.shape[0] == len(mesh.vertices):
                vc = vc_arr[:, :3] / 255.0
        except Exception:
            pass
        if vc is None:
            try:
                cv_ = mesh.visual.to_color()
                vc_arr = np.asarray(cv_.vertex_colors)
                if vc_arr.shape[0] == len(mesh.vertices):
                    vc = vc_arr[:, :3] / 255.0
            except Exception:
                pass
        if vc is not None:
            o.vertex_colors = o3d.utility.Vector3dVector(vc)
        else:
            o.paint_uniform_color([0.7, 0.7, 0.7])
    return o


_renderer = None
_renderer_size = (None, None)

def get_renderer(width, height):
    global _renderer, _renderer_size
    if _renderer is None or _renderer_size != (width, height):
        _renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        _renderer_size = (width, height)
    return _renderer


def pose7_to_mat4(p7):
    """[x, y, z, qw, qx, qy, qz] -> 4x4."""
    T = np.eye(4)
    T[:3, 3] = p7[:3]
    qw, qx, qy, qz = p7[3:]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


_AXIS_VEC = {
    "+x": np.array([ 1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([ 0.0, 1.0, 0.0]),
    "-y": np.array([ 0.0,-1.0, 0.0]),
    "+z": np.array([ 0.0, 0.0, 1.0]),
    "-z": np.array([ 0.0, 0.0,-1.0]),
}


def wrist_camera(world_T_wrist, frame_meshes,
                 offset_axis="+z", view_axis="-y",
                 dist=None, fov_deg=35.0, aspect=1.0, padding=1.15,
                 obj_mesh_for_dist=None, dist_mult=1.0, roll_deg=0.0,
                 lookat_point=None, auto_fov=True):
    """Wrist-relative camera with decoupled position and view direction.

    offset_axis: wrist-frame axis along which the camera is translated from
                 the wrist (eye = wrist + dist * R_wrist @ offset_axis).
    view_axis:   wrist-frame axis the camera optical axis points along
                 (camera looks along R_wrist @ view_axis).
    dist:        m; if None auto-sized to fit frame_meshes within fov.
    """
    R_wrist = world_T_wrist[:3, :3]
    wrist_pos = world_T_wrist[:3, 3]

    # offset_axis can be a single axis ("+z") or a comma-list summed and
    # normalized (e.g. "-z,+y" = diagonal). Same for view_axis.
    def _resolve(spec):
        vec = np.zeros(3)
        for tok in spec.split(","):
            tok = tok.strip()
            vec += _AXIS_VEC[tok]
        return vec / max(np.linalg.norm(vec), 1e-9)

    off_local  = _resolve(offset_axis)
    view_local = _resolve(view_axis)
    off_world  = R_wrist @ off_local
    view_world = R_wrist @ view_local
    off_world  /= max(np.linalg.norm(off_world),  1e-9)
    view_world /= max(np.linalg.norm(view_world), 1e-9)

    # Lookat: contact-point centroid if given, else obj+hand bbox center.
    if lookat_point is None:
        combined = trimesh.util.concatenate(list(frame_meshes))
        lookat = np.array(combined.bounding_sphere.primitive.center)
    else:
        lookat = np.asarray(lookat_point, dtype=float)

    # Forward direction = user-specified wrist view axis (fixed).
    # Camera is placed BACK along forward from lookat so the lookat point
    # is at the image center.
    forward = view_world

    combined = trimesh.util.concatenate(list(frame_meshes))
    bs = combined.bounding_sphere
    bs_center = np.array(bs.primitive.center)
    bs_rad = float(bs.primitive.radius)

    # Decompose (bs_center - lookat) into along-forward and perpendicular.
    d_lo  = bs_center - lookat
    d_par_off = float(d_lo @ forward)
    d_perp    = float(np.linalg.norm(d_lo - d_par_off * forward))

    # Use the requested fov_deg as the TARGET vertical FOV. Solve for dist
    # such that the obj+hand bounding sphere just fits: half-FOV must equal
    # angle(forward, eye→bs_center) + asin(rad / |eye→bs_center|).
    target_half = np.radians(fov_deg / 2)

    def _required_half(dist_):
        d_par = dist_ + d_par_off
        if d_par <= 0:
            return np.pi  # eye is past the sphere → invalid
        full = np.hypot(d_par, d_perp)
        if full <= bs_rad:
            return np.pi
        return np.arctan2(d_perp, d_par) + np.arcsin(bs_rad / full)

    if dist is None:
        # Binary search for the dist that gives required_half ≈ target_half/padding.
        target = target_half / padding
        lo, hi = 0.05, 2.5
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if _required_half(mid) > target:
                lo = mid
            else:
                hi = mid
        dist = hi

    dist = dist * dist_mult
    eye = lookat - dist * forward
    fov_deg_eff = fov_deg if not auto_fov else fov_deg

    # Pin image up = world +z (tabletop normal in scene frame), so every
    # trial's rendered image has a consistent gravity-aligned orientation.
    # setup_camera will orthogonalize against forward.
    forward = lookat - eye
    fnorm = np.linalg.norm(forward) + 1e-9
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) / fnorm > 0.95:
        # forward is too vertical (top-down or bottom-up) — fall back to +x
        up = np.array([1.0, 0.0, 0.0])

    if abs(roll_deg) > 1e-6:
        k = forward / fnorm
        t = np.radians(roll_deg)
        up = (up * np.cos(t)
              + np.cross(k, up) * np.sin(t)
              + k * (k @ up) * (1.0 - np.cos(t)))
    return eye, lookat, up, fov_deg_eff


def render_scene_grasp(meshes, hand_mesh, world_T_wrist,
                       width, height, fov,
                       offset_axis="+z", view_axis="-z", padding=1.15,
                       dist_mult=1.0, roll_deg=0.0, lookat_point=None,
                       auto_fov=True):
    """Render the BODex reorient scene + the chosen reset grasp.

    Camera is wrist-relative. eye = wrist + dist * (R_wrist @ offset_axis);
    optical axis = R_wrist @ view_axis. dist auto-sized to fit target+hand.
    """
    rend = get_renderer(width, height)
    rend.scene.clear_geometry()

    for name, (mesh, color, alpha) in meshes.items():
        mat = o3d.visualization.rendering.MaterialRecord()
        if alpha < 0.999:
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*color, alpha]
        else:
            mat.shader = "defaultLit"
        rend.scene.add_geometry(name, trimesh_to_o3d(mesh, color), mat)

    mat_robot = o3d.visualization.rendering.MaterialRecord(); mat_robot.shader = "defaultLit"
    rend.scene.add_geometry("robot", trimesh_to_o3d(hand_mesh, COLOR_ROBOT), mat_robot)

    frame_meshes = [meshes["target"][0], hand_mesh]
    eye, lookat, up, fov_eff = wrist_camera(
        world_T_wrist, frame_meshes,
        offset_axis=offset_axis, view_axis=view_axis,
        dist=None, fov_deg=fov, aspect=width / height, padding=padding,
        obj_mesh_for_dist=meshes["target"][0],
        dist_mult=dist_mult, roll_deg=roll_deg,
        lookat_point=lookat_point, auto_fov=auto_fov,
    )

    rend.scene.scene.set_sun_light([0.4, -0.4, -1.0], [1.0, 1.0, 1.0], 60000)
    rend.scene.scene.enable_sun_light(True)
    rend.scene.set_background([1.0, 1.0, 1.0, 1.0])
    rend.setup_camera(fov_eff, lookat, eye, up)
    img = np.asarray(rend.render_to_image())  # RGB uint8
    return img


def build_hand_mesh(hand_type, candidate_dir: Path, world_T_wrist):
    robot = RobotModule(HAND_URDF[hand_type])
    grasp_pose = np.load(candidate_dir / "grasp_pose.npy").flatten()
    n = min(robot.num_joints, len(grasp_pose))
    cfg = {name: angle for name, angle in zip(robot.joint_names[:n], grasp_pose[:n])}
    robot.update_cfg(cfg)
    mesh = robot.get_robot_mesh(collision_geometry=False)
    mesh.apply_transform(world_T_wrist)
    return mesh


def load_obj_mesh(obj_name):
    """Load the object mesh preserving its texture/visual. `process=False`
    keeps vertex/UV correspondences intact (force=mesh would otherwise merge
    submeshes and break texture mapping)."""
    p = OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    m = trimesh.load(str(p), process=False)
    if isinstance(m, trimesh.Scene):
        # Concatenate scene geometries into a single textured mesh.
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return m


def state_to_frame_idx(states, target_state, ts_array):
    """Map a logged state's ISO datetime → video frame index via timestamp.npy.
    Returns None if state not present or before/after the recording window."""
    rec = next((s for s in states if s["state"] == target_state), None)
    if rec is None:
        return None
    t_event = dt.datetime.fromisoformat(rec["time"]).timestamp()
    if t_event < ts_array[0] or t_event > ts_array[-1]:
        return None
    i = int(np.searchsorted(ts_array, t_event))
    if i >= len(ts_array):
        i = len(ts_array) - 1
    return i


def extract_frame(video_path: Path, frame_idx: int):
    """Read a single frame from an AVI. Returns BGR uint8 or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_idx, max(0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_label(img, text, color=(255, 255, 0)):
    """Stamp a small text label in the top-left corner of img (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, img.shape[1] / 600.0)
    thickness = max(1, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(img, (4, 4), (4 + tw + 2 * pad, 4 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (4 + pad, 4 + pad + th),
                font, font_scale, color, thickness, cv2.LINE_AA)


def build_paired_camera_grid(videos_dir: Path, frame_a: int, frame_b: int,
                             cell_w: int, cell_h: int,
                             rows: int = 2, cols: int = 2,
                             label_a: str = "grasp",
                             label_b: str = "descent",
                             serials_filter=None):
    """Per-camera paired grid. Each cell is [frame_a | frame_b] for one camera.
    Output dims: rows*cell_h tall, cols*(2*cell_w + sep)*1 wide (sep=2px gap).

    serials_filter: optional list of camera serials to use (else all .avi found,
                    truncated to rows*cols)."""
    if serials_filter is not None:
        serials = [s for s in serials_filter
                   if (videos_dir / f"{s}.avi").exists()]
    else:
        serials = sorted(p.stem for p in videos_dir.glob("*.avi"))
    n_cells = rows * cols
    if len(serials) > n_cells:
        serials = serials[:n_cells]

    SEP = 2  # thin black separator between the two views in a cell
    pair_w = 2 * cell_w + SEP
    grid = np.full((rows * cell_h, cols * pair_w, 3), 32, dtype=np.uint8)

    for i, serial in enumerate(serials):
        r, c = i // cols, i % cols
        vid = videos_dir / f"{serial}.avi"

        def _cell(idx):
            f = extract_frame(vid, idx)
            if f is None:
                return np.full((cell_h, cell_w, 3), 64, dtype=np.uint8)
            return cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        left  = _cell(frame_a)
        right = _cell(frame_b)

        # serial on left half, frame-label hints on top-left of each half
        _draw_label(left,  f"{serial}  [{label_a}]")
        _draw_label(right, f"[{label_b}]")

        cell = np.full((cell_h, pair_w, 3), 0, dtype=np.uint8)
        cell[:, :cell_w]              = left
        cell[:, cell_w + SEP:pair_w]  = right

        grid[r * cell_h:(r + 1) * cell_h, c * pair_w:(c + 1) * pair_w] = cell

    return grid  # BGR


def banner(img, text, color=(255, 255, 255)):
    """Add a thin title bar at the top. Mutates a copy."""
    h, w = img.shape[:2]
    bh = max(28, h // 30)
    bar = np.full((bh, w, 3), 0, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.6, bh / 50.0)
    th = max(1, int(fs * 2))
    cv2.putText(bar, text, (8, bh - 8), font, fs, color, th, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


# ───────────────────────── per-trial driver ─────────────────────────

def render_trial(hand, obj_name, ts_name, args):
    ep_dir = EXP_BASE / hand / obj_name / ts_name
    out_path = OUTPUT_DIR / f"{obj_name}_{ts_name}.png"
    if out_path.exists() and not args.overwrite:
        return "skip"

    result_path = ep_dir / "result.json"
    if not result_path.exists():
        return "no result.json"
    result = json.load(open(result_path))

    states = result.get("states") or []
    scene_info = result.get("scene_info") or {}
    cand_src = scene_info.get("source")
    if not cand_src:
        return "no scene_info.source"

    # Re-root the candidate path so it works regardless of where it was logged
    # (logs sometimes have /home/robot/... — rewrite the user prefix).
    cand_dir = Path(cand_src)
    if not cand_dir.exists():
        for prefix in ("/home/robot", "/home/mingi"):
            if str(cand_dir).startswith(prefix):
                alt = Path(str(cand_dir).replace(prefix, str(Path.home()), 1))
                if alt.exists():
                    cand_dir = alt
                    break
    if not cand_dir.exists():
        return f"missing candidate dir: {cand_src}"

    # Reorient scene parameters: which (i, j) pose pair and release height h
    # were used when this reset grasp was generated.
    cell = scene_info.get("cell", "")
    try:
        i_idx, j_idx = (int(x) for x in cell.split("_"))
    except Exception:
        return f"bad cell={cell!r}"
    h_cm = scene_info.get("h_cm", 0)
    h_m = float(h_cm) / 100.0

    # Re-generate the BODex reorient scene used at grasp generation time —
    # object at pose i (scene origin), table_i below, virtual support pillars
    # spanning to table_j (the pose-j placement surface).
    try:
        scene = gen_reorient_scene(obj_name, i_idx, j_idx, h_m)
    except Exception as e:
        return f"gen_reorient_scene failed: {e!r}"

    target_T = pose7_to_mat4(scene["scene"]["mesh"]["target"]["pose"])

    # wrist_se3.npy on disk is in object frame; compose to scene frame.
    wrist_obj = np.load(cand_dir / "wrist_se3.npy")
    world_T_wrist = target_T @ wrist_obj

    obj_mesh = load_obj_mesh(obj_name)
    obj_world = obj_mesh.copy(); obj_world.apply_transform(target_T)
    hand_mesh = build_hand_mesh(hand, cand_dir, world_T_wrist)

    # Contact-point centroid (lookat target) — BODex stores per-finger
    # contact points in the object frame as a (1, N, 6) array (2 points
    # per finger). Take mean of all valid points, transform to scene frame.
    lookat_point = None
    try:
        binfo = np.load(cand_dir / "bodex_info.npy", allow_pickle=True).item()
        cp = np.asarray(binfo["contact_point"]).reshape(-1, 3)
        cp_mean_obj = cp.mean(axis=0)
        lookat_point = (target_T @ np.r_[cp_mean_obj, 1.0])[:3]
    except Exception:
        # Fall back to obj+hand bounding sphere center (default in wrist_camera).
        pass

    # Assemble scene meshes: target + table_i + pillars + table_j.
    # COLOR_OBJECT=None → trimesh_to_o3d bakes the mesh's own texture/vertex
    # colors instead of painting a flat color.
    scene_meshes = {
        "target": (obj_world, None, 1.0),
    }
    for name, info in scene["scene"]["cuboid"].items():
        box = trimesh.creation.box(extents=info["dims"])
        box.apply_transform(pose7_to_mat4(info["pose"]))
        if name == "table_i":
            scene_meshes[name] = (box, COLOR_TABLE_I, 0.9)
        elif name == "table_j":
            scene_meshes[name] = (box, COLOR_TABLE_J, 0.45)
        elif name.startswith("pillar"):
            scene_meshes[name] = (box, COLOR_PILLAR, 0.6)
        else:
            scene_meshes[name] = (box, np.array([0.5, 0.5, 0.5]), 0.8)

    # ── (1) reorient-scene + grasp render ──
    left = render_scene_grasp(
        scene_meshes, hand_mesh, world_T_wrist,
        args.scene_size, args.scene_size, args.fov,
        offset_axis=args.offset_axis, view_axis=args.view_axis,
        padding=args.padding, dist_mult=args.dist_mult, roll_deg=args.roll,
        lookat_point=lookat_point, auto_fov=True,
    )  # RGB

    # ── (2,3) camera grids at grasp + descent ──
    ts_dir = ep_dir / "raw" / "timestamps"
    ts_path = ts_dir / "timestamp.npy"
    videos_dir = ep_dir / "videos"
    if not ts_path.exists() or not videos_dir.exists():
        return "no timestamps/videos"
    ts_array = np.load(ts_path)

    # "grasp"     = finger-closure start (object lifted off table)
    # "hand_init" = end of descent, fingers begin opening (object placed).
    # The state timestamp logs the *command*; actual physical settling lags
    # slightly, so add a small frame offset.
    grasp_idx   = state_to_frame_idx(states, "grasp",     ts_array)
    descent_idx = state_to_frame_idx(states, "hand_init", ts_array)
    if grasp_idx is None or descent_idx is None:
        return f"missing state grasp={grasp_idx} hand_init={descent_idx}"
    if args.descent_offset:
        descent_idx = min(len(ts_array) - 1, descent_idx + args.descent_offset)
    if args.grasp_offset:
        grasp_idx = min(len(ts_array) - 1, grasp_idx + args.grasp_offset)

    paired_grid = build_paired_camera_grid(
        videos_dir, grasp_idx, descent_idx, args.cell_w, args.cell_h,
        rows=args.grid_rows, cols=args.grid_cols,
        serials_filter=args.cam_serials,
    )  # BGR

    # ── compose: convert left to BGR, match heights, concat ──
    left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    target_h = max(left_bgr.shape[0], paired_grid.shape[0])

    def fit_height(im, h):
        if im.shape[0] == h:
            return im
        new_w = int(im.shape[1] * h / im.shape[0])
        return cv2.resize(im, (new_w, h), interpolation=cv2.INTER_AREA)

    left_bgr    = fit_height(left_bgr, target_h)
    paired_grid = fit_height(paired_grid, target_h)

    chosen_grasp = scene_info.get("grasp_idx", "?")
    yaw_deg = result.get("reorient", {}).get("chosen_yaw_deg", "?")
    left_bgr    = banner(left_bgr,
                         f"scene+grasp  hand={hand}  cell={cell}  h={h_cm}cm  "
                         f"grasp_idx={chosen_grasp}  yaw={yaw_deg}deg")
    paired_grid = banner(paired_grid,
                         f"per-cam: [grasp f={grasp_idx} | descent f={descent_idx}]")

    target_h2 = max(left_bgr.shape[0], paired_grid.shape[0])
    left_bgr    = fit_height(left_bgr, target_h2)
    paired_grid = fit_height(paired_grid, target_h2)

    canvas = np.concatenate([left_bgr, paired_grid], axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return f"ok  ({canvas.shape[1]}x{canvas.shape[0]})"


# ───────────────────────── discovery + CLI ─────────────────────────

def discover_trials(hand, obj_filter=None, ts_filter=None, include_drop=False):
    hand_dir = EXP_BASE / hand
    if not hand_dir.exists():
        return []
    out = []
    objects = sorted(d.name for d in hand_dir.iterdir() if d.is_dir())
    if obj_filter:
        objects = [o for o in objects if o in obj_filter]
    for obj in objects:
        for ep in sorted((hand_dir / obj).iterdir()):
            if not ep.is_dir():
                continue
            if not include_drop and ep.name.endswith("_drop"):
                continue
            if ts_filter and ep.name not in ts_filter:
                continue
            if not (ep / "result.json").exists():
                continue
            out.append((obj, ep.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left",
                    choices=["inspire_left", "inspire", "allegro"])
    ap.add_argument("--obj", nargs="+", default=None,
                    help="object name filter (e.g. donut pepsi)")
    ap.add_argument("--ts",  nargs="+", default=None,
                    help="timestamp folder filter (e.g. 20260523_033519)")
    ap.add_argument("--include-drop", action="store_true",
                    help="also include *_drop trials")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max", type=int, default=None)

    ap.add_argument("--scene-size", type=int, default=1800,
                    help="W=H of the left scene+grasp render panel (px)")
    ap.add_argument("--cell-w", type=int, default=560)
    ap.add_argument("--cell-h", type=int, default=420)
    ap.add_argument("--grid-rows", type=int, default=2)
    ap.add_argument("--grid-cols", type=int, default=2)
    ap.add_argument("--cam-serials", nargs="+",
                    default=["25322648", "25305466", "25322644", "25305463"],
                    help="camera serials to put in the paired grid (in order)")
    ap.add_argument("--grasp-offset", type=int, default=0,
                    help="extra frames to add after the 'grasp' state for "
                         "the pickup snapshot (use to compensate for "
                         "command-vs-actuation lag)")
    ap.add_argument("--descent-offset", type=int, default=15,
                    help="extra frames to add after the 'hand_init' state "
                         "for the descent-end snapshot (~0.5s at 30 FPS by "
                         "default; raise if the hand isn't fully opened yet)")
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--dist-mult", type=float, default=1.0,
                    help="multiply the auto-computed camera distance")
    ap.add_argument("--roll", type=float, default=0.0,
                    help="in-plane camera roll, deg (rotate the image)")
    ap.add_argument("--offset-axis", default="-z,+y",
                    help="wrist-frame axis for camera offset; comma-list "
                         "of ±x/±y/±z sums (e.g. '-z,+y' = diagonal)")
    ap.add_argument("--view-axis", default="-y",
                    help="wrist-frame axis for the camera optical axis "
                         "(comma-list supported, same syntax as offset)")
    ap.add_argument("--padding", type=float, default=0.9)

    args = ap.parse_args()

    obj_filter = set(args.obj) if args.obj else None
    ts_filter  = set(args.ts)  if args.ts  else None
    trials = discover_trials(args.hand, obj_filter, ts_filter, args.include_drop)
    if args.max:
        trials = trials[:args.max]

    print(f"[{args.hand}] {len(trials)} trials -> {OUTPUT_DIR}")
    for i, (obj, ts) in enumerate(trials):
        tag = f"[{i+1}/{len(trials)}]"
        try:
            status = render_trial(args.hand, obj, ts, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            status = f"ERR {e!r}"
        print(f"{tag} {obj}/{ts} -> {status}", flush=True)


if __name__ == "__main__":
    main()
