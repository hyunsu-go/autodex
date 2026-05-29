#!/usr/bin/env python3
"""Paper figure for Sec.3 Reset (fig:reset).

For each chosen trial, build a 5-panel row:
    [start pose] [goal pose] [scene+grasp render] [grasp] [place]

Panels 1, 2, 4, 5 are real frames from a single chosen camera at the
states init, clear_view, grasp, hand_init+offset. Panel 3 is the Open3D
render reused from reset_figure.render_scene_grasp.

Multiple trials are stacked vertically.

Output: ~/CORL_2026_latex/AutoDex_CoRL/figures/method/reset_row.pdf
        (and .png alongside for preview)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path.home() / "AutoDex"))
from src.visualization.reset_figure import (  # noqa: E402
    COLOR_ROBOT,
    COLOR_TABLE_I, COLOR_TABLE_J, COLOR_PILLAR,
    build_hand_mesh, extract_frame, get_renderer, load_obj_mesh,
    pose7_to_mat4, state_to_frame_idx, trimesh_to_o3d,
)
from src.grasp_generation.reorient.gen_scene import gen_reorient_scene  # noqa: E402


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "reset_test" / "reorient"
OUT_DIR  = Path.home() / "CORL_2026_latex" / "AutoDex_CoRL" / "figures" / "method"

_NO_SCENE_AUTOCROP = False


def reroot(p: Path) -> Path:
    if p.exists():
        return p
    for prefix in ("/home/robot", "/home/mingi"):
        if str(p).startswith(prefix):
            alt = Path(str(p).replace(prefix, str(Path.home()), 1))
            if alt.exists():
                return alt
    return p


def render_scene_external(meshes, hand_mesh, width, height,
                          azim_deg: float, elev_deg: float,
                          fov_deg: float = 30.0, padding: float = 1.15,
                          dist_override=None):
    """Render the reorient scene from a world-fixed orbit camera.

    The camera is placed on a sphere around the union bounding box of all
    geometries (obj + tables + pillars + hand), at the given azimuth and
    elevation. dist is auto-sized so the bounding sphere just fits.
    """
    rend = get_renderer(width, height)
    rend.scene.clear_geometry()

    frame_verts = []
    for name, (mesh, color, alpha) in meshes.items():
        frame_verts.append(np.asarray(mesh.vertices))
        mat = o3d.visualization.rendering.MaterialRecord()
        if alpha < 0.999:
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*color, alpha] if color is not None else [0.7, 0.7, 0.7, alpha]
        else:
            mat.shader = "defaultLit"
        rend.scene.add_geometry(name, trimesh_to_o3d(mesh, color), mat)

    frame_verts.append(np.asarray(hand_mesh.vertices))
    mat_robot = o3d.visualization.rendering.MaterialRecord(); mat_robot.shader = "defaultLit"
    rend.scene.add_geometry("robot", trimesh_to_o3d(hand_mesh, COLOR_ROBOT), mat_robot)

    pts = np.vstack(frame_verts)
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * 0.5

    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    dir_world = np.array([np.cos(el) * np.cos(az),
                          np.cos(el) * np.sin(az),
                          np.sin(el)])
    # fov_deg is the VERTICAL field-of-view (Open3D convention). For non-square
    # aspect ratios the horizontal FOV differs; use whichever is smaller so the
    # bounding sphere fits both dimensions.
    half_v = np.radians(fov_deg / 2) / padding
    aspect = width / height
    half_h = np.arctan(np.tan(np.radians(fov_deg / 2)) * aspect) / padding
    half_fov = min(half_v, half_h)
    dist = dist_override if dist_override is not None else radius / np.sin(half_fov)
    eye = center + dir_world * dist
    up = np.array([0.0, 0.0, 1.0])
    if abs(dir_world @ up) > 0.95:
        up = np.array([1.0, 0.0, 0.0])

    rend.scene.scene.set_sun_light([0.4, -0.4, -1.0], [1.0, 1.0, 1.0], 60000)
    rend.scene.scene.enable_sun_light(True)
    rend.scene.set_background([1.0, 1.0, 1.0, 1.0])
    rend.setup_camera(fov_deg, center, eye, up)
    return np.asarray(rend.render_to_image())  # RGB


def render_panel_scene(hand: str, obj_name: str, ts_name: str,
                       size, azim_deg: float, elev_deg: float,
                       fov: float, padding: float,
                       dist_override=None):
    if isinstance(size, (tuple, list)):
        scene_w, scene_h = size[0], size[1]
    else:
        scene_w = scene_h = size
    """Build the reorient scene meshes and render externally (panel 3)."""
    ep_dir = EXP_BASE / hand / obj_name / ts_name
    result = json.load(open(ep_dir / "result.json"))
    scene_info = result.get("scene_info") or {}
    cand_dir = reroot(Path(scene_info["source"]))
    i_idx, j_idx = (int(x) for x in scene_info["cell"].split("_"))
    h_m = float(scene_info.get("h_cm", 0)) / 100.0

    scene = gen_reorient_scene(obj_name, i_idx, j_idx, h_m)
    target_T = pose7_to_mat4(scene["scene"]["mesh"]["target"]["pose"])
    wrist_obj = np.load(cand_dir / "wrist_se3.npy")
    world_T_wrist = target_T @ wrist_obj

    obj_mesh = load_obj_mesh(obj_name)
    obj_world = obj_mesh.copy(); obj_world.apply_transform(target_T)
    hand_mesh = build_hand_mesh(hand, cand_dir, world_T_wrist)

    # Shrink the 2x2 m collision tables to a small visualization slab so they
    # don't dominate the framing and we can see the pillars between them.
    # The "supporting face" sign depends on the table:
    #   table_i: object sits ON TOP → supporting face = +z (top) of the slab
    #   table_j: object hangs BELOW it (placement constraint above the
    #            descent volume) → supporting face = local -z (bottom),
    #            i.e. the face pointing toward the pillars.
    TABLE_VIS_XY = 0.25  # m
    TABLE_VIS_Z  = 0.015 # m

    scene_meshes = {"target": (obj_world, None, 1.0)}
    for name, info in scene["scene"]["cuboid"].items():
        pose = pose7_to_mat4(info["pose"])
        if name in ("table_i", "table_j"):
            dims = info["dims"]
            local_normal_sign = +1.0 if name == "table_i" else -1.0
            # Original supporting-face position along the table's local +z axis.
            face_offset_world = pose[:3, :3] @ (
                np.array([0.0, 0.0, local_normal_sign * dims[2] / 2])
            )
            face_center_world = pose[:3, 3] + face_offset_world
            # New thin slab centered so its supporting face lands on
            # face_center_world (same local sign).
            box = trimesh.creation.box(extents=[TABLE_VIS_XY, TABLE_VIS_XY, TABLE_VIS_Z])
            vis_pose = np.eye(4); vis_pose[:3, :3] = pose[:3, :3]
            new_center_world = face_center_world - (
                pose[:3, :3] @ np.array([0.0, 0.0, local_normal_sign * TABLE_VIS_Z / 2])
            )
            vis_pose[:3, 3] = new_center_world
            box.apply_transform(vis_pose)
            color = COLOR_TABLE_I if name == "table_i" else COLOR_TABLE_J
            alpha = 0.9 if name == "table_i" else 0.55
            scene_meshes[name] = (box, color, alpha)
        else:
            box = trimesh.creation.box(extents=info["dims"])
            box.apply_transform(pose)
            if name.startswith("pillar"):
                scene_meshes[name] = (box, COLOR_PILLAR, 0.7)
            else:
                scene_meshes[name] = (box, np.array([0.5, 0.5, 0.5]), 0.8)

    rgb = render_scene_external(
        scene_meshes, hand_mesh, scene_w, scene_h,
        azim_deg=azim_deg, elev_deg=elev_deg,
        fov_deg=fov, padding=padding,
        dist_override=dist_override,
    )
    # Force near-white pixels to pure white (so the print doesn't waste ink).
    near_white = (rgb.min(axis=2) >= 230)
    rgb[near_white] = 255
    if not _NO_SCENE_AUTOCROP:
        # Trim white margins to leave a proportional border on all sides.
        mask = ~near_white
        if mask.any():
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            pad_y = max(6, (y1 - y0) // 6)   # ~16% vertical margin
            pad_x = max(4, (x1 - x0) // 30)  # ~3% horizontal margin
            y0 = max(0, y0 - pad_y); y1 = min(rgb.shape[0], y1 + pad_y)
            x0 = max(0, x0 - pad_x); x1 = min(rgb.shape[1], x1 + pad_x)
            rgb = rgb[y0:y1, x0:x1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _shift_action_crop(crop, dy, dx=0, scale=1.0):
    x1, y1, x2, y2 = crop
    cx = (x1 + x2) / 2 + dx
    cy = (y1 + y2) / 2 + dy
    half_w = (x2 - x1) * scale / 2
    half_h = (y2 - y1) * scale / 2
    return (int(cx - half_w), int(cy - half_h),
            int(cx + half_w), int(cy + half_h))


def grab_frame(videos_dir: Path, serial: str, frame_idx: int,
               cell_w: int, cell_h: int, crop=None) -> np.ndarray:
    """Extract one frame. Optionally crop=(x1, y1, x2, y2) in original-res
    pixels before resize."""
    vid = videos_dir / f"{serial}.avi"
    f = extract_frame(vid, frame_idx)
    if f is None:
        return np.full((cell_h, cell_w, 3), 64, dtype=np.uint8)
    if crop is not None:
        x1, y1, x2, y2 = crop
        f = f[y1:y2, x1:x2]
    return cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)


def collect_panels(hand: str, obj_name: str, ts_name: str,
                   pose_serial: str, action_serial: str,
                   cell_w: int, cell_h: int, scene_size: int,
                   fov: float, azim_deg: float, elev_deg: float,
                   padding: float,
                   start_state: str, goal_state: str,
                   grasp_state: str, place_state: str,
                   grasp_offset: int, place_offset: int,
                   action_crop=None,
                   scene_shift_up: float = 0.0,
                   scene_left_trim: float = 0.0,
                   grasp_crop_override=None,
                   place_crop_override=None,
                   scene_dist=None,
                   scene_target_aspect=None) -> dict:
    """Render all five candidate panels for one trial.

    Returns dict {start, goal, scene, grasp, place} as BGR uint8 arrays.
    """
    ep_dir = EXP_BASE / hand / obj_name / ts_name
    result = json.load(open(ep_dir / "result.json"))
    states = result.get("states") or []
    ts_array = np.load(ep_dir / "raw" / "timestamps" / "timestamp.npy")
    videos_dir = ep_dir / "videos"

    def idx_or(state, fallback):
        if state == "frame0":
            return 0
        i = state_to_frame_idx(states, state, ts_array)
        return fallback if i is None else i

    start_idx = idx_or(start_state, 0)
    goal_idx  = idx_or(goal_state,  len(ts_array) - 1)
    grasp_idx = idx_or(grasp_state, 0)
    place_idx = idx_or(place_state, len(ts_array) - 1)
    grasp_idx = min(len(ts_array) - 1, max(0, grasp_idx + grasp_offset))
    place_idx = min(len(ts_array) - 1, max(0, place_idx + place_offset))

    scene_bgr = render_panel_scene(
        hand, obj_name, ts_name,
        scene_size, azim_deg, elev_deg, fov, padding,
        dist_override=scene_dist,
    )
    # Pad white space at bottom so the content shifts upward by `scene_shift_up`
    # fraction of the current cropped height.
    if scene_shift_up > 0:
        h, w = scene_bgr.shape[:2]
        pad_h = int(h * scene_shift_up)
        bottom_pad = np.full((pad_h, w, 3), 255, dtype=np.uint8)
        scene_bgr = np.concatenate([scene_bgr, bottom_pad], axis=0)
    # Trim a fraction from each horizontal side of the scene before resizing.
    if scene_left_trim > 0:
        h, w = scene_bgr.shape[:2]
        cut = int(w * scene_left_trim)
        scene_bgr = scene_bgr[:, cut:w - cut]
    # Pad scene to a target aspect (w/h) with white margins so multiple trials
    # share the same scene panel shape.
    if scene_target_aspect is not None:
        h, w = scene_bgr.shape[:2]
        cur = w / h
        if cur < scene_target_aspect:
            # too tall — pad sides
            target_w = int(h * scene_target_aspect)
            extra = target_w - w
            left = extra // 2; right = extra - left
            scene_bgr = np.concatenate([
                np.full((h, left, 3), 255, dtype=np.uint8),
                scene_bgr,
                np.full((h, right, 3), 255, dtype=np.uint8),
            ], axis=1)
        elif cur > scene_target_aspect:
            # too wide — pad top/bottom
            target_h = int(w / scene_target_aspect)
            extra = target_h - h
            top = extra // 2; bot = extra - top
            scene_bgr = np.concatenate([
                np.full((top, w, 3), 255, dtype=np.uint8),
                scene_bgr,
                np.full((bot, w, 3), 255, dtype=np.uint8),
            ], axis=0)
    new_w = int(scene_bgr.shape[1] * cell_h / scene_bgr.shape[0])
    scene_bgr = cv2.resize(scene_bgr, (new_w, cell_h), interpolation=cv2.INTER_AREA)

    grasp_crop = action_crop if grasp_crop_override is None else grasp_crop_override
    place_crop = action_crop if place_crop_override is None else place_crop_override
    return {
        "start": grab_frame(videos_dir, pose_serial,   start_idx, cell_w, cell_h),
        "goal":  grab_frame(videos_dir, pose_serial,   goal_idx,  cell_w, cell_h),
        "scene": scene_bgr,
        "grasp": grab_frame(videos_dir, action_serial, grasp_idx, cell_w, cell_h, crop=grasp_crop),
        "place": grab_frame(videos_dir, action_serial, place_idx, cell_w, cell_h, crop=place_crop),
    }


def hcat(panels, gap_px=6):
    H = max(p.shape[0] for p in panels)
    out_w = sum(p.shape[1] for p in panels) + gap_px * (len(panels) - 1)
    out = np.full((H, out_w, 3), 255, dtype=np.uint8)
    x = 0
    for p in panels:
        # Center vertically if shorter than max H
        y = (H - p.shape[0]) // 2
        out[y:y + p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + gap_px
    return out


def vcat(panels, gap_px=6):
    W = max(p.shape[1] for p in panels)
    out_h = sum(p.shape[0] for p in panels) + gap_px * (len(panels) - 1)
    out = np.full((out_h, W, 3), 255, dtype=np.uint8)
    y = 0
    for p in panels:
        x = (W - p.shape[1]) // 2
        out[y:y + p.shape[0], x:x + p.shape[1]] = p
        y += p.shape[0] + gap_px
    return out


def resize_h(im, h):
    if im.shape[0] == h: return im
    new_w = int(im.shape[1] * h / im.shape[0])
    return cv2.resize(im, (new_w, h), interpolation=cv2.INTER_AREA)


def build_row_5col(p, gap_px=6):
    """A: start | goal | scene | grasp | place  (current default)"""
    return hcat([p["start"], p["goal"], p["scene"], p["grasp"], p["place"]],
                gap_px=gap_px)


def build_row_3col_split(p, gap_px=6, sub_gap_px=4):
    """B: scene | (start / goal) | (grasp / place)"""
    H = p["scene"].shape[0]
    sub_h = (H - sub_gap_px) // 2
    col_pose   = vcat([resize_h(p["start"], sub_h), resize_h(p["goal"],  sub_h)], gap_px=sub_gap_px)
    col_action = vcat([resize_h(p["grasp"], sub_h), resize_h(p["place"], sub_h)], gap_px=sub_gap_px)
    return hcat([p["scene"], col_pose, col_action], gap_px=gap_px)


def build_row_3col(p, gap_px=6):
    """C: scene | grasp | place"""
    return hcat([p["scene"], p["grasp"], p["place"]], gap_px=gap_px)


def build_row_2col_split(p, gap_px=6, sub_gap_px=4):
    """D: scene | (grasp / place)"""
    H = p["scene"].shape[0]
    sub_h = (H - sub_gap_px) // 2
    col_action = vcat([resize_h(p["grasp"], sub_h), resize_h(p["place"], sub_h)], gap_px=sub_gap_px)
    return hcat([p["scene"], col_action], gap_px=gap_px)


def _fit_to(im, w, h, bg=255):
    """Resize im into a (w, h) box preserving aspect, pad with bg."""
    ih, iw = im.shape[:2]
    s = min(w / iw, h / ih)
    new_w, new_h = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), bg, dtype=np.uint8)
    y = (h - new_h) // 2
    x = (w - new_w) // 2
    out[y:y + new_h, x:x + new_w] = resized
    return out


def build_row_3col_equal(p, gap_px=6, mid_gap_px=None):
    """E: scene | grasp@pose_i | place@pose_j, all fit into (cell_w, cell_h).
    `mid_gap_px` widens the gap between Current and Target Pose columns so an
    arrow can be drawn there.  Defaults to gap_px when not provided."""
    if mid_gap_px is None:
        mid_gap_px = gap_px
    h, w = p["grasp"].shape[:2]
    scene = _fit_to(p["scene"], w, h)
    left = hcat([scene, p["grasp"]], gap_px=gap_px)
    return hcat([left, p["place"]], gap_px=mid_gap_px)


def _find_font(size: int, bold: bool = True):
    from PIL import ImageFont
    candidates = (
        [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ] if bold else [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    )
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def annotate_3col_header(canvas, cell_w, gap_px, font_size=None, h_band=None,
                         mid_gap_px=None):
    """Append a footer band: column labels under each image column, an arrow
    between Current Pose and Target Pose, and 'Reset' under the arrow. Font
    size auto-selected as the largest where every label width <= width of the
    element above it (column for labels, arrow span for 'Reset')."""
    if mid_gap_px is None:
        mid_gap_px = gap_px

    from PIL import ImageFont
    times_path = None
    for cand in (
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ):
        if Path(cand).exists():
            times_path = cand
            break

    arrow_pad = max(8, mid_gap_px // 8)
    arrow_span = mid_gap_px - 2 * arrow_pad
    edge_pad = 16
    constraints = [
        ("Combined Collision Scene", cell_w - 2 * edge_pad),
        ("Current Pose",             cell_w - 2 * edge_pad),
        ("Target Pose",              cell_w - 2 * edge_pad),
        ("Reset",                    max(40, arrow_span - 2 * edge_pad)),
    ]

    def text_w(size, text):
        return ImageFont.truetype(times_path, size).getlength(text)

    def max_size(text, w_avail, lo=8, hi=600):
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if text_w(mid, text) <= w_avail:
                lo = mid
            else:
                hi = mid - 1
        return lo

    auto_fs = min(max_size(t, w) for t, w in constraints)
    fs = auto_fs if font_size is None else min(font_size, auto_fs)
    if h_band is None:
        h_band = int(fs * 1.9)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.font_manager import FontProperties

    H, W = canvas.shape[:2]
    total_H = H + h_band
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, total_H / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(total_H, 0)
    ax.set_axis_off()
    ax.imshow(canvas[:, :, ::-1], extent=(0, W, H, 0), interpolation="none")

    pt_size = fs * 72 / dpi
    fp = FontProperties(fname=times_path, size=pt_size)

    col_centers = [
        cell_w / 2,
        cell_w + gap_px + cell_w / 2,
        2 * cell_w + gap_px + mid_gap_px + cell_w / 2,
    ]
    labels = ["Combined Collision Scene", "Current Pose", "Target Pose"]
    label_y = H + h_band / 2
    for label, cx in zip(labels, col_centers):
        ax.text(cx, label_y, label, ha="center", va="center", color="black",
                fontproperties=fp, clip_on=False)

    gap_x0 = 2 * cell_w + gap_px
    gap_x1 = gap_x0 + mid_gap_px
    x_left  = gap_x0 + arrow_pad
    x_right = gap_x1 - arrow_pad
    y_arrow = H / 2
    arrow = FancyArrowPatch(
        (x_left, y_arrow), (x_right, y_arrow),
        arrowstyle="-|>",
        mutation_scale=max(20, fs * 0.7),
        color="black",
        linewidth=max(2, fs / 12),
        clip_on=False,
    )
    ax.add_patch(arrow)

    cx_arrow = (x_left + x_right) / 2
    # "Reset" placed directly below the arrow (in the row gap between rows).
    ax.text(cx_arrow, y_arrow + fs * 0.55 + 4, "Reset",
            ha="center", va="top", color="black",
            fontproperties=fp, clip_on=False)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf[:, :, ::-1]


LAYOUTS = {
    "5col":       build_row_5col,
    "3col_split": build_row_3col_split,
    "3col":       build_row_3col,
    "2col_split": build_row_2col_split,
    "3col_equal": build_row_3col_equal,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--trials", nargs="+",
                    default=["donut:20260523_034840", "pepsi:20260526_145001"],
                    help="obj:ts pairs (one per row)")
    ap.add_argument("--pose-serial",   default="25305466",
                    help="camera serial for start/goal pose panels "
                         "(prefer a top-down clean view, no robot home)")
    ap.add_argument("--action-serial", default="25322648",
                    help="camera serial for grasp/place action panels "
                         "(prefer a side view showing the grip clearly)")
    ap.add_argument("--cell-w",     type=int, default=720)
    ap.add_argument("--cell-h",     type=int, default=540)
    ap.add_argument("--scene-size", type=int, default=1200,
                    help="square render size if --scene-h not given")
    ap.add_argument("--scene-h",    type=int, default=None,
                    help="render height (uses --scene-size as width if set)")
    ap.add_argument("--fov",        type=float, default=35.0,
                    help="vertical FOV in degrees")
    ap.add_argument("--azim",       type=float, default=45.0,
                    help="camera azimuth (deg, around +z)")
    ap.add_argument("--elev",       type=float, default=30.0,
                    help="camera elevation (deg, above xy plane)")
    ap.add_argument("--azims", nargs="+", type=float, default=None,
                    help="per-trial azimuths (one per --trials); overrides --azim")
    ap.add_argument("--elevs", nargs="+", type=float, default=None,
                    help="per-trial elevations (one per --trials); overrides --elev")
    ap.add_argument("--padding",    type=float, default=1.05)
    ap.add_argument("--paddings",   nargs="+", type=float, default=None,
                    help="per-trial scene-render padding (>1 = camera further "
                         "back, smaller content in render)")
    ap.add_argument("--scene-dists", nargs="+", type=float, default=None,
                    help="per-trial explicit camera distance (m); overrides "
                         "the auto-distance from bbox-sphere fit. Smaller = "
                         "closer = bigger content")
    ap.add_argument("--scene-target-aspect", type=float, default=None,
                    help="pad scene panel to this width/height aspect with "
                         "white margins so all trials share the same shape")
    ap.add_argument("--no-scene-autocrop", action="store_true",
                    help="skip the post-render auto-crop of white margins "
                         "(use raw render output)")
    ap.add_argument("--start-state", default="init",
                    help="state name for start-pose frame, or 'frame0'")
    ap.add_argument("--goal-state",  default="reset_done")
    ap.add_argument("--grasp-state", default="squeeze_done",
                    help="state for the grasp action panel "
                         "(squeeze_done = fingers fully closed, object held)")
    ap.add_argument("--place-state", default="hand_init",
                    help="state for the place action panel")
    ap.add_argument("--grasp-offset",   type=int, default=0)
    ap.add_argument("--place-offset",   type=int, default=-3,
                    help="frame offset added to place-state; keep <=0 so the "
                         "object is still in hand at the place pose")
    ap.add_argument("--gap-px",     type=int, default=6)
    ap.add_argument("--row-gap-px", type=int, default=10)
    ap.add_argument("--out-prefix", default="reset_row")
    ap.add_argument("--png-only", action="store_true",
                    help="skip PDF, write PNG only")
    ap.add_argument("--horizontal", action="store_true",
                    help="arrange multiple trials side-by-side (horizontal) "
                         "instead of stacked (vertical)")
    ap.add_argument("--scene-side-trim", type=float, default=0.0,
                    help="fraction (0..0.4) to trim from each horizontal "
                         "side of the rendered scene panel")
    ap.add_argument("--scene-shifts-up", nargs="+", type=float, default=None,
                    help="per-trial fraction of bottom padding to add to the "
                         "scene panel (positive = pushes content upward)")
    ap.add_argument("--action-crop", nargs=4, type=int, default=None,
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="crop box (in original-res pixels) for action panels")
    ap.add_argument("--action-crop-y-shifts", nargs="+", type=int, default=None,
                    help="per-trial pixel shifts to add to action-crop y1, y2 "
                         "(positive = move window downward in the image)")
    ap.add_argument("--action-crop-x-shifts", nargs="+", type=int, default=None,
                    help="per-trial pixel shifts to add to action-crop x1, x2 "
                         "(positive = move window rightward in the image, so "
                         "content appears to move leftward in the output)")
    ap.add_argument("--action-crop-scales", nargs="+", type=float, default=None,
                    help="per-trial scale (around the crop center). >1 = "
                         "larger crop window (zoom out, more area captured)")
    # Per-panel overrides (apply to only the grasp or only the place crop).
    ap.add_argument("--grasp-crop-x-shifts", nargs="+", type=int, default=None)
    ap.add_argument("--grasp-crop-y-shifts", nargs="+", type=int, default=None)
    ap.add_argument("--grasp-crop-scales",   nargs="+", type=float, default=None)
    ap.add_argument("--place-crop-x-shifts", nargs="+", type=int, default=None)
    ap.add_argument("--place-crop-y-shifts", nargs="+", type=int, default=None)
    ap.add_argument("--place-crop-scales",   nargs="+", type=float, default=None)
    ap.add_argument("--layouts", nargs="+",
                    default=list(LAYOUTS.keys()),
                    help="which layouts to emit (subset of "
                         + ",".join(LAYOUTS.keys()) + ")")
    ap.add_argument("--annotate-3col", action="store_true",
                    help="for 3col_equal layout: prepend a header band with "
                         "column labels and a Reset arrow between Current "
                         "Pose and Target Pose")
    ap.add_argument("--header-h", type=int, default=None,
                    help="footer band height (px); auto-sized from font if omitted")
    ap.add_argument("--header-font", type=int, default=None,
                    help="upper cap on footer font size; auto-fits if omitted")
    ap.add_argument("--mid-gap-px", type=int, default=None,
                    help="for 3col_equal layout: gap between Current Pose and "
                         "Target Pose columns where the Reset arrow is drawn. "
                         "Defaults to --gap-px.")
    args = ap.parse_args()
    global _NO_SCENE_AUTOCROP
    _NO_SCENE_AUTOCROP = args.no_scene_autocrop

    # Render each trial once, then assemble multiple layouts from the same panels.
    per_trial_panels = []
    for ti, spec in enumerate(args.trials):
        obj, ts = spec.split(":")
        azim = args.azims[ti] if args.azims else args.azim
        elev = args.elevs[ti] if args.elevs else args.elev
        print(f"[panels] {obj} / {ts}  azim={azim} elev={elev}", flush=True)
        scene_size_arg = ((args.scene_size, args.scene_h)
                          if args.scene_h is not None else args.scene_size)
        padding_ti = args.paddings[ti] if args.paddings else args.padding

        def _per_panel_crop(x_shifts, y_shifts):
            if not args.action_crop:
                return None
            # Per-panel x/y shifts REPLACE globals when provided. Scale is
            # always per-trial (must be the same for grasp and place so the
            # two panels stay visually the same size).
            y = (y_shifts[ti] if y_shifts is not None
                 else (args.action_crop_y_shifts[ti]
                       if args.action_crop_y_shifts else 0))
            x = (x_shifts[ti] if x_shifts is not None
                 else (args.action_crop_x_shifts[ti]
                       if args.action_crop_x_shifts else 0))
            s = (args.action_crop_scales[ti] if args.action_crop_scales else 1.0)
            return _shift_action_crop(args.action_crop, y, x, s)

        grasp_crop = _per_panel_crop(
            args.grasp_crop_x_shifts, args.grasp_crop_y_shifts)
        place_crop = _per_panel_crop(
            args.place_crop_x_shifts, args.place_crop_y_shifts)
        per_trial_panels.append(collect_panels(
            args.hand, obj, ts, args.pose_serial, args.action_serial,
            args.cell_w, args.cell_h, scene_size_arg,
            args.fov, azim, elev, padding_ti,
            args.start_state, args.goal_state,
            args.grasp_state, args.place_state,
            args.grasp_offset, args.place_offset,
            scene_left_trim=args.scene_side_trim,
            scene_shift_up=(args.scene_shifts_up[ti]
                            if args.scene_shifts_up else 0.0),
            grasp_crop_override=grasp_crop,
            place_crop_override=place_crop,
            scene_dist=(args.scene_dists[ti] if args.scene_dists else None),
            scene_target_aspect=args.scene_target_aspect,
        ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    mid_gap_px = args.mid_gap_px if args.mid_gap_px is not None else args.gap_px
    for layout_name in args.layouts:
        builder = LAYOUTS[layout_name]
        if layout_name == "3col_equal":
            rows = [builder(p, gap_px=args.gap_px, mid_gap_px=mid_gap_px)
                    for p in per_trial_panels]
        else:
            rows = [builder(p, gap_px=args.gap_px) for p in per_trial_panels]
        if args.horizontal:
            # Side-by-side: pad heights to match, hstack with gap column.
            max_h = max(r.shape[0] for r in rows)
            padded = []
            for r in rows:
                if r.shape[0] < max_h:
                    pad = np.full((max_h - r.shape[0], r.shape[1], 3), 255, dtype=np.uint8)
                    r = np.concatenate([r, pad], axis=0)
                padded.append(r)
            gap_col = np.full((max_h, args.row_gap_px, 3), 255, dtype=np.uint8)
            parts = []
            for i, r in enumerate(padded):
                if i: parts.append(gap_col)
                parts.append(r)
            canvas = np.concatenate(parts, axis=1)
        else:
            max_w = max(r.shape[1] for r in rows)
            padded = []
            for r in rows:
                if r.shape[1] < max_w:
                    pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
                    r = np.concatenate([r, pad], axis=1)
                padded.append(r)
            gap_row = np.full((args.row_gap_px, max_w, 3), 255, dtype=np.uint8)
            parts = []
            for i, r in enumerate(padded):
                if i: parts.append(gap_row)
                parts.append(r)
            canvas = np.concatenate(parts, axis=0)

        if args.annotate_3col and layout_name == "3col_equal":
            canvas = annotate_3col_header(
                canvas, args.cell_w, args.gap_px,
                font_size=args.header_font, h_band=args.header_h,
                mid_gap_px=mid_gap_px,
            )

        png_path = OUT_DIR / f"{args.out_prefix}_{layout_name}.png"
        cv2.imwrite(str(png_path), canvas)
        if not args.png_only:
            pdf_path = OUT_DIR / f"{args.out_prefix}_{layout_name}.pdf"
            Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(
                pdf_path, "PDF", resolution=300.0,
            )
        print(f"-> {png_path}  ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
