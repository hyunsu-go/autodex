"""Cylinder axis viewer.

Given a standing (tabletop) pose, the cylinder-axis hypothesis is the world
vertical line through the mesh's bbox center after applying the pose
(gravity = -world_z, so axis direction = +z).

The viewer:
  1. Applies a tabletop pose so the object stands upright.
  2. Draws the candidate axis through the bbox center.
  3. Lets the user rotate the mesh around that axis and overlays the rotated
     copy on the original — if cylindrically symmetric, they should coincide.
  4. Reports a symmetry residual: mean / max nearest-surface distance from the
     rotated point cloud to the original mesh, normalised by the bbox diagonal.
     A sweep button averages this over a full revolution.
"""

import argparse
import os
import glob
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as autodex_obj_path


COLOR_AXIS = (1.0, 0.85, 0.10)
COLOR_BBOX_CENTER = (1.0, 0.30, 0.30)
COLOR_CONTACT_A = (0.20, 0.85, 0.20)
COLOR_CONTACT_B = (0.20, 0.55, 1.00)
LINE_WIDTH_AXIS = 5.0

N_SAMPLE_POINTS = 4000      # query samples (rotated copy)
N_REF_POINTS = 20000        # reference KDTree (original surface)


parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", default=autodex_obj_path)
cli_args = parser.parse_args()


def list_objects(root):
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, "processed_data", "info", "tabletop"))
        and os.path.exists(os.path.join(root, d, "raw_mesh", f"{d}.obj"))
    ])


def list_poses(root, obj_name):
    pose_dir = os.path.join(root, obj_name, "processed_data", "info", "tabletop")
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(pose_dir, "*.npy"))])


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def apply_pose(mesh, T):
    m = mesh.copy()
    m.apply_transform(T)
    return m


AXIS_DIRS_WORLD = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}
AXIS_DIRS_OBJ = {
    "obj X": np.array([1.0, 0.0, 0.0]),
    "obj Y": np.array([0.0, 1.0, 0.0]),
    "obj Z": np.array([0.0, 0.0, 1.0]),
}


def resolve_axis_direction(name, pose_T):
    """Return the axis direction in the standing-pose world frame."""
    if name in AXIS_DIRS_WORLD:
        return AXIS_DIRS_WORLD[name]
    return pose_T[:3, :3] @ AXIS_DIRS_OBJ[name]


def make_axis_line(center, half_height, direction):
    d = direction / np.linalg.norm(direction)
    p0 = center - d * half_height
    p1 = center + d * half_height
    return np.stack([p0, p1], axis=0)


def rotation_about_axis(angle_rad, center, direction):
    """4x4 rotation by `angle_rad` about the line through `center` with
    unit `direction` (Rodrigues' formula)."""
    d = direction / np.linalg.norm(direction)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array([
        [0.0, -d[2],  d[1]],
        [d[2],  0.0, -d[0]],
        [-d[1], d[0],  0.0],
    ])
    R = np.eye(3) * c + s * K + (1.0 - c) * np.outer(d, d)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    return T


def symmetry_residual(ref_tree, query_points, center, direction, angle_rad):
    """Mean / max nearest-neighbour distance after rotating `query_points`
    about the line (center, direction) by `angle_rad`.

    `ref_tree` is a cKDTree built on dense surface samples of the original
    (un-rotated) mesh — this avoids `trimesh.proximity.closest_point`, which
    crashes on some rtree builds. Distances are in metres.
    """
    T = rotation_about_axis(angle_rad, center, direction)
    rotated = (T[:3, :3] @ query_points.T).T + T[:3, 3]
    dist, _ = ref_tree.query(rotated, k=1, workers=-1)
    return float(dist.mean()), float(dist.max())


def contact_object_frame(mesh, T):
    """Object-frame coordinates of the mesh vertex that sits lowest in world
    after applying `T` (the table contact when the object rests in pose `T`)."""
    R = T[:3, :3]; t = T[:3, 3]
    v = np.asarray(mesh.vertices)
    world_z = v @ R[2, :] + t[2]
    return v[int(np.argmin(world_z))]


# ── State ─────────────────────────────────────────────────────────────────────
state = {
    "obj_name": None,
    "pose": None,             # standing pose filename (pose 000-ish)
    "mesh_raw": None,         # mesh in object frame, untransformed
    "pose_T": None,           # 4x4 standing pose
    "mesh_world": None,       # mesh after standing pose applied
    "bbox_center": None,      # bbox center of mesh_world
    "bbox_diag": None,
    "query_pts": None,        # surface samples of mesh_world (rotated each time)
    "ref_tree": None,         # cKDTree on dense samples of mesh_world
    "axis_center": None,      # actual 3D position the axis passes through
    "axis_mode": "bbox center",
    "axis_dir_name": "Z",
    "rotated_name": "rotated",
    "axis_handle": None,
    "center_handle": None,
    "contact_handles": [],
}


vis = ViserViewer()
vis.add_floor(0.0)


def clear_scene():
    for name in list(vis.obj_dict.keys()):
        try:
            vis.obj_dict[name]["frame"].remove()
        except Exception:
            pass
    vis.obj_dict.clear()
    vis.frame_nodes.clear()
    for key in ("axis_handle", "center_handle"):
        h = state[key]
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
            state[key] = None
    for h in state["contact_handles"]:
        try:
            h.remove()
        except Exception:
            pass
    state["contact_handles"] = []


def draw_axis():
    if state["bbox_center"] is None:
        return
    if state["axis_handle"] is not None:
        try: state["axis_handle"].remove()
        except Exception: pass
    if state["center_handle"] is not None:
        try: state["center_handle"].remove()
        except Exception: pass

    center = state["axis_center"]
    direction = resolve_axis_direction(state["axis_dir_name"], state["pose_T"])
    half = float(state["bbox_diag"]) * 0.6
    positions = make_axis_line(center, half, direction)
    state["axis_handle"] = vis.server.scene.add_spline_catmull_rom(
        name="/cylinder_axis",
        positions=positions,
        color=COLOR_AXIS,
        line_width=LINE_WIDTH_AXIS,
    )
    state["center_handle"] = vis.server.scene.add_icosphere(
        name="/cylinder_axis_center",
        radius=float(state["bbox_diag"]) * 0.015,
        color=COLOR_BBOX_CENTER,
        position=center,
    )


def draw_contact_markers(p_a_world, p_b_world):
    for h in state["contact_handles"]:
        try: h.remove()
        except Exception: pass
    state["contact_handles"] = []
    r = float(state["bbox_diag"]) * 0.02
    state["contact_handles"].append(vis.server.scene.add_icosphere(
        name="/contact_A", radius=r, color=COLOR_CONTACT_A, position=p_a_world,
    ))
    state["contact_handles"].append(vis.server.scene.add_icosphere(
        name="/contact_B", radius=r, color=COLOR_CONTACT_B, position=p_b_world,
    ))


def compute_axis_center():
    """Resolve `state['axis_center']` from the current axis mode."""
    if state["axis_mode"] == "world origin":
        state["axis_center"] = np.zeros(3)
        for h in state["contact_handles"]:
            try: h.remove()
            except Exception: pass
        state["contact_handles"] = []
        return
    if state["axis_mode"] == "object origin":
        # object frame origin lands at the standing pose translation
        state["axis_center"] = state["pose_T"][:3, 3].copy()
        for h in state["contact_handles"]:
            try: h.remove()
            except Exception: pass
        state["contact_handles"] = []
        return
    if state["axis_mode"] == "pose midpoint":
        a_name = pose_a_dropdown.value
        b_name = pose_b_dropdown.value
        if not a_name or not b_name or a_name == "<none>" or b_name == "<none>":
            state["axis_center"] = state["bbox_center"].copy()
            return
        T_a = np.load(os.path.join(
            cli_args.obj_path, state["obj_name"],
            "processed_data", "info", "tabletop", a_name,
        ))
        T_b = np.load(os.path.join(
            cli_args.obj_path, state["obj_name"],
            "processed_data", "info", "tabletop", b_name,
        ))
        # contact point on the mesh in object frame, for each pose
        p_a_obj = contact_object_frame(state["mesh_raw"], T_a)
        p_b_obj = contact_object_frame(state["mesh_raw"], T_b)
        # transform into the standing-pose world frame
        R0 = state["pose_T"][:3, :3]
        t0 = state["pose_T"][:3, 3]
        p_a_world = R0 @ p_a_obj + t0
        p_b_world = R0 @ p_b_obj + t0
        # axis (x, y) = midpoint; keep z at bbox-center z so the vertical axis
        # is anchored at the object's mid-height for drawing.
        mid_xy = (p_a_world[:2] + p_b_world[:2]) / 2.0
        center = np.array([mid_xy[0], mid_xy[1], state["bbox_center"][2]])
        state["axis_center"] = center
        draw_contact_markers(p_a_world, p_b_world)
    else:
        state["axis_center"] = state["bbox_center"].copy()
        # clear contact markers
        for h in state["contact_handles"]:
            try: h.remove()
            except Exception: pass
        state["contact_handles"] = []


def load_pose(obj_name, pose_name):
    clear_scene()

    raw_path = os.path.join(cli_args.obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    pose_path = os.path.join(cli_args.obj_path, obj_name, "processed_data", "info", "tabletop", pose_name)

    mesh = load_mesh(raw_path)
    T = np.load(pose_path)
    mesh_w = apply_pose(mesh, T)

    v = np.asarray(mesh_w.vertices)
    bbmin = v.min(axis=0)
    bbmax = v.max(axis=0)
    center = (bbmin + bbmax) / 2.0
    diag = float(np.linalg.norm(bbmax - bbmin))

    query_pts, _ = trimesh.sample.sample_surface(mesh_w, N_SAMPLE_POINTS)
    ref_pts, _ = trimesh.sample.sample_surface(mesh_w, N_REF_POINTS)
    ref_tree = cKDTree(np.asarray(ref_pts))

    state.update({
        "obj_name": obj_name,
        "pose": pose_name,
        "mesh_raw": mesh,
        "pose_T": T,
        "mesh_world": mesh_w,
        "bbox_center": center,
        "bbox_diag": diag,
        "query_pts": np.asarray(query_pts),
        "ref_tree": ref_tree,
    })

    vis.add_object(f"{obj_name}_orig", mesh_w, obj_T=np.eye(4))
    rotated_mesh = mesh_w.copy()
    rotated_mesh.visual.face_colors = [200, 80, 80, 140]
    vis.add_object(state["rotated_name"], rotated_mesh, obj_T=np.eye(4))

    # repopulate pose-pair dropdowns with poses available for this object
    pose_options = list_poses(cli_args.obj_path, obj_name)
    if pose_options:
        opts = tuple(pose_options)
        pose_a_dropdown.options = opts
        pose_b_dropdown.options = opts
        default_a = "004.npy" if "004.npy" in pose_options else pose_options[0]
        default_b = "016.npy" if "016.npy" in pose_options else (pose_options[-1] if len(pose_options) > 1 else pose_options[0])
        pose_a_dropdown.value = default_a
        pose_b_dropdown.value = default_b

    compute_axis_center()
    draw_axis()
    info_text.value = (
        f"obj: {obj_name}\n"
        f"standing pose: {pose_name}\n"
        f"bbox size: {(bbmax - bbmin).round(4).tolist()} m\n"
        f"bbox center: {center.round(4).tolist()} m\n"
        f"axis center: {state['axis_center'].round(4).tolist()} m\n"
        f"diag: {diag:.4f} m"
    )
    update_rotation()


def update_rotation():
    if state["mesh_world"] is None:
        return
    angle = float(angle_slider.value) * np.pi / 180.0
    center = state["axis_center"]
    direction = resolve_axis_direction(state["axis_dir_name"], state["pose_T"])
    T = rotation_about_axis(angle, center, direction)
    rec = vis.obj_dict.get(state["rotated_name"])
    if rec is not None:
        from scipy.spatial.transform import Rotation as R
        rec["frame"].position = T[:3, 3]
        rec["frame"].wxyz = R.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mean_d, max_d = symmetry_residual(
        state["ref_tree"], state["query_pts"], center, direction, angle
    )
    diag = state["bbox_diag"]
    residual_text.value = (
        f"angle = {angle_slider.value:.1f}°\n"
        f"mean dist  = {mean_d * 1000:.2f} mm  ({mean_d / diag * 100:.2f} % of diag)\n"
        f"max  dist  = {max_d * 1000:.2f} mm  ({max_d / diag * 100:.2f} % of diag)"
    )


def sweep_symmetry():
    if state["mesh_world"] is None:
        return
    angles = np.linspace(0.0, 360.0, 25)[1:-1]  # skip 0/360
    means, maxes = [], []
    direction = resolve_axis_direction(state["axis_dir_name"], state["pose_T"])
    for a_deg in angles:
        m, mx = symmetry_residual(
            state["ref_tree"], state["query_pts"],
            state["axis_center"], direction, a_deg * np.pi / 180.0,
        )
        means.append(m); maxes.append(mx)
    means = np.array(means); maxes = np.array(maxes)
    diag = state["bbox_diag"]
    sweep_text.value = (
        f"Sweep over {len(angles)} angles (excl. 0/360):\n"
        f"  mean(mean dist) = {means.mean() * 1000:.2f} mm  ({means.mean() / diag * 100:.2f} %)\n"
        f"  max (mean dist) = {means.max()  * 1000:.2f} mm  ({means.max()  / diag * 100:.2f} %)\n"
        f"  max (max  dist) = {maxes.max()  * 1000:.2f} mm  ({maxes.max()  / diag * 100:.2f} %)"
    )


# ── GUI ──────────────────────────────────────────────────────────────────────
available_objects = list_objects(cli_args.obj_path)
if not available_objects:
    raise SystemExit(f"No objects with tabletop poses found under {cli_args.obj_path}")

with vis.server.gui.add_folder("Selection"):
    obj_dropdown = vis.server.gui.add_dropdown(
        "Object",
        options=tuple(available_objects),
        initial_value=available_objects[0],
    )
    init_poses = list_poses(cli_args.obj_path, available_objects[0])
    pose_dropdown = vis.server.gui.add_dropdown(
        "Standing pose",
        options=tuple(init_poses) if init_poses else ("<none>",),
        initial_value=init_poses[0] if init_poses else "<none>",
    )
    load_btn = vis.server.gui.add_button("Load")

with vis.server.gui.add_folder("Axis direction"):
    axis_dir_dropdown = vis.server.gui.add_dropdown(
        "Direction",
        options=("Z", "X", "Y", "obj Z", "obj X", "obj Y"),
        initial_value="Z",
    )

with vis.server.gui.add_folder("Axis center"):
    axis_mode_dropdown = vis.server.gui.add_dropdown(
        "Mode",
        options=("bbox center", "pose midpoint", "world origin", "object origin"),
        initial_value="bbox center",
    )
    pose_a_dropdown = vis.server.gui.add_dropdown(
        "Pose A",
        options=tuple(init_poses) if init_poses else ("<none>",),
        initial_value=(
            "004.npy" if init_poses and "004.npy" in init_poses
            else (init_poses[0] if init_poses else "<none>")
        ),
    )
    pose_b_dropdown = vis.server.gui.add_dropdown(
        "Pose B",
        options=tuple(init_poses) if init_poses else ("<none>",),
        initial_value=(
            "016.npy" if init_poses and "016.npy" in init_poses
            else (init_poses[-1] if init_poses else "<none>")
        ),
    )

with vis.server.gui.add_folder("Rotation"):
    angle_slider = vis.server.gui.add_slider(
        "Angle (deg)", min=0.0, max=360.0, step=1.0, initial_value=0.0
    )
    sweep_btn = vis.server.gui.add_button("Sweep symmetry residual")

with vis.server.gui.add_folder("Info"):
    info_text = vis.server.gui.add_text("Object", initial_value="(load an object)")
    residual_text = vis.server.gui.add_text("Residual", initial_value="(rotate to evaluate)")
    sweep_text = vis.server.gui.add_text("Sweep", initial_value="(press Sweep)")


@obj_dropdown.on_update
def _(_):
    poses = list_poses(cli_args.obj_path, obj_dropdown.value)
    pose_dropdown.options = tuple(poses) if poses else ("<none>",)
    if poses:
        pose_dropdown.value = poses[0]


@load_btn.on_click
def _(_):
    if pose_dropdown.value and pose_dropdown.value != "<none>":
        load_pose(obj_dropdown.value, pose_dropdown.value)


@angle_slider.on_update
def _(_):
    update_rotation()


@sweep_btn.on_click
def _(_):
    sweep_symmetry()


def _axis_changed():
    if state["mesh_world"] is None:
        return
    state["axis_mode"] = axis_mode_dropdown.value
    state["axis_dir_name"] = axis_dir_dropdown.value
    compute_axis_center()
    draw_axis()
    if state["axis_center"] is not None:
        info_text.value = (
            f"obj: {state['obj_name']}\n"
            f"standing pose: {state['pose']}\n"
            f"axis dir: {state['axis_dir_name']}\n"
            f"axis mode: {state['axis_mode']}\n"
            f"axis center: {state['axis_center'].round(4).tolist()} m\n"
            f"diag: {state['bbox_diag']:.4f} m"
        )
    update_rotation()


@axis_dir_dropdown.on_update
def _(_): _axis_changed()


@axis_mode_dropdown.on_update
def _(_): _axis_changed()


@pose_a_dropdown.on_update
def _(_): _axis_changed()


@pose_b_dropdown.on_update
def _(_): _axis_changed()


if init_poses:
    load_pose(available_objects[0], init_poses[0])

vis.start_viewer()
