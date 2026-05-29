#!/usr/bin/env python3
"""Render side-by-side (scene+grasp | lift-moment overlay grid) for every trial.

Left half: Open3D offscreen render of the candidate scene (wall/shelf/box) with
the planned hand at the executed grasp pose, on the canonical tabletop pose.
Right half: thumb_grid.png from grasp_overlay_thumbnail (24-cam overlay).

Output: <output_dir>/<obj>_<ep>.png per trial.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom

# AutoDex utils
from autodex.utils.conversion import cart2se3
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH
from autodex.utils.path import repo_dir
from paradex.visualization.robot import RobotModule


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
THUMB_BASE = Path.home() / "shared_data" / "AutoDex" / "grasp_overlay_thumbnail"
CAND_BASE = Path.home() / "shared_data" / "AutoDex" / "candidates"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"

HAND_URDF = {
    "allegro": str(Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo"
                   / "content" / "assets" / "robot" / "allegro_description"
                   / "allegro_hand_description_right.urdf"),
    "inspire": str(Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo"
                   / "content" / "assets" / "robot" / "inspire_description"
                   / "inspire_hand_right.urdf"),
}

XARM_URDF = {
    # inspire uses .bak (pre-calibration) per offline convention.
    "allegro": str(Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
                   / "allegro_description" / "xarm_allegro.urdf"),
    "inspire": str(Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
                   / "inspire_description" / "xarm_inspire.urdf.bak"),
}

COLOR_ARM = np.array([180, 180, 180]) / 255.0           # light gray
COLOR_HAND_BASE = np.array([140, 140, 140]) / 255.0     # darker gray
COLOR_FINGER = {
    "thumb":  np.array([255, 140,   0]) / 255.0,   # orange
    "index":  np.array([  0, 200, 255]) / 255.0,   # cyan
    "middle": np.array([  0, 255, 100]) / 255.0,   # lime
    "ring":   np.array([255,   0, 200]) / 255.0,   # magenta
    "pinky":  np.array([255, 220,   0]) / 255.0,   # yellow
}
ARM_LINK_NAMES = {"base.obj", "link1.obj", "link2.obj", "link3.obj",
                  "link4.obj", "link5.obj", "link6.obj"}
HAND_BASE_NAMES = {"base_link.obj", "base_link.STL"}
ALLEGRO_LINK_LABELS = {}
for i in range(4):
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj"] = "index"
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj_1"] = "middle"
    ALLEGRO_LINK_LABELS[f"link_{i}.0.obj_2"] = "ring"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj"] = "index"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj_1"] = "middle"
ALLEGRO_LINK_LABELS["link_3.0_tip.obj_2"] = "ring"
for name in ["link_12.0_right.obj", "link_12.0_left.obj",
             "link_13.0.obj", "link_14.0.obj", "link_15.0.obj", "link_15.0_tip.obj"]:
    ALLEGRO_LINK_LABELS[name] = "thumb"
FINGER_PREFIX_MAP = {
    "right_thumb_": "thumb", "right_index_": "index", "right_middle_": "middle",
    "right_ring_": "ring", "right_little_": "pinky",
    "left_thumb_": "thumb", "left_index_": "index", "left_middle_": "middle",
    "left_ring_": "ring", "left_little_": "pinky",
}


def link_color(link_name):
    if link_name in ARM_LINK_NAMES:
        return COLOR_ARM
    if link_name in HAND_BASE_NAMES:
        return COLOR_HAND_BASE
    if link_name in ALLEGRO_LINK_LABELS:
        return COLOR_FINGER[ALLEGRO_LINK_LABELS[link_name]]
    for prefix, label in FINGER_PREFIX_MAP.items():
        if link_name.startswith(prefix):
            return COLOR_FINGER[label]
    return COLOR_ARM

COLOR_OBJECT = np.array([0.0, 100/255, 0.0])     # dark green
COLOR_OBSTACLE = np.array([119, 136, 153]) / 255.0
COLOR_TABLE = np.array([205, 210, 215]) / 255.0
COLOR_ROBOT = np.array([153, 128, 224]) / 255.0


# ------- scene construction (ported from RSS recorder_*.py) -----------------

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
    return T

def transl(xyz):
    T = np.eye(4); T[:3, 3] = xyz
    return T

def mat4_to_pose(T):
    q = R.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return [T[0, 3], T[1, 3], T[2, 3], q[3], q[0], q[1], q[2]]


def get_tabletop_base(tabletop_pose):
    return {
        "mesh": {"target": {"pose": tabletop_pose}},
        "cuboid": {
            "table": {
                "dims": [66.0, 6.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }
        },
    }


def get_wall_scene(tabletop_pose, obb_info, gap=0.0, z_rotation_deg=0.0):
    angle_rad = np.radians(z_rotation_deg)
    z_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    rotated_pose = z_rotation @ tabletop_pose

    obb_transform = np.array(obb_info['obb_transform'])
    R_obb = obb_transform[:3, :3]
    obb_extents = np.array(obb_info['obb'])

    R_world = rotated_pose[:3, :3]
    t_world = rotated_pose[:3, 3]
    obb_world = R_world @ R_obb

    corners_local = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corners_local.append(np.array([i, j, k]) * obb_extents / 2)
    corners_world = np.array([(obb_world @ c + t_world) for c in corners_local])

    min_y = corners_world[:, 1].min()
    max_z = corners_world[:, 2].max()

    wall_height = max_z + 0.1
    wall_y = min_y - gap

    scene = get_tabletop_base(rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [0.6, 0.02, wall_height],
        "pose": [0.0, wall_y - 0.01, wall_height/2, 1, 0, 0, 0],
    }
    return scene


def get_box_scene(obj_mesh, tabletop_pose, height_offset=0.1):
    verts = obj_mesh.vertices
    R_obj = tabletop_pose[:3, :3]
    t_obj = tabletop_pose[:3, 3]
    verts_w = (R_obj @ verts.T).T + t_obj

    points_xy = verts_w[:, :2]
    poly = geom.MultiPoint(points_xy).convex_hull
    rect = poly.minimum_rotated_rectangle
    rect_pts = np.array(rect.exterior.coords)[:4]

    cx, cy = rect.centroid.coords[0]
    edge = rect_pts[1] - rect_pts[0]
    yaw = np.arctan2(edge[1], edge[0])

    width = np.linalg.norm(rect_pts[1] - rect_pts[0])
    depth = np.linalg.norm(rect_pts[2] - rect_pts[1])

    max_z = verts_w[:, 2].max()
    wall_height = max_z - height_offset
    if wall_height <= 0:
        return None

    THICK = 0.02
    scene = get_tabletop_base(tabletop_pose)

    T_box_center = np.eye(4)
    T_box_center[:3, :3] = rotz(yaw)[:3, :3]
    T_box_center[:3, 3] = [cx, cy, wall_height / 2]

    def add_wall(name, lx, ly, w, d):
        T_local = np.eye(4); T_local[:3, 3] = [lx, ly, 0]
        T_wall = T_box_center @ T_local
        scene["cuboid"][name] = {
            "dims": [w, d, wall_height],
            "pose": mat4_to_pose(T_wall),
        }

    hw, hd = width / 2, depth / 2
    add_wall("box_front", hw + THICK/2, 0, THICK, depth + 2*THICK)
    add_wall("box_back", -hw - THICK/2, 0, THICK, depth + 2*THICK)
    add_wall("box_right", 0, hd + THICK/2, width + 2*THICK, THICK)
    add_wall("box_left", 0, -hd - THICK/2, width + 2*THICK, THICK)
    return scene


def get_shelf_scene(tabletop_pose, obb_info, gap=0.03, z_rotation_deg=0.0,
                    up=True, side=True, back=True):
    angle = np.radians(z_rotation_deg)
    z_rot = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1],
    ])
    pose = z_rot @ tabletop_pose

    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    ext = np.array(obb_info["obb"]) / 2.0
    Rw = pose[:3, :3]; tw = pose[:3, 3]
    axes = Rw @ R_obb

    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(axes @ (np.array([sx, sy, sz]) * ext) + tw)
    corners = np.array(corners)

    THICK = 0.02
    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()

    inner_y_back = min_y - gap
    inner_y_front = max_y + gap
    inner_x_min = min_x - gap
    inner_x_max = max_x + gap
    inner_z_top = max_z + gap
    wall_h = inner_z_top

    scene = get_tabletop_base(pose)
    full_width = (inner_x_max - inner_x_min) + 2 * THICK
    full_depth = (inner_y_front - inner_y_back) + THICK

    if back:
        scene["cuboid"]["back"] = {
            "dims": [full_width, THICK, wall_h],
            "pose": [(inner_x_min + inner_x_max) / 2,
                     inner_y_back - THICK/2, wall_h / 2, 1, 0, 0, 0],
        }
    if side:
        scene["cuboid"]["side_pos"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [inner_x_max + THICK/2,
                     (inner_y_back - THICK/2 + inner_y_front)/2,
                     wall_h / 2, 1, 0, 0, 0],
        }
        scene["cuboid"]["side_neg"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [inner_x_min - THICK/2,
                     (inner_y_back - THICK/2 + inner_y_front)/2,
                     wall_h / 2, 1, 0, 0, 0],
        }
    if up:
        scene["cuboid"]["up"] = {
            "dims": [full_width, full_depth, THICK],
            "pose": [(inner_x_min + inner_x_max) / 2,
                     (inner_y_back - THICK + inner_y_front)/2,
                     inner_z_top + THICK/2, 1, 0, 0, 0],
        }
    return scene


def pose7_to_mat4(p7):
    """[x, y, z, qw, qx, qy, qz] -> 4x4."""
    T = np.eye(4)
    T[:3, 3] = p7[:3]
    qw, qx, qy, qz = p7[3:]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


# ---------------------- rendering -----------------------------------------

def trimesh_to_o3d(mesh, color=None):
    o = o3d.geometry.TriangleMesh()
    o.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o.compute_vertex_normals()
    if color is not None:
        o.paint_uniform_color(color)
    else:
        try:
            cv = mesh.visual.to_color()
            vc = np.array(cv.vertex_colors)[:, :3] / 255.0
            o.vertex_colors = o3d.utility.Vector3dVector(vc)
        except Exception:
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


def compute_camera(combined_mesh, fov_deg, aspect, elev_deg, padding=1.3):
    bs = combined_mesh.bounding_sphere
    center = np.array(bs.primitive.center)
    rad = bs.primitive.radius

    vfov = np.radians(fov_deg)
    hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect)
    eff = min(vfov, hfov) / 2.0
    dist = (rad * padding) / np.sin(eff)
    return center, dist


def render_left(obj_mesh_world, robot_link_meshes, cuboid_meshes_world,
                width, height, fov, elev_deg, azim_deg, padding):
    """robot_link_meshes: list of (trimesh, color_rgb_0to1)."""
    rend = get_renderer(width, height)
    rend.scene.clear_geometry()

    mat_obj = o3d.visualization.rendering.MaterialRecord(); mat_obj.shader = "defaultLit"

    obj_o3d = trimesh_to_o3d(obj_mesh_world, color=COLOR_OBJECT)
    rend.scene.add_geometry("object", obj_o3d, mat_obj)

    for li, (m, color) in enumerate(robot_link_meshes):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        o3d_m = trimesh_to_o3d(m, color=color)
        rend.scene.add_geometry(f"robot_{li}", o3d_m, mat)

    for name, m in cuboid_meshes_world.items():
        if name == "table":
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*COLOR_TABLE, 0.9]
            o3d_m = trimesh_to_o3d(m, color=COLOR_TABLE)
            rend.scene.add_geometry(name, o3d_m, mat)
        else:
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*COLOR_OBSTACLE, 0.35]
            o3d_m = trimesh_to_o3d(m, color=COLOR_OBSTACLE)
            rend.scene.add_geometry(name, o3d_m, mat)

    combined_for_cam = [obj_mesh_world] + [m for m, _ in robot_link_meshes]
    combined = trimesh.util.concatenate(combined_for_cam)
    aspect = width / height
    center, cam_dist = compute_camera(combined, fov, aspect, elev_deg, padding)

    elev_rad = np.radians(elev_deg)
    azim_rad = np.radians(azim_deg)
    horiz = cam_dist * np.cos(elev_rad)
    vert = cam_dist * np.sin(elev_rad)
    eye = np.array([
        center[0] + horiz * np.cos(azim_rad),
        center[1] + horiz * np.sin(azim_rad),
        center[2] + vert,
    ])
    lookat = center.copy()
    up = np.array([0.0, 0.0, 1.0])

    rend.scene.scene.set_sun_light([0.4, -0.4, -1.0], [1.0, 1.0, 1.0], 60000)
    rend.scene.scene.enable_sun_light(True)
    rend.scene.set_background([1.0, 1.0, 1.0, 1.0])
    rend.setup_camera(fov, lookat, eye, up)
    img = np.asarray(rend.render_to_image())
    return img  # RGB uint8


# --------- main per-trial pipeline -----------------------------------------

def load_obj_mesh(obj_name):
    p = OBJ_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    m = trimesh.load(str(p), force="mesh")
    return m


def load_obb_info(obj_name):
    p = OBJ_BASE / obj_name / "processed_data" / "info" / "simplified.json"
    return json.load(open(p))


def load_scene_json(obj_name, scene_type, scene_id):
    p = OBJ_BASE / obj_name / "scene" / scene_type / f"{scene_id}.json"
    return json.load(open(p))


def find_pregrasp_frame_idx(result_json_path, timestamps_path, state="pregrasp"):
    from datetime import datetime
    r = json.load(open(result_json_path))
    target_iso = None
    for s in r["timing"]["execution_states"]:
        if s["state"] == state:
            target_iso = s["time"]; break
    if target_iso is None:
        return None
    target_epoch = datetime.fromisoformat(target_iso).timestamp() + 0.03
    ts = np.load(timestamps_path)
    return int(np.argmin(np.abs(ts - target_epoch)))


def build_full_robot_link_meshes_left(hand_type, arm_qpos, hand_qpos):
    """FK xarm_<hand>.urdf in robot frame. Returns list of (mesh, color)."""
    robot = RobotModule(XARM_URDF[hand_type])
    n_dof = robot.num_joints
    qpos = np.zeros(n_dof, dtype=np.float64)
    qpos[:len(arm_qpos)] = arm_qpos
    qpos[len(arm_qpos):len(arm_qpos)+len(hand_qpos)] = hand_qpos
    cfg = {name: angle for name, angle in zip(robot.joint_names[:len(arm_qpos)+len(hand_qpos)],
                                              qpos[:len(arm_qpos)+len(hand_qpos)])}
    robot.update_cfg(cfg)
    out = []
    for ln in list(robot.scene.geometry.keys()):
        m = robot.scene.geometry[ln].copy()
        T = robot.scene.graph.get(ln)[0]
        m.apply_transform(T)
        out.append((m, link_color(ln)))
    return out


def build_hand_mesh(hand_type, candidate_dir, world_T):
    robot = RobotModule(HAND_URDF[hand_type])
    grasp_pose = np.load(candidate_dir / "grasp_pose.npy").flatten()
    n = min(robot.num_joints, len(grasp_pose))
    cfg = {name: angle for name, angle in zip(robot.joint_names[:n], grasp_pose[:n])}
    robot.update_cfg(cfg)
    mesh = robot.get_robot_mesh(collision_geometry=False)
    mesh.apply_transform(world_T)
    return mesh


def render_trial(hand, obj_name, ep_name, args):
    out_path = Path(args.output_dir) / f"{obj_name}_{ep_name}.png"
    if out_path.exists() and not args.overwrite:
        return "skip"

    ep_dir = EXP_BASE / hand / obj_name / ep_name
    result = json.load(open(ep_dir / "result.json"))
    scene_info = result.get("scene_info")
    if not scene_info or len(scene_info) != 3:
        return "no scene_info"
    scene_type = scene_info[0]
    if scene_type not in ("wall", "shelf", "box"):
        return f"unknown scene_type {scene_type}"

    # Need: pose_world, C2R, arm/hand state, timestamps.
    needed = [ep_dir / p for p in ("pose_world.npy", "C2R.npy",
              "arm/state.npy", "hand/state.npy",
              "raw/timestamps/timestamp.npy")]
    for p in needed:
        if not p.exists():
            return f"missing {p.name}"

    pose_world = np.load(ep_dir / "pose_world.npy")
    c2r = np.load(ep_dir / "C2R.npy")
    if c2r.shape == (3, 4):
        c2r = np.vstack([c2r, [0, 0, 0, 1]])
    pose_robot = np.linalg.inv(c2r) @ pose_world  # object pose in robot frame

    frame_idx = find_pregrasp_frame_idx(
        ep_dir / "result.json", ep_dir / "raw" / "timestamps" / "timestamp.npy")
    if frame_idx is None:
        return "no pregrasp state"
    arm_state = np.load(ep_dir / "arm" / "state.npy")
    hand_state = np.load(ep_dir / "hand" / "state.npy")
    if frame_idx >= len(arm_state) or frame_idx >= len(hand_state):
        return f"frame_idx {frame_idx} oob"
    arm_q = arm_state[frame_idx]
    hand_q = hand_state[frame_idx]

    obj_mesh = load_obj_mesh(obj_name)
    obj_mesh_world = obj_mesh.copy(); obj_mesh_world.apply_transform(pose_robot)

    # Scene built around actual (perceived) object pose, in robot frame.
    if scene_type == "wall":
        obb = load_obb_info(obj_name)
        scene = get_wall_scene(pose_robot, obb, gap=0.0)
    elif scene_type == "shelf":
        obb = load_obb_info(obj_name)
        scene = get_shelf_scene(pose_robot, obb, gap=0.03)
    elif scene_type == "box":
        scene = get_box_scene(obj_mesh, pose_robot, height_offset=0.1)
        if scene is None:
            return "box scene degenerate"

    cuboid_meshes = {}
    for name, info in scene["cuboid"].items():
        box = trimesh.creation.box(extents=info["dims"])
        pose = pose7_to_mat4(info["pose"])
        box.apply_transform(pose)
        cuboid_meshes[name] = box

    robot_link_meshes = build_full_robot_link_meshes_left(hand, arm_q, hand_q)

    left = render_left(
        obj_mesh_world, robot_link_meshes, cuboid_meshes,
        args.width, args.height, args.fov, args.elev, args.azim, args.padding,
    )  # RGB

    # Right: thumb_grid.png
    grid_path = THUMB_BASE / hand / obj_name / ep_name / "thumb_grid.png"
    if grid_path.exists():
        right = cv2.imread(str(grid_path))
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        # Resize right to same height as left, preserve aspect
        h_left = left.shape[0]
        w_right = int(right.shape[1] * h_left / right.shape[0])
        right_resized = cv2.resize(right, (w_right, h_left), interpolation=cv2.INTER_AREA)
        canvas = np.concatenate([left, right_resized], axis=1)
    else:
        canvas = left  # left only; thumb not ready

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return "ok" if grid_path.exists() else "ok (no thumb)"


def discover_trials(hand, obj_filter=None):
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
            if not (ep / "result.json").exists():
                continue
            out.append((obj, ep.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", nargs="+", default=["allegro", "inspire"],
                    choices=["allegro", "inspire"])
    ap.add_argument("--obj", nargs="+", default=None)
    ap.add_argument("--ep", nargs="+", default=None)
    ap.add_argument("--output-dir", default=str(Path.home() / "CORL_2026_latex"
                    / "figures" / "grasp_examples"))
    ap.add_argument("--width", type=int, default=1536)
    ap.add_argument("--height", type=int, default=1536)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-120.0)
    ap.add_argument("--padding", type=float, default=1.2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max", type=int, default=None, help="limit trials per hand")
    args = ap.parse_args()

    for hand in args.hand:
        out_sub = Path(args.output_dir) / hand
        out_sub.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(out_sub) if False else args.output_dir  # keep as-is
        trials = discover_trials(hand, set(args.obj) if args.obj else None)
        if args.ep:
            trials = [(o, e) for o, e in trials if e in set(args.ep)]
        if args.max:
            trials = trials[:args.max]
        print(f"[{hand}] {len(trials)} trials")
        for i, (obj, ep) in enumerate(trials):
            tag = f"[{hand} {i+1}/{len(trials)}]"
            # save under hand-subdir
            cur_out = Path(args.output_dir) / hand
            cur_out.mkdir(parents=True, exist_ok=True)
            saved_args = argparse.Namespace(**vars(args))
            saved_args.output_dir = str(cur_out)
            try:
                status = render_trial(hand, obj, ep, saved_args)
            except Exception as e:
                status = f"ERR {e}"
            print(f"{tag} {obj}/{ep} -> {status}", flush=True)


if __name__ == "__main__":
    main()
