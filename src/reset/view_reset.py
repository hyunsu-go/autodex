"""Per-grasp reset trajectory viewer.

For each grasp, sweep the wrist 10cm along 12 directions and report:
- collision with table/object during the sweep (per-sphere)
- escape from object OBB expanded by 2cm at sweep end

Pick (object, tabletop pose, grasp idx, direction) and slide sweep t.

Usage:
    python src/reset/view_reset.py
"""

import os
import json
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH, load_candidate
from autodex.utils.conversion import cart2se3
from autodex.planner.planner import GraspPlanner

obj_path = DEFAULT_OBJ_PATH  # rebound from CLI in __main__

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

HAND_URDFS = {
    "allegro": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "allegro_description",
        "allegro_hand_description_right.urdf",
    ),
    "inspire": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_description",
        "inspire_hand_right.urdf",
    ),
    "inspire_left": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_description",
        "inspire_hand_left.urdf",
    ),
    "inspire_f1": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_f1_description",
        "inspire_f1_hand_right.urdf",
    ),
}

WORLD_AXES = np.array([
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
], dtype=np.float64)

DIR_NAMES = [
    "world +x", "world -x", "world +y", "world -y", "world +z", "world -z",
    "wrist +x", "wrist -x", "wrist +y", "wrist -y", "wrist +z", "wrist -z",
]

SWEEP_DIST = 0.10
SWEEP_STEPS_VIZ = 11           # cached sample count
SWEEP_STEP = SWEEP_DIST / (SWEEP_STEPS_VIZ - 1)  # 0.01 m
BBOX_EXPAND = 0.02             # expand object OBB by this much (m) for escape check

CANDIDATES_ROOT = os.path.join(REPO_ROOT, "candidates")


def list_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def load_tabletop_pose(obj_name, pose_idx, x_offset=0.0, z_rotation_rad=0.0):
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    obj_pose = np.load(os.path.join(pose_dir, f"{pose_idx}.npy"))
    obj_pose[0, 3] += x_offset
    if z_rotation_rad != 0.0:
        c, s = np.cos(z_rotation_rad), np.sin(z_rotation_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]
    return obj_pose


def list_tabletop_poses(obj_name):
    d = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(d):
        return []
    return sorted(f.replace(".npy", "") for f in os.listdir(d) if f.endswith(".npy"))


def load_scene_pose(obj_name, scene_type, scene_id):
    """Load native tabletop pose for a grasp from its scene JSON."""
    p = os.path.join(obj_path, obj_name, "scene", scene_type, f"{scene_id}.json")
    with open(p) as f:
        j = json.load(f)
    pose7 = j["scene"]["mesh"]["target"]["pose"]
    return cart2se3(np.array(pose7))


def apply_xz_perturbation(pose, x_offset, z_rad):
    """Add x_offset to translation and rotate around z (in-place safe — returns new)."""
    out = pose.copy()
    out[0, 3] += x_offset
    if z_rad != 0.0:
        c, s = np.cos(z_rad), np.sin(z_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[:3, :3] = Rz @ out[:3, :3]
    return out


def load_obb(obj_name):
    """Returns (obb_transform 4x4 in object frame, half_extent (3,))."""
    p = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    with open(p) as f:
        info = json.load(f)
    return np.array(info["obb_transform"]), np.array(info["obb"]) / 2.0


def bbox_world_frame(obj_pose, obb_tf):
    """Combine obj_pose and obb_tf into bbox-in-world SE3."""
    return obj_pose @ obb_tf


def points_inside_bbox(points_world, bbox_world, half_ext_expanded):
    """
    points_world: (..., 3)
    bbox_world:   (4, 4) SE3
    half_ext_expanded: (3,) per-axis half extent (already expanded)
    Returns (..., ) bool: True if inside bbox.
    """
    R = bbox_world[:3, :3]
    t = bbox_world[:3, 3]
    # Express points in bbox local frame
    rel = points_world - t  # (..., 3)
    local = rel @ R         # equivalent to R.T @ rel.T for the last axis
    return np.all(np.abs(local) <= half_ext_expanded, axis=-1)


class ResetViewer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.gui_playing.value = False

        self._candidates = None  # (wrist_obj, pregrasp, grasp, scene_info)
        self._table_handle = None
        self._object_handle = None
        self._line_handle = None
        self._sphere_handles = []
        self._bbox_handle = None
        self._planners = {}  # hand_name -> GraspPlanner (lazy)
        self._cache = None    # single cache: per-grasp native pose + sweep results
        self._matches = []    # (M, 2) of (grasp_idx, dir_idx) under current filter
        self._suspend_refresh = False  # while we drive sliders programmatically

        hands = [h for h in HAND_URDFS if os.path.isdir(os.path.join(CANDIDATES_ROOT, h))]
        initial_hand = "inspire_left" if "inspire_left" in hands else (hands[0] if hands else "")

        with self.server.gui.add_folder("Retreat Viewer"):
            self.hand_sel = self.server.gui.add_dropdown(
                "Hand", options=hands, initial_value=initial_hand,
            )
            self.version_sel = self.server.gui.add_dropdown(
                "Version", options=[], initial_value="",
            )
            self.obj_sel = self.server.gui.add_dropdown(
                "Object", options=[], initial_value="",
            )
            self.filter_sel = self.server.gui.add_dropdown(
                "Filter", options=[
                    "all",
                    "safe (no-coll + escape)",
                    "no-coll + no-escape",
                    "coll + escape",
                    "coll + no-escape",
                    "dead grasp (no safe dir)",
                ],
                initial_value="all",
            )
            self.match_idx = self.server.gui.add_slider(
                "Match #", min=0, max=1, step=1, initial_value=0, disabled=True,
            )
            self.scene_info_text = self.server.gui.add_text(
                "Scene (auto)", initial_value="-", disabled=True,
            )
            self.x_offset = self.server.gui.add_slider(
                "x_offset", min=0.0, max=0.6, step=0.01, initial_value=0.0,
            )
            self.z_rot = self.server.gui.add_slider(
                "z_rotation (deg)", min=0.0, max=360.0, step=5.0, initial_value=0.0,
            )
            self.grasp_idx = self.server.gui.add_slider(
                "Grasp Idx", min=0, max=1, step=1, initial_value=0,
            )
            self.dir_sel = self.server.gui.add_dropdown(
                "Direction", options=DIR_NAMES, initial_value=DIR_NAMES[0],
            )
            self.sweep_t = self.server.gui.add_slider(
                "Sweep t (m)", min=0.0, max=SWEEP_DIST, step=SWEEP_STEP, initial_value=0.0,
            )
            self.show_spheres = self.server.gui.add_checkbox(
                "Show Collision Spheres", initial_value=True,
            )
            self.show_bbox = self.server.gui.add_checkbox(
                "Show Object Bbox+2cm", initial_value=True,
            )
            self.coll_info = self.server.gui.add_text(
                "Status", initial_value="-", disabled=True,
            )

        for hand_name, urdf in HAND_URDFS.items():
            if os.path.exists(urdf):
                self.add_robot(hand_name, urdf)
                self.robot_dict[hand_name].set_visibility(False)

        # Wire callbacks
        self.hand_sel.on_update(lambda e: self._on_hand_change())
        self.version_sel.on_update(lambda e: self._on_version_change())
        self.obj_sel.on_update(lambda e: self._on_obj_change())
        self.x_offset.on_update(lambda e: self._refresh())
        self.z_rot.on_update(lambda e: self._refresh())
        self.grasp_idx.on_update(lambda e: self._on_grasp_change())
        self.dir_sel.on_update(lambda e: self._refresh())
        self.sweep_t.on_update(lambda e: self._refresh())
        self.show_spheres.on_update(lambda e: self._refresh())
        self.show_bbox.on_update(lambda e: self._refresh())
        self.filter_sel.on_update(lambda e: self._rebuild_matches())
        self.match_idx.on_update(lambda e: self._jump_to_match())

        self._draw_table()
        self._on_hand_change()

    def _draw_table(self):
        box = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
        self._table_handle = self.server.scene.add_mesh_simple(
            name="/objects/table",
            vertices=np.array(box.vertices, dtype=np.float32),
            faces=np.array(box.faces, dtype=np.uint32),
            color=(240, 240, 245),
            opacity=0.85,
            flat_shading=True,
            side="double",
            position=(0.0, 0.0, -0.1),
        )

    def _on_hand_change(self):
        self._candidates = None
        hand = self.hand_sel.value
        versions = list_dirs(os.path.join(CANDIDATES_ROOT, hand)) if hand else []
        self.version_sel.options = versions
        if versions:
            self.version_sel.value = versions[0]
        self._on_version_change()

    def _on_version_change(self):
        self._candidates = None
        hand, ver = self.hand_sel.value, self.version_sel.value
        objs = list_dirs(os.path.join(CANDIDATES_ROOT, hand, ver)) if (hand and ver) else []
        objs = [o for o in objs if list_tabletop_poses(o)]
        self.obj_sel.options = objs
        if objs:
            self.obj_sel.value = objs[0]
        self._on_obj_change()

    def _on_obj_change(self):
        # Invalidate first so any intermediate callbacks from GUI assignments
        # below see stale=None and early-return.
        self._candidates = None

        obj = self.obj_sel.value
        if not obj:
            return

        wrist_obj, pregrasp, grasp, scene_info = load_candidate(
            obj, np.eye(4), self.version_sel.value,
            shuffle=False, skip_done=False, hand=self.hand_sel.value,
        )
        N = len(wrist_obj)

        self.grasp_idx.max = max(N - 1, 0)
        if self.grasp_idx.value >= N:
            self.grasp_idx.value = 0

        # Commit AFTER GUI updates so the final _refresh sees consistent state.
        self._candidates = (wrist_obj, pregrasp, grasp, scene_info)
        self._refresh()

    def _show_object(self, obj_pose):
        if self._object_handle is not None:
            self._object_handle.remove()
            self._object_handle = None
        obj = self.obj_sel.value
        mesh_path = os.path.join(obj_path, obj, "processed_data", "mesh", "simplified.obj")
        if not os.path.exists(mesh_path):
            return
        mesh = trimesh.load(mesh_path, force="mesh")
        self._object_handle = self.server.scene.add_mesh_simple(
            name="/objects/target",
            vertices=np.array(mesh.vertices, dtype=np.float32),
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(180, 180, 200),
            opacity=1.0,
            flat_shading=False,
            wxyz=Rot.from_matrix(obj_pose[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=obj_pose[:3, 3],
        )

    def _get_planner(self, hand):
        if hand not in self._planners:
            self._planners[hand] = GraspPlanner(hand=hand)
        return self._planners[hand]

    def _cache_signature(self):
        # Pose excluded — we cache all poses under one signature.
        return (
            self.hand_sel.value, self.version_sel.value, self.obj_sel.value,
            round(float(self.x_offset.value), 4),
            round(float(self.z_rot.value), 2),
        )

    def _on_grasp_change(self):
        if self._suspend_refresh:
            return
        if self._cache is not None:
            self._suspend_refresh = True
            self._rebuild_dir_options(int(self.grasp_idx.value))
            self._suspend_refresh = False
        self._refresh()

    def _rebuild_dir_options(self, grasp_i):
        """Filter dir_sel.options to dirs matching current filter for grasp_i."""
        cache = self._cache
        if cache is None:
            return
        cf = cache["dir_collision_free"][grasp_i]
        es = cache["dir_escape"][grasp_i]
        safe = cf & es
        flt = self.filter_sel.value
        if flt == "safe (no-coll + escape)":
            mask = safe
        elif flt == "no-coll + no-escape":
            mask = cf & ~es
        elif flt == "coll + escape":
            mask = ~cf & es
        elif flt == "coll + no-escape":
            mask = ~cf & ~es
        else:
            mask = np.ones_like(cf, dtype=bool)
        dirs_filtered = [DIR_NAMES[k] for k in range(len(DIR_NAMES)) if mask[k]]
        if not dirs_filtered:
            dirs_filtered = list(DIR_NAMES)
        self.dir_sel.options = dirs_filtered
        if self.dir_sel.value not in dirs_filtered:
            self.dir_sel.value = dirs_filtered[0]

    def _rebuild_matches(self):
        """Build (grasp, dir) match list under current filter."""
        if self._cache is None:
            return
        cf = self._cache["dir_collision_free"]
        es = self._cache["dir_escape"]
        safe = cf & es
        flt = self.filter_sel.value
        if flt == "all":
            mask = np.ones_like(cf, dtype=bool)
        elif flt == "safe (no-coll + escape)":
            mask = safe
        elif flt == "no-coll + no-escape":
            mask = cf & ~es
        elif flt == "coll + escape":
            mask = ~cf & es
        elif flt == "coll + no-escape":
            mask = ~cf & ~es
        elif flt == "dead grasp (no safe dir)":
            dead = ~safe.any(axis=1, keepdims=True)
            mask = np.broadcast_to(dead, cf.shape).copy()
        else:
            mask = np.ones_like(cf, dtype=bool)

        self._matches = np.argwhere(mask)
        M = len(self._matches)
        if M == 0:
            self.match_idx.disabled = True
            self.match_idx.max = 1
            self.match_idx.value = 0
            self.coll_info.value = f"0 matches under filter '{flt}'"
            return
        self.match_idx.disabled = False
        self.match_idx.max = M - 1
        if self.match_idx.value >= M:
            self.match_idx.value = 0
        unique_grasps = len(set(int(g) for g, _ in self._matches))
        self.coll_info.value = (
            f"{M} matches under '{flt}'  |  {unique_grasps} unique grasps"
        )
        self._jump_to_match()

    def _jump_to_match(self):
        """Set grasp_idx + direction from current match_idx."""
        if self._matches is None or len(self._matches) == 0:
            return
        m = max(0, min(int(self.match_idx.value), len(self._matches) - 1))
        i, k = self._matches[m]
        self._suspend_refresh = True
        self.grasp_idx.value = int(i)
        self._rebuild_dir_options(int(i))
        self.dir_sel.value = DIR_NAMES[int(k)]
        self._suspend_refresh = False
        self._refresh()

    def _maybe_precompute(self):
        if self._candidates is None:
            return False
        sig = self._cache_signature()
        if self._cache is not None and self._cache.get("signature") == sig:
            return True

        wrist_obj, pregrasp, _, scene_info = self._candidates
        N = len(wrist_obj)
        K = 12
        T = SWEEP_STEPS_VIZ
        t_values = np.linspace(0.0, SWEEP_DIST, T)

        obb_tf, half_ext = load_obb(self.obj_sel.value)
        half_ext_expanded = half_ext + BBOX_EXPAND
        z_rad = np.radians(self.z_rot.value)
        x_off = float(self.x_offset.value)

        # Per-grasp native pose from scene JSON
        native_poses = np.zeros((N, 4, 4), dtype=np.float64)
        # Cache scene pose lookups
        scene_pose_cache = {}
        for i, (st, sid, _) in enumerate(scene_info):
            key = (st, sid)
            if key not in scene_pose_cache:
                scene_pose_cache[key] = load_scene_pose(self.obj_sel.value, st, sid)
            native_poses[i] = apply_xz_perturbation(scene_pose_cache[key], x_off, z_rad)

        wrist_world = np.einsum('nij,njk->nik', native_poses, wrist_obj)  # (N, 4, 4)

        # Directions (N, K, 3)
        dirs_world = np.broadcast_to(WORLD_AXES[None, :, :], (N, 6, 3))
        dirs_local = (wrist_world[:, :3, :3] @ WORLD_AXES.T).transpose(0, 2, 1)
        dirs = np.concatenate([dirs_world, dirs_local], axis=1)

        # Swept poses (N, K, T, 4, 4)
        deltas = dirs[:, :, None, :] * t_values[None, None, :, None]
        poses = np.broadcast_to(wrist_world[:, None, None, :, :], (N, K, T, 4, 4)).copy()
        poses[:, :, :, :3, 3] = wrist_world[:, None, None, :3, 3] + deltas

        qpos = np.broadcast_to(pregrasp[:, None, None, :], (N, K, T, pregrasp.shape[1])).copy()

        # Group by (scene_type, scene_id) → one world_cfg per group
        from collections import defaultdict
        scene_groups = defaultdict(list)
        for i, (st, sid, _) in enumerate(scene_info):
            scene_groups[(st, sid)].append(i)

        planner = self._get_planner(self.hand_sel.value)

        centers_all = None
        collide_all = None
        radii_all = None
        bbox_worlds = np.zeros((N, 4, 4))

        for gi_scene, ((st, sid), grasp_idx_list) in enumerate(scene_groups.items()):
            self.coll_info.value = (
                f"Precomputing scene {gi_scene+1}/{len(scene_groups)} ({st}/{sid}, "
                f"{len(grasp_idx_list)} grasps)..."
            )
            grasp_idx_arr = np.array(grasp_idx_list, dtype=int)
            scene_pose = native_poses[grasp_idx_arr[0]]
            world_cfg = self._build_world_cfg(self.obj_sel.value, scene_pose)

            poses_sub = poses[grasp_idx_arr].reshape(-1, 4, 4)
            qpos_sub = qpos[grasp_idx_arr].reshape(-1, qpos.shape[-1])
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

        # Per-sphere bbox check (per-grasp because each grasp has own bbox_world)
        in_bbox_all = np.zeros_like(collide_all)
        for gi in range(N):
            in_bbox_all[gi] = points_inside_bbox(centers_all[gi], bbox_worlds[gi], half_ext_expanded)

        coll_during = collide_all.any(axis=(2, 3))
        all_outside_final = ~in_bbox_all[:, :, -1, :].any(axis=2)
        dir_collision_free = ~coll_during
        dir_escape = all_outside_final
        dir_safe = dir_collision_free & dir_escape

        self._cache = {
            "signature": sig,
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
            "scene_info": scene_info,
        }
        n_safe_any = int(dir_safe.any(axis=1).sum())
        self.coll_info.value = f"Cache OK | {n_safe_any}/{N} grasps have at least one safe reset dir"
        self._rebuild_matches()
        return True

    def _build_world_cfg(self, obj_name, obj_pose):
        mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
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

    def _clear_spheres(self):
        for h in self._sphere_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._sphere_handles = []

    def _draw_spheres(self, centers, radii, collide, in_bbox):
        """red = collision, blue = inside bbox (no coll), green = outside bbox (no coll)."""
        self._clear_spheres()
        for i, (c, r, hit, inside) in enumerate(zip(centers, radii, collide, in_bbox)):
            if hit:
                color, opacity = (230, 60, 60), 0.9
            elif inside:
                color, opacity = (70, 130, 230), 0.5
            else:
                color, opacity = (90, 200, 110), 0.3
            h = self.server.scene.add_icosphere(
                name=f"/spheres/s{i}",
                radius=float(r),
                color=color,
                position=tuple(float(x) for x in c),
                opacity=opacity,
            )
            self._sphere_handles.append(h)

    def _draw_bbox(self, bbox_world, half_ext_expanded):
        if getattr(self, "_bbox_handle", None) is not None:
            self._bbox_handle.remove()
            self._bbox_handle = None
        dims = tuple(2.0 * half_ext_expanded.astype(float))
        box = trimesh.creation.box(extents=dims)
        wxyz = Rot.from_matrix(bbox_world[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self._bbox_handle = self.server.scene.add_mesh_simple(
            name="/objects/bbox",
            vertices=np.array(box.vertices, dtype=np.float32),
            faces=np.array(box.faces, dtype=np.uint32),
            color=(255, 200, 80),
            opacity=0.15,
            flat_shading=True,
            side="double",
            wxyz=wxyz,
            position=bbox_world[:3, 3],
        )

    def _draw_retreat_line(self, start_w, dir_w):
        if self._line_handle is not None:
            self._line_handle.remove()
            self._line_handle = None
        end_w = start_w + SWEEP_DIST * dir_w
        pts = np.stack([start_w, end_w], axis=0).astype(np.float32)
        self._line_handle = self.server.scene.add_spline_catmull_rom(
            name="/objects/retreat_line",
            positions=pts,
            color=(255, 80, 80),
            line_width=4.0,
        )

    def _refresh(self):
        if self._suspend_refresh:
            return
        if not self._maybe_precompute():
            return

        cache = self._cache
        wrist_obj, pregrasp, _, scene_info = self._candidates
        N = len(wrist_obj)

        i = int(self.grasp_idx.value)
        if i >= N:
            return
        if self.dir_sel.value not in DIR_NAMES:
            return
        k = DIR_NAMES.index(self.dir_sel.value)
        t_idx = int(round(self.sweep_t.value / SWEEP_STEP))
        t_idx = max(0, min(t_idx, SWEEP_STEPS_VIZ - 1))

        obj_pose = cache["native_poses"][i]
        wrist_world = cache["wrist_world"][i]
        d_w = cache["dirs"][i, k]
        t_val = cache["t_values"][t_idx]

        wrist_swept = wrist_world.copy()
        wrist_swept[:3, 3] = wrist_world[:3, 3] + t_val * d_w

        self._show_object(obj_pose)
        self._draw_retreat_line(wrist_world[:3, 3], d_w)

        # Scene info display
        st, sid, gid = scene_info[i]
        self.scene_info_text.value = f"{st}/{sid}  (grasp {gid})"

        hand = self.hand_sel.value
        for hn in HAND_URDFS:
            if hn in self.robot_dict:
                self.robot_dict[hn].set_visibility(hn == hand)

        robot = self.robot_dict.get(hand)
        if robot is None:
            return
        robot._visual_root_frame.position = wrist_swept[:3, 3]
        robot._visual_root_frame.wxyz = Rot.from_matrix(wrist_swept[:3, :3]).as_quat()[[3, 0, 1, 2]]
        robot.update_cfg(pregrasp[i])

        if self.show_bbox.value:
            self._draw_bbox(cache["bbox_worlds"][i], cache["half_ext_expanded"])
        elif getattr(self, "_bbox_handle", None) is not None:
            self._bbox_handle.remove()
            self._bbox_handle = None

        if self.show_spheres.value:
            centers = cache["centers"][i, k, t_idx]
            radii = cache["radii"]
            collide = cache["collide"][i, k, t_idx]
            in_bbox = cache["in_bbox"][i, k, t_idx]
            self._draw_spheres(centers, radii, collide, in_bbox)
            n_hit = int(collide.sum())
            n_in = int(in_bbox.sum())
            self.coll_info.value = (
                f"coll {n_hit}/{len(collide)}  inside-bbox {n_in}/{len(in_bbox)}"
            )
        else:
            self._clear_spheres()
            self.coll_info.value = "spheres hidden"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", default=DEFAULT_OBJ_PATH)
    args = parser.parse_args()

    globals()["obj_path"] = args.obj_path

    vis = ResetViewer()
    vis.start_viewer()
