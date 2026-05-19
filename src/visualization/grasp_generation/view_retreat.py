"""Visualize retreat sweep for a selected grasp + direction.

Pick (object, tabletop pose, grasp idx, direction) and slide sweep t [0, 5cm].
Shows hand (pregrasp finger config) translated by t along the chosen direction.

Usage:
    python src/visualization/grasp_generation/view_retreat.py
"""

import os
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH, load_candidate
from autodex.planner.planner import GraspPlanner

obj_path = DEFAULT_OBJ_PATH  # rebound from CLI in __main__

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

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
    "inspire_f1_left": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_f1_left_description",
        "inspire_f1_hand_left.urdf",
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

SWEEP_DIST = 0.05
SWEEP_STEPS_VIZ = 11           # cached sample count
SWEEP_STEP = SWEEP_DIST / (SWEEP_STEPS_VIZ - 1)  # 0.005 m

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


class RetreatViewer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.gui_playing.value = False

        self._candidates = None  # (wrist_obj, pregrasp, grasp, scene_info)
        self._table_handle = None
        self._object_handle = None
        self._line_handle = None
        self._sphere_handles = []
        self._planners = {}  # hand_name -> GraspPlanner (lazy)
        self._cache = None   # {signature, centers, radii, collide, obj_pose}

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
            self.pose_sel = self.server.gui.add_dropdown(
                "Tabletop Pose", options=[], initial_value="",
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
            self.coll_info = self.server.gui.add_text(
                "Collision", initial_value="-", disabled=True,
            )

        for hand_name, urdf in HAND_URDFS.items():
            if os.path.exists(urdf):
                self.add_robot(hand_name, urdf)
                self.robot_dict[hand_name].set_visibility(False)

        # Wire callbacks
        self.hand_sel.on_update(lambda e: self._on_hand_change())
        self.version_sel.on_update(lambda e: self._on_version_change())
        self.obj_sel.on_update(lambda e: self._on_obj_change())
        self.pose_sel.on_update(lambda e: self._refresh())
        self.x_offset.on_update(lambda e: self._refresh())
        self.z_rot.on_update(lambda e: self._refresh())
        self.grasp_idx.on_update(lambda e: self._refresh())
        self.dir_sel.on_update(lambda e: self._refresh())
        self.sweep_t.on_update(lambda e: self._refresh())
        self.show_spheres.on_update(lambda e: self._refresh())

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
            self.pose_sel.options = ["(none)"]
            return

        wrist_obj, pregrasp, grasp, scene_info = load_candidate(
            obj, np.eye(4), self.version_sel.value,
            shuffle=False, skip_done=False, hand=self.hand_sel.value,
        )
        N = len(wrist_obj)

        poses = list_tabletop_poses(obj)
        self.pose_sel.options = poses if poses else ["(none)"]
        if poses:
            self.pose_sel.value = poses[0]

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
        return (
            self.hand_sel.value, self.version_sel.value, self.obj_sel.value,
            self.pose_sel.value,
            round(float(self.x_offset.value), 4),
            round(float(self.z_rot.value), 2),
        )

    def _maybe_precompute(self):
        if self._candidates is None:
            return False
        if self.pose_sel.value in ("", "(none)"):
            return False
        sig = self._cache_signature()
        if self._cache is not None and self._cache["signature"] == sig:
            return True

        self.coll_info.value = "Precomputing..."

        wrist_obj, pregrasp, _, _ = self._candidates
        N = len(wrist_obj)
        K = 12
        T = SWEEP_STEPS_VIZ
        t_values = np.linspace(0.0, SWEEP_DIST, T)

        obj_pose = load_tabletop_pose(
            self.obj_sel.value, self.pose_sel.value,
            x_offset=self.x_offset.value,
            z_rotation_rad=np.radians(self.z_rot.value),
        )
        wrist_world = obj_pose @ wrist_obj  # (N, 4, 4)

        # Direction batch (N, K, 3)
        dirs_world = np.broadcast_to(WORLD_AXES[None, :, :], (N, 6, 3))
        dirs_local = (wrist_world[:, :3, :3] @ WORLD_AXES.T).transpose(0, 2, 1)
        dirs = np.concatenate([dirs_world, dirs_local], axis=1)

        # Swept poses: (N, K, T, 4, 4)
        deltas = dirs[:, :, None, :] * t_values[None, None, :, None]
        poses = np.broadcast_to(wrist_world[:, None, None, :, :], (N, K, T, 4, 4)).copy()
        poses[:, :, :, :3, 3] = wrist_world[:, None, None, :3, 3] + deltas

        qpos = np.broadcast_to(pregrasp[:, None, None, :], (N, K, T, pregrasp.shape[1])).copy()

        B = N * K * T
        poses_flat = poses.reshape(B, 4, 4)
        qpos_flat = qpos.reshape(B, -1)

        world_cfg = self._build_world_cfg(self.obj_sel.value, obj_pose)
        planner = self._get_planner(self.hand_sel.value)
        centers, radii, collide = planner.check_collision_per_sphere(world_cfg, poses_flat, qpos_flat)
        # centers (B, N_s, 3), radii (N_s,), collide (B, N_s)

        N_s = radii.shape[0]
        self._cache = {
            "signature": sig,
            "centers": centers.reshape(N, K, T, N_s, 3),
            "radii": radii,
            "collide": collide.reshape(N, K, T, N_s),
            "obj_pose": obj_pose,
            "wrist_world": wrist_world,
            "dirs": dirs,  # (N, K, 3)
            "t_values": t_values,
        }
        self.coll_info.value = f"Cache: {N}×{K}×{T} ({N_s} spheres)"
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

    def _draw_spheres(self, centers, radii, collide):
        self._clear_spheres()
        for i, (c, r, hit) in enumerate(zip(centers, radii, collide)):
            color = (230, 60, 60) if hit else (90, 200, 110)
            h = self.server.scene.add_icosphere(
                name=f"/spheres/s{i}",
                radius=float(r),
                color=color,
                position=tuple(float(x) for x in c),
                opacity=0.85 if hit else 0.35,
            )
            self._sphere_handles.append(h)

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
        if not self._maybe_precompute():
            return

        cache = self._cache
        wrist_obj, pregrasp, _, _ = self._candidates
        N = len(wrist_obj)

        i = int(self.grasp_idx.value)
        if i >= N:
            return
        k = DIR_NAMES.index(self.dir_sel.value)
        t_idx = int(round(self.sweep_t.value / SWEEP_STEP))
        t_idx = max(0, min(t_idx, SWEEP_STEPS_VIZ - 1))

        obj_pose = cache["obj_pose"]
        wrist_world = cache["wrist_world"][i]
        d_w = cache["dirs"][i, k]
        t_val = cache["t_values"][t_idx]

        wrist_swept = wrist_world.copy()
        wrist_swept[:3, 3] = wrist_world[:3, 3] + t_val * d_w

        self._show_object(obj_pose)
        self._draw_retreat_line(wrist_world[:3, 3], d_w)

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

        if self.show_spheres.value:
            centers = cache["centers"][i, k, t_idx]
            radii = cache["radii"]
            collide = cache["collide"][i, k, t_idx]
            self._draw_spheres(centers, radii, collide)
            n_hit = int(collide.sum())
            self.coll_info.value = f"{n_hit}/{len(collide)} spheres collide  (cache OK)"
        else:
            self._clear_spheres()
            self.coll_info.value = "spheres hidden"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", default=DEFAULT_OBJ_PATH)
    args = parser.parse_args()

    globals()["obj_path"] = args.obj_path

    vis = RetreatViewer()
    vis.start_viewer()
