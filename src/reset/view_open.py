"""Minimal viewer for opening results across (tabletop_pose, grasp) pairs.

Pick object → tabletop pose → grasp idx → slide Open↔Pregrasp to see the
opening animation. No filter / match / direction complexity.

Usage:
    python src/reset/view_open.py
    python src/reset/view_open.py --obj_root /home/mingi/shared_data/AutoDex/object/robothome
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs", "reset")

HAND_URDFS = {
    "allegro": os.path.join(REPO_ROOT,
        "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
        "allegro_description/allegro_hand_description_right.urdf"),
    "inspire": os.path.join(REPO_ROOT,
        "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
        "inspire_description/inspire_hand_right.urdf"),
    "inspire_left": os.path.join(REPO_ROOT,
        "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
        "inspire_description/inspire_hand_left.urdf"),
    "inspire_f1": os.path.join(REPO_ROOT,
        "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
        "inspire_f1_description/inspire_f1_hand_right.urdf"),
}

OBJ_ROOT_RUNTIME = DEFAULT_OBJ_ROOT


def list_objects_with_open_npz():
    if not os.path.isdir(OUTPUTS_ROOT):
        return []
    return sorted(
        d for d in os.listdir(OUTPUTS_ROOT)
        if os.path.isdir(os.path.join(OUTPUTS_ROOT, d))
        and any(f.startswith("open_") and f.endswith(".npz")
                for f in os.listdir(os.path.join(OUTPUTS_ROOT, d)))
    )


def list_open_npz(obj_name):
    d = os.path.join(OUTPUTS_ROOT, obj_name)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.startswith("open_") and f.endswith(".npz"))


class OpenViewer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.gui_playing.value = False

        self._cache = None
        self._object_handle = None
        self._sphere_handles = []

        objs = list_objects_with_open_npz()
        with self.server.gui.add_folder("Open Viewer"):
            self.obj_sel = self.server.gui.add_dropdown(
                "Object", options=objs, initial_value=objs[0] if objs else "",
            )
            self.file_sel = self.server.gui.add_dropdown(
                "Result npz", options=[], initial_value="",
            )
            self.pair_idx = self.server.gui.add_slider(
                "(Pose, Grasp) Pair", min=0, max=1, step=1, initial_value=0,
            )
            self.pair_info = self.server.gui.add_text(
                "Current pair", initial_value="-", disabled=True,
            )
            self.sweep_t = self.server.gui.add_slider(
                "Open(0) → Pregrasp(1)", min=0.0, max=1.0, step=0.05, initial_value=0.0,
            )
            self.show_spheres = self.server.gui.add_checkbox(
                "Show Collision Spheres", initial_value=True,
            )
            self.status_text = self.server.gui.add_text(
                "Status", initial_value="-", disabled=True,
            )

        for hand_name, urdf in HAND_URDFS.items():
            if os.path.exists(urdf):
                self.add_robot(hand_name, urdf)
                self.robot_dict[hand_name].set_visibility(False)

        self.obj_sel.on_update(lambda e: self._on_obj_change())
        self.file_sel.on_update(lambda e: self._on_file_change())
        self.pair_idx.on_update(lambda e: self._refresh())
        self.sweep_t.on_update(lambda e: self._refresh())
        self.show_spheres.on_update(lambda e: self._refresh())

        self._draw_table()
        self._on_obj_change()

    def _draw_table(self):
        box = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
        self.server.scene.add_mesh_simple(
            name="/objects/table",
            vertices=np.array(box.vertices, dtype=np.float32),
            faces=np.array(box.faces, dtype=np.uint32),
            color=(240, 240, 245),
            opacity=0.85,
            flat_shading=True,
            side="double",
            position=(0.0, 0.0, -0.1),
        )

    def _show_object(self, obj_pose, obj_name):
        if self._object_handle is not None:
            self._object_handle.remove()
            self._object_handle = None
        mesh_path = os.path.join(OBJ_ROOT_RUNTIME, obj_name,
                                 "processed_data", "mesh", "simplified.obj")
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
            opacity = 0.9 if hit else 0.35
            h = self.server.scene.add_icosphere(
                name=f"/spheres/s{i}",
                radius=float(r),
                color=color,
                position=tuple(float(x) for x in c),
                opacity=opacity,
            )
            self._sphere_handles.append(h)

    def _on_obj_change(self):
        obj = self.obj_sel.value
        files = list_open_npz(obj) if obj else []
        self.file_sel.options = files if files else ["(none)"]
        if files:
            self.file_sel.value = files[0]
        self._on_file_change()

    def _on_file_change(self):
        obj = self.obj_sel.value
        f = self.file_sel.value
        if not obj or not f or f == "(none)":
            self._cache = None
            return
        npz_path = os.path.join(OUTPUTS_ROOT, obj, f)
        d = np.load(npz_path, allow_pickle=True)
        meta = d["meta"][0] if d["meta"].dtype == object else d["meta"].item()
        c = {k: d[k] for k in d.files}
        c["meta"] = meta
        self._cache = c

        M = c["centers"].shape[0]
        self.pair_idx.max = max(M - 1, 0)
        if self.pair_idx.value >= M:
            self.pair_idx.value = 0
        self._refresh()

    def _refresh(self):
        if self._cache is None:
            return
        cache = self._cache
        M = cache["centers"].shape[0]
        m = int(self.pair_idx.value)
        m = max(0, min(m, M - 1))
        p_idx = int(cache["pair_pose_idx"][m])
        original_grasp_idx = int(cache["pair_grasp_idx"][m])
        pose_id = str(cache["pose_ids"][p_idx])

        T = cache["centers"].shape[1]
        s = max(0.0, min(float(self.sweep_t.value), 1.0))
        # 0 = open (final iter), 1 = pregrasp (iter 0)
        t_idx = int(round((1.0 - s) * (T - 1)))
        t_idx = max(0, min(t_idx, T - 1))

        meta = cache["meta"]
        obj_name = meta["obj"]
        hand = meta["hand"]

        obj_pose = cache["pose_se3"][p_idx]
        wrist_world = cache["wrist_world"][m]

        self._show_object(obj_pose, obj_name)

        for hn in HAND_URDFS:
            if hn in self.robot_dict:
                self.robot_dict[hn].set_visibility(hn == hand)

        robot = self.robot_dict.get(hand)
        if robot is None:
            return
        robot._visual_root_frame.position = wrist_world[:3, 3]
        robot._visual_root_frame.wxyz = Rot.from_matrix(wrist_world[:3, :3]).as_quat()[[3, 0, 1, 2]]
        robot.update_cfg(cache["qpos_traj"][m, t_idx])

        self.pair_info.value = (
            f"pose {pose_id}  /  grasp {original_grasp_idx}  /  pair {m+1}/{M}"
        )

        if self.show_spheres.value:
            centers = cache["centers"][m, t_idx]
            radii = cache["radii"]
            collide = cache["collide"][m, t_idx]
            self._draw_spheres(centers, radii, collide)
            final_clear_mm = float(cache["min_clearance"][m, -1]) * 1000
            init_clear_mm = float(cache["min_clearance"][m, 0]) * 1000
            self.status_text.value = (
                f"iter {t_idx+1}/{T}  clearance init {init_clear_mm:.1f}mm → "
                f"final {final_clear_mm:.1f}mm"
            )
        else:
            self._clear_spheres()
            self.status_text.value = "spheres hidden"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    args = parser.parse_args()
    globals()["OBJ_ROOT_RUNTIME"] = args.obj_root

    vis = OpenViewer()
    vis.start_viewer()
