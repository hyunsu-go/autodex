"""Viewer for precomputed reset results (translate / open_translate).

Loads outputs/reset/{obj}/{method}_{version}.npz produced by
compute_translate.py / compute_open_translate.py and renders the sweep
interactively. No cuRobo computation.

Usage:
    python src/reset/view.py
    python src/reset/view.py --obj_root /home/mingi/shared_data/AutoDex/object/robothome
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

from common import (
    DIR_NAMES, SWEEP_DIST, SWEEP_STEPS, SWEEP_STEP,
)


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


def list_objects():
    if not os.path.isdir(OUTPUTS_ROOT):
        return []
    return sorted(d for d in os.listdir(OUTPUTS_ROOT)
                  if os.path.isdir(os.path.join(OUTPUTS_ROOT, d)))


def list_npz_for_obj(obj_name):
    d = os.path.join(OUTPUTS_ROOT, obj_name)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".npz"))


class ResetViewer(ViserViewer):
    def __init__(self):
        super().__init__()
        self.gui_playing.value = False

        self._cache = None
        self._object_handle = None
        self._line_handle = None
        self._bbox_handle = None
        self._sphere_handles = []
        self._matches = []
        self._suspend = False

        objs = list_objects()
        with self.server.gui.add_folder("Reset Viewer"):
            self.obj_sel = self.server.gui.add_dropdown(
                "Object", options=objs, initial_value=objs[0] if objs else "",
            )
            self.file_sel = self.server.gui.add_dropdown(
                "Result npz", options=[], initial_value="",
            )
            self.filter_sel = self.server.gui.add_dropdown(
                "Filter", options=["all"], initial_value="all",
            )
            self.match_idx = self.server.gui.add_slider(
                "Match #", min=0, max=1, step=1, initial_value=0, disabled=True,
            )
            self.scene_info_text = self.server.gui.add_text(
                "Scene (auto)", initial_value="-", disabled=True,
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

        self.obj_sel.on_update(lambda e: self._on_obj_change())
        self.file_sel.on_update(lambda e: self._on_file_change())
        self.filter_sel.on_update(lambda e: self._rebuild_matches())
        self.match_idx.on_update(lambda e: self._jump_to_match())
        self.grasp_idx.on_update(lambda e: self._on_grasp_change())
        self.dir_sel.on_update(lambda e: self._refresh())
        self.sweep_t.on_update(lambda e: self._refresh())
        self.show_spheres.on_update(lambda e: self._refresh())
        self.show_bbox.on_update(lambda e: self._refresh())

        self._draw_table()
        self._on_obj_change()

    # ---- scene drawing ----

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

    def _draw_retreat_line(self, start_w, dir_w):
        end_w = start_w + SWEEP_DIST * dir_w
        self._draw_retreat_line_explicit(start_w, end_w)

    def _draw_retreat_line_explicit(self, start_w, end_w):
        if self._line_handle is not None:
            self._line_handle.remove()
            self._line_handle = None
        pts = np.stack([start_w, end_w]).astype(np.float32)
        self._line_handle = self.server.scene.add_spline_catmull_rom(
            name="/objects/retreat_line",
            positions=pts,
            color=(255, 80, 80),
            line_width=4.0,
        )

    def _draw_bbox(self, bbox_world, half_ext_expanded):
        if self._bbox_handle is not None:
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

    def _clear_spheres(self):
        for h in self._sphere_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._sphere_handles = []

    def _draw_spheres(self, centers, radii, collide, in_bbox):
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

    # ---- data loading ----

    def _on_obj_change(self):
        obj = self.obj_sel.value
        files = list_npz_for_obj(obj) if obj else []
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
        if "q_open" not in c:
            c["q_open"] = c["pregrasp"]

        # Adapt direction dropdown + sweep slider based on npz contents.
        dir_names_arr = c.get("dir_names")
        if dir_names_arr is not None:
            dir_names_list = [str(x) for x in dir_names_arr]
        else:
            dir_names_list = list(DIR_NAMES)
        c["dir_names_list"] = dir_names_list

        t_values = c["t_values"]
        self._suspend = True
        self.dir_sel.options = dir_names_list
        if self.dir_sel.value not in dir_names_list:
            self.dir_sel.value = dir_names_list[0]
        # sweep slider range to match t_values
        t_min = float(t_values.min())
        t_max = float(t_values.max())
        if len(t_values) > 1:
            t_step = (t_max - t_min) / (len(t_values) - 1)
        else:
            t_step = 1.0
        self.sweep_t.min = t_min
        self.sweep_t.max = t_max if t_max > t_min else t_min + 1.0
        self.sweep_t.step = t_step if t_step > 0 else 1.0
        if self.sweep_t.value < t_min or self.sweep_t.value > t_max:
            self.sweep_t.value = t_min
        self._suspend = False

        self._cache = c
        N = len(c["wrist_world"])
        self.grasp_idx.max = max(N - 1, 0)
        if self.grasp_idx.value >= N:
            self.grasp_idx.value = 0
        self._rebuild_matches()

    # ---- filter / match ----

    def _rebuild_matches(self):
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
        uniq = len(set(int(g) for g, _ in self._matches))
        self.coll_info.value = f"{M} matches '{flt}'  |  {uniq} unique grasps"
        self._jump_to_match()

    def _rebuild_dir_options(self, grasp_i):
        if self._cache is None:
            return
        dnames = self._cache.get("dir_names_list", list(DIR_NAMES))
        cf = self._cache["dir_collision_free"][grasp_i]
        es = self._cache["dir_escape"][grasp_i]
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
        dirs_filtered = [dnames[k] for k in range(len(dnames)) if mask[k]]
        if not dirs_filtered:
            dirs_filtered = list(dnames)
        self.dir_sel.options = dirs_filtered
        if self.dir_sel.value not in dirs_filtered:
            self.dir_sel.value = dirs_filtered[0]

    def _jump_to_match(self):
        if not len(self._matches):
            return
        m = max(0, min(int(self.match_idx.value), len(self._matches) - 1))
        i, k = self._matches[m]
        self._suspend = True
        self.grasp_idx.value = int(i)
        self._rebuild_dir_options(int(i))
        dnames = self._cache.get("dir_names_list", list(DIR_NAMES))
        self.dir_sel.value = dnames[int(k)]
        self._suspend = False
        self._refresh()

    def _on_grasp_change(self):
        if self._suspend:
            return
        if self._cache is not None:
            self._suspend = True
            self._rebuild_dir_options(int(self.grasp_idx.value))
            self._suspend = False
        self._refresh()

    # ---- render ----

    def _refresh(self):
        if self._suspend or self._cache is None:
            return
        cache = self._cache
        N = len(cache["wrist_world"])
        i = int(self.grasp_idx.value)
        if i >= N:
            return
        dnames = cache.get("dir_names_list", list(DIR_NAMES))
        if self.dir_sel.value not in dnames:
            return
        k = dnames.index(self.dir_sel.value)
        # Snap sweep_t to nearest stored t_value index
        t_values = cache["t_values"]
        t_idx = int(np.argmin(np.abs(t_values - float(self.sweep_t.value))))

        meta = cache["meta"]
        obj_name = meta["obj"]
        hand = meta["hand"]

        obj_pose = cache["native_poses"][i]
        wrist_world = cache["wrist_world"][i]
        d_w = cache["dirs"][i, k]
        t_val = float(cache["t_values"][t_idx])

        # If gradient method has stored trajectories, use those for wrist position;
        # otherwise compute from sweep direction × t.
        if "trajectories" in cache:
            wrist_swept = cache["trajectories"][i, t_idx].copy()
            line_end = cache["trajectories"][i, -1, :3, 3]
        else:
            wrist_swept = wrist_world.copy()
            wrist_swept[:3, 3] = wrist_world[:3, 3] + t_val * d_w
            line_end = wrist_world[:3, 3] + (t_values.max()) * d_w

        self._show_object(obj_pose, obj_name)
        self._draw_retreat_line_explicit(wrist_world[:3, 3], line_end)

        st, sid, gid = cache["scene_info"][i]
        self.scene_info_text.value = f"{st}/{sid} (grasp {gid})  method={meta.get('method', '?')}"

        for hn in HAND_URDFS:
            if hn in self.robot_dict:
                self.robot_dict[hn].set_visibility(hn == hand)

        robot = self.robot_dict.get(hand)
        if robot is None:
            return
        robot._visual_root_frame.position = wrist_swept[:3, 3]
        robot._visual_root_frame.wxyz = Rot.from_matrix(wrist_swept[:3, :3]).as_quat()[[3, 0, 1, 2]]
        # If per-step finger qpos trajectory is saved (open method), animate it.
        if "qpos_traj" in cache:
            robot.update_cfg(cache["qpos_traj"][i, t_idx])
        else:
            robot.update_cfg(cache["q_open"][i])

        if self.show_bbox.value:
            self._draw_bbox(cache["bbox_worlds"][i], cache["half_ext_expanded"])
        elif self._bbox_handle is not None:
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
    parser.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    args = parser.parse_args()
    OBJ_ROOT_RUNTIME = args.obj_root  # noqa: F841  (kept for symmetry; module-level set below)
    globals()["OBJ_ROOT_RUNTIME"] = args.obj_root

    vis = ResetViewer()
    vis.start_viewer()
