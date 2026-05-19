"""Browse openpose results saved per (tabletop_pose, grasp) pair.

Reads:
    - per-grasp:  {candidates_root}/{hand}/{version}/{obj}/{scene_type}/{scene_id}/{grasp_idx}/
                    wrist_se3.npy, pregrasp_pose.npy, openpose_{pose_id}.npy
    - per-object: outputs/reset/{obj}/open_{version}.npz (for valid_final flag)
    - tabletop:   {obj_path}/{obj}/processed_data/info/tabletop/{pose_id}.npy

Usage:
    python src/visualization/grasp_generation/view_openpose.py
    python src/visualization/grasp_generation/view_openpose.py \
        --candidates_root /home/mingi/shared_data/AutoDex/candidates
"""

import os
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESET_OUT_ROOT = os.path.join(REPO_ROOT, "outputs", "reset")

HAND_URDFS = {
    "allegro": os.path.join(
        REPO_ROOT, "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
                   "allegro_description/allegro_hand_description_right.urdf"),
    "inspire": os.path.join(
        REPO_ROOT, "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
                   "inspire_description/inspire_hand_right.urdf"),
    "inspire_left": os.path.join(
        REPO_ROOT, "src/grasp_generation/BODex/src/curobo/content/assets/robot/"
                   "inspire_description/inspire_hand_left.urdf"),
}

SCENE_RGBA = {
    "table":    [240, 240, 245, 230],
    "floor":    [200, 200, 210, 180],
}

# Has to match what compute_open.py uses for the scene.
DEFAULT_TABLE = {"dims": [2, 2, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0]}
DEFAULT_FLOOR = {"dims": [10.0, 10.0, 0.05], "pose": [0.0, 0.0, -0.025, 1, 0, 0, 0]}


def _list_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _find_versions(hand_root, max_depth=4):
    """Find all version paths under hand_root. A version is the path prefix such
    that {version}/{obj}/{scene_type}/{scene_id}/{grasp_idx}/wrist_se3.npy exists,
    so versions may themselves be nested (e.g. ``reset/0``)."""
    versions = set()
    if not os.path.isdir(hand_root):
        return []
    base_depth = hand_root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(hand_root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "wrist_se3.npy" in filenames:
            rel = os.path.relpath(dirpath, hand_root)
            parts = rel.split(os.sep)
            if len(parts) >= 4:
                version = os.sep.join(parts[:-4])
                versions.add(version)
            dirnames[:] = []  # stop descending once we hit a grasp dir
    return sorted(versions)


def _list_tabletop_poses(obj_path_root, obj_name):
    pose_dir = os.path.join(obj_path_root, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted(f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy"))


def _load_tabletop_pose(obj_path_root, obj_name, pose_id, x_offset=0.0, z_rotation=0.0):
    p = np.load(os.path.join(obj_path_root, obj_name, "processed_data", "info",
                              "tabletop", f"{pose_id}.npy"))
    p[0, 3] += x_offset
    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p[:3, :3] = Rz @ p[:3, :3]
    return p


def _load_validity_index(obj_name, hand, version):
    """Read outputs/reset/{obj}/open_{version}.npz and build a dict:
    (pose_id_str, scene_type, scene_id_str, grasp_idx_str) -> bool valid_final
    """
    path = os.path.join(RESET_OUT_ROOT, obj_name, f"open_{version}.npz")
    if not os.path.exists(path):
        print(f"[validity] no npz at {path}; all entries marked Unknown")
        return None
    d = np.load(path, allow_pickle=True)
    pose_ids = [str(p) for p in d["pose_ids"].tolist()]
    pair_pi = d["pair_pose_idx"]
    pair_gi = d["pair_grasp_idx"]
    valid_final = d["valid_final"]
    scene_info = d["scene_info"]
    out = {}
    for m in range(len(valid_final)):
        st, sid, gid = scene_info[int(pair_gi[m])]
        key = (pose_ids[int(pair_pi[m])], str(st), str(sid), str(gid))
        out[key] = bool(valid_final[m])
    return out


class OpenPoseBrowser(ViserViewer):
    def __init__(self, obj_path_root, candidates_root):
        super().__init__()
        self.obj_path_root = obj_path_root
        self.candidates_root = candidates_root
        self.obj_pose = np.eye(4)
        self.gui_playing.value = True

        self.current_hand = None
        self.current_version = None
        self.current_obj = None
        self.current_pose_id = None
        self.current_scene_type = None
        self.current_scene_idx = None
        self.all_grasp_dirs = []
        self.filtered_grasp_dirs = []  # after All/Safe/Unsafe filter
        self.validity_index = None     # (pose_id, st, sid, gid) -> bool

        with self.server.gui.add_folder("OpenPose Viewer"):
            self.hand_selector = self.server.gui.add_dropdown(
                "Hand", options=[], initial_value="")
            self.version_selector = self.server.gui.add_dropdown(
                "Version", options=[], initial_value="")
            self.obj_selector = self.server.gui.add_dropdown(
                "Object", options=[], initial_value="")
            self.pose_id_selector = self.server.gui.add_dropdown(
                "Tabletop Pose", options=[], initial_value="")
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type", options=[], initial_value="")
            self.scene_idx_selector = self.server.gui.add_dropdown(
                "Scene", options=[], initial_value="")
            self.filter_selector = self.server.gui.add_dropdown(
                "Filter", options=["All", "Safe", "Unsafe"], initial_value="All")
            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Slot (filtered)", min=0, max=1, step=1, initial_value=0)
            self.grasp_id_label = self.server.gui.add_text(
                "Grasp Dir ID", initial_value="-", disabled=True)
            self.pose_blend = self.server.gui.add_slider(
                "Pose Blend (0=pregrasp ↔ 1=openpose)",
                min=0.0, max=1.0, step=0.01, initial_value=1.0,
            )
            self.metric_text = self.server.gui.add_text(
                "Metrics", initial_value="No grasp loaded", disabled=True)

        # Pre-load all hand URDFs
        for hn, urdf in HAND_URDFS.items():
            if os.path.exists(urdf):
                self.add_robot(hn, urdf)
                self.robot_dict[hn].set_visibility(False)
            # Pregrasp overlay clone
            ov_name = f"{hn}__pregrasp"
            if os.path.exists(urdf):
                self.add_robot(ov_name, urdf)
                self.robot_dict[ov_name].set_visibility(False)

        # Wire callbacks
        @self.hand_selector.on_update
        def _(_): self._on_hand_change()

        @self.version_selector.on_update
        def _(_): self._on_version_change()

        @self.obj_selector.on_update
        def _(_): self._on_object_change()

        @self.pose_id_selector.on_update
        def _(_): self._on_pose_id_change()

        @self.scene_type_selector.on_update
        def _(_): self._on_scene_type_change()

        @self.scene_idx_selector.on_update
        def _(_): self._on_scene_idx_change()

        @self.filter_selector.on_update
        def _(_): self._rebuild_filtered()

        @self.grasp_idx_slider.on_update
        def _(_): self._load_current_grasp()

        @self.pose_blend.on_update
        def _(_): self._load_current_grasp()

        self._cuboid_handles = []
        self._init_top_level()

    # ── cascading updates ────────────────────────────────────────────────

    def _init_top_level(self):
        hands = _list_dirs(self.candidates_root)
        self.hand_selector.options = hands if hands else ["(none)"]
        if hands:
            self.hand_selector.value = hands[0]
        self._on_hand_change()

    def _on_hand_change(self):
        self.current_hand = self.hand_selector.value
        versions = _list_dirs(os.path.join(self.candidates_root, self.current_hand or ""))
        self.version_selector.options = versions if versions else ["(none)"]
        if versions:
            self.version_selector.value = versions[0]
        self._on_version_change()

    def _on_version_change(self):
        self.current_version = self.version_selector.value
        objs = _list_dirs(os.path.join(self.candidates_root, self.current_hand or "",
                                        self.current_version or ""))
        # Only objects that actually have at least one openpose file
        objs = [o for o in objs if self._object_has_openpose(o)]
        self.obj_selector.options = objs if objs else ["(none)"]
        if objs:
            self.obj_selector.value = objs[0]
        self._on_object_change()

    def _object_has_openpose(self, obj_name):
        root = os.path.join(self.candidates_root, self.current_hand, self.current_version, obj_name)
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.startswith("openpose_") and f.endswith(".npy"):
                    return True
        return False

    def _on_object_change(self):
        self.current_obj = self.obj_selector.value
        if not self.current_obj or self.current_obj == "(none)":
            return
        self.validity_index = _load_validity_index(
            self.current_obj, self.current_hand, self.current_version,
        )
        pose_ids = _list_tabletop_poses(self.obj_path_root, self.current_obj)
        self.pose_id_selector.options = pose_ids if pose_ids else ["(none)"]
        if pose_ids:
            self.pose_id_selector.value = pose_ids[0]
        self._on_pose_id_change()

    def _on_pose_id_change(self):
        self.current_pose_id = self.pose_id_selector.value
        if not self.current_pose_id or self.current_pose_id == "(none)":
            return
        # Place object + cuboids in the scene at this tabletop pose.
        self._load_scene()
        # Refresh scene_type options
        scene_types = _list_dirs(os.path.join(self.candidates_root, self.current_hand,
                                               self.current_version, self.current_obj))
        self.scene_type_selector.options = scene_types if scene_types else ["(none)"]
        if scene_types:
            self.scene_type_selector.value = scene_types[0]
        self._on_scene_type_change()

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        if not self.current_scene_type or self.current_scene_type == "(none)":
            return
        scenes = _list_dirs(os.path.join(self.candidates_root, self.current_hand,
                                          self.current_version, self.current_obj,
                                          self.current_scene_type))
        scenes = sorted(scenes, key=lambda x: int(x) if x.isdigit() else x)
        self.scene_idx_selector.options = scenes if scenes else ["(none)"]
        if scenes:
            self.scene_idx_selector.value = scenes[0]
        self._on_scene_idx_change()

    def _on_scene_idx_change(self):
        self.current_scene_idx = self.scene_idx_selector.value
        if not self.current_scene_idx or self.current_scene_idx == "(none)":
            return
        scene_path = os.path.join(self.candidates_root, self.current_hand,
                                   self.current_version, self.current_obj,
                                   self.current_scene_type, self.current_scene_idx)
        self.all_grasp_dirs = _list_dirs(scene_path)
        self.all_grasp_dirs = sorted(self.all_grasp_dirs,
                                      key=lambda x: int(x) if x.isdigit() else x)
        self._rebuild_filtered()

    def _rebuild_filtered(self):
        flt = self.filter_selector.value
        out = []
        for gid in self.all_grasp_dirs:
            op_path = os.path.join(self.candidates_root, self.current_hand,
                                    self.current_version, self.current_obj,
                                    self.current_scene_type, self.current_scene_idx,
                                    gid, f"openpose_{self.current_pose_id}.npy")
            if not os.path.exists(op_path):
                continue
            if flt == "All":
                out.append(gid)
                continue
            key = (self.current_pose_id, self.current_scene_type,
                   self.current_scene_idx, gid)
            v = self.validity_index.get(key) if self.validity_index else None
            if flt == "Safe" and v is True:
                out.append(gid)
            elif flt == "Unsafe" and v is False:
                out.append(gid)
            elif flt == "Unsafe" and v is None and self.validity_index is None:
                # Without validity info we can't decide; skip Unsafe filter.
                continue
        self.filtered_grasp_dirs = out

        if not self.filtered_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            self.grasp_idx_slider.max = 1
            self.grasp_idx_slider.value = 0
            self.metric_text.value = (
                f"No grasps matching filter='{flt}' "
                f"({len(self.all_grasp_dirs)} total before filter)"
            )
            # Hide robots
            for n in HAND_URDFS:
                if n in self.robot_dict:
                    self.robot_dict[n].set_visibility(False)
                if f"{n}__pregrasp" in self.robot_dict:
                    self.robot_dict[f"{n}__pregrasp"].set_visibility(False)
            return

        self.grasp_idx_slider.disabled = False
        self.grasp_idx_slider.max = len(self.filtered_grasp_dirs) - 1
        if self.grasp_idx_slider.value >= len(self.filtered_grasp_dirs):
            self.grasp_idx_slider.value = 0
        self._load_current_grasp()

    # ── scene + grasp loading ────────────────────────────────────────────

    def clear_scene(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]
        for h in self._cuboid_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._cuboid_handles = []

    def _load_scene(self):
        self.clear_scene()
        # Object at tabletop pose
        obj_pose = _load_tabletop_pose(self.obj_path_root, self.current_obj,
                                        self.current_pose_id)
        self.obj_pose = obj_pose
        simp = os.path.join(self.obj_path_root, self.current_obj, "processed_data",
                             "mesh", "simplified.obj")
        raw = os.path.join(self.obj_path_root, self.current_obj, "raw_mesh",
                            f"{self.current_obj}.obj")
        if os.path.exists(raw):
            mesh = trimesh.load(raw, force="mesh", process=False)
        elif os.path.exists(simp):
            mesh = trimesh.load(simp, force="mesh")
        else:
            print(f"[scene] no mesh for {self.current_obj}")
            return
        self.add_object("target", mesh, obj_T=obj_pose)

        # Table + floor (matches compute_open.py world)
        for name, info, rgba in (("table", DEFAULT_TABLE, SCENE_RGBA["table"]),
                                  ("floor", DEFAULT_FLOOR, SCENE_RGBA["floor"])):
            box = trimesh.creation.box(extents=info["dims"])
            cpose = np.eye(4)
            cpose[:3, 3] = info["pose"][:3]
            q = info["pose"][3:]  # wxyz
            cpose[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            handle = self.server.scene.add_mesh_simple(
                name=f"/objects/{name}",
                vertices=np.array(box.vertices, dtype=np.float32),
                faces=np.array(box.faces, dtype=np.uint32),
                color=rgba[:3],
                opacity=rgba[3] / 255.0,
                flat_shading=True,
                side="double",
                wxyz=Rot.from_matrix(cpose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=cpose[:3, 3],
            )
            self._cuboid_handles.append(handle)

    def _update_robot_pose(self, hand_name, pose, qpos=None):
        robot = self.robot_dict.get(hand_name)
        if robot is None:
            return
        robot._visual_root_frame.position = pose[:3, 3]
        robot._visual_root_frame.wxyz = Rot.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if qpos is not None:
            try:
                robot.update_cfg(np.asarray(qpos, dtype=np.float32))
            except Exception as ex:
                print(f"[update_cfg] {hand_name}: {ex}")

    def _load_current_grasp(self):
        if not self.filtered_grasp_dirs:
            return
        gi = self.grasp_idx_slider.value
        if gi >= len(self.filtered_grasp_dirs):
            gi = 0
        gid = self.filtered_grasp_dirs[gi]
        self.grasp_id_label.value = str(gid)
        grasp_path = os.path.join(self.candidates_root, self.current_hand,
                                   self.current_version, self.current_obj,
                                   self.current_scene_type, self.current_scene_idx, gid)

        wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
        pregrasp = np.load(os.path.join(grasp_path, "pregrasp_pose.npy"))
        op_path = os.path.join(grasp_path, f"openpose_{self.current_pose_id}.npy")
        if not os.path.exists(op_path):
            self.metric_text.value = f"no openpose for pose {self.current_pose_id} @ {gid}"
            return
        openpose = np.load(op_path)
        robot_T = self.obj_pose @ wrist_se3

        # Show only current hand; hide pregrasp overlay clones (deprecated).
        for hn in HAND_URDFS:
            for n in (hn, f"{hn}__pregrasp"):
                if n in self.robot_dict and n != self.current_hand:
                    self.robot_dict[n].set_visibility(False)

        # Linear blend between pregrasp (alpha=0) and openpose (alpha=1).
        alpha = float(self.pose_blend.value)
        blended = (1.0 - alpha) * np.asarray(pregrasp, dtype=np.float32) \
                  + alpha * np.asarray(openpose, dtype=np.float32)
        self._update_robot_pose(self.current_hand, robot_T, blended)
        if self.current_hand in self.robot_dict:
            self.robot_dict[self.current_hand].set_visibility(True)

        # Metrics
        key = (self.current_pose_id, self.current_scene_type,
               self.current_scene_idx, gid)
        v = self.validity_index.get(key) if self.validity_index else None
        v_str = "Safe" if v is True else ("UNSAFE" if v is False else "Unknown")
        self.metric_text.value = (
            f"obj={self.current_obj}  pose={self.current_pose_id}  "
            f"scene={self.current_scene_type}/{self.current_scene_idx}  "
            f"grasp={gid}  [{gi+1}/{len(self.filtered_grasp_dirs)}]  → {v_str}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", default=DEFAULT_OBJ_PATH)
    parser.add_argument("--candidates_root",
                        default=os.path.join(os.path.expanduser("~"), "AutoDex", "candidates"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    vis = OpenPoseBrowser(args.obj_path, args.candidates_root)
    vis.start_viewer()
