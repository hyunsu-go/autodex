"""Per-grasp scene-coverage viewer.

Two sliders:
- scene_type: pick a scene_type (box / float / shelf / table / wall)
- grasp_rank: pick a grasp from setcover_order

For the selected (scene_type, grasp_rank), every scene of that type is tiled
in a 3D grid with the hand placed in world frame. Tiles are colored green
when the grasp is collision-free for that scene, red otherwise.

Requires `scene_list.json` + `valid_array_full.npy` from build_scene_list.py.

Usage:
    conda run -n mingi python src/visualization/grasp_generation/grasp_coverage.py \\
        --hand inspire_left --version v3 --obj smallbowl
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as Rot

from autodex.utils.path import obj_path, repo_dir
from autodex.utils.conversion import cart2se3

# Reuse the procedural scene builders from the existing viewer.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "mesh_process"
))
from scene_grid_viewer import build_scene  # noqa: E402


# Scene types shown in the viewer (in slider order).
SCENE_TYPES_SHOWN = ["box", "shelf", "wall"]


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
}


def _quat_wxyz_from_mat(R3):
    q = Rot.from_matrix(R3).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _load_grasps(candidate_root, obj_name):
    wrist_list, pregrasp_list, info_list = [], [], []
    root = os.path.join(candidate_root, obj_name)
    for scene_type in sorted(os.listdir(root)):
        sd = os.path.join(root, scene_type)
        if not os.path.isdir(sd):
            continue
        for scene_name in sorted(os.listdir(sd)):
            sd2 = os.path.join(sd, scene_name)
            if not os.path.isdir(sd2):
                continue
            for grasp_name in sorted(os.listdir(sd2)):
                gd = os.path.join(sd2, grasp_name)
                if not os.path.isdir(gd):
                    continue
                wrist_list.append(np.load(os.path.join(gd, "wrist_se3.npy")))
                pregrasp_list.append(np.load(os.path.join(gd, "pregrasp_pose.npy")))
                info_list.append((scene_type, scene_name, grasp_name))
    return np.array(wrist_list), np.array(pregrasp_list), info_list


def _grid_offsets(n, pitch):
    cols = max(1, int(np.ceil(np.sqrt(n * 1.4))))
    rows = int(np.ceil(n / cols))
    offsets = []
    for i in range(n):
        r, c = i // cols, i % cols
        # Center grid around origin (visually nicer)
        x = (c - (cols - 1) / 2) * pitch
        y = -(r - (rows - 1) / 2) * pitch
        offsets.append(np.array([x, y, 0.0], dtype=np.float64))
    return offsets, cols, rows


class GraspCoverageViewer:
    def __init__(self, server, hand, version, obj_name, pitch):
        self.server = server
        self.hand = hand
        self.version = version
        self.obj_name = obj_name
        self.pitch = pitch

        # --- Load order data ---
        order_dir = os.path.join(repo_dir, "order", hand, version, obj_name)
        with open(os.path.join(order_dir, "setcover_order.json")) as f:
            self.order = json.load(f)
        with open(os.path.join(order_dir, "scene_list.json")) as f:
            self.scene_list = json.load(f)
        self.valid_array = np.load(os.path.join(order_dir, "valid_array_full.npy"))

        assert self.valid_array.shape[0] == len(self.scene_list), \
            f"valid_array rows {self.valid_array.shape[0]} != scene_list {len(self.scene_list)}"

        # --- Load candidates ---
        candidate_root = os.path.join(repo_dir, "candidates", hand, version)
        self.wrist_se3, self.pregrasp, self.grasp_paths = _load_grasps(candidate_root, obj_name)
        assert len(self.wrist_se3) == self.valid_array.shape[1], \
            f"grasp count {len(self.wrist_se3)} != valid_array cols {self.valid_array.shape[1]}"

        # --- Object mesh + OBB info (used by procedural shelf/wall builders) ---
        mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
        self.obj_mesh = trimesh.load(mesh_path, force="mesh")
        obb_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
        with open(obb_path) as f:
            self.obb_info = json.load(f)

        # --- URDF + cached visual mesh data ---
        self.urdf = yourdfpy.URDF.load(
            HAND_URDFS[hand], load_meshes=True, build_collision_scene_graph=False
        )
        self.joint_names = list(self.urdf.actuated_joint_names)
        # Cache (name, vertices, faces) per visual mesh in the URDF — reused
        # across all tile instances instead of letting ViserUrdf load N times.
        self.link_visuals = []
        for vname, gmesh in self.urdf.scene.geometry.items():
            self.link_visuals.append((
                vname,
                np.asarray(gmesh.vertices, dtype=np.float32),
                np.asarray(gmesh.faces, dtype=np.uint32),
            ))

        # --- Group scenes (only wall/shelf/box) ---
        present = set(s["scene_type"] for s in self.scene_list)
        self.scene_types = [st for st in SCENE_TYPES_SHOWN if st in present]
        if not self.scene_types:
            raise RuntimeError(f"None of {SCENE_TYPES_SHOWN} present in scene_list")
        self.row_idx_by_type = {
            st: [i for i, s in enumerate(self.scene_list) if s["scene_type"] == st]
            for st in self.scene_types
        }

        # --- State ---
        self.tile_state = []  # list per tile: {row_idx, T_tile, obj_se3, hand_frame, hand_urdf, bg, handles}

        self._build_gui()
        self._on_scene_type_change()

    def _build_gui(self):
        with self.server.gui.add_folder(f"{self.obj_name} ({self.hand}/{self.version})"):
            self.st_slider = self.server.gui.add_slider(
                "Scene Type", min=0, max=len(self.scene_types) - 1, step=1, initial_value=0,
            )
            self.st_label = self.server.gui.add_text(
                "  type", initial_value=self.scene_types[0], disabled=True,
            )
            self.rank_slider = self.server.gui.add_slider(
                "Grasp Rank", min=0, max=len(self.order) - 1, step=1, initial_value=0,
            )
            self.grasp_info = self.server.gui.add_text(
                "  grasp", initial_value="--", disabled=True,
            )
            self.cover_info = self.server.gui.add_text(
                "  coverage", initial_value="--", disabled=True,
            )

        self.st_slider.on_update(lambda _: self._on_scene_type_change())
        self.rank_slider.on_update(lambda _: self._update_grasp())

    def _clear_tiles(self):
        for tile in self.tile_state:
            for h in tile["handles"]:
                try:
                    h.remove()
                except Exception:
                    pass
        self.tile_state = []
        try:
            self.server.scene.reset()
        except Exception:
            pass

    def _on_scene_type_change(self):
        self._clear_tiles()

        st_idx = int(self.st_slider.value)
        scene_type = self.scene_types[st_idx]
        self.st_label.value = scene_type

        rows = self.row_idx_by_type[scene_type]
        offsets, _, _ = _grid_offsets(len(rows), self.pitch)

        for tile_i, (row_idx, offset) in enumerate(zip(rows, offsets)):
            self._add_tile(tile_i, row_idx, offset)

        self._update_grasp()

    def _add_tile(self, tile_i, row_idx, offset):
        scene_info = self.scene_list[row_idx]
        scene_type = scene_info["scene_type"]
        scene_path = os.path.join(
            obj_path, self.obj_name, "scene", scene_type, f"{scene_info['scene_id']}.json"
        )
        with open(scene_path) as f:
            cfg_full = json.load(f)

        # Procedural scene (matches scene_grid_viewer.build_scene → walls/shelf
        # cuboids generated from object footprint rather than the JSON's only
        # table+floor).
        built = build_scene(self.obj_name, scene_type, cfg_full, self.obb_info)

        T_tile = np.eye(4)
        T_tile[:3, 3] = offset

        handles = []

        # Three overlaid plates: green (valid for current grasp), red (invalid),
        # yellow (this scene fails for ALL grasps — impossible). Toggle visible.
        # Place bg plate just below the canonical table top (z=0) so it visually
        # reads as the floor under the object.
        bg_dims = (self.pitch * 0.95, self.pitch * 0.95, 0.005)
        bg_pos = offset + np.array([0, 0, -bg_dims[2] / 2.0])
        is_impossible = bool(not self.valid_array[row_idx].any())
        bg_green = self.server.scene.add_box(
            name=f"/tiles/{tile_i}/bg_green",
            dimensions=bg_dims, position=bg_pos, color=(90, 200, 110),
        )
        bg_red = self.server.scene.add_box(
            name=f"/tiles/{tile_i}/bg_red",
            dimensions=bg_dims, position=bg_pos, color=(220, 90, 90),
        )
        bg_yellow = self.server.scene.add_box(
            name=f"/tiles/{tile_i}/bg_yellow",
            dimensions=bg_dims, position=bg_pos, color=(230, 210, 70),
        )
        bg_green.visible = False
        bg_red.visible = False
        bg_yellow.visible = is_impossible
        handles.append(bg_green)
        handles.append(bg_red)
        handles.append(bg_yellow)

        # Label
        label = self.server.scene.add_label(
            f"/tiles/{tile_i}/label",
            text=f"{scene_info['scene_type']}/{scene_info['scene_id']}",
            position=offset + np.array([0, self.pitch * 0.4, 0.0]),
        )
        handles.append(label)

        # Object (from procedurally-built scene)
        obj_se3 = cart2se3(built["mesh"]["target"]["pose"])
        T_obj_world = T_tile @ obj_se3
        obj_h = self.server.scene.add_mesh_trimesh(
            f"/tiles/{tile_i}/target", mesh=self.obj_mesh,
            position=T_obj_world[:3, 3],
            wxyz=_quat_wxyz_from_mat(T_obj_world[:3, :3]),
        )
        handles.append(obj_h)

        # Cuboids from the built scene. Skip the canonical 2×2 table backdrop
        # (Visualization/scene.py:206-207 reference pattern).
        for cub_name, info in built.get("cuboid", {}).items():
            if cub_name == "table":
                continue
            T_cub = cart2se3(info["pose"])
            T_cub_world = T_tile @ T_cub
            box = trimesh.creation.box(extents=info["dims"])
            box.visual.face_colors = np.array([140, 170, 220, 200])
            h = self.server.scene.add_mesh_trimesh(
                f"/tiles/{tile_i}/cub_{cub_name}", mesh=box,
                position=T_cub_world[:3, 3],
                wxyz=_quat_wxyz_from_mat(T_cub_world[:3, :3]),
            )
            handles.append(h)

        # Hand: parent frame moved per grasp; per-link mesh handles share mesh
        # data with all other tiles (much cheaper than N × ViserUrdf instances).
        hand_frame = self.server.scene.add_frame(
            f"/tiles/{tile_i}/hand_pose",
            show_axes=False,
            position=offset,
            wxyz=np.array([1, 0, 0, 0]),
        )
        handles.append(hand_frame)

        link_handles = {}
        for vname, verts, faces in self.link_visuals:
            h = self.server.scene.add_mesh_simple(
                name=f"/tiles/{tile_i}/hand_pose/{vname}",
                vertices=verts, faces=faces,
                color=(180, 180, 180),
                flat_shading=True,
            )
            link_handles[vname] = h
            handles.append(h)

        self.tile_state.append({
            "row_idx": row_idx,
            "T_tile": T_tile,
            "obj_se3": obj_se3,
            "hand_frame": hand_frame,
            "link_handles": link_handles,
            "bg_green": bg_green,
            "bg_red": bg_red,
            "bg_yellow": bg_yellow,
            "is_impossible": is_impossible,
            "handles": handles,
        })

    def _update_grasp(self):
        rank = int(self.rank_slider.value)
        info = self.order[rank]
        # Format-robust: last 4 fields are always (scene_type, scene_id, grasp_name, grasp_idx)
        grasp_idx = int(info[-1])
        grasp_name = str(info[-2])
        src_scene_id = str(info[-3])
        src_scene_type = str(info[-4])
        self.grasp_info.value = (
            f"rank #{rank} | source: {src_scene_type}/{src_scene_id}/{grasp_name} (idx={grasp_idx})"
        )

        wrist_obj = self.wrist_se3[grasp_idx]
        pregrasp = self.pregrasp[grasp_idx]
        cfg_dict = {jn: float(pregrasp[i]) for i, jn in enumerate(self.joint_names)}

        # Update URDF joint config once; link visual transforms are read from
        # the shared scene graph and applied per-tile.
        self.urdf.update_cfg(cfg_dict)
        link_T_local = {
            vname: np.asarray(self.urdf.scene.graph.get(vname)[0])
            for vname, _, _ in self.link_visuals
        }

        n_valid = 0
        for tile in self.tile_state:
            T_w = tile["T_tile"] @ tile["obj_se3"] @ wrist_obj
            tile["hand_frame"].position = T_w[:3, 3]
            tile["hand_frame"].wxyz = _quat_wxyz_from_mat(T_w[:3, :3])
            for vname, h in tile["link_handles"].items():
                T_link = link_T_local[vname]
                # hand_frame is positioned at T_w; link is added under hand_frame
                # → set local pose relative to hand_frame.
                h.position = T_link[:3, 3]
                h.wxyz = _quat_wxyz_from_mat(T_link[:3, :3])

            valid = bool(self.valid_array[tile["row_idx"], grasp_idx])
            n_valid += int(valid)
            if tile["is_impossible"]:
                tile["bg_yellow"].visible = True
                tile["bg_green"].visible = False
                tile["bg_red"].visible = False
            else:
                tile["bg_yellow"].visible = False
                tile["bg_green"].visible = valid
                tile["bg_red"].visible = not valid

        self.cover_info.value = f"{n_valid}/{len(self.tile_state)} valid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire_left", choices=list(HAND_URDFS))
    parser.add_argument("--version", default="v3")
    parser.add_argument("--obj", default="smallbowl")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--pitch", type=float, default=0.7,
                        help="Tile spacing (m) in the grid")
    args = parser.parse_args()

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser at http://localhost:{args.port}")

    GraspCoverageViewer(server, args.hand, args.version, args.obj, args.pitch)

    print("Viewer ready. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
