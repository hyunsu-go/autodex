"""
Polar placement preview.

Visualize the (r, theta) reachability grid: xarm at base, and the object
placed at every (r, theta) combination so we can sanity-check the
parameterization before running IK.

    python src/validation/planning/preview_polar_grid.py --obj attached_container --hand inspire_left
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path, project_dir
from autodex.utils.robot_config import INIT_STATE, XARM_INIT, INSPIRE_INIT
from autodex.utils.conversion import se32cart


TABLE_DIMS = [2, 3, 0.2]
TABLE_POSE_XYZ = [1.1, 0, -0.1]
COLOR_TABLE = (0.94, 0.94, 0.96, 0.6)

HAND_URDF = {
    "allegro": os.path.join(project_dir, "content", "assets", "robot",
                            "allegro_description", "xarm_allegro.urdf"),
    "inspire": os.path.join(project_dir, "content", "assets", "robot",
                            "inspire_description", "xarm_inspire.urdf"),
    "inspire_left": os.path.join(project_dir, "content", "assets", "robot",
                                 "inspire_left_description", "xarm_inspire_left.urdf"),
}


def place_pose(tabletop_pose: np.ndarray, r: float, theta_rad: float) -> np.ndarray:
    """Polar (r, theta) placement: only POSITION rotates around base z; orientation stays."""
    p = tabletop_pose.copy()
    p[0, 3] += r
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p[:3, 3] = Rz @ p[:3, 3]
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="attached_container")
    parser.add_argument("--hand", default="inspire_left", choices=list(HAND_URDF.keys()))
    parser.add_argument("--r_min", type=float, default=0.20)
    parser.add_argument("--r_max", type=float, default=0.60)
    parser.add_argument("--r_step", type=float, default=0.10)
    parser.add_argument("--theta_step", type=float, default=30.0, help="degrees")
    parser.add_argument("--version", default="selected_100",
                        help="Grasp candidate version for IK check")
    parser.add_argument("--reach_json", default=None,
                        help="Precomputed reachability JSON "
                             "(default: outputs/reachability/<hand>/<obj>/reachability_<version>.json)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    radii = np.arange(args.r_min, args.r_max + 1e-6, args.r_step).round(3)
    thetas_deg = np.arange(0, 360, args.theta_step)
    print(f"Radii (m): {radii.tolist()}")
    print(f"Thetas (deg): {thetas_deg.tolist()}")
    print(f"Total placements: {len(radii) * len(thetas_deg)}")

    tabletop_dir = os.path.join(obj_path, args.obj, "processed_data", "info", "tabletop")
    pose_indices = sorted(f.replace(".npy", "") for f in os.listdir(tabletop_dir)
                          if f.endswith(".npy"))
    if not pose_indices:
        raise FileNotFoundError(f"No tabletop poses in {tabletop_dir}")
    print(f"Tabletop poses: {pose_indices}")

    # Prefer textured raw_mesh; bake texture into vertex colors so viser renders it.
    raw_mesh_path = os.path.join(obj_path, args.obj, "raw_mesh", f"{args.obj}.obj")
    simp_mesh_path = os.path.join(obj_path, args.obj, "processed_data", "mesh", "simplified.obj")
    mesh_path = raw_mesh_path if os.path.exists(raw_mesh_path) else simp_mesh_path
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    obj_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    try:
        color_visual = obj_mesh.visual.to_color()
        obj_mesh.visual = color_visual
    except Exception:
        pass

    vis = ViserViewer(port_number=args.port)

    # Robot at INIT
    vis.add_robot("robot", HAND_URDF[args.hand])
    if args.hand.startswith("inspire"):
        init = np.concatenate([XARM_INIT, INSPIRE_INIT]).astype(np.float32)
    else:
        init = INIT_STATE
    vis.robot_dict["robot"].update_cfg(init)

    # Table
    table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
    table_pose = np.eye(4)
    table_pose[:3, 3] = TABLE_POSE_XYZ
    vis.add_object("table", table_mesh, table_pose)
    vis.change_color("table", COLOR_TABLE)

    # Pre-add object placements (poses updated on tabletop change)
    obj_names = []
    for ri in range(len(radii)):
        for ti in range(len(thetas_deg)):
            name = f"obj_r{ri}_t{ti}"
            vis.add_object(name, obj_mesh, np.eye(4))
            obj_names.append((name, ri, ti))

    def refresh(pose_idx_str: str):
        tabletop_pose = np.load(os.path.join(tabletop_dir, f"{pose_idx_str}.npy"))
        for name, ri, ti in obj_names:
            pose = place_pose(tabletop_pose, float(radii[ri]),
                              float(np.radians(thetas_deg[ti])))
            frame = vis.frame_nodes[name]
            frame.position = pose[:3, 3].astype(np.float32)
            from scipy.spatial.transform import Rotation as Rot
            wxyz = Rot.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            frame.wxyz = wxyz.astype(np.float32)

    # Radial guide rings
    for r in radii:
        n_seg = 96
        phis = np.linspace(0, 2 * np.pi, n_seg + 1)
        pts = np.stack([r * np.cos(phis), r * np.sin(phis),
                        np.full_like(phis, TABLE_POSE_XYZ[2] + TABLE_DIMS[2] / 2)], axis=1)
        vis.server.scene.add_spline_catmull_rom(
            f"/ring/r{int(r * 100)}",
            positions=pts.astype(np.float32),
            color=(0.5, 0.5, 0.5),
            line_width=2.0,
        )

    pose_radio = vis.server.gui.add_dropdown(
        "Tabletop Pose", options=tuple(pose_indices), initial_value=pose_indices[0],
    )

    # ── Precomputed IK reachability overlay ──────────────────────────────────
    reach_json_path = args.reach_json
    if reach_json_path is None:
        reach_json_path = os.path.join(
            "outputs", "reachability", args.hand, args.obj,
            f"reachability_{args.version}.json",
        )

    reach_aggregate = {}  # (pose_idx, r, theta_deg) -> rate in [0,1]
    reach_per_grasp = {}  # grasp_key -> {(pose_idx, r, theta_deg) -> count in [0, n_trials]}
    reach_n_trials = 1
    grasp_keys = []
    viz_index = {}  # (pose_idx, r, theta_deg) -> {"qpos_list": [...], "obj_pose": [...]}
    if os.path.exists(reach_json_path):
        with open(reach_json_path) as f:
            reach = json.load(f)
        reach_n_trials = reach.get("n_trials", 1)
        max_ik = max((e["ik_mean"] for e in reach["grid"]), default=1.0)
        max_ik = max(max_ik, 1.0)
        grasp_key_set = set()
        for entry in reach["grid"]:
            key = (entry["pose_idx"], round(float(entry["r"]), 3),
                   float(entry["theta_deg"]))
            reach_aggregate[key] = entry["ik_mean"] / max_ik
            for gk, c in entry.get("per_grasp_succ", {}).items():
                reach_per_grasp.setdefault(gk, {})[key] = c
                grasp_key_set.add(gk)
        grasp_keys = sorted(grasp_key_set, key=lambda s: tuple(s.split("/")))
        print(f"Loaded reachability ({len(reach_aggregate)} cells, "
              f"{len(grasp_keys)} grasps) from {reach_json_path}, max ik_mean={max_ik:.1f}")

        viz_path = reach_json_path.replace(".json", "_viz.json")
        if os.path.exists(viz_path):
            with open(viz_path) as f:
                viz = json.load(f)
            for e in viz:
                k = (e["pose_idx"], round(float(e["r"]), 3),
                     float(e["theta_deg"]))
                viz_index[k] = {
                    "per_grasp_qpos": e.get("per_grasp_qpos", {}),
                    "obj_pose": e["obj_pose"],
                }
            print(f"Loaded viz qpos for {len(viz_index)} cells from {viz_path}")
    else:
        print(f"[no reachability data at {reach_json_path}]")
    has_data = bool(reach_aggregate)

    ik_status = vis.server.gui.add_text(
        "IK", initial_value=("loaded" if has_data else "no data"), disabled=True,
    )
    mode_dropdown = vis.server.gui.add_dropdown(
        "Mode",
        options=("Object", "Grasp", "Texture") if has_data else ("Texture",),
        initial_value="Object" if has_data else "Texture",
    )
    grasp_slider = vis.server.gui.add_slider(
        "Grasp #",
        min=0, max=max(len(grasp_keys) - 1, 0), step=1, initial_value=0,
    )
    grasp_label = vis.server.gui.add_text(
        "Grasp info",
        initial_value=(grasp_keys[0] if grasp_keys else "(none)"),
        disabled=True,
    )

    class _GraspRef:
        @property
        def value(self):
            return grasp_keys[grasp_slider.value] if grasp_keys else "(none)"

    grasp_dropdown = _GraspRef()

    def _color_for_rate(rate: float):
        if rate >= 0.5:
            t = (rate - 0.5) * 2
            return (1.0 - t, 1.0, 0.0)
        t = rate * 2
        return (1.0, t, 0.0)

    # Plain (no-vertex-color) copy for uniform IK coloring.
    obj_mesh_plain = trimesh.Trimesh(vertices=np.asarray(obj_mesh.vertices),
                                     faces=np.asarray(obj_mesh.faces), process=False)

    def _remove_old_mesh(name: str):
        try:
            vis.obj_dict[name]["handle"].remove()
        except Exception:
            pass

    def restore_texture():
        for name, _, _ in obj_names:
            _remove_old_mesh(name)
            handle = vis.server.scene.add_mesh_trimesh(
                name=f"/objects/{name}_frame/{name}",
                mesh=obj_mesh,
            )
            vis.obj_dict[name]["handle"] = handle

    verts_np = np.asarray(obj_mesh_plain.vertices, dtype=np.float32)
    faces_np = np.asarray(obj_mesh_plain.faces, dtype=np.uint32)

    def _set_color(name: str, color):
        _remove_old_mesh(name)
        rgb255 = tuple(int(round(c * 255)) for c in color[:3])
        handle = vis.server.scene.add_mesh_simple(
            name=f"/objects/{name}_frame/{name}",
            vertices=verts_np,
            faces=faces_np,
            color=rgb255,
            opacity=0.95,
        )
        vis.obj_dict[name]["handle"] = handle

    def color_by_reachability(pose_idx_str: str, mode: str, grasp_key: str):
        if not has_data:
            return
        if mode == "Object":
            cell_data = reach_aggregate
            n_match = 0
            for name, ri, ti in obj_names:
                key = (pose_idx_str, round(float(radii[ri]), 3),
                       float(thetas_deg[ti]))
                color = _color_for_rate(cell_data[key]) if key in cell_data else (0.4, 0.4, 0.4)
                _set_color(name, color)
                if key in cell_data:
                    n_match += 1
            ik_status.value = f"Object  pose={pose_idx_str}  {n_match}/{len(obj_names)} cells"
        else:  # Grasp
            gd = reach_per_grasp.get(grasp_key, {})
            n_match = 0
            for name, ri, ti in obj_names:
                key = (pose_idx_str, round(float(radii[ri]), 3),
                       float(thetas_deg[ti]))
                count = gd.get(key, 0)
                rate = count / max(reach_n_trials, 1)
                if count > 0:
                    n_match += 1
                    color = _color_for_rate(rate)
                else:
                    color = (0.25, 0.25, 0.25)  # gray = grasp not reachable here
                _set_color(name, color)
            ik_status.value = (f"Grasp [{grasp_key}]  pose={pose_idx_str}  "
                               f"reachable in {n_match}/{len(obj_names)} cells")

    # ── r / theta / IK sliders for cell selection ────────────────────────────
    r_slider = vis.server.gui.add_slider(
        "r index", min=0, max=max(len(radii) - 1, 0), step=1, initial_value=0,
    )
    r_label = vis.server.gui.add_text(
        "r (m)", initial_value=f"{radii[0]:.2f}", disabled=True,
    )
    theta_slider = vis.server.gui.add_slider(
        "theta index", min=0, max=max(len(thetas_deg) - 1, 0), step=1, initial_value=0,
    )
    theta_label = vis.server.gui.add_text(
        "theta (deg)", initial_value=f"{thetas_deg[0]:.0f}", disabled=True,
    )
    ik_slider = vis.server.gui.add_slider(
        "IK Solution #", min=0, max=1, step=1, initial_value=0,
    )
    cell_label = vis.server.gui.add_text("Cell info", initial_value="", disabled=True)

    def _cell_key(pose_idx_str: str, ri: int, ti: int):
        return (pose_idx_str, round(float(radii[ri]), 3), float(thetas_deg[ti]))

    def _name_for(ri: int, ti: int):
        return f"obj_r{ri}_t{ti}"

    selected_name = {"current": None}  # (ri, ti) or None

    def _restore_one_cell_color(ri: int, ti: int,
                                 pose_idx_str: str, mode: str, grasp_key: str):
        name = _name_for(ri, ti)
        key = _cell_key(pose_idx_str, ri, ti)
        if mode == "Object" and key in reach_aggregate:
            _set_color(name, _color_for_rate(reach_aggregate[key]))
        elif mode == "Grasp":
            gd = reach_per_grasp.get(grasp_key, {})
            count = gd.get(key, 0)
            rate = count / max(reach_n_trials, 1)
            color = _color_for_rate(rate) if count > 0 else (0.25, 0.25, 0.25)
            _set_color(name, color)
        elif mode == "Texture":
            _remove_old_mesh(name)
            handle = vis.server.scene.add_mesh_trimesh(
                name=f"/objects/{name}_frame/{name}", mesh=obj_mesh,
            )
            vis.obj_dict[name]["handle"] = handle

    def update_robot_from_slider():
        pose_idx_str = pose_radio.value
        ri = min(r_slider.value, len(radii) - 1)
        ti = min(theta_slider.value, len(thetas_deg) - 1)
        r_label.value = f"{radii[ri]:.2f}"
        theta_label.value = f"{thetas_deg[ti]:.0f}"
        name = _name_for(ri, ti)
        key = _cell_key(pose_idx_str, ri, ti)
        mode = mode_dropdown.value
        grasp_key = grasp_dropdown.value

        prev = selected_name["current"]
        if prev is not None and prev != (ri, ti):
            _restore_one_cell_color(prev[0], prev[1], pose_idx_str, mode, grasp_key)
        if mode != "Texture":
            _set_color(name, (0.2, 0.5, 1.0))
        selected_name["current"] = (ri, ti)

        info = viz_index.get(key)
        if info is None:
            cell_label.value = (f"r={radii[ri]:.2f} θ={thetas_deg[ti]:.0f}°  "
                                f"(no IK qpos)")
            vis.robot_dict["robot"].update_cfg(init)
            ik_slider.max = 1
            return
        pgq = info["per_grasp_qpos"]

        if mode == "Grasp":
            qpos = pgq.get(grasp_key)
            if qpos is None:
                cell_label.value = (f"r={radii[ri]:.2f} θ={thetas_deg[ti]:.0f}°  "
                                    f"[{grasp_key}] NOT reachable here")
                vis.robot_dict["robot"].update_cfg(init)
                ik_slider.max = 1
                return
            vis.robot_dict["robot"].update_cfg(np.asarray(qpos, dtype=np.float32))
            ik_slider.max = 1
            cell_label.value = (f"r={radii[ri]:.2f} θ={thetas_deg[ti]:.0f}°  "
                                f"[{grasp_key}]")
        else:  # Object / Texture: iterate all grasps reachable at this cell
            keys_sorted = sorted(pgq.keys())
            n_q = len(keys_sorted)
            if n_q == 0:
                cell_label.value = (f"r={radii[ri]:.2f} θ={thetas_deg[ti]:.0f}°  "
                                    f"(no IK)")
                vis.robot_dict["robot"].update_cfg(init)
                ik_slider.max = 1
                return
            ik_slider.max = max(n_q - 1, 1)
            qi = min(ik_slider.value, n_q - 1)
            gk = keys_sorted[qi]
            qpos = pgq[gk]
            vis.robot_dict["robot"].update_cfg(np.asarray(qpos, dtype=np.float32))
            cell_label.value = (f"r={radii[ri]:.2f} θ={thetas_deg[ti]:.0f}°  "
                                f"IK {qi+1}/{n_q} = [{gk}]")

    def apply_view():
        pose_idx_str = pose_radio.value
        # Drop highlight memory so it's not reapplied with stale mode.
        selected_name["current"] = None
        if mode_dropdown.value == "Texture":
            restore_texture()
            ik_status.value = "Texture mode"
        else:
            color_by_reachability(pose_idx_str, mode_dropdown.value, grasp_dropdown.value)
        update_robot_from_slider()

    @pose_radio.on_update
    def _(_):
        refresh(pose_radio.value)
        apply_view()

    @mode_dropdown.on_update
    def _(_):
        apply_view()

    @grasp_slider.on_update
    def _(_):
        if grasp_keys:
            grasp_label.value = grasp_keys[grasp_slider.value]
        apply_view()

    @r_slider.on_update
    def _(_):
        update_robot_from_slider()

    @theta_slider.on_update
    def _(_):
        update_robot_from_slider()

    @ik_slider.on_update
    def _(_):
        update_robot_from_slider()

    refresh(pose_indices[0])
    apply_view()
    vis.add_floor(0.0)
    vis.start_viewer()


if __name__ == "__main__":
    main()
