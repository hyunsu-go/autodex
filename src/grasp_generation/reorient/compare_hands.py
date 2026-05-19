"""
Overlap visualization: BODex inspire-left floating hand vs same hand with
xarm-URDF mimic values applied. Identical mesh/origin/axis — only mimic
relationships differ. Same slider input drives both; finger shapes diverge.

Usage:
    python src/grasp_generation/reorient/compare_hands.py --port 8081
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np
import yourdfpy

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer


BODEX_FLOATING = (
    "/home/mingi/AutoDex/src/grasp_generation/BODex/src/curobo/content/assets/"
    "robot/inspire_description/inspire_left_floating.urdf"
)

XARM_MIMICS = {
    "left_thumb_3_joint": ("left_thumb_2_joint", 1.114, 0.0),
    "left_thumb_4_joint": ("left_thumb_2_joint", 0.62, 0.0),
    "left_index_2_joint": ("left_index_1_joint", 0.67, 0.22),
    "left_middle_2_joint": ("left_middle_1_joint", 0.88, 0.15),
    "left_ring_2_joint": ("left_ring_1_joint", 1.05, 0.0),
    "left_little_2_joint": ("left_little_1_joint", 0.9889, -0.03126),
}

ACT_FINGER_JOINTS = [
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_index_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_little_1_joint",
]

JOINT_LIMITS = {
    "left_thumb_1_joint":  (0.0, 1.15),
    "left_thumb_2_joint":  (0.0, 0.55),
    "left_index_1_joint":  (0.0, 1.60),
    "left_middle_1_joint": (0.0, 1.60),
    "left_ring_1_joint":   (0.0, 1.60),
    "left_little_1_joint": (0.0, 1.60),
}


def make_xarm_mimic_urdf(src_path: str) -> str:
    with open(src_path) as f:
        urdf = f.read()
    for joint_name, (driver, mult, offset) in XARM_MIMICS.items():
        pattern = (
            r'(<joint name="' + re.escape(joint_name) + r'"[^>]*>'
            r'(?:[^<]|<(?!/joint>))*?<mimic\s+)'
            r'joint="[^"]*"\s+multiplier="[^"]*"\s+offset="[^"]*"'
        )
        replacement = (r'\1joint="' + driver + r'" multiplier="' + f"{mult}"
                       + r'" offset="' + f"{offset}" + r'"')
        urdf = re.sub(pattern, replacement, urdf, count=1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_xarmmimic.urdf", delete=False,
        dir=os.path.dirname(src_path),
    )
    tmp.write(urdf)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    xarm_urdf_path = make_xarm_mimic_urdf(BODEX_FLOATING)

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("bodex", BODEX_FLOATING)
    vis.add_robot("xarm", xarm_urdf_path)

    # uint8 colors so viser MeshHandle.color receives the format it was
    # initialized with (vertex_colors from trimesh = uint8 0-255).
    vis.change_color("bodex", (255, 70, 70))     # red
    vis.change_color("xarm",  (70, 130, 255))    # blue

    bodex_urdf = yourdfpy.URDF.load(BODEX_FLOATING, build_scene_graph=True)
    xarm_urdf = yourdfpy.URDF.load(xarm_urdf_path, build_scene_graph=True)

    sliders = {}
    with vis.server.gui.add_folder("Finger joints"):
        for j in ACT_FINGER_JOINTS:
            lo, hi = JOINT_LIMITS[j]
            sliders[j] = vis.server.gui.add_slider(
                j.replace("left_", "").replace("_joint", ""),
                min=lo, max=hi, step=0.01, initial_value=0.0,
            )
    reset_btn = vis.server.gui.add_button("Open all (0)")
    pregrasp_btn = vis.server.gui.add_button("BODex sample pregrasp")
    vis.server.gui.add_markdown("**Red = BODex mimic** &nbsp; **Blue = xarm mimic**")

    def build_cfg(urdf, fv):
        return np.array([fv.get(n, 0.0) for n in urdf.actuated_joint_names],
                        dtype=np.float32)

    def push():
        fv = {j: s.value for j, s in sliders.items()}
        vis.robot_dict["bodex"].update_cfg(build_cfg(bodex_urdf, fv))
        vis.robot_dict["xarm"].update_cfg(build_cfg(xarm_urdf, fv))

    for s in sliders.values():
        @s.on_update
        def _(_): push()

    @reset_btn.on_click
    def _(_):
        for s in sliders.values():
            s.value = 0.0
        push()

    @pregrasp_btn.on_click
    def _(_):
        vals = [0.64, 0.13, 0.52, 0.72, 1.01, 1.02]
        for j, v in zip(ACT_FINGER_JOINTS, vals):
            sliders[j].value = float(v)
        push()

    push()
    print(f"[compare_hands] serving on http://localhost:{args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
