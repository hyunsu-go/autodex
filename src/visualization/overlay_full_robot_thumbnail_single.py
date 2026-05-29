"""Render lift-moment overlays with the FULL xarm+hand (not floating hand).

For each camera, seek to the video frame closest to a specified execution state
timestamp (default: pregrasp), use the ACTUAL arm/state.npy and hand/state.npy
qpos at that frame to do FK on `xarm_{hand}.urdf`, and overlay every link plus
the object onto the frame. Saves 24 per-camera PNGs + 1 grid PNG.

Each finger gets its own color; the arm gets a separate color.

Prints `[thumb_progress] N/TOTAL` lines for parent tqdm parsing.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))
from paradex.image.projection import intr_opencv_to_opengl_proj
from paradex.image.grid import make_image_grid
from paradex.visualization.robot import RobotModule


ROBOT_VIDEO_OFFSET_S = 0.03
ALPHA_LINK = 0.55
ALPHA_OBJ = 0.5

FINGER_COLORS_BGR = {
    "thumb":  (  0, 140, 255),  # orange
    "index":  (255, 200,   0),  # cyan
    "middle": (100, 255,   0),  # lime
    "ring":   (200,   0, 255),  # magenta
    "pinky":  (  0, 220, 255),  # yellow
}
ARM_COLOR_BGR = (180, 180, 180)         # light gray
OBJECT_BGR = (255, 80, 80)              # blue
HAND_BASE_BGR = (140, 140, 140)         # slightly darker gray

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
        return ARM_COLOR_BGR
    if link_name in HAND_BASE_NAMES:
        return HAND_BASE_BGR
    if link_name in ALLEGRO_LINK_LABELS:
        return FINGER_COLORS_BGR[ALLEGRO_LINK_LABELS[link_name]]
    for prefix, label in FINGER_PREFIX_MAP.items():
        if link_name.startswith(prefix):
            return FINGER_COLORS_BGR[label]
    return ARM_COLOR_BGR  # fallback


_GLCAM_IN_CVCAM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)


class MultiMeshOverlayRenderer:
    def __init__(self, link_meshes, link_colors_bgr, link_alphas,
                 intrinsics, extrinsics_cw, H, W, device="cuda"):
        self.device = device
        self.serials = sorted(intrinsics.keys())
        self.N = len(self.serials)
        self.H, self.W = H, W
        self.glctx = dr.RasterizeCudaContext()

        glcam = torch.from_numpy(_GLCAM_IN_CVCAM).to(device)
        cam_extrs, proj_list = [], []
        for s in self.serials:
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :] = extrinsics_cw[s][:3, :]
            cam_extrs.append(torch.from_numpy(ext).to(device))
            proj = intr_opencv_to_opengl_proj(intrinsics[s], W, H, near=0.01, far=5).astype(np.float32)
            proj_list.append(torch.from_numpy(proj).to(device))
        self.mtx = (torch.stack(proj_list) @ glcam[None] @ torch.stack(cam_extrs)).contiguous()

        per_v, per_f, per_lid = [], [], []
        v_off = 0
        self.link_vert_ranges = []
        for i, mesh in enumerate(link_meshes, start=1):
            v = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
            f = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
            nv = v.shape[0]
            per_v.append(v)
            per_f.append(f + v_off)
            per_lid.append(torch.full((nv,), float(i), dtype=torch.float32, device=device))
            self.link_vert_ranges.append((v_off, v_off + nv))
            v_off += nv

        self.base_verts = torch.cat(per_v, dim=0)
        self.faces = torch.cat(per_f, dim=0)
        self.vert_lid = torch.cat(per_lid, dim=0)[:, None]
        self.n_links = len(link_meshes)
        self.V = v_off

        color_lut = np.zeros((self.n_links + 1, 3), dtype=np.float32)
        alpha_lut = np.zeros((self.n_links + 1,), dtype=np.float32)
        for i, (bgr, a) in enumerate(zip(link_colors_bgr, link_alphas), start=1):
            color_lut[i] = bgr
            alpha_lut[i] = a
        self.color_lut = torch.from_numpy(color_lut).to(device)
        self.alpha_lut = torch.from_numpy(alpha_lut).to(device)[:, None]

    def render(self, link_poses, frames_bgr_list):
        device = self.device
        poses = torch.as_tensor(np.stack(link_poses), dtype=torch.float32, device=device)
        verts_world = torch.empty((self.V, 3), dtype=torch.float32, device=device)
        for i, (start, end) in enumerate(self.link_vert_ranges):
            v = self.base_verts[start:end]
            v_h = torch.cat([v, torch.ones(v.shape[0], 1, device=device)], dim=1)
            verts_world[start:end] = (v_h @ poses[i].T)[:, :3]

        v_homo = torch.cat([verts_world, torch.ones(self.V, 1, device=device)], dim=1)
        pos_clip = torch.einsum("nij,vj->nvi", self.mtx, v_homo).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, self.faces, resolution=(self.H, self.W))
        id_map, _ = dr.interpolate(self.vert_lid, rast_out, self.faces)
        id_map = torch.flip(id_map, dims=[1])
        ids = torch.clamp(torch.round(id_map[..., 0]).long(), 0, self.n_links)
        colors = self.color_lut[ids]
        alphas = self.alpha_lut[ids]

        frames_np = np.stack(frames_bgr_list)
        frames_gpu = torch.from_numpy(frames_np).to(device).float()
        overlay = frames_gpu * (1.0 - alphas) + colors * alphas
        overlay_u8 = overlay.clamp(0, 255).to(torch.uint8).cpu().numpy()
        return [overlay_u8[i] for i in range(self.N)]


def load_cam_param(cam_param_dir):
    intr_raw = json.load(open(cam_param_dir / "intrinsics.json"))
    extr_raw = json.load(open(cam_param_dir / "extrinsics.json"))
    intrinsics, extrinsics = {}, {}
    for s in intr_raw:
        intrinsics[s] = np.array(intr_raw[s]["intrinsics_undistort"], dtype=np.float32).reshape(3, 3)
        T = np.array(extr_raw[s], dtype=np.float64).reshape(-1)
        T = np.vstack([T.reshape(3, 4), [0, 0, 0, 1]]) if T.size == 12 else T.reshape(4, 4)
        extrinsics[s] = T
    return intrinsics, extrinsics


def find_state_frame(result_json_path, timestamps_npy_path, state_name):
    with open(result_json_path) as f:
        result = json.load(f)
    target_iso = None
    for s in result["timing"]["execution_states"]:
        if s["state"] == state_name:
            target_iso = s["time"]
            break
    if target_iso is None:
        raise RuntimeError(f"no {state_name} state in {result_json_path}")
    target_epoch = datetime.fromisoformat(target_iso).timestamp() + ROBOT_VIDEO_OFFSET_S
    ts = np.load(timestamps_npy_path)
    idx = int(np.argmin(np.abs(ts - target_epoch)))
    dt = float(ts[idx] - target_epoch)
    return idx, dt, target_iso, ts[idx]


def read_frame(video_path, frame_idx, H, W):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    return frame


def build_full_robot_link_meshes(hand_type, arm_qpos, hand_qpos, c2r):
    """FK on xarm_{hand}.urdf at the trial's actual arm+hand qpos.

    FK is in the robot frame; pre-multiply by C2R so link poses live in the
    camera-world frame (same frame as pose_world / extrinsics).

    Returns: (link_meshes, link_poses_world, link_colors_bgr, link_alphas)."""
    # Inspire xarm needs the pre-calibration URDF; allegro uses current.
    urdf_name = f"xarm_{hand_type}.urdf.bak" if hand_type == "inspire" else f"xarm_{hand_type}.urdf"
    urdf_path = (Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
                 / f"{hand_type}_description" / urdf_name)
    robot = RobotModule(str(urdf_path))
    n_dof = robot.num_joints

    qpos = np.zeros(n_dof, dtype=np.float64)
    n_arm = len(arm_qpos)
    n_hand = len(hand_qpos)
    qpos[:n_arm] = arm_qpos
    qpos[n_arm:n_arm + n_hand] = hand_qpos
    cfg = {name: angle for name, angle in zip(robot.joint_names[:n_arm + n_hand],
                                              qpos[:n_arm + n_hand])}
    robot.update_cfg(cfg)

    scene = robot.scene
    meshes, poses, colors, alphas = [], [], [], []
    for ln in list(scene.geometry.keys()):
        T_robot = scene.graph.get(ln)[0]
        T_world = c2r @ T_robot
        meshes.append(scene.geometry[ln])
        poses.append(T_world)
        colors.append(link_color(ln))
        alphas.append(ALPHA_LINK)
    return meshes, poses, colors, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--cam_param_dir", required=True)
    ap.add_argument("--arm_state", required=True, help="path to arm/state.npy")
    ap.add_argument("--hand_state", required=True, help="path to hand/state.npy")
    ap.add_argument("--pose_world", required=True)
    ap.add_argument("--object_mesh", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--hand", required=True, choices=["allegro", "inspire"])
    ap.add_argument("--c2r", required=True, help="path to C2R.npy")
    ap.add_argument("--state", default="pregrasp",
                    help="execution state to snapshot (default: pregrasp)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    cam_param_dir = Path(args.cam_param_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = load_cam_param(cam_param_dir)
    available = {p.stem for p in videos_dir.glob("*.avi")}
    serials = sorted(s for s in intrinsics if s in available)
    intrinsics = {s: intrinsics[s] for s in serials}
    extrinsics = {s: extrinsics[s] for s in serials}
    if not serials:
        print("[error] no serials match between cam_param and videos", flush=True)
        sys.exit(1)

    frame_idx, dt, target_iso, _ = find_state_frame(
        args.result_json, args.timestamps, args.state)
    print(f"[thumb] {args.state}={target_iso} frame_idx={frame_idx} dt={dt*1000:.1f}ms",
          flush=True)
    c2r = np.load(args.c2r)
    if c2r.shape == (3, 4):
        c2r = np.vstack([c2r, [0, 0, 0, 1]])

    # arm/hand state.npy is sampled per-camera-frame; index aligns with frame_idx.
    arm_state = np.load(args.arm_state)
    hand_state = np.load(args.hand_state)
    if frame_idx >= len(arm_state) or frame_idx >= len(hand_state):
        print(f"[error] frame_idx {frame_idx} out of range for state arrays "
              f"(arm={len(arm_state)}, hand={len(hand_state)})", flush=True)
        sys.exit(1)
    arm_q = arm_state[frame_idx]
    hand_q = hand_state[frame_idx]

    pose_world = np.load(args.pose_world)

    cap0 = cv2.VideoCapture(str(videos_dir / f"{serials[0]}.avi"))
    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    link_meshes, link_poses, link_colors, link_alphas = build_full_robot_link_meshes(
        args.hand, arm_q, hand_q, c2r)

    obj_mesh = trimesh.load(args.object_mesh, process=False)
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = trimesh.util.concatenate(list(obj_mesh.geometry.values()))

    all_meshes = link_meshes + [obj_mesh]
    all_poses = link_poses + [pose_world]
    all_colors = link_colors + [OBJECT_BGR]
    all_alphas = link_alphas + [ALPHA_OBJ]

    renderer = MultiMeshOverlayRenderer(all_meshes, all_colors, all_alphas,
                                        intrinsics, extrinsics, H, W)
    ordered = renderer.serials

    total = len(ordered) + 1
    frames = []
    for i, s in enumerate(ordered):
        frames.append(read_frame(videos_dir / f"{s}.avi", frame_idx, H, W))
        print(f"[thumb_progress] {i+1}/{total}", flush=True)

    overlays = renderer.render(all_poses, frames)
    for s, img in zip(ordered, overlays):
        cv2.imwrite(str(out_dir / f"thumb_{s}.png"), img)

    grid = make_image_grid([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in overlays])
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "thumb_grid.png"), grid_bgr)
    print(f"[thumb_progress] {total}/{total}", flush=True)
    print(f"[thumb] wrote {len(ordered)} thumbs + grid to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
