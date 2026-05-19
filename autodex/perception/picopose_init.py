"""In-process PicoPose first-frame init wrapper.

Mirrors :class:`autodex.perception.foundpose_init.FoundPoseInit` so the two
methods can be swapped behind the same call site::

    init = PicoPoseInit(mesh_path, assets_root, obj_name, device='cuda:0')
    per_view = init.estimate_per_view(images_rgb, masks_bool, intrinsics, extrinsics)
    # per_view[serial] = {pose_world (4x4 m), quality, inliers, template_id, timings}

Templates are rendered in-process via nvdiffrast (no Panda3D / BlenderProc) using
the level-1 viewpoint poses bundled with PicoPose
(`utils/predefined_poses/obj_poses_level1.npy`).

Must run in the ``picopose`` conda env (torch 2.0 + xformers 0.0.18 + mmcv 2.0).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_PICOPOSE_ROOT = Path(__file__).resolve().parent / "thirdparty/PicoPose"
# Only add the repo root — adding utils/ to sys.path would shadow stdlib `logging`.
# `from utils.xxx import ...` works with just the root on sys.path.
_s = str(_PICOPOSE_ROOT)
if _s not in sys.path:
    sys.path.insert(0, _s)


# Fixed templates camera (paper convention, do not change).
TEMPLATE_K = np.array([
    [572.4114, 0.0, 320.0],
    [0.0, 573.57043, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
TEMPLATE_W = 640
TEMPLATE_H = 480
DEFAULT_N_VIEWS = 162  # level-1


# ── nvdiffrast template renderer ──

def _glcam_in_cvcam() -> np.ndarray:
    return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _projection_matrix(K: np.ndarray, H: int, W: int, znear: float = 0.001, zfar: float = 100.0) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 1 - 2 * cx / W
    P[1, 2] = 2 * cy / H - 1
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * zfar * znear / (zfar - znear)
    P[3, 2] = -1
    return P


def _load_mesh_for_render(mesh_path: Path) -> Dict[str, np.ndarray]:
    import trimesh
    m = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type: {type(m)} for {mesh_path}")
    out: Dict[str, np.ndarray] = {
        "vertices": np.asarray(m.vertices, dtype=np.float32),
        "faces": np.asarray(m.faces, dtype=np.int32),
        "extents": np.asarray(m.extents, dtype=np.float32),
    }
    has_uv = (
        hasattr(m.visual, "uv") and m.visual.uv is not None
        and hasattr(m.visual, "material") and getattr(m.visual.material, "image", None) is not None
    )
    if has_uv:
        tex = np.asarray(m.visual.material.image.convert("RGB"), dtype=np.float32) / 255.0
        out["uv"] = np.asarray(m.visual.uv, dtype=np.float32)
        out["texture"] = tex  # (H, W, 3)
    elif hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors is not None:
        vc = np.asarray(m.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        out["vertex_colors"] = vc
    return out


def _render_template_views(
    mesh_path: Path,
    poses_obj_in_cam: np.ndarray,  # (N, 4, 4) in mm units
    out_dir: Path,
    obj_id: int,
    K: np.ndarray = TEMPLATE_K,
    H: int = TEMPLATE_H,
    W: int = TEMPLATE_W,
) -> None:
    """Render N RGBA + depth template views via nvdiffrast and save to disk."""
    import nvdiffrast.torch as dr

    mesh = _load_mesh_for_render(mesh_path)
    device = torch.device("cuda")
    pos = torch.tensor(mesh["vertices"], device=device).contiguous()  # in mesh units (m)
    faces = torch.tensor(mesh["faces"], device=device).contiguous()

    has_uv = "texture" in mesh
    has_vc = "vertex_colors" in mesh

    if has_uv:
        uv = torch.tensor(mesh["uv"], device=device)  # (V, 2)
        # Flip V (texture image is y-down, OpenGL UV is y-up)
        uv = torch.stack([uv[:, 0], 1.0 - uv[:, 1]], dim=-1).contiguous()
        tex = torch.tensor(mesh["texture"], device=device).contiguous()  # (Ht, Wt, 3)
    elif has_vc:
        vc = torch.tensor(mesh["vertex_colors"], device=device).contiguous()
    # else fall back to constant gray

    # Work in meters: mesh vertices are already in meters; convert input poses
    # (which are in mm) to meters for the projection.
    pos_homo = torch.cat([pos, torch.ones(pos.shape[0], 1, device=device)], dim=-1)  # (V, 4)

    glcam = torch.tensor(_glcam_in_cvcam(), device=device)
    proj = torch.tensor(_projection_matrix(K, H=H, W=W, znear=0.01, zfar=10.0), device=device)

    glctx = dr.RasterizeCudaContext(device=device)

    img_dir = out_dir / f"{obj_id:06d}"
    img_dir.mkdir(parents=True, exist_ok=True)
    poses_dir = out_dir / "object_poses"
    poses_dir.mkdir(parents=True, exist_ok=True)

    # Convert poses from mm → m for rendering (vertices are in m).
    poses_m = poses_obj_in_cam.astype(np.float32).copy()
    poses_m[:, :3, 3] *= 1e-3
    poses_t = torch.tensor(poses_m, device=device)  # (N, 4, 4) m

    for i in range(poses_t.shape[0]):
        T = poses_t[i]                       # mesh→cam (cv), m
        T_gl = glcam @ T                     # mesh→cam (gl)
        mvp = proj @ T_gl                    # full clip-space transform
        pos_clip = (mvp @ pos_homo.T).T.contiguous()      # (V, 4)

        rast_out, _ = dr.rasterize(glctx, pos_clip[None].contiguous(), faces, resolution=[H, W])
        # rast_out: (1, H, W, 4) — (u, v, z/w, triangle_id+1); alpha = (rast[..., 3] > 0)

        alpha = (rast_out[..., 3:4] > 0).float()  # (1, H, W, 1)

        if has_uv:
            uv_pix, _ = dr.interpolate(uv[None], rast_out, faces)  # (1, H, W, 2)
            color = dr.texture(tex[None], uv_pix, filter_mode="linear")  # (1, H, W, 3)
        elif has_vc:
            color, _ = dr.interpolate(vc[None], rast_out, faces)
        else:
            shading = torch.full_like(alpha, 0.5).expand(-1, -1, -1, 3)
            color = shading
        color = color * alpha  # mask out background

        # depth from clip-space z/w → world Z in cam frame (mm).
        # We compute Z directly from the linear-Z buffer of the rasterizer:
        # rast_out[..., 2] is z/w in NDC ∈ [-1, 1], not directly metric.
        # Easier: interpolate per-vertex Z (in cam space) using barycentric coords.
        z_cam = (T_gl @ pos_homo.T).T[:, 2:3].contiguous()  # (V, 1) cam z gl (negative forward, m)
        z_cv_mm = (-z_cam * 1000.0).contiguous()             # gl→cv positive forward, mm
        z_buf, _ = dr.interpolate(z_cv_mm, rast_out, faces)   # (1, H, W, 1) mm
        depth_mm = (z_buf * alpha).clamp(min=0).squeeze().detach().cpu().numpy()
        depth_uint16 = depth_mm.astype(np.uint16)

        rgba = torch.cat([color, alpha], dim=-1).squeeze(0).detach().cpu().numpy()
        rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
        # Flip vertically (nvdiffrast renders upside-down vs OpenCV).
        rgba = np.flip(rgba, axis=0).copy()
        depth_uint16 = np.flip(depth_uint16, axis=0).copy()

        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(img_dir / f"{i:06d}.png"), bgra)
        cv2.imwrite(str(img_dir / f"{i:06d}_depth.png"), depth_uint16)

    np.save(poses_dir / f"{obj_id:06d}.npy", poses_obj_in_cam.astype(np.float32))


def _onboard_object(
    mesh_path: Path,
    template_root: Path,
    obj_id: int = 1,
    n_views: int = DEFAULT_N_VIEWS,
    diameter_scale: float = 1.0,
) -> None:
    """Render N template views using level-1 predefined viewpoint poses.

    diameter_scale lets us push the camera in/out (1.0 = z=diameter mm, paper default).
    """
    pose_path = _PICOPOSE_ROOT / "utils/predefined_poses/obj_poses_level1.npy"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing predefined poses: {pose_path}")
    poses = np.load(pose_path)[:n_views].astype(np.float64)  # (N, 4, 4) mm

    # Set translation = (0, 0, diameter) per object (matches BOP renderer behavior).
    import trimesh
    m = trimesh.load(str(mesh_path), process=False, force="mesh")
    diameter_mm = float(np.linalg.norm(m.extents)) * 1000.0 * diameter_scale
    poses[:, :3, 3] = np.array([0.0, 0.0, diameter_mm])[None].repeat(n_views, axis=0)

    template_root.mkdir(parents=True, exist_ok=True)
    _render_template_views(
        mesh_path=mesh_path,
        poses_obj_in_cam=poses,
        out_dir=template_root,
        obj_id=obj_id,
    )


def _picopose_template_dir(assets_root: Path, obj_name: str) -> Path:
    return assets_root / "templates" / obj_name


def _picopose_repre_paths(assets_root: Path, obj_name: str, obj_id: int = 1) -> Tuple[Path, Path]:
    tdir = _picopose_template_dir(assets_root, obj_name)
    return tdir / f"{obj_id:06d}", tdir / "object_poses" / f"{obj_id:06d}.npy"


# ── inference wrapper ──

def _bbox_from_mask(mask_bool: np.ndarray) -> Optional[List[int]]:
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(rmin), int(rmax) + 1, int(cmin), int(cmax) + 1]


def _square_bbox(rmin: int, rmax: int, cmin: int, cmax: int, H: int, W: int) -> List[int]:
    rb = rmax - rmin
    cb = cmax - cmin
    b = min(max(rb, cb), min(H, W))
    cy = (rmin + rmax) // 2
    cx = (cmin + cmax) // 2
    rmin = cy - b // 2
    rmax = cy + b // 2
    cmin = cx - b // 2
    cmax = cx + b // 2
    if rmin < 0:
        rmax += -rmin
        rmin = 0
    if cmin < 0:
        cmax += -cmin
        cmin = 0
    if rmax > H:
        rmin -= rmax - H
        rmax = H
    if cmax > W:
        cmin -= cmax - W
        cmax = W
    return [rmin, rmax, cmin, cmax]


def _init_points2d(img_size: int, patch_size: float) -> np.ndarray:
    n = int(img_size / patch_size)
    coords = np.linspace(patch_size / 2.0, img_size - patch_size / 2.0, n)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    pts = np.stack([xx, yy], axis=-1).astype(np.float32)
    return pts  # (n, n, 2)


class PicoPoseInit:
    """Per-view PicoPose pose estimation, in-process.

    Mirrors :class:`FoundPoseInit` API but uses the PicoPose 3-stage progressive
    correspondence model instead.
    """

    DEFAULT_OPTS = dict(
        img_size=224,
        pts_size=64,
        n_template_view=DEFAULT_N_VIEWS,
        hypothesis=5,
        rgb_mask_flag=False,
        min_mask_pixels=200,
        translation_scale=1e-3,  # cv2 PnP returns translation in meters when 3D points are in meters
    )

    def __init__(
        self,
        mesh_path: str,
        assets_root: str,
        obj_name: str,
        object_id: int = 1,
        device: str = "cuda:0",
        checkpoint_path: str = "/home/mingi/shared_data/AutoDex/weights/picopose/picopose.pth",
        config_path: Optional[str] = None,
        opts: Optional[Mapping[str, Any]] = None,
        force_onboard: bool = False,
    ):
        self.mesh_path = Path(mesh_path).resolve()
        self.assets_root = Path(assets_root).resolve()
        self.obj_name = obj_name
        self.object_id = int(object_id)
        self.device = device
        self.opts = dict(self.DEFAULT_OPTS)
        if opts:
            self.opts.update(opts)
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else (_PICOPOSE_ROOT / "config/base.yaml")

        tem_dir, pose_npy = _picopose_repre_paths(self.assets_root, obj_name, self.object_id)
        n_views = self.opts["n_template_view"]
        need_onboard = (
            force_onboard
            or not pose_npy.is_file()
            or not tem_dir.is_dir()
            or sum(1 for p in tem_dir.glob("*.png") if "_depth" not in p.name) < n_views
        )
        if need_onboard:
            t0 = time.perf_counter()
            logger.info(f"[PicoPoseInit] Onboarding {obj_name} (mesh={self.mesh_path.name})")
            _onboard_object(
                mesh_path=self.mesh_path,
                template_root=_picopose_template_dir(self.assets_root, obj_name),
                obj_id=self.object_id,
                n_views=n_views,
            )
            logger.info(f"[PicoPoseInit] Onboarded {obj_name} in {time.perf_counter() - t0:.1f}s")

        self._build_model()
        self._build_template_features()

    def _build_model(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(self.config_path)
        # Inject runtime overrides into model config.
        if "model" not in cfg:
            raise RuntimeError(f"Bad picopose config: {self.config_path}")
        from model.picopose import Net
        t0 = time.perf_counter()
        model = Net(cfg.model)

        # Load lightning checkpoint, strip "network." prefix from state_dict.
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("network."):
                new_sd[k[len("network."):]] = v
            else:
                new_sd[k] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            logger.warning(f"[PicoPoseInit] missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            logger.warning(f"[PicoPoseInit] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
        model = model.to(self.device)
        model.eval()
        self.model = model
        self._cfg = cfg
        self.model_load_sec = time.perf_counter() - t0
        logger.info(f"[PicoPoseInit] Model loaded in {self.model_load_sec:.1f}s")

        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def _build_template_features(self) -> None:
        """Load all N template views, run them through the feature extractor once,
        and cache the resulting per-view feature map for matching."""
        from utils.data_utils import get_point_cloud_from_depth, load_im, get_bbox

        tem_dir, pose_npy = _picopose_repre_paths(self.assets_root, self.obj_name, self.object_id)
        n_views = self.opts["n_template_view"]
        img_size = int(self.opts["img_size"])
        pts_size = int(self.opts["pts_size"])

        all_rgb, all_pts3d, all_mask, all_M, all_K, all_pose = [], [], [], [], [], []
        templates_K = TEMPLATE_K.copy()

        obj_poses = np.load(pose_npy)  # mm
        for i in range(n_views):
            img_path = tem_dir / f"{i:06d}.png"
            depth_path = tem_dir / f"{i:06d}_depth.png"
            if not img_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"Template view missing: {img_path}")
            rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                raise RuntimeError(f"Failed to read {img_path}")
            if rgba.shape[2] == 4:
                bgr, alpha = rgba[..., :3], rgba[..., 3]
            else:
                raise RuntimeError(f"Template image must be RGBA: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = (alpha > 127).astype(np.float32)
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
            mask_c = mask[y1:y2, x1:x2]

            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm→m
            pts = get_point_cloud_from_depth(depth, templates_K, [y1, y2, x1, x2])
            pts = cv2.resize(pts, (pts_size, pts_size), interpolation=cv2.INTER_NEAREST)

            rgb_c = rgb[y1:y2, x1:x2, :].astype(np.float32) / 255.0
            if self.opts["rgb_mask_flag"]:
                rgb_c = rgb_c * (mask_c[:, :, None] > 0).astype(np.float32)
            rgb_c = cv2.resize(rgb_c, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            mask_c = cv2.resize(mask_c[:, :, None].astype(int), (img_size, img_size),
                                interpolation=cv2.INTER_NEAREST)
            rgb_t = self._transform(np.array(rgb_c)).float()

            M_crop = np.array([[1, 0, -bbox[2]], [0, 1, -bbox[0]], [0, 0, 1]], dtype=np.float32)
            M_resize = np.array([
                [img_size / (y2 - y1), 0, 0],
                [0, img_size / (x2 - x1), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            M = M_resize @ M_crop

            pose_m = obj_poses[i].astype(np.float32).copy()
            pose_m[:3, 3] /= 1000.0  # mm → m

            all_rgb.append(rgb_t)
            all_pts3d.append(torch.from_numpy(pts).float())
            all_mask.append(torch.from_numpy(mask_c).float())
            all_M.append(torch.from_numpy(M).float())
            all_K.append(torch.from_numpy(templates_K).float())
            all_pose.append(torch.from_numpy(pose_m).float())

        device = self.device
        self._tem = {
            "tem_rgb": torch.stack(all_rgb).to(device),       # (N, 3, H, W)
            "tem_pts3d": torch.stack(all_pts3d).to(device),  # (N, p, p, 3)
            "tem_mask": torch.stack(all_mask).to(device),    # (N, H, W)
            "tem_M": torch.stack(all_M).to(device),
            "tem_K": torch.stack(all_K).to(device),
            "tem_pose": torch.stack(all_pose).to(device),
        }

        # Pre-compute feature for matching (last-stage feature map, normalized).
        bs = 16
        feats = []
        with torch.no_grad():
            for j in range(0, n_views, bs):
                f = self.model.feature_extractor(self._tem["tem_rgb"][j:j + bs].contiguous())
                feats.append(f[-1])
        feats = torch.cat(feats, dim=0)  # (N, C, hf, wf)
        # Squeeze if needed; matching expects (N, C, hf, wf)
        self._tem["template_feature"] = feats

    # ── per-view inference ──

    def estimate_one_view(
        self,
        image_rgb: np.ndarray,
        mask_bool: np.ndarray,
        K: np.ndarray,
        ext_cw: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Run PicoPose on one view. Returns dict matching FoundPoseInit shape."""
        from utils.pose_recovery import pose_recovery_ransac_pnp
        from utils.data_utils import get_bbox
        from utils.torch_utils import init_points2d_numpy

        H, W = image_rgb.shape[:2]
        n_pixels = int(mask_bool.sum())
        if n_pixels < int(self.opts["min_mask_pixels"]):
            return None
        bb = _bbox_from_mask(mask_bool)
        if bb is None:
            return None
        # Use square bbox, aligned with picopose's get_bbox semantics.
        bbox = get_bbox(mask_bool.astype(np.uint8))
        y1, y2, x1, x2 = bbox

        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        img_size = int(self.opts["img_size"])
        pts_size = int(self.opts["pts_size"])

        mask_c = mask_bool[y1:y2, x1:x2].astype(np.float32)
        rgb_c = image_rgb[y1:y2, x1:x2].astype(np.float32) / 255.0
        if self.opts["rgb_mask_flag"]:
            rgb_c = rgb_c * (mask_c[:, :, None] > 0).astype(np.float32)
        rgb_c = cv2.resize(rgb_c, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        mask_c = cv2.resize(mask_c[:, :, None].astype(int), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        rgb_t = self._transform(np.array(rgb_c)).float()

        M_crop = np.array([[1, 0, -bbox[2]], [0, 1, -bbox[0]], [0, 0, 1]], dtype=np.float32)
        M_resize = np.array([
            [img_size / (y2 - y1), 0, 0],
            [0, img_size / (x2 - x1), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        M = M_resize @ M_crop

        # pts2d: per-pixel original-image coords (pts_size, pts_size, 2)
        pts2d = init_points2d_numpy(img_size, patch_size=img_size / pts_size)
        ones = np.ones_like(pts2d[..., :1])
        pts_h = np.concatenate([pts2d, ones], axis=-1)
        Minv = np.linalg.inv(M)
        pts2d_orig = (Minv @ pts_h.reshape(-1, 3).T)
        pts2d_orig = (pts2d_orig[:2] / pts2d_orig[2:]).T.reshape(pts_size, pts_size, 2)

        timings["preprocess_sec"] = time.perf_counter() - t0

        device = self.device
        # Build inputs matching forward_test() signature.
        # batch dim B=1
        real_pose = torch.eye(4, dtype=torch.float32, device=device)
        inputs = {
            "real_rgb": rgb_t.unsqueeze(0).to(device),
            "real_mask": torch.from_numpy(mask_c).float().unsqueeze(0).to(device),
            "real_pts2d": torch.from_numpy(pts2d_orig).float().unsqueeze(0).to(device),
            "real_K": torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device),
            "real_M": torch.from_numpy(M).unsqueeze(0).to(device),
            "real_pose": real_pose.unsqueeze(0),
            # template entries: prepend a batch dim of 1 (templates indexed by view via topk).
            "template_feature": self._tem["template_feature"].unsqueeze(0),
            "tem_rgb": self._tem["tem_rgb"].unsqueeze(0),
            "tem_pts3d": self._tem["tem_pts3d"].unsqueeze(0),
            "tem_mask": self._tem["tem_mask"].unsqueeze(0),
            "tem_M": self._tem["tem_M"].unsqueeze(0),
            "tem_K": self._tem["tem_K"].unsqueeze(0),
            "tem_pose": self._tem["tem_pose"].unsqueeze(0),
        }

        hyp = int(self.opts["hypothesis"])
        t1 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.forward_test(inputs, hyp=hyp)
        torch.cuda.synchronize()
        timings["forward_sec"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        preds = []
        for tk in range(hyp):
            pred_r, pred_t, inliers_ratio, success = pose_recovery_ransac_pnp(
                outputs[tk]["tar_pts_2d"][0],
                outputs[tk]["src_pts_3d"][0],
                inputs["real_K"][0],
                outputs[tk]["tem_pose"][0],
                outputs[tk]["pred_tar_pts"][0],
                outputs[tk]["pred_src_pts"][0],
            )
            if not success:
                pred_r = outputs[tk]["pred_poses"][0][:3, :3].detach().cpu().numpy()
                pred_t = outputs[tk]["pred_poses"][0][:3, 3].detach().cpu().numpy().reshape(3, 1)
            preds.append({
                "R": pred_r.reshape(3, 3),
                "t": pred_t.reshape(3, 1),
                "inliers_ratio": float(inliers_ratio),
                "template_id": int(outputs[tk].get("template_id", -1)) if "template_id" in outputs[tk] else -1,
            })
        timings["pnp_sec"] = time.perf_counter() - t2

        # Pick top-1 by inliers_ratio (matches run_test.py sort).
        preds.sort(key=lambda p: p["inliers_ratio"], reverse=True)
        best = preds[0]

        R_m2c = best["R"].astype(np.float64)
        t_m2c = best["t"].reshape(3).astype(np.float64)
        # cv2.solvePnP returns t in the same units as input 3D points.
        # In pose_recovery_ransac_pnp, 3D points come from tem_pts3d (in meters),
        # so t is already in meters. Do not multiply by 1e-3.
        pose_camera_m = np.eye(4, dtype=np.float64)
        pose_camera_m[:3, :3] = R_m2c
        pose_camera_m[:3, 3] = t_m2c

        # World pose: ext_cw is camera-from-world; pose_world = inv(ext_cw) @ pose_camera
        T_wc = np.linalg.inv(np.asarray(ext_cw, dtype=np.float64))
        pose_world_m = T_wc @ pose_camera_m

        return {
            "pose_world": pose_world_m,
            "pose_camera": pose_camera_m,
            "quality": float(best["inliers_ratio"]),
            "inliers": int(round(best["inliers_ratio"] * (int(self.opts["pts_size"]) ** 2))),
            "template_id": int(best["template_id"]),
            "mask_pixels": n_pixels,
            "timings": timings,
        }

    def estimate_per_view(
        self,
        images_rgb: Dict[str, np.ndarray],
        masks_bool: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        extrinsics: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """Run PicoPose on every camera. Returns {serial: result_or_None}."""
        out: Dict[str, Dict[str, Any]] = {}
        for s, img in images_rgb.items():
            if s not in masks_bool or s not in intrinsics or s not in extrinsics:
                out[s] = None
                continue
            try:
                out[s] = self.estimate_one_view(
                    img, masks_bool[s], intrinsics[s], extrinsics[s],
                )
            except Exception as exc:
                logger.warning(f"[PicoPoseInit] {s} failed: {exc}")
                out[s] = None
        return out
