"""In-process GoTrack stage 1-4 wrapper for one capture PC (N cameras).

Wraps the MV-GoTrack online tracker so a daemon can call ``process_frame``
once per multi-cam frame, get back per-camera anchor observations, and
forward them to the central robot PC for triangulation + Kabsch fit (stages
5-6).

Stages handled here (per-camera, run on the local GPU):
  1. Template render (nvdiffrast) using the prior pose
  2. Crop (template + live image) to model.opts.crop_size
  3. DINOv2 forward → flow + confidence
  4. Anchor 2D observations (project canonical anchors → sample flow → live uv)

Stages NOT handled (run on robot PC after gathering 24-cam obs):
  5. Multi-view triangulation
  6. Robust Kabsch fit (RANSAC) → world pose

Must run inside the ``gotrack`` conda env (depends on torch 2.0 + xformers
+ the MV-GoTrack code under autodex/perception/thirdparty/MV-GoTrack/).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GOTRACK_ROOT = Path(__file__).resolve().parent / "thirdparty/MV-GoTrack"
if str(_GOTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GOTRACK_ROOT))

# Default GoTrack runtime knobs (matches gotrack_pipeline_debug.py).
_DEFAULT_CONFIG = "configs/model/gotrack.yaml"
_DEFAULT_CKPT = _GOTRACK_ROOT / "gotrack_checkpoint.pt"


@dataclass
class CameraIntrinsics:
    serial: str
    K: np.ndarray            # 3x3
    extrinsic_cw: np.ndarray  # 4x4 world->cam
    width: int
    height: int


def _build_camera_model(intr: CameraIntrinsics):
    """Wrap our intrinsics+extrinsics into the GoTrack PinholePlaneCameraModel."""
    from utils import structs
    T_world_from_eye = np.linalg.inv(np.asarray(intr.extrinsic_cw, dtype=np.float64))
    return structs.PinholePlaneCameraModel(
        width=int(intr.width), height=int(intr.height),
        f=(float(intr.K[0, 0]), float(intr.K[1, 1])),
        c=(float(intr.K[0, 2]), float(intr.K[1, 2])),
        T_world_from_eye=T_world_from_eye,
    )


def _build_args_namespace(
    *,
    object_id: int = 1,
    mask_free: bool = True,
    skip_pnp: bool = True,
    confidence_threshold: float = 0.25,
    anchor_confidence_threshold: float = 0.25,
    anchor_depth_tolerance_m: float = 0.05,
    max_active_anchors_per_view: int = 256,
    mask_threshold: int = 127,
    min_mask_pixels: int = 200,
) -> argparse.Namespace:
    """Build a minimal argparse-style namespace consumed by GoTrack helpers."""
    return argparse.Namespace(
        object_id=object_id,
        mask_free=mask_free,
        skip_pnp=skip_pnp,
        confidence_threshold=confidence_threshold,
        anchor_confidence_threshold=anchor_confidence_threshold,
        anchor_depth_tolerance_m=anchor_depth_tolerance_m,
        max_active_anchors_per_view=max_active_anchors_per_view,
        mask_threshold=mask_threshold,
        min_mask_pixels=min_mask_pixels,
        # Optimized input pipeline knobs (off; we batch per-frame on this PC).
        optimized_input_pipeline=False,
        optimized_input_pipeline_v2=False,
        optim_crop_update_interval=0,
        optim_template_update_interval=1,
        optim_template_render_workers=1,
        optim_v2_crop_camera_workers=1,
        optim_v2_warp_grid_workers=1,
        template_renderer_backend="nvdiffrast",
    )


class GoTrackEngine:
    """One-PC GoTrack inference for N cameras.

    Args:
        mesh_path: object mesh (used for template render + anchor projection).
        anchor_bank_path: pre-generated anchor bank (.npz from
            scripts/generate_anchor_bank.py).
        cameras: list of CameraIntrinsics for the N cams attached to this PC.
        object_id: integer object id (1 for single-object setup).
        checkpoint_path: GoTrack checkpoint .pt.
        config_path: GoTrack model config .yaml.
        device: cuda device string.
        mesh_scale: 1.0 (mesh is in meters) or 1000.0 (mm).
        unit_scale_mode: "auto" | "meter" | "mm" — passed through to GoTrack
            unit-scale resolver.
        first_frame_num_iters / num_iters: refinement iterations. Online
            tracking uses 1 (warm prior); first frame can use 5 for safety.
    """

    def __init__(
        self,
        mesh_path: str,
        anchor_bank_path: str,
        cameras: List[CameraIntrinsics],
        object_id: int = 1,
        object_name: str = "object",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        mesh_scale: float = 1.0,
        unit_scale_mode: str = "auto",
        num_iters: int = 1,
        first_frame_num_iters: int = 5,
        mask_free: bool = True,
        skip_pnp: bool = True,
    ):
        import torch
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")

        self.device_str = device
        self.device = torch.device(device)
        if device.startswith("cuda"):
            torch.cuda.set_device(self.device)

        self.mesh_path = str(Path(mesh_path).expanduser().resolve())
        self.anchor_bank_path = str(Path(anchor_bank_path).expanduser().resolve())
        self.object_id = int(object_id)
        self.object_name = object_name

        self.cameras = {c.serial: c for c in cameras}
        self.serials = [c.serial for c in cameras]

        # GoTrack helpers (lazy import — these touch torch + nvdiffrast).
        from run_singleview_multi_object import (
            load_gotrack_model,
            setup_renderer,
        )
        from utils.anchor_bank import (
            load_anchor_bank,
            prepare_anchor_bank_for_gotrack,
        )
        from utils.unit_scale import resolve_unit_scale_for_object

        cfg_path = config_path or str(_GOTRACK_ROOT / _DEFAULT_CONFIG)
        ckpt_path = checkpoint_path or str(_DEFAULT_CKPT)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"GoTrack checkpoint missing: {ckpt_path}")

        # Cap iterations to first_frame value here; we manually swap to num_iters
        # after the first frame in process_frame().
        self._num_iters = int(num_iters)
        self._first_frame_num_iters = int(first_frame_num_iters)
        self._frame_count = 0

        logger.info(f"[GoTrackEngine] loading model on {device}")
        self.model = load_gotrack_model(
            config_path=cfg_path,
            checkpoint_path=ckpt_path,
            device=self.device,
            num_iters=self._first_frame_num_iters,
        )

        # Resolve unit scale (handles meter↔mm conversion between mesh and GoTrack).
        unit_info = resolve_unit_scale_for_object(
            mesh_path=self.mesh_path,
            mesh_scale=float(mesh_scale),
            mode=str(unit_scale_mode),
        )
        self._unit_info = unit_info
        self._translation_scale = float(unit_info.translation_scale_to_gotrack)

        self.renderer = setup_renderer(
            model=self.model,
            obj_ids=[self.object_id],
            mesh_paths=[self.mesh_path],
            unit_scale_infos={self.object_id: unit_info},
            backend="nvdiffrast",
        )

        # Anchor bank — canonical FPS anchors on mesh (mesh frame, mm or meters
        # depending on mesh). prepare_anchor_bank_for_gotrack normalises into
        # GoTrack's internal unit system.
        raw_bank = load_anchor_bank(self.anchor_bank_path)
        self.anchor_bank = prepare_anchor_bank_for_gotrack(
            raw_bank,
            translation_scale_to_gotrack=unit_info.translation_scale_to_gotrack,
        )

        # GoTrack-side camera models (cached).
        self.gotrack_camera_models: Dict[str, Any] = {
            s: _build_camera_model(c) for s, c in self.cameras.items()
        }

        # argparse-like namespace consumed by GoTrack helpers.
        self.args = _build_args_namespace(
            mask_free=mask_free,
            skip_pnp=skip_pnp,
        )
        self.mask_free = mask_free

    def _set_iters(self, n: int) -> None:
        """GoTrackOpts is a NamedTuple (immutable). Try to replace; fall back
        to no-op if model rejects assignment (in which case `num_iters` stays at
        whatever was set at load time)."""
        try:
            self.model.opts = self.model.opts._replace(num_iterations_test=int(n))
        except (AttributeError, TypeError):
            pass

    def process_frame(
        self,
        prior_pose_world: np.ndarray,
        frames_bgr: Dict[str, np.ndarray],
        masks: Optional[Dict[str, np.ndarray]] = None,
        frame_index: int = 0,
        time_sec: float = 0.0,
        include_debug_images: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Run GoTrack stage 1-4 on N cameras for one frame.

        Args:
            prior_pose_world: 4x4 SE(3), object-in-world pose to use as prior.
            frames_bgr: {serial: HxWx3 uint8 BGR}. Must contain entries for
                every camera passed to __init__; missing cameras are skipped.
            masks: optional {serial: HxW uint8/bool} foreground mask. Required
                if mask_free=False.
            frame_index: monotonic frame id for record-keeping (passed back).
            time_sec: optional timestamp; passed through.
            include_debug_images: forward to GoTrack (heavy, off by default).

        Returns:
            {
              serial: {
                "frame_index": int,
                "status": "ok" | "empty_mask" | "missing",
                "anchor_uv_orig": np.float32 (M, 2)  — anchor (u, v) in original image,
                "anchor_conf":    np.float32 (M,),
                "valid_mask":     np.bool_  (M,)  — anchor was projected + flow-sampled successfully,
                "selected_mask":  np.bool_  (M,)  — top-K by conf (max_active_anchors_per_view),
                "compute_sec":    float,
              },
              ...
            }
            Anchor count M == len(self.anchor_bank["positions_o"]).
        """
        import time as _time
        import torch
        from run_multiview_gotrack_anchor_online import (
            _build_anchor_observations_for_frame,
            _process_group_for_timestep_anchor,
            DeviceState,
        )

        # Adjust iters based on first-frame.
        if self._frame_count == 0:
            self._set_iters(self._first_frame_num_iters)
        else:
            self._set_iters(self._num_iters)

        prior_pose_world = np.asarray(prior_pose_world, dtype=np.float64).reshape(4, 4)

        # Build frame_batch in GoTrack's expected shape.
        frame_batch: Dict[str, Dict[str, Any]] = {}
        for s in self.serials:
            if s not in frames_bgr:
                continue
            mask_frame = None
            if masks is not None and s in masks:
                m = masks[s]
                if m.dtype == np.bool_:
                    m = (m.astype(np.uint8) * 255)
                mask_frame = m
            frame_batch[s] = {
                "frame_bgr": frames_bgr[s],
                "mask_frame": mask_frame,
                "frame_index": int(frame_index),
                "time_sec": float(time_sec),
            }

        if not frame_batch:
            return {}

        # extrinsics dict + camera_models dict in expected shapes.
        extrinsics_map = {s: self.cameras[s].extrinsic_cw for s in frame_batch}
        camera_models = self.gotrack_camera_models

        device_state = DeviceState(
            device=self.device,
            device_name=str(self.device),
            model=self.model,
            camera_ids=list(frame_batch.keys()),
        )

        t0 = _time.perf_counter()
        per_camera_records = _process_group_for_timestep_anchor(
            device_state=device_state,
            frame_batch=frame_batch,
            camera_models=camera_models,
            gotrack_camera_models=camera_models,
            extrinsics_map=extrinsics_map,
            init_pose_world=prior_pose_world,
            init_pose_source="prior",
            args=self.args,
            include_debug_images=include_debug_images,
        )

        # _process_group_for_timestep_anchor stores debug payloads inside each
        # frame_record under "debug_data". Extract them for stage 4.
        per_camera_debug: Dict[str, Dict[str, Any]] = {}
        for cam, rec in per_camera_records.items():
            dbg = rec.pop("debug_data", None) or rec.get("debug", None)
            if dbg is not None:
                per_camera_debug[cam] = dbg

        # === DIAG: save crop images to ~/shared_data/AutoDex/debug/gotrack_crops/{obj}/{session_pid}/{frame:06d}/{serial}_*.png ===
        try:
            import os, cv2 as _cv2
            if self._frame_count < 5:
                home = os.path.expanduser("~")
                # session_pid groups all crops from this daemon's lifetime (resets on restart).
                session_id = getattr(self, "_diag_session", None)
                if session_id is None:
                    self._diag_session = f"{int(time.time())}_{os.getpid()}"
                    session_id = self._diag_session
                save_dir = f"{home}/shared_data/AutoDex/debug/gotrack_crops/{self.obj_name}/{session_id}/{int(frame_index):06d}"
                os.makedirs(save_dir, exist_ok=True)
                for s in self.serials:
                    dbg = per_camera_debug.get(s)
                    if dbg is None:
                        continue
                    q = dbg.get("query_rgb_crop")
                    t = dbg.get("template_rgb_crop")
                    if q is not None:
                        q_arr = np.asarray(q)
                        if q_arr.dtype != np.uint8:
                            q_arr = np.clip(q_arr * 255 if q_arr.max() <= 1.0 else q_arr, 0, 255).astype(np.uint8)
                        if q_arr.ndim == 3 and q_arr.shape[2] == 3:
                            q_arr = _cv2.cvtColor(q_arr, _cv2.COLOR_RGB2BGR)
                        _cv2.imwrite(f"{save_dir}/{s}_query.png", q_arr)
                    if t is not None:
                        t_arr = np.asarray(t)
                        if t_arr.dtype != np.uint8:
                            t_arr = np.clip(t_arr * 255 if t_arr.max() <= 1.0 else t_arr, 0, 255).astype(np.uint8)
                        if t_arr.ndim == 3 and t_arr.shape[2] == 3:
                            t_arr = _cv2.cvtColor(t_arr, _cv2.COLOR_RGB2BGR)
                        _cv2.imwrite(f"{save_dir}/{s}_template.png", t_arr)
                logger.info(f"[diag-crops] saved to {save_dir}")
        except Exception as exc:
            logger.warning(f"[diag-crops] save failed: {exc}")

        # === DIAG: per-cam debug state right after batch refinement ===
        diag_records: Dict[str, str] = {}
        for s in self.serials:
            rec = per_camera_records.get(s)
            dbg = per_camera_debug.get(s)
            if rec is None:
                diag_records[s] = "rec=None"
                continue
            status = rec.get("status", "?")
            if dbg is None:
                diag_records[s] = f"status={status} dbg=None"
                continue
            fmap = dbg.get("flow_map")
            cmap = dbg.get("confidence_map")
            tw = dbg.get("T_world_from_crop_cam")
            ci = dbg.get("crop_intrinsic")
            cis = dbg.get("crop_image_size")
            f_stat = (
                f"flow shape={list(fmap.shape) if hasattr(fmap, 'shape') else None}"
            ) if hasattr(fmap, "min") else "flow=None"
            tw_str = "Tw=None"
            if tw is not None:
                t_arr = np.asarray(tw)
                tw_str = f"Tw=[t={t_arr[:3,3].round(3).tolist()}]"
            ci_str = "ci=None"
            if ci is not None:
                ci_arr = np.asarray(ci)
                ci_str = f"ci_fx={ci_arr[0,0]:.1f} cx={ci_arr[0,2]:.1f}"
            diag_records[s] = (f"status={status} {f_stat} {tw_str} {ci_str} cis={cis}")
        logger.info("[engine.diag] per-cam debug after refine batch:")
        for s, txt in diag_records.items():
            logger.info(f"  {s}: {txt}")

        external_unit_scale_to_meter = float(self._unit_info.external_unit_scale_to_meter)
        per_view_anchor_data, _obs_by_anchor, _summary = _build_anchor_observations_for_frame(
            anchor_bank=self.anchor_bank,
            per_camera_records=per_camera_records,
            per_camera_debug=per_camera_debug,
            frame_batch=frame_batch,
            camera_models=camera_models,
            args=self.args,
            external_unit_scale_to_meter=external_unit_scale_to_meter,
        )

        # === DIAG: per-cam valid_mask after anchor obs build ===
        logger.info("[engine.diag] per-cam anchor obs:")
        for s in self.serials:
            obs = per_view_anchor_data.get(s)
            if obs is None:
                logger.info(f"  {s}: obs=None")
                continue
            vm = obs.get("valid_mask")
            sm = obs.get("selected_mask")
            conf = obs.get("confidence")
            v_sum = int(np.asarray(vm).sum()) if vm is not None else -1
            s_sum = int(np.asarray(sm).sum()) if sm is not None else -1
            c_stat = ""
            if conf is not None:
                ca = np.asarray(conf)
                c_stat = f"conf[min={ca.min():.3f} max={ca.max():.3f} mean={ca.mean():.3f}]"
            logger.info(f"  {s}: valid={v_sum}/256 selected={s_sum} {c_stat}")
        compute_sec = _time.perf_counter() - t0
        self._frame_count += 1

        # Pack per-cam payload for the robot PC. `uv_curr` is in CROP-image
        # coordinates and must be back-projected with `crop_intrinsic` (also
        # cropped). `T_world_from_crop_cam` lets the robot PC build the world-
        # frame ray. `position_o` is the canonical anchor in mesh frame (used
        # by Kabsch fit).
        out: Dict[str, Dict[str, Any]] = {}
        for s, rec in per_camera_records.items():
            obs = per_view_anchor_data.get(s)
            base = {
                "frame_index": int(frame_index),
                "status": rec.get("status", "missing"),
                "compute_sec": compute_sec / max(len(per_camera_records), 1),
            }
            if obs is None:
                out[s] = base
                continue
            dbg = per_camera_debug.get(s, {})
            crop_intrinsic = dbg.get("crop_intrinsic")
            T_world_from_crop_cam = dbg.get("T_world_from_crop_cam")
            base.update({
                "uv_curr":               np.asarray(obs["uv_curr"], dtype=np.float32),
                "confidence":            np.asarray(obs["confidence"], dtype=np.float32),
                "valid_mask":            np.asarray(obs["valid_mask"], dtype=np.bool_),
                "selected_mask":         np.asarray(obs["selected_mask"], dtype=np.bool_),
                "anchor_ids":            np.asarray(obs["anchor_ids"], dtype=np.int64),
                "positions_o":           np.asarray(obs["positions_o"], dtype=np.float32),
                "crop_intrinsic":        (np.asarray(crop_intrinsic, dtype=np.float32)
                                          if crop_intrinsic is not None else None),
                "T_world_from_crop_cam": (np.asarray(T_world_from_crop_cam, dtype=np.float64)
                                          if T_world_from_crop_cam is not None else None),
            })
            out[s] = base
        return out
