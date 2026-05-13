"""Build a merged debug video from a trial directory with mesh overlay.

Per (fid, serial) tile: 1/4-scale undistorted frame with green nvdiffrast
mesh silhouette overlay rendered from pose_log's pose_world. Header text
shows fid + fit stats.

Inputs (per trial):
- `{crops_root}/{obj}/{fid:06d}/{serial}_frame.jpg`    — 1/4-scale undistorted full frame
- `{trial_dir}/cam_param/{intrinsics,extrinsics}.json` — calibration saved by track_interactive
- `{trial_dir}/pose_log.json`                          — per-frame pose_world + fit info

Output: `{trial_dir}/track_debug.mp4`

Run in `foundationpose` conda env (needs pytorch3d for SilhouetteOptimizer).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"


def _resolve_mesh(obj_name: str) -> Path:
    p = MESH_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    if not p.exists():
        raise SystemExit(f"mesh not found: {p}")
    return p


def _list_fids(crops_dir: Path) -> List[int]:
    if not crops_dir.exists():
        return []
    return sorted(int(p.name) for p in crops_dir.iterdir() if p.is_dir() and p.name.isdigit())


def _collect_serials(crops_dir: Path, fids: List[int]) -> List[str]:
    seen = set()
    for fid in fids:
        for p in (crops_dir / f"{fid:06d}").glob("*_frame.jpg"):
            seen.add(p.stem.rsplit("_frame", 1)[0])
    return sorted(seen)


def _load_pose_log(trial_dir: Path) -> Dict[int, Dict]:
    path = trial_dir / "pose_log.json"
    if not path.exists():
        return {}
    log = json.loads(path.read_text())
    return {int(rec["frame_id"]): rec for rec in log}


def _load_cam_params(calib_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int]:
    """Returns K_undist per serial, extrinsic_cw per serial, H, W (originals).
    Supports both layouts: the trial-side dump (K_undist key, written by
    track_interactive at trial start) and the system calib dir
    (intrinsics_undistort key)."""
    intr_path = calib_dir / "intrinsics.json"
    extr_path = calib_dir / "extrinsics.json"
    intr_raw = json.loads(intr_path.read_text())
    extr_raw = json.loads(extr_path.read_text())
    K = {}
    for s, v in intr_raw.items():
        if "K_undist" in v:
            K[s] = np.asarray(v["K_undist"], dtype=np.float64)
        else:
            K[s] = np.asarray(v["intrinsics_undistort"], dtype=np.float64)
    T = {}
    for s, v in extr_raw.items():
        e = np.asarray(v, dtype=np.float64)
        if e.shape == (3, 4):
            e = np.vstack([e, [0, 0, 0, 1]])
        T[s] = e
    sample = next(iter(intr_raw.values()))
    return K, T, int(sample["height"]), int(sample["width"])


def _draw_bbox(img: np.ndarray, corners_orig: List[List[float]],
               scale: float, color=(0, 0, 255), thickness=2) -> None:
    pts = np.asarray(corners_orig, dtype=np.float32) * scale
    pts = pts.round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def _render_overlay(image_rgb: np.ndarray, pose_world: np.ndarray,
                    K: np.ndarray, T: np.ndarray, H: int, W: int,
                    glctx, mt, color=(0, 200, 0), alpha=0.5) -> np.ndarray:
    """Render mesh silhouette under pose_world via nvdiffrast and alpha-composite
    on top of image_rgb (HxWx3 uint8). K/T must match the supplied H/W."""
    import torch
    sys.path.insert(0, str(REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render  # type: ignore
    pose_cam = T @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    rc, _, _ = nvdiffrast_render(K=np.asarray(K, np.float32), H=H, W=W,
                                  ob_in_cams=pt, glctx=glctx, mesh_tensors=mt,
                                  use_light=False)
    sil = (rc[0].sum(dim=2) > 0).detach().cpu().numpy()
    out = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[sil] = (out[sil] * (1 - alpha) + color_arr * alpha).astype(np.uint8)
    return out


def build_video(crops_dir: Path, trial_dir: Path, calib_dir: Path,
                mesh_path: Path, out_path: Path, fps: int = 10) -> None:
    fids = _list_fids(crops_dir)
    if not fids:
        raise SystemExit(f"no fids in {crops_dir}")
    serials = _collect_serials(crops_dir, fids)
    if not serials:
        raise SystemExit(f"no frames in {crops_dir}")
    pose_log = _load_pose_log(trial_dir)
    K_orig_all, T_all, H_orig, W_orig = _load_cam_params(calib_dir)

    sample = cv2.imread(str(next(crops_dir.glob(f"{fids[0]:06d}/*_frame.jpg"))))
    if sample is None:
        raise SystemExit("could not read sample frame")
    tile_h, tile_w = sample.shape[:2]
    scale = tile_h / float(H_orig)
    # Scale intrinsics to the tile resolution for rendering.
    K_tile = {s: (K_orig_all[s].copy() * scale) for s in K_orig_all}
    for s in K_tile:
        K_tile[s][2, 2] = 1.0  # restore K[2,2]=1 after scaling

    from autodex.perception.silhouette import SilhouetteOptimizer
    sil_opt = SilhouetteOptimizer(str(mesh_path), device="cuda")
    glctx, mt = sil_opt.glctx, sil_opt.mesh_tensors

    n = len(serials)
    cols = math.ceil(math.sqrt(n * 1.6))
    rows = math.ceil(n / cols)
    grid_h, grid_w = tile_h * rows, tile_w * cols
    header_h = 60
    out_h, out_w = grid_h + header_h, grid_w
    # Write with mp4v (OpenCV's FFmpeg fails on H264 here because it picks
    # h264_v4l2m2m, a hardware encoder that isn't available). We transcode
    # to libx264 at the end with a subprocess ffmpeg call for universal
    # playback (VSCode / browsers).
    tmp_path = out_path.with_suffix(".mp4v.mp4")
    vw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (out_w, out_h))
    if not vw.isOpened():
        raise SystemExit(f"could not open {tmp_path} (mp4v)")
    serial_idx = {s: i for i, s in enumerate(serials)}
    print(f"[build] serials={n} grid={rows}x{cols} tile={tile_w}x{tile_h} fids={len(fids)}")

    for fid in fids:
        fid_dir = crops_dir / f"{fid:06d}"
        rec = pose_log.get(fid)
        pose_world = np.asarray(rec["pose_world"], dtype=np.float64).reshape(4, 4) if rec else None
        bbox_path = fid_dir / "bbox.json"
        bboxes = json.loads(bbox_path.read_text()) if bbox_path.exists() else {}

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for s in serials:
            idx = serial_idx[s]
            r, c = divmod(idx, cols)
            y0, x0 = header_h + r * tile_h, c * tile_w
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            img_path = fid_dir / f"{s}_frame.jpg"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    if img.shape[:2] != (tile_h, tile_w):
                        img = cv2.resize(img, (tile_w, tile_h))
                    if pose_world is not None and s in K_tile:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        try:
                            img_rgb = _render_overlay(img_rgb, pose_world,
                                                       K_tile[s], T_all[s],
                                                       tile_h, tile_w, glctx, mt)
                            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        except Exception as exc:
                            cv2.putText(img, f"overlay err: {exc}", (5, tile_h - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    # Draw crop bbox in red — corners in original undistorted
                    # image coords, scale to tile resolution.
                    if s in bboxes:
                        _draw_bbox(img, bboxes[s], scale=tile_h / float(H_orig),
                                   color=(0, 0, 255))
                    tile = img
            else:
                cv2.putText(tile, "missing", (10, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
            cv2.putText(tile, s, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        if rec is None:
            status = f"fid={fid}  (no pose_log entry)"
        else:
            status = (f"fid={fid}  n_inliers={rec.get('n_inliers','?')}  "
                      f"resid_mm={float(rec.get('mean_residual_mm', -1)):.2f}")
        cv2.putText(canvas, status, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    # Transcode mp4v → libx264 H264 for universal playback (VSCode, browsers).
    import shutil as _sh, subprocess as _sp
    ffmpeg = _sh.which("ffmpeg")
    if ffmpeg:
        print(f"[build] transcoding {tmp_path.name} → {out_path.name} (libx264)")
        ret = _sp.run(
            [ffmpeg, "-y", "-i", str(tmp_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", str(out_path)],
            capture_output=True,
        )
        if ret.returncode == 0:
            tmp_path.unlink(missing_ok=True)
            print(f"[build] wrote {out_path}")
        else:
            print(f"[build] ffmpeg transcode failed (rc={ret.returncode}); keeping {tmp_path}")
            print(ret.stderr.decode(errors="ignore")[-500:])
    else:
        tmp_path.rename(out_path.with_suffix(".mp4"))
        print(f"[build] ffmpeg not found; wrote mp4v as {out_path.with_suffix('.mp4')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", required=True,
                    help="experiment trial dir; trial_ts is its basename")
    ap.add_argument("--crops-root", default="~/shared_data/AutoDex/debug/gotrack_crops")
    ap.add_argument("--calib-dir", default=None,
                    help="Defaults to {crops_root}/{obj}/{trial_ts}/cam_param/ "
                         "(written by track_interactive)")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir).expanduser()
    trial_ts = trial_dir.name
    crops_dir = Path(args.crops_root).expanduser() / args.obj / trial_ts
    if args.calib_dir:
        calib_dir = Path(args.calib_dir).expanduser()
    else:
        calib_dir = crops_dir / "cam_param"
        if not calib_dir.exists():
            cam_root = Path.home() / "shared_data/cam_param"
            calib_dir = sorted(cam_root.iterdir())[-1]
    print(f"[calib] {calib_dir}")
    print(f"[crops] {crops_dir}")
    mesh_path = _resolve_mesh(args.obj)
    out = Path(args.out).expanduser() if args.out else trial_dir / "track_debug.mp4"
    build_video(crops_dir, trial_dir, calib_dir, mesh_path, out, fps=args.fps)


if __name__ == "__main__":
    main()
