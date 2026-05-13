"""Build a merged debug video from a trial directory.

Inputs (per trial):
- `{crops_root}/{obj}/{fid:06d}/{serial}_frame.jpg`    — 1/4-scale undistorted full frame
- `{crops_root}/{obj}/{fid:06d}/bbox.json`             — {serial: [[u,v]x4]} in ORIG-image coords
- `{trial_dir}/pose_log.json`                          — per-frame fit info

Output:
- `{trial_dir}/track_debug.mp4`

Layout: all unique serials sorted; arranged into a (rows × cols) grid where
each (i, j) tile is permanently assigned to one serial. Missing frames for a
serial at a given fid show a black tile labelled "missing". The fid + fit
status (n_inliers, mean_residual_mm, reason if failed) is overlaid as text
on top of the merged grid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _list_fids(crops_dir: Path) -> List[int]:
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


def _draw_bbox(img: np.ndarray, corners_orig: List[List[float]],
               scale: float, color=(0, 255, 0), thickness=2) -> None:
    pts = np.asarray(corners_orig, dtype=np.float32) * scale
    pts = pts.round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def build_video(crops_dir: Path, trial_dir: Path, out_path: Path,
                fps: int = 10, fixed_tile_wh: Tuple[int, int] | None = None) -> None:
    fids = _list_fids(crops_dir)
    if not fids:
        raise SystemExit(f"no fids in {crops_dir}")
    serials = _collect_serials(crops_dir, fids)
    if not serials:
        raise SystemExit(f"no frames in {crops_dir}")
    pose_log = _load_pose_log(trial_dir)

    if fixed_tile_wh is None:
        sample = cv2.imread(str(next(crops_dir.glob(f"{fids[0]:06d}/*_frame.jpg"))))
        if sample is None:
            raise SystemExit("could not read sample frame")
        tile_h, tile_w = sample.shape[:2]
    else:
        tile_w, tile_h = fixed_tile_wh

    n = len(serials)
    cols = math.ceil(math.sqrt(n * 1.6))
    rows = math.ceil(n / cols)
    grid_h, grid_w = tile_h * rows, tile_w * cols
    header_h = 60
    out_h, out_w = grid_h + header_h, grid_w

    # Original frame size = 4× downscaled tile size (because we save at /4).
    scale_to_tile = 1.0 / 4.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not vw.isOpened():
        raise SystemExit(f"could not open {out_path} for writing")

    serial_idx = {s: i for i, s in enumerate(serials)}

    print(f"[build] serials={n} grid={rows}x{cols} tile={tile_w}x{tile_h} fids={len(fids)}")
    for fid in fids:
        fid_dir = crops_dir / f"{fid:06d}"
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
                    tile = img
                    if s in bboxes:
                        _draw_bbox(tile, bboxes[s], scale=scale_to_tile)
            else:
                cv2.putText(tile, "missing", (10, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
            cv2.putText(tile, s, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

        rec = pose_log.get(fid)
        if rec is None:
            status = f"fid={fid}  (no pose_log entry)"
        else:
            status = (f"fid={fid}  n_inliers={rec.get('n_inliers','?')}  "
                      f"resid_mm={rec.get('mean_residual_mm','?'):.2f}")
        cv2.putText(canvas, status, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        vw.write(canvas)
    vw.release()
    print(f"[build] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", required=True, help="e.g. ~/shared_data/AutoDex/experiment/.../{ts}")
    ap.add_argument("--crops-root", default="~/shared_data/AutoDex/debug/gotrack_crops",
                    help="where daemon rsynced crops to; obj subdir is appended")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None, help="default {trial_dir}/track_debug.mp4")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir).expanduser()
    crops_dir = Path(args.crops_root).expanduser() / args.obj
    out = Path(args.out).expanduser() if args.out else trial_dir / "track_debug.mp4"
    build_video(crops_dir, trial_dir, out, fps=args.fps)


if __name__ == "__main__":
    main()
