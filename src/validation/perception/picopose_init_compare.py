#!/usr/bin/env python3
"""Compare PicoPose-based first-frame init vs current PerceptionPipeline.

Mirrors :file:`foundpose_init_compare.py` but uses
:class:`autodex.perception.picopose_init.PicoPoseInit` for the per-view
first-frame pose estimate. Reference target is the same `pose_world.npy`
saved by the existing run-time pipeline (SAM3 + DA3 + FPose + sil).

Reports for every (object, episode):

    pp_iou_best_view, pp_iou_mean
    trans_err_pre_mm, rot_err_pre_deg     (after IoU select, before sil refine)
    trans_err_post_mm, rot_err_post_deg   (after silhouette refinement)
    sil_loss, sil_sec
    pp_compute_sec, pp_n_views_ok

Run inside the ``picopose`` conda env::

    conda run -n picopose python src/validation/perception/picopose_init_compare.py \\
        --experiment-root ~/shared_data/AutoDex/experiment/selected_100/allegro \\
        --output-dir outputs/picopose_init_compare/selected_100 \\
        --objects white_soap_dish clock icecream_scoop metal_scoop_small \\
                  organizer_beige pepsi_light toothbrush_holder beige_brush \\
                  wood_organizer
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# FoundationPose Utils is needed by silhouette + pose_select for nvdiffrast helpers.
_FPOSE_THIRDPARTY = REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"
if str(_FPOSE_THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(_FPOSE_THIRDPARTY))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

MESH_ROOT = Path("/home/mingi/shared_data/AutoDex/object/paradex")

DEFAULT_OBJECTS = [
    "beige_brush", "clock", "icecream_scoop", "metal_scoop_small",
    "organizer_beige", "pepsi_light", "toothbrush_holder",
    "white_soap_dish", "wood_organizer",
]


# ── data discovery ──

def _list_episodes(obj_dir: Path) -> List[Path]:
    eps = []
    for p in sorted(obj_dir.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "pose_world.npy").exists():
            continue
        if not (p / "_pipeline_tmp" / "masks").exists():
            continue
        if not (p / "images").exists():
            continue
        if not (p / "cam_param" / "intrinsics.json").exists():
            continue
        eps.append(p)
    return eps


def _select_objects(
    experiment_root: Path,
    object_names: List[str],
    n_episodes: int,
    seed: int = 0,
    episode_csv: Optional[Path] = None,
) -> List[Tuple[str, List[Path]]]:
    """Select episodes per object.

    If ``episode_csv`` is given (e.g. foundpose results.csv), use the exact
    (object, episode) pairs from that file so PicoPose evaluates the SAME
    episodes as FoundPose for an apples-to-apples comparison.
    Otherwise, randomly sample ``n_episodes`` per object using ``seed``.
    """
    import random
    rng = random.Random(seed)
    out: List[Tuple[str, List[Path]]] = []

    fixed: Dict[str, List[str]] = {}
    if episode_csv is not None and episode_csv.is_file():
        with open(episode_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fixed.setdefault(row["object"], []).append(row["episode"])
        print(f"[compare] using fixed episode set from {episode_csv}: "
              f"{sum(len(v) for v in fixed.values())} pairs across {len(fixed)} objects")

    for name in object_names:
        odir = experiment_root / name
        if not odir.is_dir():
            print(f"[skip {name}] missing experiment dir")
            continue
        if name in fixed:
            eps = []
            for ep_name in fixed[name]:
                ep = odir / ep_name
                if (ep / "pose_world.npy").exists() and (ep / "_pipeline_tmp" / "masks").exists():
                    eps.append(ep)
                else:
                    print(f"  [warn {name}/{ep_name}] missing pose_world.npy or masks, skipping")
            eps.sort()
            out.append((name, eps))
            continue
        eps = _list_episodes(odir)
        if len(eps) < n_episodes:
            print(f"[skip {name}] only {len(eps)} episodes, need {n_episodes}")
            continue
        sample = rng.sample(eps, n_episodes)
        sample.sort()
        out.append((name, sample))
    return out


def _resolve_mesh_path(obj_name: str) -> Path:
    candidates = [
        MESH_ROOT / obj_name / "raw_mesh" / f"{obj_name}.obj",
        MESH_ROOT / obj_name / "processed_data" / "mesh" / "raw.obj",
        MESH_ROOT / obj_name / "processed_data" / "mesh" / "simplified.obj",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No mesh for {obj_name} under {MESH_ROOT}")


def _load_episode(ep_dir: Path) -> Dict[str, object]:
    images_dir = ep_dir / "images"
    masks_dir = ep_dir / "_pipeline_tmp" / "masks"
    cam_param_dir = ep_dir / "cam_param"

    with open(cam_param_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(cam_param_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    serials = sorted(p.stem for p in images_dir.glob("*.png"))
    images_rgb: Dict[str, np.ndarray] = {}
    masks_bool: Dict[str, np.ndarray] = {}
    intrinsics: Dict[str, np.ndarray] = {}
    extrinsics: Dict[str, np.ndarray] = {}

    H, W = None, None
    for s in serials:
        if s not in intr_raw or s not in extr_raw:
            continue
        bgr = cv2.imread(str(images_dir / f"{s}.png"))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        images_rgb[s] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(masks_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        masks_bool[s] = (m > 127) if m is not None else np.zeros((H, W), dtype=bool)
        intrinsics[s] = np.asarray(intr_raw[s]["intrinsics_undistort"], dtype=np.float64)
        T = np.asarray(extr_raw[s], dtype=np.float64)
        if T.shape == (3, 4):
            T = np.vstack([T, [0, 0, 0, 1]])
        extrinsics[s] = T

    return {
        "images_rgb": images_rgb,
        "masks_bool": masks_bool,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "H": H, "W": W,
        "serials": list(images_rgb.keys()),
    }


# ── geometry ──

def _pose_errors(p_ref: np.ndarray, p_test: np.ndarray) -> Tuple[float, float]:
    t_err_m = float(np.linalg.norm(p_ref[:3, 3] - p_test[:3, 3]))
    R_ref, R_test = p_ref[:3, :3], p_test[:3, :3]
    cos = (np.trace(R_ref.T @ R_test) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return t_err_m * 1000.0, float(np.degrees(np.arccos(cos)))


# ── main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=str, required=True,
                        help="e.g. ~/shared_data/AutoDex/experiment/selected_100/allegro")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--objects", type=str, nargs="*", default=DEFAULT_OBJECTS,
                        help="Object names to evaluate (defaults to the 9 objects already in foundpose results.csv)")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assets-root", type=str,
                        default=str(REPO_ROOT / "outputs/picopose_assets"))
    parser.add_argument("--checkpoint-path", type=str,
                        default="/home/mingi/shared_data/AutoDex/weights/picopose/picopose.pth")
    parser.add_argument("--force-onboard", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-sil", action="store_true",
                        help="Skip silhouette refinement (faster, only pre-error reported)")
    parser.add_argument("--episode-csv", type=str, default=None,
                        help="Path to a foundpose results.csv. If given, evaluate "
                             "the same (object, episode) pairs (apples-to-apples).")
    args = parser.parse_args()

    from autodex.perception.picopose_init import PicoPoseInit
    from autodex.perception.pose_select import select_best_pose_by_iou
    if not args.skip_sil:
        from autodex.perception.silhouette import SilhouetteOptimizer

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_root = Path(args.assets_root).expanduser().resolve()
    assets_root.mkdir(parents=True, exist_ok=True)

    ep_csv = Path(args.episode_csv).expanduser().resolve() if args.episode_csv else None
    obj_eps = _select_objects(experiment_root, args.objects, args.n_episodes,
                              seed=args.seed, episode_csv=ep_csv)
    print(f"[compare] selected {len(obj_eps)} objects:")
    for obj, eps in obj_eps:
        print(f"  - {obj}: {len(eps)} episodes")

    csv_path = output_dir / "results.csv"
    csv_header = [
        "object", "episode",
        "pp_iou_best_view", "pp_iou_mean",
        "trans_err_pre_mm", "rot_err_pre_deg",
        "trans_err_post_mm", "rot_err_post_deg",
        "sil_loss", "sil_sec",
        "pp_compute_sec", "pp_n_views_ok",
    ]
    done: set = set()
    if csv_path.exists():
        with open(csv_path) as fr:
            for row in csv.DictReader(fr):
                done.add((row["object"], row["episode"]))
        print(f"[compare] resuming: {len(done)} (obj, ep) already in {csv_path.name}")
        fcsv = open(csv_path, "a", newline="")
        writer = csv.writer(fcsv)
    else:
        fcsv = open(csv_path, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(csv_header)

    try:
        for obj_name, episodes in obj_eps:
            episodes = [ep for ep in episodes if (obj_name, ep.name) not in done]
            if not episodes:
                print(f"[{obj_name}] all episodes already done, skipping")
                continue

            try:
                mesh_path = _resolve_mesh_path(obj_name)
            except FileNotFoundError as e:
                print(f"[skip {obj_name}] {e}")
                continue

            asset_dir = assets_root / obj_name

            print(f"\n[{obj_name}] init (mesh={mesh_path.name})")
            try:
                pp_init = PicoPoseInit(
                    mesh_path=str(mesh_path),
                    assets_root=str(asset_dir),
                    obj_name=obj_name,
                    object_id=1,
                    device=args.device,
                    checkpoint_path=args.checkpoint_path,
                    force_onboard=args.force_onboard,
                )
            except Exception as exc:
                print(f"[skip {obj_name}] PicoPoseInit failed: {exc}")
                continue

            sil_optimizer = None
            mesh_tensors = None
            glctx = None
            if not args.skip_sil:
                sil_optimizer = SilhouetteOptimizer(str(mesh_path), device="cuda")
                mesh_tensors = sil_optimizer.mesh_tensors
                glctx = sil_optimizer.glctx
            else:
                # We still need glctx + mesh_tensors for IoU select.
                from Utils import make_mesh_tensors
                import nvdiffrast.torch as dr
                import trimesh
                m = trimesh.load(str(mesh_path), process=False)
                if isinstance(m, trimesh.Scene):
                    m = trimesh.util.concatenate(list(m.geometry.values()))
                mesh_tensors = make_mesh_tensors(m, device="cuda")
                glctx = dr.RasterizeCudaContext()

            for ep in episodes:
                p_ref = np.load(ep / "pose_world.npy")
                ep_data = _load_episode(ep)

                t0 = time.perf_counter()
                per_view = pp_init.estimate_per_view(
                    ep_data["images_rgb"], ep_data["masks_bool"],
                    ep_data["intrinsics"], ep_data["extrinsics"],
                )
                pp_compute_sec = time.perf_counter() - t0

                ok_views = {s: r for s, r in per_view.items() if r is not None}
                if not ok_views:
                    print(f"  [{ep.name}] all views failed")
                    continue

                candidates = {s: r["pose_world"] for s, r in ok_views.items()}
                best_iou_serial, p_pre, mean_iou, _ = select_best_pose_by_iou(
                    candidates=candidates,
                    masks=ep_data["masks_bool"],
                    intrinsics=ep_data["intrinsics"],
                    extrinsics=ep_data["extrinsics"],
                    H=ep_data["H"], W=ep_data["W"],
                    glctx=glctx, mesh_tensors=mesh_tensors,
                )
                if p_pre is None:
                    print(f"  [{ep.name}] iou select failed")
                    continue
                t_err_pre, r_err_pre = _pose_errors(p_ref, p_pre)

                if not args.skip_sil:
                    sil_views = []
                    for s, m in ep_data["masks_bool"].items():
                        if int(m.sum()) < 100:
                            continue
                        sil_views.append({
                            "mask": (m.astype(np.uint8) * 255),
                            "K": ep_data["intrinsics"][s].astype(np.float32),
                            "extrinsic": ep_data["extrinsics"][s].astype(np.float64),
                        })
                    t_sil0 = time.perf_counter()
                    p_post, sil_loss = sil_optimizer.optimize(
                        p_pre, sil_views, iters=100, lr=0.002, antialias=True,
                    )
                    sil_sec = time.perf_counter() - t_sil0
                    t_err_post, r_err_post = _pose_errors(p_ref, p_post)
                else:
                    p_post = p_pre
                    sil_loss = float("nan")
                    sil_sec = 0.0
                    t_err_post = t_err_pre
                    r_err_post = r_err_pre

                pose_dir = output_dir / "poses" / obj_name
                pose_dir.mkdir(parents=True, exist_ok=True)
                np.savez(pose_dir / f"{ep.name}.npz", pre=p_pre, post=p_post)

                print(f"  [{ep.name}] iou_best={best_iou_serial} mean_iou={mean_iou:.3f} "
                      f"pre t={t_err_pre:.1f}mm r={r_err_pre:.2f}° -> "
                      f"post t={t_err_post:.1f}mm r={r_err_post:.2f}° "
                      f"(sil_loss={sil_loss:.4f}, {sil_sec:.1f}s) | "
                      f"pp={pp_compute_sec:.1f}s nv={len(ok_views)}/{len(per_view)}")

                writer.writerow([
                    obj_name, ep.name,
                    best_iou_serial, f"{mean_iou:.4f}",
                    f"{t_err_pre:.2f}", f"{r_err_pre:.3f}",
                    f"{t_err_post:.2f}", f"{r_err_post:.3f}",
                    f"{sil_loss:.6f}", f"{sil_sec:.2f}",
                    f"{pp_compute_sec:.2f}", len(ok_views),
                ])
                fcsv.flush()

        print(f"\n[done] wrote {csv_path}")
    finally:
        fcsv.close()


if __name__ == "__main__":
    main()
