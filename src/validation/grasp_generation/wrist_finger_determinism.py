"""Per-object wrist->finger determinism analysis for inspire_left grasps.

For each object, gathers all (wrist_se3, grasp_pose) pairs from candidates dir,
computes KNN-based conditional std of finger joints given wrist pose, and
reports the ratio vs global finger std. Ratio << 1 means wrist effectively
determines finger configuration; ratio ~ 1 means independent.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


CAND_ROOT = Path.home() / "AutoDex" / "candidates"


def wrist_feature(se3):
    """4x4 SE(3) -> 9D feature: translation + first 2 columns of rotation (6D rot rep)."""
    t = se3[:3, 3]
    r6 = se3[:3, :2].reshape(-1)  # 6D
    return np.concatenate([t, r6])


def collect_object(obj_dir):
    wrists, grasps = [], []
    for w in obj_dir.rglob("wrist_se3.npy"):
        g = w.parent / "grasp_pose.npy"
        if not g.exists():
            continue
        wrists.append(np.load(w))
        grasps.append(np.load(g))
    if not wrists:
        return None, None
    return np.stack(wrists), np.stack(grasps)


def determinism_score(wrists, grasps, k=5):
    """Returns dict with per-joint and overall determinism ratios.

    ratio = mean local std (within k-NN wrist neighborhood) / global std.
    """
    feats = np.stack([wrist_feature(w) for w in wrists])
    # Z-normalize wrist features so trans/rot scaled comparably
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)

    n = len(feats)
    k_eff = min(k + 1, n)  # +1 because the point itself is included
    nn = NearestNeighbors(n_neighbors=k_eff).fit(feats)
    _, idx = nn.kneighbors(feats)

    local_std = np.zeros((n, grasps.shape[1]))
    for i in range(n):
        local_std[i] = grasps[idx[i]].std(0)
    mean_local = local_std.mean(0)
    global_std = grasps.std(0)
    ratio = mean_local / (global_std + 1e-9)

    return {
        "n": int(n),
        "global_std": global_std.tolist(),
        "mean_local_std": mean_local.tolist(),
        "ratio_per_joint": ratio.tolist(),
        "ratio_mean": float(ratio.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--version", default="selected_100")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = CAND_ROOT / args.hand / args.version
    out_path = Path(args.out) if args.out else (
        Path(__file__).parent / f"determinism_{args.hand}_{args.version}.json"
    )

    results = {}
    obj_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Found {len(obj_dirs)} objects under {root}")

    for od in obj_dirs:
        wrists, grasps = collect_object(od)
        if wrists is None or len(wrists) < args.k + 2:
            print(f"  skip {od.name} (n={0 if wrists is None else len(wrists)})")
            continue
        score = determinism_score(wrists, grasps, k=args.k)
        results[od.name] = score
        print(f"  {od.name:40s} n={score['n']:4d}  ratio_mean={score['ratio_mean']:.3f}")

    ratios = np.array([v["ratio_mean"] for v in results.values()])
    summary = {
        "hand": args.hand,
        "version": args.version,
        "k": args.k,
        "n_objects": len(results),
        "ratio_mean_overall": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "ratio_p25": float(np.percentile(ratios, 25)),
        "ratio_p75": float(np.percentile(ratios, 75)),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_object": results}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
