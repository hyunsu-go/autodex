"""
Render heatmap(s) of sweep_reset results — which (pickup_x, pickup_tz) cells are
resetable per (i, j) tabletop-pose pair.

Three input modes (pick one):
    # 1) Single pair
    python src/grasp_generation/reorient/plot_sweep.py \
        --sweep_dir outputs/reset_cache/inspire_left/attached_container/reorient_0/0_16

    # 2) Object root — render every pair under it, save heatmap.png per pair
    python src/grasp_generation/reorient/plot_sweep.py \
        --obj attached_container --h_cm 0 --hand inspire_left

    # 3) Same as (2) but also write a single combined grid PNG
    python src/grasp_generation/reorient/plot_sweep.py \
        --obj attached_container --h_cm 0 --combined
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autodex.utils.path import repo_dir


def plot_single_pair(sweep_dir: Path, out_path: Path = None) -> tuple:
    """Render one pair's heatmap. Returns (n_ok, n_total, summary)."""
    summary_file = sweep_dir / "sweep_summary.json"
    if not summary_file.exists():
        return 0, 0, None
    summary = json.load(open(summary_file))

    xs = summary["x_values"]
    tzs = summary["tz_values"]
    nx, ntz = len(xs), len(tzs)

    grid = np.zeros((ntz, nx), dtype=np.int8)
    for c in summary["cells"]:
        if c["x"] in xs and c["tz"] in tzs:
            grid[tzs.index(c["tz"]), xs.index(c["x"])] = 1 if c["status"] == "ok" else 0

    n_ok = int(grid.sum()); n_tot = grid.size
    title = (f"{summary['obj_name']}  reorient_{summary['h_cm']}  "
             f"{summary['i']}→{summary['j']}  hand={summary['hand']}  "
             f"ok={n_ok}/{n_tot}")

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * nx + 2), max(4, 0.4 * ntz + 1.5)))
    ax.imshow(grid, aspect="auto", cmap=plt.get_cmap("RdYlGn"),
              vmin=0, vmax=1, origin="lower",
              extent=[min(xs) - 0.025, max(xs) + 0.025,
                      min(tzs) - 15, max(tzs) + 15])
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=45)
    ax.set_yticks(tzs); ax.set_yticklabels([f"{int(t)}°" for t in tzs])
    ax.set_xlabel("pickup x (m)"); ax.set_ylabel("pickup θ_z (deg)")
    ax.set_title(title)
    for c in summary["cells"]:
        if c["x"] in xs and c["tz"] in tzs:
            mark = "✓" if c["status"] == "ok" else "✗"
            ax.text(c["x"], c["tz"], mark, ha="center", va="center",
                    color="black" if c["status"] == "ok" else "white", fontsize=9)
    plt.tight_layout()
    if out_path is None:
        out_path = sweep_dir / "heatmap.png"
    plt.savefig(out_path, dpi=120); plt.close(fig)
    return n_ok, n_tot, summary


def plot_combined(obj_root: Path, pair_dirs: list, out_path: Path):
    """Single PNG with subplot grid for all pairs."""
    n = len(pair_dirs)
    if n == 0:
        return
    cols = min(5, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.8 * rows),
                              squeeze=False)
    for idx, pd in enumerate(pair_dirs):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        summary_file = pd / "sweep_summary.json"
        if not summary_file.exists():
            ax.set_title(f"{pd.name} (no data)"); ax.axis("off"); continue
        summary = json.load(open(summary_file))
        xs = summary["x_values"]; tzs = summary["tz_values"]
        grid = np.zeros((len(tzs), len(xs)), dtype=np.int8)
        for cell in summary["cells"]:
            if cell["x"] in xs and cell["tz"] in tzs:
                grid[tzs.index(cell["tz"]), xs.index(cell["x"])] = 1 if cell["status"] == "ok" else 0
        n_ok = int(grid.sum()); n_tot = grid.size
        ax.imshow(grid, aspect="auto", cmap=plt.get_cmap("RdYlGn"),
                  vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=45, fontsize=7)
        ax.set_yticks(range(len(tzs))); ax.set_yticklabels([f"{int(t)}" for t in tzs], fontsize=7)
        ax.set_title(f"{summary['i']}→{summary['j']}  {n_ok}/{n_tot}", fontsize=10)
    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")
    fig.suptitle(f"{pair_dirs[0].parent.parent.name}  /  {pair_dirs[0].parent.name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    # Mode A: single pair
    p.add_argument("--sweep_dir", default=None,
                    help="explicit pair dir (containing sweep_summary.json)")
    # Mode B: object root — process all pairs
    p.add_argument("--obj", default=None)
    p.add_argument("--h_cm", type=int, default=None)
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--obj_root", default=None,
                    help="override outputs/reset_cache/{hand}/{obj}/reorient_{h_cm}")
    p.add_argument("--combined", action="store_true",
                    help="also write a combined grid PNG with all pairs")
    p.add_argument("--out", default=None,
                    help="output PNG (single-pair mode only)")
    args = p.parse_args()

    if args.sweep_dir:
        sd = Path(args.sweep_dir)
        out_path = Path(args.out) if args.out else (sd / "heatmap.png")
        n_ok, n_tot, _ = plot_single_pair(sd, out_path)
        print(f"[plot] {sd.name}: {n_ok}/{n_tot} → {out_path}")
        return

    if not args.obj or args.h_cm is None:
        p.error("specify --sweep_dir, or (--obj + --h_cm)")

    obj_root = Path(args.obj_root) if args.obj_root else (
        Path(repo_dir) / "outputs" / "reset_cache" / args.hand / args.obj
        / f"reorient_{args.h_cm}"
    )
    if not obj_root.exists():
        print(f"[plot] no sweep root: {obj_root}")
        return

    pair_dirs = sorted(
        [d for d in obj_root.iterdir()
         if d.is_dir() and (d / "sweep_summary.json").exists()],
        key=lambda d: tuple(int(x) for x in d.name.split("_")),
    )
    if not pair_dirs:
        print(f"[plot] no pair sweeps under {obj_root}")
        return

    grand_ok = grand_total = 0
    for pd in pair_dirs:
        n_ok, n_tot, _ = plot_single_pair(pd)
        grand_ok += n_ok; grand_total += n_tot
        print(f"[plot] {pd.name}: {n_ok}/{n_tot} → {pd / 'heatmap.png'}")

    print(f"\n[plot] all pairs done: {grand_ok}/{grand_total} ok across {len(pair_dirs)} pairs")

    if args.combined:
        combined_out = obj_root / "heatmap_combined.png"
        plot_combined(obj_root, pair_dirs, combined_out)
        print(f"[plot] combined -> {combined_out}")


if __name__ == "__main__":
    main()
