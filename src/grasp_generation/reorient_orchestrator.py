"""Adaptive BODex+sim_filter orchestrator for reorient (reset) scenes.

Mirrors `adaptive_orchestrator.py` (box/shelf/wall), but escalates over a
different axis: instead of growing a `gap`, it walks a *config schedule* of
(h_cm, yml_variant, seed_num) — varying lift height and contact / inflation
strategy until at least SUCCESS_THRESHOLD sim-filter-passing grasps exist for
each (i, j) tabletop-pose pair.

Per (obj, i, j) state is tracked across rounds; a round runs BODex + sim_filter
on all scenes that are still unmet at the current schedule step. exp_name is
forced unique per variant so candidate directories don't collide.

Output:
  obj/scene/reorient_{h}/{i}_{j}.json       (from gen_reorient_scene)
  bodex_outputs/{hand}/{exp_name}/{obj}/reorient_{h}/{i}_{j}/{seed}/
  candidates/{hand}/{exp_name}/{obj}/reorient_{h}/{i}_{j}/{seed}/
  {output_dir}/{obj}/reorient_summary.json
"""

import os
import sys
import re
import json
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grasp_generation", "reorient"))

from gen_scene import _obj_dir, gen_reorient_scene  # noqa: E402

from autodex.utils.path import obj_path as default_obj_path  # noqa: E402


# --- Schedules ---
H_SWEEP_CM = [0, 4]
N_SWEEP = [200, 1000]
SUCCESS_THRESHOLD = 5

PARALLEL_PER_N = {200: 10, 1000: 2, 5000: 1}

PYTHON_BODEX = "/home/mingi/miniconda3/envs/bodex/bin/python"
PYTHON_MINGI = "/home/mingi/miniconda3/envs/mingi/bin/python"

HAND_BODEX_CFG_PREFIX = {
    "allegro": "sim_allegro",
    "inspire": "sim_inspire",
    "inspire_left": "sim_inspire_left",
    "inspire_f1": "sim_inspire_f1",
}

BODEX_CFG_ROOT = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex",
                              "src", "curobo", "content", "configs", "manip")


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------

def discover_pose_pairs(obj_name: str, obj_root: str):
    tt_dir = Path(obj_root) / obj_name / "processed_data" / "info" / "tabletop"
    ids = sorted(int(p.stem) for p in tt_dir.glob("*.npy"))
    return [(i, j) for i in ids for j in ids if i != j]


def ensure_scenes(obj_name: str, h_cm: int, obj_root: str, pairs):
    h_m = h_cm / 100.0
    out_dir = Path(obj_root) / obj_name / "scene" / f"reorient_{h_cm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, j in pairs:
        p = out_dir / f"{i}_{j}.json"
        if p.exists():
            written.append(f"{i}_{j}")
            continue
        scene = gen_reorient_scene(obj_name, i, j, h_m)
        scene["meta"]["scene_type"] = f"reorient_{h_cm}"
        with open(p, "w") as f:
            json.dump(scene, f, indent=2)
        written.append(f"{i}_{j}")
    return written


# ---------------------------------------------------------------------------
# YAML variant discovery
# ---------------------------------------------------------------------------

def discover_yml_variants(hand: str, h_cm: int):
    """Return ordered list of (variant_tag, yml_relpath) for this (hand, h_cm).

    Base reorient first, then inflate variants. The new pressure_constraints
    treats all fingers equally, so pinch variants are skipped.
    `variant_tag` is appended to exp_name so output dirs don't collide.
    """
    prefix = HAND_BODEX_CFG_PREFIX[hand]
    cfg_dir = Path(BODEX_CFG_ROOT) / prefix
    out = []
    base = cfg_dir / f"paradex_reorient_{h_cm}.yml"
    if base.is_file():
        out.append(("base", f"{prefix}/{base.name}"))

    # inflate variants: paradex_reorient_{N}_inflate*.yml
    inflate_re = re.compile(rf"^paradex_reorient_{h_cm}_(\w+)\.yml$")
    for p in sorted(cfg_dir.glob(f"paradex_reorient_{h_cm}_*.yml")):
        m = inflate_re.match(p.name)
        if m:
            out.append((m.group(1), f"{prefix}/{p.name}"))
    return out


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_bodex(yml_relpath, exp_name, obj_list_file, scene_type, scene_ids,
              N, obj_root_dir=None):
    parallel = PARALLEL_PER_N.get(N, 1)
    bodex_dir = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex")
    filter_file = os.path.join("/tmp", f"reorient_scene_filter_{os.getpid()}.json")
    with open(filter_file, "w") as f:
        json.dump({scene_type: list(scene_ids)}, f)
    cmd = [
        PYTHON_BODEX, "generate.py",
        "-c", yml_relpath,
        "-w", str(parallel),
        "--obj_list_file", obj_list_file,
        "--seed_num", str(N),
        "--exp_name", exp_name,
        "--scene_filter_file", filter_file,
    ]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.run(cmd, cwd=bodex_dir, env=env, capture_output=True, text=True)
    try:
        os.remove(filter_file)
    except OSError:
        pass
    if proc.returncode != 0:
        print(f"[BODex FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"BODex failed (rc={proc.returncode})")
    return proc.stdout


def run_sim_filter(hand, exp_name, obj_name, obj_root_dir=None):
    script = os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter", "run_sim_filter.py")
    cmd = [PYTHON_MINGI, script, "--hand", hand, "--version", exp_name, "--obj", obj_name]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[sim_filter FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"sim_filter failed (rc={proc.returncode})")
    return proc.stdout


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def count_passing(hand, exp_name, obj_name, scene_type, scene_id):
    cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                            obj_name, scene_type, scene_id)
    if not os.path.isdir(cand_dir):
        return 0
    return sum(1 for d in os.listdir(cand_dir)
               if os.path.isdir(os.path.join(cand_dir, d)))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_obj(obj_name, hand, obj_root, obj_list_file,
                h_sweep=H_SWEEP_CM, n_sweep=N_SWEEP, threshold=SUCCESS_THRESHOLD):
    """Walk schedule of (h, variant, N). Per (i, j) scene, mark done when count >= threshold."""
    pairs = discover_pose_pairs(obj_name, obj_root)
    if not pairs:
        return {"status": "no_tabletop_poses"}

    # Per (h, i_j) state. Same (i, j) at different h is treated independently
    # (different scene; reset h is a task parameter).
    state = {}  # (h_cm, scene_id) -> {"valid": int, "done": bool, "history": [...]}
    for h_cm in h_sweep:
        ensure_scenes(obj_name, h_cm, obj_root, pairs)
        for i, j in pairs:
            state[(h_cm, f"{i}_{j}")] = {"valid": 0, "done": False, "history": []}

    # Build schedule: outer h, middle variant, inner N
    for h_cm in h_sweep:
        variants = discover_yml_variants(hand, h_cm)
        if not variants:
            print(f"  [h={h_cm}] no yml variants — skip")
            continue
        scene_type = f"reorient_{h_cm}"

        for variant_tag, yml_relpath in variants:
            exp_name = f"reset_{h_cm}" if variant_tag == "base" else f"reset_{h_cm}_{variant_tag}"
            for N in n_sweep:
                active_ids = [sid for (h, sid), s in state.items()
                              if h == h_cm and not s["done"]]
                if not active_ids:
                    break  # all (i, j) at this h satisfied
                print(f"  [h={h_cm}] variant={variant_tag} N={N}: {len(active_ids)} active "
                      f"(exp={exp_name})")
                run_bodex(yml_relpath, exp_name, obj_list_file, scene_type,
                          active_ids, N,
                          obj_root_dir=obj_root if obj_root != default_obj_path else None)
                run_sim_filter(hand, exp_name, obj_name,
                               obj_root_dir=obj_root if obj_root != default_obj_path else None)
                for sid in active_ids:
                    cnt = count_passing(hand, exp_name, obj_name, scene_type, sid)
                    s = state[(h_cm, sid)]
                    s["valid"] = max(s["valid"], cnt)
                    s["history"].append({"variant": variant_tag, "exp": exp_name,
                                          "N": N, "valid": cnt})
                    if cnt >= threshold:
                        s["done"] = True
                cnts = sorted([state[(h_cm, sid)]["valid"] for sid in active_ids],
                              reverse=True)[:5]
                print(f"    -> top5 valid: {cnts}")

    return {f"{h}_{sid}": s for (h, sid), s in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=list(HAND_BODEX_CFG_PREFIX))
    parser.add_argument("--obj", required=True)
    parser.add_argument("--obj_root", default=default_obj_path)
    parser.add_argument("--h_sweep", type=int, nargs="+", default=H_SWEEP_CM,
                        help="Lift heights in cm to try, in order (default: 0 4)")
    parser.add_argument("--n_sweep", type=int, nargs="+", default=N_SWEEP)
    parser.add_argument("--threshold", type=int, default=SUCCESS_THRESHOLD)
    parser.add_argument("--output_dir", default=None,
                        help="Where to write reorient_summary.json (default: REPO/logging/reorient/)")
    args = parser.parse_args()

    obj_list_file = os.path.join("/tmp", f"reorient_obj_list_{os.getpid()}.txt")
    with open(obj_list_file, "w") as f:
        f.write(args.obj + "\n")

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "logging", "reorient", args.hand)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {args.obj} ({args.hand}) ===")
    summary = process_obj(args.obj, args.hand, args.obj_root, obj_list_file,
                          h_sweep=args.h_sweep, n_sweep=args.n_sweep,
                          threshold=args.threshold)

    summary_path = os.path.join(out_dir, f"{args.obj}_reorient_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {summary_path}")

    try:
        os.remove(obj_list_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
