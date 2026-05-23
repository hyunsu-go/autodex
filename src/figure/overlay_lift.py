"""Save figure-style overlays (object + robot mesh) for one experiment trial.

- Multiple frames in one run: --frames F1 F2 ... or --frame_range start end step
- Default: render the "done" state (end of lift) if result.json present, else peak EE Z
- Thumb mimic overrides: generate a temp URDF with patched right_thumb_3/4 mimic
  multiplier+offset so the user can sweep mimic values without editing the URDF

Output:
    outputs/figure/overlay/{hand}/{obj}/{date}[__{out_tag}]/
        meta.json
        frame_{idx:04d}/
            raw/{serial}.png
            overlay/{serial}.png
"""
import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
PARADEX_ROOT = Path.home() / "paradex"
sys.path.insert(0, str(PARADEX_ROOT))
sys.path.insert(0, str(REPO))

from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule

from autodex.utils.sync import convert_inspire_raw, ALLEGRO_POS_TO_URDF


def _import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_obj_mod = _import_from_path(
    "overlay_object_video_single",
    REPO / "src" / "visualization" / "overlay_object_video_single.py",
)
_rob_mod = _import_from_path(
    "overlay_robot_video",
    REPO / "src" / "visualization" / "overlay_robot_video.py",
)
ObjectOverlayRenderer = _obj_mod.ObjectOverlayRenderer
RobotOverlayRenderer = _rob_mod.RobotOverlayRenderer
_label_for_link = _rob_mod._label_for_link


EXP_BASE = Path.home() / "shared_data" / "AutoDex" / "experiment"
OBJ_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"
URDF_BASE = REPO / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"


# ── small helpers ────────────────────────────────────────────────────────────

def _state_time(result_json, state_name):
    states = result_json["timing"]["execution_states"]
    for s in states:
        if s["state"] == state_name:
            return datetime.fromisoformat(s["time"]).timestamp()
    available = [s["state"] for s in states]
    raise ValueError(f"state '{state_name}' not in result.json (have: {available})")


def _read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _load_pose_at_frame(records_path, frame_idx):
    records = json.load(open(records_path))
    pose_by_idx = {int(r["frame_index"]): r for r in records}
    if frame_idx in pose_by_idx and pose_by_idx[frame_idx].get("pose_world"):
        return np.array(pose_by_idx[frame_idx]["pose_world"], dtype=np.float64)
    nearest = min(pose_by_idx.keys(), key=lambda k: abs(k - frame_idx))
    pose = pose_by_idx[nearest].get("pose_world")
    if pose is None:
        return None
    return np.array(pose, dtype=np.float64)


def _sync_one_frame(exp_dir, hand_type, t_target,
                    arm_time_offset=0.03, hand_time_offset=0.03):
    """Single-frame robust qpos sync (nearest-neighbor; tolerates length mismatch).
    If raw/hand/ is absent, falls back to squeeze_hand.npy as a constant.
    """
    arm_dir = exp_dir / "raw" / "arm"
    hand_dir = exp_dir / "raw" / "hand"

    arm_t = np.load(arm_dir / "time.npy") + arm_time_offset
    arm_p = np.load(arm_dir / "position.npy")
    n = min(len(arm_t), len(arm_p))
    arm_t, arm_p = arm_t[:n], arm_p[:n]
    arm_row = arm_p[int(np.argmin(np.abs(arm_t - t_target)))]

    has_hand_time = (hand_dir / "time.npy").exists() and (hand_dir / "position.npy").exists()
    if has_hand_time:
        hand_t = np.load(hand_dir / "time.npy") + hand_time_offset
        hand_p = np.load(hand_dir / "position.npy")
        n = min(len(hand_t), len(hand_p))
        hand_t, hand_p = hand_t[:n], hand_p[:n]
        hidx = int(np.argmin(np.abs(hand_t - t_target)))
        hand_row = hand_p[hidx:hidx + 1]
    else:
        sq_path = exp_dir / "squeeze_hand.npy"
        if not sq_path.exists():
            raise FileNotFoundError(
                f"no raw/hand/ and no squeeze_hand.npy in {exp_dir}")
        hand_row = np.load(sq_path).reshape(1, -1)

    if hand_type in ("inspire", "inspire_left"):
        hand_row = convert_inspire_raw(hand_row)
    elif hand_type == "allegro":
        hand_row = hand_row[:, ALLEGRO_POS_TO_URDF]
    return arm_row, hand_row[0]


def _patch_thumb_mimic(src_urdf, hand_type,
                       t3_mult, t3_off, t4_mult, t4_off):
    """Write a sibling URDF with right/left_thumb_3/4_joint mimic values patched.
    Returns the new path; mesh refs are relative so it MUST live in the same dir.
    """
    side = "left" if hand_type == "inspire_left" else "right"
    txt = Path(src_urdf).read_text()

    def patch(txt, owner, mult, off):
        pat = re.compile(
            r'(<joint name="' + re.escape(owner) + r'"[^>]*>.*?<mimic joint="[^"]+" multiplier=)"[^"]+"( offset=)"[^"]+"',
            re.S,
        )
        new, n = pat.subn(rf'\1"{mult}"\2"{off}"', txt, count=1)
        if n != 1:
            raise RuntimeError(f"failed to patch mimic for {owner}")
        return new

    txt = patch(txt, f"{side}_thumb_3_joint", t3_mult, t3_off)
    txt = patch(txt, f"{side}_thumb_4_joint", t4_mult, t4_off)

    src = Path(src_urdf)
    dst = src.parent / f"{src.stem}_tmp_thumb.urdf"
    dst.write_text(txt)
    return dst


def _resolve_frames(args, video_times, max_frame, exp_dir):
    """Return list[int] of video frame indices to render, plus a one-line source string."""
    if args.frames:
        out = sorted({int(f) for f in args.frames})
        return out, f"explicit list {out}"
    if args.frame_range:
        s, e, st = args.frame_range
        out = list(range(int(s), int(e) + 1, int(st)))
        return out, f"range {s}..{e} step {st}"
    if args.frame is not None:
        return [int(args.frame)], f"single explicit frame={args.frame}"
    if (exp_dir / "result.json").exists():
        result = json.load(open(exp_dir / "result.json"))
        t_target = _state_time(result, args.state)
        return [int(np.argmin(np.abs(video_times[:max_frame + 1] - t_target)))], \
               f"state={args.state}"
    # peak EE Z fallback
    arm_act = np.load(exp_dir / "raw" / "arm" / "action.npy")
    arm_t = np.load(exp_dir / "raw" / "arm" / "time.npy")
    n = min(len(arm_t), len(arm_act))
    arm_t, arm_act = arm_t[:n], arm_act[:n]
    t_lo, t_hi = float(video_times[0]), float(video_times[max_frame])
    in_range = (arm_t >= t_lo) & (arm_t <= t_hi)
    z = np.where(in_range, arm_act[:, 2], -np.inf)
    t_target = float(arm_t[int(np.argmax(z))])
    return [int(np.argmin(np.abs(video_times[:max_frame + 1] - t_target)))], \
           "peak EE z"


def _get_qpos(exp_dir, hand_type, frame_idx, t_target):
    if (exp_dir / "arm" / "state.npy").exists() and (exp_dir / "hand" / "state.npy").exists():
        arm_state = np.load(exp_dir / "arm" / "state.npy")
        hand_state = np.load(exp_dir / "hand" / "state.npy")
        return np.concatenate([arm_state[frame_idx], hand_state[frame_idx]], axis=-1)
    arm_row, hand_row = _sync_one_frame(exp_dir, hand_type, t_target)
    return np.concatenate([arm_row, hand_row], axis=-1)


def _load_thumb2_calibration(calib_dir):
    """Build interp arrays from {calib_dir}/{N}/lookup_thumb.json.
    Returns (regs, t2, t3, t4) sorted by reg.
    """
    rows = []
    for sub in sorted(Path(calib_dir).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1e18):
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        fp = sub / "lookup_thumb.json"
        if not fp.exists():
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        v = next(iter(d.values()))
        if v.get("thumb_2") is None:
            continue
        rows.append((int(sub.name), v["thumb_2"], v.get("thumb_3", np.nan), v.get("thumb_4", np.nan)))
    rows.sort()
    regs = np.array([r[0] for r in rows])
    t2 = np.array([r[1] for r in rows])
    t3 = np.array([r[2] for r in rows])
    t4 = np.array([r[3] for r in rows])
    return regs, t2, t3, t4


def _raw_thumb2_register(exp_dir, frame_idx, t_target, hand_time_offset=0.03):
    """Return the controller thumb_2 register (0-1000) at the frame's time.
    Raw layout: [little, ring, middle, index, thumb_2, thumb_1] -> col 4.
    """
    raw_pos = np.load(exp_dir / "raw" / "hand" / "position.npy")
    raw_t = np.load(exp_dir / "raw" / "hand" / "time.npy") + hand_time_offset
    n = min(len(raw_pos), len(raw_t))
    raw_pos, raw_t = raw_pos[:n], raw_t[:n]
    ridx = int(np.argmin(np.abs(raw_t - t_target)))
    return int(raw_pos[ridx, 4])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--exp_name", default="selected_100")
    ap.add_argument("--out_root", default=str(REPO / "outputs" / "figure" / "overlay"))
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--urdf", default=None,
                    help="override URDF path")

    # frame selection
    ap.add_argument("--frame", type=int, default=None, help="single frame")
    ap.add_argument("--frames", nargs="+", type=int, default=None,
                    help="list of frame indices, e.g. --frames 100 200 300 358")
    ap.add_argument("--frame_range", nargs=3, type=int, default=None,
                    metavar=("START", "END", "STEP"),
                    help="frame range, e.g. --frame_range 0 358 30")
    ap.add_argument("--state", default="done",
                    help="execution_state name when no frame given (default: done)")
    ap.add_argument("--serial", nargs="+", default=None,
                    help="restrict to specific camera serial(s)")

    # thumb mimic override (right/left_thumb_3_joint and right/left_thumb_4_joint)
    ap.add_argument("--thumb3_mult", type=float, default=None)
    ap.add_argument("--thumb3_offset", type=float, default=0.0)
    ap.add_argument("--thumb4_mult", type=float, default=None)
    ap.add_argument("--thumb4_offset", type=float, default=0.0)

    # Per-frame thumb_2 override (sets URDF thumb_2_joint angle directly,
    # bypassing the controller's linear formula). Useful when calibration shows
    # the controller register underestimates the true thumb_2 angle.
    ap.add_argument("--thumb1_override", type=float, default=None,
                    help="set URDF thumb_1_joint angle (rad) directly for all frames")
    ap.add_argument("--thumb2_override", type=float, default=None,
                    help="set URDF thumb_2_joint angle (rad) directly for all frames")
    ap.add_argument("--palette", default="current",
                    choices=["current", "mono", "pastel", "current_low_alpha"],
                    help="robot mesh color palette")

    args = ap.parse_args()

    # Palette: monkey-patch overlay_robot_video module-level constants
    PALETTES = {
        "current": {
            "ARM_COLOR": (40, 200, 40), "ARM_ALPHA": 0.35, "FINGER_ALPHA": 0.55,
            "FINGER_COLORS": {
                "thumb":  (255, 140,   0), "index":  (  0, 200, 255),
                "middle": (  0, 255, 100), "ring":   (255,   0, 200),
                "pinky":  (255, 220,   0),
            },
        },
        "mono": {
            "ARM_COLOR": (60, 60, 60), "ARM_ALPHA": 0.35, "FINGER_ALPHA": 0.5,
            "FINGER_COLORS": {f: (100, 150, 220) for f in ("thumb","index","middle","ring","pinky")},
        },
        "pastel": {
            "ARM_COLOR": (130, 130, 130), "ARM_ALPHA": 0.35, "FINGER_ALPHA": 0.55,
            "FINGER_COLORS": {
                "thumb":  (252, 141,  98), "index":  (102, 194, 165),
                "middle": (141, 160, 203), "ring":   (231, 138, 195),
                "pinky":  (166, 216,  84),
            },
        },
        "current_low_alpha": {
            "ARM_COLOR": (40, 200, 40), "ARM_ALPHA": 0.25, "FINGER_ALPHA": 0.35,
            "FINGER_COLORS": {
                "thumb":  (255, 140,   0), "index":  (  0, 200, 255),
                "middle": (  0, 255, 100), "ring":   (255,   0, 200),
                "pinky":  (255, 220,   0),
            },
        },
    }
    pal = PALETTES[args.palette]
    _rob_mod.ARM_COLOR = pal["ARM_COLOR"]
    _rob_mod.ARM_ALPHA = pal["ARM_ALPHA"]
    _rob_mod.FINGER_ALPHA = pal["FINGER_ALPHA"]
    _rob_mod.FINGER_COLORS = pal["FINGER_COLORS"]
    print(f"[info] palette: {args.palette}")

    exp_dir = EXP_BASE / args.exp_name / args.hand / args.obj / args.date
    if not exp_dir.is_dir():
        raise FileNotFoundError(exp_dir)

    # discover usable frame range from videos
    videos_dir = exp_dir / "videos"
    video_frame_counts = []
    for vp in sorted(videos_dir.glob("*.avi")):
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            video_frame_counts.append(n)
    video_times = np.load(exp_dir / "raw" / "timestamps" / "timestamp.npy")
    max_frame = (min(video_frame_counts) - 1) if video_frame_counts else (len(video_times) - 1)
    print(f"[info] usable frame range: [0, {max_frame}]  "
          f"(video counts min/max = {min(video_frame_counts)}/{max(video_frame_counts)})")

    frames, src_msg = _resolve_frames(args, video_times, max_frame, exp_dir)
    frames = [max(0, min(f, max_frame)) for f in frames]
    print(f"[info] frames: {frames}  (source: {src_msg})")

    # resolve URDF, optionally patch thumb mimic
    base_urdf = Path(args.urdf) if args.urdf else (
        URDF_BASE / f"{args.hand}_description" / f"xarm_{args.hand}.urdf"
    )
    patched_urdf = None
    if args.thumb3_mult is not None or args.thumb4_mult is not None:
        # need both to make a complete patch — fill missing from base URDF defaults
        # (parse current values)
        side = "left" if args.hand == "inspire_left" else "right"
        txt = base_urdf.read_text()
        def _cur(owner):
            m = re.search(
                r'<joint name="' + re.escape(owner) + r'"[^>]*>.*?<mimic joint="[^"]+" multiplier="([^"]+)" offset="([^"]+)"',
                txt, re.S,
            )
            if not m:
                raise RuntimeError(f"no mimic for {owner} in {base_urdf}")
            return float(m.group(1)), float(m.group(2))
        cur_t3 = _cur(f"{side}_thumb_3_joint")
        cur_t4 = _cur(f"{side}_thumb_4_joint")
        t3m = args.thumb3_mult if args.thumb3_mult is not None else cur_t3[0]
        t3o = args.thumb3_offset if args.thumb3_mult is not None else cur_t3[1]
        t4m = args.thumb4_mult if args.thumb4_mult is not None else cur_t4[0]
        t4o = args.thumb4_offset if args.thumb4_mult is not None else cur_t4[1]
        patched_urdf = _patch_thumb_mimic(base_urdf, args.hand, t3m, t3o, t4m, t4o)
        urdf_path = patched_urdf
        print(f"[info] thumb mimic patched: thumb_3 mult={t3m} off={t3o}, "
              f"thumb_4 mult={t4m} off={t4o}")
    else:
        urdf_path = base_urdf
    print(f"[info] urdf: {urdf_path}")

    # camera params (constant across frames)
    intrinsic, extrinsic_cw = load_camparam(str(exp_dir))
    c2r = np.load(exp_dir / "C2R.npy")
    serials = sorted(s for s in intrinsic if (videos_dir / f"{s}.avi").exists())
    if args.serial:
        wanted = set(args.serial)
        serials = [s for s in serials if s in wanted]
    if not serials:
        raise RuntimeError("no overlapping serials between intrinsics and videos/")

    # object mesh (constant across frames)
    track_path = exp_dir / "object_tracking" / "gotrack_output" / "world_pose_records.json"
    has_track = track_path.exists()
    static_pose_path = exp_dir / "pose_world.npy"

    # robot module (constant; we only update_cfg per frame)
    robot = RobotModule(str(urdf_path))
    dof = robot.get_num_joints()

    # read a frame from each cam first to know H, W
    probe_frame = frames[0]
    raw0 = {}
    for s in serials:
        fr = _read_frame(videos_dir / f"{s}.avi", probe_frame)
        if fr is not None:
            raw0[s] = fr
    serials = [s for s in serials if s in raw0]
    H, W = raw0[serials[0]].shape[:2]
    intrinsics_K = {s: intrinsic[s]["intrinsics_undistort"] for s in serials}
    extr_cw_clean = {s: extrinsic_cw[s] for s in serials}

    # build robot mesh structures once
    robot.update_cfg(np.zeros(dof))
    scene = robot.scene
    link_names_ordered = list(scene.geometry.keys())
    scene_meshes = [scene.geometry[ln] for ln in link_names_ordered]
    link_labels = {ln: _label_for_link(ln) for ln in link_names_ordered}

    intrinsic_subset = {s: intrinsic[s] for s in serials}
    robot_extr_world = {s: extr_cw_clean[s][:3, :] for s in serials}
    robot_renderer = RobotOverlayRenderer(
        scene_meshes, link_names_ordered, link_labels,
        intrinsic_subset, robot_extr_world, H, W,
    )
    ordered = robot_renderer.serials

    # object renderer (constructed once; mesh constant, pose varies per frame)
    obj_renderer = None
    if has_track or static_pose_path.exists():
        mesh_path = OBJ_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
        mesh = trimesh.load(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        obj_renderer = ObjectOverlayRenderer(mesh, intrinsics_K, extr_cw_clean, H, W)
        assert obj_renderer.serials == ordered

    # output dir
    date_dir = args.date if not args.out_tag else f"{args.date}__{args.out_tag}"
    out_root = Path(args.out_root) / args.hand / args.obj / date_dir
    out_root.mkdir(parents=True, exist_ok=True)

    per_frame_meta = []
    static_pose = np.load(static_pose_path) if static_pose_path.exists() else None

    try:
        for frame_idx in frames:
            t_target = float(video_times[frame_idx])

            # frames per camera
            raw_frames = {}
            for s in ordered:
                fr = _read_frame(videos_dir / f"{s}.avi", frame_idx)
                if fr is None:
                    print(f"[warn] failed to read frame {frame_idx} from {s}.avi")
                    fr = np.zeros((H, W, 3), dtype=np.uint8)
                raw_frames[s] = fr
            frames_list = [raw_frames[s] for s in ordered]

            # qpos
            qpos = _get_qpos(exp_dir, args.hand, frame_idx, t_target)
            # URDF qpos layout: [arm6, thumb_1, thumb_2, index_1, middle_1, ring_1, little_1]
            if args.thumb1_override is not None:
                qpos[6] = args.thumb1_override
            if args.thumb2_override is not None:
                qpos[7] = args.thumb2_override
            robot.update_cfg(qpos[:dof])
            scene_now = robot.scene
            link_poses_robot = [scene_now.graph.get(ln)[0] for ln in link_names_ordered]
            link_poses_world = [c2r @ p for p in link_poses_robot]

            # object pose
            if has_track:
                pose_world = _load_pose_at_frame(track_path, frame_idx)
                pose_kind = "gotrack"
            elif static_pose is not None:
                pose_world = static_pose
                pose_kind = "static pose_world.npy"
            else:
                pose_world = None
                pose_kind = "none"

            # render
            after_obj = obj_renderer.render(pose_world, frames_list) if (obj_renderer and pose_world is not None) else frames_list
            final = robot_renderer.render(link_poses_world, after_obj)

            # save
            frame_dir = out_root / f"frame_{frame_idx:04d}"
            (frame_dir / "raw").mkdir(parents=True, exist_ok=True)
            (frame_dir / "overlay").mkdir(parents=True, exist_ok=True)
            for s, raw, ovl in zip(ordered, frames_list, final):
                cv2.imwrite(str(frame_dir / "raw" / f"{s}.png"), raw)
                cv2.imwrite(str(frame_dir / "overlay" / f"{s}.png"), ovl)

            per_frame_meta.append({
                "frame_idx": int(frame_idx),
                "t": t_target,
                "object_pose_source": pose_kind,
            })
            print(f"[done] frame {frame_idx}: {len(ordered)} views -> {frame_dir}")

    finally:
        if patched_urdf is not None and patched_urdf.exists():
            patched_urdf.unlink()

    meta = {
        "hand": args.hand,
        "obj": args.obj,
        "date": args.date,
        "exp_name": args.exp_name,
        "frame_source": src_msg,
        "n_frames": len(per_frame_meta),
        "frames": per_frame_meta,
        "serials": ordered,
        "urdf_base": str(base_urdf),
        "thumb3_mult": args.thumb3_mult,
        "thumb3_offset": args.thumb3_offset,
        "thumb4_mult": args.thumb4_mult,
        "thumb4_offset": args.thumb4_offset,
        "out_tag": args.out_tag,
    }
    json.dump(meta, open(out_root / "meta.json", "w"), indent=2)
    print(f"\n[all done] wrote {len(per_frame_meta)} frames to {out_root}")


if __name__ == "__main__":
    main()
