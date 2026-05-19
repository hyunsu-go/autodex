import os
import random
import numpy as np
import trimesh

home_path = os.path.expanduser("~")
code_path = os.path.join(home_path, "RSS_2026")
shared_dir = os.path.join(home_path, "shared_data")
project_dir = os.path.join(shared_dir, "AutoDex")
bodex_path = os.path.join(code_path, "BODex_outputs")
repo_dir = os.path.join(home_path, "AutoDex")
candidate_path = os.path.join(project_dir, "candidates", "allegro")  # default, use get_candidate_path() for other hands

robot_configs_path = os.path.join(project_dir, "content", "configs", "robot")
obj_path = os.path.join(project_dir, "object", "paradex")
urdf_path = os.path.join(project_dir, "content", "assets", "robot", "allegro_description")


def get_candidate_path(hand: str = "allegro") -> str:
    return os.path.join(project_dir, "candidates", hand)


def get_object_mesh(obj_name):
    mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_candidate(obj_name, obj_pose, version, shuffle=True, skip_done=True, success_only=False, hand="allegro"):
    """Load all grasp candidates under ``{candidates}/{hand}/{version}/{obj}``.

    Supports both layouts (auto-detected by walking until ``wrist_se3.npy`` is found):
        nested: ``{obj}/{scene_type}/{scene_id}/{grasp_idx}/wrist_se3.npy``
        flat:   ``{obj}/{scene_id}/{grasp_idx}/wrist_se3.npy``

    In the flat case the returned scene_info has ``scene_type=""``.
    """
    wrist_se3_list = []
    pregrasp_pose_list = []
    grasp_pose_list = []
    scene_info = []

    candidate_obj_path = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(candidate_obj_path):
        return np.empty((0, 4, 4)), np.empty((0, 0)), np.empty((0, 0)), []

    # Walk to find every grasp dir (one containing wrist_se3.npy).
    grasp_dirs = []
    for dirpath, dirnames, filenames in os.walk(candidate_obj_path):
        if "wrist_se3.npy" in filenames:
            grasp_dirs.append(dirpath)
            dirnames[:] = []  # don't descend further

    if shuffle:
        random.shuffle(grasp_dirs)
    else:
        grasp_dirs.sort()

    for base in grasp_dirs:
        rel = os.path.relpath(base, candidate_obj_path)
        parts = rel.split(os.sep)
        if len(parts) == 3:
            scene_type, scene_id, grasp_idx = parts
        elif len(parts) == 2:
            scene_type = ""
            scene_id, grasp_idx = parts
        else:
            # Unexpected depth — skip.
            continue

        result_path = os.path.join(base, "result.json")
        has_result = os.path.exists(result_path)
        if success_only:
            if not has_result:
                continue
            import json
            with open(result_path) as f:
                if not json.load(f).get("success", False):
                    continue
        elif skip_done and has_result:
            continue

        pregrasp = np.load(os.path.join(base, "pregrasp_pose.npy"))
        pregrasp_pose_list.append(pregrasp)
        grasp_file = os.path.join(base, "grasp_pose.npy")
        grasp_pose_list.append(np.load(grasp_file) if os.path.exists(grasp_file) else pregrasp)
        wrist_se3_obj = np.load(os.path.join(base, "wrist_se3.npy"))
        wrist_se3_list.append(obj_pose @ wrist_se3_obj)
        scene_info.append((scene_type, scene_id, grasp_idx))

    wrist_se3 = np.array(wrist_se3_list)
    grasp_pose = np.array(grasp_pose_list)
    pregrasp_pose = np.array(pregrasp_pose_list)

    return wrist_se3, pregrasp_pose, grasp_pose, scene_info
