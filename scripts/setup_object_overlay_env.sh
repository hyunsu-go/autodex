#!/usr/bin/env bash
# Prepare the conda env used by overlay_object_video_single.py.
#
# This is separate from the GoTrack env. The batch wrapper usually resolves
# this interpreter to ~/anaconda3/envs/paradex/bin/python on capture PCs.
set -euo pipefail

ENV_NAME="${ENV_NAME:-paradex}"
CONDA_DIR="${CONDA_DIR:-$HOME/anaconda3}"
PY="${PY:-$CONDA_DIR/envs/$ENV_NAME/bin/python}"

if [[ ! -x "$PY" ]]; then
    echo "[overlay-env] missing python interpreter: $PY" >&2
    exit 1
fi

echo "[overlay-env] host=$(hostname)"
echo "[overlay-env] python=$PY"

"$PY" -m pip install transforms3d trimesh

if ! "$PY" - <<'PY'
import nvdiffrast.torch as dr
print("nvdiffrast_ok")
PY
then
    "$PY" -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
fi

"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "paradex"))

import cv2
import nvdiffrast.torch as dr
import torch
import transforms3d
import trimesh
from paradex.image.projection import intr_opencv_to_opengl_proj

print("overlay_imports_ok")
PY

echo "[overlay-env] done"
