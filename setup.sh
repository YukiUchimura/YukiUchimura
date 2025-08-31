#!/usr/bin/env bash
set -euo pipefail

# HFトークンがあれば非対話でログイン
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  python - <<'PY'
import os, subprocess
tok=os.environ.get("HUGGING_FACE_HUB_TOKEN","")
if tok:
    subprocess.run(["python","-m","huggingface_hub.cli","login","--token",tok,"--add-to-git-credential","--quiet"], check=False)
PY
fi

# 主要モデルの事前取得（キャッシュへ）
python - <<'PY'
from huggingface_hub import snapshot_download
import os

cache=os.environ.get("HF_HOME","/workspace/.cache/huggingface")
os.makedirs(cache, exist_ok=True)

# SDXL Base
snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", local_dir="/workspace/models/sdxl-base", local_dir_use_symlinks=False)
# ControlNet OpenPose
snapshot_download(repo_id="lllyasviel/sd-controlnet-openpose", local_dir="/workspace/models/controlnet-openpose", local_dir_use_symlinks=False)
# IP-Adapter（SDXL向け）
snapshot_download(repo_id="h94/IP-Adapter", local_dir="/workspace/models/ip-adapter", local_dir_use_symlinks=False)
# InstantID
snapshot_download(repo_id="InstantX/InstantID", local_dir="/workspace/models/instantid", local_dir_use_symlinks=False)
# insightface の顔モデル（antelopev2）
snapshot_download(repo_id="deepinsight/insightface", local_dir="/workspace/models/insightface", local_dir_use_symlinks=False, allow_patterns=["models/*","*.onnx","**/*.onnx"])
PY
