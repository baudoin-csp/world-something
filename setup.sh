#!/usr/bin/env bash
# One-time setup, on a machine with internet (e.g. a cluster login node):
#   1. third-party code at pinned commits (Depth Anything 3, Boxer) into third_party/
#   2. Python packages into the active environment (Python 3.10-3.12)
#   3. model weights: DA3 nested metric model (Hugging Face cache), Boxer/DINOv3/OWLv2 checkpoints
#   4. example 1: the CA-1M capture val-42898570 into data/ (asks you to accept its licence;
#      ACCEPT_CA1M_LICENSE=1 accepts it non-interactively)
# Set SKIP_PIP=1 to manage packages yourself (see requirements.txt and DA3's pyproject.toml).
set -euo pipefail
cd "$(dirname "$0")"

DA3_COMMIT=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4    # ByteDance-Seed/Depth-Anything-3, 2026-07-27
BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148  # facebookresearch/boxer, 2026-07-27
DA3_MODEL=${DA3_MODEL:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}

clone() {  # name url commit
  if [ ! -d "third_party/$1/.git" ]; then
    git clone --filter=blob:none "$2" "third_party/$1"
  fi
  git -C "third_party/$1" fetch -q origin
  git -C "third_party/$1" checkout -q "$3"
}
mkdir -p third_party
clone Depth-Anything-3 https://github.com/ByteDance-Seed/Depth-Anything-3 "$DA3_COMMIT"
clone boxer https://github.com/facebookresearch/boxer "$BOXER_COMMIT"

if [ "${SKIP_PIP:-0}" != 1 ]; then
  # DA3's documented install (pulls torch, xformers, open3d, evo, ...). If your cluster needs a
  # specific CUDA build of torch, install it first; pip keeps it when it satisfies DA3.
  python -m pip install "torch>=2" torchvision xformers
  python -m pip install -e third_party/Depth-Anything-3
  python -m pip install -r requirements.txt   # Boxer's runtime deps + ours
fi

(cd third_party/boxer && bash scripts/download_ckpts.sh)
python -c "from huggingface_hub import snapshot_download as d; print(d('$DA3_MODEL'))"

if [ "${ACCEPT_CA1M_LICENSE:-0}" = 1 ]; then
  python -m world_something.ca1m fetch --accept-license
else
  python -m world_something.ca1m fetch
fi
echo "Setup done. Run ./run_example.sh (needs a GPU)."
