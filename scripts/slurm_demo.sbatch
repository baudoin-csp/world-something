#!/bin/bash
# The room + physics demo on one GPU node: real Phases 1-2 (DA3 + Boxer), then Phases 3-7.
#   sbatch scripts/slurm_demo.sbatch                      # rendered with the CPU stand-in (ibr)
#   RENDER=sglang SERVER=http://<node>:30011 sbatch scripts/slurm_demo.sbatch
# Run ./setup.sh once on a login node first (code, weights, CA-1M capture). Adjust the GPU line to
# your cluster; Phases 1-2 fit on one 24 GB GPU.
#SBATCH --job-name=mundo-demo
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
export MUJOCO_GL=egl
nvidia-smi --query-gpu=name,memory.total --format=csv
./run_demo.sh
