#!/bin/bash
# Phase 8 on a Slurm cluster (e.g. ETH Euler): one node with 8 large GPUs.
# Check the GPU names and limits for your cluster (`sinfo`, the cluster docs) and adjust the
# #SBATCH lines; H3's transformer needs about 66 GB in bf16, so 8 x 80 GB GPUs are the target.
#SBATCH --job-name=mundo-h3-lora
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
PAIRS=${PAIRS:-outputs/pairs}
OUT=${OUT:-outputs/phase8}
export HF_HOME=${HF_HOME:-$SCRATCH/hf}            # H3 weights (~150 GB with the text encoder)
python -m world_something.phase8_train precompute --pairs "$PAIRS"     # VAE + Qwen3-VL, one GPU is enough
accelerate launch --config_file configs/accelerate_fsdp.yaml -m world_something.phase8_train train \
  --pairs "$PAIRS" --out "$OUT"
