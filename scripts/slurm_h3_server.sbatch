#!/bin/bash
# Serve MiniMax-H3 Ref2VA with SGLang for `phase7_render --backend sglang` (zero-shot rendering).
# Adjust the GPU lines for your cluster; the H3 README uses 4 GPUs with Ulysses parallelism.
#SBATCH --job-name=h3-ref2va
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs
export HF_HOME=${HF_HOME:-$SCRATCH/hf}
echo "H3 server on $(hostname):30011"
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --host 0.0.0.0 --port 30011 --model-variant ref2va
