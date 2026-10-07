#!/bin/bash
#SBATCH --job-name=hipmri-benchmark
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=hipmri_benchmark_%j.out
#SBATCH --error=hipmri_benchmark_%j.err

set -euo pipefail

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate"
conda activate torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
RUN_ROOT="$HOME/comp3710_runs/gpu_benchmark_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MODEL_NAME in baseline proposed; do
    echo "Benchmarking ${MODEL_NAME}"
    python train.py \
        --data-root /home/groups/comp3710/HipMRI_Study_open \
        --output-dir "$RUN_ROOT/$MODEL_NAME" \
        --model "$MODEL_NAME" \
        --epochs 1 \
        --patch-size 64 128 128 \
        --base-channels 16 \
        --batch-size 1 \
        --workers 0 \
        --device cuda \
        --max-train-batches 10 \
        --max-validation-batches 3
done

echo "Output directory: $RUN_ROOT"
echo "Finished $(date)"
