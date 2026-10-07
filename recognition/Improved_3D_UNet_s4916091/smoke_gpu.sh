#!/bin/bash
#SBATCH --job-name=hipmri-smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=hipmri_smoke_%j.out
#SBATCH --error=hipmri_smoke_%j.err

set -euo pipefail

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate"
conda activate torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/gpu_smoke_proposed_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
nvidia-smi
python -c 'import torch; print("torch:",torch.__version__); print("cuda:",torch.cuda.is_available()); print("device:",torch.cuda.get_device_name(0))'

python train.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open \
    --output-dir "$OUTPUT_DIR" \
    --model proposed \
    --epochs 1 \
    --patch-size 64 128 128 \
    --base-channels 16 \
    --batch-size 1 \
    --workers 0 \
    --device cuda \
    --max-train-batches 1 \
    --max-validation-batches 1

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
