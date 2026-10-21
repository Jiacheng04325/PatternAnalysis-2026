#!/bin/bash
#SBATCH --job-name=hipmri-train-2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hipmri_train_2d_%j.out
#SBATCH --error=hipmri_train_2d_%j.err

set -euo pipefail

EPOCHS="${EPOCHS:-30}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
SLICES_PER_VOLUME="${SLICES_PER_VOLUME:-16}"
SLICE_BATCH_SIZE="${SLICE_BATCH_SIZE:-16}"

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate" torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/formal_baseline_2d_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "Model: baseline_2d"
echo "Epochs: ${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_2d.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --base-channels "$BASE_CHANNELS" \
    --slices-per-volume "$SLICES_PER_VOLUME" \
    --slice-batch-size "$SLICE_BATCH_SIZE" \
    --workers 0 \
    --seed 3710 \
    --selection-class 5 \
    --device cuda

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
