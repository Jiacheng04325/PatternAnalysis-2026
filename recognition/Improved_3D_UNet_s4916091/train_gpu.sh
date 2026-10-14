#!/bin/bash
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=hipmri_train_%j.out
#SBATCH --error=hipmri_train_%j.err

set -euo pipefail

MODEL_NAME="${1:-}"
if [[ "$MODEL_NAME" != "baseline" && "$MODEL_NAME" != "proposed" ]]; then
    echo "Usage: sbatch train_gpu.sh {baseline|proposed}" >&2
    exit 1
fi

EPOCHS="${EPOCHS:-30}"
PATCH_DEPTH="${PATCH_DEPTH:-64}"
PATCH_HEIGHT="${PATCH_HEIGHT:-128}"
PATCH_WIDTH="${PATCH_WIDTH:-128}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate" torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/formal_${MODEL_NAME}_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "Model: ${MODEL_NAME}"
echo "Epochs: ${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --patch-size "$PATCH_DEPTH" "$PATCH_HEIGHT" "$PATCH_WIDTH" \
    --base-channels "$BASE_CHANNELS" \
    --batch-size "$BATCH_SIZE" \
    --workers 0 \
    --seed 3710 \
    --selection-class 5 \
    --device cuda

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
