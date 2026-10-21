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
MAX_TRAIN_VOLUMES="${MAX_TRAIN_VOLUMES:-}"
MAX_VALIDATION_VOLUMES="${MAX_VALIDATION_VOLUMES:-}"
RUN_PREFIX="${RUN_PREFIX:-formal_baseline_2d}"

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate" torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/${RUN_PREFIX}_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "Model: baseline_2d"
echo "Epochs: ${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TRAIN_ARGUMENTS=(
    --data-root /home/groups/comp3710/HipMRI_Study_open
    --output-dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --base-channels "$BASE_CHANNELS"
    --slices-per-volume "$SLICES_PER_VOLUME"
    --slice-batch-size "$SLICE_BATCH_SIZE"
    --workers 0
    --seed 3710
    --selection-class 5
    --device cuda
)

if [[ -n "$MAX_TRAIN_VOLUMES" ]]; then
    TRAIN_ARGUMENTS+=(--max-train-volumes "$MAX_TRAIN_VOLUMES")
fi
if [[ -n "$MAX_VALIDATION_VOLUMES" ]]; then
    TRAIN_ARGUMENTS+=(--max-validation-volumes "$MAX_VALIDATION_VOLUMES")
fi

python train_2d.py "${TRAIN_ARGUMENTS[@]}"

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
