#!/bin/bash
#SBATCH --job-name=hipmri-evaluate-2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=hipmri_evaluate_2d_%j.out
#SBATCH --error=hipmri_evaluate_2d_%j.err

set -euo pipefail

if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "CHECKPOINT must contain a trained 2D checkpoint path" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

export TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp"

source "$HOME/miniconda3/bin/activate" torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/evaluation_2d_${SLURM_JOB_ID}"
SPLIT="${SPLIT:-test}"
MAX_CASES="${MAX_CASES:-}"
SLICE_BATCH_SIZE="${SLICE_BATCH_SIZE:-16}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "Checkpoint: $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PREDICT_ARGUMENTS=(
    --data-root /home/groups/comp3710/HipMRI_Study_open
    --checkpoint "$CHECKPOINT"
    --output-dir "$OUTPUT_DIR"
    --split "$SPLIT"
    --slice-batch-size "$SLICE_BATCH_SIZE"
    --device cuda
)

if [[ -n "$MAX_CASES" ]]; then
    PREDICT_ARGUMENTS+=(--max-cases "$MAX_CASES")
fi

python predict_2d.py "${PREDICT_ARGUMENTS[@]}"

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
