#!/bin/bash
#SBATCH --job-name=hipmri-evaluate
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=hipmri_evaluate_%j.out
#SBATCH --error=hipmri_evaluate_%j.err

set -euo pipefail

if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "CHECKPOINT must contain a trained checkpoint path" >&2
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

source "$HOME/miniconda3/bin/activate"
conda activate torch

PROJECT_DIR="$HOME/PatternAnalysis-2026/recognition/Improved_3D_UNet_s4916091"
OUTPUT_DIR="$HOME/comp3710_runs/evaluation_${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"
echo "Checkpoint: $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python predict.py \
    --data-root /home/groups/comp3710/HipMRI_Study_open \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --split test \
    --patch-size 64 128 128 \
    --overlap 0.5 \
    --window-batch-size 2 \
    --device cuda \
    --max-cases 1

echo "Output directory: $OUTPUT_DIR"
echo "Finished $(date)"
