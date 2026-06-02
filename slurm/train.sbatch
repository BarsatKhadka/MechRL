#!/bin/bash
#SBATCH --job-name=mechrl_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Magnolia (Ole Miss MCSR) — L40S nodes, 46GB VRAM, bf16 native.
#
# Multi-task (all 13 verified tasks):
#   sbatch slurm/train.sbatch
# Single-task signal run (train on IOI only — the first "does it learn" test):
#   sbatch --export=ALL,TASKS=IOITask slurm/train.sbatch
# A whole family:
#   sbatch --export=ALL,TASKS=greaterthan slurm/train.sbatch
# Override scale:
#   sbatch --export=ALL,TASKS=all,ITERS=800,STEPS=256,BUDGET=400,SEED=1 slurm/train.sbatch

set -euo pipefail

module purge
module load python/2025.12-2
module load cuda12.8/toolkit/12.8.1

source "${HOME}/MechRL/venv/bin/activate"

cd "${HOME}/MechRL"
mkdir -p logs runs

# Compute nodes are usually offline — GPT-2 must already be in the HF cache.
# Pre-fetch once on the login node, then keep these on:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TASKS="${TASKS:-all}"
SEED="${SEED:-0}"
ITERS="${ITERS:-500}"
STEPS="${STEPS:-256}"
BUDGET="${BUDGET:-400}"

echo "[start] $(date) tasks=${TASKS} seed=${SEED} iters=${ITERS} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m mechrl.train.train_agent \
    --tasks "${TASKS}" \
    --seed "${SEED}" \
    --device cuda \
    --total-iterations "${ITERS}" \
    --num-steps "${STEPS}" \
    --step-budget "${BUDGET}" \
    --out runs

echo "[done]  $(date)"
