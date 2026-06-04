#!/bin/bash
#SBATCH --job-name=mechrl_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclude=node049   # CONFIRMED bad: uncorrectable ECC twice. Skip it (admins haven't drained it yet).

# Magnolia (Ole Miss MCSR) — L40S nodes, 46GB VRAM, bf16 native.
#
# Env-var knobs (all optional):
#   TASKS=IOITask|ioi|greaterthan|docstring|all   which task(s)        (default all)
#   BUDGET=1500     max actions per episode (raises the pruning ceiling) (default 400)
#   STEPS=...       rollout length; defaults to BUDGET so each rollout completes >=1 episode
#   TAU=0.8         faithfulness threshold the circuit must clear        (default 0.8)
#   PENALTY=3.0     how hard the threshold is (lambda)                   (default 3.0)
#   LR=2.5e-4       learning rate                                        (default 2.5e-4)
#   ITERS=800       PPO iterations                                       (default 500)
#   SEED=0
#
# Examples:
#   sbatch --export=ALL,TASKS=IOITask slurm/train.sbatch                 # single-task IOI
#   sbatch --export=ALL,TASKS=IOITask,BUDGET=1500,ITERS=800 slurm/train.sbatch   # deeper pruning
#   sbatch slurm/train.sbatch                                            # all 13 tasks

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
export PYTHONUNBUFFERED=1   # stream python stdout live into the .out log

TASKS="${TASKS:-all}"
SEED="${SEED:-0}"
ITERS="${ITERS:-500}"
BUDGET="${BUDGET:-400}"
STEPS="${STEPS:-$BUDGET}"        # default rollout length to the budget => each rollout finishes >=1 episode (no nan metrics)
TAU="${TAU:-0.8}"
PENALTY="${PENALTY:-3.0}"
LR="${LR:-2.5e-4}"
RESUME="${RESUME:-}"             # set to a run dir to continue from its latest checkpoint

echo "[start] $(date) tasks=${TASKS} seed=${SEED} iters=${ITERS} budget=${BUDGET} steps=${STEPS} tau=${TAU} penalty=${PENALTY} lr=${LR} resume=${RESUME:-none} host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m mechrl.train.train_agent \
    --tasks "${TASKS}" \
    --seed "${SEED}" \
    --device cuda \
    --total-iterations "${ITERS}" \
    --num-steps "${STEPS}" \
    --step-budget "${BUDGET}" \
    ${RESUME:+--resume-from "${RESUME}"} \
    --faith-threshold "${TAU}" \
    --threshold-penalty "${PENALTY}" \
    --lr "${LR}" \
    --out runs

echo "[done]  $(date)"
