#!/bin/bash
# VALIDITY BATTERY for the 2-task agent (IOI + GreaterThan), on a CIC compute node.
# Agent plays first (best-of-K rollout -> circuit), then scores each circuit on its
# paper's metric (IOI: 26 heads + logit-diff; GreaterThan: ACDC prob-diff). Pins the
# iter-600 policy by default. Auto-finds the 2-task run dir.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/battery_2task_cic.sbatch
# Read  :  tail -f mechrl_battery.<jobid>.out   (look for the two "=== ... BATTERY ===")
#
#SBATCH --job-name=mechrl_batt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --exclude=c1,c2
#SBATCH --output=mechrl_battery.%j.out
#SBATCH --error=mechrl_battery.%j.err
# NOTE: main does NOT use --gres (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] battery start $(date)"
nvidia-smi || true
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

# auto-find the 2-task run dir (newest IOI+GreaterThan run)
RUN_DIR=$(ls -dt runs/IOITask,GreaterThanOriginal_seed0_*/ 2>/dev/null | head -1 || true)
if [ -z "${RUN_DIR}" ]; then echo "ERROR: no 2-task run dir found"; exit 1; fi
RUN_DIR=${RUN_DIR%/}
echo "battery on: ${RUN_DIR}"
ls "${RUN_DIR}"/policy_iter*.pt 2>/dev/null | tail -1 || echo "(no policy_iter*.pt found)"

# If gpt2 tries to download and hangs (no internet on compute node), the offline
# vars below force the cached copy.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.battery_test \
    --run "${RUN_DIR}" --device cuda --num-rollouts 8 \
    --out "${RUN_DIR}/battery_2task.json"

echo "[$(hostname)] battery done $(date)"
