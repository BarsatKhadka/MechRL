#!/bin/bash
# TRANSFER SMOKE-TEST (CIC): does the 2-task policy transfer to a HELD-OUT task?
#
# Runs the SAME held-out task two ways, back to back, same seed/config:
#   1. SCRATCH    -- random init (the baseline: how long without prior skill)
#   2. WARM-START -- --init-from the 2-task IOI+GreaterThan policy (loads its
#                    actor+critic weights, then trains on the held-out task)
#
# THE THESIS: if WARM-START clears the held-out task's bar in FEWER iters than
# SCRATCH (first_success_iter), the policy transferred a general cutting skill.
# This is the make-or-break experiment, and 2 tasks is enough to scout it -- no
# need to wait for 12-task. It even works off a mid-run 2-task CHECKPOINT (the
# auto-find below picks the latest policy_iter*.pt), so you can run it today.
#
# Held-out = a task NOT in training (training was IOI + GreaterThan). Default
# GenderedPronounTask (ceiling 0.996, cleanest bar). Swap TASK if it doesn't
# resolve (try CopySuppressionTask or SubjectVerbAgreementTask).
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/transfer_smoke_cic.sbatch
# Read  :  tail -f mechrl_transfer.<jobid>.out  -- compare the two "first_success_iter"
#
#SBATCH --job-name=mechrl_xfer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --exclude=c1,c2
#SBATCH --output=mechrl_transfer.%j.out
#SBATCH --error=mechrl_transfer.%j.err
# NOTE: main does NOT use --gres (one GPU/node, auto-allocated).

set -euo pipefail
TASK=GenderedPronounTask          # held-out task (NOT in training)
SEED=0
ITERS=150

echo "[$(hostname)] transfer smoke-test start $(date)  task=${TASK}"
nvidia-smi || true
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

# auto-find the 2-task source policy (latest run; --init-from picks its latest ckpt)
SRC=$(ls -dt runs/IOITask,GreaterThanOriginal_seed0_*/ 2>/dev/null | head -1 || true)
if [ -z "${SRC}" ]; then echo "ERROR: no 2-task run dir to warm-start from"; exit 1; fi
SRC=${SRC%/}
echo "warm-start source policy: ${SRC}"
ls "${SRC}"/policy_iter*.pt 2>/dev/null | tail -1 || echo "(will use policy_final.pt)"

COMMON="--policy batch --tasks ${TASK} --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --seed ${SEED} --step-budget 150 --num-steps 512 --total-iterations ${ITERS} \
    --device cuda"

echo "==================== SCRATCH (random init) ===================="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent ${COMMON} \
    --out runs/transfer_scratch

echo "==================== WARM-START (from 2-task) ===================="
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent ${COMMON} \
    --init-from "${SRC}" --out runs/transfer_warm

echo "[$(hostname)] done $(date)"
echo "COMPARE: in each section above, find the first iter the '└ ${TASK}✓' mark appears"
echo "  (or grep the metrics). Warm-start clearing the bar EARLIER = transfer works."
