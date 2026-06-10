#!/bin/bash
# 2-TASK PCGrad CONFIRMATION run on Magnolia (L40S, gpuq) -- validates the
# sum->mean fix in _pcgrad_combine before the 12-task scale-up.
#
# WHAT CHANGED: PCGrad now AVERAGES the de-conflicted per-task gradients (was sum).
# Summing made the combined gradient grow ~T x with #tasks -> at 12 tasks it would
# slam the grad-norm clip (max_grad_norm=0.5) EVERY step and flatten the surgery
# into a fixed-length step. Averaging keeps the update on the same scale as the
# non-PCGrad mean path, so LR + clip stay calibrated from 2 tasks up to 12.
#
# WHAT TO WATCH (new field): "gn" = pre-clip gradient norm.
#   - gn mostly <= 0.50  -> clip is NOT binding most steps  -> GOOD (fix works)
#   - gn consistently >  0.50 -> clip binding every step    -> averaging not enough,
#                                                              tell me (we'd raise the clip)
# Also confirm the science still holds: IOI faith holds ~0.85-0.90 (no 0.77 collapse),
# GreaterThan ~0.95+, expl_var healthy (0.6-0.9), gcos still dips negative (conflict
# present + being projected out). Behaviour should match the Aquaman run, just with a
# correctly-scaled step. Diagnostic (gn) is readable from iter ~1-10; the IOI verdict
# needs ~200 iters. Kill early once gn looks right if you only want the fix confirmed.
#
# If it passes -> scale up WITHOUT wasting this run:
#   --init-from runs/<this_run_dir> --tasks train --pcgrad --total-iterations 500
#
# Pull first on Magnolia:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_2task_pcgrad.sbatch
# Watch :  tail -f mechrl_2task_pcgrad.<jobid>.out   (watch the gn field)
#
#SBATCH --job-name=mechrl_2t_pcg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=mechrl_2task_pcgrad.%j.out
#SBATCH --error=mechrl_2task_pcgrad.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Same config as the Aquaman PCGrad run (margin 0 = tau is each task's ceiling,
# lambda=5), now with the AVERAGED PCGrad combine. --pcgrad implies mixed sampling
# (it is mutually exclusive with --round-robin: PCGrad needs BOTH tasks' gradients
# in the same step to de-conflict them; round-robin gives one task per iter).
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
