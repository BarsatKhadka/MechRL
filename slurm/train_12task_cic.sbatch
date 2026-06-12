#!/bin/bash
# THE 12-TASK run on CIC: IOI x4 + GreaterThan x4 + Docstring x4 (--tasks train).
# This is the core amortization result -- one shared policy across 12 tasks.
#
# Everything proven at 2-task is ON:
#   --pcgrad      : de-conflicts per-task gradients (AVERAGED combine -> scales to 12
#                   without clip-flattening). REQUIRED for multi-task; was missing before.
#   headroom obs  : automatic (globals 5->6) -> lets the policy specialize per-task cut depth.
#   --faith-margin 0 : per-task tau = each task's measured CEILING (printed as "[bar] ...").
#   lambda=5.
#
# Watch the per-task "└" line: every task climbs toward its tau and holds (✓). Expect the
# same shape as 2-task (specialize cut depth, hit bars) but it needs MORE total iters since
# each task gets ~1/12 of the episodes. Checkpoints every 50 -> --resume-from to extend.
#
# WHY 12-task matters: a 2-task donor already transferred 5x better than scratch on a
# held-out task. A 12-task donor (more diverse skill) should transfer even harder -> the
# headline figure. After this finishes, re-run transfer_smoke_cic.sbatch off THIS policy.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/train_12task_cic.sbatch
# Watch :  tail -f mechrl_12task.<jobid>.out
#
#SBATCH --job-name=mechrl_12t
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=48:00:00          # main is MaxTime=UNLIMITED; 12-task + PCGrad is slower/iter.
#SBATCH --exclude=c1,c2          # If it needs more than 48h, resume from the latest ckpt.
#SBATCH --output=mechrl_12task.%j.out
#SBATCH --error=mechrl_12task.%j.err
# NOTE: main does NOT use --gres (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] 12-task start $(date)"
nvidia-smi || true
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# 12 tasks share ONE frozen GPT-2. --pcgrad implies mixed sampling (it needs >1 task per
# update to de-conflict). If gpt2 tries to download and hangs, the offline vars force cache.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent --policy batch \
    --tasks train --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad \
    --step-budget 150 --num-steps 512 --total-iterations 800 \
    --device cuda --out runs

echo "[$(hostname)] 12-task done $(date)"
