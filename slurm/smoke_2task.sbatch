#!/bin/bash
# 2-task SMOKE on Magnolia (L40S) -- verify the multi-task plumbing works:
#   * shared frozen GPT-2 across both tasks (ONE model, not two -> no OOM)
#   * both bundles build (IOI + GreaterThan prefilters)
#   * episodes run across both tasks for a few iters
# This is a 3-iter sanity check, NOT a real run. Green light = it prints
# "[shared] 1 GPT-2 across 2 task(s)", both bundles build, no OOM, 3 iter lines.
#
# Pull first on Magnolia:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/smoke_2task.sbatch
# Watch :  squeue --me ; tail -f smoke2.<jobid>.out
#
#SBATCH --job-name=smoke2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=smoke2.%j.out
#SBATCH --error=smoke2.%j.err

echo "[$(hostname)] start $(date)"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv || true

cd ~/MechRL
source venv/bin/activate           # NOTE: not using `set -u` -- activate trips it
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# 2-task, 3 iters. Watch for "[shared] 1 GPT-2 across 2 task(s)" after the two
# "built ..." lines, then 3 iter lines with no CUDA OOM.
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --step-budget 150 --num-steps 512 \
    --total-iterations 3 --faith-threshold 0.95 --threshold-penalty 6 \
    --device cuda --out runs

echo "--- GPU memory after run (one model + 2 task caches, NOT two models) ---"
nvidia-smi --query-gpu=memory.used --format=csv || true
echo "[$(hostname)] done $(date)"
