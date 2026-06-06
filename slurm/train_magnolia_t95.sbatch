#!/bin/bash
# Higher-fidelity batch+KILL retrain on Magnolia (L40S, gpuq).
#
# Why: the tau=0.9 run gave a small but TASK-WEAK circuit (610 edges, faith 0.887
# but argmax only 0.65 / logit-diff 0.27 -- the soft threshold let it over-cut
# into the low-fidelity zone). Raise the bar (tau=0.95) and the penalty (lambda=6)
# so it holds faith high. Goal: a task-faithful circuit, then compare its size to
# flat's 1595 AT MATCHED faith (the real "did batch+KILL beat flat" test).
#
# Submit:  sbatch slurm/train_magnolia_t95.sbatch
# Watch :  squeue --me ; tail -f mechrl_t95.<jobid>.out
#
#SBATCH --job-name=mechrl_t95
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=mechrl_t95.%j.out
#SBATCH --error=mechrl_t95.%j.err

echo "[$(hostname)] start $(date)"
nvidia-smi || true

cd ~/MechRL
source venv/bin/activate           # NOTE: not using `set -u` -- activate trips it
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# tau=0.95, lambda=6 -> hold faith high (don't over-cut). IOI prefilter recomputes
# on GPU (~20s) since the cache isn't in git. Imports take ~1-2 min (cold NFS) before
# the first iter line -- that's normal, not a hang.
python -m mechrl.train.train_agent \
    --policy batch --tasks IOITask --num-examples 20 \
    --batch-sizes 1 3 10 30 \
    --faith-threshold 0.95 --threshold-penalty 6 \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
