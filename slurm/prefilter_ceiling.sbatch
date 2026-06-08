#!/bin/bash
# THE baseline check: for every task/variant, can the top-K candidate set reproduce
# the full model (KL-faith)? This is the ceiling -- the agent can never beat it, so
# if a task's top-K caps low, that task is broken before training. Sweeps K x
# attribution (logit-diff vs KL) so we can read off the fix per task: better
# selection (kl), more edges (K), or drop it.
#
# Covers all 13 training tasks (4 IOI + 4 GreaterThan + 5 docstring) + the 3
# held-out candidates (copy-suppression / successor / induction). One shared GPT-2.
#
# Pull first on Magnolia:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/prefilter_ceiling.sbatch
# Watch :  squeue --me ; tail -f prefilter_ceiling.<jobid>.out
#
#SBATCH --job-name=pref_ceil
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=prefilter_ceiling.%j.out
#SBATCH --error=prefilter_ceiling.%j.err

echo "[$(hostname)] start $(date)"
nvidia-smi --query-gpu=memory.total,memory.used --format=csv || true

cd ~/MechRL
source venv/bin/activate           # NOTE: not using `set -u` -- activate trips it
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# logit-diff vs KL attribution, K in {1500,3000,5000,8000}, all tasks + held-out.
# First run computes the KL attribution scores (cached as *_kl.pt for reuse);
# the logit-diff scores reuse the existing *_taskmetric.pt cache where present.
python -m scripts.verify_prefilter_kl \
    --tasks all,CopySuppressionTask,SuccessorHeadsTask,InductionTask \
    --ks 1500,3000,5000,8000 \
    --num-examples 20 --ig-steps 5 --device cuda

echo "[$(hostname)] done $(date)"
