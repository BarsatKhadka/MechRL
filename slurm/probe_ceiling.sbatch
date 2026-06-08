#!/bin/bash
# Diagnose WHY induction/successor/diffuse-docstring cap low, and test whether a
# longer (more in-distribution) induction sequence fixes it. Reports KL_cut (is the
# counterfactual strong?) + faith vs K up to ALL edges (diffuse?), plus an induction
# half_len sweep. --force ignores cached prefilter scores (also re-confirms the
# earlier ceiling table wasn't stale: Part-1 faith@3000 should match its logitdiff col).
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/probe_ceiling.sbatch
#
#SBATCH --job-name=probe_ceil
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=probe_ceiling.%j.out
#SBATCH --error=probe_ceiling.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate

python -m scripts.probe_task_ceiling \
    --tasks InductionTask,SuccessorHeadsTask,DocstringGPT2Google5Task,DocstringGPT2Sphinx7Task,IOITask,CopySuppressionTask \
    --induction-sweep --force --device cuda

echo "[$(hostname)] done $(date)"
