#!/bin/bash
# COLLATE RESULTS -- CPU-only job (NO GPU, NO model load; pure JSON parsing).
# Reads the eval JSONs (zeroshot_*/amort_*/validity_*/acdc_*) and prints the Results tables.
# Run as a job, NOT on the login node (standing rule: nothing runs on a login node here).
#
# Submit (after the result jobs have written their JSONs -- or anytime for partial tables):
#   cd ~/MechRL && git pull && sbatch slurm/collate_results_cpu.sbatch
# Read:  cat mechrl_collate.<jobid>.out
#
# NOTE: --gres is intentionally omitted (no GPU). If Magnolia has a dedicated CPU partition,
# swap the --partition line to it; gpuq accepts CPU-only jobs as a fallback.
#
#SBATCH --job-name=mechrl_collate
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=mechrl_collate.%j.out
#SBATCH --error=mechrl_collate.%j.err

echo "[$(hostname)] collate results start $(date)"
cd ~/MechRL
source venv/bin/activate
RUNS=${1:-runs}    # pass runs_lam4 / runs_lam6 to collate a lambda run
python -m scripts.collate_results --runs "${RUNS}"
echo "[$(hostname)] collate results done $(date)"
