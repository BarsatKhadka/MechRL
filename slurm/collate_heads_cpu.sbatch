#!/bin/bash
# HEAD-RECOVERY collation -- CPU-only (pure JSON, no torch). Reads circuit_*.json dumps and
# prints canonical / recovered / MISSING heads per task (intersection recomputed), writes CSV.
# Run as a job, not on the login node.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/collate_heads_cpu.sbatch              # runs/
#          sbatch slurm/collate_heads_cpu.sbatch runs_lam4                                # a lambda run
# Read  :  cat mechrl_heads.<jobid>.out   (and open the CSV it writes)
#
#SBATCH --job-name=mechrl_heads
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=mechrl_heads.%j.out
#SBATCH --error=mechrl_heads.%j.err

echo "[$(hostname)] head collation start $(date)"
cd ~/MechRL
source venv/bin/activate
RUNS=${1:-runs}
python -m scripts.collate_heads --runs "${RUNS}" --out "head_recovery_${RUNS//\//_}.csv"
echo "[$(hostname)] head collation done $(date)"
