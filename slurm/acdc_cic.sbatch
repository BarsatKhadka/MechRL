#!/bin/bash
# ACDC greedy baseline on the Cyber Innovation Cluster (single GPU, no DDP).
# Sweeps thresholds to trace ACDC's size-vs-faith curve on IOI, so we can read
# off the edge count at faith ~0.85 and compare to our RL agent.
#
# Submit from the LOGIN node:
#   sbatch slurm/acdc_cic.sbatch
# Monitor:
#   squeue -u $USER ; tail -f acdc_cic.<jobid>.out
#
#SBATCH --job-name=acdc_ioi
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --exclude=c1,c2            # leave desk machines free; prefer c3/c4
#SBATCH --output=acdc_cic.%j.out
#SBATCH --error=acdc_cic.%j.err

set -euo pipefail

echo "[$(hostname)] starting at $(date)"
nvidia-smi || true

# --- activate the conda env you created in setup (step 3 of the guide) ---
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env

cd ~/MechRL

# GPT-2 downloads automatically on first run (small). Let HF reach the hub.
python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Sweep thresholds low->high. Higher tau removes more edges -> lower faith.
# The faith~0.85 point is the number we want; the curve brackets it.
python -m scripts.run_acdc_greedy \
    --task IOITask \
    --device cuda \
    --num-examples 20 \
    --thresholds 0.001 0.003 0.01 0.03 0.06 0.1 0.2 \
    --out runs/acdc_ioi.json

echo "[$(hostname)] done at $(date)"
