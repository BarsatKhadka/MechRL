#!/bin/bash
# Batch-cut agent training on the Cyber Innovation Cluster (single GPU).
# Fresh run (batch policy can't resume the flat checkpoint). ~1.5-2h on a 5070 Ti.
#
# Submit from the LOGIN node:
#   sbatch slurm/train_cic.sbatch
# Monitor:
#   squeue -u $USER ; tail -f mechrl_batch.<jobid>.out
#
#SBATCH --job-name=mechrl_batch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=c1,c2            # leave desk machines free; prefer c3/c4
#SBATCH --output=mechrl_batch.%j.out
#SBATCH --error=mechrl_batch.%j.err
# NOTE: this cluster does NOT use --gres for GPUs (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] start $(date)"
nvidia-smi || true

module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# IOI prefilter is computed fresh on the GPU (~20s) since the cache isn't in git.
# If gpt2 tries to download and hangs (compute node has no internet), prefix the
# python call with: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m mechrl.train.train_agent \
    --policy batch --tasks IOITask --num-examples 20 \
    --batch-sizes 1 3 10 30 100 \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --faith-threshold 0.9 --device cuda --out runs

echo "[$(hostname)] done $(date)"
