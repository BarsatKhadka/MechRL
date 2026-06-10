#!/bin/bash
# 2-TASK PCGrad CONFIRMATION run on the Cyber Innovation Cluster (CIC, single GPU/node).
# Same experiment as slurm/train_2task_pcgrad.sbatch (Magnolia), CIC-flavoured:
# partition=main, NO --gres (one GPU/node, auto-allocated), conda env ~/mechrl-env.
#
# WHAT THIS VALIDATES: the sum->mean fix in _pcgrad_combine. PCGrad now AVERAGES the
# de-conflicted per-task gradients (was sum). Summing grew the combined gradient ~T x
# with #tasks -> at 12 tasks it would slam the grad-norm clip (max_grad_norm=0.5) every
# step and flatten the surgery. Averaging keeps LR + clip calibrated 2 -> 12.
#
# WATCH the new "gn" field (pre-clip gradient norm):
#   - gn mostly <= 0.50  -> clip NOT binding most steps   -> GOOD (fix works)
#   - gn consistently >  0.50 -> clip binding every step  -> tell me (we raise the clip)
# Science check: IOI faith holds ~0.85-0.90 (no 0.77 collapse), GreaterThan ~0.95+,
# expl_var 0.6-0.9, gcos still dips negative. gn is readable by iter ~10; the IOI
# verdict needs ~200 iters. Kill early once gn looks right if you only want the fix
# confirmed.
#
# Submit from the LOGIN node:  cd ~/MechRL && git pull && sbatch slurm/train_2task_pcgrad_cic.sbatch
# Monitor:  squeue -u $USER ; tail -f mechrl_2task_pcgrad.<jobid>.out   (watch gn)
#
#SBATCH --job-name=mechrl_2t_pcg
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=c1,c2            # leave desk machines free; prefer c3/c4
#SBATCH --output=mechrl_2task_pcgrad.%j.out
#SBATCH --error=mechrl_2task_pcgrad.%j.err
# NOTE: this cluster does NOT use --gres for GPUs (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] start $(date)"
nvidia-smi || true

module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Same config as the Aquaman/Magnolia PCGrad run (margin 0 = tau is each task's ceiling,
# lambda=5), now with the AVERAGED PCGrad combine. --pcgrad implies mixed sampling
# (mutually exclusive with --round-robin: PCGrad needs both tasks' gradients in one step).
# If gpt2 tries to download and hangs (no internet on compute node), prefix python with:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
