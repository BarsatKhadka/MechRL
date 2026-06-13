#!/bin/bash
# 12-TASK on Aquaman (direct GPU box, NO SLURM) -- SEED 2, the third replication seed.
# CIC = seed 0, Magnolia = seed 1, Aquaman = seed 2 -> three seeds for error bars.
# Aquaman is slow but has no walltime limit, so it runs 1200 iters to completion.
#
# Run INSIDE tmux so it survives disconnects:
#     tmux new -s mech12
#     bash scripts/run_12task_aquaman.sh
#   detach: Ctrl-B then D   |   reattach: tmux attach -t mech12
#
# If a specific GPU is free (check nvidia-smi), prefix:  CUDA_VISIBLE_DEVICES=0 bash ...
#
# Same config as the CIC/Magnolia seeds (only --seed differs): 4 IOI + 4 GreaterThan +
# 4 Docstring variants, --pcgrad, headroom (auto), faith-margin 0 (tau = ceiling), lambda=5.

set -euo pipefail
source ~/mechrl-venv/bin/activate
cd ~/MechRL
git pull
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Aquaman's GPU is only ~8GB (vs L40S 46GB), and building 12 task prefilters is tight.
# expandable_segments reclaims fragmented (reserved-but-unallocated) memory to fit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent --policy batch \
    --tasks train --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad --seed 2 \
    --step-budget 150 --num-steps 512 --total-iterations 1200 \
    --device cuda --out runs 2>&1 | tee 12task_seed2_aquaman.out
