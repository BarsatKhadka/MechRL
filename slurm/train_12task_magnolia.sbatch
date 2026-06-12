#!/bin/bash
# 12-TASK run on Magnolia (gpuq / L40S) -- SEED 1, the replication seed for error bars.
# CIC runs seed 0; this is seed 1 so you get TWO seeds of the headline amortization
# result. Same config as slurm/train_12task_cic.sbatch (only --seed differs):
#   4 IOI + 4 GreaterThan + 4 Docstring variants, --pcgrad, headroom (auto in code),
#   --faith-margin 0 (per-task tau = each task's ceiling), lambda=5, 1200 iters.
#
# Queue it now -- gpuq is full, so it'll sit PENDING and start the moment a slot frees
# (no need to babysit). node049 stays excluded (it has issues). Checkpoints every 50.
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_12task_magnolia.sbatch
# Watch :  squeue -u $USER ; tail -f mechrl_12task_mag.<jobid>.out
#
#SBATCH --job-name=mechrl_12t_s1
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049         # node049 has known issues -- keep excluded
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                 # Magnolia nodes are large (~450G), 24G is fine here
#SBATCH --time=0                  # unlimited (gpuq allows it)
#SBATCH --output=mechrl_12task_mag.%j.out
#SBATCH --error=mechrl_12task_mag.%j.err

echo "[$(hostname)] 12-task seed1 start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# --pcgrad de-conflicts the 12 tasks' gradients (averaged combine -> scales to 12).
# If gpt2 tries to download and hangs on a compute node, the offline vars force cache.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent --policy batch \
    --tasks train --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad --seed 1 \
    --step-budget 150 --num-steps 512 --total-iterations 1200 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
