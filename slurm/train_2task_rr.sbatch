#!/bin/bash
# 2-TASK ROUND-ROBIN: train ONE task per iteration (IOI, GreaterThan, IOI, ...) so each
# update is single-task-clean -- no cross-task mixing, no overshadowing, no adv-norm noise.
# THE diagnostic: does IOI reach its ~0.90 bar SMOOTHLY (like the single-task runs) when it
# gets clean un-mixed updates? If yes -> mixing was the culprit. Watch the per-task "└" line:
# on IOI iters only IOI shows, on GreaterThan iters only GreaterThan -- each should climb.
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_2task_rr.sbatch
#
#SBATCH --job-name=mechrl_rr
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=mechrl_2task_rr.%j.out
#SBATCH --error=mechrl_2task_rr.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.05 --threshold-penalty 5 \
    --round-robin \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
