#!/bin/bash
# THE 12-task multi-task run: IOI x4 + GreaterThan x4 + Docstring x4 (--tasks train).
# Standard mixed PPO (no round-robin, no per-task adv-norm -- those were fixes for the
# wrong problem). lambda=5. faith-margin 0 => per-task tau = each task's measured CEILING
# (the proven Aquaman setting that held IOI at 0.91; margin 0.05 was below IOI's cliff).
# Per-task tau printed at startup as "[bar] <Task> ceiling=.. -> tau=.." (= ceiling).
#
# Watch the per-task "└" line: every task should climb toward its tau and hold (✓).
# IOI especially -- it should lock ~0.91 now, not slide to 0.77.
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_12task.sbatch
#
#SBATCH --job-name=mechrl_12t
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0                 # unlimited (gpuq allows it). If rejected: sinfo -p gpuq -o "%l"
#SBATCH --output=mechrl_12task.%j.out
#SBATCH --error=mechrl_12task.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# 12 tasks share ONE frozen GPT-2. ~800 iters (each task gets ~1/12 of episodes, so it
# needs more total iters than 2-task). Checkpoints every 50; --resume-from to extend.
python -m mechrl.train.train_agent --policy batch \
    --tasks train --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --step-budget 150 --num-steps 512 --total-iterations 800 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
