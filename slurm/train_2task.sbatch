#!/bin/bash
# 2-TASK VALIDATION run (IOI + GreaterThan -- two different mechanisms, both 0.95+
# ceiling). Answers the core multi-task question before the expensive 12-task run:
# can ONE shared policy learn two different circuits (both hit faith>=tau, both
# compress), with no collapse and no OOM? Watch the per-task "└ ..." line.
#
# If it passes, scale up WITHOUT wasting this run:
#   --init-from runs/<this_run_dir> --tasks train --total-iterations 500
# (warm-starts a fresh 12-task run from this policy.)
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_2task.sbatch
# Watch :  tail -f mechrl_2task.<jobid>.out   (look for BOTH tasks' faith climbing)
#
#SBATCH --job-name=mechrl_2t
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=mechrl_2task.%j.out
#SBATCH --error=mechrl_2task.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# per-task tau auto-derived from each task's ceiling (--faith-margin 0.05); lambda=5.
# Watch the "[bar]" lines at startup (ceiling->tau per task) and the per-task "└ ..."
# lines for BOTH tasks' faith climbing and clearing their tau (✓).
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.05 --threshold-penalty 5 \
    --step-budget 150 --num-steps 512 --total-iterations 200 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
