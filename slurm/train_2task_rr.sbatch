#!/bin/bash
# 2-TASK ROUND-ROBIN @ MARGIN 0 -- the A/B vs mixed-margin-0 (the untested cell).
# Train ONE task per iteration (IOI, GreaterThan, IOI, ...) so each update is single-task
# clean -- no cross-task mixing. faith-margin 0 = correct bar (tau=ceiling). THE test:
# mixed-margin-0 settled IOI ~0.83 (below single-task's 0.91). Does round-robin (clean
# per-task gradients) recover IOI toward ~0.91? If yes -> multi-task interference WAS the
# gap, schedule matters. If it also lands ~0.83 -> IOI just gives up a bit in multi-task.
# Watch the per-task "└" line: IOI iters show only IOI, GreaterThan iters only GreaterThan.
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
#SBATCH --time=0                 # 0 = UNLIMITED walltime (gpuq allows it). If SLURM
                                 # rejects, set the partition max (sinfo -p gpuq -o "%l").
#SBATCH --output=mechrl_2task_rr.%j.out
#SBATCH --error=mechrl_2task_rr.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --round-robin \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
