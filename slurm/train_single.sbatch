#!/bin/bash
# Single-task ACHIEVABILITY check: can the AGENT (not just the top-K ceiling) reach
# high faith on this task? Set TAU at/below the task's ceiling (see TASKS.md) so the
# bar is actually clearable. IOI is already confirmed (~0.94); run this for the other
# train families before committing to multi-task.
#
# Submit (defaults shown):
#   sbatch --export=ALL,TASK=GreaterThanOriginal,TAU=0.95,LAM=6 slurm/train_single.sbatch
#   sbatch --export=ALL,TASK=DocstringGPT2Task,TAU=0.85,LAM=6   slurm/train_single.sbatch
# Watch: tail -f mechrl_single.<jobid>.out  -> does `faith` climb to ~TAU and `kept` drop?
#
#SBATCH --job-name=mechrl_1t
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_single.%j.out
#SBATCH --error=mechrl_single.%j.err

TASK=${TASK:-IOITask}
TAU=${TAU:-0.95}
LAM=${LAM:-6}
ITERS=${ITERS:-150}

echo "[$(hostname)] start $(date)  TASK=$TASK TAU=$TAU LAM=$LAM ITERS=$ITERS"
cd ~/MechRL
source venv/bin/activate

python -m mechrl.train.train_agent --policy batch \
    --tasks "$TASK" --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-threshold "$TAU" --threshold-penalty "$LAM" \
    --step-budget 150 --num-steps 512 --total-iterations "$ITERS" \
    --device cuda --out runs

echo "[$(hostname)] done $(date)"
