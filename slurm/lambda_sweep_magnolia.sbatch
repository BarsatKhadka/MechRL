#!/bin/bash
# LAMBDA SWEEP -- full 12-task training at a chosen faith-penalty weight (Magnolia gpuq / L40S).
# Reward is Phi = w(1 - k/K) - lambda * max(0, tau - f). lambda is the threshold-penalty:
# higher lambda keeps more edges to protect faithfulness, lower lambda cuts harder for smaller
# circuits. Sweeping it traces the AGENT'S faith-vs-size Pareto (the answer to "ACDC is smaller").
#
# The existing donor is lambda=5 (runs/train_seed1_1781368586). This runs lambda=4 and lambda=6
# from scratch, SAME seed (1) and SAME everything else, so only lambda differs -- a clean
# controlled comparison. Output goes to runs_lam<lambda>/ so it never collides with the donor
# in runs/ (and so the eval scripts' "ls -dt runs/train_seed1_*" donor detection is unaffected).
# The prefilter is attribution-based and lambda-independent, so candidate sets are identical
# across runs; only the reward changes.
#
# Submit BOTH (each ~a full 1200-iter run, long; gpuq --time=0):
#   cd ~/MechRL && git pull
#   sbatch slurm/lambda_sweep_magnolia.sbatch 4
#   sbatch slurm/lambda_sweep_magnolia.sbatch 6
# Watch: tail -f mechrl_lam.<jobid>.out   (per-task faith/edges by iter, like the original run)
#
# AFTER training, evaluate each run by pointing --run at its dir explicitly (NOT auto-detect):
#   python -m scripts.zero_shot_eval --run runs_lam4/train_seed1_* --task <T> ...
#   python -m scripts.collate_results --runs runs_lam4
# then plot faith vs |C| across lambda = 4, 5, 6.
#
#SBATCH --job-name=mechrl_lam
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_lam.%j.out
#SBATCH --error=mechrl_lam.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LAMBDA=${1:?usage: sbatch slurm/lambda_sweep_magnolia.sbatch <lambda>   e.g. 4 or 6}

echo "[$(hostname)] lambda-sweep lambda=${LAMBDA} start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Identical to the donor's 12-task command, only --threshold-penalty and --out differ.
python -m mechrl.train.train_agent --policy batch --tasks train \
    --num-examples 20 --batch-sizes 1 3 10 30 \
    --faith-margin 0.0 --threshold-penalty "${LAMBDA}" --pcgrad --seed 1 \
    --step-budget 150 --num-steps 512 --total-iterations 1200 \
    --device cuda --out "runs_lam${LAMBDA}"

echo "[$(hostname)] lambda-sweep lambda=${LAMBDA} done $(date)"
