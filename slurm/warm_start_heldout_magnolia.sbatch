#!/bin/bash
# WARM-START transfer (Results 5.1) on Magnolia (gpuq / L40S).
# Take the frozen 12-task donor policy as the INITIALISATION (--init-from: fresh run,
# iter 0, fresh LR schedule -- NOT --resume-from, which would resume at LR~0 and not
# learn) and adapt it a SHORT budget on a HELD-OUT behaviour. Measures iters-to-bar:
# how cheaply the learned cutting skill adapts to a distant mechanism vs from scratch.
#
#   CopySuppression  -- zero-shot FAILED (suppression is the opposite of move/copy);
#                       the key warm-start test: does a handful of iters rescue it?
#   SubjectVerbAgree -- zero-shot near-miss (0.921 vs strict bar 0.958); should clear fast.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/warm_start_heldout_magnolia.sbatch
# Watch :  tail -f mechrl_ws.<jobid>.out
#   -> the "[bar] <Task> ceiling=.. -> tau=.." line is the bar; then watch the per-iter
#      faith climb. iters-to-bar = first iter whose per-task faith >= tau. Checkpoints
#      every 10 iters (policy_iter10/20/..) -> feed the first clearing one to zero_shot_eval
#      / battery_test for the validity + head-recovery check.
#
#SBATCH --job-name=mechrl_ws
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=mechrl_ws.%j.out
#SBATCH --error=mechrl_ws.%j.err

echo "[$(hostname)] warm-start (12-task donor) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Auto-find the newest 12-task seed-1 run (the trained donor policy).
DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor (init-from): ${DONOR}"
ls "${DONOR}/config.json" || { echo "ERROR: missing config.json"; exit 1; }

# Short adaptation: fresh LR schedule annealed over 150 iters, checkpoints every 10 so
# we can pin the first iter that clears the bar. Single task -> PCGrad is moot (omitted).
# faith-margin 0 keeps the same strict per-task tau (= ceiling) as the 12-task training.
for TASK in CopySuppressionTask SubjectVerbAgreementTask; do
    echo ""
    echo "######## WARM-START -> ${TASK} ########"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m mechrl.train.train_agent \
        --policy batch --tasks "${TASK}" \
        --num-examples 20 --batch-sizes 1 3 10 30 \
        --faith-margin 0.0 --threshold-penalty 5 --seed 1 \
        --step-budget 150 --num-steps 512 \
        --total-iterations 150 --save-every 10 --log-every 1 \
        --init-from "${DONOR}" --device cuda --out runs
done

echo "[$(hostname)] warm-start done $(date)"
