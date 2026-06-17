#!/bin/bash
# ZERO-SHOT TRANSFER eval (Results 5.1) on Magnolia (gpuq / L40S).
# Frozen 12-task donor policy -> each of the 3 HELD-OUT behaviours (never trained on),
# NO fine-tuning, best-of-16 rollouts. Per task: faith + argmax recovery + necessity
# (knockout) + specificity (random same-size) -- the held-out validity check.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_12task_magnolia.sbatch
# Watch :  tail -f mechrl_zs12.<jobid>.out   (three "=== ZERO-SHOT ===" blocks)
#
#SBATCH --job-name=mechrl_zs12
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=mechrl_zs12.%j.out
#SBATCH --error=mechrl_zs12.%j.err

echo "[$(hostname)] zero-shot (12-task donor) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Auto-find the newest 12-task seed-1 run (the trained donor policy).
DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"
ls "${DONOR}/config.json" "${DONOR}/policy_final.pt" || { echo "ERROR: missing config.json / policy_final.pt"; exit 1; }

# offline vars in case the compute node has no internet (gpt2 is cached).
for TASK in CopySuppressionTask GenderedPronounTask SubjectVerbAgreementTask; do
    echo ""
    echo "######## ZERO-SHOT -> ${TASK} ########"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.zero_shot_eval \
        --run "${DONOR}" --task "${TASK}" --device cuda --num-rollouts 16 \
        --out "${DONOR}/zeroshot_${TASK}.json"
done

echo "[$(hostname)] zero-shot done $(date)"
