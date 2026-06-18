#!/bin/bash
# AMORTISATION eval (Results 5.2) on Magnolia (gpuq / L40S).
# Replay the FROZEN 12-task donor on each of its OWN 12 training behaviours and score the
# circuit it produces -- the core amortisation evidence: ONE policy -> 12 faithful circuits,
# at amortised cost. Per task: faith + size + argmax/logit-diff recovery + necessity (knockout)
# + specificity (random same-size). For canonical IOI, zero_shot_eval reports 26-head recovery.
# These are also the AGENT's points on the faith-vs-size / faith-vs-cost Pareto against the
# ACDC + EAP-IG baselines (baselines_acdc_eapig job).
#
# This is frozen-replay (the tasks ARE in the donor's training set) -- zero_shot_eval prints
# "in training set (frozen-replay sanity)", which is exactly what we want for §5.2.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/amortisation_12task_eval_magnolia.sbatch
# Watch :  tail -f mechrl_amort.<jobid>.out   (one "=== ZERO-SHOT (<task>) ===" block per task)
#
#SBATCH --job-name=mechrl_amort
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_amort.%j.out
#SBATCH --error=mechrl_amort.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] amortisation eval start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1); DONOR=${DONOR%/}
[ -z "${DONOR}" ] && { echo "ERROR: no donor"; exit 1; }
echo "donor: ${DONOR}"

# The 12 trained behaviours (3 families x 4 variants).
TRAIN_TASKS="IOITask IOIAfterOpener IOINoPlaceObject IOIFriendsFound \
GreaterThanOriginal GreaterThanReversed GreaterThanBeganEnded GreaterThanTookPlace \
DocstringGPT2Task DocstringGPT2ClassSphinxTask DocstringSphinxDescTask DocstringFuncSphinxTask"

for TASK in ${TRAIN_TASKS}; do
    echo ""
    echo "======== AMORTISATION REPLAY -> ${TASK} ========"
    python -m scripts.zero_shot_eval --run "${DONOR}" --task "${TASK}" \
        --device cuda --num-rollouts 16 \
        --out "${DONOR}/amort_${TASK}.json" || echo "[FAILED] amort ${TASK}"
done

echo ""
echo "[$(hostname)] amortisation eval done $(date)"
