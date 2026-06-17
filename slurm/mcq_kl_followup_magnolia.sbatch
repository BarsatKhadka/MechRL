#!/bin/bash
# MCQ follow-up under KL-ATTRIBUTION PREFILTER (Magnolia gpuq / L40S).
# The gate showed MCQAnchoredBiasTask is invisible to the logit-diff prefilter (faith ~0 at
# every K) but reaches ~0.95 faith under KL-attribution prefilter -- the prefilter must match
# the KL objective for this task. The main pipeline (results_5_1_full) runs MCQ under logit-diff
# (artifact failure); this evaluates it under KL where its circuit actually lives.
#
#   zero-shot  : frozen donor -> MCQ, --prefilter-metric kl, best-of-16
#   warm-start : --init-from donor, --prefilter-metric kl, 80 iters
#   validity   : re-evaluate the warm-started circuit, --prefilter-metric kl
#
# NOTE: the donor trained with logit-diff prefilter, so MCQ here is zero-shot transfer onto a
# KL-prefiltered candidate set (a mild train/test prefilter mismatch -- report honestly). If we
# decide to retrain the donor with --prefilter-metric kl, this becomes fully in-distribution.
#
# Submit (AFTER results_5_1_full finishes, or anytime -- it is independent):
#   cd ~/MechRL && git pull && sbatch slurm/mcq_kl_followup_magnolia.sbatch
# Watch: tail -f mechrl_mcqkl.<jobid>.out
#
#SBATCH --job-name=mechrl_mcqkl
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=mechrl_mcqkl.%j.out
#SBATCH --error=mechrl_mcqkl.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] MCQ KL-prefilter follow-up start $(date)"
cd ~/MechRL
source venv/bin/activate

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1); DONOR=${DONOR%/}
[ -z "${DONOR}" ] && { echo "ERROR: no donor"; exit 1; }
echo "donor: ${DONOR}"

echo ""
echo "======== ZERO-SHOT (KL prefilter) -> MCQAnchoredBiasTask ========"
python -m scripts.zero_shot_eval --run "${DONOR}" --task MCQAnchoredBiasTask \
    --prefilter-metric kl --device cuda --num-rollouts 16 \
    --out "${DONOR}/zeroshot_MCQAnchoredBiasTask_kl.json" || echo "[FAILED] zero-shot MCQ kl"

echo ""
echo "======== WARM-START (KL prefilter) -> MCQAnchoredBiasTask ========"
python -m mechrl.train.train_agent --policy batch --tasks MCQAnchoredBiasTask \
    --prefilter-metric kl --num-examples 20 --batch-sizes 1 3 10 30 \
    --faith-margin 0.0 --threshold-penalty 5 --seed 1 \
    --step-budget 150 --num-steps 512 \
    --total-iterations 80 --save-every 10 --log-every 1 \
    --init-from "${DONOR}" --device cuda --out runs || echo "[FAILED] warm-start MCQ kl"

WS=$(ls -dt runs/MCQAnchoredBiasTask_seed1_*/ 2>/dev/null | head -1); WS=${WS%/}
if [ -n "${WS}" ]; then
    echo ""
    echo "======== VALIDITY (KL prefilter) -> MCQAnchoredBiasTask  [${WS}] ========"
    python -m scripts.zero_shot_eval --run "${WS}" --task MCQAnchoredBiasTask \
        --prefilter-metric kl --device cuda --num-rollouts 16 \
        --out "${WS}/validity_MCQAnchoredBiasTask_kl.json" || echo "[FAILED] validity MCQ kl"
fi

echo "[$(hostname)] MCQ KL-prefilter follow-up done $(date)"
