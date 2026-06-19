#!/bin/bash
# LAMBDA-RUN EVALUATION (Magnolia gpuq / L40S). Evaluates a finished lambda-sweep run in
# runs_lam<lambda>/ exactly like the main pipeline: zero-shot transfer on every held-out
# behaviour (+ MCQ under KL prefilter, + circuit dumps for head verification) and the
# amortisation replay on all twelve training behaviours. Produces the same JSONs the
# collator reads, so afterwards `collate_results --runs runs_lam<lambda>` gives the tables.
#
# Run it AFTER the matching training job, chained by dependency so it waits automatically:
#   cd ~/MechRL && git pull
#   J4=$(sbatch --parsable slurm/lambda_sweep_magnolia.sbatch 4)
#   sbatch --dependency=afterok:$J4 slurm/lambda_eval_magnolia.sbatch 4
#   J6=$(sbatch --parsable slurm/lambda_sweep_magnolia.sbatch 6)
#   sbatch --dependency=afterok:$J6 slurm/lambda_eval_magnolia.sbatch 6
# (If the training jobs are already queued/running, use their job ids in --dependency instead.)
#
# Warm-start is intentionally omitted here -- the lambda sweep tests the zero-shot/amortised
# operating point (faith vs size), not adaptation, and warm-starting every task x2 lambdas
# would add days for no bearing on the Pareto.
#
#SBATCH --job-name=mechrl_leval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_leval.%j.out
#SBATCH --error=mechrl_leval.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LAMBDA=${1:?usage: sbatch slurm/lambda_eval_magnolia.sbatch <lambda>   e.g. 4 or 6}
ROOT="runs_lam${LAMBDA}"

echo "[$(hostname)] lambda=${LAMBDA} eval start $(date)"
cd ~/MechRL
source venv/bin/activate

DONOR=$(ls -dt ${ROOT}/train_seed1_*/ 2>/dev/null | head -1); DONOR=${DONOR%/}
[ -z "${DONOR}" ] && { echo "ERROR: no ${ROOT}/train_seed1_* run dir"; exit 1; }
echo "lambda=${LAMBDA} donor: ${DONOR}"

HELD_OUT="GenderedPronounTask SubjectVerbAgreementTask AcronymTask SimpleSyllogismTask OppositeSyllogismTask CountryCapitalTask"
TRAIN="IOITask IOIAfterOpener IOINoPlaceObject IOIFriendsFound \
GreaterThanOriginal GreaterThanReversed GreaterThanBeganEnded GreaterThanTookPlace \
DocstringGPT2Task DocstringGPT2ClassSphinxTask DocstringSphinxDescTask DocstringFuncSphinxTask"

echo ""
echo "################ ZERO-SHOT TRANSFER (held-out) ################"
for TASK in ${HELD_OUT}; do
    echo "======== zero-shot -> ${TASK} ========"
    python -m scripts.zero_shot_eval --run "${DONOR}" --task "${TASK}" \
        --device cuda --num-rollouts 16 --dump-circuit \
        --out "${DONOR}/zeroshot_${TASK}.json" || echo "[FAILED] zero-shot ${TASK}"
done
echo "======== zero-shot -> MCQAnchoredBiasTask (KL prefilter) ========"
python -m scripts.zero_shot_eval --run "${DONOR}" --task MCQAnchoredBiasTask \
    --prefilter-metric kl --device cuda --num-rollouts 16 --dump-circuit \
    --out "${DONOR}/zeroshot_MCQAnchoredBiasTask_kl.json" || echo "[FAILED] MCQ kl"

echo ""
echo "################ AMORTISATION (12 training behaviours, frozen replay) ################"
for TASK in ${TRAIN}; do
    echo "======== amort -> ${TASK} ========"
    python -m scripts.zero_shot_eval --run "${DONOR}" --task "${TASK}" \
        --device cuda --num-rollouts 16 \
        --out "${DONOR}/amort_${TASK}.json" || echo "[FAILED] amort ${TASK}"
done

echo ""
echo "[$(hostname)] lambda=${LAMBDA} eval done $(date). Collate with: python -m scripts.collate_results --runs ${ROOT}"
