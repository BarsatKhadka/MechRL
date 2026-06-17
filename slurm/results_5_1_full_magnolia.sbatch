#!/bin/bash
# RESULTS 5.1 -- FULL transfer pipeline on Magnolia (gpuq / L40S), all held-out tasks,
# sequentially, from the frozen 12-task donor. Four stages, every task treated equally:
#
#   STAGE 0  GATE      verify_prefilter_kl (faith vs K, logitdiff vs kl prefilter) +
#                      probe_task_ceiling (KL_cut = counterfactual strength). The DATA
#                      says which tasks have a strong counterfactual + sparse circuit.
#   STAGE 1  ZERO-SHOT frozen donor -> each task, best-of-16, NO training. faith + argmax +
#                      logit-diff recovery + necessity + specificity + canonical-head recovery.
#   STAGE 2  WARM-START --init-from donor (fresh run, iter 0, live LR) -> each task, 80 iters,
#                      checkpoints every 10. iters-to-bar = first per-task faith >= tau.
#   STAGE 3  VALIDITY  re-evaluate each warm-started circuit (necessity/specificity/heads).
#
# Long job by design (warm-start x 8 tasks). gpuq allows --time=0. Each task writes its own
# files/run dir as it finishes, so partial progress survives a node failure.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/results_5_1_full_magnolia.sbatch
# Watch :  tail -f mechrl_r51.<jobid>.out
#   STAGE banners separate the phases; per task look for "=== ZERO-SHOT ===" and the
#   warm-start "[bar] ... tau=" line then the faith climb.
#
#SBATCH --job-name=mechrl_r51
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_r51.%j.out
#SBATCH --error=mechrl_r51.%j.err

set +e                                  # one task failing must not kill the whole pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] RESULTS 5.1 full pipeline start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"
ls "${DONOR}/config.json" || { echo "ERROR: missing config.json"; exit 1; }

# All eight held-out behaviours, treated equally.
HELD_OUT="GenderedPronounTask SubjectVerbAgreementTask CopySuppressionTask AcronymTask \
SimpleSyllogismTask OppositeSyllogismTask CountryCapitalTask MCQAnchoredBiasTask"
HELD_OUT_CSV=$(echo ${HELD_OUT} | tr ' ' ',')

echo ""
echo "################ STAGE 0: GATE (counterfactual strength + sparse-circuit ceiling) ################"
python -m scripts.probe_task_ceiling --tasks "${HELD_OUT_CSV}" --num-examples 50 --device cuda
echo ""
echo "---- prefilter comparison (logitdiff vs kl-attr, faith vs K) ----"
python -m scripts.verify_prefilter_kl --tasks "${HELD_OUT_CSV}" \
    --ks 300,600,1000,1500,3000 --num-examples 50 --device cuda

echo ""
echo "################ STAGE 1: ZERO-SHOT TRANSFER (frozen donor) ################"
for TASK in ${HELD_OUT}; do
    echo ""
    echo "======== ZERO-SHOT -> ${TASK} ========"
    python -m scripts.zero_shot_eval --run "${DONOR}" --task "${TASK}" \
        --device cuda --num-rollouts 16 --out "${DONOR}/zeroshot_${TASK}.json" \
        || echo "[FAILED] zero-shot ${TASK}"
done

echo ""
echo "################ STAGE 2: WARM-START (--init-from donor, 80 iters each) ################"
for TASK in ${HELD_OUT}; do
    echo ""
    echo "======== WARM-START -> ${TASK} ========"
    python -m mechrl.train.train_agent --policy batch --tasks "${TASK}" \
        --num-examples 20 --batch-sizes 1 3 10 30 \
        --faith-margin 0.0 --threshold-penalty 5 --seed 1 \
        --step-budget 150 --num-steps 512 \
        --total-iterations 80 --save-every 10 --log-every 1 \
        --init-from "${DONOR}" --device cuda --out runs \
        || echo "[FAILED] warm-start ${TASK}"
done

echo ""
echo "################ STAGE 3: VALIDITY on warm-started circuits ################"
for TASK in ${HELD_OUT}; do
    WS=$(ls -dt runs/${TASK}_seed1_*/ 2>/dev/null | head -1)
    [ -z "${WS}" ] && { echo "[skip] no warm-start run dir for ${TASK}"; continue; }
    WS=${WS%/}
    echo ""
    echo "======== VALIDITY (warm-started) -> ${TASK}  [${WS}] ========"
    python -m scripts.zero_shot_eval --run "${WS}" --task "${TASK}" \
        --device cuda --num-rollouts 16 --out "${WS}/validity_${TASK}.json" \
        || echo "[FAILED] validity ${TASK}"
done

echo ""
echo "[$(hostname)] RESULTS 5.1 full pipeline done $(date)"
