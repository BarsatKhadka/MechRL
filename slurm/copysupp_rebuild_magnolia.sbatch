#!/bin/bash
# COPY-SUPPRESSION REBUILD test on Magnolia (gpuq / L40S).
# Copy-suppression was the one held-out task NOT built as a clean contrastive minimal pair
# (it measured a single absolute logit with a garbage distractor, and the prefilter/edge
# features had no sharp contrast to lock onto -- against our own method). It is now fixed to
# a contrastive metric -(logit[seen] - logit[unseen]), same shape as the tasks that transfer.
# This re-gates it, PROVES the circuit is findable by a principled non-RL search (ACDC), then
# re-runs transfer. If the contrastive version transfers, copy-suppression flips from
# "honest failure" to "works once we follow our own contrastive design".
#
#   STAGE 0  GATE   probe_task_ceiling + verify_prefilter_kl on CopySuppressionTask.
#   STAGE A  ACDC   run_acdc_greedy on CopySuppressionTask -> findability proof (size @ faith).
#                   If ACDC finds a small faithful circuit, the circuit was always there and
#                   the gap was the task/agent, not the model.
#   STAGE 1  ZERO-SHOT  frozen donor -> copy-suppression (contrastive), best-of-16.
#   STAGE 2  WARM-START --init-from donor, 80 iters, checkpoints every 10.
#   STAGE 3  VALIDITY   re-evaluate the warm-started circuit.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/copysupp_rebuild_magnolia.sbatch
# Watch :  tail -f mechrl_cs.<jobid>.out
#
#SBATCH --job-name=mechrl_cs
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_cs.%j.out
#SBATCH --error=mechrl_cs.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] copy-suppression rebuild start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1); DONOR=${DONOR%/}
[ -z "${DONOR}" ] && { echo "ERROR: no donor"; exit 1; }
echo "donor: ${DONOR}"

echo ""
echo "################ STAGE 0: GATE (contrastive copy-suppression) ################"
python -m scripts.probe_task_ceiling --tasks CopySuppressionTask --num-examples 50 --device cuda
python -m scripts.verify_prefilter_kl --tasks CopySuppressionTask \
    --ks 300,600,1000,1500,3000 --num-examples 50 --device cuda

echo ""
echo "################ STAGE A: ACDC findability proof (principled non-RL search) ################"
python -m scripts.run_acdc_greedy --task CopySuppressionTask --device cuda \
    --num-examples 20 --thresholds 0.001 0.003 0.01 \
    --out runs/acdc_copysuppression.json || echo "[FAILED] ACDC copy-suppression"

echo ""
echo "################ STAGE 1: ZERO-SHOT (frozen donor, contrastive) ################"
python -m scripts.zero_shot_eval --run "${DONOR}" --task CopySuppressionTask \
    --device cuda --num-rollouts 16 \
    --out "${DONOR}/zeroshot_CopySuppressionTask_contrastive.json" || echo "[FAILED] zero-shot"

echo ""
echo "################ STAGE 2: WARM-START (--init-from donor, 80 iters) ################"
python -m mechrl.train.train_agent --policy batch --tasks CopySuppressionTask \
    --num-examples 20 --batch-sizes 1 3 10 30 \
    --faith-margin 0.0 --threshold-penalty 5 --seed 1 \
    --step-budget 150 --num-steps 512 \
    --total-iterations 80 --save-every 10 --log-every 1 \
    --init-from "${DONOR}" --device cuda --out runs || echo "[FAILED] warm-start"

WS=$(ls -dt runs/CopySuppressionTask_seed1_*/ 2>/dev/null | head -1); WS=${WS%/}
if [ -n "${WS}" ]; then
    echo ""
    echo "################ STAGE 3: VALIDITY (warm-started) ################"
    python -m scripts.zero_shot_eval --run "${WS}" --task CopySuppressionTask \
        --device cuda --num-rollouts 16 \
        --out "${WS}/validity_CopySuppressionTask.json" || echo "[FAILED] validity"
fi

echo "[$(hostname)] copy-suppression rebuild done $(date)"
