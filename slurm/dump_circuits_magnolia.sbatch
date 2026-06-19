#!/bin/bash
# DUMP head-recovery circuits for manual verification (Magnolia gpuq / L40S).
# Re-runs the frozen donor on each held-out behaviour that has canonical heads, with
# --dump-circuit, writing circuit_<task>.json = EVERY edge + every head present + the
# canonical set + the intersection. Verify the recovery counts by hand, do not trust them.
#
# The circuit is re-extracted (best-of-16), so it is a fresh draw; the dumped edges ARE the
# circuit whose recovery that run reports, so heads_present vs canonical_heads is internally
# consistent. (Counts may differ slightly from the first table run, which was a different draw;
# treat these dumped runs as the auditable version and update the table from them.)
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/dump_circuits_magnolia.sbatch
# Read  :  cat runs/train_seed1_*/circuit_<task>.json   (heads_present vs canonical_heads)
#
#SBATCH --job-name=mechrl_dump
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_dump.%j.out
#SBATCH --error=mechrl_dump.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] dump circuits start $(date)"
cd ~/MechRL
source venv/bin/activate

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1); DONOR=${DONOR%/}
[ -z "${DONOR}" ] && { echo "ERROR: no donor"; exit 1; }
echo "donor: ${DONOR}"

# Held-out behaviours with published canonical heads (logit-diff prefilter).
for TASK in AcronymTask SimpleSyllogismTask OppositeSyllogismTask; do
    echo ""
    echo "======== DUMP -> ${TASK} ========"
    python -m scripts.zero_shot_eval --run "${DONOR}" --task "${TASK}" \
        --device cuda --num-rollouts 16 --dump-circuit \
        --out "${DONOR}/zeroshot_${TASK}.json" || echo "[FAILED] dump ${TASK}"
done

# MCQ needs the KL-attribution prefilter (its circuit is invisible to logit-diff).
echo ""
echo "======== DUMP -> MCQAnchoredBiasTask (KL prefilter) ========"
python -m scripts.zero_shot_eval --run "${DONOR}" --task MCQAnchoredBiasTask \
    --prefilter-metric kl --device cuda --num-rollouts 16 --dump-circuit \
    --out "${DONOR}/zeroshot_MCQAnchoredBiasTask_kl.json" || echo "[FAILED] dump MCQ"

echo "[$(hostname)] dump circuits done $(date)"
