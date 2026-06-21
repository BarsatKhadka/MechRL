#!/bin/bash
# WARM-START head recovery (Magnolia gpuq / L40S). Re-evaluates each warm-started circuit with
# the CURRENT (complete) canonical sets and --dump-circuit, so warm-start head recovery is
# computed + auditable for ALL held-out tasks (incl. SVA, country-capital, gendered pronoun,
# which had no canonical set when the original warm-start eval ran). Frozen replay on each
# warm-start run dir; writes validity_<task>.json + circuit_<task>.json in that dir.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/warmstart_heads_magnolia.sbatch
# Then  :  sbatch slurm/collate_heads_cpu.sbatch          # head numbers for all warm-start dirs
#
#SBATCH --job-name=mechrl_wsh
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_wsh.%j.out
#SBATCH --error=mechrl_wsh.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] warm-start head eval start $(date)"
cd ~/MechRL
source venv/bin/activate

# task-metric-prefilter held-out (the warm-start runs used the default prefilter).
for TASK in GenderedPronounTask SubjectVerbAgreementTask AcronymTask SimpleSyllogismTask OppositeSyllogismTask CountryCapitalTask; do
    WS=$(ls -dt runs/${TASK}_seed1_*/ 2>/dev/null | head -1); WS=${WS%/}
    [ -z "${WS}" ] && { echo "[skip] no warm-start dir for ${TASK}"; continue; }
    echo "======== warm-start heads -> ${TASK}  [${WS}] ========"
    python -m scripts.zero_shot_eval --run "${WS}" --task "${TASK}" \
        --device cuda --num-rollouts 16 --dump-circuit \
        --out "${WS}/validity_${TASK}.json" || echo "[FAILED] ${TASK}"
done

# MCQ warm-start used the KL prefilter.
WS=$(ls -dt runs/MCQAnchoredBiasTask_seed1_*/ 2>/dev/null | head -1); WS=${WS%/}
if [ -n "${WS}" ]; then
    echo "======== warm-start heads -> MCQAnchoredBiasTask (KL) [${WS}] ========"
    python -m scripts.zero_shot_eval --run "${WS}" --task MCQAnchoredBiasTask \
        --prefilter-metric kl --device cuda --num-rollouts 16 --dump-circuit \
        --out "${WS}/validity_MCQAnchoredBiasTask_kl.json" || echo "[FAILED] MCQ"
fi

echo "[$(hostname)] warm-start head eval done $(date)"
