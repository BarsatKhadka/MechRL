#!/bin/bash
# ZERO-SHOT TRANSFER across ALL held-out tasks, one job. Frozen 2-task donor ->
# each held-out task (never trained on), NO fine-tuning. best of {greedy} u {16 samples}.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_all_cic.sbatch
# Read  :  tail -f mechrl_zsall.<jobid>.out   (three "=== ZERO-SHOT ===" blocks)
#
#SBATCH --job-name=mechrl_zsall
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=c1,c2
#SBATCH --output=mechrl_zsall.%j.out
#SBATCH --error=mechrl_zsall.%j.err
# NOTE: main does NOT use --gres (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] zero-shot ALL start $(date)"
nvidia-smi || true
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

DONOR=$(ls -dt runs/IOITask,GreaterThanOriginal_seed0_*/ 2>/dev/null | head -1 || true)
if [ -z "${DONOR}" ]; then echo "ERROR: no donor run dir found"; exit 1; fi
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

for TASK in GenderedPronounTask CopySuppressionTask SubjectVerbAgreementTask; do
    echo ""
    echo "########################################################################"
    echo "#  ZERO-SHOT TRANSFER -> ${TASK}"
    echo "########################################################################"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.zero_shot_eval \
        --run "${DONOR}" --task "${TASK}" --device cuda --num-rollouts 16 \
        --out "${DONOR}/zeroshot_${TASK}.json"
done

echo ""
echo "[$(hostname)] zero-shot ALL done $(date)"
