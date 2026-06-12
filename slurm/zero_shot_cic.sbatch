#!/bin/bash
# ZERO-SHOT eval on a CIC compute node: frozen donor policy -> target task, NO training.
# Default target = GreaterThanOriginal. Change TASK to a HELD-OUT task (e.g.
# GenderedPronounTask, CopySuppressionTask, SubjectVerbAgreementTask) for true
# zero-shot TRANSFER (donor never trained on it).
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_cic.sbatch
# Read  :  tail -f mechrl_zeroshot.<jobid>.out
#
#SBATCH --job-name=mechrl_zs
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --exclude=c1,c2
#SBATCH --output=mechrl_zeroshot.%j.out
#SBATCH --error=mechrl_zeroshot.%j.err
# NOTE: main does NOT use --gres (one GPU/node, auto-allocated).

set -euo pipefail
TASK=GreaterThanOriginal          # change to a held-out task for zero-shot TRANSFER
echo "[$(hostname)] zero-shot start $(date)  task=${TASK}"
nvidia-smi || true
module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

# auto-find the 2-task donor (newest IOI+GreaterThan run); pins its latest ckpt (iter600)
DONOR=$(ls -dt runs/IOITask,GreaterThanOriginal_seed0_*/ 2>/dev/null | head -1 || true)
if [ -z "${DONOR}" ]; then echo "ERROR: no donor run dir found"; exit 1; fi
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.zero_shot_eval \
    --run "${DONOR}" --task "${TASK}" --device cuda --num-rollouts 16 \
    --out "${DONOR}/zeroshot_${TASK}.json"

echo "[$(hostname)] zero-shot done $(date)"
