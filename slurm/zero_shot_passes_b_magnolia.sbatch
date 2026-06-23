#!/bin/bash
# EXACT zero-shot inference forward-pass count, group B = the remaining four held-out
# behaviours (Simple/Opposite syllogism, Country-capital, Multiple choice) for completeness
# beyond Table 4. MCQ uses the KL-attribution prefilter inside the script (as in the paper).
# See zero_shot_passes_a_magnolia.sbatch for the method.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_passes_b_magnolia.sbatch
# Watch :  tail -f mechrl_passesB.<jobid>.out   ;  result -> runs/zeroshot_passes_b.json
#
#SBATCH --job-name=mechrl_passesB
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_passesB.%j.out
#SBATCH --error=mechrl_passesB.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] zero-shot pass count (B) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

python -m scripts.zero_shot_passes \
    --run "${DONOR}" --device cuda --num-rollouts 16 \
    --tasks "SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask" \
    --out runs/zeroshot_passes_b.json \
    || echo "[FAILED] zero_shot_passes B"

echo "[$(hostname)] zero-shot pass count (B) done $(date)"
