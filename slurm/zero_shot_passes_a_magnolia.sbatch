#!/bin/bash
# EXACT zero-shot inference forward-pass count, group A = the six behaviours in Table 4
# (§5.2): the three training-set tasks (IOI, Greater-than, Docstring) + three held-out
# (Gendered pronoun, Acronyms, Subject-verb). Wraps engine.run_with_mask (the same
# one-forward-pass primitive run_acdc_greedy counts) and tallies calls across the best-of-16
# frozen rollouts -> a Policy `passes` number directly comparable to ACDC's column.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_passes_a_magnolia.sbatch
# Watch :  tail -f mechrl_passesA.<jobid>.out   ;  result -> runs/zeroshot_passes_a.json
#
#SBATCH --job-name=mechrl_passesA
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_passesA.%j.out
#SBATCH --error=mechrl_passesA.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] zero-shot pass count (A) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

python -m scripts.zero_shot_passes \
    --run "${DONOR}" --device cuda --num-rollouts 16 \
    --tasks "IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,AcronymTask,SubjectVerbAgreementTask" \
    --out runs/zeroshot_passes_a.json \
    || echo "[FAILED] zero_shot_passes A"

echo "[$(hostname)] zero-shot pass count (A) done $(date)"
