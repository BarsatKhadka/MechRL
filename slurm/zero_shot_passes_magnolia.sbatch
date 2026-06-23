#!/bin/bash
# EXACT zero-shot inference forward-pass count per behaviour (for the §5.2 cost column).
# Wraps engine.run_with_mask (the same one-forward-pass primitive run_acdc_greedy counts) and
# tallies calls across the best-of-16 frozen rollouts -> a number directly comparable to ACDC's
# `passes`. Runs the frozen 12-task donor on the §5.2 table behaviours + all held-out ones.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/zero_shot_passes_magnolia.sbatch
# Watch :  tail -f mechrl_passes.<jobid>.out  -> per-task "total fwd passes = ..." line.
# Result:  runs/zeroshot_passes.json (+ the SUMMARY table at the end of the log).
#
#SBATCH --job-name=mechrl_passes
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_passes.%j.out
#SBATCH --error=mechrl_passes.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] zero-shot pass count start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Frozen 12-task donor (newest seed-1 training run), same one the transfer results use.
DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

python -m scripts.zero_shot_passes \
    --run "${DONOR}" --device cuda --num-rollouts 16 \
    --tasks "IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask" \
    --out runs/zeroshot_passes.json \
    || echo "[FAILED] zero_shot_passes"

echo "[$(hostname)] zero-shot pass count done $(date)"
