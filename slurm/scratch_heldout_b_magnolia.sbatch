#!/bin/bash
# FROM-SCRATCH held-out baseline, group B (4 tasks) -- Magnolia (gpuq / L40S).
# See scratch_heldout_a_magnolia.sbatch for the full rationale. Trains a FRESH policy per
# held-out task (NO --init-from), same bar (--faith-margin 0.0) and hyperparameters as the
# warm-start runs, so the from-scratch vs warm-start learning curves are directly comparable.
#
# MCQAnchoredBiasTask is invisible to the default logit-diff prefilter (faith ~0 at every K)
# and only has a faithful circuit under KL-attribution prefilter, so it -- and ONLY it -- runs
# with --prefilter-metric kl, matching how it appears in the transfer table (the dagger).
#
# 200 iters each, no --time limit. set +e: one task failing does not lose the rest.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/scratch_heldout_b_magnolia.sbatch
# Watch :  tail -f mechrl_scratchB.<jobid>.out
#
#SBATCH --job-name=mechrl_scratchB
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_scratchB.%j.out
#SBATCH --error=mechrl_scratchB.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] from-scratch held-out (group B) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

for TASK in SimpleSyllogismTask OppositeSyllogismTask CountryCapitalTask MCQAnchoredBiasTask; do
    echo ""
    echo "######## FROM-SCRATCH -> ${TASK} ########"
    EXTRA=""
    [ "${TASK}" = "MCQAnchoredBiasTask" ] && EXTRA="--prefilter-metric kl"
    python -m mechrl.train.train_agent \
        --policy batch --tasks "${TASK}" \
        --num-examples 20 --batch-sizes 1 3 10 30 \
        --faith-margin 0.0 --threshold-penalty 5 --seed 1 \
        --step-budget 150 --num-steps 512 \
        --total-iterations 200 --log-every 1 --save-every 50 \
        --device cuda --out runs ${EXTRA} \
        || echo "[FAILED] from-scratch ${TASK}"
done

echo "[$(hostname)] from-scratch held-out (group B) done $(date)"
