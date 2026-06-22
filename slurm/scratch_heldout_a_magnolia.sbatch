#!/bin/bash
# FROM-SCRATCH held-out baseline, group A (3 tasks) -- Magnolia (gpuq / L40S).
# Backs the §5.1 claim "warm-start reaches the bar several times faster than from scratch".
# The warm-start curves exist; the matched FROM-SCRATCH curves on held-out behaviours do NOT
# (every held-out run on disk was --init-from). This trains a FRESH policy per held-out task
# (NO --init-from), same bar (--faith-margin 0.0 -> tau = ceiling) and same hyperparameters as
# the warm-start runs, so the two learning curves are directly comparable.
#
# Per task -> runs/<Task>_seed1_<ts>/metrics.jsonl : per-iter faith + per-task tau +
# first_success_iter (the iter it first clears the bar). That JSONL is the figure's data;
# first_success_iter is the denominator of the speed-up ratio.
#
# 200 iters each so scratch gets a FAIR shot at the bar (a starved scratch run is not a fair
# baseline). No --time limit. set +e: one task failing does not lose the rest.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/scratch_heldout_a_magnolia.sbatch
# Watch :  tail -f mechrl_scratchA.<jobid>.out
#   -> per-task "[bar] <Task> ceiling=.. -> tau=.." then watch `faith` climb; iters-to-bar =
#      first iter whose faith >= tau (also recorded as first_success_iter in metrics.jsonl).
#
#SBATCH --job-name=mechrl_scratchA
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_scratchA.%j.out
#SBATCH --error=mechrl_scratchA.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] from-scratch held-out (group A) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

for TASK in GenderedPronounTask SubjectVerbAgreementTask AcronymTask; do
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

echo "[$(hostname)] from-scratch held-out (group A) done $(date)"
