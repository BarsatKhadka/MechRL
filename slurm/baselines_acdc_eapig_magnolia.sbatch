#!/bin/bash
# BASELINES for the comparison / amortisation sections (Magnolia gpuq / L40S).
# Everything per-task-method needs for the Pareto + break-even-cost story, in one overnight job:
#
#   PHASE 1  EAP-IG / prefilter Pareto  (cheaper)
#       verify_prefilter_kl over ALL tasks (12 training + held-out): faith vs K under both
#       logit-diff (EAP-IG, the attribution-only baseline) and KL attribution. Doubles as the
#       per-task ceiling certificate for the method section.
#
#   PHASE 2  ACDC baseline  (expensive: ~full-graph greedy, thousands of forward passes/threshold)
#       run_acdc_greedy per task -> (edges, faith, FORWARD-PASS COST) at 3 thresholds. ACDC runs
#       on the full graph from scratch (no prefilter) -- the honest per-task cost we amortise
#       against. Training reps (IOI / GreaterThan / Docstring, 1 per family) + all held-out.
#       Each task's circuits saved to runs/acdc_<task>.json (+ acdc_circuit_tau*.json).
#
# All GPU (every step is GPT-2 forward passes; no CPU-only chunk). --time=0. Continues past
# any single-task failure so one slow/odd task does not lose the rest.
#
# Submit (queues behind the 5.1 jobs; fine -- baselines are for later sections):
#   cd ~/MechRL && git pull && sbatch slurm/baselines_acdc_eapig_magnolia.sbatch
# Watch: tail -f mechrl_base.<jobid>.out
#
#SBATCH --job-name=mechrl_base
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
#SBATCH --output=mechrl_base.%j.out
#SBATCH --error=mechrl_base.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] baselines (ACDC + EAP-IG) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Held-out + training-representative task list for ACDC (one ACDC run each).
ACDC_TASKS="IOITask GreaterThanOriginal DocstringGPT2Task \
CopySuppressionTask GenderedPronounTask SubjectVerbAgreementTask \
AcronymTask SimpleSyllogismTask OppositeSyllogismTask \
MCQAnchoredBiasTask CountryCapitalTask"

echo ""
echo "################ PHASE 1: EAP-IG / prefilter Pareto (faith vs K, all tasks) ################"
python -m scripts.verify_prefilter_kl \
    --tasks "train,CopySuppressionTask,GenderedPronounTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,MCQAnchoredBiasTask,CountryCapitalTask" \
    --ks 100,300,600,1000,1500,3000 --num-examples 50 --device cuda \
    || echo "[FAILED] EAP-IG prefilter Pareto"

echo ""
echo "################ PHASE 2: ACDC baseline (size / faith / forward-pass cost) ################"
for TASK in ${ACDC_TASKS}; do
    echo ""
    echo "======== ACDC -> ${TASK} ========"
    python -m scripts.run_acdc_greedy --task "${TASK}" --device cuda \
        --num-examples 20 --thresholds 0.001 0.003 0.01 \
        --out "runs/acdc_${TASK}.json" || echo "[FAILED] ACDC ${TASK}"
done

echo ""
echo "[$(hostname)] baselines done $(date)"
