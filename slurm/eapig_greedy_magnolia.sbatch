#!/bin/bash
# EAP-IG greedy baseline (Hanna et al. 2024, App. E) for Table 4 (§5.3), on OUR graph + KL engine
# so size/faith are directly comparable to ACDC and the policy. Greedy selection is forward-pass
# free; cost is one-time EAP-IG attribution + one eval per circuit size -- tens of passes, vs
# ACDC's ~10^4. Per task -> runs/eapig_<task>.json with the (edges, faith) curve.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/eapig_greedy_magnolia.sbatch
# Watch :  tail -f mechrl_eapig.<jobid>.out
#
#SBATCH --job-name=mechrl_eapig
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_eapig.%j.out
#SBATCH --error=mechrl_eapig.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] EAP-IG greedy start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

# Table-4 behaviours (all logit-diff attribution). Add MCQAnchoredBiasTask with --prefilter-metric
# kl if you want it. Sizes span the range where faith crosses ~0.9.
for TASK in IOITask GreaterThanOriginal DocstringGPT2Task GenderedPronounTask AcronymTask \
            SubjectVerbAgreementTask SimpleSyllogismTask OppositeSyllogismTask \
            CountryCapitalTask MCQAnchoredBiasTask; do
    echo ""
    echo "======== EAP-IG greedy -> ${TASK} ========"
    EXTRA=""
    [ "${TASK}" = "MCQAnchoredBiasTask" ] && EXTRA="--prefilter-metric kl"
    python -m scripts.run_eapig_greedy --task "${TASK}" --device cuda \
        --num-examples 20 --ig-steps 5 \
        --sizes 50 100 200 400 700 1000 1500 2000 3000 \
        --out "runs/eapig_${TASK}.json" ${EXTRA} || echo "[FAILED] EAP-IG greedy ${TASK}"
done

echo "[$(hostname)] EAP-IG greedy done $(date)"
