#!/bin/bash
# EAP-IG greedy with KL attribution (the fair EAP-IG for Table 4: it attributes toward the SAME
# KL metric everything is scored on, unlike the logit-diff run). EAP-IG-KL is the paper's KL
# variant; this is its best shot at small KL-faithful circuits. Per task -> runs/eapig_kl_<task>.json.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/eapig_greedy_kl_magnolia.sbatch
# Watch :  tail -f mechrl_eapigkl.<jobid>.out
#
#SBATCH --job-name=mechrl_eapigkl
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_eapigkl.%j.out
#SBATCH --error=mechrl_eapigkl.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] EAP-IG greedy (KL attribution) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

for TASK in IOITask GreaterThanOriginal DocstringGPT2Task GenderedPronounTask AcronymTask \
            SubjectVerbAgreementTask SimpleSyllogismTask OppositeSyllogismTask \
            CountryCapitalTask MCQAnchoredBiasTask; do
    echo ""
    echo "======== EAP-IG-KL greedy -> ${TASK} ========"
    python -m scripts.run_eapig_greedy --task "${TASK}" --device cuda \
        --num-examples 20 --ig-steps 5 --prefilter-metric kl \
        --sizes 50 100 200 400 700 1000 1500 2000 3000 \
        --out "runs/eapig_kl_${TASK}.json" || echo "[FAILED] EAP-IG-KL ${TASK}"
done

echo "[$(hostname)] EAP-IG greedy (KL) done $(date)"
