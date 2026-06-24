#!/bin/bash
# Candidate-set faithfulness vs K, per behaviour -- justifies K=3000 (§5.2 / method).
# verify_prefilter_kl computes the EAP-IG scores once per behaviour, then reads the KL
# faithfulness of the top-K candidate set for each K in the sweep (cheap: one forward pass
# per K). Output JSON -> plotted by scripts.plot_prefilter_ceiling into figures/fig_prefilter_k.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/prefilter_ceiling_magnolia.sbatch
# Watch :  tail -f mechrl_pfk.<jobid>.out   ;  result -> runs/prefilter_ceiling.json
#
#SBATCH --job-name=mechrl_pfk
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_pfk.%j.out
#SBATCH --error=mechrl_pfk.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] prefilter faith-vs-K start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

python -m scripts.verify_prefilter_kl \
    --tasks "IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask" \
    --ks "100,300,600,1000,1500,2000,3000,5000,8000" \
    --num-examples 20 --ig-steps 5 --device cuda \
    --out runs/prefilter_ceiling.json \
    || echo "[FAILED] prefilter faith-vs-K"

echo "[$(hostname)] prefilter faith-vs-K done $(date)"
