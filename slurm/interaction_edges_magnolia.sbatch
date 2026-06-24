#!/bin/bash
# Probe 2 (§5.2 "policy keeps edges attribution undervalues"): per behaviour, rank the K=3000
# candidates by |EAP-IG score|, run the frozen donor to get its |C|-edge circuit, and measure
#   (1) agent faith vs top-|C|-by-score faith (selection quality),
#   (2) # of kept edges ranked BELOW the top-|C| cutoff vs the random expectation,
#   (3) faith drop when those discordant low-rank edges are ablated (are they load-bearing).
# Output -> runs/interaction_edges.json (plotted locally).
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/interaction_edges_magnolia.sbatch
# Watch :  tail -f mechrl_p2.<jobid>.out
#
#SBATCH --job-name=mechrl_p2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_p2.%j.out
#SBATCH --error=mechrl_p2.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] probe2 interaction-edges start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

DONOR=$(ls -dt runs/train_seed1_*/ 2>/dev/null | head -1)
[ -z "${DONOR}" ] && { echo "ERROR: no runs/train_seed1_* donor"; exit 1; }
DONOR=${DONOR%/}
echo "donor: ${DONOR}"

python -m scripts.interaction_edges \
    --run "${DONOR}" --device cuda --num-rollouts 16 \
    --tasks "IOITask,GreaterThanOriginal,DocstringGPT2Task,GenderedPronounTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask" \
    --out runs/interaction_edges.json \
    || echo "[FAILED] interaction_edges"

echo "[$(hostname)] probe2 done $(date)"
