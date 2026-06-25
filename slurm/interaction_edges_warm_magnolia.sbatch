#!/bin/bash
# Probe 2 / Fig 4b, WARM-START variant: same rank analysis, but each behaviour uses its OWN
# warm-start checkpoint (newest runs/<Task>_seed1_* with init_from) instead of the frozen donor.
# Lets Fig 4b show, per held-out behaviour, a dotted line from the zero-shot gain to the warm-start
# gain -- e.g. subject-verb climbing from the red centre out into the green. Only the seven held-out
# behaviours have warm-start runs. MCQ's warm run already used the KL prefilter.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/interaction_edges_warm_magnolia.sbatch
# Watch :  tail -f mechrl_p2warm.<jobid>.out  ;  result -> runs/interaction_edges_warm.json
#
#SBATCH --job-name=mechrl_p2warm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_p2warm.%j.out
#SBATCH --error=mechrl_p2warm.%j.err

set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "[$(hostname)] probe2 warm-start start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

python -m scripts.interaction_edges --warm --device cuda --num-rollouts 16 \
    --tasks "GenderedPronounTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,CountryCapitalTask,MCQAnchoredBiasTask" \
    --out runs/interaction_edges_warm.json \
    || echo "[FAILED] interaction_edges warm"

echo "[$(hostname)] probe2 warm-start done $(date)"
