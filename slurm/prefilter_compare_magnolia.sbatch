#!/bin/bash
# PREFILTER COMPARISON on Magnolia (gpuq / L40S) -- the decisive copy-suppression diagnostic.
#
# The agent is judged on KL faithfulness, but the candidate set (top-K edges) is chosen by
# the prefilter. Default = logit-diff/EAP-IG attribution on the TASK metric. For copy-
# suppression the task metric is a weak single-token (food) logit, so logit-diff attribution
# may rank the WRONG edges and leave no SPARSE KL-faithful subset -> the agent cuts to ~600
# edges and lands at faith 0 (the warm-start stall) for a prefilter reason, NOT a transfer one.
#
# verify_prefilter_kl prints candidate-set KL-faith for BOTH prefilters (logitdiff vs kl-attr)
# across SMALL K (the agent's real operating regime ~300-600 edges) up to 3000.
#
# READ:
#   - copy-suppression kl-attr@K~600 HIGH but logitdiff@K~600 ~0  -> PREFILTER is the culprit;
#     re-run copy-suppression zero-shot + warm-start with --prefilter-metric kl.
#   - both LOW at small K, only climb near K=3000                 -> genuinely diffuse; no
#     prefilter rescues it, "resists transfer" is simply correct.
#   - SVA / acronyms / syllogisms: sanity that logit-diff is fine for them (task metric ~ KL).
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/prefilter_compare_magnolia.sbatch
# Watch :  tail -f mechrl_pf.<jobid>.out
#
#SBATCH --job-name=mechrl_pf
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=mechrl_pf.%j.out
#SBATCH --error=mechrl_pf.%j.err

echo "[$(hostname)] prefilter compare start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.verify_prefilter_kl \
    --tasks CopySuppressionTask,SubjectVerbAgreementTask,AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask \
    --ks 300,600,1000,1500,3000 --num-examples 50 --device cuda

echo "[$(hostname)] prefilter compare done $(date)"
