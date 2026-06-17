#!/bin/bash
# CEILING PROBE for the NEW held-out candidates on Magnolia (gpuq / L40S).
# Gate before any transfer run: is each task a real, SPARSE GPT-2-small circuit task?
#   - Acronyms          (Garcia-Carrasco et al. 2024, AISTATS)  -- 8 letter-mover heads
#   - SimpleSyllogism   (Saraipour & Zhang 2025)                -- truth heads
#   - OppositeSyllogism (Saraipour & Zhang 2025)                -- neg-truth heads + MLPs
#   - MCQAnchoredBias   (Li & Gao 2025)  ON PROBATION -- positional bias; the content-swap
#                       counterfactual may give a WEAK KL_cut (expected). The probe decides.
#   - CountryCapital    factual recall   ON PROBATION -- likely DIFFUSE (MLP-heavy). The probe decides.
#
# Reads (per task): fullKL (~0 sanity), KL_cut (counterfactual strength), and top-K
# faith at K in {3000, 8000, 16000, ALL}.
#   VIABLE  : KL_cut healthy (>~1.5) AND faith@3000 clears ~0.85 (the training-task gate).
#   WEAK    : KL_cut < 1.5  -> weak counterfactual; redesign the corrupted prompt or drop.
#   DIFFUSE : KL_cut healthy but faith@3000 low and only ~1.0 at K=ALL -> no sparse subset; drop.
#
# Submit:  cd ~/MechRL && git pull && sbatch slurm/ceiling_probe_heldout_magnolia.sbatch
# Watch :  tail -f mechrl_probe.<jobid>.out   (the "Part 1: diagnosis" table is the verdict)
#
#SBATCH --job-name=mechrl_probe
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=mechrl_probe.%j.out
#SBATCH --error=mechrl_probe.%j.err

echo "[$(hostname)] ceiling probe (new held-out) start $(date)"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.probe_task_ceiling \
    --tasks AcronymTask,SimpleSyllogismTask,OppositeSyllogismTask,MCQAnchoredBiasTask,CountryCapitalTask \
    --num-examples 50 --device cuda

echo "[$(hostname)] ceiling probe done $(date)"
