#!/bin/bash
# 2-TASK + HEADROOM feature, 3 SEEDS, on the Cyber Innovation Cluster (CIC).
# Tests the observation fix: globals now carries headroom = current_faith - tau (the
# signed distance to THIS task's bar). Before, the policy was blind to its own bar, so
# it applied ONE global cut-depth to both tasks -> lockstep circuit sizes + IOI over-cut
# below its bar (Aquaman converged: IOI ~0.82 @ ~800, GT ~0.96 @ ~800, both locked).
#
# 3 SEEDS because the 2-task result is seed-sensitive (one CIC seed diverged while
# Aquaman converged). This is a SLURM job array: seed = array index (0,1,2), each on
# its own node. Fresh from scratch -- the obs shape changed (N_GLOBALS 5->6), so old
# checkpoints can't warm-start this.
#
# DID IT WORK? Compare the per-task "└" line at convergence (~iter 250-300) vs Aquaman:
#   1. LOCKSTEP BREAKS: GT settles SMALLER (~540-700) than IOI (~1000-1200), not both ~800.
#   2. IOI FAITH RISES toward its 0.951 bar (stops AT the bar instead of cutting through).
# If both happen across seeds -> headroom worked. If sizes stay locked -> bottleneck is
# elsewhere (policy can't condition / critic), and we escalate.
# Also still watch: expl_var climbing to ~0.6-0.9 (healthy critic), gn (~1-3 is normal,
# the clip binds -- ignore it), gcos dipping negative (conflict present).
#
# Submit from the LOGIN node:  cd ~/MechRL && git pull && sbatch slurm/train_2task_headroom_cic.sbatch
# Monitor:  squeue -u $USER ; tail -f mechrl_2t_hr.<arrayjobid>_0.out   (seed 0)
#
#SBATCH --job-name=mechrl_2t_hr
#SBATCH --partition=main
#SBATCH --array=0-2                # 3 seeds: array index -> --seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --exclude=c1,c2            # leave desk machines free; prefer c3/c4
#SBATCH --output=mechrl_2t_hr.%A_%a.out
#SBATCH --error=mechrl_2t_hr.%A_%a.err
# NOTE: this cluster does NOT use --gres for GPUs (one GPU/node, auto-allocated).

set -euo pipefail
echo "[$(hostname)] start $(date)  seed=${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true

module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

python -c "import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Headroom obs fix is in the code (globals 5->6, automatic). Same config as the
# Aquaman/CIC PCGrad runs; only the observation changed + a seed sweep.
# If gpt2 tries to download and hangs (no internet on compute node), prefix python with:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad --seed "${SLURM_ARRAY_TASK_ID}" \
    --step-budget 150 --num-steps 512 --total-iterations 300 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)  seed=${SLURM_ARRAY_TASK_ID}"
