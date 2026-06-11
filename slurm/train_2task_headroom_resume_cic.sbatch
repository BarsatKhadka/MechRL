#!/bin/bash
# STAGE 2: resume the headroom 3-seed run (300 -> 600 iters) on CIC.
# Submit this with a DEPENDENCY on the stage-1 array so it auto-fires when stage 1
# finishes -- no need to babysit / wake up at night.
#
#   # find the stage-1 array job id first:  squeue -u $USER   (e.g. 23547)
#   sbatch --dependency=aftercorr:23547 slurm/train_2task_headroom_resume_cic.sbatch
#
#   aftercorr = resume seed i starts after stage-1 seed i finishes SUCCESSFULLY
#   (per-seed pipelining; a failed/diverged seed is NOT resumed). If your SLURM
#   rejects aftercorr, fall back to:  --dependency=afterok:23547  (waits for ALL
#   stage-1 seeds, then runs all resumes).
#
# Each seed auto-finds its own stage-1 run dir (newest runs/IOITask,...seed<i>_*),
# loads policy_iter300.pt (actor+critic weights) and continues from iter 300.
# NOTE: anneal_lr restarts the LR at ~50% of base at iter 301 (the schedule now spans
# 600), so expect a visible bump in dynamics there -- that's intended (you need nonzero
# LR to keep learning), not instability.
#
#SBATCH --job-name=mechrl_2t_hr_rs
#SBATCH --partition=main
#SBATCH --array=0-2                # must match stage-1 indices for aftercorr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00            # 300 more iters of 2-task PCGrad
#SBATCH --exclude=c1,c2
#SBATCH --output=mechrl_2t_hr_rs.%A_%a.out
#SBATCH --error=mechrl_2t_hr_rs.%A_%a.err
# NOTE: this cluster does NOT use --gres for GPUs (one GPU/node, auto-allocated).

set -euo pipefail
SEED=${SLURM_ARRAY_TASK_ID}
echo "[$(hostname)] resume start $(date)  seed=${SEED}"
nvidia-smi || true

module load miniconda/miniconda3 2>/dev/null || true
conda activate ~/mechrl-env
cd ~/MechRL

# Auto-find this seed's stage-1 run dir (newest matching, by mtime).
RUN_DIR=$(ls -dt runs/IOITask,GreaterThanOriginal_seed${SEED}_*/ 2>/dev/null | head -1 || true)
if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: no stage-1 run dir found for seed ${SEED} (runs/IOITask,GreaterThanOriginal_seed${SEED}_*)"
    exit 1
fi
RUN_DIR=${RUN_DIR%/}
echo "resuming seed ${SEED} from: ${RUN_DIR}"
ls "${RUN_DIR}"/policy_iter*.pt 2>/dev/null || echo "(no policy_iter*.pt -- will fall back to policy_final.pt, start_iter=0)"

# Same args as stage 1 (must match so policy/env shapes line up); only --total-iterations
# and --resume-from differ. If gpt2 tries to download and hangs, prefix python with:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad --seed ${SEED} \
    --step-budget 150 --num-steps 512 --total-iterations 600 \
    --resume-from "${RUN_DIR}" \
    --device cuda --out runs

echo "[$(hostname)] resume done $(date)  seed=${SEED}"
