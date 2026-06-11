#!/bin/bash
# 2-TASK + HEADROOM, 3 SEEDS, 600 iters, on Magnolia (L40S, gpuq).
# Mirror of slurm/train_2task_headroom_cic.sbatch but with Magnolia conventions
# (gpuq, --gres=gpu:1, venv). The headroom obs fix (globals 5->6) is in the code,
# so this just runs it. See the CIC script header for the full "did it work?" notes.
#
# WATCH at convergence (~iter 250-600): lockstep should be BROKEN -- GT settles small
# (~500-700, hits its bar ✓) while IOI keeps more (~800-1100). Then watch whether IOI
# faith climbs past ~0.89 toward its 0.951 bar (seed 0 on CIC touched 0.94).
#
# CHECK FIRST that gpuq allows a 24h job:  scontrol show partition gpuq | grep -i maxtime
#   If MaxTime < 24h, lower --time below to fit (600 iters ~14h @ ~1.4 min/iter; if the
#   cap is short, drop --total-iterations or chain with --dependency=afterany).
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/train_2task_headroom_magnolia.sbatch
# Watch :  tail -f mechrl_2t_hr_mag.<arrayjobid>_0.out
#
#SBATCH --job-name=mechrl_2t_hr
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --array=0-2                # 3 seeds: array index -> --seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00            # 600 iters ~14h; lower if gpuq MaxTime is shorter
#SBATCH --output=mechrl_2t_hr_mag.%A_%a.out
#SBATCH --error=mechrl_2t_hr_mag.%A_%a.err

set -euo pipefail
echo "[$(hostname)] start $(date)  seed=${SLURM_ARRAY_TASK_ID}"
cd ~/MechRL
source venv/bin/activate
python -c "import torch; print('cuda?', torch.cuda.is_available())"

python -m mechrl.train.train_agent --policy batch \
    --tasks IOITask,GreaterThanOriginal --num-examples 20 \
    --batch-sizes 1 3 10 30 --faith-margin 0.0 --threshold-penalty 5 \
    --pcgrad --seed "${SLURM_ARRAY_TASK_ID}" \
    --step-budget 150 --num-steps 512 --total-iterations 600 \
    --device cuda --out runs

echo "[$(hostname)] done $(date)  seed=${SLURM_ARRAY_TASK_ID}"
