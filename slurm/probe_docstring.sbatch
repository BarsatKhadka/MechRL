#!/bin/bash
# Verify the new candidate tasks all hit a healthy KL ceiling before training.
# Part 1 = IOI (control ~0.95) + google_5 (weak contrast) + the TWO NEW tasks
#          (Subject-Verb-Agreement, Gendered-Pronoun -- from the literature dive).
# Part 3 = docstring BASE + class_sphinx + 3 new strong-cue variants.
# No induction sweep / dead experiments (settled).
#
# Pull first:  cd ~/MechRL && git pull
# Submit:  sbatch slurm/probe_docstring.sbatch
#
#SBATCH --job-name=probe_doc
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node049
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=probe_docstring.%j.out
#SBATCH --error=probe_docstring.%j.err

echo "[$(hostname)] start $(date)"
cd ~/MechRL
source venv/bin/activate

python -m scripts.probe_task_ceiling \
    --tasks IOITask,DocstringGPT2Google5Task,SubjectVerbAgreementTask,GenderedPronounTask \
    --force --device cuda

echo "[$(hostname)] done $(date)"
