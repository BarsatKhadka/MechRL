# SLURM scripts (Magnolia / Ole Miss MCSR, L40S)

Mirrors the weightBench convention: `gpuq` partition, one L40S, `module load` +
repo venv, env-var driven so the same script covers single- and multi-task runs.

## One-time setup on the cluster

```bash
# on the login node (has internet):
cd ~/MechRL
module load python/2025.12-2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# install the two vendored libs the same way as locally (editable / path):
pip install -e paperCodes/Automatic-Circuit-Discovery
# eap-ig is used via path import (see mechrl/env) — no install needed

# pre-fetch GPT-2 into the HF cache so compute nodes (offline) can load it:
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')"
```

## Running

```bash
# multi-task: all 13 verified tasks
sbatch slurm/train.sbatch

# single-task signal run — the FIRST "does it learn" test (train on IOI only)
sbatch --export=ALL,TASKS=IOITask slurm/train.sbatch

# one family
sbatch --export=ALL,TASKS=greaterthan slurm/train.sbatch

# override scale / seed
sbatch --export=ALL,TASKS=all,ITERS=800,STEPS=256,BUDGET=400,SEED=1 slurm/train.sbatch
```

`TASKS` accepts: `all`, `ioi`, `greaterthan`, `docstring`, or a single class
name (`IOITask`, `GreaterThanOriginal`, `DocstringGPT2Numpy5Task`, ...).

## Outputs

`runs/<run_name>/` contains:
- `config.json` — the exact args of the run
- `metrics.jsonl` — one line per PPO iteration (return, faith, kept, losses)
- `policy_iter*.pt`, `policy_final.pt` — checkpoints

Logs land in `logs/mechrl_train_<jobid>.out`.

## Suggested first runs

1. **Single-task signal**: `TASKS=IOITask`, a few hundred iterations. Watch
   `metrics.jsonl`: does `return` climb and `kept` drop while `faith` stays high?
   That is the make-or-break "is there signal" test.
2. **Multi-task**: `TASKS=all` once single-task shows learning.
