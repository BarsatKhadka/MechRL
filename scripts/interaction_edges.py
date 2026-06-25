"""Probe 2: does the agent keep edges the EAP-IG ranking undervalues, and do they matter?

The candidate set is the top-K=3000 edges by |EAP-IG score|. The agent returns a circuit of
|C| edges -- a subset of the K. A size-matched naive baseline keeps the top-|C| by score. We
ask three things per behaviour:

  (1) selection: is the agent's circuit more faithful than the top-|C| by score?
  (2) rank mix: how many of the agent's edges are ranked BELOW the top-|C| cutoff ("discordant",
      i.e. rescued from the lower band) -- and how does that compare to a random |C|-subset's
      expectation, |C|(K-|C|)/K? (Keeping 1k of 3k naturally puts some below rank 1k; the
      question is whether it is fewer than chance, i.e. the agent DOES use the ranking, while
      still rescuing some.)
  (3) load-bearing: does ablating those discordant low-rank edges drop faithfulness? If yes, the
      agent is correcting the score's known ranking error (EAP-IG Appendix F), not keeping noise.

Run on the box with ckpt + prefilter cache (loads GPT-2 -> compute node, NOT login):
    python -m scripts.interaction_edges --run runs/train_seed1_<ts> --device cuda \
        --tasks IOITask,GreaterThanOriginal,... --num-rollouts 16 \
        --out runs/interaction_edges.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent.batch_policy import BatchCutPolicy
from scripts.battery_test import TASKS, latest_ckpt, move_obs


def agent_circuit(policy, env, engine, device, num_rollouts):
    """best-of-K frozen rollouts; return the most faithful final mask (over full edge list)."""
    def rollout(greedy):
        obs = env.reset(bundle_idx=0)
        with torch.no_grad():
            while not env.done:
                a, _, _, _ = policy.act(move_obs(obs, device), greedy=greedy)
                obs, _, _, _ = env.step(a)
        m = env.mask.clone().cpu()
        return m, float(engine.faithfulness(m))
    best = rollout(greedy=True)
    for _ in range(num_rollouts):
        m, f = rollout(greedy=False)
        if f > best[1]:
            best = (m, f)
    return best


def analyze(policy, cfg, task_name, device, num_rollouts):
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[task_name](**tkwargs)
    pm = "kl" if task_name == "MCQAnchoredBiasTask" else None
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000), prefilter_metric=pm)
    engine = bundle.engine
    env = CircuitEnv([bundle], step_budget=cfg.get("step_budget", 150),
                     faith_threshold=cfg.get("faith_threshold", 0.8),
                     threshold_penalty=cfg.get("threshold_penalty", 5.0),
                     invalid_penalty=cfg.get("invalid_penalty", -0.01),
                     seed=cfg.get("seed", 0), faith_margin=cfg.get("faith_margin"))

    cand = bundle.cand_edge_idx.cpu().numpy()                       # full-edge idx of K candidates
    K = len(cand)
    absc = np.array([abs(float(engine.edge_list[int(i)].score)) for i in cand])
    order = np.argsort(-absc)                                       # cand positions, |score| desc
    rank_of_cand = np.empty(K, dtype=int)
    rank_of_cand[order] = np.arange(K)                             # rank_of_cand[j] = rank (0=top)
    full_to_pos = {int(idx): j for j, idx in enumerate(cand)}

    mask, faith_agent = agent_circuit(policy, env, engine, device, num_rollouts)
    kept_full = mask.nonzero(as_tuple=True)[0].cpu().numpy()
    kept_pos = np.array([full_to_pos[int(i)] for i in kept_full if int(i) in full_to_pos])
    C = int(len(kept_pos))
    kept_ranks = rank_of_cand[kept_pos]

    # discordant = kept but ranked >= C (a top-|C| cut would have dropped them)
    discordant_pos = kept_pos[kept_ranks >= C]
    n_disc = int(len(discordant_pos))

    # (1) top-|C| by score, as a mask of exactly C edges
    topC_full = cand[order[:C]]
    topC_mask = torch.zeros_like(mask)
    topC_mask[torch.as_tensor(topC_full, dtype=torch.long)] = True
    faith_topC = float(engine.faithfulness(topC_mask))

    # (3) ablate the discordant low-rank edges from the agent's circuit -> faith drop
    mask_concord = mask.clone()
    if n_disc:
        mask_concord[torch.as_tensor(cand[discordant_pos], dtype=torch.long)] = False
    faith_concord = float(engine.faithfulness(mask_concord))

    hist = np.histogram(kept_ranks, bins=10, range=(0, K))[0].tolist()  # kept counts per rank-decile

    return {
        "task": task_name, "K": K, "C": C,
        "faith_agent": faith_agent, "faith_topC": faith_topC,
        "faith_gap_vs_topC": faith_agent - faith_topC,
        "n_discordant": n_disc, "frac_discordant": (n_disc / C) if C else 0.0,
        "expected_discordant_random": C * (K - C) / K,
        "faith_after_ablating_discordant": faith_concord,
        "faith_drop_from_discordant": faith_agent - faith_concord,
        "kept_rank_hist_deciles": hist,
    }


def find_warm_run(runs_dir, task):
    """Newest runs/<task>_seed1_* whose config has a non-null init_from (a warm-start run)."""
    best, best_ts = None, -1
    for d in Path(runs_dir).glob(f"{task}_seed1_*"):
        cfgp = d / "config.json"
        if cfgp.exists() and json.loads(cfgp.read_text()).get("init_from"):
            try:
                ts = int(d.name.split("_")[-1])
            except ValueError:
                ts = 0
            if ts > best_ts:
                best, best_ts = d, ts
    return best


def load_policy(run_dir, ckpt_arg, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt = Path(ckpt_arg) if ckpt_arg else latest_ckpt(run_dir)
    policy = BatchCutPolicy(hidden=cfg.get("hidden", 128),
                            batch_sizes=tuple(cfg.get("batch_sizes", [1, 3, 10, 30]))).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    return policy, cfg, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="donor run dir (zero-shot); not needed with --warm")
    p.add_argument("--warm", action="store_true",
                   help="use each task's OWN warm-start run (newest <task>_seed1_* with init_from)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-rollouts", type=int, default=16)
    p.add_argument("--tasks", required=True, help="comma-separated task class names")
    p.add_argument("--out", default="runs/interaction_edges.json")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable; cpu", flush=True); device = "cpu"
    if not args.warm and not args.run:
        raise SystemExit("provide --run (zero-shot) or --warm (per-task warm-start)")

    donor = None
    if not args.warm:
        donor = load_policy(Path(args.run), args.ckpt, device)
        print(f"[probe2] donor={Path(args.run).name} ckpt={donor[2].name} device={device}", flush=True)

    rows = []
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if t not in TASKS:
            print(f"[skip] unknown task {t}", flush=True); continue
        if args.warm:
            wdir = find_warm_run(args.runs_dir, t)
            if wdir is None:
                print(f"[skip] no warm-start run for {t}", flush=True); continue
            policy, cfg, _ = load_policy(wdir, None, device)
            print(f"\n=== {t}  (warm: {wdir.name}) ===", flush=True)
        else:
            policy, cfg = donor[0], donor[1]
            print(f"\n=== {t} ===", flush=True)
        r = analyze(policy, cfg, t, device, args.num_rollouts)
        rows.append(r)
        print(f"  |C|={r['C']}  agent f={r['faith_agent']:.3f} vs top-|C| f={r['faith_topC']:.3f} "
              f"(gap {r['faith_gap_vs_topC']:+.3f})", flush=True)
        print(f"  discordant {r['n_discordant']}/{r['C']} = {r['frac_discordant']:.0%}  "
              f"(random {r['expected_discordant_random']/r['C']:.0%})  drop "
              f"{r['faith_drop_from_discordant']:+.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mode = "warm" if args.warm else "zero_shot"
    Path(args.out).write_text(json.dumps({"mode": mode, "rows": rows}, indent=2, default=float))
    print(f"\n[probe2] saved -> {args.out}  (mode={mode})", flush=True)


if __name__ == "__main__":
    main()
