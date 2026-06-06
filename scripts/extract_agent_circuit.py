"""Roll out a trained policy for ONE greedy episode and dump the circuit it finds.

The agent's "circuit" is just the edge-mask at the end of an episode. We load a
checkpoint, run the policy deterministically (argmax, not sampling) on IOI, and
save the surviving edges (by name) + faith/KL to a small JSON. That JSON feeds
the validity battery (evaluate_circuit.py) -- this is Gate-1 prep.

Reads the run's config.json so the env is built exactly as in training
(step_budget, faith_threshold, k, hidden, ...).

Run (on Aquaman, where the checkpoint + prefilter cache live):
    python -m scripts.extract_agent_circuit \
        --run runs/IOITask_seed0_1780540820 --device cuda \
        --out runs/agent_circuit_ioi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechrl.tasks import IOITask
from mechrl.env import CircuitEnv, TaskBundle
from mechrl.agent import CircuitPolicy
from mechrl.agent.batch_policy import BatchCutPolicy

TASKS = {"IOITask": IOITask}


def latest_ckpt(run_dir: Path) -> Path:
    import re
    ckpts = sorted(run_dir.glob("policy_iter*.pt"),
                   key=lambda p: int(re.search(r"iter(\d+)", p.name).group(1)))
    if not ckpts:
        raise FileNotFoundError(f"no policy_iter*.pt in {run_dir}")
    return ckpts[-1]


def move_obs(obs, device):
    return {k: v.to(device) for k, v in obs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir containing config.json + policy_iter*.pt")
    p.add_argument("--ckpt", default=None, help="specific checkpoint (default: latest in run dir)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--num-rollouts", type=int, default=8,
                   help="number of SAMPLED rollouts; keep the most faithful (best-of-K)")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available; using cpu", flush=True)
        device = "cpu"

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "config.json").read_text())
    task_name = cfg.get("task_classes", ["IOITask"])[0]
    if task_name not in TASKS:
        raise ValueError(f"this extractor only supports {list(TASKS)} for now, got {task_name}")
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(run_dir)
    print(f"[extract] run={run_dir.name} ckpt={ckpt.name} task={task_name} device={device}", flush=True)

    # Build the task + bundle EXACTLY as training did (prefilter cache reused).
    tkwargs = {"device": device}
    if cfg.get("num_examples") is not None:
        tkwargs["num_examples"] = cfg["num_examples"]
    task = TASKS[task_name](**tkwargs)
    bundle = TaskBundle.build(task, k=cfg.get("k", 3000))
    engine = bundle.engine

    env = CircuitEnv(
        [bundle],
        step_budget=cfg.get("step_budget", 400),
        faith_threshold=cfg.get("faith_threshold", 0.8),
        threshold_penalty=cfg.get("threshold_penalty", 3.0),
        invalid_penalty=cfg.get("invalid_penalty", -0.01),
        seed=cfg.get("seed", 0),
    )
    is_batch = cfg.get("policy", "flat") == "batch"
    if is_batch:
        policy = BatchCutPolicy(hidden=cfg.get("hidden", 128),
                                batch_sizes=tuple(cfg.get("batch_sizes", [1, 3, 10, 30, 100]))).to(device)
    else:
        policy = CircuitPolicy(hidden=cfg.get("hidden", 128)).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    # The policy is STOCHASTIC and was trained/measured by SAMPLING, so greedy
    # (argmax) is off-distribution and can be degenerate (it was: faith ~0,
    # circuit disconnected from the output). Extract by SAMPLING, best-of-K
    # (matches training + the preprint's best-of-K planning). One greedy rollout
    # is run first, only to show how degenerate the argmax path is.
    def rollout(greedy: bool):
        obs = env.reset(bundle_idx=0)
        steps, info = 0, {}
        with torch.no_grad():
            while not env.done:
                if is_batch:
                    action, _, _, _ = policy.act(move_obs(obs, device), greedy=greedy)
                elif greedy:
                    logits, _ = policy(move_obs(obs, device))
                    action = int(torch.argmax(logits).item())
                else:
                    action, _, _, _ = policy.act(move_obs(obs, device))
                obs, _, done, info = env.step(action)
                steps += 1
        m = env.mask.clone().cpu()
        return m, engine.faithfulness(m), int(m.sum().item()), steps, info

    gm, gf, gk, gs, _ = rollout(greedy=True)
    print(f"[extract] GREEDY:  kept {gk:5d}  faith {gf:+.4f}  (argmax = off-distribution)", flush=True)

    best = None
    for k in range(args.num_rollouts):
        m, f, kp, st, info = rollout(greedy=False)
        print(f"[extract] sample {k}: kept {kp:5d}  faith {f:+.4f}", flush=True)
        if best is None or f > best[1]:
            best = (m, f, kp, st, info)
    mask, faith, kept, steps, last_info = best
    print(f"[extract] BEST-of-{args.num_rollouts} (by faith): kept {kept}  faith {faith:.4f}", flush=True)

    # --- the circuit = best sampled final mask ---
    edge_names = [engine.edge_list[i].name for i in mask.nonzero(as_tuple=True)[0].tolist()]
    kl = engine.run_with_mask(mask)
    kl_cut = engine.corrupted_baseline()

    payload = {
        "source": "agent",
        "run": run_dir.name,
        "ckpt": ckpt.name,
        "task": task_name,
        "n_edges": len(edge_names),
        "faith": faith,
        "kl": kl,
        "kl_cut": kl_cut,
        "stop_reason": last_info.get("reason"),
        "steps": steps,
        "num_rollouts": args.num_rollouts,
        "greedy_faith": gf,
        "greedy_edges": gk,
        "edges": edge_names,
    }
    out = Path(args.out) if args.out else (run_dir / "agent_circuit_ioi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n[extract] circuit (best of {args.num_rollouts} sampled): {len(edge_names)} edges "
          f"@ faith {faith:.4f} (KL {kl:.4f}, cut {kl_cut:.4f})", flush=True)
    print(f"[extract] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
