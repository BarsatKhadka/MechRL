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
    policy = CircuitPolicy(hidden=cfg.get("hidden", 128)).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()

    # --- greedy rollout (deterministic policy = the circuit it commits to) ---
    obs = env.reset(bundle_idx=0)
    steps, last_info = 0, {}
    with torch.no_grad():
        while not env.done:
            logits, _ = policy(move_obs(obs, device))
            action = int(torch.argmax(logits).item())
            obs, _, done, last_info = env.step(action)
            steps += 1
    print(f"[extract] episode ended: reason={last_info.get('reason')} steps={steps}", flush=True)

    # --- the circuit = final mask ---
    mask = env.mask.clone().cpu()
    edge_names = [engine.edge_list[i].name for i in mask.nonzero(as_tuple=True)[0].tolist()]
    faith = engine.faithfulness(mask)
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
        "edges": edge_names,
    }
    out = Path(args.out) if args.out else (run_dir / "agent_circuit_ioi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n[extract] circuit: {len(edge_names)} edges @ faith {faith:.4f} "
          f"(KL {kl:.4f}, cut {kl_cut:.4f})", flush=True)
    print(f"[extract] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
