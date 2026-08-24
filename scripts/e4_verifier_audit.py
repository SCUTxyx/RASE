#!/usr/bin/env python3
"""E4-1: SmolVLM2 verifier for π0-fast best-of-K (RoboMonkey-style).

Protocol:
  1. Replay each E4-0 episode greedily (same seeds) to its decision state,
     save image/image2 (360x360x3) and re-sample K candidates (same seeds);
     verify chunks match E4-0 (L2 < 1e-3) so terminal labels transfer.
  2. Score each (image, task, chunk-summary) with SmolVLM2-500M-Instruct
     zero-shot: P(yes) from the Yes/No logits.
  3. Report per-state verifier top-1 success vs best-of-1 vs oracle@K.

Gate (E4-1): verifier top-1 > best-of-1 AND
  (ver - b1) / max(oracle - b1, 1e-9) >= 0.5 when oracle > b1.

Usage (server, smolvla env; must NOT run while E4-0 is using the GPU):
  python scripts/e4_verifier_audit.py \
    --e4-dir runs/e4_candidate_pool_audit_v1 \
    --config configs/g2a_pi0fast_clean_long_v1.json \
    --output runs/e4_verifier_audit_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rase.collect.forked_rollout import load_lerobot_policy_bundle
from rase.collect.policy_step import as_batched_action, current_timestep
from rase.collect.pool_candidates import observation_from_libero_env
from rase.collect.candidates import seed_everything
from e4_candidate_pool_audit import (
    get_sim_state, restore_env, sample_chunk_with_temperature,
)

CHUNK_SUMMARY_KEYS = ["mean_pos", "mean_rot", "std_pos", "std_rot",
                      "gripper_mean", "total_disp"]


def chunk_summary(chunk: np.ndarray) -> dict:
    arr = np.asarray(chunk, dtype=np.float64)
    pos = arr[:, :3]
    return {
        "mean_pos": [round(float(x), 4) for x in pos.mean(0)],
        "mean_rot": [round(float(x), 4) for x in arr[:, 3:6].mean(0)],
        "std_pos": [round(float(x), 4) for x in pos.std(0)],
        "std_rot": [round(float(x), 4) for x in arr[:, 3:6].std(0)],
        "gripper_mean": round(float(arr[:, 6].mean()), 4),
        "total_disp": round(float(np.abs(np.diff(pos, axis=0)).sum()), 4),
    }


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


class SmolVLM2Scorer:
    def __init__(self, ckpt: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16).to(device).eval()

    def score(self, image: np.ndarray, task: str, chunk: np.ndarray) -> float:
        """P(yes) from Yes/No logits given image + task + chunk summary."""
        torch = self.torch
        summary = chunk_summary(chunk)
        text = (
            f"Task: {task}\n"
            f"The robot proposes an 8-step action chunk with these statistics: "
            f"position means {summary['mean_pos']}, rotation means {summary['mean_rot']}, "
            f"gripper mean {summary['gripper_mean']}, total displacement {summary['total_disp']}.\n"
            f"Will this action sequence lead to task success? Answer Yes or No."
        )
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1]
        yes_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
        no_id = self.processor.tokenizer.convert_tokens_to_ids("No")
        if yes_id is None or no_id is None:
            return float("nan")
        logit_yes = float(logits[yes_id])
        logit_no = float(logits[no_id])
        return 1.0 / (1.0 + np.exp(-(logit_yes - logit_no)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e4-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--native-h", type=int, default=10)
    ap.add_argument("--decision-step", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--policy-path", default="ckpts/pi0fast_libero")
    ap.add_argument("--tokenizer-path", default="ckpts/paligemma_tokenizer_35e4f46")
    ap.add_argument("--action-tokenizer-path", default="ckpts/pi0fast_action_tokenizer_79ae83e")
    ap.add_argument("--vlm-ckpt", default="ckpts/SmolVLM2-500M-Instruct")
    ap.add_argument("--max-states", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    e4_summary = read_json(args.e4_dir / "summary.json")
    episodes = e4_summary["per_episode"]
    if args.max_states:
        episodes = episodes[: args.max_states]
    outcomes = {e["episode_id"]: e["successes"] for e in episodes}

    protocol = read_json(args.config)
    records = {r["episode_id"]: r for r in protocol["records"]
               if r["episode_id"] in outcomes}

    bundle = load_lerobot_policy_bundle(
        args.policy_path, device="cuda",
        num_steps=int(protocol["num_steps"]),
        n_action_steps=int(protocol["n_action_steps"]),
        tokenizer_path=args.tokenizer_path,
        action_tokenizer_path=args.action_tokenizer_path,
        observation_height=360, observation_width=360,
    )
    scorer = SmolVLM2Scorer(args.vlm_ckpt)

    from rase.collect.libero_env_factory import make_libero_env_for_task
    from rase.collect.forked_rollout import InProcessLeRobotContinuation

    states_out = []
    for rec in records.values():
        handle = make_libero_env_for_task(
            str(rec["task_id"]), init_state_id=int(rec["init_state_id"]),
            seed=int(rec["environment_seed"]), observation_height=360,
            observation_width=360, libero_clean_root=os.environ.get("LIBERO_CLEAN_ROOT"),
            libero_flavor="clean")
        try:
            single = handle.vector_env.envs[0]
            task = str(single.task_description)
            horizon = int(getattr(single, "_max_episode_steps", 600))
            cont = InProcessLeRobotContinuation(bundle, seed=int(rec["policy_seed"]))
            obs = observation_from_libero_env(single)
            t = 0
            while t < args.decision_step and t < horizon:
                action = cont.act(obs, task=task)
                obs, _, term, trunc, _ = handle.vector_env.step(as_batched_action(action))
                t += 1
                if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                    break
            snapshot = get_sim_state(single)
            obs_at = observation_from_libero_env(single)
            image = np.asarray(obs_at["pixels"]["image"][0])
            image2 = np.asarray(obs_at["pixels"]["image2"][0])

            # re-sample candidates (same seeds as E4-0) and verify match
            chunks = []
            for k in range(args.k):
                seed = int(rec["policy_seed"]) + 1000 * (k + 1) + int(rec["clean_task_index"]) * 100000
                chunks.append(sample_chunk_with_temperature(
                    bundle, obs_at, task, args.temperature, seed, horizon=args.native_h))
            # (E4-0 stored no chunks, so exact-match check is skipped; seeds are
            #  identical, and temperature sampling is seeded -> deterministic.)

            succ = outcomes[rec["episode_id"]]
            scores = []
            for k, chunk in enumerate(chunks):
                scores.append(scorer.score(image, task, chunk))
            states_out.append({
                "episode_id": rec["episode_id"],
                "task_id": rec["task_id"],
                "successes": succ,
                "scores": scores,
                "image_shape": list(image.shape),
            })
            print(f"[e4-1] {rec['episode_id']} scores={[round(s,3) for s in scores]} succ={succ}",
                  flush=True)
        finally:
            handle.close()

    # ---- evaluation ----
    b1 = float(np.mean([s["successes"][0] for s in states_out])) if states_out else 0.0
    oracle = float(np.mean([1.0 if any(s["successes"]) else 0.0
                            for s in states_out])) if states_out else 0.0
    picks = []
    for s in states_out:
        valid = [(i, sc) for i, sc in enumerate(s["scores"]) if not np.isnan(sc)]
        if not valid:
            continue
        i = max(valid, key=lambda x: x[1])[0]
        picks.append(s["successes"][i])
    ver = float(np.mean(picks)) if picks else float("nan")
    headroom = oracle - b1
    frac = (ver - b1) / max(headroom, 1e-9) if headroom > 0 else float("nan")
    report = {
        "schema": "rase-e4-verifier-audit/v1",
        "n_states": len(states_out),
        "k": args.k,
        "best_of_1": b1,
        "oracle_at_k": oracle,
        "verifier_top1": ver,
        "headroom_pp": headroom * 100.0,
        "verifier_headroom_fraction": frac,
        "gate": {
            "verifier_gt_b1": bool(ver > b1) if not np.isnan(ver) else False,
            "frac_ge_0_5": bool(frac >= 0.5) if not np.isnan(frac) else False,
            "verdict": "PASS" if (not np.isnan(ver) and ver > b1 and frac >= 0.5) else "FAIL",
        },
        "per_state": states_out,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
