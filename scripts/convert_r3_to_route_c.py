#!/usr/bin/env python3
"""Convert R3 paired_chunks teacher_preferred into Route C training data.

Creates fresh LIBERO envs (does NOT require state pool restore).
For each teacher_preferred boundary, creates env, runs SmolVLA → OFT, extracts delta.
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.policy_step import as_batched_action, select_env_action
from rase.collect.pool_candidates import observation_from_libero_env
from rase.oracle.client import OracleClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-chunks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/pre_c0/pre_c0_48_state_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smolvla-ckpt", type=Path, default=ROOT / "ckpts" / "smolvla_libero")
    parser.add_argument("--oft-port", type=int, default=5555)
    parser.add_argument("--max-steps-before-sample", type=int, default=1,
                        help="how many env steps to take before sampling delta")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    r0_out = output_dir / "R0"
    r0_out.mkdir(exist_ok=True)

    paired = [json.loads(l) for l in args.paired_chunks.read_text().splitlines() if l.strip()]
    teacher_pref = [p for p in paired if p.get("label") == "teacher_preferred"]

    # Build manifest lookup: state_key → (task_id, suite)
    with open(args.manifest) as f:
        mf = json.load(f)
    manifest_lookup = {}
    for rec in mf.get("records", mf if isinstance(mf, list) else []):
        sk = rec.get("state_key", "")
        if sk:
            manifest_lookup[sk] = {
                "task_id": rec.get("task_id", rec.get("concrete_task_id", "")),
                "suite": rec.get("suite", ""),
                "step": rec.get("step", 1),
            }

    print(f"Teacher preferred: {len(teacher_pref)} pairs, manifest: {len(manifest_lookup)} states")

    # Group by unique state_key
    from collections import defaultdict
    by_boundary: dict[str, list[dict]] = defaultdict(list)
    for p in teacher_pref:
        by_boundary[p["state_key"]].append(p)

    unique_boundaries = sorted(by_boundary.keys())
    print(f"Unique boundaries: {len(unique_boundaries)}")

    # Helper: determine flavor from task_id
    def _flavor(task_id: str) -> str:
        import re
        m = re.match(r'libero_\w+_(\d+)', task_id)
        return "plus" if (m and int(m.group(1)) > 10) else "clean"

    # Split
    train_keys = unique_boundaries[:3]
    dev_keys = unique_boundaries[3:4] if len(unique_boundaries) > 3 else []

    # Load SmolVLA
    vlm_cache = str(Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467")
    bundle = load_smolvla_policy_bundle(
        args.smolvla_ckpt, device="cuda", tokenizer_path=vlm_cache,
        observation_height=360, observation_width=360,
    )

    client = OracleClient(f"tcp://127.0.0.1:{args.oft_port}", timeout_ms=60000)
    n_converted = 0

    for split_name, keys in [("train", train_keys), ("dev", dev_keys)]:
        for state_key in keys:
            pairs = by_boundary[state_key]
            info = manifest_lookup.get(state_key)
            if not info:
                print(f"[{split_name}] SKIP {state_key[:16]}: not in manifest")
                continue

            task_id = info["task_id"]
            suite = info["suite"]

            for pi in range(min(3, len(pairs))):
                try:
                    seed = args.seed + pi * 7
                    handle = make_libero_env_for_task(
                        task_id, init_state_id=seed % 50, seed=seed,
                        libero_flavor=_flavor(task_id),
                    )
                    single = handle.vector_env.envs[0]
                    instruction = str(getattr(single, "task_description", "") or "")

                    # Step forward a few steps to approximate boundary
                    obs = observation_from_libero_env(single)
                    for _ in range(args.max_steps_before_sample):
                        action = select_env_action(bundle, obs, task=instruction)
                        obs, _, _, _, _ = handle.vector_env.step(as_batched_action(action))
                        obs = observation_from_libero_env(single)
                        # Stop if episode ended
                        if hasattr(single, '_terminated') and single._terminated:
                            break

                    # SmolVLA student action
                    student_a = select_env_action(bundle, obs, task=instruction)
                    student_a = np.asarray(student_a, dtype=np.float32).flatten()[:7]

                    # OFT teacher action
                    oft = OracleChunkContinuation(client, instruction=instruction, control_env=handle.control_env)
                    teacher_a = oft.act(obs, task=instruction)
                    teacher_a = np.asarray(teacher_a, dtype=np.float32).flatten()[:7]

                    delta = np.clip(teacher_a - student_a, -0.5, 0.5)

                    sample = {
                        "state_key": state_key, "suite": suite, "task": pairs[0]["task"],
                        "split": split_name, "seed_idx": pi,
                        "student_action": student_a.tolist(),
                        "teacher_action": teacher_a.tolist(),
                        "delta_target": delta.tolist(),
                        "source": "r3_fresh_env",
                    }

                    out_path = r0_out / f"r3_{split_name}_{state_key[:12]}_s{pi}.json"
                    out_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
                    n_converted += 1

                    n_s = np.linalg.norm(student_a).item()
                    n_t = np.linalg.norm(teacher_a).item()
                    n_d = np.linalg.norm(delta).item()
                    print(f"[{split_name}] {suite} {pairs[0]['task'][:40]} S={n_s:.3f} T={n_t:.3f} D={n_d:.3f}")

                    handle.close()
                except Exception as e:
                    print(f"  SKIP {state_key[:16]} s{pi}: {e}")

    client.close()

    n_r0 = len(list(r0_out.glob("*.json")))
    gate = {"R0_count": n_r0, "unique_boundaries": len(unique_boundaries),
            "train_keys": train_keys, "dev_keys": dev_keys, "gate_pass": n_r0 >= 8}
    (output_dir / "round0_plugin_data_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    print(f"\nConversion: {n_converted}/{len(teacher_pref)} converted, {n_r0} R0 samples")
    return 0 if n_r0 >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
