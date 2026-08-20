#!/usr/bin/env python3
"""Synthetic smoke test for the R6-D --wm-features path through the R6-C
CandidateArmStudent trainer: builds a tiny candidate-arm npz + report + a
wm_features.jsonl cache with the pre-registered layout, then runs
train_r6c_candidate_arm_student.py --wm-features end to end.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/RASE")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "smoke_r6d_wm_path"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)
N_GROUPS = 24
HORIZON = 600.0
arm_success = []
arm_steps = []
rows = []
image = []
proprio = []
action = []
history = []
task_id = []
state_key = []
group_id = []
policy_id = []
elapsed = []
within = {"8": [], "16": [], "32": []}
for group in range(N_GROUPS):
    policy = "pi0fast_libero" if group % 2 == 0 else "pi05_libero"
    n_b = 3 + group % 3
    for b in range(n_b):
        succ = bool(rng.random() < 0.55)
        persistent = bool(rng.random() < 0.72)
        persistent_steps = float(rng.integers(40, 320))
        rows.append({"group_id": f"smoke_group_{group:03d}"})
        image.append(rng.integers(0, 255, (2, 3, 96, 96), dtype=np.uint8))
        proprio.append(rng.normal(size=8).astype(np.float32))
        action.append(rng.normal(size=20).astype(np.float32))
        history.append(rng.normal(size=28).astype(np.float32))
        task_id.append(f"task_{group // 6}")
        state_key.append(f"state_{group:03d}")
        group_id.append(f"smoke_group_{group:03d}")
        policy_id.append(policy)
        elapsed.append(float(group * 10 + b * 5))
        arm_success.append([succ, persistent])
        arm_steps.append([0.0, persistent_steps])
        within["8"].append(bool(rng.random() < 0.6))
        within["16"].append(bool(rng.random() < 0.5))
        within["32"].append(bool(rng.random() < 0.4))

n = len(rows)
elapsed_arr = np.asarray(elapsed, dtype=np.float32)
data = {
    "image": np.stack(image), "proprio": np.stack(proprio),
    "action_summary": np.stack(action), "history": np.stack(history),
    "elapsed_progress": elapsed_arr / HORIZON,
    "elapsed_source_steps": elapsed_arr.astype(np.int32),
    "instruction": np.asarray([f"instr {i}" for i in range(n)]),
    "language_hash": np.asarray(rng.normal(size=(n, 256)).astype(np.float32)),
    "state_key": np.asarray(state_key), "task_id": np.asarray(task_id),
    "suite": np.asarray(["smoke"] * n), "group_id": np.asarray(group_id),
    "policy_id": np.asarray(policy_id),
    "policy_index": np.asarray([0 if p == "pi0fast_libero" else 1 for p in policy_id], dtype=np.int64),
    "source_success": np.asarray([s for s, _ in arm_success], dtype=np.float32),
    "source_trials": np.ones(n, dtype=np.float32),
    "source_within_8": np.asarray(within["8"], dtype=np.float32),
    "source_within_16": np.asarray(within["16"], dtype=np.float32),
    "source_within_32": np.asarray(within["32"], dtype=np.float32),
    "persistent_success": np.asarray([p for _, p in arm_success], dtype=np.float32),
    "persistent_teacher_steps": np.asarray([s for _, s in arm_steps], dtype=np.float32),
    "arm_success": np.asarray(arm_success, dtype=np.float32),
    "arm_teacher_steps": np.asarray(arm_steps, dtype=np.float32),
    "arm_ids": np.asarray([0, 1], dtype=np.int64),
}
npz_path = OUT / "candidate_arm_smoke.npz"
np.savez_compressed(npz_path, **data)
sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
(OUT / "candidate_arm_smoke.npz.report.json").write_text(json.dumps({
    "schema_version": "rase-r6c-candidate-arm-dataset/v1",
    "status": "complete",
    "dataset": str(npz_path.resolve()), "dataset_sha256": sha,
    "n_rows": n, "n_groups": N_GROUPS, "diagnostic": "synthetic_wm_path_smoke",
}, indent=2) + "\n")

# WM feature cache with the pre-registered layout (K=1/4/8).
wm_rows = []
for i, g in enumerate(group_id):
    wm_rows.append({
        "group_id": g,
        "elapsed_source_steps": int(elapsed_arr[i]),
        "state_key": state_key[i], "policy_id": policy_id[i],
        "task_id": task_id[i], "suite": "smoke", "seed_index": 0,
        "k_values": [1, 4, 8],
        "latent_z_t": rng.normal(size=16).tolist(),
        "predicted_latents": {"1": rng.normal(size=16).tolist(),
                              "4": rng.normal(size=16).tolist(),
                              "8": rng.normal(size=16).tolist()},
        "disagreement": {"delta_direction_var": float(rng.random()),
                         "delta_magnitude_var": float(rng.random())},
        "residual": {"1": rng.normal(size=32).tolist(),
                     "4": rng.normal(size=32).tolist(),
                     "8": rng.normal(size=32).tolist()},
    })
wm_path = OUT / "wm_features_smoke.jsonl"
wm_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in wm_rows))
wm_dim = None
from rase.risk.wm_features import feature_vector
wm_dim = feature_vector(wm_rows[0]).shape[0]
print(f"wm_dim={wm_dim} (expected 3*residual_dim(16)+2+latent(16)=66)", flush=True)
assert wm_dim == 3 * 16 + 2 + 16, wm_dim

out = OUT / "oof_wm_smoke.json"
python_bin = Path("/root/autodl-tmp/envs/oft/bin/python")
cmd = [
    str(python_bin), str(ROOT / "scripts/train_r6c_candidate_arm_student.py"),
    "--dataset", str(npz_path), "--dataset-report", str(npz_path.with_suffix(".npz.report.json")),
    "--output", str(out), "--mode", "per_vla", "--target-policy", "pi0fast_libero",
    "--seed", "1", "--folds", "3", "--fold-seed", "1", "--members", "2",
    "--epochs", "3", "--dwell", "2", "--device", "cpu",
    "--wm-features", str(wm_path),
]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
result = json.loads(out.read_text())
assert result["wm_feature_dim"] == wm_dim, result
assert all("persistent_success" in row and "source_success" in row for row in result["predictions"])

baseline_out = OUT / "oof_baseline_smoke.json"
cmd = [
    str(python_bin), str(ROOT / "scripts/train_r6c_candidate_arm_student.py"),
    "--dataset", str(npz_path), "--dataset-report", str(npz_path.with_suffix(".npz.report.json")),
    "--output", str(baseline_out), "--mode", "per_vla", "--target-policy", "pi0fast_libero",
    "--seed", "1", "--folds", "3", "--fold-seed", "1", "--members", "2",
    "--epochs", "3", "--dwell", "2", "--device", "cpu",
]
subprocess.run(cmd, check=True)

pareto = OUT / "pareto_smoke.json"
cmd = [
    str(python_bin), str(ROOT / "scripts/eval_r6d_wm_ablation.py"),
    "--baseline", str(baseline_out), "--wm", str(out),
    "--output", str(pareto),
]
print(" ".join(cmd), flush=True)
# A synthetic dataset is expected to reject the WM arm (exit 2 is the design's
# "reject" signal); both keep and reject are valid smoke outcomes.
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as exc:
    if exc.returncode != 2:
        raise
pareto_result = json.loads(pareto.read_text())
assert "decision" in pareto_result
print(json.dumps({"status": "WM smoke PASS", "wm_dim": result["wm_feature_dim"],
                  "pareto_decision": pareto_result["decision"],
                  "metrics": result["metrics"]}, indent=2))
