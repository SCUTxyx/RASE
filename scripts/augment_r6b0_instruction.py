#!/usr/bin/env python3
"""Add deployable instruction features to a frozen R6-B0 dataset without rerollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_instruction(text: str, dim: int = 256) -> np.ndarray:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    words = normalized.split()
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    features += [normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))]
    value = np.zeros(dim, dtype=np.float32)
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        value[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--initial-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input_report.read_text())
    if report["dataset_sha256"] != sha256(args.input):
        raise ValueError("input dataset lock mismatch")
    initial = json.loads(args.initial_keys.read_text())
    from rase.collect.state_pool import StatePool
    pool = StatePool(Path(str(initial["pool"])).resolve())
    instruction_by_key = {
        str(key): str(pool.read_state(str(key), load_observations=False).metadata.instruction)
        for key in initial["state_keys"]
    }
    raw = np.load(args.input)
    arrays = {key: raw[key] for key in raw.files}
    instructions = np.asarray([instruction_by_key[str(key)] for key in arrays["state_key"]])
    arrays["instruction"] = instructions
    arrays["language_hash"] = np.stack([hashed_instruction(text) for text in instructions])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    output_report = dict(report)
    output_report.update({
        "schema_version": "rase-r6b0-takeover-dataset/v2-language-exploratory",
        "dataset": str(args.output.resolve()),
        "dataset_sha256": sha256(args.output),
        "parent_dataset": str(args.input.resolve()),
        "parent_dataset_sha256": sha256(args.input),
        "protocol_status": "post-hoc exploratory repair; requires independent validation",
        "feature_policy": "two RGB views + 8D proprio + 256D hashed instruction + seed0 canonical 10-step action summary + policy ID",
    })
    output_report_path = args.output.with_suffix(".report.json")
    output_report_path.write_text(json.dumps(output_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
