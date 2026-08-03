#!/usr/bin/env python3
"""Validate and combine suite-specific OFT prefix-ablation summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use SUITE=PATH")
    name, raw = value.split("=", 1)
    return name, Path(raw)


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# W7 OFT action-prefix causal ablation",
        "",
        "## Classification counts",
        "",
    ]
    lines.extend(
        f"- {label}: {count}"
        for label, count in result["classification_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Per state",
            "",
            "| state | suite | dim | level | direct | zero | candidates | mechanism |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in result["per_state"]:
        lines.append(
            f"| {row['state_key']} | {row.get('suite', '')} | {row.get('dim', '')} | "
            f"{row.get('level', '')} | {int(row['direct_oft_success'])} | "
            f"{int(row['zero_prefix_success'])} | "
            f"{row['candidate_hits']}/{row['candidate_trials']} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "Candidate-specific rescue is assigned only when direct OFT and the "
            "time-matched zero prefix both fail but at least one frozen candidate succeeds.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys", type=Path, required=True)
    parser.add_argument("--summary", action="append", type=_named_path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    key_artifact = _load(args.state_keys.resolve())
    keys = [str(key) for key in key_artifact.get("state_keys") or []]
    checksum = hashlib.sha256(
        json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if checksum != key_artifact.get("state_keys_sha256"):
        raise SystemExit("state-key checksum mismatch")

    summaries = []
    manifests = []
    for suite, path in args.summary:
        summary = _load(path)
        if summary.get("suite") not in {suite, None}:
            raise SystemExit(f"summary suite mismatch: {summary.get('suite')} != {suite}")
        if summary.get("state_keys_sha256") != checksum:
            raise SystemExit(f"state-key provenance mismatch in {path}")
        summaries.append(summary)
        manifest_path = path.resolve().parent / "run_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"run manifest missing: {manifest_path}")
        manifests.append(_load(manifest_path))
    candidate_hashes = {item.get("candidates_dir_sha256") for item in manifests}
    pool_hashes = {item.get("pool_manifest_sha256") for item in manifests}
    if None in candidate_hashes or len(candidate_hashes) != 1:
        raise SystemExit("candidate provenance mismatch across suite summaries")
    if None in pool_hashes or len(pool_hashes) != 1:
        raise SystemExit("pool provenance mismatch across suite summaries")

    from rase.collect.prefix_ablation import aggregate_prefix_summaries

    result = aggregate_prefix_summaries(keys, summaries)
    result["state_keys_sha256"] = checksum
    result["candidates_dir_sha256"] = next(iter(candidate_hashes))
    result["pool_manifest_sha256"] = next(iter(pool_hashes))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_states": result["n_states"],
                "classification_counts": result["classification_counts"],
                "candidate_specific_rescue_states": result[
                    "candidate_specific_rescue_states"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"WROTE {args.output_json}")
    print(f"WROTE {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
