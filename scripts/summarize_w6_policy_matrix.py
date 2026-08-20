#!/usr/bin/env python3
"""Validate and summarize the W6 matched one-shot Smol/OFT policy matrix."""

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


def _keys_checksum(keys: list[str]) -> str:
    encoded = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _suite_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use SUITE=PATH")
    suite, raw_path = value.split("=", 1)
    return suite, Path(raw_path)


def _run_manifest(summary_path: Path) -> dict[str, Any]:
    path = summary_path.resolve().parent / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest missing beside summary: {path}")
    return _load(path)


def _render_markdown(result: dict[str, Any], *, title: str) -> str:
    state = result["state_pair_counts"]
    candidate = result["candidate_pair_counts_descriptive"]
    smol = result["smol_portfolio"]
    oft = result["oft_portfolio"]
    smol_candidate = result["smol_candidate"]
    oft_candidate = result["oft_candidate"]
    effect = result["paired_state_effect"]
    lines = [
        f"# {title}",
        "",
        "## Headline",
        "",
        f"- Smol candidate hits: {smol_candidate['hits']}/{smol_candidate['trials']}",
        f"- OFT candidate hits: {oft_candidate['hits']}/{oft_candidate['trials']}",
        f"- Smol portfolio states: {smol['hits']}/{smol['trials']}",
        f"- OFT portfolio states: {oft['hits']}/{oft['trials']}",
        "- State pairs: "
        f"both-hit={state['both_hit']}, Smol-only={state['smol_only']}, "
        f"OFT-only={state['oft_only']}, both-miss={state['both_miss']}",
        f"- Exact state-level McNemar p: {result['state_mcnemar_exact_p_two_sided']}",
        "- Paired portfolio risk difference (OFT - Smol): "
        f"{effect['risk_difference_oft_minus_smol']:.4f}",
        f"- Discordant state pairs: {effect['discordant_pairs']}; "
        "OFT-win fraction among discordant pairs: "
        f"{effect['oft_win_fraction_among_discordant']}",
        "",
        "## Candidate pairs (descriptive only)",
        "",
        f"both-hit={candidate['both_hit']}, Smol-only={candidate['smol_only']}, "
        f"OFT-only={candidate['oft_only']}, both-miss={candidate['both_miss']}",
        "",
        "## Per cell",
        "",
        "| dim | level | n | both hit | Smol only | OFT only | both miss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_cell_state_pairs"]:
        lines.append(
            f"| {row['dim']} | {row['level']} | {row['n_states']} | "
            f"{row['both_hit']} | {row['smol_only']} | {row['oft_only']} | {row['both_miss']} |"
        )
    lines.extend(
        [
            "",
            "## Per suite",
            "",
            "| suite | n | both hit | Smol only | OFT only | both miss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_suite_state_pairs"]:
        lines.append(
            f"| {row['suite']} | {row['n_states']} | "
            f"{row['both_hit']} | {row['smol_only']} | "
            f"{row['oft_only']} | {row['both_miss']} |"
        )
    lines.extend(["", "## Interpretation constraints", ""])
    lines.extend(f"- {warning}" for warning in result["warnings"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-keys", type=Path, required=True)
    parser.add_argument("--smol-summary", type=Path, required=True)
    parser.add_argument(
        "--oft-summary", type=_suite_path, action="append", required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="W6 L1–L2 paired one-shot policy matrix",
        help="Markdown report title",
    )
    args = parser.parse_args()

    key_artifact = _load(args.state_keys)
    keys = [str(key) for key in key_artifact.get("state_keys") or ()]
    checksum = _keys_checksum(keys)
    if checksum != key_artifact.get("state_keys_sha256"):
        raise SystemExit("frozen state-key checksum mismatch")

    summaries = [(suite, _load(path)) for suite, path in args.oft_summary]
    manifests = [_run_manifest(args.smol_summary)] + [
        _run_manifest(path) for _, path in args.oft_summary
    ]
    candidate_hashes = {manifest.get("candidates_dir_sha256") for manifest in manifests}
    if None in candidate_hashes or len(candidate_hashes) != 1:
        raise SystemExit(f"candidate artifact hash mismatch across arms: {candidate_hashes}")
    pool_hashes = {manifest.get("pool_manifest_sha256") for manifest in manifests}
    if None in pool_hashes or len(pool_hashes) != 1:
        raise SystemExit(f"pool manifest hash mismatch across arms: {pool_hashes}")

    from rase.collect.policy_matrix import aggregate_one_shot_policy_matrix
    from rase.collect.state_pool import StatePool

    pool = StatePool(Path(key_artifact["pool"]))
    pool_meta = {
        key: pool.read_state(key, load_observations=False).metadata.to_dict()
        for key in keys
    }
    result = aggregate_one_shot_policy_matrix(
        keys,
        _load(args.smol_summary),
        summaries,
        pool_meta=pool_meta,
        state_keys_sha256=checksum,
        candidate_artifact_sha256=next(iter(candidate_hashes)),
    )
    result["pool_manifest_sha256"] = next(iter(pool_hashes))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(
        _render_markdown(result, title=args.title), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in (
        "status", "n_states", "smol_candidate", "oft_candidate",
        "smol_portfolio", "oft_portfolio", "state_pair_counts",
        "state_mcnemar_exact_p_two_sided",
    )}, indent=2), flush=True)
    print(f"WROTE {args.output_json}")
    print(f"WROTE {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
