#!/usr/bin/env python3
"""Print compact progress for the resume-safe vNext discovery collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = {str(job["job_id"]): job for job in manifest["jobs"]}
    rows = []
    invalid = []
    for path in sorted((args.output_dir / "groups").glob("*.json")):
        try:
            rows.extend(json.loads(path.read_text())["rows"])
        except Exception as exc:
            invalid.append(f"{path.name}:{type(exc).__name__}")
    observed = {str(row["job_id"]): row for row in rows}
    counts = Counter(
        (str(row["suite"]), str(row["policy_id"])) for row in observed.values()
    )
    result = {
        "status": "COMPLETE" if set(observed) == set(expected) and not invalid else "RUNNING",
        "expected_jobs": len(expected), "observed_jobs": len(observed),
        "progress_fraction": len(observed) / len(expected),
        "available_jobs": sum(row.get("available") is True for row in observed.values()),
        "masked_jobs": sum(row.get("available") is False for row in observed.values()),
        "success_jobs": sum(row.get("available") is True and bool(row.get("success")) for row in observed.values()),
        "by_suite_policy": {f"{suite}/{policy}": count for (suite, policy), count in sorted(counts.items())},
        "invalid_group_files": invalid,
        "complete_artifact": (args.output_dir / "COMPLETE.json").exists(),
    }
    if args.compact:
        print(
            f"{result['status']} jobs={result['observed_jobs']}/{result['expected_jobs']} "
            f"available={result['available_jobs']} masked={result['masked_jobs']} "
            f"success={result['success_jobs']} invalid={len(result['invalid_group_files'])}",
            flush=True,
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
