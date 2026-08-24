#!/usr/bin/env python3
"""Wait for V6 Stage-0 lanes, validate completeness, then merge and analyse."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", action="append", required=True, type=int)
    parser.add_argument("--lane", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if len(args.pid) != len(args.lane):
        parser.error("--pid and --lane counts must match")
    output = args.output_dir.resolve()
    while any(process_alive(pid) for pid in args.pid):
        time.sleep(args.poll_seconds)
    summaries: list[dict[str, Any]] = []
    valid = True
    for lane in args.lane:
        path = lane.resolve() / "collection_summary.json"
        if not path.is_file():
            summaries.append({"lane": str(lane), "error": "missing_collection_summary"})
            valid = False
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        expected = 6 * int(summary.get("n_planned_roots", 0))
        if int(summary.get("n_branch_rows", -1)) != expected or int(summary.get("uncaught_root_errors", -1)) != 0:
            valid = False
        summaries.append({"lane": str(lane), "summary": summary, "expected_branch_rows": expected})
    manifest = {
        "schema_version": "rase-v6-stage0-watcher/v1",
        "pids": args.pid,
        "lanes": summaries,
        "complete_and_auditable": valid,
    }
    if not valid:
        atomic_json(output / "postprocess_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True), flush=True)
        return 2
    merge = Path(__file__).with_name("merge_v6_stage0_records.py")
    analyse = Path(__file__).with_name("analyze_v6_refresh_opportunity.py")
    merged = output / "stage0_records.jsonl"
    merge_command = [sys.executable, str(merge), *sum((["--input", str(path.resolve())] for path in args.lane), []), "--output", str(merged)]
    merge_result = subprocess.run(merge_command, text=True, capture_output=True)
    manifest["merge_returncode"] = merge_result.returncode
    manifest["merge_stdout"] = merge_result.stdout[-4000:]
    manifest["merge_stderr"] = merge_result.stderr[-4000:]
    if merge_result.returncode != 0:
        atomic_json(output / "postprocess_manifest.json", manifest)
        return merge_result.returncode
    analysis_path = output / "pilot_analysis.json"
    analysis_command = [
        sys.executable, str(analyse), "--input", str(merged), "--mode", "pilot",
        "--expected-r-new-k", "4", "--bootstrap", "10000", "--output", str(analysis_path),
    ]
    analysis_result = subprocess.run(analysis_command, text=True, capture_output=True)
    manifest["analysis_returncode"] = analysis_result.returncode
    manifest["analysis_stdout"] = analysis_result.stdout[-4000:]
    manifest["analysis_stderr"] = analysis_result.stderr[-4000:]
    manifest["analysis"] = str(analysis_path)
    atomic_json(output / "postprocess_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    # A valid experimental FAIL is an expected control-flow result.  Preserve
    # it in the manifest but let the watcher exit cleanly for monitoring.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
