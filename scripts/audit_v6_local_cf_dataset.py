#!/usr/bin/env python3
"""Audit V6 local C/R data and materialize root-level relative labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def audit_record(record: Mapping[str, Any], repeats: int) -> tuple[dict[str, Any] | None, list[str]]:
    root = record.get("root")
    if not isinstance(root, Mapping): return None, ["missing_root"]
    errors: list[str] = []
    if record.get("schema_version") != "rase-v6-local-cf-root/v1" or record.get("status") != "complete":
        errors.append("incomplete_or_wrong_schema")
    rows = record.get("branches")
    if not isinstance(rows, list): return None, errors + ["missing_branches"]
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema_version") != "rase-v6-local-cf-branch/v1":
            errors.append("wrong_branch_schema"); continue
        by_kind[str(row.get("candidate_kind"))].append(row)
    if set(by_kind) != {"C", "R"}: errors.append(f"candidate_kinds={sorted(by_kind)}")
    expected = [int(seed) for seed in root.get("downstream_seeds") or []]
    if len(expected) != repeats or len(set(expected)) != repeats: errors.append("invalid_planned_downstream_seeds")
    qpos_values: set[str] = set()
    for kind in ("C", "R"):
        kind_rows = by_kind.get(kind, [])
        seeds = [int(row.get("downstream_seed", -1)) for row in kind_rows]
        if sorted(seeds) != sorted(expected): errors.append(f"{kind}_downstream_seeds_not_matched")
        if len({str(row.get("candidate_chunk_sha256")) for row in kind_rows}) != 1: errors.append(f"{kind}_candidate_not_frozen")
        qpos_values.update(str(row.get("root_snapshot_qpos_sha256")) for row in kind_rows)
        if any(not isinstance(row.get("success"), bool) for row in kind_rows): errors.append(f"{kind}_nonbool_success")
    if len(qpos_values) != 1: errors.append("same_root_snapshot_drift")
    if errors: return None, errors
    outcomes = {kind: {int(row["downstream_seed"]): int(bool(row["success"])) for row in by_kind[kind]} for kind in ("C", "R")}
    qc = float(np.mean(list(outcomes["C"].values()))); qr = float(np.mean(list(outcomes["R"].values())))
    return {"root_id": str(root["root_id"]), "task_id": str(root["task_id"]), "task_number": int(root["task_number"]),
            "cursor": int(root["cursor"]), "pre_decision_chunks": int(root["pre_decision_chunks"]),
            "q_continue": qc, "q_refresh": qr, "adv_refresh": qr-qc,
            "y_pref": int(qr > qc), "y_risk": int(qc < 1.0), "outcomes": outcomes,
            "instruction": record.get("instruction"), "artifact": (record.get("boundary") or {}).get("artifact")}, []


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--repeats",type=int,default=2); args=p.parse_args()
    roots_dir=args.input/"roots" if (args.input/"roots").is_dir() else args.input
    good=[]; failures={}
    for path in sorted(roots_dir.glob("*.json")):
        record=json.loads(path.read_text(encoding="utf-8")); rid=str((record.get("root") or {}).get("root_id",path.stem)); value,err=audit_record(record,args.repeats)
        if err: failures[rid]=err
        elif value: good.append(value)
    adv=np.array([x["adv_refresh"] for x in good],dtype=float)
    result={"schema_version":"rase-v6-local-cf-audit/v1","n_auditable_roots":len(good),"n_audit_failures":len(failures),"audit_failures":failures,
      "summary":{"mean_q_continue":float(np.mean([x["q_continue"] for x in good])) if good else None,"mean_q_refresh":float(np.mean([x["q_refresh"] for x in good])) if good else None,
        "mean_adv_refresh":float(adv.mean()) if len(adv) else None,"refresh_better_roots":int((adv>0).sum()),"continue_better_roots":int((adv<0).sum()),"ties":int((adv==0).sum()),
        "tasks":dict(sorted(Counter(x["task_id"] for x in good).items())),"label_pref_positive":int(sum(x["y_pref"] for x in good)),"label_risk_positive":int(sum(x["y_risk"] for x in good))},"roots":good}
    atomic_json(args.output,result); print(json.dumps({k:result[k] for k in ("n_auditable_roots","n_audit_failures","summary")},sort_keys=True)); return 0 if good else 2
if __name__=="__main__": raise SystemExit(main())
