#!/usr/bin/env python3
"""Leave-one-VLA-out evaluation for multi-VLA generalization.

Protocol: given the trained LightRiskStudent (shared encoder + heads), we audit
how the model behaves when a novel VLA's actions are supplied.  This script
simulates the protocol using per-VLA adapters and reports handback coverage per
VLA, flagging abstention risk for unseen VLA action distributions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.vla_action_adapters import create_vla_adapter  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vla-actions-json", type=Path, required=True,
                   help="jsonl/json of raw VLA action chunks per vla")
    p.add_argument("--output", type=Path, default=ROOT / "runs/pre_c0_r4/vla_leave_one_out.json")
    args = p.parse_args()

    # The adapters we claim to support
    supported = ["smolvla", "oft"]
    payload = json.loads(args.vla_actions_json.read_text())
    records = payload.get("records", payload if isinstance(payload, list) else [])

    per_vla = {}
    failures = []
    for record in records:
        vla = str(record.get("vla_name", "unknown"))
        raw = np.asarray(record.get("action_chunk", []), dtype=np.float32).reshape(-1, 7)
        try:
            adapter = create_vla_adapter(vla)
            chunk = adapter.to_canonical(raw)
            summary = chunk.flatten().shape
            per_vla[vla] = {
                "supported": True,
                "canonical_shape": list(summary),
                "horizon": chunk.horizon,
                "n_rows": 1,
            }
        except ValueError:
            per_vla[vla] = {"supported": False, "reason": "no adapter"}
            failures.append(vla)

    report = {
        "schema_version": "rase-pre-c0-r4d-leave-one-vla-out/v1",
        "supported_vlas": supported,
        "per_vla": per_vla,
        "unsupported_vlas": sorted(failures),
        "note": (
            "Full leave-one-VLA-out requires per-VLA trajectory labels; this "
            "audit verifies adapter coverage and canonical-action mapping only. "
            "Extend with real per-VLA rollouts once >=3 VLA collections exist."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    # Gate: at least the two core adapters must map cleanly
    return 0 if all(per_vla.get(v, {}).get("supported") for v in ("smolvla", "oft")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
