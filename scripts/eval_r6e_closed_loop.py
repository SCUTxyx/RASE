#!/usr/bin/env python3
"""R6-E independent closed-loop paired evaluation harness.

Sealed until the R6-C stage gate passes (see
``configs/r6b1_dynamic_boundary_protocol_v1.json`` validation_test_lock).  The
harness is the executable entry point once unlocked; it refuses to run before
the unlock and records the pre-registered protocol otherwise.

Workflow (per validation state, after unlock):

1. Restore the frozen task-disjoint validation snapshot.
2. Run the source VLA until a scheduled boundary; evaluate the frozen R6-C
   risk model (3-member ensemble LCB) at each boundary.
3. With two-boundary dwell, decide CONTINUE_SOURCE vs ENTER_PERSISTENT_OFT
   using the frozen per-policy threshold.
4. Record the paired arms: source-only outcome and selector outcome.
5. Aggregate >= 100 paired episodes across four suites and both source VLA
   seeds; report success, teacher steps, false continue, and savings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = ROOT / "configs/r6b1_dynamic_boundary_protocol_v1.json"
MIN_EPISODES = 100


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-keys", type=Path, required=True)
    parser.add_argument("--frozen-thresholds", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="R6-C candidate-arm dataset for feature normalization")
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--seed-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=MIN_EPISODES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol = _load(PROTOCOL)
    lock = protocol.get("validation_test_lock", {})
    independent_validation = lock.get("independent_validation", "locked")
    if independent_validation != "unlocked":
        print(f"R6E sealed: independent_validation={independent_validation!r}; "
              f"harness is inert until the R6-C stage gate passes.")
        return 2

    # Unlocked path.  Import the model and simulator wiring lazily so the sealed
    # path above never loads torch/lerobot.
    from rase.collect.state_pool import StatePool  # noqa: F401
    from rase.risk.light_risk_student import CandidateArmStudent  # noqa: F401

    raise NotImplementedError(
        "R6E closed-loop rollout body is implemented by the unlock run; "
        "see the pre-registered protocol in the module docstring.")

    # --- Protocol body (after unlock) ---
    # val = _load(args.val_keys)
    # thresholds = _load(args.frozen_thresholds)["by_policy"][args.policy_id]
    # pool = StatePool(Path(val["pool"]).resolve())
    # dataset = np.load(args.dataset)
    # ... restore, run source to boundary, LCB ensemble scoring with dwell,
    # ... switch to OFT when risky, record paired arms, aggregate report.
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())
