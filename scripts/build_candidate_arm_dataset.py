#!/usr/bin/env python3
"""Build the R6-C candidate-arm dataset from the frozen B1.2 collection.

This is the formal entry point for the candidate-arm schema.  It reuses the
shared extraction in ``build_r6c_dynamic_dataset.build_dataset`` and adds the
arm metadata / arm-level labels required by the multi-arm selector plan:

- every boundary row records the deployment inputs and, per current arm,
  ``arm_success`` (P(success | enter arm)) and ``arm_teacher_steps``
  (action/teacher cost) for ``CONTINUE_SOURCE`` and ``ENTER_PERSISTENT_OFT``;
- the dataset report embeds the arm schema, so future arms (recovery student,
  other VLA/planner, safe abort) only append labels without changing the rows;
- ``--atlas-root`` enables the R6-A source-parity hard gate inside the build.

Run only after ``audit_r6b1_source_parity.py`` passes on the full B1.2 output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_r6c_dynamic_dataset import build_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True,
                        help="collection output root(s) (contains suite_*); may be repeated to merge B1.2 with R6-C.1B re-collection")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="frozen r6b1_dynamic_boundary_protocol_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=0,
                        help="diagnostic: limit the number of trajectory groups")
    parser.add_argument("--atlas-root", type=Path, default=None,
                        help="R6-A reference root (policy_pair_atlas_v1); enables parity hard gate")
    parser.add_argument("--exclusions", type=Path, default=None,
                        help="frozen exclusion manifest of known nondeterministic groups")
    args = parser.parse_args()

    input_roots = args.input_root if len(args.input_root) > 1 else args.input_root[0]

    _, report = build_dataset(
        input_root=input_roots,
        protocol=args.protocol,
        output=args.output,
        history_window=args.history_window,
        max_groups=args.max_groups,
        atlas_root=args.atlas_root,
        exclusions=args.exclusions,
    )
    if report["status"] != "complete":
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
