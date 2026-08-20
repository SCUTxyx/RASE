#!/usr/bin/env bash
# Continue PRE-C1.2 after partial E0: finish missing Long suite, then E1→DAgger→E3→E4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SKIP_E0=0
# Re-score existing successor with robust interface rule, then run full pipeline
# which skips already-complete Spatial/Object/Goal suites.
exec bash scripts/run_pre_c1_2_full_pipeline.sh
