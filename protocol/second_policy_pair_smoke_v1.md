# Second policy-pair smoke protocol v1

**Purpose:** strengthen the benchmark claim that recoverability is
policy-relative, not SmolVLA↔OFT specific.

## Design

- Reuse a frozen 16-state episode-disjoint subset from W7/W8 or PRE-A3 train
- Arms:
  1. frozen backbone A continue
  2. frozen backbone B continue
  3. A→B direct escalation
  4. B→A direct escalation (optional asymmetry check)
- No learned selector
- Report paired McNemar + task-cluster bootstrap

## Candidate backbones

Priority order:

1. A robustified OpenVLA / OFT variant already available locally
2. Another publicly released LIBERO VLA checkpoint with reproducible eval

## Success criterion for inclusion in main text

- At least one direction shows significant policy-relative recovery asymmetry
  on held-out episodes
- Otherwise report as appendix negative / limited generalization

## Status

Scaffold only as of 2026-08-04. Execution waits for PRE-A3 collection/smoke
bandwidth; does not reopen killed selector method branch.
