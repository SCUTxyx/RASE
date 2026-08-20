#!/usr/bin/env bash
# PRE-C1.4-R3: Causal-Unit-Gated Paired Recovery Distillation
# Gate-JSON-driven pipeline. Phases 1+ unlocked only by prior gate files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
RASE_DIR="$ROOT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---- Phase toggles (default: Phase 0 only) ----
PHASE0_IDENTITY="${PHASE0_IDENTITY:-1}"
PHASE0_RESTORE="${PHASE0_RESTORE:-1}"
PHASE0_REWARD="${PHASE0_REWARD:-1}"
PHASE0_CAUSAL_UNIT="${PHASE0_CAUSAL_UNIT:-1}"
PHASE1_COLLECT="${PHASE1_COLLECT:-0}"
PHASE2_BUILD="${PHASE2_BUILD:-0}"
PHASE3_TRAIN="${PHASE3_TRAIN:-0}"
PHASE4_DEV="${PHASE4_DEV:-0}"
PHASE5_ONLINE="${PHASE5_ONLINE:-0}"
PHASE6_CONFIRM="${PHASE6_CONFIRM:-0}"

# ---- Paths ----
PROTOCOL_DIR="$RASE_DIR/runs/rase_pre_c1_4_r3_protocol"
COLLECT_DIR="$RASE_DIR/runs/rase_pre_c1_4_counterfactual"
DATASET_DIR="$RASE_DIR/runs/rase_pre_c1_4_dataset"
TRAIN_DIR="$RASE_DIR/runs/rase_pre_c1_4_train"
EVAL_DIR="$RASE_DIR/runs/rase_pre_c1_4_eval"
ONLINE_DIR="$RASE_DIR/runs/rase_pre_c1_4_online_awr"
CONFIRM_DIR="$RASE_DIR/runs/rase_pre_c1_4_confirmation"

IDENTITY_MANIFEST="$PROTOCOL_DIR/pre_c1_4_r3_identity_manifest.json"

# Gate files
GATE_RESTORE="$PROTOCOL_DIR/phase0_restore_pass.json"
GATE_REWARD="$PROTOCOL_DIR/phase0_reward_pass.json"
GATE_CAUSAL_UNIT="$PROTOCOL_DIR/phase0_causal_unit_pass.json"
GATE_DATA="$COLLECT_DIR/data_gate_pass.json"
GATE_DEV="$EVAL_DIR/dev_selection_frozen.json"
GATE_ONLINE="$ONLINE_DIR/online_trigger.json"
GATE_CONFIRM="$CONFIRM_DIR/confirmation_protocol_frozen.json"

# ---- Utility: require_gate ----
require_gate() {
    local gate_file="$1"
    local label="$2"
    if [[ ! -f "$gate_file" ]]; then
        echo "ERROR: $label gate file not found: $gate_file"
        echo "  Cannot proceed without this gate. Run the prior phase first."
        exit 1
    fi
    local passed
    passed=$( "$PY" -c "import json; g=json.load(open('$gate_file')); print(str(g.get('passed', False)).lower())" )
    local status
    status=$( "$PY" -c "import json; g=json.load(open('$gate_file')); print(g.get('status', ''))" )
    if [[ "$passed" == "false" && "$status" != "pending_live_run" ]]; then
        echo "ERROR: $label gate FAILED: $gate_file"
        cat "$gate_file"
        exit 1
    fi
    echo "  Gate: $label — passed=$passed status=$status"
}

# ---- Phase 0: Identity manifest ----
echo "========================================"
echo "PRE-C1.4-R3 Pipeline — $TIMESTAMP"
echo "========================================"

if [[ "$PHASE0_IDENTITY" == "1" ]]; then
    echo ""
    echo "=== PHASE 0: IDENTITY MANIFEST ==="
    mkdir -p "$PROTOCOL_DIR"
    "$PY" scripts/audit_pre_c1_4_identity_and_splits.py \
        --output-dir "$PROTOCOL_DIR"
    echo "  DONE: $IDENTITY_MANIFEST"
fi

# ---- Phase 0A: Restore parity ----
if [[ "$PHASE0_RESTORE" == "1" ]]; then
    echo ""
    echo "=== PHASE 0A: RESTORE & BRANCH PARITY ==="
    "$PY" scripts/audit_pre_c1_4_restore_and_branch_parity.py \
        --manifest "$IDENTITY_MANIFEST" \
        --output-dir "$PROTOCOL_DIR"
    echo "  DONE"
fi

# ---- Phase 0B: Reward certification ----
if [[ "$PHASE0_REWARD" == "1" ]]; then
    echo ""
    echo "=== PHASE 0B: DENSE REWARD CERTIFICATION ==="
    "$PY" scripts/certify_pre_c1_4_reward.py \
        --manifest "$IDENTITY_MANIFEST" \
        --output-dir "$PROTOCOL_DIR"
    echo "  DONE"
fi

# ---- Phase 0C: Causal-unit pilot ----
if [[ "$PHASE0_CAUSAL_UNIT" == "1" ]]; then
    echo ""
    echo "=== PHASE 0C: CAUSAL-UNIT PILOT ==="
    "$PY" scripts/run_pre_c1_4_causal_unit_pilot.py \
        --manifest "$IDENTITY_MANIFEST" \
        --output-dir "$PROTOCOL_DIR" \
        --dry-run
    echo "  DONE"
fi

# ---- Phase 0 gate check ----
echo ""
echo "=== PHASE 0 GATE SUMMARY ==="
H_STAR=0
if [[ -f "$GATE_CAUSAL_UNIT" ]]; then
    H_STAR=$( "$PY" -c "import json; g=json.load(open('$GATE_CAUSAL_UNIT')); print(g.get('H_star', 0))" )
    echo "  H_star = $H_STAR"
fi
echo ""

# ---- Phase 1: Paired collection (requires Phase 0) ----
if [[ "$PHASE1_COLLECT" == "1" ]]; then
    echo "=== PHASE 1: PAIRED COUNTERFACTUAL COLLECTION ==="
    require_gate "$GATE_CAUSAL_UNIT" "causal_unit"

    mkdir -p "$COLLECT_DIR"
    "$PY" scripts/collect_pre_c1_4_counterfactual_pairs.py \
        --manifest "$IDENTITY_MANIFEST" \
        --causal-unit-gate "$GATE_CAUSAL_UNIT" \
        --output-dir "$COLLECT_DIR" \
        --h-star "$H_STAR" \
        --dry-run

    # Verify labels
    if [[ -f "$COLLECT_DIR/labeled_pairs.jsonl" ]]; then
        "$PY" scripts/verify_pre_c1_4_pair_labels.py \
            --pairs "$COLLECT_DIR/labeled_pairs.jsonl" \
            --output-dir "$COLLECT_DIR"
    fi

    # Check data gate
    if [[ -f "$GATE_DATA" ]]; then
        require_gate "$GATE_DATA" "data"
    fi
    echo "  DONE"
fi

# ---- Phase 2: Dataset build ----
if [[ "$PHASE2_BUILD" == "1" ]]; then
    echo ""
    echo "=== PHASE 2: DATASET BUILD ==="
    require_gate "$GATE_DATA" "data"

    mkdir -p "$DATASET_DIR"
    "$PY" scripts/build_pre_c1_4_dataset.py \
        --verified-pairs "$COLLECT_DIR/verified_pairs.jsonl" \
        --output-dir "$DATASET_DIR"
    echo "  DONE"
fi

# ---- Phase 3: Training ----
if [[ "$PHASE3_TRAIN" == "1" ]]; then
    echo ""
    echo "=== PHASE 3: TRAINING (V0, V1, V2) ==="

    C11_ADAPTER="$RASE_DIR/runs/rase_pre_c1_1_lora_train_v1/adapter_final"
    DATASET_JSONL="$DATASET_DIR/train.jsonl"
    SPLITS_JSON="$DATASET_DIR/benchmark_splits.json"
    CONFIG="$RASE_DIR/configs/collect_pre_c0_deviation_pilot24.json"
    MAX_STEPS="${MAX_OPTIMIZER_STEPS:-500}"

    echo "  C1.1 adapter: $C11_ADAPTER"
    echo "  Dataset: $DATASET_JSONL"
    echo "  Max steps: $MAX_STEPS"

    for variant in V0 V1 V2; do
        echo "  --- Training $variant ---"
        output_dir="$TRAIN_DIR/${variant}_seed0"
        "$PY" scripts/train_pre_c1_4_recovery_lora.py \
            --variant "$variant" \
            --config "$CONFIG" \
            --dataset-jsonl "$DATASET_JSONL" \
            --splits-json "$SPLITS_JSON" \
            --c11-adapter-dir "$C11_ADAPTER" \
            --output-dir "$output_dir" \
            --training-seed 0 \
            --max-optimizer-steps "$MAX_STEPS" \
            --h-star "$H_STAR"
    done
    echo "  DONE"
fi

# ---- Phase 4: Development selection ----
if [[ "$PHASE4_DEV" == "1" ]]; then
    echo ""
    echo "=== PHASE 4: DEVELOPMENT SELECTION ==="

    mkdir -p "$EVAL_DIR"

    # Find all trained adapter dirs
    V0_ADAPTER="$TRAIN_DIR/V0_seed0/adapter_final"
    V1_ADAPTER="$TRAIN_DIR/V1_seed0/adapter_final"
    V2_ADAPTER="$TRAIN_DIR/V2_seed0/adapter_final"

    "$PY" scripts/evaluate_pre_c1_4_hierarchical.py \
        --manifest "$IDENTITY_MANIFEST" \
        --eval-type development \
        --v0-adapter-dir "$V0_ADAPTER" \
        --variant-adapter-dirs "$V1_ADAPTER" "$V2_ADAPTER" \
        --training-seeds 0 1 \
        --eval-seeds 5 \
        --output-dir "$EVAL_DIR"

    echo "  DONE"
fi

# ---- Phase 5: Online AWR (conditional) ----
if [[ "$PHASE5_ONLINE" == "1" ]]; then
    echo ""
    echo "=== PHASE 5: ONLINE AWR ==="

    # Check if dev gate passed
    require_gate "$GATE_DEV" "development"

    SELECTED_VARIANT=$( "$PY" -c "import json; g=json.load(open('$GATE_DEV')); print(g.get('details',{}).get('selected_variant','V1'))" )
    echo "  Selected variant: $SELECTED_VARIANT"

    mkdir -p "$ONLINE_DIR"

    "$PY" scripts/train_pre_c1_4_online_awr.py \
        --dev-gate "$GATE_DEV" \
        --phase0-dir "$PROTOCOL_DIR" \
        --selected-variant "$SELECTED_VARIANT" \
        --adapter-dir "$TRAIN_DIR/${SELECTED_VARIANT}_seed0/adapter_final" \
        --output-dir "$ONLINE_DIR" \
        --max-iterations 3

    echo "  DONE"
fi

# ---- Phase 6: Locked confirmation (conditional) ----
if [[ "$PHASE6_CONFIRM" == "1" ]]; then
    echo ""
    echo "=== PHASE 6: LOCKED CONFIRMATION ==="

    require_gate "$GATE_DEV" "development"

    SELECTED_VARIANT=$( "$PY" -c "import json; g=json.load(open('$GATE_DEV')); print(g.get('details',{}).get('selected_variant','V1'))" )
    echo "  Selected variant: $SELECTED_VARIANT"

    mkdir -p "$CONFIRM_DIR"

    V0_ADAPTER="$RASE_DIR/runs/rase_pre_c1_1_lora_train_v1/adapter_final"
    VARIANT_ADAPTER="$TRAIN_DIR/${SELECTED_VARIANT}_seed0/adapter_final"

    "$PY" scripts/freeze_pre_c1_4_confirmation.py \
        --manifest "$IDENTITY_MANIFEST" \
        --v0-adapter-dir "$V0_ADAPTER" \
        --variant-adapter-dir "$VARIANT_ADAPTER" \
        --variant-name "$SELECTED_VARIANT" \
        --output-dir "$CONFIRM_DIR" \
        --training-seeds 0 1 2 \
        --eval-seeds 5

    echo "  DONE"
fi

echo ""
echo "========================================"
echo "PRE-C1.4-R3 Pipeline complete."
echo "========================================"
echo "Phase 0 outputs: $PROTOCOL_DIR"
echo "Phase 1 outputs: $COLLECT_DIR"
echo "Phase 2 outputs: $DATASET_DIR"
echo "Phase 3 outputs: $TRAIN_DIR"
echo "Phase 4 outputs: $EVAL_DIR"
echo "Phase 5 outputs: $ONLINE_DIR"
echo "Phase 6 outputs: $CONFIRM_DIR"
