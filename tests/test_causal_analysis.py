import json

from rase.collect.causal_analysis import (
    attach_pool_meta,
    build_dual_yield_tables,
    build_yield_table,
    yield_table_markdown,
)
from rase.collect.state_pool import StatePool


def _write_state(root, key, *, suite, dim, level, step=10):
    path = root / "states" / key
    path.mkdir(parents=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "perturb_dim": dim,
                "level": level,
                "task_id": 0,
                "seed": 0,
                "instruction": "x",
                "step": step,
            }
        ),
        encoding="utf-8",
    )
    return f"states/{key}"


def test_build_yield_table_by_cell(tmp_path):
    root = tmp_path / "pool"
    states = {
        "sp1_a": {"path": _write_state(root, "sp1_a", suite="Spatial", dim="camera", level=4)},
        "sp1_b": {"path": _write_state(root, "sp1_b", suite="Spatial", dim="camera", level=4)},
        "sp1_c": {"path": _write_state(root, "sp1_c", suite="Object", dim="robot", level=3)},
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "per_state": [
            {"state_key": "sp1_a", "recoverable_smolvla": False, "recoverable_oft": True},
            {"state_key": "sp1_b", "recoverable_smolvla": True, "recoverable_oft": True},
            {"state_key": "sp1_c", "recoverable_smolvla": False, "recoverable_oft": False},
        ]
    }
    table = build_yield_table(summary, StatePool(root), ngc_oracle="smolvla")
    assert table["n_states"] == 3
    assert table["n_ngc"] == 2
    assert abs(table["yield"] - 2 / 3) < 1e-12
    by_cell = {(r["suite"], r["dim"], r["level"]): r for r in table["cells"]}
    assert by_cell[("Spatial", "camera", 4)]["ngc"] == 1
    assert by_cell[("Spatial", "camera", 4)]["n"] == 2
    assert by_cell[("Object", "robot", 3)]["ngc"] == 1
    assert table["outcome"] == "smolvla_ngc"
    assert table["warnings"]
    md = yield_table_markdown(table)
    assert "Spatial" in md and "camera" in md

    tables = build_dual_yield_tables(summary, StatePool(root))
    oft = tables["oft_portfolio_unrecovered"]
    assert oft["n_ngc"] == 1
    assert oft["outcome"] == "oft_portfolio_unrecovered"
    assert any("non-independent" in warning for warning in oft["warnings"])

    enriched = attach_pool_meta(summary, StatePool(root))
    assert enriched[0]["t0"] == 10


def test_unknown_new_semantics_are_excluded(tmp_path):
    root = tmp_path / "pool"
    states = {
        "a": {"path": _write_state(root, "a", suite="Spatial", dim="camera", level=4)},
        "b": {"path": _write_state(root, "b", suite="Spatial", dim="camera", level=4)},
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": "ngc-state-pool-manifest/v1",
                "schema_version": "ngc-state-pool/v1",
                "states": states,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "per_state": [
            {
                "state_key": "a",
                "set_label_smolvla": "uncertain",
                "recoverable_smolvla": None,
                "recoverable_oft": None,
            },
            {
                "state_key": "b",
                "set_label_smolvla": "C",
                "recoverable_smolvla": False,
                "recoverable_oft": False,
            },
        ]
    }

    smol = build_yield_table(summary, StatePool(root), ngc_oracle="smolvla")
    oft = build_yield_table(summary, StatePool(root), ngc_oracle="oft")
    assert smol["n_states"] == 1 and smol["n_ngc"] == 1
    assert oft["n_states"] == 1 and oft["n_ngc"] == 1
    assert smol["excluded_unknown"] == 1
    assert oft["excluded_unknown"] == 1
