from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_builder_strips_source_outcomes(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    states = {}
    for root in range(2):
        key = f"state-{root}"
        path = f"task/episode-{root}/000000"
        state_dir = pool / path
        state_dir.mkdir(parents=True)
        (state_dir / "meta.json").write_text(json.dumps({
            "task_id": "libero_spatial_000000",
            "episode_id": f"episode-{root}",
            "init_state_id": root,
            "seed": 100 + root,
            "suite": "Spatial",
            "step": 0,
            "episode_outcome": "failure",
        }))
        states[key] = {
            "path": path, "step": 0, "outcome": "failure",
            "bundle_sha256": f"hash-{root}",
        }
    (pool / "manifest.json").write_text(json.dumps({"states": states}))
    output = tmp_path / "catalog.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "build_rase_vnext_root_catalog.py"),
        "--pool", str(pool), "--output", str(output),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    catalog = json.loads(output.read_text())
    assert catalog["n_records"] == 2
    assert all("outcome" not in json.dumps(row).lower() for row in catalog["records"])
    assert {row["init_state_id"] for row in catalog["records"]} == {0, 1}
