from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.filter_selector_dataset import filter_dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/filter_selector_dataset.py"


def _row(key: str, suite: str, cohort: str) -> dict[str, object]:
    return {
        "state_key": key,
        "suite": suite,
        "cohort": cohort,
        "provenance": {"frozen": True, "source": "W9C"},
        "arms": {"continue_smol": {"observed": True}},
    }


def test_filter_preserves_selected_rows_and_writes_hash_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    lines = [
        json.dumps(_row("object-clean", "Object", "clean_control"), separators=(",", ":")),
        json.dumps(_row("goal-clean", "Goal", "clean_control"), sort_keys=True),
        json.dumps(_row("spatial-clean", "Spatial", "clean_control"), ensure_ascii=False),
        json.dumps(_row("object-failure", "Object", "failure_challenge")),
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = tmp_path / "filtered.jsonl"
    manifest_path = tmp_path / "manifest.json"

    manifest = filter_dataset(
        source,
        output,
        manifest_path,
        suites={"Object", "Spatial"},
        cohort="clean_control",
    )

    expected = (lines[0] + "\n" + lines[2] + "\n").encode()
    assert output.read_bytes() == expected
    assert manifest["rows_preserved_verbatim"] is True
    assert manifest["source"]["n_rows"] == 4
    assert manifest["output"]["n_rows"] == 2
    assert manifest["output"]["suite_counts"] == {"Object": 1, "Spatial": 1}
    assert manifest["output"]["cohort_counts"] == {"clean_control": 2}
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["output"]["sha256"] == hashlib.sha256(expected).hexdigest()
    assert json.loads(manifest_path.read_text()) == manifest


def test_filter_rejects_duplicate_state_keys_before_filtering(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    source.write_text(
        json.dumps(_row("same", "Goal", "clean_control"))
        + "\n"
        + json.dumps(_row("same", "Object", "clean_control"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate state_key"):
        filter_dataset(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "manifest.json",
            suites={"Object"},
            cohort="clean_control",
        )


def test_filter_cli_supports_repeatable_suite(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(_row(key, suite, "clean_control")) + "\n"
            for key, suite in (("a", "Object"), ("b", "Spatial"), ("c", "Goal"))
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(source),
            "--suite",
            "Object",
            "--suite",
            "Spatial",
            "--cohort",
            "clean_control",
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line)["state_key"] for line in output.read_text().splitlines()] == ["a", "b"]


def test_filter_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    original = json.dumps(_row("a", "Object", "clean_control")) + "\n"
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="must be distinct paths"):
        filter_dataset(
            source, source, tmp_path / "manifest.json",
            suites={"Object"}, cohort="clean_control",
        )
    assert source.read_text(encoding="utf-8") == original
