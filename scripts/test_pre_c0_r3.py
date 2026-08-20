#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


audit_mod = load("r3_audit", "audit_pre_a3_operator_opportunity.py")
wm_mod = load("r3_wm", "train_counterfactual_latent_world_model.py")
from counterfactual_selector import ConservativeCounterfactualSelector


class OpportunityAuditTest(unittest.TestCase):
    def test_oracle_gap_and_harm(self):
        records = []
        # No fixed operator solves all states, while the oracle does.
        for state in range(12):
            for operator in ("CONTINUE", "OFT_H8", "OFT_H32", "OFT_PERSISTENT"):
                success = ((operator == "CONTINUE" and state < 3)
                           or (operator == "OFT_H8" and state in {3, 4, 5})
                           or (operator == "OFT_H32" and state in {6, 7, 8})
                           or (operator == "OFT_PERSISTENT" and state >= 9))
                records.append({"state_key": f"s{state}", "task_id": f"t{state % 3}",
                                "suite": "spatial", "operator": operator,
                                "success": success, "source": "synthetic"})
        report = audit_mod.audit(records, 12, .5, 2, 2, 1.0)
        self.assertEqual(report["status"], "ready")
        self.assertAlmostEqual(report["oracle_rate"], 1.0)
        self.assertGreaterEqual(len(report["task_diverse_winning_operators"]), 2)


class LatentWorldModelTest(unittest.TestCase):
    def test_arrays_and_conservative_selector(self):
        rows = []
        for i in range(8):
            rows.append({"state_key": f"s{i}", "task_id": f"t{i % 3}",
                         "latent": [float(i), 1.0], "action": [0.1, -0.2],
                         "proprio": [0.3], "next_latent": [float(i) + .1, .9],
                         "operator_success": {"CONTINUE": int(i % 2 == 0),
                                              "OFT_H8": int(i % 2 == 1)}})
        data, _ = wm_mod.arrays(rows, ["CONTINUE", "OFT_H8"])
        self.assertEqual(data["x"].shape, (8, 5))
        probs = np.asarray([[.8, .2] if i % 2 == 0 else [.2, .8] for i in range(8)])
        std = np.zeros_like(probs)
        result = wm_mod.selector_metrics(rows, ["CONTINUE", "OFT_H8"], probs, .1, std)
        self.assertEqual(result["rescue"], 4)
        self.assertEqual(result["harm"], 0)

    def test_checkpoint_roundtrip(self):
        rows = [{"state_key": f"s{i}", "task_id": f"t{i % 3}",
                 "latent": [float(i) / 10, 1.0], "action": [0.1, -0.2],
                 "proprio": [0.3], "next_latent": [float(i) / 10 + .1, .9],
                 "operator_success": {"CONTINUE": int(i % 2 == 0),
                                      "OFT_H8": int(i % 2 == 1)}} for i in range(12)]
        operators = ["CONTINUE", "OFT_H8"]
        with tempfile.TemporaryDirectory() as tmp:
            for member in range(2):
                model, config, normalizers = wm_mod.fit_full(
                    rows, operators, device="cpu", seed=member, epochs=2, lr=1e-3,
                    hidden_dim=8, dynamics_weight=1.0)
                import torch
                torch.save({"schema_version": "rase-counterfactual-latent-wm-checkpoint/v1",
                            "state_dict": model.state_dict(), "config": config,
                            "normalizers": normalizers, "operators": operators,
                            "selector_margin": .1}, Path(tmp) / f"member_{member:02d}.pt")
            selector = ConservativeCounterfactualSelector(tmp)
            result = selector.predict(rows[0]["latent"], rows[0]["action"], rows[0]["proprio"])
            self.assertIn(result["operator"], operators)
            self.assertIn("latent_delta_disagreement", result)

    def test_training_cli_writes_oof_and_deployment_ensemble(self):
        rows = []
        for i in range(18):
            risk = i % 2
            rows.append({"state_key": f"s{i}", "task_id": f"t{i % 3}",
                         "latent": [float(i) / 20, float(risk)],
                         "action": [float(risk), -0.2], "proprio": [0.3],
                         "next_latent": [float(i) / 20 + .1, float(risk) * .9],
                         "operator_success": {"CONTINUE": 1 - risk, "OFT_H8": risk}})
        with tempfile.TemporaryDirectory() as tmp:
            dataset, output = Path(tmp) / "data.jsonl", Path(tmp) / "out"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = subprocess.run([
                sys.executable, str(HERE / "train_counterfactual_latent_world_model.py"),
                "--dataset", str(dataset), "--output-dir", str(output),
                "--ensemble-size", "2", "--epochs", "2", "--hidden-dim", "8",
                "--patience", "2", "--device", "cpu"], capture_output=True, text=True)
            self.assertIn(result.returncode, {0, 2}, result.stderr)
            self.assertTrue((output / "report.json").exists())
            self.assertEqual(len(list(output.glob("member_*.pt"))), 2)
            self.assertEqual(len((output / "oof_predictions.jsonl").read_text().splitlines()), 18)


if __name__ == "__main__":
    unittest.main()
