"""Runtime wrapper for the conservative PRE-C0-R3 operator selector."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from train_counterfactual_latent_world_model import CounterfactualLatentWM


class ConservativeCounterfactualSelector:
    def __init__(self, checkpoint_dir: str | Path, *, device: str = "cpu",
                 uncertainty_z: float = 1.64, costs: Mapping[str, float] | None = None,
                 margin: float | None = None):
        paths = sorted(Path(checkpoint_dir).glob("member_*.pt"))
        if len(paths) < 2:
            raise ValueError(f"Need at least two ensemble members in {checkpoint_dir}")
        self.device = device
        self.uncertainty_z = float(uncertainty_z)
        self.costs = dict(costs or {})
        self.models, self.normalizers = [], []
        operators = None
        checkpoint_margin = None
        for path in paths:
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload.get("schema_version") != "rase-counterfactual-latent-wm-checkpoint/v1":
                raise ValueError(f"Unsupported checkpoint schema: {path}")
            if operators is not None and operators != payload["operators"]:
                raise ValueError("Ensemble members have different operator registries")
            operators = list(payload["operators"])
            checkpoint_margin = float(payload["selector_margin"])
            model = CounterfactualLatentWM(**payload["config"]).to(device)
            model.load_state_dict(payload["state_dict"]); model.eval()
            self.models.append(model); self.normalizers.append(payload["normalizers"])
        self.operators = operators
        if "CONTINUE" not in self.operators:
            raise ValueError("Checkpoint has no strict CONTINUE operator")
        self.continue_idx = self.operators.index("CONTINUE")
        self.margin = checkpoint_margin if margin is None else float(margin)

    @staticmethod
    def _normalize(value: np.ndarray, state: dict | None) -> np.ndarray:
        if state is None:
            return value
        mean = np.asarray(state["mean"], np.float32)
        std = np.asarray(state["std"], np.float32)
        if value.size != mean.size:
            raise ValueError(f"feature size {value.size} != checkpoint size {mean.size}")
        return (value - mean) / std

    @torch.no_grad()
    def predict(self, latent: Sequence[float], action: Sequence[float],
                proprio: Sequence[float] = ()) -> dict:
        latent = np.asarray(latent, np.float32).reshape(-1)
        action = np.asarray(action, np.float32).reshape(-1)
        proprio = np.asarray(proprio, np.float32).reshape(-1)
        op_predictions, risk_predictions, delta_predictions = [], [], []
        for model, norms in zip(self.models, self.normalizers):
            x = np.concatenate([self._normalize(latent, norms["latent"]),
                                self._normalize(action, norms["action"]),
                                self._normalize(proprio, norms["proprio"])])
            tensor = torch.from_numpy(x).to(self.device).unsqueeze(0)
            delta, risk, operators = model(tensor)
            delta_predictions.append(delta.squeeze(0).cpu().numpy())
            risk_predictions.append(torch.sigmoid(risk).item())
            op_predictions.append(torch.sigmoid(operators).squeeze(0).cpu().numpy())
        op_predictions = np.stack(op_predictions)
        mean, std = op_predictions.mean(0), op_predictions.std(0)
        lcb = mean - self.uncertainty_z * std
        cost = np.asarray([float(self.costs.get(op, 0.0)) for op in self.operators])
        utility = lcb - cost
        advantage = utility - utility[self.continue_idx]
        best = int(np.argmax(advantage))
        selected = (best if best != self.continue_idx and advantage[best] >= self.margin
                    else self.continue_idx)
        return {
            "operator": self.operators[selected],
            "abstained": selected == self.continue_idx,
            "risk_mean": float(np.mean(risk_predictions)),
            "risk_std": float(np.std(risk_predictions)),
            "operator_probability_mean": dict(zip(self.operators, map(float, mean))),
            "operator_probability_std": dict(zip(self.operators, map(float, std))),
            "operator_utility_lcb": dict(zip(self.operators, map(float, utility))),
            "selected_advantage": float(advantage[selected]),
            "latent_delta_disagreement": float(np.mean(np.std(delta_predictions, axis=0))),
        }
