"""Task-conditioned H=8 residual chunk model used by E3-B DAgger."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


HORIZON = 8
ACTION_DIM = 7
STATE_DIM = 8 + HORIZON * ACTION_DIM + 8 * 23 + 64
IMAGE_SIZE = 24


def state_features(
    proprio: np.ndarray,
    source_chunk: np.ndarray,
    history: np.ndarray,
    language_hash: np.ndarray,
) -> np.ndarray:
    values = [
        np.asarray(proprio, dtype=np.float32).reshape(8),
        np.asarray(source_chunk, dtype=np.float32).reshape(HORIZON * ACTION_DIM),
        np.asarray(history, dtype=np.float32).reshape(8 * 23),
        np.asarray(language_hash, dtype=np.float32).reshape(64),
    ]
    result = np.concatenate(values).astype(np.float32)
    if result.shape != (STATE_DIM,):
        raise ValueError(result.shape)
    return result


def vision_features(agentview: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    images = []
    for value in (agentview, wrist):
        image = np.asarray(value, dtype=np.float32)
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {image.shape}")
        images.append(image.reshape(-1) / 127.5 - 1.0)
    return np.concatenate(images).astype(np.float32)


def make_network():
    import torch.nn as nn

    class ChunkResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(), nn.LayerNorm(128)
            )
            self.vision_encoder = nn.Sequential(
                nn.Linear(2 * IMAGE_SIZE * IMAGE_SIZE * 3, 96),
                nn.ReLU(),
                nn.LayerNorm(96),
            )
            self.trunk = nn.Sequential(nn.Linear(224, 192), nn.ReLU())
            self.delta_head = nn.Linear(192, HORIZON * ACTION_DIM)
            self.gate_head = nn.Linear(192, 1)

        def forward(self, state, vision):
            hidden = self.trunk(
                __import__("torch").cat(
                    [self.state_encoder(state), self.vision_encoder(vision)], dim=-1
                )
            )
            return self.delta_head(hidden), self.gate_head(hidden).squeeze(-1)

    return ChunkResidualNet()


def load_ensemble(path: str, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != "rase-e3b-chunk-residual/v1":
        raise ValueError("unsupported E3-B chunk residual checkpoint")
    models = []
    for state_dict in payload["state_dicts"]:
        model = make_network().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return {**payload, "models": models, "device": device}


def predict_ensemble(
    ensemble: Mapping[str, Any],
    *,
    proprio: np.ndarray,
    source_chunk: np.ndarray,
    history: np.ndarray,
    language_hash: np.ndarray,
    agentview: np.ndarray,
    wrist: np.ndarray,
) -> tuple[np.ndarray, float]:
    import torch

    state = state_features(proprio, source_chunk, history, language_hash)
    state = (state - np.asarray(ensemble["state_mean"], dtype=np.float32)) / np.asarray(
        ensemble["state_std"], dtype=np.float32
    )
    vision = vision_features(agentview, wrist)
    state_tensor = torch.from_numpy(state[None]).to(ensemble["device"])
    vision_tensor = torch.from_numpy(vision[None]).to(ensemble["device"])
    deltas = []
    gates = []
    with torch.no_grad():
        for model in ensemble["models"]:
            delta, gate = model(state_tensor, vision_tensor)
            deltas.append(delta.cpu().numpy()[0].reshape(HORIZON, ACTION_DIM))
            gates.append(float(torch.sigmoid(gate).cpu().numpy()[0]))
    return np.mean(deltas, axis=0).astype(np.float32), float(np.mean(gates))
