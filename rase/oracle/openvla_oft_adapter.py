"""OpenVLA-OFT adapter for the RASE oracle server (oft conda env only)."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rase.oracle.wire_schema import (
    PREPROCESS_REVISION,
    WIRE_SCHEMA_VERSION,
    proprio_to_policy_state,
    validate_predict_inputs,
)

SUITE_TO_UNNORM = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


def _checkpoint_bundle_sha256(checkpoint: Path) -> str:
    files = []
    for pattern in (
        "model*.safetensors",
        "action_head*.pt",
        "proprio_projector*.pt",
        "dataset_statistics.json",
        "config.json",
    ):
        files.extend(sorted(checkpoint.glob(pattern)))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _AdapterConfig:
    pretrained_checkpoint: str
    unnorm_key: str
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    center_crop: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_open_loop_steps: int = 8


class OpenVLAOFTAdapter:
    """Thin wrapper around official ``get_vla_action`` + LIBERO action postprocess."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        suite: str,
        unnorm_key: str | None = None,
        max_batch: int = 2,
        device: str = "cuda:0",
        images_already_flipped: bool = False,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(self.checkpoint)
        self.suite = suite
        self.unnorm_key = unnorm_key or SUITE_TO_UNNORM.get(suite)
        if not self.unnorm_key:
            raise ValueError(f"cannot resolve unnorm_key for suite {suite!r}")
        self.max_batch = int(max_batch)
        self.device = device
        self.images_already_flipped = bool(images_already_flipped)
        self.bundle_sha256 = _checkpoint_bundle_sha256(self.checkpoint)
        self._cfg = _AdapterConfig(
            pretrained_checkpoint=str(self.checkpoint),
            unnorm_key=self.unnorm_key,
        )
        self._load()

    def _load(self) -> None:
        # Official helpers live in the OpenVLA-OFT checkout / PYTHONPATH.
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
        )

        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        self.vla = get_vla(self._cfg)
        if self.unnorm_key not in self.vla.norm_stats:
            # Match run_libero_eval.check_unnorm_key fallback.
            alt = f"{self.suite}_no_noops"
            if alt in self.vla.norm_stats:
                self.unnorm_key = alt
                self._cfg.unnorm_key = alt
            else:
                raise KeyError(
                    f"unnorm_key {self.unnorm_key!r} missing; "
                    f"have {list(self.vla.norm_stats)}"
                )
        self.processor = get_processor(self._cfg)
        self.action_head = get_action_head(self._cfg, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(
            self._cfg, self.vla.llm_dim, proprio_dim=8
        )

    def model_info(self) -> Mapping[str, Any]:
        return {
            "name": "openvla-oft",
            "suite": self.suite,
            "unnorm_key": self.unnorm_key,
            "action_dim": 7,
            "chunk_size": 8,
            "checkpoint": str(self.checkpoint),
            "checkpoint_bundle_sha256": self.bundle_sha256,
            "precision": "bfloat16",
            "preprocessing": PREPROCESS_REVISION,
            "wire_schema_version": WIRE_SCHEMA_VERSION,
            "num_images_in_input": 2,
            "use_proprio": True,
            "use_l1_regression": True,
            "center_crop": True,
            "deterministic": True,
            "max_batch": self.max_batch,
            "device": self.device,
        }

    def _maybe_flip(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if self.images_already_flipped:
            return array
        return array[::-1, ::-1]

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        from experiments.robot.robot_utils import (
            invert_gripper_action,
            normalize_gripper_action,
        )

        processed = normalize_gripper_action(np.asarray(action), binarize=True)
        return invert_gripper_action(processed).astype(np.float32)

    def predict(
        self, arrays: Mapping[str, np.ndarray], payload: Mapping[str, Any]
    ) -> Mapping[str, np.ndarray]:
        from experiments.robot.openvla_utils import get_vla_action

        payload = dict(payload)
        payload.setdefault("max_batch", self.max_batch)
        batch, instructions, proprio_format = validate_predict_inputs(arrays, payload)
        return_mode = str(payload.get("return_mode", "chunk"))
        if return_mode not in {"step", "chunk"}:
            raise ValueError("return_mode must be 'step' or 'chunk'")
        flip = payload.get("images_already_flipped")
        already_flipped = self.images_already_flipped if flip is None else bool(flip)

        chunks = []
        for index in range(batch):
            agentview = np.asarray(arrays["agentview"][index])
            wrist = np.asarray(arrays["wrist"][index])
            if not already_flipped:
                agentview = agentview[::-1, ::-1]
                wrist = wrist[::-1, ::-1]
            state = proprio_to_policy_state(
                arrays["proprio"][index], proprio_format=proprio_format
            )
            obs = {
                "full_image": agentview,
                "wrist_image": wrist,
                "state": state,
            }
            actions = get_vla_action(
                self._cfg,
                self.vla,
                self.processor,
                obs,
                instructions[index],
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=None,
                use_film=False,
            )
            env_actions = np.stack(
                [self._process_action(step) for step in actions], axis=0
            )
            chunks.append(env_actions.astype(np.float32))
        stacked = np.stack(chunks, axis=0)  # [B, 8, 7]
        if return_mode == "step":
            return {"actions": stacked[:, 0, :]}
        return {"actions": stacked}


def create_adapter() -> OpenVLAOFTAdapter:
    """Zero-argument factory for ``python -m rase.oracle.server --adapter ...``."""
    checkpoint = os.environ.get("RASE_OFT_CHECKPOINT")
    suite = os.environ.get("RASE_OFT_SUITE")
    if not checkpoint or not suite:
        raise RuntimeError(
            "set RASE_OFT_CHECKPOINT and RASE_OFT_SUITE before starting the server"
        )
    unnorm = os.environ.get("RASE_OFT_UNNORM_KEY")
    max_batch = int(os.environ.get("RASE_OFT_MAX_BATCH", "2"))
    device = os.environ.get("RASE_OFT_DEVICE", "cuda:0")
    flipped = os.environ.get("RASE_OFT_IMAGES_FLIPPED", "0") in {"1", "true", "True"}
    return OpenVLAOFTAdapter(
        checkpoint=checkpoint,
        suite=suite,
        unnorm_key=unnorm,
        max_batch=max_batch,
        device=device,
        images_already_flipped=flipped,
    )
