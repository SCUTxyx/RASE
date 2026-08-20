"""Recovery LoRA attach/detach helpers for SmolVLA action expert."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "action_in_proj",
    "action_out_proj",
)


@dataclass
class RecoveryLoraHandle:
    """Tracks base policy and optional PEFT wrapper."""

    base_policy: Any
    peft_policy: Any | None
    enabled: bool = False

    @property
    def policy(self) -> Any:
        return self.peft_policy if self.peft_policy is not None else self.base_policy


def _freeze_non_lora(policy: Any) -> None:
    for name, param in policy.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def enable_full_action_expert_training(
    policy: Any,
    *,
    freeze_vision: bool = True,
    freeze_vlm_backbone: bool = True,
) -> list[str]:
    """Phase-4 C4-D helper: unfreeze action-expert-ish modules, freeze the rest.

    Returns names of parameters set ``requires_grad=True``. Optimizer config must
    come from the preregistered protocol block (do not retune from gate results).
    """

    trainable: list[str] = []
    allow_substrings = (
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp",
        "state_proj",
        "lm_expert",
        "expert",
    )
    block_substrings = ("vision", "vision_model", "connector") if freeze_vision else ()
    for name, param in policy.named_parameters():
        lower = name.lower()
        if freeze_vlm_backbone and any(token in lower for token in block_substrings):
            param.requires_grad = False
            continue
        if any(token in lower for token in allow_substrings):
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
    return trainable


def attach_recovery_lora(
    policy: Any,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
) -> RecoveryLoraHandle:
    """Inject PEFT LoRA into SmolVLA; freeze non-LoRA weights."""

    modules = list(target_modules or DEFAULT_TARGET_MODULES)
    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=modules,
        bias="none",
        task_type=None,
    )
    # Prefer wrapping the inner `model` when present so VLM backbone modules
    # outside lm_expert can still be matched by short module-name suffixes.
    root = policy.model if hasattr(policy, "model") else policy
    peft_root = get_peft_model(root, config)
    if hasattr(policy, "model"):
        policy.model = peft_root
        peft_policy = policy
    else:
        peft_policy = peft_root
    _freeze_non_lora(peft_policy)
    return RecoveryLoraHandle(base_policy=policy, peft_policy=peft_policy, enabled=True)


def set_adapter_enabled(handle: RecoveryLoraHandle, enabled: bool) -> None:
    """Enable or disable LoRA adapters without unloading weights."""

    policy = handle.policy
    handle.enabled = bool(enabled)
    # PeftModel API
    root = policy.model if hasattr(policy, "model") else policy
    if not isinstance(root, PeftModel):
        return
    if enabled:
        if hasattr(root, "enable_adapter_layers"):
            root.enable_adapter_layers()
        if hasattr(root, "set_adapter"):
            try:
                root.set_adapter("default")
            except Exception:
                pass
    else:
        if hasattr(root, "disable_adapter_layers"):
            root.disable_adapter_layers()


def lora_trainable_parameter_count(policy: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    lora_only = 0
    for name, param in policy.named_parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
        if "lora_" in name:
            lora_only += n
    return {"total": total, "trainable": trainable, "lora": lora_only}


def save_lora_only(handle: RecoveryLoraHandle, path: str) -> None:
    root = handle.policy.model if hasattr(handle.policy, "model") else handle.policy
    if isinstance(root, PeftModel):
        root.save_pretrained(path)
    else:
        torch.save(handle.policy.state_dict(), path)


def load_lora_onto_policy(policy: Any, adapter_dir: str) -> RecoveryLoraHandle:
    root = policy.model if hasattr(policy, "model") else policy
    peft_root = PeftModel.from_pretrained(root, adapter_dir)
    if hasattr(policy, "model"):
        policy.model = peft_root
        wrapped = policy
    else:
        wrapped = peft_root
    _freeze_non_lora(wrapped)
    return RecoveryLoraHandle(base_policy=policy, peft_policy=wrapped, enabled=True)
