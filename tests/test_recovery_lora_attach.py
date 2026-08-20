from __future__ import annotations

import torch

from rase.adapt.recovery_lora import (
    attach_recovery_lora,
    lora_trainable_parameter_count,
    set_adapter_enabled,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.action_out_proj = torch.nn.Linear(8, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_out_proj(self.v_proj(self.q_proj(x)))


def test_attach_recovery_lora_trainable_only_adapters():
    model = _Tiny()
    handle = attach_recovery_lora(
        model,
        rank=4,
        alpha=8,
        dropout=0.0,
        target_modules=["q_proj", "v_proj", "action_out_proj"],
    )
    counts = lora_trainable_parameter_count(handle.policy)
    assert counts["lora"] > 0
    assert counts["trainable"] == counts["lora"]
    set_adapter_enabled(handle, False)
    set_adapter_enabled(handle, True)
