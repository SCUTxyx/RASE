"""Physical reversibility labels ρ(s) (Guide1 §5.3).

Thin interface + rule placeholders. Full task-specific predicates are filled
once Set B/C video QC is available; until then callers can supply explicit
signals or accept the conservative default ``reversible``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReversibilityLabel(str, Enum):
    REVERSIBLE = "reversible"
    CONTACT_IRREVERSIBLE = "contact-irreversible"
    TASK_IRREVERSIBLE = "task-irreversible"


# Default contact thresholds from Guide1 §5.3 (meters / radians).
DEFAULT_CONTACT_POS_M = 0.03
DEFAULT_CONTACT_ORI_RAD = 0.2617993877991494  # ~15 degrees


@dataclass(frozen=True)
class ReversibilityEvidence:
    """Optional signals used by the automatic annotator."""

    task_irreversible: bool = False
    non_grasp_contact: bool = False
    object_pos_delta_m: float | None = None
    object_ori_delta_rad: float | None = None
    notes: str | None = None


def classify_reversibility(
    evidence: ReversibilityEvidence | Mapping[str, Any] | None = None,
    *,
    contact_pos_m: float = DEFAULT_CONTACT_POS_M,
    contact_ori_rad: float = DEFAULT_CONTACT_ORI_RAD,
) -> ReversibilityLabel:
    """Apply Guide1 §5.3 priority rules.

    Priority: task-irreversible > contact-irreversible > reversible.
    Missing evidence defaults to ``reversible`` (conservative for Set C yield;
    does not invent irreversibility without signals).
    """
    if evidence is None:
        return ReversibilityLabel.REVERSIBLE
    if isinstance(evidence, Mapping):
        ev = ReversibilityEvidence(
            task_irreversible=bool(evidence.get("task_irreversible", False)),
            non_grasp_contact=bool(evidence.get("non_grasp_contact", False)),
            object_pos_delta_m=(
                float(evidence["object_pos_delta_m"])
                if evidence.get("object_pos_delta_m") is not None
                else None
            ),
            object_ori_delta_rad=(
                float(evidence["object_ori_delta_rad"])
                if evidence.get("object_ori_delta_rad") is not None
                else None
            ),
            notes=(
                str(evidence["notes"]) if evidence.get("notes") is not None else None
            ),
        )
    else:
        ev = evidence

    if ev.task_irreversible:
        return ReversibilityLabel.TASK_IRREVERSIBLE

    contact_shift = False
    if ev.object_pos_delta_m is not None and ev.object_pos_delta_m > contact_pos_m:
        contact_shift = True
    if (
        ev.object_ori_delta_rad is not None
        and ev.object_ori_delta_rad > contact_ori_rad
    ):
        contact_shift = True
    if ev.non_grasp_contact and contact_shift:
        return ReversibilityLabel.CONTACT_IRREVERSIBLE

    return ReversibilityLabel.REVERSIBLE


def annotate_state(
    meta: Mapping[str, Any] | None = None,
    evidence: ReversibilityEvidence | Mapping[str, Any] | None = None,
    **thresholds: float,
) -> dict[str, Any]:
    """Return a JSON-serializable ρ(s) annotation record."""
    label = classify_reversibility(evidence, **thresholds)
    return {
        "rho": label.value,
        "label": label.value,
        "state_key": None if meta is None else meta.get("state_key"),
        "evidence": None
        if evidence is None
        else (
            dict(evidence)
            if isinstance(evidence, Mapping)
            else {
                "task_irreversible": evidence.task_irreversible,
                "non_grasp_contact": evidence.non_grasp_contact,
                "object_pos_delta_m": evidence.object_pos_delta_m,
                "object_ori_delta_rad": evidence.object_ori_delta_rad,
                "notes": evidence.notes,
            }
        ),
    }
