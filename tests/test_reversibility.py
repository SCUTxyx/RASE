from rase.collect.reversibility import (
    ReversibilityEvidence,
    ReversibilityLabel,
    annotate_state,
    classify_reversibility,
)


def test_default_is_reversible():
    assert classify_reversibility(None) is ReversibilityLabel.REVERSIBLE
    assert classify_reversibility({}) is ReversibilityLabel.REVERSIBLE


def test_task_irreversible_priority():
    label = classify_reversibility(
        ReversibilityEvidence(
            task_irreversible=True,
            non_grasp_contact=True,
            object_pos_delta_m=1.0,
        )
    )
    assert label is ReversibilityLabel.TASK_IRREVERSIBLE


def test_contact_irreversible_requires_contact_and_shift():
    assert (
        classify_reversibility(
            {"non_grasp_contact": True, "object_pos_delta_m": 0.01}
        )
        is ReversibilityLabel.REVERSIBLE
    )
    assert (
        classify_reversibility(
            {"non_grasp_contact": True, "object_pos_delta_m": 0.05}
        )
        is ReversibilityLabel.CONTACT_IRREVERSIBLE
    )
    assert (
        classify_reversibility({"non_grasp_contact": False, "object_pos_delta_m": 0.05})
        is ReversibilityLabel.REVERSIBLE
    )


def test_annotate_state_record():
    rec = annotate_state(
        {"state_key": "sp1_x"},
        {"task_irreversible": True},
    )
    assert rec["rho"] == "task-irreversible"
    assert rec["state_key"] == "sp1_x"
