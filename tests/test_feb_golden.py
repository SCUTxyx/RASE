"""Golden test for protocol.eval_feb (fixed input → fixed metrics)."""

from protocol.eval_feb import evaluate_feb, metrics_nearly_equal

GOLDEN_EPISODES = [
    # Set C: candidate-internal forced execution, fail
    {
        "set_label": "C",
        "chose_from_candidates": True,
        "task_success": False,
        "oracle_recoverable": False,
        "used_fallback": False,
    },
    # Set C: abstain/fallback path, fail
    {
        "set_label": "C",
        "chose_from_candidates": False,
        "task_success": False,
        "oracle_recoverable": False,
        "used_fallback": True,
    },
    # Set C: fallback recovers
    {
        "set_label": "C",
        "chose_from_candidates": False,
        "task_success": True,
        "oracle_recoverable": False,
        "used_fallback": True,
    },
    # Set A: recoverable, fallback breaks success
    {
        "set_label": "A",
        "chose_from_candidates": False,
        "task_success": False,
        "oracle_recoverable": True,
        "used_fallback": True,
    },
    # Set B: recoverable, execute candidate, success
    {
        "set_label": "B",
        "chose_from_candidates": True,
        "task_success": True,
        "oracle_recoverable": True,
        "used_fallback": False,
    },
    # Set B: recoverable, fallback still succeeds
    {
        "set_label": "B",
        "chose_from_candidates": False,
        "task_success": True,
        "oracle_recoverable": True,
        "used_fallback": True,
    },
]


GOLDEN_METRICS = {
    "protocol_version": "feb-protocol/v2",
    "n_episodes": 6,
    "n_set_c": 3,
    "n_set_ab": 3,
    "feb": 1 / 3,
    "net_success": 1 / 3,
    "broken_success": 1 / 6,
    "clean_regret": 1 / 3,
    "feb_wilson_95": {
        "lower": 0.06149194472039621,
        "upper": 0.7923403991979523,
        "hits": 1,
        "n": 3,
    },
    "net_success_wilson_95": {
        "lower": 0.06149194472039621,
        "upper": 0.7923403991979523,
        "hits": 1,
        "n": 3,
    },
    "broken_success_wilson_95": {
        "lower": 0.030053369748306635,
        "upper": 0.5635028221864702,
        "hits": 1,
        "n": 6,
    },
    "clean_regret_wilson_95": {
        "lower": 0.06149194472039621,
        "upper": 0.7923403991979523,
        "hits": 1,
        "n": 3,
    },
    "n_feb_hits": 1,
    "n_net_success": 1,
    "n_broken_success": 1,
    "n_broken_success_denom": 6,
    "n_clean_regret_hits": 1,
    "n_clean_regret_denom": 3,
}


def test_feb_golden_fixed_log():
    metrics = evaluate_feb(GOLDEN_EPISODES).to_dict()
    assert metrics_nearly_equal(metrics, GOLDEN_METRICS)


def test_feb_candidate_internal_identity():
    # All Set C chose candidates → FEB ≡ 1
    eps = [
        {
            "set_label": "C",
            "chose_from_candidates": True,
            "task_success": False,
            "oracle_recoverable": False,
            "used_fallback": False,
        }
        for _ in range(5)
    ]
    m = evaluate_feb(eps)
    assert m.feb == 1.0
    assert m.net_success == 0.0
