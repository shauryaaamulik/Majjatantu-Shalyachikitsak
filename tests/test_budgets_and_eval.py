import pytest

from majjatantu.evaluation import RegressionGate, decide_patch
from majjatantu.types import ForwardPassBudget, PatchDecision


def test_forward_pass_budget_consume_and_exceed():
    b = ForwardPassBudget(max_passes=2)
    b.consume(1)
    assert b.remaining() == 1
    b.consume(1)
    assert b.remaining() == 0
    with pytest.raises(RuntimeError, match="budget exceeded"):
        b.consume(1)


def test_decide_patch_accept_and_reject():
    gate = RegressionGate(min_target_improvement=0.1, max_regression=0.05)
    before = {"target": 0.2, "other": 0.9}
    after_good = {"target": 0.5, "other": 0.88}
    after_bad_target = {"target": 0.25, "other": 0.9}
    after_regression = {"target": 0.5, "other": 0.8}

    assert (
        decide_patch(
            before,
            after_good,
            target_metric="target",
            regression_metrics=["other"],
            gate=gate,
        )
        == PatchDecision.ACCEPT
    )
    assert (
        decide_patch(
            before,
            after_bad_target,
            target_metric="target",
            regression_metrics=["other"],
            gate=gate,
        )
        == PatchDecision.REJECT
    )
    assert (
        decide_patch(
            before,
            after_regression,
            target_metric="target",
            regression_metrics=["other"],
            gate=gate,
        )
        == PatchDecision.REJECT
    )
