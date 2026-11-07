"""Torch-free tests for the synthetic testbed (run with plain numpy + pydantic)."""

import numpy as np
import pytest

from majjatantu.reporting.edit_log import EditLog
from majjatantu.synthetic.discovery import discover_failure_slices, reproducibility_report, wilson_lower
from majjatantu.synthetic.evaluate import SyntheticGate, judge, persistence_ok, split_metrics
from majjatantu.synthetic.localization import unit_ablation_effects
from majjatantu.synthetic.planted import make_planted_failure, make_splits
from majjatantu.synthetic.repair import ablate_direction, zero_columns
from majjatantu.synthetic.runner import run_trial, summarize
from majjatantu.synthetic.toy_model import BudgetExhausted, ToyConfig, ToyResidualMLP
from majjatantu.types import ForwardPassBudget, PatchDecision


def test_budget_enforced_and_failed_call_not_charged():
    m = ToyResidualMLP.random(ToyConfig(), 0)
    b = ForwardPassBudget(max_passes=2)
    x = np.zeros((2, 16))
    m.forward(x, budget=b); m.forward(x, budget=b)
    with pytest.raises(BudgetExhausted):
        m.forward(x, budget=b)
    assert b.used_passes == 2


def test_planted_defect_ground_truth():
    pf = make_planted_failure(0)
    sp = make_splits(pf, 0)
    base, clean = pf.corrupted, pf.clean
    assert split_metrics(clean, sp.dev) == {"target_error": 0.0, "retention_acc": 1.0}  # labels come from clean
    assert split_metrics(base, sp.dev)["target_error"] > 0.5      # failure is real
    assert split_metrics(base, sp.dev)["retention_acc"] > 0.95    # and confined to the trigger region
    fixed, touched = zero_columns(base, list(pf.defect.units))
    assert fixed.fingerprint() != base.fingerprint() and touched == 32 * 4
    assert split_metrics(fixed, sp.dev) == {"target_error": 0.0, "retention_acc": 1.0}  # a perfect repair exists
    assert persistence_ok(fixed, sp.val)


def test_test_split_is_guarded():
    sp = make_splits(make_planted_failure(1), 1)
    assert not sp.test_revealed
    sp.reveal_test()
    assert sp.test_revealed


def test_discovery_finds_trigger_feature_and_reproduces():
    pf = make_planted_failure(2)
    sp = make_splits(pf, 2)
    failed = pf.corrupted.predict(sp.dev.x) != sp.dev.y
    top = discover_failure_slices(sp.dev.x, failed)
    assert top[0].feature == pf.defect.trigger_feature
    rep = reproducibility_report(sp.dev.x, failed, top[0])
    assert rep.reproducibility_rate == 1.0 and rep.n_trials == 5
    # a healthy model produces no strong slice
    ok = pf.clean.predict(sp.dev.x) != sp.dev.y
    assert discover_failure_slices(sp.dev.x, ok)[0].lower_bound == 0.0


def test_wilson_lower_sane():
    assert wilson_lower(0, 0) == 0.0
    assert 0.0 < wilson_lower(50, 100) < 0.5
    assert wilson_lower(500, 1000) > wilson_lower(50, 100)


def test_causal_localization_ranks_planted_units_first_with_full_budget():
    pf = make_planted_failure(3)
    sp = make_splits(pf, 3)
    base = pf.corrupted
    failed = base.predict(sp.dev.x) != sp.dev.y
    eff, n = unit_ablation_effects(base, sp.dev.x[failed], sp.dev.y[failed], sp.dev.x[~failed],
                                   ForwardPassBudget(max_passes=100), np.random.default_rng(0))
    assert n == 64
    assert set(np.argsort(-eff)[:4]) == set(pf.defect.units)


def test_localization_is_anytime_under_small_budget():
    pf = make_planted_failure(3)
    sp = make_splits(pf, 3)
    failed = pf.corrupted.predict(sp.dev.x) != sp.dev.y
    b = ForwardPassBudget(max_passes=12)
    eff, n = unit_ablation_effects(pf.corrupted, sp.dev.x[failed], sp.dev.y[failed], sp.dev.x[~failed],
                                   b, np.random.default_rng(0))
    assert n == 10 and b.used_passes == 12 and np.isnan(eff).sum() == 54


def test_direction_ablation_removes_direction_and_is_gradient_free():
    m = ToyResidualMLP.random(ToyConfig(), 5)
    d = np.random.default_rng(0).normal(size=32)
    new, touched = ablate_direction(m, d, columns=[3, 7])
    d_hat = d / np.linalg.norm(d)
    assert np.allclose(d_hat @ new.params["W_out"][:, [3, 7]], 0, atol=1e-12)
    assert np.array_equal(new.params["W_out"][:, 0], m.params["W_out"][:, 0])  # other columns untouched
    assert touched == 64
    with pytest.raises(ValueError):
        ablate_direction(m, np.zeros(32))


def test_gate_rejects_over_budget_and_no_improvement():
    pf = make_planted_failure(4)
    sp = make_splits(pf, 4)
    base = pf.corrupted
    perfect, _ = zero_columns(base, list(pf.defect.units))
    dec, _, why = judge(base, perfect, sp.val, 0.027, SyntheticGate())
    assert dec == PatchDecision.ACCEPT and not why
    dec, _, why = judge(base, perfect, sp.val, 0.5, SyntheticGate())
    assert dec == PatchDecision.REJECT and "parameter budget" in why[0]
    dec, _, _ = judge(base, base.copy(), sp.val, 0.0, SyntheticGate())
    assert dec == PatchDecision.REJECT  # no-op edit gives no improvement


def test_trial_logs_every_edit_and_respects_budget(tmp_path):
    log = tmp_path / "log.jsonl"
    rows = run_trial(0, 100, log_path=log)
    assert {r["strategy"] for r in rows} == {"random_k", "diffmeans_topk", "diffmeans_all", "causal_k"}
    assert all(r["passes_used"] <= 100 for r in rows)
    entries = EditLog(log).entries()
    assert len(entries) == 4  # accepted AND rejected edits are logged
    assert {e.decision for e in entries} >= {PatchDecision.ACCEPT, PatchDecision.REJECT}
    assert all(e.model_hash_before != e.model_hash_after for e in entries)
    by = {r["strategy"]: r for r in rows}
    assert by["causal_k"]["decision"] == "accept" and by["causal_k"]["localization_overlap"] == 1.0
    assert by["random_k"]["decision"] == "reject"
    assert by["diffmeans_all"]["decision"] == "reject"  # dense edit is over the parameter budget


def test_tiny_budget_yields_no_proposal_not_a_crash():
    rows = run_trial(0, 1)
    by = {r["strategy"]: r for r in rows}
    assert by["causal_k"]["decision"] == "no_proposal"
    assert by["random_k"]["decision"] in {"reject", "accept"}  # needs no passes


def test_trial_is_deterministic():
    a, b = run_trial(7, 50), run_trial(7, 50)
    assert [r["test_target_error_after"] for r in a] == [r["test_target_error_after"] for r in b]


def test_summarize_shapes():
    rows = run_trial(0, 50) + run_trial(1, 50)
    s = summarize(rows)
    assert len(s) == 4 and all(r["n"] == 2 for r in s)
