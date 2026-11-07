"""Run one trial: discover failure, try each strategy at a fixed budget, judge, log everything."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from majjatantu.reporting.edit_log import EditLog
from majjatantu.synthetic.discovery import discover_failure_slices, reproducibility_report
from majjatantu.synthetic.evaluate import SyntheticGate, judge, split_metrics
from majjatantu.synthetic.planted import make_planted_failure, make_splits
from majjatantu.synthetic.repair import STRATEGIES, Strategy
from majjatantu.synthetic.toy_model import BudgetExhausted
from majjatantu.types import EditLogEntry, ForwardPassBudget, PatchDecision


def run_trial(
    seed: int,
    budget_limit: int,
    *,
    k: int = 4,
    strategies: dict[str, Strategy] | None = None,
    gate: SyntheticGate | None = None,
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    strategies = strategies or STRATEGIES
    gate = gate or SyntheticGate()
    pf = make_planted_failure(seed)
    splits = make_splits(pf, seed)
    base = pf.corrupted
    truth = set(pf.defect.units)
    log = EditLog(log_path) if log_path else None

    # 1. discovery (1 unmetered evaluation pass; diagnostic, does not steer the repair strategies)
    failed = base.predict(splits.dev.x) != splits.dev.y
    top = discover_failure_slices(splits.dev.x, failed)
    report = reproducibility_report(splits.dev.x, failed, top[0]) if top else None
    found = bool(top and top[0].feature == pf.defect.trigger_feature)

    # 2. strategies, each with its own identical budget
    pending = []
    for name, strat in strategies.items():
        budget = ForwardPassBudget(max_passes=budget_limit)
        row: dict[str, Any] = {
            "seed": seed, "strategy": name, "budget": budget_limit, "k": k,
            "discovery_found_trigger": found,
            "failure_reproducibility": report.reproducibility_rate if report else 0.0,
        }
        try:
            prop = strat(base, splits.dev, budget, seed, k)
        except (BudgetExhausted, ValueError) as e:
            row.update(decision="no_proposal", passes_used=budget.used_passes, note=str(e))
            pending.append((row, None))
            continue
        decision, metrics, reasons = judge(base, prop.model, splits.val, prop.edit.fraction_touched, gate)
        cols = prop.meta.get("columns")
        row.update(
            decision=decision.value, passes_used=budget.used_passes, note="; ".join(reasons),
            params_touched=prop.edit.params_touched, fraction_touched=prop.edit.fraction_touched,
            localization_overlap=(len(truth & set(cols)) / k) if cols is not None else None,
            units_evaluated=prop.meta.get("units_evaluated"), **metrics,
        )
        if log is not None:
            log.append(EditLogEntry(
                edit=prop.edit, model_hash_before=base.fingerprint(),
                model_hash_after=prop.model.fingerprint(), decision=decision, metrics=metrics,
                forward_passes_used=budget.used_passes, notes=f"{name} seed={seed} {row['note']}".strip(),
            ))
        pending.append((row, prop))

    # 3. only after every decision is made is the held-out test split revealed (for reporting)
    test = splits.reveal_test()
    t0 = split_metrics(base, test)
    rows = []
    for row, prop in pending:
        t = split_metrics(prop.model, test) if prop else t0
        row.update(test_target_error_before=t0["target_error"], test_target_error_after=t["target_error"],
                   test_retention_before=t0["retention_acc"], test_retention_after=t["retention_acc"])
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per (strategy, budget) across seeds."""
    out = []
    keys = sorted({(r["strategy"], r["budget"]) for r in rows}, key=lambda t: (t[0], t[1]))
    for strat, bud in keys:
        rs = [r for r in rows if r["strategy"] == strat and r["budget"] == bud]
        made = [r for r in rs if r["decision"] != "no_proposal"]
        def mean(key, src=made):
            v = [r[key] for r in src if r.get(key) is not None]
            return float(np.mean(v)) if v else float("nan")
        out.append({
            "strategy": strat, "budget": bud, "n": len(rs),
            "proposal_rate": len(made) / len(rs),
            "accept_rate": sum(r["decision"] == "accept" for r in rs) / len(rs),
            "test_err_before": mean("test_target_error_before", rs),
            "test_err_after": mean("test_target_error_after", rs),
            "test_retention_drop": mean("test_retention_before", rs) - mean("test_retention_after", rs),
            "overlap": mean("localization_overlap"),
            "param_fraction": mean("fraction_touched"),
        })
    return out
