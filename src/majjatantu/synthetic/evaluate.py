"""Verification of a proposed edit: target repair, regression, parameter budget, persistence."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from majjatantu.synthetic.planted import Split
from majjatantu.synthetic.toy_model import ToyResidualMLP
from majjatantu.types import PatchDecision


@dataclass
class SyntheticGate:
    """Accept/reject rules (mirrors RegressionGate plus a parameter budget)."""

    min_target_improvement: float = 0.2   # absolute drop in error on the failure region
    max_regression: float = 0.05          # max drop in accuracy outside the failure region
    max_param_fraction: float = 0.05      # toy analogue of the paper-scale 0.1% target


def split_metrics(model: ToyResidualMLP, split: Split) -> dict[str, float]:
    ok = model.predict(split.x) == split.y
    return {
        "target_error": float(1.0 - ok[split.trigger].mean()),
        "retention_acc": float(ok[~split.trigger].mean()),
    }


def persistence_ok(model: ToyResidualMLP, split: Split) -> bool:
    """Edit must survive save/reload with identical predictions."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.npz"
        model.save(p)
        return bool(np.array_equal(ToyResidualMLP.load(p).predict(split.x), model.predict(split.x)))


def judge(
    base: ToyResidualMLP, patched: ToyResidualMLP, val: Split, fraction_touched: float, gate: SyntheticGate
) -> tuple[PatchDecision, dict[str, float], list[str]]:
    b, a = split_metrics(base, val), split_metrics(patched, val)
    improvement = b["target_error"] - a["target_error"]
    regression = b["retention_acc"] - a["retention_acc"]
    reasons = []
    if fraction_touched > gate.max_param_fraction:
        reasons.append(f"over parameter budget ({fraction_touched:.3f} > {gate.max_param_fraction})")
    if improvement < gate.min_target_improvement:
        reasons.append(f"target improvement {improvement:.3f} < {gate.min_target_improvement}")
    if regression > gate.max_regression:
        reasons.append(f"regression {regression:.3f} > {gate.max_regression}")
    if not persistence_ok(patched, val):
        reasons.append("edit not stable after save/reload")
    metrics = {
        "val_target_error_before": b["target_error"], "val_target_error_after": a["target_error"],
        "val_retention_before": b["retention_acc"], "val_retention_after": a["retention_acc"],
        "val_improvement": improvement, "val_regression": regression,
    }
    return (PatchDecision.REJECT if reasons else PatchDecision.ACCEPT), metrics, reasons
