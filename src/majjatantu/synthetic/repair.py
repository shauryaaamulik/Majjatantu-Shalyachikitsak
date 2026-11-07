"""Gradient-free repair operators and the strategies built from them.

No gradients and no training loops anywhere in this module (README hard constraint).
``ablate_direction`` follows the semantics of ``weight_surgery.direction_ablation``:
for W of shape [out, in] and a direction of length ``out``, each selected column j
becomes ``W[:, j] - strength * d * (d . W[:, j])``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from majjatantu.synthetic.localization import unit_ablation_effects
from majjatantu.synthetic.planted import Split
from majjatantu.synthetic.toy_model import ToyResidualMLP, charge
from majjatantu.types import EditSpec, ForwardPassBudget


@dataclass
class Proposal:
    model: ToyResidualMLP
    edit: EditSpec
    meta: dict[str, Any] = field(default_factory=dict)


def ablate_direction(
    model: ToyResidualMLP, direction: np.ndarray, columns: list[int] | None = None, strength: float = 1.0
) -> tuple[ToyResidualMLP, int]:
    d = np.asarray(direction, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(d)
    if n < 1e-8:
        raise ValueError("direction has near-zero norm")
    d = d / n
    new = model.copy()
    W = new.params["W_out"]
    cols = list(range(W.shape[1])) if columns is None else list(columns)
    for j in cols:
        W[:, j] = W[:, j] - strength * d * (d @ W[:, j])
    return new, W.shape[0] * len(cols)


def zero_columns(model: ToyResidualMLP, columns: list[int]) -> tuple[ToyResidualMLP, int]:
    new = model.copy()
    new.params["W_out"][:, list(columns)] = 0.0
    return new, new.params["W_out"].shape[0] * len(columns)


def _edit(model: ToyResidualMLP, operator: str, touched: int, description: str, **meta: Any) -> EditSpec:
    return EditSpec(
        edit_id=str(uuid.uuid4()),
        operator=operator,
        target_params=["W_out"],
        params_touched=touched,
        fraction_touched=touched / model.n_params,
        description=description,
        metadata=meta,
    )


Strategy = Callable[[ToyResidualMLP, Split, ForwardPassBudget, int, int], Proposal]


def random_columns(base: ToyResidualMLP, dev: Split, budget: ForwardPassBudget, seed: int, k: int) -> Proposal:
    """Control: zero k random W_out columns (same parameter footprint, no information used)."""
    cols = sorted(int(c) for c in np.random.default_rng(seed + 1).choice(base.cfg.d_hidden, k, replace=False))
    m, touched = zero_columns(base, cols)
    return Proposal(m, _edit(base, "zero_columns", touched, f"zero random columns {cols}"), {"columns": cols})


def _diffmeans_direction(base: ToyResidualMLP, dev: Split, budget: ForwardPassBudget) -> np.ndarray:
    out = base.forward(dev.x, budget=budget)  # 1 pass
    failed = out.logits.argmax(axis=1) != dev.y
    if not failed.any() or failed.all():
        raise ValueError("need both failing and passing dev inputs")
    delta = out.resid_post - out.resid_pre  # what the block wrote
    return delta[failed].mean(axis=0) - delta[~failed].mean(axis=0)


def diffmeans_topk(base: ToyResidualMLP, dev: Split, budget: ForwardPassBudget, seed: int, k: int) -> Proposal:
    """Non-LLM baseline 5: activation-difference direction, ablated from the k most aligned columns."""
    d = _diffmeans_direction(base, dev, budget)
    d_hat = d / np.linalg.norm(d)
    scores = np.abs(d_hat @ base.params["W_out"])
    cols = sorted(int(c) for c in np.argsort(-scores)[:k])
    m, touched = ablate_direction(base, d, cols)
    return Proposal(m, _edit(base, "direction_ablation", touched, f"diff-means direction, columns {cols}"),
                    {"columns": cols})


def diffmeans_all(base: ToyResidualMLP, dev: Split, budget: ForwardPassBudget, seed: int, k: int) -> Proposal:
    """Reference: same direction ablated from every column (dense edit, usually over the parameter budget)."""
    d = _diffmeans_direction(base, dev, budget)
    m, touched = ablate_direction(base, d, None)
    return Proposal(m, _edit(base, "direction_ablation", touched, "diff-means direction, all columns"), {})


def causal_columns(base: ToyResidualMLP, dev: Split, budget: ForwardPassBudget, seed: int, k: int) -> Proposal:
    """Causally-informed: patch hidden units one at a time, zero the k with the largest effect."""
    pred = base.predict(dev.x, budget=budget)
    failed = pred != dev.y
    if not failed.any() or failed.all():
        raise ValueError("need both failing and passing dev inputs")
    effects, n_eval = unit_ablation_effects(
        base, dev.x[failed], dev.y[failed], dev.x[~failed], budget, np.random.default_rng(seed + 2)
    )
    order = np.argsort(-np.nan_to_num(effects, nan=-np.inf))
    cols = sorted(int(c) for c in order[:k] if not np.isnan(effects[c]))
    if not cols:
        raise ValueError("budget too small to evaluate any unit")
    m, touched = zero_columns(base, cols)
    return Proposal(m, _edit(base, "zero_columns", touched, f"zero causally-ranked columns {cols}"),
                    {"columns": cols, "units_evaluated": n_eval})


STRATEGIES: dict[str, Strategy] = {
    "random_k": random_columns,
    "diffmeans_topk": diffmeans_topk,
    "diffmeans_all": diffmeans_all,
    "causal_k": causal_columns,
}
