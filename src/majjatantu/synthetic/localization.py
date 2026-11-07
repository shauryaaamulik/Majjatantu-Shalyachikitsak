"""Causal localization by activation patching on hidden units (anytime under a budget)."""

from __future__ import annotations

import numpy as np

from majjatantu.synthetic.toy_model import BudgetExhausted, ToyResidualMLP
from majjatantu.types import ForwardPassBudget


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def unit_ablation_effects(
    model: ToyResidualMLP,
    x_fail: np.ndarray,
    y_fail: np.ndarray,
    x_ref: np.ndarray,
    budget: ForwardPassBudget,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Effect of replacing each hidden unit with its mean activation on ``x_ref`` (non-failing inputs).

    effect[i] = loss(failing inputs) - loss(failing inputs, unit i patched). Large positive
    means unit i is causally responsible for the failure. Costs 2 setup passes plus one pass
    per unit; units are visited in random order and the loop stops when the budget runs out,
    so unevaluated units stay NaN. Returns (effects, n_units_evaluated).
    """
    n_units = model.cfg.d_hidden
    effects = np.full(n_units, np.nan)
    ref_mean = model.forward(x_ref, budget=budget).hidden.mean(axis=0)
    base_loss = cross_entropy(model.forward(x_fail, budget=budget).logits, y_fail)
    done = 0
    for i in rng.permutation(n_units):
        def hook(h: np.ndarray, i: int = int(i)) -> np.ndarray:
            h = h.copy()
            h[:, i] = ref_mean[i]
            return h

        try:
            out = model.forward(x_fail, budget=budget, hidden_hook=hook)
        except BudgetExhausted:
            break
        effects[i] = base_loss - cross_entropy(out.logits, y_fail)
        done += 1
    return effects, done
