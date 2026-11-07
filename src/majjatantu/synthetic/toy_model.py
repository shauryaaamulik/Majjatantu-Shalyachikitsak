"""A small residual MLP classifier used as a fully inspectable testbed.

    r0     = x @ E.T
    h      = relu(r0 @ W_in.T + b_in)         (optionally patched via hidden_hook)
    r1     = r0 + h @ W_out.T                 (block writes into the residual stream)
    logits = r1 @ U.T

``W_out`` has the same [out, in] layout as an MLP output projection, so
direction ablation on it mirrors the paper's Exp. 8 edit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from majjatantu.types import ForwardPassBudget

PARAM_NAMES = ("E", "W_in", "b_in", "W_out", "U")


class BudgetExhausted(RuntimeError):
    """A strategy asked for more forward passes than it was given."""


def charge(budget: ForwardPassBudget | None, n: int = 1) -> None:
    """Charge ``n`` forward passes (1 pass = 1 batch call). ``None`` = unmetered (evaluation only)."""
    if budget is None:
        return
    if budget.remaining() < n:
        raise BudgetExhausted(f"budget {budget.max_passes} exhausted (used {budget.used_passes})")
    budget.consume(n)


@dataclass(frozen=True)
class ToyConfig:
    d_in: int = 16
    d_model: int = 32
    d_hidden: int = 64
    n_classes: int = 4


@dataclass
class ForwardOutput:
    logits: np.ndarray
    resid_pre: np.ndarray
    hidden: np.ndarray
    resid_post: np.ndarray


class ToyResidualMLP:
    def __init__(self, params: dict[str, np.ndarray], cfg: ToyConfig):
        expected = {
            "E": (cfg.d_model, cfg.d_in),
            "W_in": (cfg.d_hidden, cfg.d_model),
            "b_in": (cfg.d_hidden,),
            "W_out": (cfg.d_model, cfg.d_hidden),
            "U": (cfg.n_classes, cfg.d_model),
        }
        for name, shape in expected.items():
            if name not in params or params[name].shape != shape:
                raise ValueError(f"param {name!r} must have shape {shape}")
        self.params = {k: np.array(params[k], dtype=np.float64) for k in PARAM_NAMES}
        self.cfg = cfg

    @classmethod
    def random(cls, cfg: ToyConfig, seed: int) -> "ToyResidualMLP":
        rng = np.random.default_rng(seed)
        return cls(
            {
                "E": rng.normal(0, 1 / np.sqrt(cfg.d_in), (cfg.d_model, cfg.d_in)),
                "W_in": rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.d_hidden, cfg.d_model)),
                "b_in": np.zeros(cfg.d_hidden),
                "W_out": rng.normal(0, 0.5 / np.sqrt(cfg.d_hidden), (cfg.d_model, cfg.d_hidden)),
                "U": rng.normal(0, 1 / np.sqrt(cfg.d_model), (cfg.n_classes, cfg.d_model)),
            },
            cfg,
        )

    def forward(
        self,
        x: np.ndarray,
        *,
        budget: ForwardPassBudget | None = None,
        hidden_hook: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> ForwardOutput:
        charge(budget)
        P = self.params
        r0 = x @ P["E"].T
        h = np.maximum(r0 @ P["W_in"].T + P["b_in"], 0.0)
        if hidden_hook is not None:
            h = hidden_hook(h)
        r1 = r0 + h @ P["W_out"].T
        return ForwardOutput(r1 @ P["U"].T, r0, h, r1)

    def predict(self, x: np.ndarray, *, budget: ForwardPassBudget | None = None) -> np.ndarray:
        return self.forward(x, budget=budget).logits.argmax(axis=1)

    def copy(self) -> "ToyResidualMLP":
        return ToyResidualMLP(self.params, self.cfg)

    @property
    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k in PARAM_NAMES:
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.params[k]).tobytes())
        return h.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        c = self.cfg
        with open(path, "wb") as f:
            np.savez(f, cfg=np.array([c.d_in, c.d_model, c.d_hidden, c.n_classes]), **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "ToyResidualMLP":
        with np.load(path) as z:
            return cls({k: z[k] for k in PARAM_NAMES}, ToyConfig(*[int(v) for v in z["cfg"]]))
