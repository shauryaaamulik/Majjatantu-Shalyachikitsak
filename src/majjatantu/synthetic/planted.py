"""Plant a known defect in spare hidden units, plus guarded train/dev/val/test splits.

The *clean* model has a few dead hidden units (their ``W_out`` columns are zero).
Corruption turns those units into a trigger detector that writes a fixed
direction ``u`` into the residual stream whenever input feature ``f`` exceeds a
threshold, pushing the prediction to one wrong class. Ground truth is therefore
known exactly: the units, the write direction, and the trigger. Zeroing those
``W_out`` columns restores the clean model, so a perfect repair exists.

Labels come from the clean model, so "error" means "disagrees with the clean model".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from majjatantu.synthetic.toy_model import ToyConfig, ToyResidualMLP


@dataclass(frozen=True)
class PlantedDefect:
    units: tuple[int, ...]
    write_direction: np.ndarray  # unit vector in residual space
    trigger_feature: int
    trigger_threshold: float
    target_class: int


@dataclass
class PlantedFailure:
    clean: ToyResidualMLP
    corrupted: ToyResidualMLP
    defect: PlantedDefect


def make_planted_failure(
    seed: int,
    cfg: ToyConfig = ToyConfig(),
    *,
    n_units: int = 4,
    trigger_feature: int = 0,
    threshold: float = 1.0,
    gain: float = 3.0,
    strength: float = 3.0,
) -> PlantedFailure:
    rng = np.random.default_rng(seed + 10_000)
    clean = ToyResidualMLP.random(cfg, seed)
    units = tuple(sorted(int(i) for i in rng.choice(cfg.d_hidden, n_units, replace=False)))
    clean.params["W_out"][:, list(units)] = 0.0  # spare capacity in the clean model

    cls = int(rng.integers(cfg.n_classes))
    u = clean.params["U"][cls] / np.linalg.norm(clean.params["U"][cls])

    bad = clean.copy()
    reader = np.linalg.pinv(bad.params["E"])[trigger_feature]  # reader @ (E x) == x[trigger_feature]
    for i in units:
        bad.params["W_in"][i] = gain * reader
        bad.params["b_in"][i] = -gain * threshold
        bad.params["W_out"][:, i] = strength * u
    return PlantedFailure(clean, bad, PlantedDefect(units, u, trigger_feature, threshold, cls))


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray        # labels from the clean model
    trigger: np.ndarray  # bool: input lies in the planted failure region (evaluation-only ground truth)


class Splits:
    """dev / val are open to repair strategies; test is reachable only via ``reveal_test``."""

    def __init__(self, dev: Split, val: Split, test: Split):
        self.dev, self.val, self._test = dev, val, test
        self.test_revealed = False

    def reveal_test(self) -> Split:
        self.test_revealed = True
        return self._test


def make_splits(pf: PlantedFailure, seed: int, n: int = 2000) -> Splits:
    rng = np.random.default_rng(seed + 20_000)
    d = pf.defect

    def one() -> Split:
        x = rng.normal(size=(n, pf.clean.cfg.d_in))
        return Split(x, pf.clean.predict(x), x[:, d.trigger_feature] > d.trigger_threshold)

    return Splits(one(), one(), one())
