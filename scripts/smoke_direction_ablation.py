#!/usr/bin/env python3
"""CPU-only smoke: ablate a planted direction and verify orthogonality + logging."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Allow running without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majjatantu.orchestration import record_edit, rollback, start_experiment
from majjatantu.types import PatchDecision
from majjatantu.weight_surgery import apply_direction_ablation


def main() -> None:
    torch.manual_seed(42)
    model = nn.Linear(16, 32, bias=False)
    state = start_experiment(
        "smoke_direction_ablation",
        model,
        max_forward_passes=10,
        log_dir=ROOT / "artifacts" / "edit_logs",
    )

    # Plant a direction aligned with column 0, then ablate it.
    direction = model.weight.detach()[:, 0].clone()
    d_hat = direction / torch.linalg.vector_norm(direction)
    before_dot = float(torch.dot(d_hat, model.weight.detach()[:, 0]))
    result = apply_direction_ablation(
        model, "weight", direction, columns=[0], strength=1.0
    )
    after_dot = float(torch.dot(d_hat, model.weight.detach()[:, 0]))
    print(f"before projection: {before_dot:.6f}")
    print(f"after  projection: {after_dot:.6f}")
    print(f"params touched:    {result.params_touched}")
    print(f"edit id:           {result.edit.edit_id}")

    record_edit(
        state,
        result.edit,
        model=model,
        decision=PatchDecision.ACCEPT if abs(after_dot) < 1e-4 else PatchDecision.REJECT,
        metrics={"abs_projection": abs(after_dot)},
        notes="smoke test",
    )
    rollback(model, state)
    print(f"rolled back; log entries: {len(state.edit_log)}")
    print("OK")


if __name__ == "__main__":
    main()
