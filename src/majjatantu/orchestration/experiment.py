"""Minimal experimental orchestrator: budgets, rollback, edit logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from majjatantu.models.loader import apply_state_dict, clone_state_dict, hash_state_dict
from majjatantu.reporting.edit_log import EditLog
from majjatantu.types import EditLogEntry, EditSpec, ForwardPassBudget, PatchDecision


@dataclass
class ExperimentState:
    name: str
    model_hash_baseline: str
    budget: ForwardPassBudget
    edit_log: EditLog
    baseline_state: dict[str, torch.Tensor] = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)


def start_experiment(
    name: str,
    model: nn.Module,
    *,
    max_forward_passes: int,
    log_dir: str | Path = "artifacts/edit_logs",
) -> ExperimentState:
    baseline = clone_state_dict(model)
    model_hash = hash_state_dict(baseline)
    log_path = Path(log_dir) / f"{name}.jsonl"
    return ExperimentState(
        name=name,
        model_hash_baseline=model_hash,
        budget=ForwardPassBudget(max_passes=max_forward_passes),
        edit_log=EditLog(log_path),
        baseline_state=baseline,
    )


def rollback(model: nn.Module, state: ExperimentState) -> None:
    """Restore the pre-edit baseline weights."""
    apply_state_dict(model, state.baseline_state)


def record_edit(
    state: ExperimentState,
    edit: EditSpec,
    *,
    model: nn.Module,
    decision: PatchDecision,
    metrics: dict[str, float] | None = None,
    notes: str = "",
) -> EditLogEntry:
    after_hash = hash_state_dict(clone_state_dict(model))
    entry = EditLogEntry(
        edit=edit,
        model_hash_before=state.model_hash_baseline,
        model_hash_after=after_hash,
        decision=decision,
        metrics=metrics or {},
        forward_passes_used=state.budget.used_passes,
        notes=notes,
    )
    state.edit_log.append(entry)
    return entry
