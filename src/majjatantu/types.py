"""Shared typed records used across subsystems."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PatchDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"


class ParameterBudget(BaseModel):
    """Hard caps on how much of a model may be touched."""

    max_params: int | None = None
    max_fraction: float = 0.001  # 0.1% default experimental target
    max_rank: int | None = None
    layers: list[str] | None = None


class ForwardPassBudget(BaseModel):
    """Explicit forward-pass budget — must be logged with every result."""

    max_passes: int
    used_passes: int = 0

    def remaining(self) -> int:
        return max(0, self.max_passes - self.used_passes)

    def consume(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError("cannot consume a negative number of passes")
        if self.used_passes + n > self.max_passes:
            raise RuntimeError(
                f"forward-pass budget exceeded: used={self.used_passes}, "
                f"requested={n}, max={self.max_passes}"
            )
        self.used_passes += n


class ModelIdentity(BaseModel):
    """Fingerprint of a loaded checkpoint for reproducibility."""

    name_or_path: str
    architecture: str | None = None
    checkpoint_hash: str
    num_parameters: int
    config: dict[str, Any] = Field(default_factory=dict)
    tokenizer_name: str | None = None


class WeightTensorSummary(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    mean: float
    std: float
    min: float
    max: float
    frobenius_norm: float
    spectral_top_k: list[float] = Field(default_factory=list)


class EditSpec(BaseModel):
    """Machine-readable description of a candidate weight edit."""

    edit_id: str
    operator: str
    target_params: list[str]
    params_touched: int
    fraction_touched: float
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class EditLogEntry(BaseModel):
    """Every candidate edit — accepted or rejected — must be logged."""

    edit: EditSpec
    model_hash_before: str
    model_hash_after: str | None = None
    decision: PatchDecision
    metrics: dict[str, float] = Field(default_factory=dict)
    forward_passes_used: int = 0
    notes: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class FailureReport(BaseModel):
    """Structured diagnostic for a reproducible behavioural failure."""

    failure_id: str
    category: str
    description: str
    reproducibility_rate: float
    n_trials: int
    example_prompts: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
