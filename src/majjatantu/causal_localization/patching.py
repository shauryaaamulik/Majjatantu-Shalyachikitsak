"""Activation patching stubs for Phase I → II localization work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PatchResult:
    module_name: str
    metric_before: float
    metric_after: float
    delta: float
    metadata: dict[str, Any]


def activation_patch_effect(
    clean_activation: torch.Tensor,
    corrupt_activation: torch.Tensor,
) -> torch.Tensor:
    """
    Return the patched activation used in causal tracing style interventions.

    Phase I helper: replaces corrupt with clean (or a blend). Full circuit
    localization comes in Phase II.
    """
    if clean_activation.shape != corrupt_activation.shape:
        raise ValueError(
            f"shape mismatch: clean={tuple(clean_activation.shape)} "
            f"corrupt={tuple(corrupt_activation.shape)}"
        )
    return clean_activation.detach().clone()


def list_linear_modules(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
