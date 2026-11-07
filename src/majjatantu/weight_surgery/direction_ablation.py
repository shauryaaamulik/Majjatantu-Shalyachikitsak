"""
Direction ablation / projection removal — gradient-free weight edit.

Matches the paper's Exp. 8 style edit and README baseline 5:
subtract a direction from selected output columns of a weight matrix.

Primary repair path: NO gradients, NO training loops.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import torch
from torch import nn

from majjatantu.types import EditSpec, ParameterBudget


@dataclass
class DirectionAblationResult:
    edit: EditSpec
    params_touched: int
    direction_norm: float


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.vector_norm(v)
    if float(n) < eps:
        raise ValueError("direction has near-zero norm; cannot ablate")
    return v / n


def ablate_direction_in_weight(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    columns: slice | list[int] | None = None,
    strength: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Remove `direction` from the column-space of a 2-D weight matrix.

    For W of shape [out, in] (nn.Linear.weight layout), we treat each selected
    output row as a vector in input-space and subtract the projection onto
    the (normalized) direction living in input-space... 

    Paper Exp. 8 subtracted a direction from MLP *output* columns. Here we
    support ablating a direction `d` in the *output* space of W by projecting
    each column:  W[:, j] ← W[:, j] - strength * d * (d · W[:, j])
    when d is length `out`. If d is length `in`, we project each row instead.

    Returns (edited_weight, params_touched).
    """
    if weight.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {tuple(weight.shape)}")

    w = weight.detach().clone()
    d = direction.detach().float().reshape(-1).to(device=w.device, dtype=w.dtype)
    d = _normalize(d)

    out_features, in_features = w.shape
    touched = 0

    if d.numel() == out_features:
        # Project each column onto output-space direction and subtract.
        col_idx = _resolve_indices(in_features, columns)
        for j in col_idx:
            col = w[:, j]
            proj = torch.dot(d, col.float()).to(dtype=w.dtype)
            w[:, j] = col - strength * proj * d
            touched += out_features
    elif d.numel() == in_features:
        # Project each row onto input-space direction and subtract.
        row_idx = _resolve_indices(out_features, columns)
        for i in row_idx:
            row = w[i, :]
            proj = torch.dot(d, row.float()).to(dtype=w.dtype)
            w[i, :] = row - strength * proj * d
            touched += in_features
    else:
        raise ValueError(
            f"direction length {d.numel()} matches neither out={out_features} "
            f"nor in={in_features}"
        )

    return w, touched


def _resolve_indices(size: int, columns: slice | list[int] | None) -> list[int]:
    if columns is None:
        return list(range(size))
    if isinstance(columns, slice):
        return list(range(*columns.indices(size)))
    return list(columns)


def apply_direction_ablation(
    model: nn.Module,
    param_name: str,
    direction: torch.Tensor,
    *,
    columns: slice | list[int] | None = None,
    strength: float = 1.0,
    budget: ParameterBudget | None = None,
) -> DirectionAblationResult:
    """
    Apply direction ablation in-place to a named parameter.

    Raises if the edit would violate the parameter budget.
    """
    named = dict(model.named_parameters())
    if param_name not in named:
        raise KeyError(f"parameter not found: {param_name}")

    param = named[param_name]
    edited, touched = ablate_direction_in_weight(
        param.data, direction, columns=columns, strength=strength
    )

    total = sum(p.numel() for p in model.parameters())
    fraction = touched / total if total else 0.0
    if budget is not None:
        if budget.max_params is not None and touched > budget.max_params:
            raise RuntimeError(
                f"edit touches {touched} params > max_params={budget.max_params}"
            )
        if fraction > budget.max_fraction:
            raise RuntimeError(
                f"edit fraction {fraction:.6f} > max_fraction={budget.max_fraction}"
            )
        if budget.layers is not None and not any(layer in param_name for layer in budget.layers):
            raise RuntimeError(f"parameter {param_name} outside allowed layers {budget.layers}")

    with torch.no_grad():
        param.copy_(edited)

    edit = EditSpec(
        edit_id=str(uuid.uuid4()),
        operator="direction_ablation",
        target_params=[param_name],
        params_touched=touched,
        fraction_touched=fraction,
        description=(
            f"Ablate direction (strength={strength}) on {param_name} "
            f"columns={columns!r}"
        ),
        metadata={
            "strength": strength,
            "columns": _columns_repr(columns),
            "direction_dim": int(direction.numel()),
        },
    )
    return DirectionAblationResult(
        edit=edit,
        params_touched=touched,
        direction_norm=float(torch.linalg.vector_norm(direction.float())),
    )


def _columns_repr(columns: slice | list[int] | None) -> str | list[int] | None:
    if isinstance(columns, slice):
        return f"slice({columns.start},{columns.stop},{columns.step})"
    return columns
