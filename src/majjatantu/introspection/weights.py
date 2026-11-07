"""Weight-tensor inspection without dumping entire models into prompts."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from majjatantu.types import WeightTensorSummary


def summarize_tensor(name: str, tensor: torch.Tensor, *, top_k_singular: int = 5) -> WeightTensorSummary:
    """Compact numeric fingerprint of a single parameter tensor."""
    t = tensor.detach().float().cpu()
    flat = t.reshape(-1)
    spectral: list[float] = []
    if t.ndim == 2 and min(t.shape) > 0:
        # Economy SVD on the smaller side; avoid materialising huge matrices fully when possible.
        try:
            k = min(top_k_singular, min(t.shape))
            if k > 0:
                # torch.svd_lowrank is cheaper for tall/wide matrices
                if max(t.shape) > 512 and k < min(t.shape):
                    u, s, _ = torch.svd_lowrank(t, q=k)
                    spectral = s[:k].tolist()
                else:
                    s = torch.linalg.svdvals(t)
                    spectral = s[:k].tolist()
        except RuntimeError:
            spectral = []

    return WeightTensorSummary(
        name=name,
        shape=list(t.shape),
        dtype=str(tensor.dtype),
        numel=int(flat.numel()),
        mean=float(flat.mean()),
        std=float(flat.std(unbiased=False)),
        min=float(flat.min()),
        max=float(flat.max()),
        frobenius_norm=float(torch.linalg.vector_norm(flat)),
        spectral_top_k=spectral,
    )


def iter_named_parameters(
    model: torch.nn.Module,
    *,
    name_contains: str | None = None,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        if name_contains is None or name_contains in name:
            yield name, param.data


def summarize_model(
    model: torch.nn.Module,
    *,
    name_contains: str | None = None,
    top_k_singular: int = 5,
    limit: int | None = None,
) -> list[WeightTensorSummary]:
    """Summarise selected parameters — streaming, no full activation store."""
    out: list[WeightTensorSummary] = []
    for i, (name, tensor) in enumerate(iter_named_parameters(model, name_contains=name_contains)):
        if limit is not None and i >= limit:
            break
        out.append(summarize_tensor(name, tensor, top_k_singular=top_k_singular))
    return out


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def slice_tensor(
    tensor: torch.Tensor,
    *,
    rows: slice | None = None,
    cols: slice | None = None,
) -> torch.Tensor:
    """Safe 2-D slice helper for structured tool interfaces."""
    t = tensor.detach()
    if t.ndim != 2:
        raise ValueError(f"slice_tensor expects a 2-D tensor, got shape {tuple(t.shape)}")
    r = rows if rows is not None else slice(None)
    c = cols if cols is not None else slice(None)
    return t[r, c].cpu().clone()
