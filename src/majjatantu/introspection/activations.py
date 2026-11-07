"""Streaming activation capture — never keep all activations in memory at once."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch
from torch import nn


@dataclass
class ActivationStats:
    module_name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    l2_norm: float


@dataclass
class ActivationCapture:
    """Collects reduced stats (not raw tensors) unless explicitly requested."""

    stats: list[ActivationStats] = field(default_factory=list)
    last_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    keep_tensors: bool = False

    def record(self, name: str, tensor: torch.Tensor) -> None:
        t = tensor.detach()
        flat = t.float().reshape(-1)
        self.stats.append(
            ActivationStats(
                module_name=name,
                shape=tuple(t.shape),
                mean=float(flat.mean()),
                std=float(flat.std(unbiased=False)),
                l2_norm=float(torch.linalg.vector_norm(flat)),
            )
        )
        if self.keep_tensors:
            # Store on CPU to avoid GPU OOM during multi-example runs.
            self.last_tensors[name] = t.detach().cpu()


def _resolve_modules(
    model: nn.Module,
    module_names: Sequence[str] | None,
) -> list[tuple[str, nn.Module]]:
    named = dict(model.named_modules())
    if module_names is None:
        # Default: leaf Linear / LayerNorm style modules only — keep hooks sparse.
        picked: list[tuple[str, nn.Module]] = []
        for name, mod in named.items():
            if len(list(mod.children())) == 0 and isinstance(mod, (nn.Linear, nn.LayerNorm)):
                picked.append((name, mod))
        return picked
    missing = [n for n in module_names if n not in named]
    if missing:
        raise KeyError(f"unknown modules: {missing}")
    return [(n, named[n]) for n in module_names]


@contextmanager
def capture_activations(
    model: nn.Module,
    module_names: Sequence[str] | None = None,
    *,
    keep_tensors: bool = False,
    reduce_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[ActivationCapture]:
    """Context manager that hooks selected modules and records activation stats."""
    capture = ActivationCapture(keep_tensors=keep_tensors)
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            if reduce_fn is not None:
                tensor = reduce_fn(tensor)
            capture.record(name, tensor)

        return hook

    try:
        for name, mod in _resolve_modules(model, module_names):
            handles.append(mod.register_forward_hook(make_hook(name)))
        yield capture
    finally:
        for h in handles:
            h.remove()


def mean_activation_difference(
    success_acts: dict[str, torch.Tensor],
    failure_acts: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Per-module mean difference vectors (failure − success).

    This is the core statistic for the non-LLM direction-ablation baseline
    called out in README §9 baseline 5.
    """
    shared = set(success_acts) & set(failure_acts)
    diffs: dict[str, torch.Tensor] = {}
    for name in shared:
        s = success_acts[name].float()
        f = failure_acts[name].float()
        if s.shape != f.shape:
            # Reduce sequence dim if present: take mean over tokens.
            s = s.mean(dim=tuple(range(1, s.ndim - 1))) if s.ndim > 2 else s
            f = f.mean(dim=tuple(range(1, f.ndim - 1))) if f.ndim > 2 else f
        # Average over batch if present.
        if s.ndim >= 2:
            s = s.mean(dim=0)
            f = f.mean(dim=0)
        diffs[name] = f - s
    return diffs


def pool_token_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce [batch, seq, hidden] → [batch, hidden] via token mean."""
    if tensor.ndim == 3:
        return tensor.mean(dim=1)
    return tensor
