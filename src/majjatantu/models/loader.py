"""Model loading with checkpoint hashing for reproducible experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import PreTrainedTokenizerBase

from majjatantu.types import ModelIdentity


def hash_state_dict(state_dict: dict[str, torch.Tensor], *, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 over ordered (name, dtype, shape, raw bytes) of every tensor."""
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        flat = tensor.numpy().tobytes()
        for i in range(0, len(flat), chunk_bytes):
            h.update(flat[i : i + chunk_bytes])
    return h.hexdigest()


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    identity: ModelIdentity


def load_causal_lm(
    name_or_path: str,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | str | None = "auto",
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load an open-weight causal LM and record its identity fingerprint."""
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if device is not None:
        model = model.to(device)
    model.eval()

    state = {k: v for k, v in model.state_dict().items()}
    checkpoint_hash = hash_state_dict(state)
    num_parameters = sum(p.numel() for p in model.parameters())

    identity = ModelIdentity(
        name_or_path=name_or_path,
        architecture=getattr(config, "model_type", None),
        checkpoint_hash=checkpoint_hash,
        num_parameters=num_parameters,
        config=_safe_config_dict(config),
        tokenizer_name=getattr(tokenizer, "name_or_path", None),
    )
    return LoadedModel(model=model, tokenizer=tokenizer, identity=identity)


def _safe_config_dict(config: Any) -> dict[str, Any]:
    try:
        raw = config.to_dict()
    except Exception:
        return {}
    # Keep only JSON-serialisable scalars / short lists.
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 32:
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = list(v)
    return out


def clone_state_dict(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """CPU deepcopy of parameters for isolated edits / rollback."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def apply_state_dict(model: PreTrainedModel, state: dict[str, torch.Tensor]) -> None:
    """Load a previously cloned state dict (moves tensors onto model device)."""
    device = next(model.parameters()).device
    moved = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(moved, strict=True)
