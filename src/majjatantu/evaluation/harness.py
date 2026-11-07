"""Reproducible evaluation harness with explicit forward-pass accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from majjatantu.types import ForwardPassBudget, PatchDecision


MetricFn = Callable[[str, str], float]


@dataclass
class EvalExample:
    prompt: str
    reference: str | None = None
    split: str = "dev"  # "dev" | "val" | "test" — agent must never see "test"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    split: str
    n: int
    metrics: dict[str, float]
    forward_passes_used: int
    per_example: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RegressionGate:
    """Accept/reject rules from README §7 / §15."""

    min_target_improvement: float = 0.0
    max_regression: float = 0.05
    require_save_reload_stable: bool = True


def exact_match(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


@torch.inference_mode()
def run_generation_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[EvalExample],
    *,
    split: str,
    budget: ForwardPassBudget,
    max_new_tokens: int = 64,
    metric_fns: dict[str, MetricFn] | None = None,
    forbid_test_to_agent: bool = True,
) -> EvalResult:
    """
    Evaluate generation on a fixed split.

    Hard constraint: when forbid_test_to_agent is True and split == 'test',
    callers acting as the repair agent should not invoke this; orchestration
    owns final test evaluation.
    """
    if forbid_test_to_agent and split == "test":
        # Soft guard — orchestration should set this False for the final judge.
        pass

    metric_fns = metric_fns or ({"exact_match": exact_match} if examples and examples[0].reference else {})
    device = next(model.parameters()).device
    scores: dict[str, list[float]] = {k: [] for k in metric_fns}
    per_example: list[dict[str, Any]] = []

    split_examples = [e for e in examples if e.split == split]
    for ex in split_examples:
        budget.consume(1)
        inputs = tokenizer(ex.prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred = tokenizer.decode(
            out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        row: dict[str, Any] = {"prompt": ex.prompt, "prediction": pred, "reference": ex.reference}
        if ex.reference is not None:
            for name, fn in metric_fns.items():
                val = fn(pred, ex.reference)
                scores[name].append(val)
                row[name] = val
        per_example.append(row)

    metrics = {
        name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in scores.items()
    }
    return EvalResult(
        split=split,
        n=len(split_examples),
        metrics=metrics,
        forward_passes_used=budget.used_passes,
        per_example=per_example,
    )


def decide_patch(
    before: dict[str, float],
    after: dict[str, float],
    *,
    target_metric: str,
    regression_metrics: list[str],
    gate: RegressionGate,
) -> PatchDecision:
    """Apply improvement / regression gates. Failed repair is a valid result."""
    if target_metric not in after or target_metric not in before:
        return PatchDecision.REJECT
    improvement = after[target_metric] - before[target_metric]
    if improvement < gate.min_target_improvement:
        return PatchDecision.REJECT
    for m in regression_metrics:
        if m not in before or m not in after:
            continue
        if before[m] - after[m] > gate.max_regression:
            return PatchDecision.REJECT
    return PatchDecision.ACCEPT
