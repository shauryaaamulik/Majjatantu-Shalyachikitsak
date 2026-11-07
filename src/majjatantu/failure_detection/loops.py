"""Detect repetitive reasoning loops (paper Exp. 8 style failure)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from majjatantu.types import FailureReport, ForwardPassBudget


@dataclass
class GenerationOutcome:
    prompt: str
    text: str
    n_tokens: int
    hit_token_limit: bool
    repetition_ratio: float


def token_repetition_ratio(token_ids: list[int], *, ngram: int = 8) -> float:
    """Fraction of n-grams that are duplicates — proxy for looping."""
    if len(token_ids) < ngram * 2:
        return 0.0
    grams = [tuple(token_ids[i : i + ngram]) for i in range(len(token_ids) - ngram + 1)]
    counts = Counter(grams)
    dupes = sum(c - 1 for c in counts.values() if c > 1)
    return dupes / max(1, len(grams))


@torch.inference_mode()
def generate_once(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    budget: ForwardPassBudget | None = None,
) -> GenerationOutcome:
    """Single generation under an optional forward-pass budget (counts as 1 pass)."""
    if budget is not None:
        budget.consume(1)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, inputs["input_ids"].shape[-1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    hit_limit = len(new_ids) >= max_new_tokens
    return GenerationOutcome(
        prompt=prompt,
        text=text,
        n_tokens=len(new_ids),
        hit_token_limit=hit_limit,
        repetition_ratio=token_repetition_ratio(new_ids),
    )


def detect_loop_failures(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    max_new_tokens: int = 512,
    repetition_threshold: float = 0.25,
    budget: ForwardPassBudget | None = None,
) -> FailureReport:
    """
    Flag prompts where generation hits the token limit or shows high n-gram repetition.
    A failure must be reproducible across the provided prompt set rates.
    """
    outcomes: list[GenerationOutcome] = []
    for p in prompts:
        outcomes.append(
            generate_once(
                model, tokenizer, p, max_new_tokens=max_new_tokens, budget=budget
            )
        )

    flagged = [
        o
        for o in outcomes
        if o.hit_token_limit or o.repetition_ratio >= repetition_threshold
    ]
    rate = len(flagged) / max(1, len(outcomes))
    return FailureReport(
        failure_id="repetitive_reasoning_loop",
        category="repetitive_reasoning_loops",
        description=(
            "Generation hits max_new_tokens and/or shows high n-gram repetition "
            "(Exp. 8 style loop failure)."
        ),
        reproducibility_rate=rate,
        n_trials=len(outcomes),
        example_prompts=[o.prompt for o in flagged[:5]],
        metrics={
            "token_limit_hit_rate": sum(o.hit_token_limit for o in outcomes) / max(1, len(outcomes)),
            "mean_repetition_ratio": sum(o.repetition_ratio for o in outcomes)
            / max(1, len(outcomes)),
            "flagged_rate": rate,
        },
        metadata={"max_new_tokens": max_new_tokens, "repetition_threshold": repetition_threshold},
    )
