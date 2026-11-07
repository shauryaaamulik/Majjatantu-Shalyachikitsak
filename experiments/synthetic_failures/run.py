#!/usr/bin/env python3
"""Synthetic planted-defect experiment: compare repair strategies at equal forward-pass budgets.

    python experiments/synthetic_failures/run.py --seeds 30 --out results/synthetic_failures.md

Everything is deterministic given --seeds. All numbers are on a toy model with a *planted*
defect: they validate the harness, not any claim about real language models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from majjatantu.synthetic.runner import run_trial, summarize  # noqa: E402


def table(summary: list[dict], extra_cols: str = "") -> str:
    head = ("| strategy | budget | accepted | test error before → after | retention drop | "
            "unit overlap | params touched |\n|---|---|---|---|---|---|---|")
    lines = [head]
    for r in summary:
        ov = "–" if r["overlap"] != r["overlap"] else f"{r['overlap']:.2f}"
        lines.append(
            f"| {r['strategy']} | {r['budget']} | {r['accept_rate']:.0%} | "
            f"{r['test_err_before']:.2f} → {r['test_err_after']:.2f} | {r['test_retention_drop']:+.3f} | "
            f"{ov} | {r['param_fraction']:.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None, help="JSONL edit log (all candidate edits)")
    a = ap.parse_args()

    sections = []
    rows = [r for s in range(a.seeds) for b in a.budgets for r in run_trial(s, b, log_path=a.log)]
    disc = sum(r["discovery_found_trigger"] for r in rows if r["strategy"] == "random_k") / (a.seeds * len(a.budgets))
    sections.append(("A. Budget sweep (4 planted units, k=4)", table(summarize(rows))))

    for tag, n_units in (("B1", 1), ("B2", 8)):
        rows = [r for s in range(a.seeds)
                for r in run_trial(s, 100, k=n_units, defect_kwargs={"n_units": n_units})]
        sections.append((f"{tag}. Defect size: {n_units} planted unit(s), k={n_units}, budget 100", table(summarize(rows))))

    md = [f"# Synthetic planted-defect results ({a.seeds} seeds)\n",
          "Toy residual MLP, planted trigger defect, labels from the clean model. "
          "**These validate the harness; they are not results about real language models.**\n",
          f"Failure discovery named the true trigger feature in {disc:.0%} of trials.\n"]
    for title, t in sections:
        md += [f"## {title}\n", t, ""]
    text = "\n".join(md)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)


if __name__ == "__main__":
    main()
