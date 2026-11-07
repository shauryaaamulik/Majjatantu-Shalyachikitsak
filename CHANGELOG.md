# Changelog

All notable changes to MAJJATANTU are recorded here so human and AI collaborators
(Claude, Cursor agents, etc.) can see what landed and why.

Format: reverse-chronological. Each entry notes phase, intent, and hard-constraint impact.

---

## [0.2.0] — 2026-11-07

### Synthetic testbed with planted ground-truth defect (Claude session)

Implements the "Synthetic corruption experiment with known ground-truth direction" item
from *Planned next*. Lives in `src/majjatantu/synthetic/` and is **torch-free** (NumPy +
the existing pydantic records), so it runs and is tested without a GPU or model download.

#### Added
- `synthetic/toy_model.py` — residual MLP; `W_out` has the same [out, in] layout as an MLP output projection. Forward passes are charged to the existing `ForwardPassBudget` (1 batch call = 1 pass).
- `synthetic/planted.py` — plants a trigger-detector defect in spare hidden units; ground truth (units, write direction, trigger) is known; `Splits` keeps the test split behind `reveal_test()`.
- `synthetic/discovery.py` — failure-slice discovery (Wilson lower bound) + reproducibility check producing a `FailureReport`.
- `synthetic/localization.py` — per-unit activation patching, *anytime* under a budget.
- `synthetic/repair.py` — gradient-free operators (`ablate_direction` mirrors `weight_surgery.direction_ablation` semantics; `zero_columns`) and four strategies: `random_k`, `diffmeans_topk` (README baseline 5), `diffmeans_all`, `causal_k`.
- `synthetic/evaluate.py`, `synthetic/runner.py` — accept/reject gate (target gain, regression, parameter budget, save/reload stability); every candidate edit, accepted or rejected, is written to the `EditLog`.
- `tests/test_synthetic.py` — 13 tests.

#### Added (experiment)
- `experiments/synthetic_failures/run.py` → `results/synthetic_failures.md` (30 seeds; budget sweep 10/25/50/100; defect size 1/4/8).

#### Findings (synthetic only; do not generalize)
- Failure discovery named the true trigger feature in 100% of trials.
- `diffmeans_topk` (1 pass) was accepted in 100% of trials and reached ≈0.05 test error; `causal_k` reached exact repair (0.00) but needed ≈66 passes (accepted 33% at budget 50, 0% at ≤25). The cheap non-LLM baseline matched causal localization on acceptance in this easy setting.
- Dense direction ablation (`diffmeans_all`) lowers target error but costs ≈7% retention and 43% of parameters; the gate rejects it.
- Gate bug caught while running: a 5% toy parameter cap rejected a perfect 8-unit repair (5.3%); cap set to 6%.

#### Verification status
- Torch is **not installable** in this session's sandbox (package index unreachable), so the torch-based modules and their tests from 0.1.0 were **not re-run** here. New synthetic tests: 13/13 pass.

## [0.1.0] — 2026-11-07

### Phase I scaffolding (foundations)

Repo previously contained **only** `README.md` (plan, no code). This release starts
Phase I from README §13: inspection toolkit, direction-ablation baseline, eval harness.

#### Added
- `pyproject.toml` — installable `majjatantu` package (PyTorch / Transformers / Pydantic).
- Package layout matching README §12 under `src/majjatantu/`:
  - `models/` — causal-LM loader + SHA-256 checkpoint hashing + state clone/rollback
  - `introspection/` — weight summaries (incl. top singular values), streaming activation capture, mean activation-difference helper
  - `failure_detection/` — loop / token-limit / n-gram repetition detector (Exp. 8 style)
  - `weight_surgery/` — **gradient-free** direction ablation operator (README baseline 5)
  - `evaluation/` — generation eval harness + accept/reject regression gates
  - `orchestration/` — experiment state, forward-pass budget, rollback, edit recording
  - `reporting/` — JSONL `EditLog` (accepted **and** rejected edits)
  - `causal_localization/` — activation-patch helper stub (Phase II will expand)
  - `agent/`, `optimization/` — placeholders for Phase III
- `configs/phase1_defaults.yaml` — default budgets and gate settings
- Unit tests for direction ablation, budgets, weight summaries, edit log, patch decision
- `scripts/smoke_direction_ablation.py` — CPU-only synthetic Linear ablation smoke test
- CLI: `majjatantu info`

#### Constraints enforced in code
- Repairs do **not** use training loops / gradients (`weight_surgery.direction_ablation`).
- `ForwardPassBudget` raises if exceeded; eval/failure paths consume passes explicitly.
- Every edit can be logged with model hash before/after via `EditLog`.
- Parameter-fraction budget checked before applying ablation (`ParameterBudget`).

#### README fixes
- Status line updated: code scaffolding exists; still R&D, no empirical results claimed.
- §12 note updated: intended layout is now partially implemented.
- Added pointer to this `CHANGELOG.md` for AI collaborators (§15).

#### Known gaps (intentional — later phases)
- No real Qwen / Gemma repair runs yet
- No TransformerLens / NNsight integration wired
- LLM research agent not implemented
- Activation patching is a stub, not a full causal-tracing suite
- Frontend / FastAPI / Docker not started

---

## [Unreleased]

### Verified — 2026-11-07
- Created conda env `majjatantu` (Python 3.11); system Python 3.14 lacks usable torch wheels.
- `pip install -e ".[dev]"` succeeded; **11/11 pytest passed**; `scripts/smoke_direction_ablation.py` OK.

### Planned next
- ~~Synthetic corruption experiment with known ground-truth direction~~ (done in 0.2.0)
- Harder synthetic defects where diff-means is *not* sufficient (entangled units, distributed direction)
- Unify `synthetic.evaluate.judge` with `evaluation.harness.decide_patch` (needs torch-optional `evaluation/__init__`)
- Wire tiny open model smoke eval for loop detection
- Activation-difference → direction ablation end-to-end baseline script
- Reproduce reference edit pattern from paper logs (layers / column counts) on a small proxy
