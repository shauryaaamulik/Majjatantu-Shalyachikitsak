# Changelog

All notable changes to MAJJATANTU are recorded here so human and AI collaborators
(Claude, Cursor agents, etc.) can see what landed and why.

Format: reverse-chronological. Each entry notes phase, intent, and hard-constraint impact.

---

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
- Synthetic corruption experiment with known ground-truth direction
- Wire tiny open model smoke eval for loop detection
- Activation-difference → direction ablation end-to-end baseline script
- Reproduce reference edit pattern from paper logs (layers / column counts) on a small proxy
