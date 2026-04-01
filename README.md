## Symbolica + Spenso Feynman-Rule Prototype

This repository explores a Python-based analogue of FeynRules built around:

- Symbolica for symbolic expressions, rewriting, and simplification
- Spenso for tensor structures, spinor/Lorentz objects, and index-aware building blocks

The current codebase is centered on a Symbolica contraction engine plus a thin
model layer that maps FeynRules-style declarations into that engine.

### Repository layout

Main source files:

- `src/model_symbolica.py`
  - core contraction engine
  - direct parallel-list API
  - simplification helpers
  - compact derivative-sum helpers
- `src/model.py`
  - model-layer dataclasses
  - `Field`, `FieldOccurrence`, `ExternalLeg`, `DerivativeAction`,
    `InteractionTerm`, `Model`
  - bridge from model declarations to `vertex_factor(...)`
- `src/spenso_structures.py`
  - Spenso-backed wrappers for gamma matrices, metrics, and gauge generators
- `src/examples.py`
  - runnable examples and regression checks
  - covers both the direct API and the model layer
- `src/spenso_gamma_checks.py`
  - focused gamma/tensor sandbox
  - currently stale and not runnable as-is because it still imports `model_legacy`

Supporting notes live under `docs/notes/`.

### Current status

What is working in the active code path:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with permutation-aware momentum assignment
- fermion permutation signs
- stripped and unstripped external fermion factors
- open spinor-index output through Spenso bispinor metrics
- gamma-matrix and gauge-generator structures supplied through wrappers
- a model-layer API that compiles to the engine
- direct/model cross-checks in the main regression script
- gauge-ready examples such as quark-gluon and complex-scalar current structures

What is not yet solid:

- object-based matching is incomplete; parts of the engine still rely on string comparisons
- the model layer does not yet distinguish all non-fermion roles precisely
- `src/spenso_gamma_checks.py` is stale
- general multi-fermion tensor support is still narrower than a full FeynRules-like system
- gauge-boson self-interactions and a broader gauge-model workflow are not implemented

### Conventions

The main engine entry point is:

- `vertex_factor(...)` in `src/model_symbolica.py`

It accepts either:

- model-layer input: `interaction=...`, `external_legs=...`
- direct input: `coupling`, `alphas`, `betas`, `ps`, plus optional role/index metadata

Important output conventions:

- `strip_externals=True` by default
  - external wavefunctions are amputated from the displayed vertex
- `include_delta=True` by default
  - the returned expression keeps the overall momentum-conservation factor
    `(2*pi)^d Delta(sum p)`
- use `include_delta=False` when you want the reduced vertex with that universal factor stripped

Related helpers:

- `simplify_deltas(...)`
- `simplify_spinor_indices(...)`
- `simplify_vertex(...)`
- `compact_vertex_sum_form(...)`
- `compact_sum_notation(...)`

### Main workflow

The current workflow is:

1. declare an interaction, either directly or through `InteractionTerm`
2. provide external legs
3. let `vertex_factor(...)` perform contraction sums and derivative bookkeeping
4. optionally simplify the result with `simplify_vertex(...)` or more targeted helpers

The main runnable entry point for this workflow is `src/examples.py`.

### Setup

Create or refresh the local virtual environment:

- `bash setup_env.sh`

This installs the dependencies listed in `requirements.txt` into `.venv`.

### Usage

Run the main example and regression script from the repository root:

- `./.venv/bin/python src/examples.py`
- `./.venv/bin/python src/examples.py --suite scalar`
- `./.venv/bin/python src/examples.py --suite fermion`
- `./.venv/bin/python src/examples.py --suite gauge`
- `./.venv/bin/python src/examples.py --suite model`
- `./.venv/bin/python src/examples.py --suite cross`

Notes:

- `src/examples.py --suite all` is the main validation target
- `src/spenso_gamma_checks.py` is currently stale and needs an import update before it can be used again

For notebooks, use the repository virtual environment:

- `.venv/bin/python`

### Current notes

Project notes are kept in:

- `docs/notes/PROJECT_GOAL.md`
- `docs/notes/ROADMAP.md`
- `docs/notes/RESEARCH_LOG.md`
- `docs/notes/THESIS_PROGRESS.md`
- `docs/notes/FEYNRULES_STYLE_STRATEGY.md`

### Immediate priorities

The highest-value next steps in the codebase are:

1. remove the remaining string-based matching from the engine
2. repair `src/spenso_gamma_checks.py`
3. tighten the model-layer role and index semantics
4. centralize tensor/operator builders instead of hand-building structures in examples
5. move the runnable assertions toward a dedicated test harness
