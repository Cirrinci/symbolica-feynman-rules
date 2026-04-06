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
- `src/operators.py`
  - reusable operator builders for bilinears, currents, and scalar-gauge structures
- `src/gauge_compiler.py`
  - minimal structural gauge compiler plus convention-fixed physical compiler
  - covers minimal matter currents/contact terms, matter covariant derivatives,
    and pure-gauge kinetic / Yang-Mills self-interactions
- `src/examples.py`
  - runnable examples and regression checks
  - covers both the direct API and the model layer
- `src/spenso_gamma_checks.py`
  - focused gamma/tensor sandbox
  - runnable against the live source tree

Supporting notes live under `docs/notes/`.

Walkthrough material:

- `notebooks/codebase_workflow_walkthrough.ipynb`
  - first end-to-end walkthrough of the live code path
  - currently still work in progress, but already runnable

### Current status

What is working in the active code path:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with permutation-aware momentum assignment
- fermion permutation signs
- stripped and unstripped external fermion factors
- open spinor-index output through Spenso bispinor metrics
- gamma-matrix and gauge-generator structures supplied through wrappers
- reusable operator builders in `src/operators.py`
- a model-layer API that compiles to the engine
- direct/model cross-checks in the main regression script
- gauge-ready examples such as quark-gluon and complex-scalar current structures
- a minimal gauge compiler driven by `GaugeGroup`, `GaugeRepresentation`, and field metadata
- a convention-fixed physical compiler for:
  - `psibar i gamma^mu D_mu psi`
  - `(D_mu phi)^dagger (D^mu phi)`
  - `-1/4 F_{mu nu} F^{mu nu}`
  - `-1/4 F^a_{mu nu} F^{a mu nu}` with Yang-Mills 3- and 4-gauge vertices
- compiled gauge-model checks for quark-gluon and abelian complex-scalar interactions
- runnable gamma/tensor checks in `src/spenso_gamma_checks.py`

What is not yet solid:

- general multi-fermion tensor support is still narrower than a full FeynRules-like system
- most regression checks still live in `src/examples.py` instead of a dedicated test harness
- background-field-gauge scaffolding, gauge fixing, and ghosts are still absent
- the public API boundary between the minimal structural compiler and the
  physical compiler should be tightened further

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

Gauge/compiler conventions:

- derivatives map to `-i p_mu`
- `vertex_factor(...)` contributes the universal overall `+i`
- matter uses `D_mu = partial_mu + i g A_mu`
- pure gauge uses
  `F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu`

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
- `./.venv/bin/python src/examples.py --suite covariant`
- `./.venv/bin/python src/examples.py --suite cross`

Notes:

- `src/examples.py --suite all` is the main validation target
- `./.venv/bin/python src/spenso_gamma_checks.py` is a second live validation path for gamma/tensor structures

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

1. keep conventions documented once across code, docs, and tests
2. move the runnable assertions in `src/examples.py` toward a dedicated test harness
3. fix repeated same-kind index-slot handling in the model/compiler boundary
4. add gauge-fixing declarations and compilation through the physical compiler path
5. add ghost-sector support after gauge fixing is stable
6. add background/quantum gauge-field splitting on top of the ordinary gauge-fixed path
7. improve canonical/readable pure-gauge output while keeping the raw compiled form available

Suggested implementation order:

1. keep conventions frozen in one place
2. extract tests from `src/examples.py`
3. fix repeated same-kind index-slot handling
4. add ordinary gauge fixing
5. add the ghost sector
6. add background/quantum gauge-field splitting
7. improve canonical/readable pure-gauge output on top of the raw compiled form
