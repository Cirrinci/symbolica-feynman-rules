## Symbolica Feynman Rule Prototype

This repository focuses on deriving Feynman vertices directly with Symbolica,
using canonical quantization plus Wick-contraction sums.

Main code path:

- `code/model_symbolica.py` is the primary implementation.
- `code/examples_symbolica.py` contains runnable examples/tests for this implementation.
- `code/spenso_gamma_checks.py` contains focused Spenso/gamma-matrix experiments and checks.
- `Notebooks/symbolica_interaction_term.ipynb` is the interactive notebook walkthrough.

Legacy prototype files have been archived outside this repository path.

### Current scope

- Bosonic polynomial interactions.
- Permutation-summed Wick contractions.
- Derivative interactions via momentum factors.
- Fermionic role-aware contractions with permutation signs.
- Optional spinor-index handling through Spenso bispinor metrics.
- Amputated fermion vertices with open spinor indices for both repeated-dummy
  bilinears and explicit coupling tensors such as `gamma(mu, i_bar, i_psi)`.

### Current status

The code is in a good state for the currently covered examples, but it is not
yet a general FeynRules-like fermion engine.

What is solid right now:

- scalar polynomial and derivative vertices
- fermion bilinears encoded by repeated dummy spinor labels, e.g.
  `field_spinor_indices=[alpha, alpha, None]` for `psibar psi phi`
- open-spinor remapping inside explicit coupling tensors such as
  `gamma(mu, i_bar, i_psi)`
- amputated open-index output for scalar fermion bilinears, e.g.
  `i y g(i1,i2)` for Yukawa and
  `-i g [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]` for `-(g/2)(psibar psi)^2`
- amputated open-index output for a current-current operator such as
  `gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)`
- unamputated matrix-element output with `UF/UbarF` kept explicitly
- rejection of underspecified multi-fermion products such as bare
  `psi * psibar * psi * psibar` with no spinor-contraction data

What is not solid yet:

- general multi-fermion operators whose spinor structure lives in the coupling
  tensor beyond the currently exercised bilinear/current-current patterns
- general gamma-chain extraction from a Lagrangian
- normalization and symmetry-factor conventions are still script-level choices
  rather than a single centralized policy

Physics convention to keep in mind:

- the physically correct fermion vertex is the amputated open-index object
- the unstripped output is a matrix element diagnostic, not the final vertex
- a four-fermion term should not become `0` just because the external spinors
  were stripped; if it does, the spinor structure was erased too early

### Core pipeline

The function `vertex_factor(...)` in `model_symbolica.py` performs:

1. Contraction sum over external legs.
2. Derivative momentum factors per contraction.
3. Plane-wave replacement by momentum conservation delta.
4. External wavefunction amputation to an open-index vertex (optional).
5. Final multiplication by `i`.

Related helpers in `model_symbolica.py`:

- `contract_to_full_expression`
- `infer_derivative_targets`
- `simplify_deltas`
- `simplify_spinor_indices`
- `simplify_vertex`
- `compact_vertex_sum_form`
- `compact_sum_notation`

### Current session checklist (2026-03-26)

Recently completed:

1. Explicit open-spinor labels in the coupling are remapped to external leg
   spinor slots during compatible fermion contractions.
2. Runnable regressions now cover explicit gamma-current examples, including:
   - `psibar gamma^mu psi A_mu`
   - `gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)`
3. Focused gamma checks now display the simplified current-current structures
   clearly enough to inspect direct and exchange terms.

Next:

1. Centralize normalization and symmetry-factor conventions for fermion
   operators.
2. Add validation for ambiguous encodings, especially repeated field spinor
   labels combined with explicit tensor endpoints in the coupling.
3. Move the script-level checks toward a dedicated test harness.

Recommended first command:

- `./.venv/bin/python code/examples_symbolica.py --suite fermion`

### File overview

- `code/model_symbolica.py`
  - Main symbolic engine and simplification helpers.
  - Supports both bosonic and fermionic workflows in the current scope.

- `code/examples_symbolica.py`
  - Scripted examples for scalar, derivative, and fermion-structure checks.
  - Good quick validation target after edits.

- `code/spenso_gamma_checks.py`
  - Focused gamma/gamma5/Clifford-identity sandbox.
  - Good place to extend spinor/Lorentz tensor experiments before adding gauge fields.

- `Notebooks/symbolica_interaction_term.ipynb`
  - Step-by-step notebook using the same API as `model_symbolica.py`.

### Setup

Create or refresh the local virtual environment:

- `bash setup_env.sh`

This installs dependencies from `requirements.txt` into `.venv`.

### Usage

Recommended run commands from repository root:

- `./.venv/bin/python code/examples_symbolica.py`
- `./.venv/bin/python code/spenso_gamma_checks.py`
- `./.venv/bin/python code/examples_symbolica.py --suite scalar`
- `./.venv/bin/python code/examples_symbolica.py --suite fermion`

For notebooks, ensure the kernel uses the repository virtual environment:

- `.venv/bin/python`

### Roadmap

Short term:

- Centralize normalization/symmetry-factor conventions for fermion operators.
- Add more regression tests for mixed scalar-fermion permutations.
- Expand compact-sum formulas for broader derivative patterns.
- Add guardrails for ambiguous fermion spinor-label encodings.

Medium term:

- Extend tensor/index handling toward gauge-field interactions.
- Add stronger notebook-to-script parity tests.
