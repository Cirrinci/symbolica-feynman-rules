## Symbolica Feynman Rule Prototype

This repository focuses on deriving Feynman vertices directly with Symbolica,
using canonical quantization plus Wick-contraction sums.

Main code path:

- `code/model_symbolica.py` is the primary implementation.
- `code/examples_symbolica.py` contains runnable examples/tests for this implementation.
- `Notebooks/symbolica_interaction_term.ipynb` is the interactive notebook walkthrough.

Legacy prototype files have been archived outside this repository path.

### Current scope

- Bosonic polynomial interactions.
- Permutation-summed Wick contractions.
- Derivative interactions via momentum factors.
- Fermionic role-aware contractions with permutation signs.
- Optional spinor-index handling through Spenso bispinor metrics.
- Amputated fermion vertices with open spinor indices when bilinear
  contractions are inferable from the Lagrangian.

### Current status

The code is in a good state for the currently covered examples, but it is not
yet a general FeynRules-like fermion engine.

What is solid right now:

- scalar polynomial and derivative vertices
- fermion bilinears encoded by repeated dummy spinor labels, e.g.
  `field_spinor_indices=[alpha, alpha, None]` for `psibar psi phi`
- amputated open-index output for scalar fermion bilinears, e.g.
  `i y g(i1,i2)` for Yukawa and
  `-i g [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]` for `-(g/2)(psibar psi)^2`
- unamputated matrix-element output with `UF/UbarF` kept explicitly
- rejection of underspecified multi-fermion products such as bare
  `psi * psibar * psi * psibar` with no spinor-contraction data

What is not solid yet:

- general multi-fermion operators whose spinor structure lives in the coupling
  tensor rather than in repeated dummy labels on the fields
- automatic remapping of open spinor labels inside the coupling to the external
  leg indices
- general gamma-chain extraction from a Lagrangian

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

### Tomorrow

If resuming work tomorrow, start here:

1. Unify the fermion open-index story for couplings that already carry spinor
   labels, such as `gamma(mu,i,j)`.
2. The key missing feature is: map field spinor slots to external leg spinor
   labels and substitute that map into `coupling` for each compatible
   contraction, instead of only adding `g(i,j)` metrics for repeated dummy
   bilinears.
3. After that, add regression tests showing that explicit
   `leg_spinor_indices=[i1,i2,...]` propagate into the coupling for:
   - `psibar gamma^mu psi A_mu`
   - a four-fermion current-current operator such as
     `(psibar gamma^mu psi)(psibar gamma_mu psi)`
4. Only after those tests pass should we broaden the allowed multi-fermion
   validation beyond repeated-dummy scalar bilinears.

Recommended first command tomorrow:

- `./.venv/bin/python code/examples_symbolica.py --suite fermion`

### File overview

- `code/model_symbolica.py`
  - Main symbolic engine and simplification helpers.
  - Supports both bosonic and fermionic workflows in the current scope.

- `code/examples_symbolica.py`
  - Scripted examples for scalar, derivative, and fermion-structure checks.
  - Good quick validation target after edits.

- `Notebooks/symbolica_interaction_term.ipynb`
  - Step-by-step notebook using the same API as `model_symbolica.py`.

### Setup

Create or refresh the local virtual environment:

- `bash setup_env.sh`

This installs dependencies from `requirements.txt` into `.venv`.

### Usage

Recommended run commands from repository root:

- `./.venv/bin/python code/examples_symbolica.py`
- `./.venv/bin/python code/examples_symbolica.py --suite scalar`
- `./.venv/bin/python code/examples_symbolica.py --suite fermion`

For notebooks, ensure the kernel uses the repository virtual environment:

- `.venv/bin/python`

### Roadmap

Short term:

- Improve fermion-chain automation for gamma/index structures.
- Add more regression tests for mixed scalar-fermion permutations.
- Expand compact-sum formulas for broader derivative patterns.

Medium term:

- Extend tensor/index handling toward gauge-field interactions.
- Add stronger notebook-to-script parity tests.
