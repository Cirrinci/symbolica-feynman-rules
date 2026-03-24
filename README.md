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

### Core pipeline

The function `vertex_factor(...)` in `model_symbolica.py` performs:

1. Contraction sum over external legs.
2. Derivative momentum factors per contraction.
3. Plane-wave replacement by momentum conservation delta.
4. External wavefunction stripping (optional).
5. Final multiplication by `i`.

Related helpers in `model_symbolica.py`:

- `contract_to_full_expression`
- `infer_derivative_targets`
- `simplify_deltas`
- `simplify_spinor_indices`
- `simplify_vertex`
- `compact_vertex_sum_form`
- `compact_sum_notation`

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
