## Project Goal

Implement a Python-based analogue of FeynRules using Symbolica and Spenso.

### Core objective

Build a reusable symbolic pipeline that starts from a field-theory model
declaration and derives Feynman interaction vertices in Python, using:

- Symbolica for symbolic expressions, rewriting, simplification, and rule-based manipulation
- Spenso for tensor structures, indices, Lorentz/spinor objects, and gauge-ready building blocks

### Current architecture

The active repository layout is now `src/`, not `code/`.

Core pieces:

- `src/model_symbolica.py`
  - contraction engine
  - direct API
  - simplification helpers
- `src/model.py`
  - model-layer dataclasses
  - translation from structured interactions into the engine inputs
- `src/spenso_structures.py`
  - tensor wrappers for gamma matrices, metrics, and generators
- `src/examples.py`
  - examples, regression checks, and direct/model cross-checks

### Target capabilities

- define fields, parameters, and gauge groups in Python objects
- encode interaction terms in a model-layer format
- derive vertices from either structured model objects or lower-level direct input
- perform Wick-contraction style combinatorics for external legs
- handle derivative interactions as momentum factors
- track tensor, Lorentz, spinor, and later gauge indices with Spenso objects
- simplify Kronecker deltas, spinor metrics, and basic tensor structures
- expose results either with or without the overall momentum-conservation delta
- support scalar, fermion, and gauge-ready interaction patterns within one engine

### Working interpretation

The project is no longer just a collection of notebook experiments.

The right direction is:

- keep `src/model_symbolica.py` as the engine
- keep `src/model.py` as the declarative layer
- move reusable tensor/operator definitions out of example code
- remove engine logic that still depends on stringified symbols or indices
- grow toward a model-driven FeynRules-style workflow, not more parallel-list plumbing

### Current support boundary

What is already credible in the code:

- scalar interactions
- derivative interactions
- fermion sign handling
- amputated open-index fermion output
- gamma-matrix and generator structures passed through wrapped symbolic objects
- a first model-layer bridge to the engine

What still needs work before the project feels structurally sound:

- the model layer is still thinner than a real model compiler
- there is still no covariant-derivative compiler
- gauge support is broader, but still not gauge-complete
- examples still carry too much of the live regression burden

### Session handoff

Current conventions to remember:

- `vertex_factor(...)` keeps `(2*pi)^d Delta(sum p)` by default
- use `include_delta=False` for the reduced vertex
- `src/examples.py` is the main runnable validation target
- `src/spenso_gamma_checks.py` is a second live validation path for gamma/tensor structures

Recommended interpretation for future work:

- prioritize turning the current minimal gauge compiler into a covariant-derivative compiler
- broaden gauge support only after the compiler/model boundary for `D_mu` structures is stable
