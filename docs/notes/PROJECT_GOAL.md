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
- `src/gauge_compiler.py`
  - minimal structural gauge compiler plus convention-fixed physical compiler
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
- a working matter-sector covariant compiler for:
  - `psibar i gamma^mu D_mu psi`
  - `(D_mu phi)^dagger (D^mu phi)`
- a working pure-gauge compiler for:
  - `-1/4 F_{mu nu} F^{mu nu}`
  - `-1/4 F^a_{mu nu} F^{a mu nu}` with Yang-Mills 3- and 4-gauge vertices

What still needs work before the project feels structurally sound:

- the model layer is still thinner than a real model compiler
- fields carrying repeated identical index kinds are not represented robustly enough yet
- general multi-fermion tensor support is still narrower than a full FeynRules-like system
- background-field-gauge scaffolding, gauge fixing, and ghosts are still missing
- gauge support is broader, but still not BFM-complete
- examples still carry too much of the live regression burden

### Session handoff

Current conventions to remember:

- `vertex_factor(...)` keeps `(2*pi)^d Delta(sum p)` by default
- use `include_delta=False` for the reduced vertex
- `src/examples.py` is the main runnable validation target
- `src/spenso_gamma_checks.py` is a second live validation path for gamma/tensor structures

Recommended interpretation for future work:

- keep the minimal gauge compiler as a structural helper layer
- treat the convention-fixed physical compiler as the user-facing path
- build ordinary gauge fixing first on top of the now-working ordinary matter
  and pure-gauge sectors
- then build BFM-specific background/quantum splitting on top of that
  ordinary gauge-fixed base

Priority now:

1. harden the current ordinary gauge baseline
   - keep conventions frozen in one place across code/docs/tests
   - move the main assertions out of `src/examples.py` into a dedicated test harness
   - fix the current weakness around repeated same-kind index slots
2. add the next ordinary physics sector
   - gauge-fixing declarations and compilation
   - then ghosts on top of the gauge-fixed path
3. only after that, add BFM-specific splitting
   - background/quantum gauge-field splitting
   - background-field-gauge-specific declarations
4. continue with broader physics growth
   - Weyl/Majorana support if needed
   - spontaneous symmetry breaking and field mixing
   - electroweak and later EFT-facing structures

What can reasonably be done next week:

1. extract the current covariant and pure-gauge checks into a first real test suite
2. add one short conventions note that becomes the stable sign/normalization reference
3. tighten index-label handling so fields with repeated identical index kinds do not collapse
4. sketch the model/compiler API for gauge-fixing terms, even if the full sector is not implemented yet
