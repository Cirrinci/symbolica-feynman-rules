## Roadmap

This roadmap turns the current Symbolica/Spenso prototype into a more coherent
Python analogue of FeynRules.

### Current baseline

Already working in the active `src/` tree:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with permutation-aware momentum assignment
- fermion permutation signs
- stripped and unstripped external fermion factors
- spinor-delta output through Spenso bispinor metrics
- gamma-matrix and generator structures passed as symbolic tensor factors
- a model layer in `src/model.py`
- reusable operator builders in `src/operators.py`
- a minimal gauge compiler in `src/gauge_compiler.py`
- direct/model agreement checks in `src/examples.py`
- gauge-ready examples such as quark-gluon and complex-scalar current interactions
- compiled gauge-model checks for quark-gluon and abelian complex-scalar interactions
- runnable gamma/tensor checks in `src/spenso_gamma_checks.py`

Current implementation entry points:

- `src/model_symbolica.py`
- `src/model.py`
- `src/spenso_structures.py`
- `src/examples.py`

### Current handoff

The current state is better described as:

- one working contraction engine
- one usable model-definition layer
- one reusable operator vocabulary
- one minimal gauge compiler
- one working covariant-derivative compiler for covered matter-sector cases
- one main regression script
- one focused gamma/tensor validation script

That is a usable baseline, but not yet a durable library layout.

Most important current technical limits:

- conventions are now implemented in code but not yet frozen/documented in one authoritative place
- the gauge compiler now covers matter-sector covariant terms, but not the pure-gauge sector
- examples still carry too much of the live regression burden
- pure-gauge and broader gauge-complete structures are still absent

### Recommended build order

The safest order from here is:

1. stabilize gauge-normalization conventions
2. broaden gauge support into the pure-gauge sector
3. add a real test harness
4. then widen multi-fermion and export support

This order matters because new physics features will compound the current
structural weaknesses if they are added first.

### Phase 1: Covariant-derivative compilation

Goal:
Turn gauge interactions into consequences of model declarations rather than hand-assembled terms.

Deliverables:

- builders for `D_mu psi` and `D_mu phi`
- automatic expansion of `|D_mu phi|^2`
- automatic expansion of `psibar i gamma^mu D_mu psi`
- explicit gauge-coupling and charge insertion rules

Success criteria:

- the scalar and fermion gauge currents no longer need to be assembled manually in examples
- the compiler, not the example file, becomes the source of the basic gauge interactions

Status:

- reached for the covered matter-sector cases

### Phase 2: Gauge-structure expansion

Goal:
Broaden the gauge compiler beyond the current minimal abelian/non-abelian current coverage.

Deliverables:

- non-abelian scalar gauge interactions
- explicit field-strength structures
- later 3-gauge and 4-gauge self-interactions

Success criteria:

- QED-like and Yang-Mills-like interactions arise from model/compiler logic
- gauge structures are no longer just a small handwritten subset

### Phase 3: Validation and test structure

Goal:
Make verification trustworthy and easy to run as the compiler grows.

Deliverables:

- move runnable assertions from `src/examples.py` toward a dedicated test suite
- keep `src/examples.py` as a showcase script, not the only regression harness
- keep `src/spenso_gamma_checks.py` as a focused tensor-identity sandbox

Success criteria:

- both the example script and the gamma sandbox remain useful
- the core compiler behavior is covered by repeatable tests

### Phase 4: Model-layer and usability growth

Goal:
Push the current thin model/compiler layer toward a more complete FeynRules-style workflow.

Deliverables:

- richer model declarations for transformations and later field strengths
- clearer normalization and convention handling
- eventual export-facing workflows

Success criteria:

- the model layer becomes the main place where interactions are declared
- the codebase starts to resemble a durable library rather than an example-driven prototype

### Immediate next tasks

These are the next concrete tasks recommended for the codebase:

1. freeze and document the active conventions once across code/docs/tests
2. split the current `src/examples.py` assertions into proper tests
3. extend the compiler into pure non-abelian gauge self-interactions
4. keep the minimal compiler as a structural helper layer and make the covariant compiler the main physics-facing layer
5. then continue toward broader gauge-complete structures

### Rule of thumb

For each new physics feature, implement it in this order:

1. tensor/index representation
2. model declaration
3. engine support
4. simplification
5. tests

That order keeps the project extensible and reduces example-only logic.
