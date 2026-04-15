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
- convention-fixed covariant matter compilation for the covered fermion and scalar cases
- pure-gauge Yang-Mills 2/3/4-point compilation
- ordinary gauge-fixing compilation
- ordinary non-abelian ghost compilation
- dedicated `pytest` coverage for the core covariant / pure-gauge / ordinary gauge-fixed matrix
- a walkthrough notebook in `notebooks/codebase_workflow_walkthrough.ipynb`
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
- one convention-fixed physical compiler for covered ordinary gauge-theory cases
- one growing dedicated `pytest` layer
- one main demo/regression script
- one walkthrough notebook
- one focused gamma/tensor validation script

That is a usable baseline, but not yet a durable library layout.

Most important current technical limits:

- conventions are now mostly frozen and have one main long-lived source of truth, but they still need to stay stable across future compiler growth
- repeated same-kind index slots are handled for the covered compiler paths, but the broader model/declaration layer still needs further hardening
- general multi-fermion tensor support is still intentionally narrow
- the ordinary gauge-fixed baseline now works, but BFM-specific scaffolding is still absent
- examples still carry too much of the broader direct/model regression burden
- background/quantum splitting and background-field-gauge-specific declarations are still absent
- raw pure-gauge / gauge-fixing / ghost output is still less canonical than the compact display forms used in demos and tests

### Recommended build order

The safest order from here is:

1. keep gauge-normalization conventions frozen
2. add a real test harness
3. tighten the remaining model/declaration validation outside the current compiler entry points
4. add background/quantum splitting on top of the current ordinary gauge sector
5. add BFM gauge fixing and ghosts on top of that split
6. improve canonical output for the ordinary and BFM gauge sectors
7. then widen multi-fermion, symmetry-breaking, and export support

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

Status:

- reached for the ordinary abelian and Yang-Mills pure-gauge sector

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

Status:

- reached for the current core compiler matrix, with broader extraction still ongoing

### Phase 4: Index-signature hardening

Goal:
Make the current generic index story actually safe before broader model growth.

Deliverables:

- represent repeated same-kind field slots without collapsing them into one `kind -> label` entry
- tighten compatibility checks so they can depend on richer index signatures when needed
- keep the engine generic while moving slot-specific meaning into model metadata

Success criteria:

- a field carrying two slots of the same representation can be compiled without label loss
- future adjoint/family/flavor extensions do not require ad hoc workarounds

Status:

- reached for the covered compiler paths, with broader model/declaration hardening still remaining

### Phase 5: Ordinary gauge fixing and ghosts

Goal:
Add the next physical gauge-theory sector on top of the now-working ordinary matter and pure-gauge foundation.

Deliverables:

- gauge-fixing declarations at the model level
- compilation of gauge-fixing terms through the physical compiler path
- ghost-field declarations and ghost interactions after gauge fixing is stable

Success criteria:

- gauge fixing is expressed by model/compiler logic rather than handwritten examples
- ghosts are generated from the same conventions instead of being bolted on separately

Status:

- reached for the ordinary unbroken gauge-fixed path

### Phase 6: Model-layer and usability growth

Goal:
Push the current thin model/compiler layer toward a more complete FeynRules-style workflow.

Deliverables:

- richer model declarations for transformations and later field strengths
- clearer normalization and convention handling
- eventual export-facing workflows

Success criteria:

- the model layer becomes the main place where interactions are declared
- the codebase starts to resemble a durable library rather than an example-driven prototype

### Phase 7: BFM-Oriented Gauge Support

Goal:
Extend the ordinary gauge foundation toward something structurally comparable to `UnbrokenSM_BFM`.

Deliverables:

- background/quantum gauge-field splitting
- background-field-gauge declarations
- BFM gauge-fixing and ghost terms
- clearer public API boundaries around the physical compiler

Success criteria:

- the model layer can express the core algorithmic ingredients of a BFM-style gauge model
- BFM-specific structures build on the ordinary gauge compiler instead of bypassing it

### Phase 8: Broader physics growth

Goal:
Move beyond the current ordinary gauge baseline once the foundation is durable.

Deliverables:

- wider multi-fermion tensor support
- Weyl/Majorana support if thesis scope requires it
- spontaneous symmetry breaking, vevs, and field mixing
- electroweak and later EFT-facing operators

Success criteria:

- the project can express more realistic model-building workflows without losing the current clarity of conventions

### Immediate next tasks

These are the next concrete tasks recommended for the codebase:

1. keep the active conventions documented once across code/docs/tests
2. keep the minimal compiler as a structural helper layer and the physical compiler as the main physics-facing layer
3. keep widening the dedicated `pytest` split beyond the current core matrix
4. tighten the remaining model/declaration validation outside the current compiler entry points
5. add background/quantum gauge-field splitting on top of the ordinary gauge-fixed path
6. add BFM gauge fixing and ghosts on top of that split
7. improve canonical output for pure-gauge, gauge-fixing, and ghost structures
8. then continue toward broader BFM-style and post-BFM model support

### Priority now

There are two distinct priorities and they should not be conflated:

1. Lagrangian.feynman_rule() doesnt accept a whole lagrangian without giving all the fields for one specific vertex.

2. L = Lagrangian(terms=(term_dphi4,)) unnatural!!
3. Ghost and gauge fixing terms are unnatural. possibily to handwrite them.
4. immediate codebase priority
   - user interface
   - conventions
   - tests
   - remaining declaration/model hardening
   - output cleanup
5. next physics priority
   - background/quantum splitting
   - BFM gauge fixing and ghosts
6. later structural extension
   - broader fermion / symmetry-breaking / EFT layers

### Immediate implementation order

Use this order for the next work cycle:

1. Test extraction
   - keep moving broader direct/model checks out of `src/examples.py`
   - keep the example script as an inspection/demo tool
2. Convention/source-of-truth cleanup
   - keep one short reference for Fourier/sign/vertex conventions
   - use it as the thing code and tests are checked against
3. Remaining model/declaration hardening
   - keep compiler-side validation strict
   - reduce permissive object-resolution paths outside the current compiler entry points
4. Background/quantum split
   - extend `GaugeGroup` / model declarations so a gauge sector can distinguish ordinary, background, and quantum gauge fields
   - make the pure-gauge compiler expand with `A -> B + Q` without breaking the ordinary non-BFM case
5. BFM gauge fixing and ghosts
   - build them on top of the split gauge-field declarations, not as a separate path
6. Output cleanup
   - add a more canonical simplification/display layer for pure-gauge, gauge-fixing, and ghost vertices
   - keep the raw vertex available, but make the readable form the default thing to inspect

### What can be done next week

This is the realistic next-week slice:

1. widen the dedicated `pytest` split beyond the current core compiler matrix
2. keep the conventions note stable and use it as the reference for the next gauge-sector work
3. draft and implement the declaration/model API for background and quantum gauge fields
4. extend the pure-gauge compiler with `A -> B + Q` without breaking the ordinary non-BFM path

### Rule of thumb

For each new physics feature, implement it in this order:

1. tensor/index representation
2. model declaration
3. engine support
4. simplification
5. tests

That order keeps the project extensible and reduces example-only logic.
