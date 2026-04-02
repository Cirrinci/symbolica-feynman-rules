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
- one convention-fixed physical compiler for covered matter-sector and pure-gauge cases
- one main regression script
- one focused gamma/tensor validation script

That is a usable baseline, but not yet a durable library layout.

Most important current technical limits:

- conventions are now mostly frozen in code/tests, but still need one cleaner long-lived source of truth
- the ordinary matter and pure-gauge sectors work, but BFM-specific scaffolding is still absent
- examples still carry too much of the live regression burden
- background/quantum splitting, gauge fixing, and ghosts are still absent

### Recommended build order

The safest order from here is:

1. keep gauge-normalization conventions frozen
2. add a real test harness
3. build BFM-specific scaffolding on top of the ordinary gauge sector
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

### Phase 5: BFM-Oriented Gauge Support

Goal:
Extend the ordinary gauge foundation toward something structurally comparable to `UnbrokenSM_BFM`.

Deliverables:

- background/quantum gauge-field splitting
- background-field-gauge declarations
- ghost and gauge-fixing terms
- clearer public API boundaries around the physical compiler

Success criteria:

- the model layer can express the core algorithmic ingredients of a BFM-style gauge model
- BFM-specific structures build on the ordinary gauge compiler instead of bypassing it

### Immediate next tasks

These are the next concrete tasks recommended for the codebase:

1. keep the active conventions documented once across code/docs/tests
2. keep the minimal compiler as a structural helper layer and the physical compiler as the main physics-facing layer
3. add background/quantum gauge-field splitting at the model and compiler level
4. add gauge-fixing declarations and compilation on top of that split
5. add the ghost sector after gauge fixing is in place
6. split the current `src/examples.py` assertions into proper tests
7. improve canonical output for pure-gauge structures so the main display is closer to textbook forms
8. then continue toward broader BFM-style model support

### Immediate implementation order

Use this order for the next work cycle:

1. Background/quantum split
   - extend `GaugeGroup` / model declarations so a gauge sector can distinguish ordinary, background, and quantum gauge fields
   - make the pure-gauge compiler expand with `A -> B + Q` without breaking the ordinary non-BFM case
2. Gauge fixing
   - add explicit model declarations for the gauge-fixing sector
   - compile those terms through the same physical compiler path, not as handwritten examples
3. Ghosts
   - add ghost-field declarations and ghost interactions after gauge fixing is stable
   - keep ghost support in the model/compiler layer, not as a special example-only path
4. Test extraction
   - move matter-sector and pure-gauge convention checks out of `src/examples.py`
   - keep the example script as an inspection/demo tool
5. Output cleanup
   - add a more canonical simplification/display layer for pure-gauge vertices
   - keep the raw vertex available, but make the readable form the default thing to inspect

### Rule of thumb

For each new physics feature, implement it in this order:

1. tensor/index representation
2. model declaration
3. engine support
4. simplification
5. tests

That order keeps the project extensible and reduces example-only logic.
