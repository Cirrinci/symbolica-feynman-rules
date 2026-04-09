## Research Log

This file is a chronological summary of the work completed so far on the
Symbolica/Spenso FeynRules-style prototype.

### Current status snapshot

As of 2026-04-09:

- the active source tree is `src/`
- the main runnable validation script is `src/examples.py`
- the core engine is `src/model_symbolica.py`
- the current model layer is `src/model.py`
- reusable operator builders live in `src/operators.py`
- the minimal gauge compiler lives in `src/gauge_compiler.py`
- the walkthrough notebook is `notebooks/codebase_workflow_walkthrough.ipynb`
- `src/spenso_gamma_checks.py` is runnable against the current source tree
- the long-term goal remains a Python analogue of FeynRules using Symbolica for symbolic rewriting and Spenso for tensor/index structures

### 2026-03-11: repository setup

What happened:

- initial repository was created
- project structure and version control baseline were added

What this achieved:

- established the working environment for the prototype

### 2026-03-14: thesis-workspace integration

What happened:

- the prototype was folded into the broader MSc thesis workspace

What this achieved:

- gave the project a stable home inside the actual thesis repository

### 2026-03-16 to 2026-03-17: scalar prototype

What happened:

- the first symbolic pipeline for scalar interactions was implemented
- early derivative handling and repository documentation were added

What this achieved:

- proved that Symbolica could support the basic contraction and rewriting logic
- established a first working reference point for later extensions

### 2026-03-19: fermion signs and Symbolica-first direction

What happened:

- fermionic sign handling and delta/index structure were introduced
- the implementation direction became more explicitly Symbolica-native

What this achieved:

- moved the project beyond a scalar-only proof of concept
- established the first serious path toward a mixed scalar/fermion engine

### 2026-03-23 to 2026-03-24: mixed scalar, fermion, and derivative support

What happened:

- scalar, fermion, and derivative cases were made to work together
- delta handling and Spenso integration were improved
- older prototype structure was cleaned up

What this achieved:

- reached the first genuinely usable symbolic core
- improved trust in the simplification layer
- reduced dependence on notebook-only logic

### 2026-03-24 to 2026-03-26: first tensor-structured fermion work

What happened:

- gamma-matrix and related spinor structures were introduced through Spenso-backed wrappers
- explicit open-spinor remapping inside coupling tensors was added for the covered fermion patterns
- current-current and gauge-ready fermion examples were added

What this achieved:

- shifted the project from plain fermion combinatorics toward actual tensor-structured fermion vertices
- demonstrated that open-index amputated output is viable in the current design

### 2026-03-31: repository refactor review

What happened:

- the active tree now consists of `src/model_symbolica.py`, `src/model.py`,
  `src/spenso_structures.py`, and `src/examples.py`
- the main example suite was confirmed runnable
- the documentation was updated to describe the live `src/` layout instead of the archived `code/` layout
- a code review identified the main structural gaps still worth addressing

What this achieved:

- clarified the actual architecture of the current repository
- separated working validation paths from stale scripts
- established a sharper set of next engineering tasks

### 2026-04-01: engine hardening and minimal gauge-model compiler

What happened:

- role handling in the engine/model boundary was hardened around typed field roles
- reusable operator builders were moved into `src/operators.py`
- `src/spenso_gamma_checks.py` was brought back onto the live source layout
- a minimal model-driven gauge compiler was added in `src/gauge_compiler.py`
- `src/examples.py` was extended to exercise compiled gauge interactions and print them clearly in both demo and validation paths

What this achieved:

- removed the main dependency on ad hoc operator assembly inside examples for the covered structures
- made the model layer capable of generating non-abelian fermion-gauge currents from gauge metadata
- made the model layer capable of generating abelian complex-scalar current/contact terms from field charge plus gauge-group declarations
- restored a second live validation path for gamma/tensor identities
- established the first genuinely model-driven gauge workflow in the repository

### 2026-04-01 (later): covariant-derivative compiler, fixed conventions, and non-abelian matter support

What happened:

- a working covariant-derivative compiler was added on top of the model/gauge layer
- the convention was fixed consistently as
  `D_mu = partial_mu + i g A_mu`
- the covered covariant cases were checked explicitly:
  - fermion QCD
  - fermion QED
  - scalar QED 3-point
  - scalar QED 4-point
  - scalar QCD 3-point
  - scalar QCD 4-point
  - mixed QCD+QED fermion coupling
- one covariant kinetic term now expands into the sum over all gauge groups acting on the field
- non-abelian complex-scalar current/contact compilation was added, including explicit representation labels and spectator-index identities
- the output/docs were clarified so the minimal structural compiler is separated from the physical covariant compiler

What this achieved:

- the project now has a working covariant-expansion layer, not only a minimal structural gauge compiler
- the representation/generic structure layer is working for the covered matter-sector cases
- the covariant expansion layer is working for the covered matter-sector cases
- basic validation on standard abelian and non-abelian matter interactions is working
- a major source of fake sign/confvention bugs was removed by separating:
  - the minimal compiler as a generic interaction-structure layer
  - the covariant compiler as the convention-fixed `D_mu` expansion layer

Current interpretation:

- representation/generic structure layer: working
- covariant expansion layer: working
- basic validation on standard interactions: working
- convention confusion: mostly resolved

Practical next steps:

1. freeze conventions in one place across code/docs/tests
   - Fourier convention
   - derivative-to-momentum rule
   - overall vertex `i`
   - covariant-derivative sign
2. turn the current checks into stronger regression tests so refactors cannot silently flip signs or factors
3. extend benchmarks and coverage toward the next harder sector
   - pure Yang-Mills 3-gluon and 4-gluon
   - ghosts if they become relevant to the chosen scope
   - more mixed-representation / multi-gauge-group cases
   - later chiral/projector structures if needed
4. keep improving output readability with clearer labels and compact interpretations
5. decide which compiler is the public physics-facing API; the covariant compiler is the strongest candidate

### 2026-04-02: pure-gauge foundation and Yang-Mills self-interactions

What happened:

- the model layer was extended with a gauge-kinetic declaration via `GaugeKineticTerm`
- the convention-fixed physical compiler was extended from matter kinetic terms into the pure-gauge sector
- the compiler now covers:
  - abelian `-1/4 F_{mu nu} F^{mu nu}` bilinears
  - non-abelian `-1/4 F^a_{mu nu} F^{a mu nu}` bilinears
  - Yang-Mills 3-gauge self-interactions
  - Yang-Mills 4-gauge self-interactions
- reusable pure-gauge tensor/operator builders were added for both:
  - the raw compiled vertex form
  - the more readable compact convention-fixed form
- the `covariant` demo suite was extended to print the new pure-gauge outputs
- focused regression checks were added for:
  - abelian gauge bilinear
  - non-abelian gauge bilinear
  - Yang-Mills cubic vertex
  - Yang-Mills quartic vertex
- top-level docs were updated so the live repository description matches the code again

What this achieved:

- the project now has a working ordinary gauge-theory foundation for both:
  - matter-sector covariant derivatives
  - pure-gauge field-strength expansion
- the physical compiler can now derive the standard Yang-Mills 2/3/4-point structures from model metadata
- the next architectural step is clearer: BFM-specific background/quantum splitting can now be built on top of an ordinary gauge sector that already works
- the minimal structural compiler and the convention-fixed physical compiler are now separated more clearly in both code and docs

Current interpretation after today:

- representation/generic structure layer: working for the covered matter and pure-gauge cases
- convention-fixed physical compiler: working for the covered ordinary gauge-theory cases
- pure-gauge sector: working for abelian bilinears and Yang-Mills self-interactions
- BFM-specific layer: not implemented

Practical next steps after today:

1. add background/quantum gauge-field splitting
2. add gauge-fixing declarations and compilation
3. add ghosts after gauge fixing is stable
4. move regression logic out of `src/examples.py`

### 2026-04-02 (later): source review, reprioritization, and documentation sync

What happened:

- the active `src/` code was reviewed module by module rather than only through the demos
- the live validation paths were rerun:
  - `src/examples.py --suite all`
  - `src/spenso_gamma_checks.py`
- the review confirmed that the ordinary matter and pure-gauge sectors are working for the covered cases
- the main nontrivial structural weakness identified in the current code is repeated identical index kinds:
  the compiler currently stores labels by `index.kind`, which is not robust for fields carrying two slots of the same kind
- the review also confirmed that multi-fermion support is still intentionally narrow and that the main regression burden still sits inside `src/examples.py`

What this achieved:

- clarified the difference between:
  - what is already working physics
  - what is the next physics sector to add
  - what should be hardened first so the next physics sector does not land on weak foundations
- sharpened the practical priority order:
  1. freeze conventions and extract tests
  2. fix repeated same-kind index-slot handling
  3. add ordinary gauge fixing
  4. add ghosts
  5. only then add BFM-specific background/quantum splitting

What should be done next week:

1. move the current covariant and pure-gauge checks into a first dedicated test harness
2. write one stable conventions reference shared across code/docs/tests
3. tighten the model/compiler index-slot representation so repeated same-kind slots remain distinct
4. draft the declaration/compilation API for gauge-fixing terms
5. improve the canonical readability of pure-gauge output while keeping the raw form available

### 2026-04-08: repeated-slot hardening, bislot covariant support, and walkthrough sync

What happened:

- the model/compiler boundary was hardened for fields carrying multiple slots of the same index kind
- `GaugeRepresentation` now supports:
  - `slot_policy="unique"` as the strict default
  - `slot_policy="sum"` as an explicit opt-in for tensor-product-style expansion over all matching slots
  - multi-slot resolution helpers used by the gauge compiler
- `Field.pack_slot_labels(...)` was made ordinal-stable for repeated kinds by preserving tuple length and `None` placeholders
- the gauge compiler was updated so the covered current/contact paths now loop over active slots instead of assuming one unique slot
- ambiguous repeated-slot cases are now rejected by default, while the bislot QCD scalar case can be expanded intentionally under `slot_policy="sum"`
- non-abelian scalar contact compilation now includes the ordered slot-pair expansion needed for same-slot and cross-slot placements
- spectator identities on inactive repeated slots are now inserted systematically in both minimal and covariant gauge compilation
- the walkthrough notebook was expanded and synchronized with the live source tree, especially around:
  - section 9 / 9.1 for the covariant compiler
  - the bislotted scalar `slot_policy="sum"` example
  - the pure-gauge follow-up section
- a first dedicated `pytest` regression file was added for the bislot covariant case while the broader matrix remains in `src/examples.py`
- the live validation paths were rerun successfully:
  - `src/examples.py --suite all --no-demo`
  - `pytest -q`

What this achieved:

- repeated same-kind index slots no longer collapse at the model/compiler boundary for the covered code paths
- the covariant compiler now has a real, explicit semantics for repeated slots:
  - ambiguity is an error by default
  - tensor-product summation is an explicit metadata choice
- the bislotted scalar covariant expansion is now working for the covered QCD-style case, including ordered contact-term slot pairs
- the source tree, executable examples, and notebook walkthrough now tell the same story about the current compiler layers
- the project now has the start of a real test split instead of keeping absolutely all covariant validation inside demo code

Current interpretation after this update:

- repeated same-kind slot handling: working for the covered minimal and covariant gauge-compiler paths
- covariant matter compiler: working for the covered single-group fermion/scalar cases and mixed-group fermion case
- pure-gauge compiler: working for the covered abelian and Yang-Mills cases
- dedicated `pytest` coverage: started, but still too small compared with the live regression matrix

What remains unfinished around this area:

- mixed-group complex-scalar covariant terms still do not generate cross-group two-gauge contact terms
- model-level validation is still too permissive when undeclared `Field` or `GaugeGroup` objects are passed directly
- explicit abelian group selection can still compile zero-coupling interactions instead of rejecting bad metadata
- most covariant and pure-gauge regression coverage still lives in `src/examples.py`

Practical next steps:

1. complete mixed-group complex-scalar covariant contact generation
2. tighten model membership checks for fields and gauge groups used by compiler declarations
3. expand the dedicated `pytest` layer so the main covariant/pure-gauge matrix no longer depends on demo scripts
4. then continue into gauge fixing, ghosts, and later BFM-specific work on top of the ordinary gauge baseline

### 2026-04-09: validation hardening, pytest extraction, and conventions freeze

What happened:

- the covariant and gauge-kinetic compiler entry points were hardened so they now require
  fields and gauge groups declared in the parent `Model`
- explicit gauge-group selections now raise when:
  - an abelian choice is made for a neutral field
  - a non-abelian choice is made for a singlet / missing representation
- dedicated `pytest` coverage was added for those compiler-validation rules
- the main covariant and pure-gauge regression matrix was extracted from `src/examples.py`
  into a dedicated `pytest` file
- a short frozen conventions note was added so the active sign and normalization
  choices now live in one main reference file
- the top-level status docs were synchronized with the live code path

What this achieved:

- metadata mistakes now fail earlier instead of compiling into silent zero-coupling output
- the core ordinary covariant / pure-gauge compiler path no longer depends only on
  the demo script for regression coverage
- the active sign conventions are now documented in one place instead of being spread
  across code comments, README bullets, and test expectations

Current interpretation after this update:

- mixed-group complex-scalar covariant contacts: working
- compiler-side validation for undeclared fields/groups and invalid explicit gauge selections: working
- dedicated `pytest` coverage for the main covariant / pure-gauge matrix: working
- broader direct/model regression extraction: still incomplete

Practical next steps:

1. keep widening the `pytest` split beyond the covariant / pure-gauge matrix
2. tighten the remaining model/declaration validation outside the current compiler entry points
3. draft and implement gauge-fixing declarations through the physical compiler path
4. then add ghosts and later BFM-specific background/quantum splitting

### 2026-04-09 (later): ordinary gauge fixing, ghosts, demos, and notebook sync

What happened:

- the model layer was extended with:
  - `GaugeFixingTerm`
  - `GhostTerm`
  - `GaugeGroup.ghost_field`
  - explicit ghost / antighost roles on top of the shared `Field` dataclass
- the physical compiler was extended with:
  - `compile_gauge_fixing_term(...)`
  - `compile_ghost_term(...)`
  - strict ghost-field resolution through the parent `Model`
- the ordinary gauge-fixing path now compiles the linear-covariant bilinear
  `-(1/2 xi) (partial.A)^2` for both abelian and non-abelian gauge groups
- the ordinary non-abelian ghost path now compiles the integrated Faddeev-Popov
  sector
  `-cbar partial.D c = (partial cbar)(partial c) - g f (partial cbar) A c`
- reusable raw/compact operator helpers were added for the new gauge-fixing and
  ghost display forms
- `src/examples.py` gained a dedicated `gaugefix` suite that prints:
  - compiled interaction labels
  - compiled `InteractionTerm` objects
  - the evaluated vertices
  - compact readability rewrites
- the notebook walkthrough gained a simple section `10.5 Ordinary Gauge Fixing And Ghosts`
  in the same manual-print style as sections 9.1 and 10
- dedicated `pytest` coverage was added in `tests/test_gauge_fixing_and_ghosts.py`
  for:
  - abelian and non-abelian gauge fixing
  - the non-abelian ghost bilinear
  - the ghost-gluon interaction
  - rejection of abelian ghost terms
  - rejection of self-conjugate ghost fields
- the live validation paths were rerun successfully:
  - `pytest -q`
  - `src/examples.py --suite gaugefix`
  - `src/examples.py --suite all --no-demo`

What this achieved:

- the project now has a working ordinary gauge-fixed baseline at the
  model/compiler level, not only in handwritten example algebra
- the ordinary gauge-theory path now covers, in one compiler stack:
  - matter covariant derivatives
  - pure-gauge field strengths
  - ordinary gauge fixing
  - ordinary non-abelian ghosts
- the examples, notebook, and tests now all expose the same new physics sector
- the next step is clearer and cleaner: build the first BFM-specific layer on
  top of the ordinary gauge-fixed base, rather than mixing ordinary and BFM
  work together

Current interpretation after this update:

- ordinary gauge-fixed baseline: working for the covered unbroken abelian and
  non-abelian cases
- compiler-generated gauge fixing: working
- compiler-generated ordinary non-abelian ghosts: working
- notebook/demo/test synchronization for the new sector: working
- BFM-specific background/quantum splitting: still not implemented

Practical next steps:

1. add background/quantum gauge-field splitting on top of the current ordinary
   gauge-field declarations
2. extend the pure-gauge compiler with `A -> B + Q` while preserving the
   ordinary non-BFM path
3. add BFM gauge-fixing declarations and compilation on top of that split
4. add the corresponding BFM ghost sector
5. keep improving canonical readability for pure-gauge, gauge-fixing, and ghost
   output while widening the dedicated regression split

### Where we are in the overall progress

Best current summary:

- foundation phase: done
- scalar + derivative phase: done
- fermion combinatorics phase: done at prototype level
- first tensor-structured fermion/gauge-ready phase: working
- model-layer phase: working and usable
- minimal gauge-compiler phase: working
- covariant-derivative compiler phase: working for covered matter-sector cases
- ordinary pure-gauge field-strength phase: working for the covered abelian and Yang-Mills cases
- ordinary gauge-fixing and ghost phase: working for the covered unbroken cases
- gauge-complete / BFM phase: not implemented
- full FeynRules-style compilation layer: not implemented
- export/usability layer: not implemented

### Current assessment

The project now has a real core, not just experiments:

- one working contraction engine
- one usable model layer
- one reusable operator vocabulary
- one minimal gauge compiler
- one working convention-fixed physical compiler for the covered matter, pure-gauge, gauge-fixing, and ghost cases
- one growing dedicated `pytest` layer
- one walkthrough notebook that now tracks the same compiler layers
- two runnable validation/demo scripts

The main remaining risks are structural rather than conceptual:

- broader direct/model regression still leans too heavily on `src/examples.py`
- the ordinary gauge-fixed sector now works, but BFM-specific scaffolding is still absent
- the remaining declaration/model validation is still narrower than a fuller library-quality API
- raw pure-gauge / gauge-fixing / ghost output still benefits from compact
  readability rewrites in displays and tests

### Immediate next milestone

The next milestone should be:

"the first BFM-specific layer on top of the ordinary gauge-fixed foundation"

That means:

1. add background/quantum gauge-field splitting
2. compile BFM gauge fixing on top of that split
3. compile the corresponding BFM ghost sector
4. keep improving the canonical readability of pure-gauge, gauge-fixing, and ghost output
