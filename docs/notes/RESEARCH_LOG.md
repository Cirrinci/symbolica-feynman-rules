## Research Log

This file is the main chronological record of work completed so far on the
Symbolica/Spenso FeynRules-style prototype. It should stay readable as a
single narrative of progress, while the other notes carry stable plans,
conventions, and design discussions.

Rule:

- keep dated work sessions
- do not collapse past days into milestone summaries
- new sessions should be appended as new dated entries, not used to overwrite
  earlier ones

### Current status snapshot

As of 2026-04-23:

- the active source tree is modularized under `src/`
- the core symbolic extraction engine now lives in `src/symbolic/vertex_engine.py`
- tensor and canonicalization helpers live in:
  - `src/symbolic/spenso_structures.py`
  - `src/symbolic/tensor_canonicalization.py`
- the model/declaration layer now lives in:
  - `src/model/core.py`
  - `src/model/interactions.py`
  - `src/model/declared.py`
  - `src/model/lagrangian.py`
  - `src/model/lowering.py`
  - `src/model/metadata.py`
- declarative helper/building-block code lives in:
  - `src/lagrangian/operators.py`
  - `src/lagrangian/lowering.py`
- gauge/covariant compiler logic now lives in `src/compiler/gauge.py`
- runnable validation/examples now live in:
  - `examples/examples.py`
  - `examples/examples_lagrangian.py`
  - `examples/examples_su2.py`
  - `examples/examples_electroweak_unbroken.py`
  - `examples/examples_electroweak_ssb.py`
- dedicated regression coverage now lives in `tests/`
- the walkthrough notebook is `notebooks/codebase_workflow_walkthrough.ipynb`
- the final user-facing walkthrough notebook is
  `notebooks/final_walkthrough_capabilities_and_usage.ipynb`
- the main extraction path is now `model.lagrangian().feynman_rule(...)`
- the recommended public model-building API is now:
  `Model(..., lagrangian_decl=...)`
- canonical declarative building blocks now include:
  - `CovD(field, mu)`
  - `Gamma(mu)`
  - `FieldStrength(group, mu, nu)`
  - `GaugeFixing(group, xi=...)`
  - `GhostLagrangian(group)`
- manual `InteractionTerm(...)` monomials still compose with the declarative
  forms in the same Lagrangian declaration
- the backend architecture is still the same:
  source-level declarations are lowered into the existing
  `InteractionTerm` / compiler pipeline rather than introducing a second
  symbolic engine
- legacy split declaration slots are still supported for compatibility, but are
  now deprecated in favor of the unified Lagrangian entry point
- broken-phase electroweak helpers now live in `src/model/ssb.py`
- the long-term goal remains a Python analogue of FeynRules using Symbolica for
  symbolic rewriting and Spenso for tensor/index structures

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

- gamma-matrix and related spinor structures were introduced through
  Spenso-backed wrappers
- explicit open-spinor remapping inside coupling tensors was added for the
  covered fermion patterns
- current-current and gauge-ready fermion examples were added

What this achieved:

- shifted the project from plain fermion combinatorics toward actual
  tensor-structured fermion vertices
- demonstrated that open-index amputated output is viable in the current design

### 2026-03-30: gauge-ready tensor checks

What happened:

- slot-label handling was generalized for gauge-ready tensor structures
- complex-scalar and gauge interaction examples were added
- tensor/gamma validation scripts were cleaned up

What this achieved:

- prepared the tensor layer for the later model-driven gauge compiler
- made gauge-index checks less dependent on one-off notebook experiments

### 2026-03-31: repository refactor review

What happened:

- the active tree was confirmed around `src/model_symbolica.py`,
  `src/model.py`, `src/spenso_structures.py`, and `src/examples.py`
- the main example suite was confirmed runnable
- the documentation was updated to describe the live `src/` layout instead of
  the archived `code/` layout
- a code review identified the main structural gaps still worth addressing

What this achieved:

- clarified the actual architecture of the current repository
- separated working validation paths from stale scripts
- established a sharper set of next engineering tasks

### 2026-04-01: engine hardening and minimal gauge-model compiler

What happened:

- role handling in the engine/model boundary was hardened around typed field
  roles
- reusable operator builders were moved into `src/operators.py`
- `src/spenso_gamma_checks.py` was brought back onto the live source layout
- a minimal model-driven gauge compiler was added in `src/gauge_compiler.py`
- `src/examples.py` was extended to exercise compiled gauge interactions and
  print them clearly in both demo and validation paths

What this achieved:

- removed the main dependency on ad hoc operator assembly inside examples for
  the covered structures
- made the model layer capable of generating non-abelian fermion-gauge currents
  from gauge metadata
- made the model layer capable of generating abelian complex-scalar
  current/contact terms from field charge plus gauge-group declarations
- restored a second live validation path for gamma/tensor identities
- established the first genuinely model-driven gauge workflow in the repository

### 2026-04-01 (later): covariant-derivative compiler, fixed conventions, and non-abelian matter support

What happened:

- a working covariant-derivative compiler was added on top of the model/gauge
  layer
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
- one covariant kinetic term now expands into the sum over all gauge groups
  acting on the field
- non-abelian complex-scalar current/contact compilation was added, including
  explicit representation labels and spectator-index identities
- the output/docs were clarified so the minimal structural compiler is
  separated from the physical covariant compiler

What this achieved:

- the project now has a working covariant-expansion layer, not only a minimal
  structural gauge compiler
- the representation/generic structure layer is working for the covered
  matter-sector cases
- the covariant expansion layer is working for the covered matter-sector cases
- basic validation on standard abelian and non-abelian matter interactions is
  working
- a major source of fake sign/convention bugs was removed by separating:
  - the minimal compiler as a generic interaction-structure layer
  - the covariant compiler as the convention-fixed `D_mu` expansion layer

Current interpretation:

- representation/generic structure layer: working
- covariant expansion layer: working
- basic validation on standard interactions: working
- convention confusion: mostly resolved

Practical next steps:

1. freeze conventions in one place across code/docs/tests
2. turn the current checks into stronger regression tests so refactors cannot
   silently flip signs or factors
3. extend benchmarks and coverage toward the next harder sector
4. keep improving output readability with clearer labels and compact
   interpretations
5. decide which compiler is the public physics-facing API

### 2026-04-02: pure-gauge foundation and Yang-Mills self-interactions

What happened:

- the model layer was extended with a gauge-kinetic declaration via
  `GaugeKineticTerm`
- the convention-fixed physical compiler was extended from matter kinetic terms
  into the pure-gauge sector
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
- top-level docs were updated so the live repository description matches the
  code again

What this achieved:

- the project now has a working ordinary gauge-theory foundation for both:
  - matter-sector covariant derivatives
  - pure-gauge field-strength expansion
- the physical compiler can now derive the standard Yang-Mills 2/3/4-point
  structures from model metadata
- the next architectural step became clearer:
  BFM-specific background/quantum splitting can be built on top of an ordinary
  gauge sector that already works
- the minimal structural compiler and the convention-fixed physical compiler are
  now separated more clearly in both code and docs

### 2026-04-02 (later): source review, reprioritization, and documentation sync

What happened:

- the active `src/` code was reviewed module by module rather than only through
  the demos
- the live validation paths were rerun:
  - `src/examples.py --suite all`
  - `src/spenso_gamma_checks.py`
- the review confirmed that the ordinary matter and pure-gauge sectors are
  working for the covered cases
- the main nontrivial structural weakness identified in the current code was
  repeated identical index kinds:
  the compiler was storing labels by `index.kind`, which is not robust for
  fields carrying two slots of the same kind
- the review also confirmed that multi-fermion support is still intentionally
  narrow and that the main regression burden still sat inside `src/examples.py`

What this achieved:

- clarified the difference between:
  - what is already working physics
  - what is the next physics sector to add
  - what should be hardened first so the next physics sector does not land on
    weak foundations
- sharpened the practical priority order:
  1. freeze conventions and extract tests
  2. fix repeated same-kind index-slot handling
  3. add ordinary gauge fixing
  4. add ghosts
  5. only then add BFM-specific background/quantum splitting

### 2026-04-06 to 2026-04-07: documentation pass and repeated-slot preparation

What happened:

- the README, setup scripts, requirements, and source docstrings were refreshed
- the main walkthrough notebook was added and normalized under `notebooks/`
- repeated index-slot handling started to be hardened in the model/compiler
  path

What this achieved:

- made the repository easier to run and inspect
- prepared the ground for the bislot covariant-derivative work that followed

### 2026-04-08: repeated-slot hardening, bislot covariant support, and walkthrough sync

What happened:

- the model/compiler boundary was hardened for fields carrying multiple slots of
  the same index kind
- `GaugeRepresentation` now supports:
  - `slot_policy="unique"` as the strict default
  - `slot_policy="sum"` as an explicit opt-in for tensor-product-style
    expansion over all matching slots
  - multi-slot resolution helpers used by the gauge compiler
- `Field.pack_slot_labels(...)` was made ordinal-stable for repeated kinds by
  preserving tuple length and `None` placeholders
- the gauge compiler was updated so the covered current/contact paths now loop
  over active slots instead of assuming one unique slot
- ambiguous repeated-slot cases are now rejected by default, while the bislot
  QCD scalar case can be expanded intentionally under `slot_policy="sum"`
- non-abelian scalar contact compilation now includes the ordered slot-pair
  expansion needed for same-slot and cross-slot placements
- spectator identities on inactive repeated slots are now inserted
  systematically in both minimal and covariant gauge compilation
- the walkthrough notebook was expanded and synchronized with the live source
  tree, especially around:
  - section 9 / 9.1 for the covariant compiler
  - the bislotted scalar `slot_policy="sum"` example
  - the pure-gauge follow-up section
- a first dedicated `pytest` regression file was added for the bislot covariant
  case while the broader matrix remained in `src/examples.py`
- the live validation paths were rerun successfully:
  - `src/examples.py --suite all --no-demo`
  - `pytest -q`

What this achieved:

- repeated same-kind index slots no longer collapse at the model/compiler
  boundary for the covered code paths
- the covariant compiler now has a real, explicit semantics for repeated slots:
  - ambiguity is an error by default
  - tensor-product summation is an explicit metadata choice
- the bislotted scalar covariant expansion is now working for the covered
  QCD-style case, including ordered contact-term slot pairs
- the source tree, executable examples, and notebook walkthrough now tell the
  same story about the current compiler layers
- the project now has the start of a real test split instead of keeping all
  covariant validation inside demo code

### 2026-04-09: validation hardening, pytest extraction, and conventions freeze

What happened:

- the covariant and gauge-kinetic compiler entry points were hardened so they
  now require fields and gauge groups declared in the parent `Model`
- explicit gauge-group selections now raise when:
  - an abelian choice is made for a neutral field
  - a non-abelian choice is made for a singlet or missing representation
- dedicated `pytest` coverage was added for those compiler-validation rules
- the main covariant and pure-gauge regression matrix was extracted from
  `src/examples.py` into a dedicated `pytest` file
- a short frozen conventions note was added so the active sign and
  normalization choices now live in one main reference file
- the top-level status docs were synchronized with the live code path

What this achieved:

- metadata mistakes now fail earlier instead of compiling into silent
  zero-coupling output
- the core ordinary covariant/pure-gauge compiler path no longer depends only
  on the demo script for regression coverage
- the active sign conventions are now documented in one place instead of being
  spread across code comments, README bullets, and test expectations

### 2026-04-09 (later): ordinary gauge fixing, ghosts, demos, and notebook sync

What happened:

- the model layer was extended with:
  - `GaugeFixingTerm`
  - `GhostTerm`
  - `GaugeGroup.ghost_field`
  - explicit ghost and antighost roles on top of the shared `Field` dataclass
- the physical compiler was extended with:
  - `compile_gauge_fixing_term(...)`
  - `compile_ghost_term(...)`
  - strict ghost-field resolution through the parent `Model`
- the ordinary gauge-fixing path now compiles the linear-covariant bilinear
  `-(1/2 xi) (partial.A)^2` for both abelian and non-abelian gauge groups
- the ordinary non-abelian ghost path now compiles the integrated
  Faddeev-Popov sector
  `-cbar partial.D c = (partial cbar)(partial c) - g f (partial cbar) A c`
- reusable raw/compact operator helpers were added for the new gauge-fixing and
  ghost display forms
- `src/examples.py` gained a dedicated `gaugefix` suite
- the notebook walkthrough gained a simple section
  `10.5 Ordinary Gauge Fixing And Ghosts`
- dedicated `pytest` coverage was added in
  `tests/test_gauge_fixing_and_ghosts.py`
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
- the next step became clearer:
  build the first BFM-specific layer on top of the ordinary gauge-fixed base

### 2026-04-14 to 2026-04-15: first declarative Lagrangian transition

What happened:

- the first `lagrangian_decl` tests, examples, and transition notes were added
- source-preserving declarations began lowering into the existing compiler path
- partial-derivative and dressed-covariant examples were worked through in the
  notebook

What this achieved:

- established the shape of the public declarative API
- kept the new front end connected to the existing Symbolica/Spenso backend

### 2026-04-16: declarative Lagrangian front-end and API unification

What happened:

- the public model-building workflow was moved toward one FeynRules-style entry
  point:
  `Model(..., lagrangian_decl=...)`
- canonical declarative building blocks were introduced:
  - `CovD(field, mu)`
  - `Gamma(mu)`
  - `FieldStrength(group, mu, nu)`
  - `GaugeFixing(group, xi=...)`
  - `GhostLagrangian(group)`
- manual `InteractionTerm(...)` monomials were kept composable with these new
  declarative objects inside the same declaration
- a source-preserving `DeclaredLagrangian` was added:
  - `source_terms` keeps the user-written declaration
  - cached `lowered_terms` feed the existing compiler backend
- lowering now maps canonical declarative forms to the current model-level
  term classes:
  - `DiracKineticTerm`
  - `ComplexScalarKineticTerm`
  - `GaugeKineticTerm`
  - `GaugeFixingTerm`
  - `GhostTerm`
- `with_compiled_covariant_terms(...)` was updated so precompiled models remain
  idempotent under `Model.lagrangian()`
- `Lagrangian.feynman_rule(...)` field matching was tightened to avoid
  same-symbol field collisions
- `(Field, bool)` input is now accepted in the Lagrangian API
- `InteractionTerm + InteractionTerm` and mixed declarative composition were
  unified through the same `DeclaredLagrangian` path
- `GaugeFixing(...)` and `GhostLagrangian(...)` gained scalar-prefactor support
  while preserving their source form in demos and notebooks
- the canonical examples and docs were updated to present `lagrangian_decl=...`
  as the main workflow
- legacy split declaration slots were kept only as a compatibility path and now
  emit `DeprecationWarning`
- validation was rerun successfully on the branch:
  - `./.venv/bin/python -m pytest -q`
  - `./.venv/bin/python src/examples.py --suite all --no-demo`
  - `./.venv/bin/python src/examples_lagrangian.py --suite all --no-demo`
  - result reported in the branch note: `89 passed`

What this achieved:

- the project now has a real declarative front-end instead of exposing only a
  collection of split physical declaration slots
- the user-facing API is much closer to a FeynRules-style workflow without
  discarding the existing Symbolica/Spenso backend
- source-level declarations and lowered compiler terms are now kept distinct,
  which is the correct architectural boundary for future growth
- the main extraction story is cleaner:
  declare a Lagrangian once, then derive vertices through
  `model.lagrangian().feynman_rule(...)`
- the codebase can now evolve toward broader usability without maintaining two
  symbolic engines or two divergent physics paths

### 2026-04-17: package split and import cleanup

What happened:

- the old monolithic source files were split into `src/model/`,
  `src/compiler/`, `src/lagrangian/`, and `src/symbolic/`
- examples were moved under `examples/`
- imports, tests, and notebook references were updated to the new layout
- the symbolic engine was renamed around the vertex-extraction role

What this achieved:

- made the architecture easier to navigate
- separated model declarations, compiler logic, Lagrangian helpers, and symbolic
  extraction into clearer modules

### 2026-04-21: compiler refactor review, SU(2)L baseline, and docs sync

What happened:

- the covariant assembly path in `src/compiler/gauge.py` was consolidated
- shared fermion-current emission replaced duplicated compiler logic
- spectator and no-spectator covariant paths were merged
- lowering and declarative `CovD` expectations were cleaned up
- SU(2)L and unbroken electroweak examples/tests were added
- notes were synchronized with the package-based source layout

What this achieved:

- the gauge/lowering stack became less duplicated and easier to test
- the package layout, docs, examples, and tests moved back into alignment
- the unbroken electroweak baseline gave the SU(2) x U(1) path a clearer
  regression target

### 2026-04-22: broken electroweak and SSB baseline

What happened:

- standalone local Lagrangian declarations were restored
- a broken-phase electroweak example was added
- Higgs expansion, physical W/Z/A mixing, masses, self-couplings, gauge fixing,
  ghosts, Yukawas, and CKM-style charged currents were covered
- dedicated electroweak SSB tests were added

What this achieved:

- moved the project from unbroken electroweak examples into a first physical
  broken-phase workflow
- checked photon masslessness and Goldstone/gauge mixing cancellation in the
  covered example

### 2026-04-23: vertex discovery API and final walkthrough notebook

What happened:

- `feynman_rule()` without field arguments now discovers available vertices
- explicit vertex extraction remains unchanged
- name-keyed and object-keyed discovery are both supported
- ambiguous readable names and invalid discovery options now fail explicitly
- the final walkthrough notebook was created and exercised against the current
  API
- SSB tests and titles were refined

What this achieved:

- users can inspect a compiled Lagrangian before choosing a specific analytic
  rule to study
- the public API now supports both targeted extraction and exploratory
  discovery
- the latest notebook gives a compact demonstration of the current scalar,
  fermion, derivative, gauge, ghost, QED, QCD, and electroweak-style workflows
