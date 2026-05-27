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

As of 2026-05-13:

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
  - `examples/examples_flavor_expansion.py`
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
- flavor-class declarations now use explicit field metadata:
  - `dirac_field(..., class_members=..., flavor_index=...)`
  - `scalar_field(..., class_members=..., flavor_index=...)`
- selective flavor expansion is available through
  `flavor_expand=True`, one flavor index, or an iterable of flavor indices
- plain symbolic slot labels are now validated monomial-wide against one exact
  `IndexType` per label name before lowering continues
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

### 2026-04-28: same-species Dirac-current products, bilinear identities, and FeynRules-style signs

What happened:

- the final walkthrough notebook case
  `I * qbar * Gamma(mu) * CovD(q, mu) * qbar * Gamma(nu) * CovD(q, nu)` was
  traced through the declarative lowering and covariant compiler paths
- the generic local monomial lowering / covariant expansion path was exercised
  on the same-species QCD current-current operator so that:
  - `Gamma(qbar, q, qbar, q, G)` no longer fails structurally
  - `Gamma(qbar, q, qbar, q, G, G)` is also emitted
- the saved final notebook example cell in
  `notebooks/final_walkthrough_capabilities_and_usage.ipynb` was inspected
  against the actual `qbar, q, qbar, q, G` output and its simplification path
- the failed whole-expression tensor canonicalization attempt on that notebook
  cell was diagnosed:
  - identical-quark Wick-contraction sums do not share one uniform external
    index assignment across all terms
  - `canonize_spenso_tensors(...)` therefore rejects the full 5-point sum
    when asked to canonize it as if one fixed external index set applied to
    every term
- the surviving unsimplified `mu` / `nu` derivative labels in the same-species
  current-current-gluon output were traced to derivative dummy labels being
  carried through term-by-term instead of being normalized before comparison
- auto-generated external leg labels were also inspected in that same notebook
  example:
  - the old generic numbering led to labels such as `i9`, `i10` on a 5-leg
    vertex
  - the current typed labeling now produces readable slots like `i1`, `c1`,
    `mu5`, `a5`
- the simplest fermion bilinear `Quark.bar * Quark` was debugged in the final
  notebook / Lagrangian API path:
  - spin contraction was already emitted as the bispinor metric
  - color contraction was not being materialized as an explicit external color
    identity
  - the raw/simplified output was leaving only species deltas like
    `delta(q,q) * delta(qbar,qbar)` instead of an external
    `g(cof(3, c1), cof(3, c2))`
- the vertex contraction engine was updated so repeated explicit non-spinor
  field-slot labels now generate identity tensors by index
  type/representation, while the existing spinor-chain handling was left intact
- the multi-fermion sign mismatch with FeynRules was then traced carefully on
  three benchmark operators:
  - `(qbar q)(qbar q)`
  - `(qbar Gamma(mu) q)(qbar Gamma(mu) q)`
  - `(qbar Gamma(mu) CovD(q, mu))(qbar Gamma(nu) CovD(q, nu))`
- the direct source of the mismatch was identified in the existing engine
  design:
  - fermion signs were always computed from the flat permutation of fermion
    slots
  - closed bilinear-current structure was either ignored or only partially
    visible through repeated spinor labels
  - this produced `direct - crossed` for identical-fermion current products
    even when FeynRules gives `direct + crossed`
- a targeted metadata path was implemented around
  `InteractionTerm.closed_dirac_bilinears`:
  - `src/model/lowering.py` now records explicitly known adjacent closed
    `psibar ... psi` Dirac bilinears while lowering local monomials
  - this covers:
    - `qbar q`
    - `qbar Gamma(mu) q`
    - `qbar Gamma(mu) PartialD(q, mu)`
    - `qbar Gamma(mu) CovD(q, mu)` after covariant expansion into local terms
  - `src/compiler/gauge.py` now tags compiler-built fermion gauge currents and
    Dirac partial terms with the same bilinear metadata
  - spectator decoration now preserves and shifts those bilinear slot pairs
    when explicit `psibar * psi` spectator pairs are appended to a covariant
    core
  - `src/symbolic/vertex_engine.py` now suppresses flat Grassmann permutation
    signs only when all Dirac fermion slots are covered exactly once by closed
    bilinear metadata; otherwise the old behavior is preserved
- the existing repeated-spinor-label fermion-chain inference was kept as a
  compatibility fallback for already explicit closed bilinears, but no new
  generic field-order inference was introduced for open multi-fermion tensors
- the declared-vs-explicit current-current reference test in
  `tests/test_lagrangian_api.py` was updated so the explicit reference
  `InteractionTerm(...)` carries the same closed-bilinear provenance as the
  declared lowering path
- concrete sign expectations were updated in the runnable examples looked at
  during the investigation:
  - `examples/examples_lagrangian.py`
    - `DECL_psibar_psi_sq` now checks `direct + crossed`
    - `DECL_current_current` now checks `direct + crossed`
  - `examples/examples.py`
    - `L_psibar_psi_sq`
    - `L_current_current`
    - `TERM_psibar_psi_sq`
    - `TERM_current_current`
    now all use the explicit closed-bilinear metadata and their expected signs
    were flipped from `direct - crossed` to `direct + crossed`
  - `src/symbolic/spenso_gamma_checks.py`
    - the stripped current-current gamma check was updated to the same
      all-plus bilinear-current convention
- the specific same-species QCD 5-point benchmark from the final notebook was
  locked down in `tests/test_lagrangian_api.py`:
  - the four `qbar, q, qbar, q, G` direct/crossed structures are now checked
    to carry the same relative sign
  - the test still allows one common overall sign from the covariant-derivative
    convention
- focused validation was rerun on the updated paths:
  - `PYTHONPATH=src ./.venv/bin/pytest tests -q`
    - result: `182 passed`
  - `PYTHONPATH=src ./.venv/bin/python examples/examples_lagrangian.py --no-demo`
    - result: all Lagrangian API example checks passed

What this achieved:

- same-species multi-current Dirac operators now work as real declarative
  inputs instead of remaining fragile notebook-only experiments
- the engine now distinguishes a genuine product of closed Dirac bilinear
  currents from a generic open multi-fermion tensor, which is the right
  boundary for adopting the FeynRules-style sign convention without broadening
  it too far
- the simplest colored fermion bilinear now exposes explicit spin and color
  identity structure at the vertex level instead of hiding the color identity
  inside species deltas
- the final notebook example and the fermion-current examples now reflect the
  intended sign convention more faithfully for identical-fermion current
  products

Here is the clean version in plain markdown text:

---

### 2026-04-29: output-policy API, documentation sync, and model validation diagnostics

What happened:

- the high-level `feynman_rule(...)` API was extended to expose vertex
  postprocessing options:

  - `include_delta` (default `True`) to control delta-function removal
  - `strip_externals` (default `True`) to toggle removal of external wavefunctions (amputated vs unamputated form)
  - `simplify_gamma` (default `False`) to enable gamma-matrix chain cleanup
- these options now wire into the existing lower-level vertex postprocessing
  layer rather than introducing new processing logic
- dedicated test coverage was added for delta removal, external wavefunction
  stripping, and gamma simplification paths
- the final walkthrough notebook was updated to demonstrate the new output
  policy options in practice
- external wavefunction notation was clarified:

  - `strip_externals=False` exposes explicit external factors
  - default behavior returns amputated vertices
- the high-level API boundary between `Lagrangian(...)` and
  `Model(..., lagrangian_decl=...)` was clarified in documentation:

  - `Lagrangian(...)` takes local/operator-level terms as input
  - `Model(..., lagrangian_decl=...)` takes declarations that depend on model
    metadata (gauge groups, fields, representations)
  - `T(...)` and `StructureConstant(...)` are currently local-scope only
- README and core module docstrings were updated to reflect this boundary
- a new model validation layer was implemented:

  - `ValidationIssue` and `ValidationReport` classes capture validation state
  - `Model.validate()` now performs focused cross-checks on declared model
    metadata
  - the validator detects:

    - undeclared gauge-fixing or ghost term group references
    - abelian gauge sectors with ghost declarations (invalid)
    - missing ghost fields referenced in ghost terms
    - missing structure constants for non-abelian gauge groups used in ghosts
  - the validation layer is intentionally narrow and diagnostic-only (no physics normalization or hermiticity checks yet)
- focused `pytest` coverage was added for these validation rules in
  `tests/test_model_validation.py`
- the final notebook was updated to show the new validation diagnostics
- the live validation paths were rerun successfully:

  - `PYTHONPATH=src ./.venv/bin/pytest tests -q`

    - result: `182 passed`
  - `PYTHONPATH=src ./.venv/bin/python examples/examples_lagrangian.py --no-demo`

    - result: all Lagrangian API example checks passed

What this achieved:

- the public vertex extraction API now exposes postprocessing choices without
  requiring lower-level function knowledge
- the API boundary between local operator-level declarations and
  metadata-dependent model-level declarations is now explicit in documentation
  and enforced in code
- common model metadata mistakes (missing ghosts, undeclared groups, invalid
  abelian ghosts) are now caught early by validation instead of producing
  silent zero-coupling output later
- the validation layer establishes a foundation for future physics checks
  (kinetic normalization, hermiticity, mass structure)
- the final walkthrough notebook now demonstrates the full declarative
  workflow including diagnostic validation

### 2026-04-30: validation diagnostics, parameter metadata, and notebook cleanup

What happened:

- the reporting/validation layer was extended beyond basic issue collection:
  - sector filtering was added to vertex reports
  - compiled-lagrangian mass-mixing diagnostics were added for off-diagonal
    bilinears
  - the mass-mixing warnings were kept diagnostic-only rather than promoted to
    hard failures
- the first version of those checks was then tightened:
  - stable field keys were used instead of display names
  - warnings were restricted to canonical conjugated bilinears
  - mixed-sector counts were made sector-local
- the model metadata layer was extended with basic parameter lookup and
  assumptions infrastructure:
  - parameter assumptions can now be stored and queried through the model
  - validation/reporting code can consult declared parameter metadata without
    reaching into raw expressions
- focused regression coverage was added for:
  - sector-filtered reporting
  - mass-mixing diagnostics
  - parameter lookup / parameter assumptions
- the main final walkthrough notebook was cleaned up and shortened so the live
  examples track the current API more directly

What this achieved:

- validation moved from a minimal diagnostic pass toward a more useful model
  inspection layer
- the reporting API can now separate sectors and flag suspicious mass
  structure without conflating those checks with hard compilation failures
- parameter metadata is now available as a first-class model-level concept,
  which prepares later validation and assumption-aware workflows
- the final walkthrough notebook became easier to use as the compact reference
  for the current public API

### 2026-05-04: branch cleanup, Python-compatibility pass, and first compact Lagrangian-list notebook

What happened:

- the accidental reintroduction of the old top-level `src/model.py` file was
  removed so the source tree returned to the intended modular `src/model/*`
  layout
- the direct `src/symbolic/spenso_gamma_checks.py` entrypoint was fixed so it
  can be run successfully from the repository venv as documented
- a compatibility pass replaced Python 3.10 union-annotation syntax in the
  covered source files with older-style `typing.Optional[...]` /
  `typing.Union[...]` forms
- the repository was reviewed after that cleanup:
  - syntax compilation succeeded on `src`, `tests`, and `examples`
  - the full test suite passed
  - the main runnable example suites passed
- a new compact notebook,
  `notebooks/list_lagrangians.ipynb`,
  was started as a short reference organized around
  `Model(..., lagrangian_decl=...)`
- the first version of that notebook covered:
  - basic scalar examples
  - derivative scalar examples
  - a first fermion section in the same compact model-layer style

What this achieved:

- the repository tree was brought back into a cleaner post-refactor state
  before continuing notebook/API work
- the documented gamma-check validation path became reliable again
- the covered source files are now easier to run on slightly older Python
  versions
- the project gained a second, much shorter notebook focused specifically on
  “how to write Lagrangians with the current model layer,” separate from the
  longer workflow walkthroughs

### 2026-05-05: typed ghost declarations, lowering-based shorthand resolution, and compact SU(2) notebook coverage

What happened:

- the ghost declaration API was cleaned up around a typed
  `GhostField(..., ghost_of=...)` helper while keeping compatibility with the
  older `kind="ghost"` path
- the ordinary `CovD(...)` machinery was generalized so adjoint ghost fields
  are treated through the same representation-aware compiler path as covered
  scalar, fermion, and quark matter fields
- this enabled direct FeynRules-style ghost input such as
  `- Ghost.bar * PartialD(CovD(Ghost, mu), mu)` without adding a ghost-only
  backend path
- `GaugeFixing(...)` was reworked so the helper lowers through the same local
  derivative-expression machinery as a manual tensor declaration instead of
  bypassing the ordinary lowering path
- compact field-occurrence syntax sugar was added at the metadata layer:
  - `Photon(mu)`
  - `Gluon(mu, a)`
  - `GhostG.bar(a)`
- the shorthand semantics were then resolved in local lowering rather than in
  the Wick-contraction engine:
  - divergence-like forms such as `PartialD(Photon(mu), mu)` and
    `PartialD(Gluon(mu, a), mu)` now lower to explicit metric-contracted
    derivatives with fresh internal Lorentz labels
  - the same lowering pass was generalized to the ghost-gluon product-rule
    branch so direct FeynRules-style ghost terms use the external gluon
    Lorentz slot consistently on both momentum terms
- the vertex engine was simplified again after that refactor:
  it now consumes ordinary lowered tensor structure instead of carrying a
  divergence-specific index-remapping rule
- focused regression coverage was added and updated for:
  - positive and negative shorthand-derivative cases
  - scalar and vector-current regressions
  - helper-vs-manual gauge-fixing equality for U(1) and SU(3)
  - direct FeynRules-style ghost-gluon vertices with exact leg-order checks
- `notebooks/list_lagrangians.ipynb` was rebuilt as a compact reference
  notebook around `Model(..., lagrangian_decl=...)` and extended with:
  - scalar examples
  - fermion examples
  - gauge, gauge-fixing, and ghost examples
  - pure `SU(2)_L` examples
  - mixed `SU(2)_L x U(1)_Y` examples showing the full set of emitted current
    and contact vertices, including the mixed `W B` scalar contact
- the live validation paths were rerun successfully:
  - `./.venv/bin/pytest -q`
    - result: `272 passed`
  - notebook execution check for `notebooks/list_lagrangians.ipynb`
    - result: passed

What this achieved:

- the ghost sector now fits the same declarative model-building story as the
  covered matter sectors instead of depending on special wrappers alone
- direct FeynRules-style ghost and gauge-fixing source terms are now real
  first-class inputs to the public declaration API
- the compact field-call syntax improves notebook/model readability without
  moving symbolic interpretation into the wrong architectural layer
- lowering now carries the responsibility for shorthand resolution, which is a
  cleaner boundary than keeping special symbolic-index heuristics in the core
  vertex engine
- the gauge-fixing helper and its manual tensor form are now explicitly locked
  to the same backend behavior
- the notebook coverage now includes the non-abelian `SU(2)` and mixed
  electroweak-style cases in the same compact Lagrangian-declaration style as
  the scalar, QED, and QCD sections

### 2026-05-06: gauge-fixing provenance, unbroken SM gauge-structure coverage, and SU(2) quartic validation

#### What happened

* Sector classification for compiled/reporting views was tightened so that
  `GaugeFixing(...)` helper terms and their canonical manual tensor forms now
  report consistently as `gauge_fixing`, instead of diverging at the
  `vertex_report(...)` / `vertex_signatures(...)` layer.

* This was done by moving away from label-text heuristics toward explicit term
  provenance:

  * helper-generated gauge-fixing terms now carry explicit sector/origin
    metadata;
  * the lowering layer now recognizes the canonical manual
    divergence-squared vector form conservatively and assigns the same
    reporting sector;
  * related tests were added for helper/manual parity, dummy-label
    invariance, and protection against accidental over-classification of
    unrelated vector-derivative bilinears.

* `notebooks/list_lagrangians.ipynb` was extended with a compact full
  **unbroken Standard Model gauge-structure** section using the existing
  symbolic model-building API with
  `SU(3)_c x SU(2)_L x U(1)_Y`.

* That section now includes:

  * SM-like matter fields with the expected color, weak, and hypercharge
    assignments;
  * a minimal kinetic `CovD(...)` test covering quark, lepton, and Higgs
    matter;
  * a second “full gauge” model including the three field-strength kinetic
    terms for `G`, `W`, and `B`;
  * compact cells that print only the emitted vertex signatures, without the
    full rules, for quick inspection.

* The SM-like field assignments used in the notebook are:

  * `qL : (3,2,1/6)`
  * `uR : (3,1,2/3)`
  * `dR : (3,1,-1/3)`
  * `lL : (1,2,-1/2)`
  * `eR : (1,1,-1)`
  * `H : (1,2,1/2)`

* The local symbolic notebook emits the expected **28 vertex signatures** for
  the full gauge test:

  * 19 interaction signatures;
  * 9 quadratic/two-point kinetic signatures.

* The corresponding FeynRules comparison gives the expected **19 interaction
  vertices**, because `FeynmanRules[...]` in that setup lists the interaction
  vertices and not the same quadratic/two-point kinetic signatures printed by
  the local notebook.

* An explicit notebook cell was added to print the internal external-leg
  assignment for the

  ```text
  ('H.bar', 'H', 'W')
  ```

  and

  ```text
  ('H.bar', 'H', 'B')
  ```

  vertices, together with the unsimplified rules, so the momentum ordering
  `q1/q2/q3` can be checked directly against the field order.

* An apparent sign mismatch in the scalar-gauge three-point vertices was
  resolved:

  * the local notebook prints the vertices as `('H.bar', 'H', W/B)` with
    `q1 = H.bar` and `q2 = H`;
  * FeynRules returns the same vertex ordered as `{H, Hbar, W/B}`;
  * after mapping momenta using the actual FeynRules ordering, the signs
    agree.

* The `SU(2)` quartic gauge-boson vertex was cross-checked against the
  FeynRules delta-basis form:

  * a reusable helper, `simplify_su2_ff(expr)`, was added to rewrite narrow
    `SU(2)` structure-constant products `f*f` into adjoint-metric /
    Kronecker-delta form;
  * the helper was tested on the compiled `WWWW` rule and matched the
    expected FeynRules basis exactly;
  * the notebook’s earlier `sympy`-based attempt was replaced by a
    `symbolica`-native check showing:

    * the raw `f*f` basis;
    * the rewritten delta basis;
    * the FeynRules-style target expression;
    * a final difference of `0`.

* The `WWW` and `GGG` vertices were also compared against the FeynRules
  forms:

  * both match the expected non-abelian three-gauge-boson structure;
  * the only notation differences are coupling names, momentum labels, and
    structure-constant naming.

* Fermion-gauge vertices were checked against FeynRules:

  * `qLbar qL G`, `qLbar qL W`, `qLbar qL B`;
  * `uRbar uR G`, `uRbar uR B`;
  * `dRbar dR G`, `dRbar dR B`;
  * `lLbar lL W`, `lLbar lL B`;
  * `eRbar eR B`.

* The hypercharge coefficients agree with the expected values:

  * `Y(qL) = 1/6`
  * `Y(uR) = 2/3`
  * `Y(dR) = -1/3`
  * `Y(lL) = -1/2`
  * `Y(eR) = -1`
  * `Y(H) = 1/2`

* Higgs-gauge vertices were checked against FeynRules:

  * `Hbar H W`;
  * `Hbar H B`;
  * `Hbar H W W`;
  * `Hbar H B B`;
  * `Hbar H W B`.

* The four-point Higgs-gauge vertices match directly in coefficient, sign,
  and index structure. The three-point Higgs-gauge vertices also match once
  the actual FeynRules external-leg ordering is used.

* Validation was rerun after these changes:

  * focused `SU(2)` tests passed;
  * the full test suite passed;
  * the updated notebook tail executed successfully.

#### What this achieved

* Helper/manual parity now holds not only at the level of Feynman rules, but
  also at the reporting/provenance layer. This makes the model-inspection API
  more trustworthy.

* The compact notebook now includes a genuinely full **unbroken SM
  gauge-structure** example for Standard Model-like matter content, rather
  than only sector-isolated QED/QCD/`SU(2)` fragments.

* The notebook can now inspect larger models in two complementary ways:

  * full rules when the algebraic expression is needed;
  * signature-only interaction lists when the goal is just to check which
    couplings are present.

* The explicit leg-assignment cell removes ambiguity about which external
  momentum is attached to which ordered field in selected vertices.

* The `WWWW` comparison closes an important confidence gap between the current
  symbolic non-abelian output and the more familiar FeynRules delta-basis
  presentation for `SU(2)` quartic gauge interactions.

* The validation establishes that the symbolic implementation reproduces the
  expected unbroken SM gauge-sector interactions:

  * correct matter representations;
  * correct hypercharge coefficients;
  * correct non-abelian self-interactions;
  * correct Higgs-gauge interactions;
  * consistent interpretation of external-leg ordering.

#### Validation summary

```text
Local symbolic notebook:
  28 vertex signatures
  = 19 interaction signatures
  + 9 quadratic/two-point kinetic signatures

FeynRules comparison:
  19 interaction vertices

SU(2) WWWW check:
  raw f*f basis -> SU(2) delta basis -> FeynRules-style target
  final difference: 0

Resolved ambiguity:
  apparent Hbar-H-gauge sign mismatch was due to FeynRules external-leg reordering,
  not a real convention mismatch.
```

#### Caveats

* This is not a full broken Standard Model implementation. The section
  currently tests the **unbroken gauge structure** only.

* The current notebook section does not yet include:

  * electroweak symmetry breaking;
  * physical `(A/Z/W^\pm)` mixing;
  * Yukawa interactions;
  * Higgs potential;
  * CKM structure;
  * ghost terms;
  * a complete gauge-fixing and ghost sector for the full SM.

* The `GGGG` vertex remains in the generic non-abelian `f*f` basis. This is
  appropriate for `SU(3)` and should not be simplified to a pure
  delta-delta structure in the same way as the `SU(2)` `WWWW` vertex.

### 2026-05-11: lighter `Model(..., lagrangian_decl=...)` metadata requirements

What happened:

- the model-construction path was relaxed so local
  `Model(..., lagrangian_decl=...)` usage no longer always needs an explicit
  `fields=(...)` list when the fields can be inferred from the declaration
- gauge-group requirements were also tightened to the cases that actually need
  them:
  if a declaration does not use `CovD(...)` or other gauge-dependent
  structures, the model no longer needs explicit `gauge_groups=(...)`
- the related notebook/model-building flow was cleaned up so the compact
  `lagrangian_decl=...` examples stay closer to how users will actually write
  small models

What this achieved:

- reduced boilerplate for direct declarative model building
- made the local `Model(..., lagrangian_decl=...)` workflow less fragile for
  simple non-gauge examples
- improved the separation between metadata that is genuinely required for
  compilation and metadata that can be inferred automatically

### 2026-05-12: flavor-index expansion and FeynRules-style field-class declarations

What happened:

- flavor/class expansion support was added around explicit flavor index
  metadata rather than hard-coded generation semantics:
  - `IndexRole.FLAVOR`
  - `flavor_index(...)`
  - field-level `flavor_index` and `class_members`
  - indexed `Parameter` metadata with `components` and `allow_summation`
- the compiled Lagrangian extraction path was extended so
  `feynman_rule(...)`, `feynman_rules(...)`, `vertex_signatures(...)`, and
  `vertex_report(...)` can all run with `flavor_expand=...`
- the public declaration API was then cleaned up to follow the FeynRules
  model-file workflow more closely:
  - explicit indices first
  - then gauge representations / gauge groups
  - then field classes
  - then parameters
  - then the Lagrangian
- the recommended public field-declaration helpers are now the ordinary
  metadata constructors with explicit flavor-class arguments:
  - `dirac_field(..., class_members=..., flavor_index=...)`
  - `scalar_field(..., class_members=..., flavor_index=...)`
- selected flavor expansion was also generalized beyond `True/False`:
  `flavor_expand` can now be `True`, one flavor index, or an iterable of
  flavor indices
- the main flavor example and notebook flow were rewritten around explicit
  FeynRules-like declarations for charged leptons and colored quark classes,
  including two-index Yukawa-matrix-style parameters with off-diagonal zero
  components
- regression coverage was expanded for:
  - charged-lepton / up-quark / down-quark field-class metadata
  - flavor expansion with color preserved
  - diagonal matrix-style Yukawas
  - retained one-index `allow_summation` behavior
  - selected flavor expansion

What this achieved:

- the project now has a much clearer public flavor API that matches the
  FeynRules mental model more directly
- flavor-generic classes remain explicit at declaration time, while expanded
  member fields correctly drop the selected flavor index and keep non-flavor
  indices such as color and spinor
- the main examples now present flavor structure as ordinary model metadata,
  not as a special toy constructor
- the flavor backend stayed intact while the user-facing workflow became more
  realistic for Standard Model-like model files
- validation stayed clean after the refactor:
  the full pytest suite passed with `316 passed`

### 2026-05-13: monomial-wide typed-label validation for plain symbolic indices

What happened:

- a safety review checked how plain symbolic labels such as
  `f`, `h`, and `col` are interpreted across one whole interaction monomial
- that review found a real gap in the old lowering path:
  symbolic labels were validated slot-locally, so incompatible reuse like
  `uq.bar(f, col) * uq(col, f) * Phi` could compile silently
- the lowering layer now builds one monomial-wide registry from label name to
  exact `IndexType`
- the new validation scans:
  - explicit field-slot labels
  - local tensor / derivative labels on factors such as
    `Gamma(...)`, `PartialD(...)`, `Metric(...)`, `T(...)`,
    `StructureConstant(...)`, and `FieldStrength(...)`
  - indexed parameter calls in the coefficient when the parent
    `Model(..., parameters=...)` provides the `Parameter.indices` metadata
- the validation now runs before auto-filled labels and before the local
  lowering heuristics that infer implied contractions
- dedicated regression tests were added for:
  - generation/colour label swapping inside one fermion bilinear
  - a parameter label reused across incompatible generation and colour slots
  - the standalone local `Lagrangian(...)` field-only rejection path

What this achieved:

- the public API stays FeynRules-like:
  users can keep writing compact plain-symbol labels instead of explicit typed
  wrappers
- internally, one label name is now bound to exactly one index space per
  monomial, so incompatible reuse is rejected early instead of compiling into a
  wrong contraction
- flavor/non-flavor conflicts and non-flavor/non-flavor conflicts are now both
  caught on the declared-term path
- lowering heuristics can no longer make an invalid symbolic input look valid
  by filling missing labels after the fact
- the safety fix is covered by tests and the whole suite stayed green:
  `170 passed`

### 2026-05-14: flavor-expansion hardening, cache normalization, and notebook sync

What happened:

- the flavor/index handling path was reviewed and documented further in the
  index-handling and flavor-expansion notebooks
- flavor-index simplification was tightened across the model layer so expanded
  flavor classes and their metadata behave more predictably
- the flavor-expanded term cache in `src/model/lagrangian.py` was added and
  then normalized so equivalent multi-index `flavor_expand` requests reuse the
  same cached expansion
- plain symbolic index-name handling was relaxed in the relevant compiler and
  post-processing paths so user-facing labels remain usable without weakening
  typed index checks
- the `list_lagrangians.ipynb` notebook was updated so Yukawa/flavor content
  is displayed more clearly

What this achieved:

- made flavor expansion cheaper to reuse across repeated vertex queries
- reduced avoidable distinctions between equivalent flavor-expansion requests
- kept the public notebook story aligned with the live flavor/index machinery
- improved the robustness of symbolic index-name handling without changing the
  typed contraction rules

### 2026-05-15: unbroken Standard Model builder, Yukawa cleanup, flavor classes,
and FeynRules-output comparison

What happened:

- a first non-BFM unbroken Standard Model builder was added under
  `src/model/standard_model_unbroken.py`, together with a walkthrough notebook
  and smoke tests
- the builder was then tightened so the Yukawa sector uses explicit stand-in
  conjugate matrices for the hermitian-conjugate terms and more explicit index
  contractions matching the intended FeynRules structure
- the generation-carrying fermions were declared as actual flavor classes with
  `class_members`, which made `flavor_expand=True` work for the unbroken SM
  example and enabled direct expansion tests
- helper scripts were added to align and normalize Python and FeynRules vertex
  outputs for direct comparison
- the quartic gluon `GGGG` output was inspected in detail:
  the internal adjoint dummy label was traced, structure-constant products were
  canonicalized, and a regression test was added comparing the derived vertex
  to the FeynRules form after canonicalization

What this achieved:

- established a first reusable non-BFM unbroken SM model in the live
  declarative/model-building framework
- made the Yukawa sector more honest about conjugation in the absence of a
  symbolic `HC` operator
- connected the unbroken SM example to the existing flavor-expansion machinery
- created a cleaner output-comparison path for checking Python expressions
  against FeynRules
- showed that the apparent quartic-gluon mismatch was a presentation /
  canonicalization issue rather than a real sign bug

### 2026-05-18: lowered-Lagrangian operator layer, Symbolica export, review fixes,
and notebook specification

What happened:

- added a new intermediate layer **after** compile/lower/expand and **before**
  `vertex_engine.py`, without modifying the vertex engine:
  - `src/lagrangian/operator_action.py` — `FieldOperator`, graded Leibniz on
    ordered `InteractionTerm`s, slot-wise splice of `FieldOccurrence`
    replacements, `replacement_operator` sugar
  - `src/lagrangian/symbolica_export.py` — `interaction_term_to_symbolica`,
    `lagrangian_to_symbolica` (display / scalar algebra only; fermion order not
    preserved in Symbolica)
  - `CompiledLagrangian.apply_operator(...)` and
    `CompiledLagrangian.to_symbolica(...)` in `src/model/lagrangian.py`, with
    optional `flavor_expand` on both (same semantics as `feynman_rules`)
- confirmed Symbolica v1.4 multiplication is commutative, so
  `InteractionTerm` remains the authoritative representation for ordered
  fermion/ghost products; Symbolica is a view, not a reversible source of truth
- reviewed the first implementation and fixed three correctness gaps:
  - **bilinear metadata:** `_validate_and_remap_bilinears` now requires a
    unique matching Dirac-fermion factor in the replacement when a slot is a
    `closed_dirac_bilinears` endpoint; ghosts are not accepted as bilinear
    endpoints; stale `(psibar, psi)` pairs are rejected instead of silently
    remapped
  - **derivatives on product replacements:** when a slot already carries
    `DerivativeAction`s and the replacement has length `N > 1`, the engine
    performs bosonic Leibniz fan-out across replacement slots (`N^M` terms for
    `M` derivatives), instead of retargeting all derivatives to the first
    replacement slot only
  - **flavor-expanded export:** `to_symbolica(flavor_expand=True)` and
    `apply_operator(..., flavor_expand=True)` use `_expanded_terms(...)` like
    the vertex API, not the flavor-generic `.terms` list alone
- added tests:
  - `tests/test_operator_action.py` — 29 cases (Leibniz signs, bilinear
    preserve/violate, derivative fan-out, flavor export, Symbolica commutative
    limitation, `CompiledLagrangian.apply_operator` wiring)
  - `tests/test_scalar_total_derivative_identity.py` — scalar IBP-motivation
    check that `L_1 - L_2` matches `c * ∂_μ(φ² ∂_μ φ)` after manual lowering,
    where `L_1 = c φ² □φ` and `L_2 = -2c φ (∂_μ φ)²`, via `to_symbolica()`
    canonical comparison (not an automatic IBP engine)
- wrote and expanded `notebooks/operator_action_and_symbolica_walkthrough.ipynb`
  as the primary usage/spec artifact:
  - sections 1–6: `Model` → `lagrangian()`, inspect terms, `to_symbolica()`,
    `apply_operator` (simple replacement, product replacement + derivative
    fan-out, flavor export, odd-operator signs with bilinear-safe replacements)
  - section 7: scalar total-derivative identity — manual expansion in the
    declarative DSL; documents that equivalence is **modulo a total derivative**
    under `∫ d⁴x`, not as identical local Lagrangians
  - section 8: shows the current layer is **replacement-based**, not a true
    `∂_μ` operator — `d_μ[Φ] = PartialD(Φ, μ)` cannot be encoded as a
    `FieldOccurrence` replacement because fresh derivatives live in
    `InteractionTerm.derivatives`
  - section 9: large unbroken SM + linear `GaugeFixing(...)` export check —
    `to_symbolica()` shows lowered `PartialD(...)` and fixing parameters, not
    raw `CovD` / `FieldStrength` / `GaugeFixing` wrappers
- full suite stayed green: `232 passed`

What this achieved:

- a first **replacement-style** operator calculus on fully lowered monomials,
  suitable for field substitutions, gauge/BRST-like maps that rewrite slots,
  and redistribution of **existing** partial derivatives across product-valued
  replacements
- a clear pipeline position: declarative `Model` → `CompiledLagrangian` →
  optional `apply_operator` / `to_symbolica` → unchanged `feynman_rules`
- separated three notions that must not be conflated in later work:
  - **replacement / variation** on field slots (`Phi → Chi`, `psi → c·psi`)
  - **true spacetime derivatives** (`∂_μ` creating new `DerivativeAction`s)
  - **total derivatives / IBP** (equivalence only under the integral)
- documented the main **remaining gaps** for follow-up:
  - richer operator results that can attach new `DerivativeAction`s (a real
    `d_μ` on products)
  - a dedicated IBP / total-derivative simplification layer above lowered terms
  - safe reverse conversion Symbolica → `InteractionTerm` only with a field
    registry and explicit ordering metadata (stub left as `NotImplementedError`)

### 2026-05-19: derivative-aware operator actions, infinitesimal gauge variation,
unified grouped vertex extraction, and label-safe replacements

What happened:

- the lowered-operator layer in `src/lagrangian/operator_action.py` was
  extended beyond pure slot replacement:
  `OperatorSummand` now carries `new_derivatives`, the engine translates those
  replacement-local derivative targets into absolute `InteractionTerm` slot
  indices, and the runtime action can therefore create fresh
  `DerivativeAction(...)` entries instead of only redistributing derivatives
  that were already present on the acted slot
- on top of that extension, `partial(...)` was turned into a dual-purpose API:
  - `partial(mu)` now returns a runtime `FieldOperator` that acts by graded
    Leibniz on lowered terms and attaches a fresh derivative on each acted slot
  - `partial(mu, Phi)` remains the declarative `PartialD(Phi, mu)` shortcut for
    model declarations
  - the runtime form also supports `on=...` filtering so it can act only on a
    selected field or tuple of fields
- `gauge_variation(group=..., parameter=...)` was implemented as a concrete
  infinitesimal gauge-transformation operator on compiled Lagrangians:
  - U(1) matter variations use the field charge from
    `field.quantum_numbers[group.charge]`
  - non-abelian matter variations insert the appropriate generator and rotate
    the representation-slot label explicitly
  - gauge-boson variations produce both the inhomogeneous `+ ∂_μ α`
    contribution and the homogeneous `- g f^{abc} α^b A^c_μ` contribution
  - the gauge parameter is materialized as a synthetic scalar field `alpha`
    (with the adjoint index when needed), so the existing lowered-term
    derivative bookkeeping can be reused
- the compiled-Lagrangian view gained a cleaner public surface:
  - `CompiledLagrangian.to_symbolica(...)` stayed as the display/export view
  - `Model.to_symbolica(...)` was added as the top-level forwarder, so the user
    does not need to drop to `model.lagrangian()` just to inspect the compiled
    expression
  - grouped whole-Lagrangian vertex extraction was folded into zero-argument
    `feynman_rule(...)`, with `arity=` and `select=` preserved there; the old
    plural `feynman_rules(...)` entry point was removed across code, tests,
    notebooks, and docs
- the operator-action layer was hardened again around replacement metadata:
  replacements now inherit missing compatible slot labels from the acted
  occurrence when the target is unambiguous
  (same exact `IndexType`, one compatible replacement occurrence, same slot
  multiplicity)
- that label-inheritance fix closed a real usability gap in the notebook odd
  operator example:
  a bilinear-preserving replacement such as `psi -> xi` no longer drops the
  original spinor label, so the resulting derived terms remain consumable by
  the usual `feynman_rule(...)` vertex-extraction path
- the operator/gauge notebook story was synchronized with the live code:
  `notebooks/operator_action_and_symbolica_walkthrough.ipynb` now documents the
  runtime partial-derivative operator, gauge variation on lowered terms, and
  working vertex extraction on operator-derived Lagrangians
- regression coverage was expanded materially:
  - `tests/test_operator_action.py` now covers runtime `partial(...)`,
    fresh-derivative insertion, bilinear-preserving differentiated fermion
    chains, Symbolica export forwarding, and the label-inheritance path needed
    for downstream `feynman_rule(...)`
  - `tests/test_gauge_variation.py` covers U(1) and non-abelian matter
    invariance checks, wrong-charge / non-singlet non-invariance checks, the
    explicit gauge-boson variation structure, and a well-formedness check for
    pure Yang-Mills variation
  - `tests/test_vertex_reporting.py` now pins the zero-argument
    `feynman_rule(arity=...)` / `feynman_rule(select=...)` behavior directly
- the full test suite was rerun after these changes and passed:
  `269 passed`

What this achieved:

- the operator layer crossed an important threshold:
  it is no longer only a replacement calculus on existing lowered slots, but
  can now also create fresh spacetime-derivative structure in the same
  `InteractionTerm` representation that the rest of the pipeline already trusts
- the project now has a direct lowered-term implementation of infinitesimal
  gauge variation, which makes gauge-invariance checks possible without adding
  a second symbolic engine or reinterpreting the declarative source language
- the public API became more coherent:
  users can inspect compiled expressions from either `Model` or
  `CompiledLagrangian`, and there is now one canonical vertex-extraction name
  (`feynman_rule(...)`) for both single-vertex and grouped whole-Lagrangian
  queries
- derived Lagrangians produced by custom operators became more robust
  downstream:
  index labels needed by the fermion vertex machinery are no longer lost in
  the common "same-structure replacement field" case
- the notebook/documentation story moved closer to the real code:
  the main operator walkthrough now demonstrates features that are actually
  implemented end-to-end, including fresh derivatives, gauge variation, and
  vertex extraction from operator outputs

### 2026-05-26: Yang-Mills canonicalization hardening, BRST milestone, and regression split

What happened:

- the recent Yang-Mills canonicalization work in
  `src/symbolic/tensor_canonicalization.py` (plain-head metric contraction,
  commuting `PartialD` normalization, local Jacobi reduction in the `f*f`
  basis, antisymmetry-based zero-term dropping, and one-shot
  `canonize_full(...)` orchestration) was reviewed as a safety/maintainability
  pass rather than refactored wholesale
- plain-head metric contraction was tightened around an explicit lightweight
  slot-kind registry for the currently exported heads (`G`, `alpha`) so metric
  rewrites stay tied to declared Lorentz/adjoint slots instead of matching
  arbitrary plain function heads
- derivative-index contraction was kept Lorentz-only for `PartialD(...)`,
  avoiding accidental application of non-Lorentz adjoint metrics to derivative
  slots
- the commuting-derivative pass remained local to the exact `PartialD` head and
  was guarded to leave malformed/non-standard arity calls untouched
- the Jacobi reducer documentation was clarified with the explicit coefficient
  elimination step (`p13 = p12 + p14`) used to compute deterministic basis
  coefficients
- antisymmetry zero dropping was narrowed to legal dummy-index relabelings:
  swap trials are now attempted only on symbols recognized as dummy labels
  after canonicalization, reducing the risk of dropping terms with free/external
  indices
- the internal pass was renamed to the more explicit
  `_drop_yang_mills_antisymmetric_zero_terms(...)` with a compatibility wrapper
  kept for the previous private helper name
- `canonize_full(...)` gained explicit pass toggles:
  - `run_commuting_partial_derivatives`
  - `run_jacobi_reduction`
  - `run_yang_mills_antisymmetric_zero_drop`
  so users can opt out of YM-specific assumptions when they want a more purely
  algebraic canonicalization path
- focused regression coverage was added in
  `tests/test_tensor_canonicalization_pipeline.py` for:
  - plain-head metric contraction on `G`, `alpha`, and `PartialD(G,...)`
  - commuting `PartialD` behavior on second/third derivatives and non-`PartialD`
    heads
  - Jacobi identity reduction (including sign/permutation variants) and
    preservation of non-Jacobi `f*f` structures
  - YM antisymmetry-based zero dropping and a nearby nonzero negative case
  - explicit disabling of YM antisymmetry dropping through `canonize_full(...)`
- a first BRST runtime operator was added in
  `src/lagrangian/operator_action.py` alongside the existing even
  `gauge_variation(...)` machinery, instead of trying to fake BRST as
  "gauge variation with `alpha -> c`"
- the implementation was kept deliberately narrow and Yang-Mills-focused:
  it acts only on
  - the chosen gauge boson
  - its ghost
  - its antighost
  - the auxiliary Nakanishi-Lautrup field
- the BRST operator was made explicitly odd (`parity = 1`) so the existing
  graded Leibniz rule on ordered `InteractionTerm` slots supplies the needed
  sign changes automatically
- the non-abelian elementary rules were implemented with the current
  convention
  - `s A^a_mu = partial_mu c^a + g f^{abc} A^b_mu c^c`
  - `s c^a = -1/2 g f^{abc} c^b c^c`
  - `s cbar^a = B^a`
  - `s B^a = 0`
- the abelian specialization was implemented in the same layer:
  - `s A_mu = partial_mu c`
  - `s c = 0`
  - `s cbar = B`
  - `s B = 0`
- the ghost/antighost path was aligned with the repository's existing field
  convention:
  the common constructor path is now
  `brst_transformation(group=..., ghost=c, auxiliary=B)`, with the
  antighost inferred as the conjugated occurrence `c.bar(...)` when the ghost
  declares a concrete `conjugate_symbol`
- explicit validation/negative-path checks were added for:
  - non-odd ghost input
  - non-abelian BRST without a structure constant
  - antighost handling without an auxiliary field
  - the inferred-antighost path when the ghost has no explicit
    `conjugate_symbol`
- focused regression coverage was added in `tests/test_brst_transformation.py`
  for:
  - operator/interface construction
  - elementary field variations
  - graded ghost-antighost Leibniz signs
  - nilpotency on `A`, `c`, `cbar`, and `B`
  - the abelian specialization
  - unrelated-field non-action and the negative construction cases above
- the operator walkthrough notebook
  `notebooks/operator_action_and_symbolica_walkthrough.ipynb`
  gained a new section
  `BRST transformations and nilpotency`
  that:
  - defines the ghost and auxiliary fields
  - constructs the BRST operator
  - prints `sG`, `sc`, `s cbar`, and `sB`
  - checks `s^2 G = 0` and `s^2 c = 0` with
    `canonize_full(..., run_color=False, infer_indices=True)`
- the BRST additions were validated against the surrounding operator/gauge
  stack:
  `tests/test_operator_action.py`, `tests/test_gauge_variation.py`, and
  `tests/test_brst_transformation.py` passed together after the change

What this achieved:

- preserved the working YM-cancellation pipeline while reducing over-broad
  rewrite scope in its highest-risk passes
- made the distinction clearer between:
  - generic tensor canonicalization
  - ordinary-flat-derivative assumptions
  - YM/color-specific compact identities
- improved confidence that future additions (for example, non-commuting
  covariant-derivative heads) will not be silently simplified by the current
  `PartialD` normalizer
- turned the recent notebook success (`delta L_YM -> 0`) into a broader,
  fine-grained regression safety net that can catch local regressions before
  they reach the full end-to-end YM check
- the project now has a physically correct first BRST layer on the compiled
  Lagrangian side, not just an ordinary gauge-variation surrogate
- the existing ordered-slot operator engine proved flexible enough to support
  odd derivations and ghost-sector sign logic without another architectural
  rewrite
- the crucial non-abelian identities behind BRST nilpotency are now exercised
  end-to-end in the live code path:
  ghost anticommutation through ordered factors,
  graded Leibniz signs through operator parity,
  and Jacobi cancellation through the current `f`-basis canonicalizer
- the user-facing BRST entry point became simpler in the common FP case while
  still allowing an explicit separate antighost field when needed
- the notebook story now covers a second major symmetry operation beyond
  infinitesimal gauge variation, which makes the operator-action layer much
  closer to the eventual gauge-fixing / ghost-sector workflow
- the project reached the first BRST milestone cleanly without overextending
  into matter BRST rules or integration-by-parts-dependent gauge-fixing
  identities too early

### 2026-05-27: BRST notebook cleanup and wildcard-coefficient notebook examples

What happened:

- the BRST walkthrough section in
  `notebooks/operator_action_and_symbolica_walkthrough.ipynb`
  was cleaned up so the gauge-fixing/ghost-sector example uses compiled
  Lagrangians consistently:
  - `K_brst` is now created as `Model(...).lagrangian()`
  - the combined BRST-invariance check now forms
    `L_total_brst = L_su3_ym.lagrangian() + s_K_brst`
    instead of trying to add a top-level `Model` to a
    `CompiledLagrangian`
- this removes the notebook runtime error
  `TypeError: unsupported operand type(s) for +: 'Model' and 'CompiledLagrangian'`
  and makes the intended `s(L_YM + s(K)) = 0` check executable again
- chapter 13 of the same notebook was rewritten into a shorter and more
  explicit coefficient-manipulation story centered on the distinction between:
  - native Symbolica `coefficient(...)` on exact literals
  - wildcard-aware coefficient extraction through the project helper
    `pattern_coefficient(...)`
  - the equivalent pure-Symbolica manual workaround based on
    `match(...)`, `replace_wildcards(...)`, and then literal
    `coefficient(...)`
- the new first cells in chapter 13 now use a deliberately tiny toy
  expression so the key behavior is obvious:
  `coefficient(pattern_with_wildcards)` returns `0`, while both the helper and
  the manual match-and-substitute path recover the expected summed result
- a compact physics-facing example was kept in chapter 13 as well:
  the BRST-exact gauge-fixing/ghost expression `s_K_brst.to_symbolica().expand()`
  is used to show that literal `coefficient(...)` is still useful for
  extracting pieces such as the `xi` and `gS` sectors of a concrete exported
  expression
- the old pure-SU(3) Yang-Mills subfactor-extraction example was restored as
  the final cell of chapter 13, using the same built-in-only workflow as
  before:
  - `expr = L_su3_ym.to_symbolica()`
  - wildcard `pattern = G(mu1_, a1_) * G(mu2_, a2_)`
  - `expr.match(..., partial=True)`
  - `pattern.replace_wildcards(match)`
  - `expr.coefficient(factor)`
  so the notebook still preserves the more realistic nontrivial example after
  the new minimal toy introduction
- the trailing scratch/blank cells at the end of chapter 13 were removed so
  the notebook now ends on the restored SU(3) Yang-Mills example instead of an
  incomplete experiment
- the updated chapter-13 cells were checked in the project `.venv`, confirming
  that:
  - native wildcard `coefficient(...)` returns `0`
  - the helper and manual workaround agree on the toy example
  - the restored SU(3) Yang-Mills cell reproduces the earlier sequence of
    matched `G*G` factors and residual coefficients

What this achieved:

- the BRST section of the walkthrough now matches the actual container/API
  distinction in the codebase:
  top-level `Model` convenience methods can forward to compiled behavior, but
  additive composition still has to happen at the compiled-Lagrangian level
- chapter 13 now teaches the coefficient issue in the right order:
  a minimal failure case first, then the project solution, then the pure
  Symbolica fallback, and only then the more complicated SU(3) Yang-Mills
  example
- the notebook now makes a sharper conceptual distinction between:
  - exact coefficient extraction
  - wildcard pattern extraction
  - subfactor matching inside larger interaction monomials
- the restored final SU(3) cell preserves the old research/debugging example
  without forcing the reader to infer the basic wildcard limitation from a very
  large expression
- the operator-action walkthrough is now more coherent as a document:
  the BRST section runs cleanly, and the final chapter closes with working,
  concrete Symbolica manipulation patterns that are directly useful for
  physics-side inspection of exported Lagrangians
