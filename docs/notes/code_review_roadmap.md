# Code Review Roadmap

## 1. Executive Summary

The codebase already has a usable layered design: `model` handles declarations and metadata, `lowering` normalizes source syntax, `compiler/gauge.py` assembles convention-fixed physics terms, and `symbolic/vertex_engine.py` performs contraction and vertex extraction. That is the right overall direction, and the current regression suite is strong for the implemented scope.

The main risks are not architectural collapse or “rewrite required” issues. They are concentrated, fixable problems in correctness-sensitive seams: local tensor-index binding can silently attach indices to the wrong fields, malformed field-strength declarations are accepted too easily, and the same kinetic operator can mean different things depending on which front-end path created it. On top of that, some large modules carry too many responsibilities, and the Symbolica/Spenso integration is still more workaround-heavy than it should be.

The near-term goal should be to make the system fail closed on ambiguous physics, tighten validation, and simplify only after semantic correctness is better protected.

## 2. Strengths

- The normalized backend object `InteractionTerm` is a strong design center and should remain the common target between declarations, compiler output, and extraction.
- The split between `model`, `lowering`, `compiler`, and `symbolic` is directionally correct and should be preserved.
- `GaugeGroup` / `GaugeRepresentation` / repeated-slot handling are well designed, especially `slot_policy='sum'` and explicit ambiguity checks.
- The declarative syntax for core use cases is readable: `I * psi.bar * Gamma(mu) * CovD(psi, mu)` is close to physics notation.
- Convention locking is already taken seriously through `docs/notes/CONVENTIONS.md` and focused tests.
- The regression suite is broad for the current scope: QED, QCD, SU(2), mixed-group scalar contacts, gauge fixing, ghosts, and electroweak SSB are all covered.
- The electroweak SSB layer is isolated in `src/model/ssb.py` instead of leaking broken-phase logic into the generic compiler.

## 3. Critical Correctness Issues (Must Fix First)

### 3.1 Ambiguous local free-tensor index binding can silently produce wrong vertices

- Description:
  Local lowering currently binds declared free-tensor indices to field slots by a flat “first available slot” assignment rather than by factor-local semantics.
- Why it is physically dangerous:
  A vertex can look structurally plausible while carrying the wrong Lorentz or gauge contraction. That changes the operator itself, not just its formatting.
- Exact file(s)/function(s) involved:
  - `src/model/lowering.py`
  - `_bind_declared_indices_to_field_slots(...)`
  - `_lower_local_interaction_monomial(...)`
- Concrete example of failure:
  - `A * B * C * Metric(mu, rho)` currently lowers without complaint.
  - The metric labels bind to `A` and `B`, not `A` and `C`.
  - This is a silent physics error.
- Proposed fix strategy:
  - Track declared-index provenance per factor instead of flattening all `(kind, label)` pairs into one list.
  - Bind free-tensor endpoints only when the intended field slots are unambiguous.
  - Raise a `ValueError` when a free tensor cannot be attached uniquely.
- Status checklist:
  - [x] understood
  - [x] test written
  - [x] fix implemented
  - [x] validated

### 3.2 Malformed field-strength declarations are accepted too easily

- Description:
  `FieldStrength(...) * FieldStrength(...)` lowering checks equality of the two factors, but not whether the Lorentz structure is physically sensible.
- Why it is physically dangerous:
  Inputs such as `F_{\mu\mu} F^{\mu\mu}` should vanish or be rejected. Accepting them as ordinary gauge-kinetic terms teaches the compiler the wrong operator.
- Exact file(s)/function(s) involved:
  - `src/lagrangian/lowering.py`
  - `lower_field_strength_monomial(...)`
  - `src/model/lowering.py`
  - `_lower_field_strength_monomial(...)`
- Concrete example of failure:
  - `FieldStrength(U1, mu, mu) * FieldStrength(U1, mu, mu)` compiles into ordinary gauge bilinears instead of failing.
- Proposed fix strategy:
  - Reject repeated Lorentz indices inside one field-strength factor.
  - Add explicit validation for allowed canonical forms before lowering to `GaugeKineticTerm`.
  - Cover both abelian and non-abelian cases with tests.
- Status checklist:
  - [x] understood
  - [x] test written
  - [x] fix implemented
  - [x] validated

### 3.3 Kinetic-term semantics depend on front-end path

- Description:
  Declarative `CovD(...)` monomials compile as full operators including free bilinears, while legacy `DiracKineticTerm` and `ComplexScalarKineticTerm` compile as gauge-only pieces.
- Why it is physically dangerous:
  The same intended Lagrangian term can include or omit two-point pieces depending on the entry path. That can cause silent missing terms or double counting.
- Exact file(s)/function(s) involved:
  - `src/compiler/gauge.py`
  - `compile_dirac_kinetic_term(...)`
  - `compile_complex_scalar_kinetic_term(...)`
  - `_compile_declared_covariant_core(...)`
  - `compile_covariant_terms(...)`
- Concrete example of failure:
  - `I * psi.bar * Gamma(mu) * CovD(psi, mu)` and `DiracKineticTerm(field=psi)` do not currently mean the same full operator.
- Proposed fix strategy:
  - Make “full operator” vs “interaction-only piece” explicit in the API and naming.
  - Either unify semantics or add explicit entry points for both modes.
  - Add parity tests that fail if the distinction is applied implicitly.
- Findings from implementation:
  - The split is currently intentional in behavior but was only partially explicit in code.
  - Declarative `CovD(...)` monomials are routed through analyzed source terms and compiled as full operators.
  - Legacy `DiracKineticTerm` / `ComplexScalarKineticTerm` declarations are routed through `model.covariant_terms` and compile only gauge-generated interactions.
  - The ambiguity came from both paths sharing the same normalized core types while relying on different compiler entry points.
- Design decision:
  - Keep legacy kinetic declarations gauge-only for backward compatibility.
  - Make the split explicit with an internal `include_free_bilinear` policy in `src/compiler/gauge.py` rather than changing public APIs.
- Tests added:
  - `test_legacy_qed_fermion_kinetic_term_is_gauge_only`
    - Proves legacy `DiracKineticTerm` does not create a 2-point free bilinear.
  - `test_legacy_scalar_qed_kinetic_term_is_gauge_only`
    - Proves legacy `ComplexScalarKineticTerm` does not create a 2-point free bilinear.
  - Existing declarative free-bilinear tests were tightened to assert that compiled output contains an actual 2-field term for both fermion and scalar cases.
- Status checklist:
  - [x] understood
  - [x] test written
  - [x] fix implemented
  - [x] validated

### 3.4 Fermion sign handling is only semantically safe for a narrow class of operators

- Description:
  The special `closed_dirac_bilinears` logic fixes relative signs only when the interaction is explicitly recognized as a complete product of closed Dirac bilinears. Outside that case, the engine falls back to raw permutation parity.
- Why it is physically dangerous:
  Relative signs in multi-fermion operators change amplitudes and interference. A formally similar operator can acquire the wrong sign family if provenance is missing.
- Exact file(s)/function(s) involved:
  - `src/symbolic/vertex_engine.py`
  - `_validate_supported_fermion_structure(...)`
  - `contract_to_full_expression(...)`
  - `src/model/lowering.py`
  - `_lower_local_interaction_monomial(...)`
- Concrete example of failure:
  - Four-fermion structures that are not lowered into explicit closed bilinear metadata remain sensitive to raw ordering rather than semantic chain structure.
- Proposed fix strategy:
  - Expand provenance marking for more supported multi-fermion structures.
  - Add focused sign tests for identical-fermion and mixed-species four-fermion operators.
  - Fail closed on structures whose sign semantics are not yet represented.
- Findings from diagnostics:
  - `src/model/lowering.py` currently marks `closed_dirac_bilinears` only for adjacent same-species `psibar ... psi` intervals.
  - `src/symbolic/vertex_engine.py` uses closed-bilinear sign handling only when those bilinears cover every fermion slot exactly once; otherwise it falls back to generic permutation parity.
  - Ordered four-fermion products built from disjoint closed bilinears are stable for both scalar and vector bilinears, including swapped bilinear order.
  - The reversed-order case `psi * psibar * chibar * chi` is suspicious: it is accepted, but it is not treated as a full closed-bilinear product and does not behave like a clean minus of `psibar * psi * chibar * chi`.
- Follow-up implementation:
  - `src/model/lowering.py` now infers local `closed_dirac_bilinears` and requires every fermion slot in a local monomial to be covered exactly once by those recognized ordered bilinears.
  - Unsupported local fermion orderings now fail closed with `ValueError("Unsupported fermion ordering in local monomial ...")`.
  - The previously suspicious reversed-order case `psi * psibar * chibar * chi` now rejects during lowering instead of producing a misleading vertex.
  - Partially recognized multi-fermion chains such as `psibar * psi * chi * chibar` also reject during lowering.
- Tests added:
  - `test_distinct_species_closed_bilinear_product_is_stable_and_nonzero`
    - Passes.
    - Distinct-species scalar bilinear products are deterministic, nonzero, and lower with `closed_dirac_bilinears=((0, 1), (2, 3))`.
  - `test_distinct_species_closed_bilinear_order_is_bosonic`
    - Passes.
    - Swapping `(psibar psi)` with `(chibar chi)` leaves the extracted vertex unchanged.
  - `test_reversed_fields_inside_bilinear_are_rejected`
    - Passes.
    - Converts the previous `xfail` into a normal rejection test for reversed `psi * psibar` ordering.
  - `test_partially_recognized_multi_fermion_chain_is_rejected`
    - Passes.
    - Confirms that partially covered multi-fermion local monomials fail closed.
  - `test_identical_closed_bilinear_square_is_deterministic_and_nonzero`
    - Passes.
    - Complements the existing exact same-species square test with a deterministic/nonzero diagnostic.
  - `test_distinct_species_vector_bilinear_order_is_stable`
    - Passes.
    - Distinct-species vector bilinears keep stable Lorentz contraction and fermion ordering under bilinear reordering.
- Status checklist:
  - [x] understood
  - [x] test written
  - [x] fix implemented
  - [x] validated

## 4. Important Design / Consistency Issues

### 4.1 `compiler/gauge.py` carries too many responsibilities

- Description:
  One file handles validation, metadata resolution, covariant-derivative expansion, matter currents, scalar contacts, pure-gauge terms, gauge fixing, ghosts, and compatibility paths.
- Why it matters:
  Correctness changes become harder to isolate and review. Sign or convention edits can accidentally affect unrelated sectors.
- Exact file(s)/function(s) involved:
  - `src/compiler/gauge.py`
- Concrete example of failure:
  - A future change to scalar-contact conventions can easily touch the same module region that also controls ghost or Yang-Mills compilation.
- Proposed fix strategy:
  - Split by responsibility into small internal modules without changing public behavior.
  - Keep current function names re-exported if needed.
- First extraction step completed:
  - [x] Created `src/compiler/covariant_core.py` as a small internal module for covariant-core compilation helpers.
  - [x] Moved the covariant-core implementation and tightly coupled free-bilinear builders out of `src/compiler/gauge.py`.
  - [x] Kept thin wrapper functions in `src/compiler/gauge.py` for `_compile_covariant_core(...)`, `_compile_declared_covariant_core(...)`, and `_compile_legacy_covariant_core(...)` so the old internal call sites still work.
  - [x] Left Yang-Mills, ghosts, gauge fixing, generic covariant expansion, and SSB code untouched.
- Module contents moved:
  - `src/compiler/covariant_core.py`
  - moved `_compile_covariant_core(...)`
  - moved `_compile_declared_covariant_core(...)`
  - moved `_compile_legacy_covariant_core(...)`
  - moved the related partial-term builders and full-operator assembly helper
- Additional extraction step completed:
  - [x] Created `src/compiler/spectators.py` for spectator-label and spectator-decoration helpers.
  - [x] Moved `_spectator_identity_factor(...)`, `_materialize_spectator_occurrences(...)`, and `_decorate_interactions_with_spectators(...)` out of `src/compiler/gauge.py`.
  - [x] Moved the private spectator label builders used by those helpers into the same module.
  - [x] Simplified `src/compiler/covariant_core.py` so it imports spectator helpers directly instead of receiving them through wrapper injection.
- Behavior parity validation:
  - Existing parity checks around declarative-vs-legacy covariant compilation were preserved and still pass.
  - Focused checks covering declarative free bilinears, legacy gauge-only behavior, and `with_compiled_covariant_terms(...)` parity were run after extraction.
  - Focused spectator-decorated declarative `CovD(...)` tests also pass after the second extraction.
- Status checklist:
  - [x] understood
  - [x] refactor plan written
  - [ ] split implemented
  - [ ] behavior parity validated

### 4.2 `vertex_engine.py` mixes core contraction with output-policy and cleanup logic

- Description:
  The same module owns permutation sums, fermion-sign logic, delta replacement, external stripping, and presentation-oriented simplification.
- Why it matters:
  Core semantics and display policy are harder to reason about separately.
- Exact file(s)/function(s) involved:
  - `src/symbolic/vertex_engine.py`
  - `contract_to_full_expression(...)`
  - `vertex_factor(...)`
  - `simplify_vertex(...)`
- Concrete example of failure:
  - A change intended only for more readable vector vertices can accidentally alter extraction semantics if it lands near the core pipeline.
- Proposed fix strategy:
  - Separate core contraction from output post-processing.
  - Keep `vertex_factor(...)` as the public façade, but delegate to smaller internal layers.
- Status checklist:
  - [ ] understood
  - [ ] refactor plan written
  - [ ] split implemented
  - [ ] behavior parity validated

### 4.3 The lowering layer still relies on central dispatch and large conditional chains

- Description:
  Declared-term analysis is concentrated in long dispatcher paths instead of small analyzers per semantic family.
- Why it matters:
  It is easy to add new syntax in a way that duplicates logic or creates inconsistent acceptance rules.
- Exact file(s)/function(s) involved:
  - `src/model/lowering.py`
  - `_analyze_declared_source_term(...)`
  - `_validate_declared_monomial(...)`
  - `_source_term_*`
- Concrete example of failure:
  - A new declarative operator family could be accepted by one path and rejected by another if analyzer logic drifts.
- Proposed fix strategy:
  - Replace central chains with small internal analyzer registries.
  - Preserve one normalized analyzed-term result type.
- Status checklist:
  - [ ] understood
  - [ ] refactor plan written
  - [ ] registry cleanup implemented
  - [ ] behavior parity validated

### 4.4 Documentation and note layout are ahead of README alignment

- Description:
  The live source uses the split package layout, but the top-level README still describes older flat-file structure.
- Why it matters:
  New development work can start from the wrong mental model.
- Exact file(s)/function(s) involved:
  - `README.md`
- Concrete example of failure:
  - A contributor reading the README looks for `src/model_symbolica.py` and `src/model.py`, but the actual code is now in `src/model/*`, `src/compiler/*`, and `src/symbolic/*`.
- Proposed fix strategy:
  - Update README structure and status sections to match the live package layout and current API.
- Status checklist:
  - [ ] understood
  - [ ] stale sections identified
  - [ ] README updated
  - [ ] cross-checked against source tree

### 4.5 Example scripts are still too close to the validation surface

- Description:
  The codebase has good tests, but `examples/examples.py` and `examples/examples_lagrangian.py` are still very large and carry some integration-check burden.
- Why it matters:
  Example-driven assertions are harder to maintain and easier to ignore than focused tests.
- Exact file(s)/function(s) involved:
  - `examples/examples.py`
  - `examples/examples_lagrangian.py`
- Concrete example of failure:
  - A subtle regression can remain “covered” only because an example script still prints or checks it in an ad hoc way.
- Proposed fix strategy:
  - Move remaining behavioral assertions into focused `tests/`.
  - Keep `examples/` primarily demonstrative.
- Status checklist:
  - [ ] understood
  - [ ] example-only assertions identified
  - [ ] tests extracted
  - [ ] examples simplified

## 5. Symbolica / Spenso Usage Improvements

- [ ] Reduce dependence on `to_canonical_string()` and raw `str(...)` for semantic equality and keying.
  - Current hotspots:
    - `src/lagrangian/lowering.py`
    - `src/model/interactions.py`
    - `src/symbolic/vertex_engine.py`
  - Concrete action:
    - Centralize expression/species/index comparison helpers and prefer structural comparison where Symbolica supports it.

- [ ] Apply gamma simplification in the standard high-level vertex cleanup path.
  - Current hotspot:
    - `src/symbolic/vertex_engine.py`
    - `simplify_vertex(...)`
  - Concrete action:
    - Add an optional gamma-simplification pass using `simplify_gamma_chain(...)`.
    - Keep it configurable if performance becomes an issue.

- [ ] Replace handwritten Lorentz cleanup where possible with more library-native tensor handling.
  - Current hotspot:
    - `src/symbolic/tensor_canonicalization.py`
    - `contract_spenso_lorentz_metrics(...)`
  - Concrete action:
    - Audit which contractions can be delegated to existing Spenso/Symbolica tensor machinery.
    - Keep the handwritten pass only for the residual patterns that truly need it.

- [ ] Extend tensor canonicalization usage beyond vector-only cleanup.
  - Current hotspot:
    - `src/symbolic/vertex_engine.py`
    - `_canonicalize_vector_vertex(...)`
  - Concrete action:
    - Generalize canonicalization helpers to mixed tensor structures when index-group metadata is available.

- [ ] Make dummy-index handling more provenance-aware in lowering rather than repairing it later in cleanup.
  - Current hotspots:
    - `src/model/lowering.py`
    - `src/symbolic/tensor_canonicalization.py`
  - Concrete action:
    - Preserve more slot-level intent during lowering so canonicalization becomes normalization, not rescue logic.

## 6. Physics Validation Gaps

- [ ] Hermiticity checks
  - Verify that declared Lagrangians are Hermitian when the model claims a physical action.
  - At minimum, detect clearly unmatched complex-conjugate terms.

- [ ] Kinetic normalization checks
  - Verify canonical normalization of scalar, fermion, and vector kinetic terms.
  - Catch wrong overall factors before vertices are trusted.

- [ ] Mass diagonalization checks
  - Detect non-diagonal or partially diagonal mass structures.
  - Separate “allowed but not yet diagonalized” from “accidentally malformed”.

- [ ] Gauge consistency checks
  - Verify that gauge-fixing, ghost, and gauge-sector declarations match the declared gauge structure.
  - Catch missing ghost fields, inconsistent adjoint slots, or bad representation matches earlier.

- [ ] Mass-spectrum consistency checks
  - Cross-check declared masses against mass terms generated by the compiled Lagrangian where applicable.
  - Important for SSB and future parameter evaluation.

- [ ] Vertex-selection and filtering support
  - Add a cleaner workflow for enumerating and selecting vertices by arity, sector, and field content.
  - This is standard daily workflow in FeynRules-style usage.

## 7. API / UX Improvements

- [ ] Expose `include_delta` and `strip_externals` on the high-level `feynman_rule(...)` API.
  - Example:
    - `L.feynman_rule(psi.bar, psi, A, include_delta=False, strip_externals=True)`

- [ ] Add optional high-level gamma simplification.
  - Example:
    - `L.feynman_rule(psi.bar, psi, A, simplify=True, simplify_gamma=True)`

- [ ] Improve “no matching interaction” errors.
  - Current issue:
    - `No matching interaction terms for: ...` is often too generic.
  - Concrete improvement:
    - Show available matching signatures from the compiled Lagrangian.

- [ ] Make the `Lagrangian(...)` vs `Model(..., lagrangian_decl=...)` boundary more obvious.
  - Current issue:
    - The distinction is learned by hitting runtime errors.
  - Concrete improvement:
    - Tighten docstrings and top-level docs.
    - Consider helper wording that explicitly says “local-only” vs “metadata-dependent”.

- [ ] Keep generated internal names readable but less noisy.
  - Current issue:
    - `mu1_int`, `canon_dummy_*`, and similar names are acceptable for debugging but not ideal for daily reading.
  - Concrete improvement:
    - Standardize internal naming policy and keep it deterministic.

- [ ] Clarify that `T(...)` and `StructureConstant(...)` in the local DSL are not yet fully generic group objects.
  - Current issue:
    - A physics user can naturally read them as universal group syntax.
  - Concrete improvement:
    - Document current scope explicitly and avoid overpromising in examples.

## 8. Long-Term Features

- [ ] Generalize SSB beyond the current electroweak-focused explicit builder in `src/model/ssb.py`.
  - Keep the existing EW path intact while extracting reusable broken-phase building blocks.

- [ ] Turn `Parameter` into a real model component rather than a placeholder metadata object.
  - Current issue:
    - `Parameter` exists in `src/model/metadata.py` but is not yet integrated into evaluation, validation, or dependency handling.

- [ ] Improve EFT scaling beyond explicit permutation sums.
  - Current issue:
    - `vertex_engine.py` uses explicit permutation sums, which will become expensive for larger operators.
  - Concrete direction:
    - Preserve current semantics, then add smarter combinatorics for repeated-species structures.

- [ ] Widen supported multi-fermion tensor structures.
  - Keep the current fail-closed approach for unsupported sign semantics.
  - Extend only with explicit provenance and tests.

- [ ] Add broader model-validation workflows comparable in spirit to FeynRules.
  - Focus on pre-extraction diagnostics, not symbolic decoration.

## 9. First PR Plan

### Scope

- [x] Keep the PR strictly focused on fail-closed local tensor binding and malformed field-strength validation.
- [x] Do not change public conventions or do a broad refactor in the same PR.

### Exact files to touch

- [x] `src/model/lowering.py`
  - Rework local declared-index binding so free tensors attach to intended field slots or raise on ambiguity.

- [x] `src/lagrangian/lowering.py`
  - Tighten `lower_field_strength_monomial(...)` validation.

- [x] `tests/test_lagrangian_api.py`
  - Added both new regression tests here: ambiguous free-tensor attachment and repeated-index `FieldStrength(...)` rejection.

- [ ] `tests/test_covariant_compiler_matrix.py`
  - Not needed for this PR; existing declaration-path coverage lives in `tests/test_lagrangian_api.py`.

- [ ] `docs/notes/CONVENTIONS.md`
  - Not needed for this PR; no public convention changed.

### What to implement

- [x] Make `Metric(...)` / `StructureConstant(...)` lowering fail when endpoint attachment is ambiguous.
- [x] Preserve current successful lowering behavior for unambiguous local terms.
- [x] Reject `FieldStrength(group, mu, mu)` and similar malformed repeated-index inputs before gauge-kinetic compilation.
- [x] Keep error messages explicit enough that the user can fix the declaration without reading source.

### What tests to add

- [x] A test showing that `A * B * C * Metric(mu, rho)` now raises instead of silently misbinding.
- [x] A test showing that a valid unambiguous local tensor term still lowers and extracts correctly.
  - Covered by the existing `Metric(mu, nu) * phi.bar * phi * A * A` contact-term regression.
- [x] A test showing that `FieldStrength(U1, mu, mu) * FieldStrength(U1, mu, mu)` is rejected.
- [x] A parity test showing that ordinary valid gauge-kinetic declarations still compile unchanged.
  - Covered by the existing declarative-vs-legacy field-strength parity test.

### Validation steps

- [x] Run targeted tests for the new failure modes.
- [x] Run the full pytest suite.
- [ ] Check that no existing example-based expected vertices change except where malformed input is now rejected.
