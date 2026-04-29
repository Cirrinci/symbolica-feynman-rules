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
- Final extraction step completed:
  - [x] Created `src/compiler/matter_actions.py` for bilinear gauge-action helpers used by matter currents and scalar contacts.
  - [x] Moved `_build_bilinear_gauge_action_data(...)`, `_compile_scalar_current_from_piece(...)`, `_build_fermion_current_interaction(...)`, `_default_scalar_contact_internal_label(...)`, `_build_scalar_contact_action_data(...)`, and `_compile_scalar_contact_terms(...)` out of `src/compiler/gauge.py`.
  - [x] Moved the private `_BilinearGaugeActionData` and `_ScalarContactActionData` dataclasses into the same module.
  - [x] Kept `_GaugeAction` in `src/compiler/gauge.py` because it is still shared with generic declared `CovD(...)` expansion and broader covariant metadata logic.
- Behavior parity validation:
  - Existing parity checks around declarative-vs-legacy covariant compilation were preserved and still pass.
  - Focused checks covering declarative free bilinears, legacy gauge-only behavior, and `with_compiled_covariant_terms(...)` parity were run after extraction.
  - Focused spectator-decorated declarative `CovD(...)` tests also pass after the second extraction.
  - Focused matter-current / scalar-contact parity checks now also pass after the final extraction:
    - `tests/test_covariant_bislot_sum.py`
    - `tests/test_covariant_mixed_scalar.py`
    - `tests/test_covariant_compiler_matrix.py`
    - legacy gauge-only kinetic diagnostics in `tests/test_lagrangian_api.py`
  - Full suite still passes after the three-step split.
- Current responsibility split:
  - `src/compiler/covariant_core.py`: full-operator vs gauge-only covariant-core policy
  - `src/compiler/spectators.py`: spectator labels and spectator decoration
  - `src/compiler/matter_actions.py`: matter-current and scalar-contact action builders
  - `src/compiler/gauge.py`: public compiler entry points plus remaining generic gauge-sector logic
- Status:
  - `4.1` is now substantially complete as an incremental split of the highest-density helper clusters out of `src/compiler/gauge.py`, without touching Yang-Mills, ghosts, gauge fixing, or SSB.
- Status checklist:
  - [x] understood
  - [x] refactor plan written
  - [x] split implemented
  - [x] behavior parity validated

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
- First extraction step completed:
  - [x] Created `src/symbolic/vertex_postprocessing.py` for post-contraction cleanup and output-policy helpers.
  - [x] Moved delta replacement / output-policy helpers out of `src/symbolic/vertex_engine.py`.
  - [x] Moved external wavefunction stripping helpers out of `src/symbolic/vertex_engine.py`.
  - [x] Moved simplification helpers and vector canonicalization helpers out of `src/symbolic/vertex_engine.py`.
- Module contents moved:
  - `src/symbolic/vertex_postprocessing.py`
  - moved `apply_vertex_output_policy(...)`
  - moved `replace_plane_wave_with_delta(...)`
  - moved `strip_external_wavefunctions(...)`
  - moved `simplify_deltas(...)`
  - moved `simplify_spinor_indices(...)`
  - moved `canonicalize_vector_vertex(...)`
  - moved `simplify_vertex(...)`
- Contraction core left untouched:
  - `contract_to_full_expression(...)` was not moved or edited semantically.
  - Fermion sign logic, permutation sums, derivative assignment, and contraction combinatorics remain in `src/symbolic/vertex_engine.py`.
- Behavior parity validation:
  - Added `tests/test_vertex_postprocessing.py`.
  - `test_vertex_factor_output_policy_matches_manual_postprocessing_for_quartic_scalar`
    - Passes.
    - Confirms `vertex_factor(...)` still matches the explicit post-contraction output-policy pipeline.
  - `test_simplify_vertex_matches_explicit_chain_for_four_gluon_vertex`
    - Passes.
    - Confirms `simplify_vertex(...)` still matches the explicit simplification/canonicalization chain on a nontrivial 4-vector vertex.
  - Focused regression checks in `tests/test_gauge_vertex_canonicalization.py` also still pass.
  - Full suite still passes after the extraction.
- Status checklist:
  - [x] understood
  - [x] refactor plan written
  - [x] split implemented
  - [x] behavior parity validated

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
- Completed alignment:
  - [x] Replaced the old flat-file repository map with the live split-package layout.
  - [x] Updated the documented engine entry points to `symbolic.vertex_engine(...)` and the current high-level workflow around `Lagrangian(...)` and `Model(..., lagrangian_decl=...)`.
  - [x] Corrected example and validation commands to use `examples/` and the live `src/symbolic/spenso_gamma_checks.py` path.
  - [x] Pointed day-to-day implementation priorities at `docs/notes/code_review_roadmap.md` instead of keeping a second stale priority list in the README.
- Status checklist:
  - [x] understood
  - [x] stale sections identified
  - [x] README updated
  - [x] cross-checked against source tree

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
- Completed extraction:
  - [x] Both example CLIs now run demos only; the `--skip-tests` flag remains as a no-op compatibility flag and no longer triggers embedded test runners.
  - [x] Added `tests/test_examples_regressions.py` for focused checks that had been living only in the example scripts.
  - [x] Moved reusable test-only symbols and low-level builders under `tests/support/` so the test suite no longer imports `examples.examples`.
  - [x] Avoided keeping a copied `examples/examples.py` mirror under `tests/support/`; the larger integration model cases now live in their owning test modules, while `tests/support/` stays limited to shared direct-API cases and generic constructors.
  - [x] Moved direct-API example regressions into tests:
    - unstripped Yukawa keeps external spinors,
    - underspecified multi-fermion direct API raises,
    - partial fermion leg-spinor labels raise,
    - role-based complex-scalar filtering and reversed-leg matching remain stable,
    - vector/scalar role mismatch stays filtered,
    - `FieldRole` compatibility semantics stay stable.
  - [x] Moved Lagrangian matcher regressions into tests:
    - tuple field syntax matches the complex-scalar bilinear,
    - same-symbol distinct scalar fields do not silently match,
    - same-symbol scalar/vector declarations do not silently match.
  - [x] Validated that `examples/examples.py --suite scalar` and `examples/examples_lagrangian.py --suite scalar` remain runnable as demos after the cleanup.
  - [x] The legacy `_run_*_tests()` helpers are no longer part of the example entry points; they remain as dormant internal code and can be deleted in a later cleanup without changing runtime behavior.
- Status checklist:
  - [x] understood
  - [x] example-only assertions identified
  - [x] tests extracted
  - [x] examples simplified

## 5. Symbolica / Spenso Usage Improvements

- [ ] Reduce dependence on `to_canonical_string()` and raw `str(...)` for semantic equality and keying.
  - Current hotspots:
    - `src/lagrangian/lowering.py`
    - `src/model/interactions.py`
    - `src/symbolic/vertex_engine.py`
  - Concrete action:
    - Centralize expression/species/index comparison helpers and prefer structural comparison where Symbolica supports it.

- [x] Apply gamma simplification in the standard high-level vertex cleanup path.
  - Current hotspot:
    - `src/symbolic/vertex_postprocessing.py`
    - `simplify_vertex(...)`
  - Concrete action:
    - Added optional `simplify_gamma: bool = False` to `simplify_vertex(...)`.
    - When enabled, the cleanup path applies `simplify_gamma_chain(...)` before the existing metric / canonicalization passes.
    - Default behavior remains unchanged when `simplify_gamma=False`.
  - Tests:
    - `tests/test_vertex_postprocessing.py::test_simplify_vertex_default_behavior_is_unchanged`
      - Passes.
      - Confirms the default `simplify_vertex(...)` chain is unchanged.
    - `tests/test_vertex_postprocessing.py::test_simplify_vertex_with_simplify_gamma_applies_gamma_chain`
      - Passes.
      - Confirms the opt-in flag changes the output and matches the explicit manual `simplify_gamma_chain(...)` pipeline.
    - `tests/test_vertex_postprocessing.py::test_lagrangian_feynman_rule_default_simplify_is_unchanged`
      - Passes.
      - Confirms `L.feynman_rule(..., simplify=True)` still uses the default non-gamma-simplifying behavior.

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

Goal:
- [ ] Add explicit pre-vertex validation passes so bad models fail before users trust extracted rules.
- [ ] Keep the first implementation narrow: diagnostics first, automatic repair later.

Recommended implementation shape:
- [ ] Add a small validation entry point that runs on declared or compiled Lagrangians without changing extraction semantics.
- [ ] Return structured diagnostics rather than only raising immediately, so the same machinery can support both tests and user-facing reports.
- [ ] Start with checks that are cheap, convention-stable, and easy to regression-test.

### 6.1 Hermiticity

- [ ] Check that the physical Lagrangian is Hermitian.
- Why it matters:
  - Missing or mismatched conjugate terms change the action itself, not just the presentation of a vertex.
  - A model can otherwise produce plausible vertices while still being physically inconsistent.
- Minimum viable check:
  - Detect obviously unmatched scalar, Yukawa, and current-like terms in declared local monomials.
  - Accept manifestly self-conjugate terms without requiring a separate partner.
- First implementation target:
  - Compare each declared source term against its conjugate-normalized form after lowering/canonicalization.
  - Report unmatched terms with the original source labels when possible.
- Good first tests:
  - `y * phi * psibar * psi` without its conjugate partner should fail.
  - `lam * phi.bar * phi` and manifestly real gauge bilinears should pass.

### 6.2 Kinetic Normalization

- [ ] Check canonical normalization of scalar, fermion, and vector kinetic terms.
- Why it matters:
  - Wrong factors in kinetic terms propagate into every vertex and can silently rescale the whole model.
  - This is one of the first checks physics users expect before trusting derived Feynman rules.
- Minimum viable check:
  - Recognize canonical two-point structures for:
    - scalar: `PartialD(phi.bar, mu) * PartialD(phi, mu)`
    - fermion: `I * psibar * Gamma(mu) * PartialD(psi, mu)`
    - vector: `-(1/4) F_{mu nu} F^{mu nu}`
  - Flag unexpected overall coefficients or duplicate canonical kinetic terms.
- First implementation target:
  - Run on compiled two-point interaction terms and compare extracted coefficients against convention-locked templates.
  - Treat gauge-only legacy kinetic declarations separately from full declarative `CovD(...)` operators.
- Good first tests:
  - canonical QED/QCD kinetic sectors pass;
  - scalar kinetic term with coefficient `2` fails;
  - doubled vector bilinear fails.

### 6.3 Mass Structure / Diagonalization

- [ ] Detect non-diagonal, partially diagonal, or malformed mass terms.
- Why it matters:
  - Off-diagonal mass terms are not automatically wrong, but they must be recognized explicitly rather than silently treated as a finished physical basis.
  - Mass mixing changes field interpretation and therefore the meaning of extracted vertices.
- Minimum viable check:
  - Identify scalar and fermion bilinears with no derivatives and classify them as:
    - diagonal canonical mass terms,
    - off-diagonal mixing terms,
    - suspicious malformed terms.
- First implementation target:
  - Provide diagnostics only: “diagonal”, “mixing present”, or “malformed”.
  - Do not attempt automatic diagonalization in the first pass.
- Good first tests:
  - `m * phi.bar * phi` passes as diagonal;
  - `m12 * phi1.bar * phi2` is reported as mixing;
  - incompatible bilinear structures are rejected.

### 6.4 Gauge Consistency

- [ ] Check that gauge-sector declarations are mutually consistent.
- Why it matters:
  - Gauge-fixing, ghost, and gauge-kinetic sectors are tightly coupled; inconsistency here can generate formally valid but physically meaningless vertices.
- Minimum viable check:
  - Verify that:
    - non-abelian ghost Lagrangians are only declared for groups with structure constants,
    - ghost fields exist when `GhostLagrangian(...)` is requested,
    - gauge-fixing terms refer to declared gauge groups,
    - field representations match the group slots they are supposed to couple to.
- First implementation target:
  - Reuse the existing compiler metadata checks, but surface them through a dedicated validation report instead of only as ad hoc compiler failures.
- Good first tests:
  - missing ghost field for a ghost Lagrangian fails;
  - abelian ghost sector request fails cleanly;
  - mismatched representation slots are reported before extraction.

### 6.5 Mass-Spectrum Consistency

- [ ] Cross-check declared particle masses against generated mass terms where the model encodes both.
- Why it matters:
  - This becomes essential once the parameter system and SSB layer carry more physical meaning.
  - It catches drift between metadata and the actual compiled Lagrangian.
- Minimum viable check:
  - When a field has declared mass metadata, compare it against the coefficient extracted from the corresponding compiled two-point mass term.
  - Allow “not yet derivable” as a distinct status instead of forcing a false failure.
- First implementation target:
  - Start with simple unbroken scalar and fermion sectors before attempting broken gauge sectors.
- Good first tests:
  - declared scalar mass matching the bilinear passes;
  - declared mass differing from compiled coefficient fails with a concrete diagnostic;
  - SSB-only unresolved masses are reported as unsupported, not incorrect.

### 6.6 Vertex Selection / Filtering

- [x] Add a physics-facing way to enumerate and filter vertices after validation.
- Why it matters:
  - Users need to inspect “all 3-point QCD vertices” or “all vertices containing ghosts” without manually probing signatures one by one.
  - This is standard workflow in FeynRules-style tooling and also makes validation outputs easier to consume.
- Implemented entry points:
  - [x] Added `CompiledLagrangian.vertex_signatures(...)` for deterministic grouped signature enumeration.
  - [x] Added `CompiledLagrangian.vertex_report(...)` plus structured `VertexSignature` / `VertexReport` diagnostics.
- Minimum viable feature:
  - [x] Enumerate compiled interaction signatures by field content and arity.
  - [x] Filter by exact field-content signature.
  - [x] Filter by “contains field(s)” with multiplicity awareness.
  - [ ] Filter by sector:
    - matter,
    - pure gauge,
    - gauge fixing,
    - ghosts.
- First implementation target:
  - [x] Build on `CompiledLagrangian` / compiled interaction-term metadata rather than re-deriving categories from final expressions.
- Tests:
  - [x] `tests/test_vertex_reporting.py::test_vertex_report_enumerates_scalar_signatures_deterministically`
    - Passes.
    - Confirms deterministic grouped scalar signature listing and report counts.
  - [x] `tests/test_vertex_reporting.py::test_vertex_signatures_filter_by_arity`
    - Passes.
    - Confirms arity filtering.
  - [x] `tests/test_vertex_reporting.py::test_vertex_signatures_filter_by_exact_qed_signature`
    - Passes.
    - Confirms exact-signature filtering on a compiled QED covariant model.
  - [x] `tests/test_vertex_reporting.py::test_vertex_signatures_contains_fields_is_multiplicity_aware_for_qcd`
    - Passes.
    - Confirms multiplicity-aware `contains_fields` filtering on a compiled QCD gauge model.
- Good first tests:
  - [ ] QCD model lists quark-gluon, 3-gluon, 4-gluon, ghost-gluon, and bilinear sectors distinctly by sector.
  - [x] Filtering by arity and field content returns stable deterministic results.

Suggested rollout order:
- [ ] Phase 1: hermiticity + kinetic normalization
- [ ] Phase 2: mass structure + gauge consistency
- [ ] Phase 3: mass-spectrum consistency + vertex selection/filtering

## 7. API / UX Improvements

- [ ] Expose `include_delta` and `strip_externals` on the high-level `feynman_rule(...)` API.
  - Example:
    - `L.feynman_rule(psi.bar, psi, A, include_delta=False, strip_externals=True)`

- [ ] Add optional high-level gamma simplification.
  - Example:
    - `L.feynman_rule(psi.bar, psi, A, simplify=True, simplify_gamma=True)`

- [x] Improve “no matching interaction” errors.
  - Current issue:
    - `No matching interaction terms for: ...` is often too generic.
  - Concrete improvement:
    - Show available matching signatures from the compiled Lagrangian.
  - Implemented behavior:
    - `CompiledLagrangian.feynman_rule(...)` now reuses `vertex_signatures()` when no match is found.
    - The error lists available grouped signatures in deterministic order.
    - Empty Lagrangians report `Available signatures: - (none)` instead of failing opaquely.
  - Example:
    - Before:
      - `No matching interaction terms for: Phi, Phi`
    - After:
      - `No matching interaction terms for: Phi, Phi.`
      - `Available signatures:`
      - `  - Phi, Phi, Phi, Phi`
  - Tests:
    - `tests/test_lagrangian_api.py::test_no_match_lists_available_higher_arity_signatures`
      - Passes.
      - Confirms a missing lower-arity request shows the available higher-arity signature.
    - `tests/test_lagrangian_api.py::test_no_match_lists_useful_available_signatures_for_unrelated_request`
      - Passes.
      - Confirms unrelated requests still get useful available signatures.
    - `tests/test_lagrangian_api.py::test_no_match_on_empty_lagrangian_is_clear`
      - Passes.
      - Confirms the empty-Lagrangian case stays clear and does not crash.

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
