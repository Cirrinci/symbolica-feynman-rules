# Code Review Roadmap

This note now keeps only outstanding work and the context needed to finish it.
Items that were marked complete and still match the live codebase were removed
after review against the current `main` branch and test suite.

## 1. Current Context

The architecture is still directionally sound:

- `src/model/` owns declarations, metadata, lowering, and validation entry points.
- `src/compiler/` owns convention-fixed gauge/covariant compilation.
- `src/symbolic/` owns contraction and vertex post-processing.

The immediate risks are no longer the previously documented fail-open
correctness bugs; those fixes are in place and covered by tests. The remaining
work is mostly:

- cleanup of large dispatch-heavy modules,
- removal of stale documentation and residual example-side validation code,
- reduction of string-based symbolic comparisons,
- extension of the validation/reporting layer beyond the first conservative pass.

## 2. Open Design / Maintenance Work

### 2.1 `src/model/lowering.py` still relies on central dispatch and long conditional chains

- Why it still matters:
  - declared-term analysis is concentrated in `_analyze_declared_source_term(...)`,
    `_validate_declared_monomial(...)`, and related `_source_term_*` helpers;
  - adding a new declarative operator family still risks duplicated acceptance
    logic and drift between analysis paths.
- Remaining work:
  - replace central chains with smaller analyzers or an internal registry,
  - preserve one normalized analyzed-term result type,
  - keep behavior parity with the current declarative syntax.

### 2.2 README alignment is still incomplete

- Review finding:
  - `README.md` still says kinetic normalization and mass-diagonalization are
    "under development" even though the first validation/reporting passes are
    already implemented;
  - it also still says some integration-style assertions live in `examples/`,
    which is only partially true now that the CLI entry points no longer run
    embedded test suites.
- Remaining work:
  - update the status section so it reflects the current validation surface:
    first-pass kinetic normalization, mass-structure diagnostics, gauge
    consistency diagnostics, and vertex reporting/filtering exist today;
  - keep only the genuinely open validation gaps as "under development".

### 2.3 Example scripts still carry residual validation code

- Current state:
  - `examples/examples.py` and `examples/examples_lagrangian.py` no longer run
    embedded test suites from their CLI entry points;
  - the legacy `_run_*_tests()` helpers are still present as dormant internal
    code.
- Remaining work:
  - delete the dormant `_run_*_tests()` helpers once all remaining useful
    assertions have clear homes under `tests/`,
  - keep `examples/` demo-oriented rather than validation-oriented.

## 3. Symbolica / Spenso Follow-Ups

### 3.1 Reduce dependence on `to_canonical_string()` / raw `str(...)`

- Current hotspots:
  - `src/lagrangian/lowering.py`
  - `src/model/interactions.py`
  - `src/symbolic/vertex_engine.py`
- Why it matters:
  - string-level equality is still fragile for semantic comparison, keying, and
    diagnostics;
  - internal dummy-label naming can create false mismatches.
- Remaining work:
  - centralize expression/species/index comparison helpers,
  - prefer structural comparison where Symbolica supports it,
  - stop duplicating ad hoc equality logic across modules.

### 3.2 Replace handwritten Lorentz cleanup where library-native tensor handling can do it

- Current hotspot:
  - `src/symbolic/tensor_canonicalization.py`
  - `contract_spenso_lorentz_metrics(...)`
- Remaining work:
  - audit which contractions can be delegated to Spenso/Symbolica directly,
  - keep handwritten cleanup only for residual patterns that truly need it.

### 3.3 Extend canonicalization beyond vector-only cleanup

- Current hotspot:
  - `src/symbolic/vertex_postprocessing.py`
  - `canonicalize_vector_vertex(...)`
- Remaining work:
  - generalize canonicalization helpers to mixed tensor structures when
    index-group metadata is available.

### 3.4 Preserve more dummy-index provenance during lowering

- Current hotspots:
  - `src/model/lowering.py`
  - `src/symbolic/tensor_canonicalization.py`
- Remaining work:
  - keep more slot-level intent during lowering so later canonicalization is
    normalization rather than repair.

## 4. Physics Validation Gaps

Existing context:

- `Model.validate()` already covers a conservative first pass for:
  - declaration-level kinetic normalization and duplicate kinetic declarations,
  - gauge-sector consistency checks,
  - some representation-resolution diagnostics.
- `CompiledLagrangian.validate()` already covers:
  - mass-structure mixing diagnostics,
  - grouped vertex reporting/filtering utilities live separately on
    `CompiledLagrangian`.

The remaining work is to extend that validation surface without guessing.

### 4.1 Hermiticity

- Goal:
  - detect obviously non-Hermitian declared models before users trust extracted
    vertices.
- Why this is still open:
  - coupling reality is not reliably knowable from the current symbolic layer,
  - mixed-species Yukawa partners are not captured by a simple "flip the scalar"
    heuristic,
  - broken-phase SSB output is not expressible as a trivial mirror pairing,
  - coupling comparison still suffers from string-level fragility.
- Remaining work:
  - either thread real/complex parameter metadata deeply enough that coupling
    reality is explicit,
  - or add a Hermitian-conjugate canonicalizer that handles same-species and
    cross-species fermion bilinears together with scalar partners.

### 4.2 Kinetic normalization: remaining scope beyond the implemented first pass

- Implemented already:
  - declaration-level checks for `ComplexScalarKineticTerm`,
    `DiracKineticTerm`, and `GaugeKineticTerm` coefficients and duplicates.
- Still missing:
  - arbitrary local kinetic expressions such as
    `PartialD(phi, mu) * PartialD(phi, mu)`,
  - real-scalar `1/2` normalization checks outside dedicated declarations,
  - compiled two-point pattern matching against convention-locked templates,
  - broader EFT-style operator classification.
- Context from review:
  - compiled kinetic terms still carry unstable internal labels and can split
    across multiple compiled `InteractionTerm`s, so safe compiled-template
    matching needs a deterministic canonicalization or a compiler-exposed
    template object first.

### 4.3 Gauge consistency: broaden representation/coupling diagnostics

- Implemented already:
  - undeclared gauge-group references,
  - abelian ghost-sector rejection,
  - missing structure constants / ghost fields,
  - explicit non-abelian representation-resolution failures for supported
    covariant declarations.
- Still missing:
  - broader automatic-coupling / representation-slot diagnostics for every
    inferred gauge action path, not just the currently explicit checked forms.

### 4.4 Mass-spectrum consistency

- Goal:
  - compare declared field masses against generated compiled mass terms where
    the model encodes both.
- Why this is still open:
  - `Field.mass` metadata is not yet integrated tightly enough with compiled
    couplings,
  - there is no stable mass-coefficient extractor,
  - SSB-derived masses and "not yet derivable" cases are not yet distinguishable
    from genuinely missing compiled mass terms.
- Remaining work:
  - promote parameter/coupling tracking far enough that declared masses can be
    compared safely,
  - or add a deterministic compiler-side extractor the validation layer can use.

## 5. API / UX Follow-Up

### 5.1 Keep generated internal names readable but less noisy

- Current issue:
  - names such as `mu1_int` and `canon_dummy_*` are acceptable for debugging
    but still noisy in daily output;
  - unstripped high-level output uses generic `U(...)` placeholders for
    non-fermion external legs, which is readable but physics-facingly vague.
- Remaining work:
  - standardize internal naming policy and keep it deterministic,
  - improve placeholder naming without changing semantics.

## 6. Long-Term Features

### 6.1 Generalize SSB beyond the current electroweak-focused builder

- Keep the existing electroweak path intact while extracting reusable
  broken-phase building blocks from `src/model/ssb.py`.

### 6.2 Finish turning `Parameter` into a fully integrated model component

- Implemented already:
  - structured parameter assumptions and metadata,
  - `Model.find_parameter(...)`,
  - `Model.parameter_assumptions(...)`,
  - direct use of `Parameter` objects in coefficient expressions.
- Still open:
  - dependency evaluation for internal parameters,
  - numerical substitution / parameter-card style workflows,
  - feeding parameter assumptions into hermiticity and mass-spectrum checks.

### 6.3 Improve EFT scaling beyond explicit permutation sums

- `src/symbolic/vertex_engine.py` still uses explicit permutation sums.
- Remaining work:
  - preserve current semantics,
  - add smarter combinatorics for larger repeated-species operators.

### 6.4 Widen supported multi-fermion tensor structures

- Keep the current fail-closed behavior for unsupported sign semantics.
- Extend only with explicit provenance and focused tests.

### 6.5 Add broader model-validation workflows in the FeynRules spirit

- Keep the focus on pre-extraction diagnostics and user-facing reports rather
  than silent symbolic repair.
