# FeynRules-Style Strategy

Purpose: architecture principles, not timeline/status.

## Design principles

1. One engine, one declarative front end
- keep symbolic contraction logic isolated from declaration semantics.

2. Metadata-driven compilation
- gauge action, index slots, and representation behavior come from model metadata.

3. Test-first evolution
- any new declaration family must ship with parity tests before broad examples.

4. Explicit boundary between layers
- declaration/lowering emits normalized terms.
- compiler assembles physics terms.
- vertex extraction consumes normalized terms.

## Current front end

The intended public direction is:

- `Model(..., lagrangian_decl=...)`
- `CovD(...)`
- `Gamma(...)`
- `FieldStrength(...)`
- `GaugeFixing(...)`
- `GhostLagrangian(...)`

These are typed source declarations. They should lower onto the existing
backend rather than create a second symbolic execution path.

## Practical architecture in this repo

- declaration and metadata: `src/model/*`
- declarative helpers: `src/lagrangian/*`
- compilation: `src/compiler/gauge.py`
- symbolic/tensor utilities: `src/symbolic/*`
- behavior locks: `tests/*`

## Layer responsibilities

The intended split is:

1. symbolic engine
   - contraction permutations
   - fermion signs
   - derivative-to-momentum replacement
   - open-index remapping
2. declaration/lowering
   - typed model declarations
   - canonical source-term analysis
   - lowering to normalized backend terms
3. compiler
   - covariant expansion
   - gauge-structure assembly
   - normalization/convention application

This note exists to keep physics structure out of ad hoc example code and to
keep model-specific semantics out of the generic symbolic engine.

## Near-term strategy

1. stabilize the recent gauge/lowering refactors with focused tests
2. finish API ergonomics for whole-Lagrangian extraction
3. enter BFM split only after step 1 and 2 are stable

## Architectural constraint

If a new feature requires parallel handwritten logic in examples and compiler
code, or requires adding field-specific branching into the symbolic engine, the
layering has regressed.
