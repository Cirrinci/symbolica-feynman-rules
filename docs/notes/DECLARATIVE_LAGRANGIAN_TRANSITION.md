# Declarative Lagrangian Transition

## Goal

Move the model-layer API from separate declaration slots

- `interactions`
- `covariant_terms`
- `gauge_kinetic_terms`
- `gauge_fixing_terms`
- `ghost_terms`

to one primary declaration entry point:

```python
model = Model(
    gauge_groups=(...),
    fields=(...),
    lagrangian_decl=(
        I * Psi.bar * Gamma(mu) * CovD(Psi, mu)
        + CovD(Phi.bar, mu) * CovD(Phi, mu)
        - S(1) / 4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)
        + GaugeFixingTerm(gauge_group=SU3C, xi=xiQCD)
        + GhostTerm(gauge_group=SU3C)
    ),
)
```

The new API should cover all physics structures that are already implemented in
the repository, while preserving the existing compiler back-end.

## Strategy

### 1. Add a high-level declaration layer

Introduce:

- `DeclaredLagrangian`
- `CovD(field, mu)`
- `Gamma(mu)`
- `FieldStrength(group, mu, nu)`

These objects are only a front-end DSL. They do not replace the current
compiler internals.

Status: implemented

### 2. Lower canonical declarative expressions to the existing term classes

Supported canonical forms:

- `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)` -> `DiracKineticTerm`
- `CovD(Phi.bar, mu) * CovD(Phi, mu)` -> `ComplexScalarKineticTerm`
- `-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu)` -> `GaugeKineticTerm`
- direct `GaugeFixingTerm(...)`
- direct `GhostTerm(...)`
- direct `InteractionTerm(...)`

This keeps the existing gauge compiler as the single lowering back-end.

Status: implemented

### 3. Make `Model` consume one unified declaration

Add `Model.lagrangian_decl` and let the model expose combined views:

- `all_interactions()`
- `all_covariant_terms()`
- `all_gauge_kinetic_terms()`
- `all_gauge_fixing_terms()`
- `all_ghost_terms()`

The compiler and `Model.lagrangian()` should use these combined views so that
legacy declarations and the new API can coexist during the transition.

Status: implemented

### 3b. Preserve the source declaration for demos and introspection

`DeclaredLagrangian` should keep the original declarative source terms
(`CovD(...)`, `FieldStrength(...)`, direct helper classes, manual
`InteractionTerm`s) and expose a cached lowered view for the existing compiler.

This allows the examples to print the declaration the user actually wrote,
rather than only the compiled `InteractionTerm` expansion.

Status: implemented

### 4. Preserve idempotent precompilation

`with_compiled_covariant_terms(model)` must:

- compile only the physical high-level pieces
- keep manual declared `InteractionTerm`s intact
- avoid duplication when `.lagrangian()` is called afterwards

Status: implemented

### 5. Migrate canonical example models

Move the repo’s example covariant/gauge/gauge-fixing/ghost models to
`lagrangian_decl=` so the new API is the primary path in the repository, not a
secondary compatibility layer.

Status: implemented for `src/examples.py`

### 6. Expand parity tests

Required coverage:

- declarative fermion `CovD` parity
- declarative scalar `CovD` parity
- declarative field-strength parity
- mixed manual `InteractionTerm` + declarative pieces
- precompiled-model idempotency with `lagrangian_decl`

Status: implemented

### 7. Follow-up cleanup

Remaining cleanup work:

- add more user-facing demos that show the new declaration syntax directly
- decide whether `GaugeFixingTerm` / `GhostTerm` also need fully symbolic DSL
  wrappers or whether direct helper classes are enough
- remove the remaining thin legacy-slot compatibility tests after one deprecation
  cycle
- document the recommended import surface for declarative model building
