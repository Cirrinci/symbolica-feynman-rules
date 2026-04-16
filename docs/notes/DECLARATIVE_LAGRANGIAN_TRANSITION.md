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
        + GaugeFixing(SU3C, xi=xiQCD)
        + GhostLagrangian(SU3C)
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
- `GaugeFixing(group, xi=...)`
- `GhostLagrangian(group)`

These objects are only a front-end DSL. They do not replace the current
compiler internals.

For gauge-fixing and ghosts, the stable front-end should stay typed and
canonical rather than trying to parse arbitrary `DC[...]` / `Div[...]`
expressions first. If richer symbolic sugar is added later, it should still
lower onto this same back-end.

Status: implemented

### 2. Lower canonical declarative expressions to the existing term classes

Supported canonical forms:

- `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)` -> `DiracKineticTerm`
- `CovD(Phi.bar, mu) * CovD(Phi, mu)` -> `ComplexScalarKineticTerm`
- `-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu)` -> `GaugeKineticTerm`
- `GaugeFixing(G, xi=...)` -> `GaugeFixingTerm`
- `GhostLagrangian(G)` -> `GhostTerm`
- direct `InteractionTerm(...)`

This keeps the existing gauge compiler as the single lowering back-end.

Status: implemented

### 2b. Current recommended declaration patterns

Use the canonical declarative objects as the public surface:

- fermion kinetic term:
  - `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)`
- complex scalar kinetic term:
  - `CovD(Phi.bar, mu) * CovD(Phi, mu)`
- gauge kinetic term:
  - `-(1/4) * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu)`
- ordinary gauge fixing:
  - `GaugeFixing(G, xi=xiG)`
- ordinary ghosts:
  - `GhostLagrangian(G)`

Generic higher-derivative monomials are still declared explicitly with
`InteractionTerm(...)` and `DerivativeAction(...)`.

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
(`CovD(...)`, `FieldStrength(...)`, `GaugeFixing(...)`,
`GhostLagrangian(...)`, direct helper classes, manual
`InteractionTerm`s) and expose a cached lowered view for the existing compiler.

This allows the examples to print the declaration the user actually wrote,
rather than only the compiled `InteractionTerm` expansion.

Status: implemented

### 3c. Use one normalized source-term analysis path

The declarative pipeline should not classify the same source term separately in
validation, model views, compiler entry points, and precompilation retention.

Instead, one normalized source-term analysis step should decide whether a term
is:

- a local interaction monomial
- a covariant matter term, optionally with spectators
- a gauge-kinetic term
- a gauge-fixing term
- a ghost term

All higher-level entry points should consume that same normalized result.

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
- decide whether the canonical wrappers should later grow into fully symbolic
  `Div(...)` / ghost-covariant-derivative expressions, or remain the stable
  user-facing front-end
- remove the remaining thin legacy-slot compatibility tests after one deprecation
  cycle
- document the recommended import surface for declarative model building
