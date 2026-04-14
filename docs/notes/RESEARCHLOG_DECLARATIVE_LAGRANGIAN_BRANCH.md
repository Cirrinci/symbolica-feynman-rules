# Research Log: Declarative Lagrangian Branch

## Goal

Move the public model-building workflow from split physical declaration slots
to one FeynRules-style entry point:

```python
Model(..., lagrangian_decl=...)
```

The backend stays the same: Symbolica + Spenso still evaluate vertices through
the existing `InteractionTerm` / `vertex_factor(...)` pipeline.

## New Public API

- `lagrangian_decl=` is now the recommended way to declare models.
- Canonical declarative building blocks:
  - `CovD(field, mu)`
  - `Gamma(mu)`
  - `FieldStrength(group, mu, nu)`
  - `GaugeFixing(group, xi=...)`
  - `GhostLagrangian(group)`
- Manual `InteractionTerm(...)` monomials still compose with these objects in
  the same declaration.
- `model.lagrangian().feynman_rule(...)` is now the main extraction path.

## Canonical Declaration Patterns

- Dirac kinetic term:
  - `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)`
- Complex scalar kinetic term:
  - `CovD(Phi.bar, mu) * CovD(Phi, mu)`
- Gauge kinetic term:
  - `-(1/4) * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu)`
- Ordinary gauge fixing:
  - `GaugeFixing(G, xi=xiG)`
- Ordinary Faddeev-Popov ghosts:
  - `GhostLagrangian(G)`
- Generic higher partial-derivative operators remain explicit:
  - `InteractionTerm(...)` + `DerivativeAction(...)`

## Implementation

- Added a source-preserving `DeclaredLagrangian`.
  - `source_terms` keeps the user-written declaration.
  - cached `lowered_terms` feeds the existing compiler backend.
- Lowering maps canonical declarative forms to the current model-level classes:
  - `DiracKineticTerm`
  - `ComplexScalarKineticTerm`
  - `GaugeKineticTerm`
  - `GaugeFixingTerm`
  - `GhostTerm`
- This keeps one compiler backend instead of building a second symbolic engine.
- `with_compiled_covariant_terms(...)` was updated so precompiled models remain
  idempotent under `Model.lagrangian()`.

## API Hardening

- `Lagrangian.feynman_rule(...)` field matching was tightened to avoid
  same-symbol field collisions.
- `(Field, bool)` input is accepted in the Lagrangian API.
- `InteractionTerm + InteractionTerm` and mixed declarative composition now
  feed the same `DeclaredLagrangian` path.
- `GaugeFixing(...)` and `GhostLagrangian(...)` support scalar prefactors and
  preserve their source form in demos and notebooks.

## Compatibility Policy

- Legacy split slots are still supported:
  - `covariant_terms`
  - `gauge_kinetic_terms`
  - `gauge_fixing_terms`
  - `ghost_terms`
- They now emit `DeprecationWarning` and are kept only as a compatibility path.

## Demos, Tests, Docs

- `src/examples.py` canonical models now use `lagrangian_decl=...`.
- `src/examples_lagrangian.py` prints the source declaration instead of only the
  lowered compiled terms.
- The workflow notebook now shows the declarative front-end directly.
- Docs were updated to present the unified declarative API as the main path.

## Validation

- `./.venv/bin/python -m pytest -q` -> `89 passed`
- `./.venv/bin/python src/examples.py --suite all --no-demo` -> green
- `./.venv/bin/python src/examples_lagrangian.py --suite all --no-demo` -> green

## Bottom Line

The code now has a real declarative front-end with a stable lowering boundary.
That is the correct architecture for growing toward a more FeynRules-like user
experience without abandoning the current Symbolica + Spenso backend.
