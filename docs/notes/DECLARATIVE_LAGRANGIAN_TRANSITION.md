# Declarative Lagrangian Transition

Purpose: track migration from legacy split declaration slots to unified `lagrangian_decl`.

## Target API

Use one primary model entry:

- `Model(..., lagrangian_decl=...)`

with canonical declarative builders:

- `CovD(field, mu)`
- `Gamma(mu)`
- `FieldStrength(group, mu, nu)`
- `GaugeFixing(group, xi=...)`
- `GhostLagrangian(group)`

## Canonical lowering forms

These front-end forms lower to the existing backend term classes:

- `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)` -> `DiracKineticTerm`
- `CovD(Phi.bar, mu) * CovD(Phi, mu)` -> `ComplexScalarKineticTerm`
- `-(1 / 4) * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu)` -> `GaugeKineticTerm`
- `GaugeFixing(G, xi=...)` -> `GaugeFixingTerm`
- `GhostLagrangian(G)` -> `GhostTerm`
- direct `InteractionTerm(...)` stays a direct lowered interaction

## Current state

Implemented:

- declarative front-end objects
- lowering to existing backend term classes
- coexistence with legacy slots during transition
- preservation of the source declarative objects for examples and introspection
- examples using declarative flow

Open points:

1. widen declarative parity tests for recently refactored covariant assembly paths
2. keep diagnostics explicit when declarations are incomplete/ambiguous
3. complete migration of residual example-only expectations into tests

## Transition rule

During the transition, legacy declaration slots may still coexist with `lagrangian_decl`, but new examples and new API work should prefer the unified declarative entry point.

## Non-goal

Do not replace the backend with a second symbolic engine. The declarative layer is a typed front-end over the existing lowering/compiler path.
