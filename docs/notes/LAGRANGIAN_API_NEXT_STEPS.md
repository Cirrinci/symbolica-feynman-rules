## Lagrangian API Next Steps

Purpose: user-facing API backlog only.

## Priority items

### 1. Whole-Lagrangian extraction

Add `feynman_rules(...)` on `Lagrangian` in addition to single-vertex `feynman_rule(...)`.

Why:

- current `feynman_rule(fields...)` is useful but weaker than the FeynRules-style whole-Lagrangian workflow
- users should not need to know every field tuple up front just to inspect the interaction content

Target examples:

- `L.feynman_rules()`
- `L.feynman_rules(arity=3)`
- `L.feynman_rules(select=[(Phi.bar, Phi, A)])`

Implementation shape:

- group `Lagrangian.terms` by external field content
- infer one canonical field tuple per group
- reuse the existing single-vertex extraction path internally
- return a structured mapping or list from field tuple to vertex expression

### 2. Single-term ergonomics

Allow direct term usage:

- `term.feynman_rule(...)`
- `(term1 + term2).feynman_rule(...)`

without manual wrapping in `Lagrangian(terms=(...))`.

Implementation shape:

- add a thin `feynman_rule(...)` wrapper on `InteractionTerm`
- make `term1 + term2` produce a Lagrangian-like object consistently
- keep the real extraction logic in one place so term and Lagrangian paths cannot drift

### 3. Gauge-fixing syntax sugar

Keep typed backend lowering, but allow readable front-end sugar such as divergence-based helpers that lower to `GaugeFixingTerm`.

Candidate surface:

- `-(1 / (2 * xi)) * Div(A, mu) * Div(A, mu)`

Constraint:

- accept only a small canonical family of divergence-squared patterns
- lower them to the same structured backend term used by `GaugeFixing(...)`

### 4. Keep ghost API safe

Retain `GhostLagrangian(group)` as canonical API. Add sugar only if it lowers to the same structured backend.

Reason:

- ghost sectors are derived objects and easy to misdeclare if exposed as unrestricted free-form syntax

### 5. Analyzer registry cleanup

Replace large central dispatch chains with small registry-based analyzers per semantic family, while preserving one normalized analyzed-term result type.

Minimal shape:

```python
SOURCE_TERM_ANALYZERS = [...]
DECLARED_MONOMIAL_ANALYZERS = [...]
```

Constraint:

- this is an internal registry cleanup, not a plugin system
- keep one normalized analyzed-term result type and make analyzers independently testable

## Recommended order

1. add `feynman_rules(...)`
2. make single terms directly usable
3. add gauge-fixing sugar
4. keep ghost sugar constrained to structured lowering
5. replace central dispatch chains with small analyzer registries
