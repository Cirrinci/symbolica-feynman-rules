## Lagrangian API Next Steps

This note records the highest-value follow-up work after the declarative
`lagrangian_decl` transition. The main theme is to remove the remaining places
where internal backend objects still leak into user-facing code.

### 1. Whole-Lagrangian vertex extraction

Problem:
- `Lagrangian.feynman_rule(...)` only extracts one specific vertex and requires
  the full external field list.
- This is weaker and less natural than the FeynRules-style workflow
  `FeynmanRules[L]`, where one can derive all vertices from a Lagrangian.

Target:
- Keep `feynman_rule(fields...)` for the single-vertex case.
- Add `feynman_rules(...)` for whole-Lagrangian extraction.

Suggested surface:

```python
L.feynman_rules()
L.feynman_rules(arity=3)
L.feynman_rules(select=[(Phi.bar, Phi, A), (Phi.bar, Phi, A, A)])
```

Suggested implementation:
- add a term-grouping pass over `Lagrangian.terms`
- infer unique field contents from each term
- call the existing single-vertex engine internally
- return a structured list or mapping of field content -> vertex expression

### 2. Make single terms directly usable

Problem:
- `L = Lagrangian(terms=(term_dphi4,))` is correct but unnatural.
- A single `InteractionTerm` is already a Lagrangian term and should behave as
  one.

Target:
- allow direct vertex extraction from one term without manual wrapping
- keep sums of terms natural to write and evaluate

Suggested surface:

```python
term.feynman_rule(Phi, Phi, Phi, Phi)
(term1 + term2).feynman_rule(...)
```

Suggested implementation:
- add `feynman_rule(...)` to `InteractionTerm` as a thin wrapper around
  `Lagrangian(terms=(self,))`
- do the same for other direct term objects where appropriate
- make sure `term1 + term2` consistently produces a Lagrangian-like object

### 3. More natural gauge-fixing syntax

Problem:
- `GaugeFixing(...)` is serviceable, but it still feels like a helper object
  rather than a genuine symbolic Lagrangian expression.

Target:
- keep the structured lowering backend
- add a more natural symbolic front end

Suggested surface:

```python
-(1 / (2 * xi)) * Div(A, mu) * Div(A, mu)
```

Possible path:
- introduce a typed `Div(...)` helper
- lower a small set of canonical divergence-squared patterns to the existing
  `GaugeFixingTerm` backend
- preserve the source declaration for printing and notebooks

### 4. Keep ghosts structured, but improve readability

Problem:
- `GhostLagrangian(G)` is readable enough for now, but still feels more like a
  compiler helper than a handwritten Lagrangian term.

Recommendation:
- keep `GhostLagrangian(group)` as the canonical safe API
- do not move to arbitrary free-form ghost parsing
- later add readable symbolic sugar only if it lowers to the same structured
  backend term

Reason:
- non-abelian ghost sectors are derived objects and are easy to get wrong if
  exposed as unrestricted handwritten expressions

### Recommended order

1. add whole-Lagrangian `feynman_rules(...)`
2. make single terms directly usable with `feynman_rule(...)`
3. add a symbolic front end for gauge fixing
4. improve ghost syntax only as sugar over the structured form
