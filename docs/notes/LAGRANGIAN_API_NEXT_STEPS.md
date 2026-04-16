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

### 5. Replace the central declarative dispatcher with analyzer registries

Problem:
- The current declarative path is much cleaner than before, but it still relies
  on central ordered dispatch functions that know about every supported
  semantic family.
- Today the main decision points are:
  - `_analyze_declared_source_term(...)`
  - `_analyze_declared_monomial(...)`
  - `_match_covariant_monomial(...)`
  - `_lower_local_interaction_monomial(...)`
- This is acceptable at the current scale, but every new family still tends to
  push work back into the same central dispatch path.

What “semantic family” means here:
- local monomials
- covariant matter monomials
- field-strength monomials
- gauge-fixing declarations
- ghost declarations
- and later possibly:
  - Yukawa declarations
  - Higgs-potential declarations
  - electroweak symmetry-breaking declarations
  - tensor-field kinetic declarations

Target:
- Keep one normalized source-term analysis result.
- Stop growing the system by editing one central `if isinstance(...)` chain.
- Instead, add one analyzer per semantic family and register it.

Suggested shape:

```python
SOURCE_TERM_ANALYZERS = [
    analyze_explicit_interaction_term,
    analyze_declared_monomial,
    analyze_explicit_covariant_term,
    analyze_explicit_gauge_kinetic_term,
    analyze_gauge_fixing_term,
    analyze_ghost_term,
]

DECLARED_MONOMIAL_ANALYZERS = [
    analyze_covariant_monomial,
    analyze_field_strength_monomial,
    analyze_local_monomial,
]
```

Then the dispatcher becomes generic:

```python
def analyze_source_term(term):
    for analyzer in SOURCE_TERM_ANALYZERS:
        analyzed = analyzer(term)
        if analyzed is not None:
            return analyzed
    return None
```

And similarly for declared monomials.

Why this is better:
- adding a new semantic family means adding one analyzer function and
  registering it
- the central dispatcher stops growing every time the language grows
- each analyzer becomes independently testable
- ordering stays explicit, but lives in a small registry rather than inside one
  large function
- the code scales better when more gauge groups, field kinds, or mixed
  interaction families are added

Important scope constraint:
- This does **not** mean introducing a large plugin system.
- A very small internal registry is enough:
  - one list for source-term analyzers
  - one list for monomial analyzers
  - each analyzer returns `_AnalyzedSourceTerm | None`

Recommended implementation path:
1. keep `_AnalyzedSourceTerm` as the single normalized result type
2. split the current `_analyze_declared_source_term(...)` into small analyzer
   functions
3. split the current `_analyze_declared_monomial(...)` the same way
4. make validation, model views, compilation, and precompilation use only the
   registry-backed analysis path
5. only then start adding new semantic families

### Recommended order

1. add whole-Lagrangian `feynman_rules(...)`
2. make single terms directly usable with `feynman_rule(...)`
3. add a symbolic front end for gauge fixing
4. improve ghost syntax only as sugar over the structured form
5. replace the central declarative dispatcher with small analyzer registries
