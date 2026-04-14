---
name: FeynRules-style Lagrangian API
overview: Add a `Lagrangian` object and a single `compute_feynman_rules` entry point so that users declare Lagrangian terms, specify external fields in order, and get back vertex factors with automatic index/momentum conventions -- matching the FeynRules workflow.
todos:
  - id: conjugate-field
    content: Add `ConjugateField` dataclass and `Field.bar` property to `src/model.py`
    status: completed
  - id: lagrangian-class
    content: Add `Lagrangian` class with `__add__`, `__radd__`, and `feynman_rule()` method to `src/model.py`
    status: completed
  - id: auto-conventions
    content: Implement automatic index (`i1,i2,...`) and momentum (`q1,q2,...`) assignment inside `feynman_rule()`
    status: completed
  - id: term-matching
    content: Implement field-matching logic that scans Lagrangian terms for species-compatible interactions
    status: completed
  - id: model-lagrangian
    content: Add `Model.lagrangian()` method that compiles and wraps all terms
    status: completed
  - id: tests
    content: Add tests verifying the new API against existing vertex_factor results
    status: completed
isProject: false
---

# FeynRules-Style Lagrangian API

## Current Problem

The user-facing API requires too many manual steps and has too many function variants:
- Users must manually construct `InteractionTerm`, `ExternalLeg` objects with explicit momenta, labels, species, etc.
- `vertex_factor()` has two modes (direct parallel-list vs model-layer), each with ~15 keyword arguments
- Legs must be constructed with `field.leg(momentum, conjugated=..., labels={...})`
- There is no `Lagrangian` object -- interactions live as a flat tuple on `Model`

FeynRules, by contrast, has:
1. Declarative Lagrangian terms that compose with `+`
2. A single "FeynmanRules[L]" call that extracts all vertices
3. No manual leg/momentum/index assignment

## Design

### 1. `Lagrangian` class in [src/model.py](src/model.py)

A thin container of `InteractionTerm` objects that supports composition:

```python
@dataclass
class Lagrangian:
    terms: tuple[InteractionTerm, ...] = ()
    
    def __add__(self, other: "Lagrangian") -> "Lagrangian":
        return Lagrangian(terms=self.terms + other.terms)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented
    
    def feynman_rule(self, *fields, momenta=None, simplify=True) -> Expression:
        ...
```

The `feynman_rule` method is the single entry point:

```python
L = LGauge + LFermions + LHiggs
vertex = L.feynman_rule(Phibar, Phi, A)
```

- `fields` are `(Field, bool)` tuples or plain `Field` objects (with conjugation inferred or specified via a helper like `Phi.bar` or a simple `(Phi, True)` convention)
- Automatic conventions (see section 3 below)

### 2. Field conjugation sugar

Add a property to `Field` in [src/model.py](src/model.py) that returns a lightweight marker:

```python
@dataclass(frozen=True)
class ConjugateField:
    field: Field

class Field:
    ...
    @property
    def bar(self) -> ConjugateField:
        return ConjugateField(self)
```

This lets users write `Phi.bar` instead of `(Phi, True)` for conjugated fields.

### 3. Automatic leg conventions

When `feynman_rule(*fields)` is called, the following are assigned automatically:

- **Leg order**: position in the argument list = leg position
- **Momenta**: `q1, q2, q3, ...` (Symbolica symbols), one per leg. Can be overridden via `momenta=[expr1, expr2, ...]` for algebraic expressions like `q1 = p3 - p6`
- **External indices**: `i1, i2, i3, ...` (one symbol per open index slot carried by each leg)

Internally, `feynman_rule` will:
1. For each `(field, conjugated)` argument, build an `ExternalLeg` with auto-generated momentum and index labels
2. Scan `self.terms` for all `InteractionTerm` objects whose field content matches the given external fields (species-compatible, same multiplicity)
3. Call `vertex_factor(interaction=term, external_legs=legs, ...)` for each match
4. Sum the results
5. Optionally simplify with `simplify_vertex()`

### 4. Term matching logic

A new `_match_fields` helper matches the user's external field list against each `InteractionTerm.fields`:
- Check that the number of fields is the same
- Check that each field occurrence's species matches the external field species
- A term matches if there is a valid 1:1 assignment of interaction fields to external fields

### 5. Integration with the gauge compiler

The `Model` class gets a `lagrangian()` method that returns a `Lagrangian` built from all its compiled interactions:

```python
class Model:
    def lagrangian(self) -> Lagrangian:
        compiled = compile_covariant_terms(self)
        all_terms = self.interactions + compiled
        return Lagrangian(terms=all_terms)
```

This lets users work in two styles:
- **Manual**: build `Lagrangian` from `InteractionTerm` objects directly
- **Model-level**: declare gauge groups, fields, covariant terms, then call `model.lagrangian().feynman_rule(...)`

### 6. Covariant derivative (implemented front-end, generalized back-end deferred)

The repository now supports a declarative `CovD(field, mu)` front-end inside
`lagrangian_decl=...`, e.g.

- `I * Psi.bar * Gamma(mu) * CovD(Psi, mu)`
- `CovD(Phi.bar, mu) * CovD(Phi, mu)`

The fully generalized symbolic expansion

- `partial_mu field + i sum_G g_G A_mu T_R field`

is still a future extension described in
[docs/notes/COVARIANT_DERIVATIVE_GENERALIZATION.md](docs/notes/COVARIANT_DERIVATIVE_GENERALIZATION.md).

## Files to Change

- **[src/model.py](src/model.py)**: Add `ConjugateField`, `Lagrangian` class with `feynman_rule()`, add `Field.bar` property, add `Model.lagrangian()` method
- **[src/model_symbolica.py](src/model_symbolica.py)**: No changes to the engine. The `Lagrangian.feynman_rule()` calls existing `vertex_factor()` internally
- **[src/gauge_compiler.py](src/gauge_compiler.py)**: Minor -- ensure `compile_covariant_terms` result can be wrapped into a `Lagrangian`

## Example Usage After Implementation

```python
# FeynRules-style declarations (recommended)
SU3C = GaugeGroup(name="SU3C", abelian=False, coupling=g3, ...)
G = Field(name="G", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX))
Phi = Field(name="Phi", spin=0, self_conjugate=False, indices=(SU2D_INDEX,), quantum_numbers={"Y": Fraction(1,2)})

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

# NEW: single command
L = model.lagrangian()
vertex = L.feynman_rule(Phi.bar, Phi, A)
# Returns: i * coupling * (tensor structure) * (2pi)^d Delta(q1+q2+q3)
# with q1, q2, q3 as momenta and i1, i2, i3, ... as open indices

# Or with explicit momenta:
vertex = L.feynman_rule(Phi.bar, Phi, A, momenta=[p3 - p6, p1, p2])
```
