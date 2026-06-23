# Field Transformations

The model layer supports FeynRules-style field definitions on compiled
Lagrangians. The machinery is independent of field names and can express
mixing, multiplet decomposition, vacuum shifts, products, conjugation, and
index-aware callable replacements.

## Declaring Rules

The most readable form mirrors the FeynRules `Definitions` block: pass the
replacement as a field expression. It is the second positional argument (or
`expr=`):

```python
from symbolica import Expression, S

from feynpy import FieldTransformation
from symbolic.vertex_engine import I

half = Expression.num(1) / Expression.num(2)
inv_sqrt2 = half**half

rules = (
    # B -> -sw Z + cw A
    FieldTransformation(B, -sw * Z + cw * A),
    # Phi[2] -> v/sqrt2 + H/sqrt2 + i G0/sqrt2   (the bare v is a vacuum shift)
    FieldTransformation(
        Phi,
        vev * inv_sqrt2 + inv_sqrt2 * H + I * inv_sqrt2 * G0,
        components={0: 2},
    ),
)

broken = model.transform_fields(*rules, repeat=False)
```

The expression accepts the same field-arithmetic DSL used by
`Model(lagrangian_decl=...)`: `Field`, `Field.bar`, their scalar-weighted sums
and products, bare scalar constants (vacuum shifts), and tuples mixing those.
Only plain field monomials are accepted; anything that needs fresh indices or
spinor matrices (chiral projectors, CKM rotations) must use `builder=`.

### Explicit `terms=` form

The expression form lowers to the lower-level `terms=` API, which remains
available when you want to build `ReplacementTerm`s directly:

```python
from feynpy import replacement

rules = (
    FieldTransformation(B, terms=(replacement(-sw, Z), replacement(cw, A))),
    FieldTransformation(
        Phi,
        components={0: 2},
        terms=(
            replacement(vev * inv_sqrt2),
            replacement(inv_sqrt2, H),
            replacement(I * inv_sqrt2, G0),
        ),
    ),
)
```

Exactly one of `expr`, `terms`, or `builder` may be given per transformation.

Each `ReplacementTerm` is one monomial. Its coefficient can be any Symbolica
expression and its fields can contain zero, one, or several field
occurrences. Zero fields represent a spacetime-independent constant.

`components={slot: value}` restricts a rule to one explicit multiplet
component. Compatible free labels are inherited by replacement fields;
fixed component slots disappear when the replacement no longer carries that
index.

For index-dependent rotations, use `builder=`. The builder receives a
`TransformationContext`:

```python
def rotate(context):
    source_generation = context.label(1)
    target_generation = context.fresh(Generation, "mass")
    return (
        replacement(
            CKM(source_generation, target_generation),
            down(target_generation),
        ),
    )
```

`context.fresh(...)` avoids every label already bound in the interaction term.
Callable rules should list fields they may create in `dependencies=(...)` so
dependency cycles can be rejected before evaluation.

## Ordering

One pass is simultaneous: fields created by a rule are not transformed again
within that pass. Rule declaration order therefore does not change the result
of an unambiguous pass.

With `repeat=True` (the default), passes continue until no rule matches.
This gives deterministic fixed-point behavior for dependent rules such as
`A -> B`, `B -> C`. Static and declared callable dependencies are checked for
cycles. `max_passes` remains a defensive bound for undeclared dynamic builder
behavior.

Overlapping rules for the same occurrence are rejected as ambiguous.

## Conjugation

Static replacements are conjugated automatically for occurrences of
`field.bar`:

- coefficients are complex-conjugated;
- product order is reversed;
- complex fields toggle their conjugation;
- self-conjugate fields remain unchanged.

Pass real parameters or expressions through `real_symbols=` when applying the
stage. For matrix/projector conventions that cannot be inferred
structurally, declare `conjugate_terms=` or `conjugate_builder=`.

## Derivatives

Partial derivatives are structural metadata on compiled interactions.
Replacing

```text
Phi -> X + Y Z
```

therefore gives

```text
partial(Phi) -> partial(X) + partial(Y) Z + Y partial(Z)
```

Existing nested derivatives are distributed independently over every factor,
including the mixed terms required by repeated Leibniz expansion. Partial
derivatives are parity-even, so they add no Grassmann sign. Existing
closed-Dirac-bilinear metadata and field ordering continue to control
fermionic signs.

A differentiated constant replacement contributes zero. Parameters, mixing
angles, and vacuum expectation values are consequently treated as
spacetime-independent unless they are represented as fields.

Covariant derivatives are deliberately not transformed as partial
derivatives of the replacement fields. `Model.lagrangian()` first expands
`CovD` and `FieldStrength` in the original gauge basis. Finite multiplet
indices are then expanded, and field transformations are applied to the
resulting component fields and gauge bosons. This preserves the original
gauge representation and reproduces the ordering used by `SM.fr`.

Integration-by-parts and derivative-target metadata remain attached to the
generated terms through the shared operator-splicing implementation.

## Broken Standard Model

`theories.build_standard_model()` declares the gauge-basis Standard Model, compiles its
covariant derivatives and field strengths, expands the weak doublet/triplet
indices, and applies the `SM.fr` transformations in one simultaneous stage:

```python
from theories import build_standard_model

sm = build_standard_model()
L = sm.lagrangian

wwa = L.feynman_rule(sm.fields.W.bar, sm.fields.W, sm.fields.A)
hff = L.feynman_rule(sm.fields.l.bar, sm.fields.l, sm.fields.H)
```

This is deliberate package layering: the reusable transformation engine lives
under `feynpy`, while concrete theories live under `theories`.

`sm.source_model` exposes the gauge-basis declarations for inspection.
`sm.transformations` contains the declarative rules, while `sm.model` and
`sm.lagrangian` contain the physical-basis result.

The builder includes the gauge, Higgs, fermion, Yukawa, QCD, electroweak
Feynman-gauge ghost, CKM, Higgs/Goldstone, and vacuum-shift content used by the
non-BFM `SM.fr` reference.
