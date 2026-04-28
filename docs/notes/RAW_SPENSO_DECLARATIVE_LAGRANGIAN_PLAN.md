# Raw Spenso in `lagrangian_decl`

Status: implemented.

## Goal

Support manual gauge-fixing and ghost source terms directly inside
`Model(..., lagrangian_decl=...)` while keeping the public syntax:

- readable
- close to the raw Spenso tensors already used elsewhere
- free of extra wrapper classes

The constraints stay explicit:

- do not reintroduce `InteractionTerm(...)` in user-facing examples
- do not add `IndexMetric(...)`, `Label(...)`, or a second tensor DSL

## Chosen solution

The missing feature was not "one more tensor wrapper".

Raw Spenso tensors already fit naturally in the scalar coefficient of a
declared monomial, for example:

- `COLOR_ADJ_INDEX.representation.g(a_left, a_right).to_expression()`
- `LORENTZ_INDEX.representation.g(mu_left, rho_left).to_expression()`
- `structure_constant(a_bar, a_gluon, a_ghost)`

What was missing was explicit field-slot binding inside declarative local
monomials.

The clean solution is:

1. keep raw Spenso tensors in the coefficient
2. reuse `FieldOccurrence(labels=...)` for explicit field binding
3. preserve those labels through `PartialD(...)` and local lowering
4. let auto-generated labels fill only the labels the user did not specify

This keeps the interface small and keeps the lowering logic inside the current
code path.

## Final user syntax

```python
manual_gauge_fixing_and_ghosts_model = Model(
    gauge_groups=(SU3C,),
    fields=(Gluon, GhostGluon),
    lagrangian_decl=(
        -(Expression.num(1) / (Expression.num(2) * xi))
        * COLOR_ADJ_INDEX.representation.g(a_left, a_right).to_expression()
        * LORENTZ_INDEX.representation.g(mu_left, rho_left).to_expression()
        * LORENTZ_INDEX.representation.g(mu_right, rho_right).to_expression()
        * PartialD(
            Gluon.occurrence(labels={LORENTZ_KIND: mu_left, COLOR_ADJ_KIND: a_left}),
            rho_left,
        )
        * PartialD(
            Gluon.occurrence(labels={LORENTZ_KIND: mu_right, COLOR_ADJ_KIND: a_right}),
            rho_right,
        )
        + COLOR_ADJ_INDEX.representation.g(a_bar, a_ghost).to_expression()
        * PartialD(
            GhostGluon.occurrence(conjugated=True, labels={COLOR_ADJ_KIND: a_bar}),
            mu,
        )
        * PartialD(
            GhostGluon.occurrence(labels={COLOR_ADJ_KIND: a_ghost}),
            mu,
        )
        + (
            -gs
            * structure_constant(a_bar, a_gluon, a_ghost)
            * LORENTZ_INDEX.representation.g(rho_ghost, mu_left).to_expression()
            * PartialD(
                GhostGluon.occurrence(conjugated=True, labels={COLOR_ADJ_KIND: a_bar}),
                rho_ghost,
            )
            * Gluon.occurrence(labels={LORENTZ_KIND: mu_left, COLOR_ADJ_KIND: a_gluon})
            * GhostGluon.occurrence(labels={COLOR_ADJ_KIND: a_ghost})
        )
    ),
)
```

## Implementation summary

### `src/model/interactions.py`

- `FieldOccurrence` now behaves like `Field` and `Field.bar` inside
  declarative products and sums
- the existing field/conjugation/labels payload is preserved as-is

### `src/model/declared.py`

- `_FieldFactor` now carries optional labels
- `PartialDerivativeFactor` now carries optional labels
- declarative factor coercion now accepts `FieldOccurrence`
- `PartialD(FieldOccurrence(...), mu)` now preserves labels

Scope stayed intentionally narrow:

- local declarative monomials are supported
- no extra tensor wrapper classes were introduced
- `CovD(...)` was left unchanged

### `src/model/metadata.py`

- `Field.unpack_slot_labels(...)` was added as the inverse of
  `Field.pack_slot_labels(...)`
- this keeps one label format across the codebase:
  kind-keyed labels at the public edge, slot-indexed labels internally

### `src/model/lowering.py`

- explicit labels are now carried into `_LocalFieldEntry`
- local lowering starts from those explicit labels
- auto-generated labels fill only missing slots

No compiler or vertex-engine changes were needed.

## Regression coverage

`tests/test_lagrangian_api.py` now covers the full target manual source:

- manual gauge-fixing bilinear
- manual ghost bilinear
- manual ghost-gluon vertex

The regression compares the manual raw-Spenso declaration against the existing
`GaugeFixing(...) + GhostLagrangian(...)` route and checks that the resulting
Feynman rules agree.
