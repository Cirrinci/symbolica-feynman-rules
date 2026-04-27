# Raw Spenso in `lagrangian_decl`: Implementation Plan

Purpose: document the next clean implementation step for manual gauge-fixing
and ghost source terms inside

- `Model(..., lagrangian_decl=...)`

without reintroducing `InteractionTerm(...)` in user-facing examples and
without adding new declarative wrapper classes such as `IndexMetric(...)` or
`Label(...)`.

## Current repo state

The temporary `IndexMetric(...)` / `Label(...)` experiment was reverted.

Current public declarative API remains:

- `CovD(...)`
- `Gamma(...)`
- `FieldStrength(...)`
- `GaugeFixing(...)`
- `GhostLagrangian(...)`
- local terms built from `Field`, `PartialD(...)`, `Metric(...)`, `T(...)`,
  `StructureConstant(...)`

## Actual problem

The real missing feature is not "one more tensor wrapper".

Raw Spenso tensors already fit naturally in the scalar coefficient of a
declared monomial, for example:

- `COLOR_ADJ_INDEX.representation.g(a_left, a_right).to_expression()`
- `LORENTZ_INDEX.representation.g(mu_left, rho_left).to_expression()`
- `structure_constant(a_bar, a_gluon, a_ghost)`

Those are already plain symbolic expressions.

What is missing is a way for `lagrangian_decl` to preserve explicit field-slot
labels on the field factors themselves, so those raw tensor indices can bind to
the intended Lorentz/color slots.

Today the declarative lowering path auto-generates field-slot labels, but it
cannot accept a source term like "this particular gluon carries Lorentz label
`mu_left` and adjoint label `a_left`" unless the user drops to
`InteractionTerm(...)`.

That is the gap to close.

## Correct direction

Do not add new declarative tensor classes.

Instead:

1. keep raw Spenso tensors in the monomial coefficient
2. reuse the already existing `FieldOccurrence(labels=...)` data structure
3. teach declarative lowering to accept labeled `FieldOccurrence` factors and
   preserve those labels through `PartialD(...)` lowering

This keeps the front end close to the real tensor objects already used
throughout the codebase.

## Target notebook syntax

After the implementation, the notebook example should look like this style:

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

That is the intended direction:

- raw Spenso tensors in the coefficient
- existing `FieldOccurrence(labels=...)` objects for explicit field binding
- no new user-facing declarative tensor wrapper classes

## Files to change

### 1. `src/model/interactions.py`

Why:

- `FieldOccurrence` exists already, but it is not usable as a factor in
  `lagrangian_decl`

Changes:

- add `__mul__`, `__rmul__`, `__add__`, `__radd__` on `FieldOccurrence`
  matching the behavior already present on `Field` and `ConjugateField`
- these methods should create a declarative `_FieldFactor` while preserving:
  - `field`
  - `conjugated`
  - `labels`

Optional:

- do not change external-leg behavior
- no new public class needed

### 2. `src/model/declared.py`

Why:

- the declarative parser currently accepts `Field`, `ConjugateField`,
  `_FieldFactor`, `PartialDerivativeFactor`, etc., but not labeled
  `FieldOccurrence`

Changes:

- extend `_FieldFactor` to carry optional `labels`
  - use the same kind-keyed label structure already used on `FieldOccurrence`
  - do not invent a second label format here
- extend `PartialDerivativeFactor` to carry optional field labels as well
  - this is needed because `PartialD(...)` must preserve labels from the input
    field factor
- update `_coerce_decl_factor(...)` so `FieldOccurrence` is accepted
- add a small declarative-only field parser, for example `_parse_decl_field_arg`
  or equivalent logic, that accepts:
  - `Field`
  - `ConjugateField`
  - `FieldOccurrence`
  - `(Field, bool)` if still wanted
- update `PartialD(...)` so `PartialD(FieldOccurrence(...), mu)` works and
  preserves labels

Scope rule:

- for this task, `CovD(...)` does not need labeled `FieldOccurrence` support
  unless tests show it is necessary
- keep the first patch narrowly focused on local source terms

Important:

- do not add `IndexMetricFactor`
- do not add `Label(...)`
- do not add a second tensor mini-language

### 3. `src/model/metadata.py`

Why:

- lowering needs the inverse of `Field.pack_slot_labels(...)`
- explicit labels arrive in kind-keyed form, but lowering works internally with
  slot-indexed labels

Changes:

- add an internal helper or a `Field` method that converts
  kind-keyed labels back to slot-indexed labels
- suggested name:
  - `Field.unpack_slot_labels(...)`
  - or a private helper near `pack_slot_labels(...)`

Behavior:

- single-slot kinds:
  - `{LORENTZ_KIND: mu}` maps to the one Lorentz slot
- repeated kinds:
  - tuple/list labels map ordinal-stably back onto the corresponding slots
- `None` entries should be ignored

This keeps the label representation consistent with the rest of the codebase.

### 4. `src/model/lowering.py`

Why:

- local declarative monomial lowering currently starts from empty slot-label
  maps and then auto-fills them
- explicit labels from `FieldOccurrence` need to seed that process

Changes:

- extend `_LocalFieldEntry` to carry the explicit field labels
- update `_local_field_entry_from_factor(...)` to preserve labels from:
  - `_FieldFactor`
  - `PartialDerivativeFactor`
- in `_lower_local_interaction_monomial(...)`, initialize each field's
  `slot_labels` from the explicit labels instead of always starting from `{}`
- use the new metadata helper from step 3 to convert those labels into the
  slot-indexed form used internally

Constraint:

- explicit labels must win
- auto-generated labels should only fill what the user did not specify

No change needed:

- `compiler/gauge.py`
- `symbolic/vertex_engine.py`

Those layers should remain unchanged if the declarative lowering feeds them the
correct `FieldOccurrence` objects.

### 5. `tests/test_lagrangian_api.py`

Add one focused regression test covering the target manual declarative source:

- manual gauge-fixing bilinear via raw Spenso tensors + labeled field factors
- manual ghost bilinear via raw Spenso tensors + labeled field factors
- manual ghost-gluon vertex via raw Spenso tensors + labeled field factors

Each should be compared against the canonical wrapper result from:

- `GaugeFixing(su3, xi=...)`
- `GhostLagrangian(su3)`

Comparison style:

- compare canonical strings of the simplified vertex expressions

This is the most important regression to add.

### 6. `notebooks/final_walkthrough_capabilities_and_usage.ipynb`

After the ghost section:

- add one code cell showing the manual declarative form
- do not show `InteractionTerm(...)`
- do not show `IndexMetric(...)`
- do not show `Label(...)`

The notebook cell should demonstrate:

1. the manual source term
2. the extracted vertices
3. equality checks against `GaugeFixing(...)` / `GhostLagrangian(...)`

## What not to do

Do not do any of the following:

- do not reintroduce `IndexMetric(...)`
- do not reintroduce `Label(...)`
- do not add another front-end tensor wrapper hierarchy
- do not change the compiled backend conventions
- do not replace the existing lowering/compiler path with a second symbolic path
- do not use `InteractionTerm(...)` in the notebook example

## Why this is better than the reverted attempt

The reverted attempt solved the problem by inventing new declarative wrapper
objects. That moved the API farther away from the actual tensor objects used in
the codebase.

The better approach is:

- raw Spenso tensor expressions stay raw
- explicit field labeling reuses the existing `FieldOccurrence` structure
- lowering learns how to preserve those labels

So the added API surface is minimal and the user-facing syntax stays closer to
the real symbolic objects.

## Acceptance checklist

The work is complete when all of the following are true:

- `docs/notes/RAW_SPENSO_DECLARATIVE_LAGRANGIAN_PLAN.md` has been followed
- no `IndexMetric(...)` or `Label(...)` remains in the tree
- `Model(..., lagrangian_decl=...)` accepts the manual ghost/gauge-fixing form
  written with raw Spenso tensors and labeled `FieldOccurrence(...)`
- the notebook contains that manual example
- the new regression test passes
- existing `tests/test_lagrangian_api.py`
- existing `tests/test_gauge_fixing_and_ghosts.py`

## Short execution order

1. `src/model/interactions.py`
2. `src/model/declared.py`
3. `src/model/metadata.py`
4. `src/model/lowering.py`
5. `tests/test_lagrangian_api.py`
6. notebook cell in `notebooks/final_walkthrough_capabilities_and_usage.ipynb`
