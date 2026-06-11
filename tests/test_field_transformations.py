from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from symbolica import Expression, S

from model import (
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    CovD,
    CyclicTransformationError,
    Field,
    FieldTransformation,
    GaugeGroup,
    Model,
    ReplacementTerm,
    apply_field_transformations,
    replacement,
)
from model.interactions import DerivativeAction, InteractionTerm
from model.lagrangian import CompiledLagrangian


ONE = Expression.num(1)
HALF = ONE / Expression.num(2)
INV_SQRT2 = HALF**HALF


def _scalar(name, *, complex_field=False, indices=()):
    return Field(
        name,
        spin=0,
        self_conjugate=not complex_field,
        indices=indices,
        symbol=S(name),
        conjugate_symbol=S(f"{name}bar") if complex_field else None,
    )


def _vector(name, *, complex_field=False, indices=(LORENTZ_INDEX,)):
    return Field(
        name,
        spin=1,
        self_conjugate=not complex_field,
        indices=indices,
        symbol=S(name),
        conjugate_symbol=S(f"{name}bar") if complex_field else None,
    )


def _lagrangian(*terms):
    return CompiledLagrangian(terms=tuple(terms))


def _field_names(term):
    return tuple(
        occurrence.field.name + (
            ".bar"
            if occurrence.conjugated and not occurrence.field.self_conjugate
            else ""
        )
        for occurrence in term.fields
    )


def _canon(value):
    if not hasattr(value, "expand"):
        value = Expression.num(value)
    return value.expand().to_canonical_string()


def test_linear_combination_is_simultaneous_on_every_occurrence():
    phi = _scalar("Phi")
    x = _scalar("X")
    y = _scalar("Y")
    a = S("a")
    b = S("b")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(), phi()))
    )

    result = source.transform_fields(
        FieldTransformation(
            phi,
            terms=(replacement(a, x), replacement(b, y)),
        ),
        repeat=False,
    )

    assert Counter(_field_names(term) for term in result.terms) == Counter(
        {("X", "X"): 1, ("X", "Y"): 1, ("Y", "X"): 1, ("Y", "Y"): 1}
    )
    assert Counter(_canon(term.coupling) for term in result.terms) == Counter(
        {_canon(a * a): 1, _canon(a * b): 2, _canon(b * b): 1}
    )


def test_product_replacement_splices_ordered_factors():
    phi = _scalar("Phi")
    x = _scalar("X")
    y = _scalar("Y")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(phi(),)))

    result = source.transform_fields(
        FieldTransformation(phi, terms=(replacement(3, x, y),)),
    )

    assert len(result.terms) == 1
    assert _field_names(result.terms[0]) == ("X", "Y")
    assert _canon(result.terms[0].coupling) == _canon(Expression.num(3))


def test_partial_derivative_uses_leibniz_rule_on_product_replacement():
    phi = _scalar("Phi")
    x = _scalar("X")
    y = _scalar("Y")
    mu = S("mu")
    source = _lagrangian(
        InteractionTerm(
            coupling=1,
            fields=(phi(),),
            derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        )
    )

    result = source.transform_fields(
        FieldTransformation(phi, terms=(replacement(1, x, y),)),
    )

    assert len(result.terms) == 2
    assert {_field_names(term) for term in result.terms} == {("X", "Y")}
    assert {
        tuple((action.target, str(action.lorentz_index)) for action in term.derivatives)
        for term in result.terms
    } == {((0, "mu"),), ((1, "mu"),)}


def test_vacuum_shift_produces_constant_linear_and_quadratic_terms():
    phi = _scalar("Phi")
    h = _scalar("h")
    v = S("v")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(), phi()))
    )

    result = source.transform_fields(
        FieldTransformation(
            phi,
            terms=(
                replacement(v * INV_SQRT2),
                replacement(INV_SQRT2, h),
            ),
        ),
        repeat=False,
    )

    assert Counter(len(term.fields) for term in result.terms) == Counter({0: 1, 1: 2, 2: 1})
    assert Counter(_field_names(term) for term in result.terms) == Counter(
        {(): 1, ("h",): 2, ("h", "h"): 1}
    )


def test_neutral_gauge_boson_mixing_preserves_lorentz_label():
    b_field = _vector("B")
    z_field = _vector("Z")
    photon = _vector("A")
    mu = S("mu")
    sw = S("sw")
    cw = S("cw")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(b_field(mu),))
    )

    result = source.transform_fields(
        FieldTransformation(
            b_field,
            terms=(replacement(-sw, z_field), replacement(cw, photon)),
        ),
        repeat=False,
    )

    assert {_field_names(term) for term in result.terms} == {("Z",), ("A",)}
    assert {
        str(term.fields[0].slot_labels.get(0))
        for term in result.terms
    } == {"mu"}


def test_charged_field_combinations_use_component_rules():
    weak = _vector("Wi", indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX))
    charged = _vector("W", complex_field=True)
    mu = S("mu")
    component = _lagrangian(
        InteractionTerm(coupling=1, fields=(weak(mu, 1),))
    )

    result = component.transform_fields(
        FieldTransformation(
            weak,
            components={1: 1},
            terms=(
                replacement(INV_SQRT2, charged.bar),
                replacement(INV_SQRT2, charged),
            ),
        ),
        repeat=False,
    )

    assert {_field_names(term) for term in result.terms} == {("W.bar",), ("W",)}


def test_conjugated_source_uses_conjugated_replacement():
    phi = _scalar("Phi", complex_field=True)
    charged = _scalar("G", complex_field=True)
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi.bar(),))
    )

    result = source.transform_fields(
        FieldTransformation(
            phi,
            terms=(replacement(-Expression.I, charged),),
        ),
    )

    assert len(result.terms) == 1
    assert _field_names(result.terms[0]) == ("G.bar",)
    assert _canon(result.terms[0].coupling) == _canon(Expression.I)


def test_indexed_multiplet_decomposition_drops_component_index():
    doublet = _scalar("Doublet", complex_field=True, indices=(WEAK_FUND_INDEX,))
    upper = _scalar("Upper", complex_field=True)
    lower = _scalar("Lower")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(doublet(2),))
    )

    result = source.transform_fields(
        FieldTransformation(
            doublet,
            components={0: 2},
            terms=(replacement(1, lower),),
        ),
        repeat=False,
    )

    assert _field_names(result.terms[0]) == ("Lower",)
    assert result.terms[0].fields[0].slot_labels.values == ()


def test_callable_builder_freshens_dummy_indices_against_existing_labels():
    phi = _scalar("Phi")
    x = _scalar("X", indices=(COLOR_FUND_INDEX,))
    y = _scalar("Y", indices=(COLOR_FUND_INDEX,))
    spectator = _scalar("S", indices=(COLOR_FUND_INDEX,))
    source = _lagrangian(
        InteractionTerm(
            coupling=1,
            fields=(spectator(S("c_transform_1")), phi()),
        )
    )

    def build(context):
        label = context.fresh(COLOR_FUND_INDEX)
        return (replacement(1, x(label), y(label)),)

    result = source.transform_fields(
        FieldTransformation(phi, builder=build, auto_conjugate=False),
    )

    replacement_label = result.terms[0].fields[1].slot_labels.get(0)
    assert str(replacement_label) != "c_transform_1"
    assert result.terms[0].fields[2].slot_labels.get(0) == replacement_label


def test_callable_builder_freshens_each_replaced_occurrence_independently():
    phi = _scalar("Phi")
    x = _scalar("X", indices=(COLOR_FUND_INDEX,))
    y = _scalar("Y", indices=(COLOR_FUND_INDEX,))
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(), phi()))
    )

    def build(context):
        label = context.fresh(COLOR_FUND_INDEX)
        return (replacement(1, x(label), y(label)),)

    result = source.transform_fields(
        FieldTransformation(phi, builder=build, auto_conjugate=False),
        repeat=False,
    )

    assert len(result.terms) == 1
    labels = [occurrence.slot_labels.get(0) for occurrence in result.terms[0].fields]
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_invalid_transformation_cannot_drop_a_free_index():
    phi = _scalar("Phi", indices=(COLOR_FUND_INDEX,))
    x = _scalar("X")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(S("c_open")),))
    )

    with pytest.raises(ValueError, match="free index"):
        source.transform_fields(
            FieldTransformation(phi, terms=(replacement(1, x),)),
            repeat=False,
        )


def test_invalid_transformation_cannot_introduce_a_new_free_index():
    phi = _scalar("Phi")
    x = _scalar("X", indices=(COLOR_FUND_INDEX,))
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(),))
    )

    with pytest.raises(ValueError, match="free index"):
        source.transform_fields(
            FieldTransformation(phi, terms=(replacement(1, x(S("c_new"))),)),
            repeat=False,
        )


def test_missing_conjugate_rule_with_auto_conjugate_disabled_leaves_field_unchanged():
    phi = _scalar("Phi", complex_field=True)
    x = _scalar("X")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi.bar(),))
    )

    result = source.transform_fields(
        FieldTransformation(
            phi,
            terms=(replacement(1, x),),
            auto_conjugate=False,
        ),
        repeat=False,
    )

    assert len(result.terms) == 1
    assert _field_names(result.terms[0]) == ("Phi.bar",)


def test_explicit_empty_conjugate_replacement_annihilates_term():
    phi = _scalar("Phi", complex_field=True)
    x = _scalar("X")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi.bar(),))
    )

    result = source.transform_fields(
        FieldTransformation(
            phi,
            terms=(replacement(1, x),),
            conjugate_terms=(),
            auto_conjugate=False,
        ),
        repeat=False,
    )

    assert result.terms == ()


def test_nested_partial_derivatives_expand_all_leibniz_arrangements():
    phi = _scalar("Phi")
    a = _scalar("A")
    b = _scalar("B")
    c = _scalar("C")
    source = _lagrangian(
        InteractionTerm(
            coupling=1,
            fields=(phi(),),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=S("mu")),
                DerivativeAction(target=0, lorentz_index=S("nu")),
            ),
        )
    )

    result = source.transform_fields(
        FieldTransformation(phi, terms=(replacement(1, a, b, c),)),
        repeat=False,
    )

    assert len(result.terms) == 9
    assert {_field_names(term) for term in result.terms} == {("A", "B", "C")}
    assert {
        tuple((action.target, str(action.lorentz_index)) for action in term.derivatives)
        for term in result.terms
    } == {
        ((0, "mu"), (0, "nu")),
        ((0, "mu"), (1, "nu")),
        ((0, "mu"), (2, "nu")),
        ((1, "mu"), (0, "nu")),
        ((1, "mu"), (1, "nu")),
        ((1, "mu"), (2, "nu")),
        ((2, "mu"), (0, "nu")),
        ((2, "mu"), (1, "nu")),
        ((2, "mu"), (2, "nu")),
    }


def test_dependent_transformations_reach_fixed_point():
    a_field = _scalar("A")
    b_field = _scalar("B")
    c_field = _scalar("C")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(a_field(),)))

    result = source.transform_fields(
        FieldTransformation(a_field, terms=(replacement(1, b_field),)),
        FieldTransformation(b_field, terms=(replacement(1, c_field),)),
    )

    assert _field_names(result.terms[0]) == ("C",)


def test_cyclic_transformations_are_rejected():
    a_field = _scalar("A")
    b_field = _scalar("B")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(a_field(),)))

    with pytest.raises(CyclicTransformationError, match="A -> B -> A"):
        apply_field_transformations(
            source,
            (
                FieldTransformation(a_field, terms=(replacement(1, b_field),)),
                FieldTransformation(b_field, terms=(replacement(1, a_field),)),
            ),
        )


def test_callable_transformation_dependencies_are_cycle_checked():
    a_field = _scalar("A")
    b_field = _scalar("B")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(a_field(),)))

    with pytest.raises(CyclicTransformationError, match="A -> B -> A"):
        source.transform_fields(
            FieldTransformation(
                a_field,
                builder=lambda _context: (replacement(1, b_field),),
                dependencies=(b_field,),
                auto_conjugate=False,
            ),
            FieldTransformation(
                b_field,
                builder=lambda _context: (replacement(1, a_field),),
                dependencies=(a_field,),
                auto_conjugate=False,
            ),
        )


def test_component_specific_cycles_are_rejected():
    multiplet = _scalar("A", indices=(WEAK_FUND_INDEX,))
    b_field = _scalar("B")
    c_field = _scalar("C")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(multiplet(1),)))

    with pytest.raises(CyclicTransformationError, match="A"):
        source.transform_fields(
            FieldTransformation(
                multiplet,
                components={0: 1},
                terms=(replacement(1, b_field),),
                name="A[1]",
            ),
            FieldTransformation(
                multiplet,
                components={0: 2},
                terms=(replacement(1, c_field),),
                name="A[2]",
            ),
            FieldTransformation(
                b_field,
                terms=(replacement(1, multiplet(1)),),
                name="B",
            ),
        )


def test_transformations_apply_after_covariant_derivative_expansion():
    phi = _scalar("Phi", complex_field=True)
    h = _scalar("h")
    photon = _vector("A")
    coupling = S("e")
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=coupling,
        gauge_boson=photon,
        charge="Q",
    )
    charged_phi = replace(
        phi,
        quantum_numbers={"Q": ONE},
    )
    mu = S("mu")
    model = Model(
        gauge_groups=(u1,),
        fields=(charged_phi, photon),
        lagrangian_decl=CovD(charged_phi.bar, mu) * CovD(charged_phi, mu),
    )

    result = model.transform_fields(
        FieldTransformation(charged_phi, terms=(replacement(1, h),)),
    )

    assert result.terms
    assert all(
        occurrence.field is not charged_phi
        for term in result.terms
        for occurrence in term.fields
    )
    assert any(term.derivatives for term in result.terms)
    assert any(
        photon in tuple(occurrence.field for occurrence in term.fields)
        for term in result.terms
    )


def test_component_expansion_preserves_already_fixed_numeric_labels():
    weak = _scalar("Weak", indices=(WEAK_ADJ_INDEX,))
    source = _lagrangian(
        InteractionTerm(
            coupling=S("g") + 1,
            fields=(weak(2),),
        )
    )

    result = source.expand_index_components(WEAK_ADJ_INDEX)

    assert len(result.terms) == 1
    assert str(result.terms[0].fields[0].slot_labels.get(0)) == "2"
    assert _canon(result.terms[0].coupling) == _canon(S("g") + 1)
