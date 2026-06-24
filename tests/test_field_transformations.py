from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest
from symbolica import Expression, S

from feynpy import (
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
    compiled_is_hermitian,
    replacement,
)
from feynpy.interactions import DerivativeAction, InteractionTerm
from feynpy.lagrangian import CompiledLagrangian


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

    assert Counter(len(term.fields) for term in result.terms) == Counter({0: 1, 1: 1, 2: 1})
    assert Counter(_field_names(term) for term in result.terms) == Counter(
        {(): 1, ("h",): 1, ("h", "h"): 1}
    )
    linear = next(term for term in result.terms if _field_names(term) == ("h",))
    assert _canon(linear.coupling) == _canon((2 * v) * HALF)


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


def _replacement_summary(transformation):
    return [
        (
            _canon(term.coefficient),
            tuple(
                occurrence.field.name
                + (".bar" if occurrence.conjugated and not occurrence.field.self_conjugate else "")
                for occurrence in term.occurrences()
            ),
        )
        for term in transformation.terms
    ]


def test_expression_syntax_matches_explicit_terms_for_linear_mixing():
    b_field = _vector("B")
    z_field = _vector("Z")
    photon = _vector("A")
    sw = S("sw")
    cw = S("cw")

    via_expr = FieldTransformation(b_field, -sw * z_field + cw * photon)
    via_terms = FieldTransformation(
        b_field,
        terms=(replacement(-sw, z_field), replacement(cw, photon)),
    )

    assert _replacement_summary(via_expr) == _replacement_summary(via_terms)


def test_expression_syntax_gives_same_feynman_rule_after_transform():
    b_field = _vector("B")
    z_field = _vector("Z")
    photon = _vector("A")
    mu = S("mu")
    sw = S("sw")
    cw = S("cw")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(b_field(mu),)))

    via_expr = source.transform_fields(
        FieldTransformation(b_field, -sw * z_field + cw * photon),
        repeat=False,
    )
    via_terms = source.transform_fields(
        FieldTransformation(
            b_field,
            terms=(replacement(-sw, z_field), replacement(cw, photon)),
        ),
        repeat=False,
    )

    assert {_field_names(term) for term in via_expr.terms} == {("Z",), ("A",)}
    assert {_canon(term.coupling) for term in via_expr.terms} == {
        _canon(term.coupling) for term in via_terms.terms
    }


def test_expression_syntax_supports_vacuum_shift_constant_in_sum():
    doublet = _scalar("Phi", complex_field=True, indices=(WEAK_FUND_INDEX,))
    h = _scalar("h")
    g0 = _scalar("G0")
    vev = S("vev")

    transformation = FieldTransformation(
        doublet,
        vev * INV_SQRT2 + INV_SQRT2 * h + Expression.I * INV_SQRT2 * g0,
        components={0: 2},
    )

    summary = _replacement_summary(transformation)
    assert ("h",) in [fields for _coeff, fields in summary]
    assert ("G0",) in [fields for _coeff, fields in summary]
    # the vacuum piece is a constant monomial with no fields
    assert () in [fields for _coeff, fields in summary]


def test_expression_syntax_accepts_bare_field_and_conjugate():
    a_field = _vector("A")
    z_field = _vector("Z")
    charged = _scalar("G", complex_field=True)
    phi = _scalar("Phi", complex_field=True)

    assert _replacement_summary(FieldTransformation(a_field, z_field)) == [
        ("1", ("Z",))
    ]
    assert _replacement_summary(FieldTransformation(phi, charged.bar)) == [
        ("1", ("G.bar",))
    ]


def test_expression_syntax_rejects_multiple_replacement_specs():
    b_field = _vector("B")
    z_field = _vector("Z")
    photon = _vector("A")

    with pytest.raises(ValueError, match="only one of"):
        FieldTransformation(
            b_field,
            -z_field,
            terms=(replacement(1, photon),),
        )


def test_expression_syntax_rejects_non_field_operators():
    from fractions import Fraction

    psi = Field(
        "psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(LORENTZ_INDEX,),
    )
    chi = _scalar("chi")
    mu = S("mu")

    with pytest.raises(TypeError):
        FieldTransformation(chi, CovD(psi, mu))


def _dirac(name, indices):
    from fractions import Fraction

    return Field(
        name,
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S(name),
        conjugate_symbol=S(f"{name}bar"),
        indices=indices,
    )


def test_projector_expression_compiles_to_builder_with_dependencies():
    from feynpy import SPINOR_INDEX, ProjM, flavor_index

    generation = flavor_index("Generation", 3, prefix="fl")
    source = _dirac("LL", (SPINOR_INDEX, WEAK_FUND_INDEX, generation))
    target = _dirac("l", (SPINOR_INDEX, generation))

    rule = FieldTransformation(source, ProjM * target, components={1: 2})

    assert rule.terms == ()
    assert rule.builder is not None
    assert rule.conjugate_builder is not None
    assert target in rule.dependencies


def test_projector_expression_wires_spinor_index_and_inherits_flavor():
    from feynpy import SPINOR_INDEX, ProjM, flavor_index

    generation = flavor_index("Generation", 3, prefix="fl")
    source = _dirac("LL", (SPINOR_INDEX, WEAK_FUND_INDEX, generation))
    target = _dirac("l", (SPINOR_INDEX, generation))
    s = S("s")
    g = S("g")

    lagrangian = _lagrangian(
        InteractionTerm(coupling=1, fields=(source(s, 2, g),))
    )
    result = lagrangian.transform_fields(
        FieldTransformation(source, ProjM * target, components={1: 2}),
        repeat=False,
    )

    assert len(result.terms) == 1
    (term,) = result.terms
    assert _field_names(term) == ("l",)
    field = term.fields[0]
    target_spinor = field.slot_labels.get(0)
    assert _canon(field.slot_labels.get(1)) == _canon(g)  # flavor inherited
    assert _canon(target_spinor) != _canon(s)  # spinor leg is fresh
    coupling = _canon(term.coupling)
    assert "PL(" in coupling  # the compact projector is present
    assert _canon(target_spinor) in coupling  # contracted with the target


def test_projector_expression_conjugates_for_bar_occurrence():
    from feynpy import SPINOR_INDEX, ProjM, flavor_index

    generation = flavor_index("Generation", 3, prefix="fl")
    source = _dirac("LL", (SPINOR_INDEX, WEAK_FUND_INDEX, generation))
    target = _dirac("l", (SPINOR_INDEX, generation))
    s = S("s")
    g = S("g")

    lagrangian = _lagrangian(
        InteractionTerm(coupling=1, fields=(source.bar(s, 2, g),))
    )
    result = lagrangian.transform_fields(
        FieldTransformation(source, ProjM * target, components={1: 2}),
        repeat=False,
    )

    (term,) = result.terms
    assert _field_names(term) == ("l.bar",)
    assert "PR(" in _canon(term.coupling)


def test_projector_chain_canonicalizes_to_single_projector():
    from feynpy import SPINOR_INDEX, ProjM, ProjP

    source_left = _dirac("PsiL", (SPINOR_INDEX,))
    source_right = _dirac("PsiR", (SPINOR_INDEX,))
    target = _dirac("psi", (SPINOR_INDEX,))
    higgs = _scalar("H")
    s = S("s")

    lagrangian = _lagrangian(
        InteractionTerm(coupling=1, fields=(source_left.bar(s), source_right(s), higgs()))
    )
    result = lagrangian.transform_fields(
        FieldTransformation(source_left, ProjM * target),
        FieldTransformation(source_right, ProjP * target),
        repeat=False,
    )

    assert len(result.terms) == 1
    (term,) = result.terms
    assert _field_names(term) == ("psi.bar", "psi", "H")
    coupling = _canon(term.coupling)
    assert "PR(" in coupling
    assert "gamma5(" not in coupling


def test_static_replacement_freshens_explicit_dummy_labels_per_occurrence():
    phi = _scalar("Phi")
    x = _scalar("X", indices=(COLOR_FUND_INDEX,))
    y = _scalar("Y", indices=(COLOR_FUND_INDEX,))
    shared = S("c_shared")
    source = _lagrangian(
        InteractionTerm(coupling=1, fields=(phi(), phi()))
    )

    result = source.transform_fields(
        FieldTransformation(phi, terms=(replacement(1, x(shared), y(shared)),)),
        repeat=False,
    )

    assert len(result.terms) == 1
    labels = [occurrence.slot_labels.get(0) for occurrence in result.terms[0].fields]
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_compiled_is_hermitian_handles_projector_based_yukawa_pairs():
    from feynpy import SPINOR_INDEX, ProjM, ProjP

    psi = _dirac("psi", (SPINOR_INDEX,))
    source_left = _dirac("PsiL", (SPINOR_INDEX,))
    source_right = _dirac("PsiR", (SPINOR_INDEX,))
    higgs = _scalar("H")
    s = S("s")
    y = S("y")

    source = _lagrangian(
        InteractionTerm(coupling=y, fields=(source_left.bar(s), source_right(s), higgs())),
        InteractionTerm(coupling=y, fields=(higgs(), source_right.bar(s), source_left(s))),
    )
    result = source.transform_fields(
        FieldTransformation(source_left, ProjM * psi),
        FieldTransformation(source_right, ProjP * psi),
        repeat=False,
        real_symbols=(y,),
    )

    assert compiled_is_hermitian(
        result,
        real_symbols=(y,),
        field_heads=(psi, higgs),
        run_color=False,
    )


def test_rotation_expression_matches_explicit_builder():
    from feynpy import (
        SPINOR_INDEX,
        Parameter,
        ProjM,
        TransformationContext,
        flavor_index,
        rotation,
    )
    from feynpy.transformations import _rule_terms, FreshLabelPool

    generation = flavor_index("Generation", 3, prefix="fl")
    source = _dirac("QL", (SPINOR_INDEX, WEAK_FUND_INDEX, generation, COLOR_FUND_INDEX))
    target = _dirac("dq", (SPINOR_INDEX, generation, COLOR_FUND_INDEX))
    ckm = Parameter("CKM", indices=(generation, generation), complex_param=True,
                    unitary_partner="CKMDag")
    ckm_dag = Parameter("CKMDag", indices=(generation, generation), complex_param=True,
                        unitary_partner="CKM")

    via_expr = FieldTransformation(
        source, rotation(ckm, ckm_dag) * ProjM * target, components={1: 2}
    )

    def builder(context):
        labels = {
            index: context.label(slot)
            for slot, index in enumerate(context.occurrence.field.indices)
        }
        s_src = labels[SPINOR_INDEX]
        g_src = next(lbl for idx, lbl in labels.items() if idx.is_flavor)
        # Match the resolver's allocation order: flavor chain before spinor chain.
        g_tgt = context.fresh(generation, "transform")
        s_tgt = context.fresh(SPINOR_INDEX, "transform")
        from symbolic.spenso_structures import chiral_projector_left

        coeff = ckm(g_src, g_tgt) * chiral_projector_left(s_src, s_tgt)
        occ = target.occurrence(
            labels=target.pack_slot_labels(
                {0: s_tgt, 1: g_tgt, 2: labels[COLOR_FUND_INDEX]}
            )
        )
        return (replacement(coeff, occ),)

    via_builder = FieldTransformation(source, components={1: 2}, builder=builder)

    occurrence = source(S("s"), 2, S("g"), S("c"))
    term = InteractionTerm(coupling=1, fields=(occurrence,))

    def first_terms(rule):
        return _rule_terms(
            rule,
            occurrence=occurrence,
            term=term,
            slot=0,
            real_symbols=(),
            parameters=(),
            label_pool=FreshLabelPool(),
        )

    expr_term = first_terms(via_expr)[0]
    builder_term = first_terms(via_builder)[0]
    assert _canon(expr_term.coefficient) == _canon(builder_term.coefficient)


def test_matrix_expression_rejects_multiple_target_fields():
    from feynpy import SPINOR_INDEX, ProjM, flavor_index

    generation = flavor_index("Generation", 3, prefix="fl")
    source = _dirac("LL", (SPINOR_INDEX, WEAK_FUND_INDEX, generation))
    a = _dirac("a", (SPINOR_INDEX, generation))
    b = _dirac("b", (SPINOR_INDEX, generation))

    rule = FieldTransformation(source, ProjM * a * b, components={1: 2})
    lagrangian = _lagrangian(
        InteractionTerm(coupling=1, fields=(source(S("s"), 2, S("g")),))
    )
    with pytest.raises(TypeError):
        lagrangian.transform_fields(rule, repeat=False)


def test_matrix_expression_respects_real_symbols_on_conjugation():
    from fractions import Fraction

    from feynpy import SPINOR_INDEX, ProjM

    a = S("a")
    psi = Field(
        "psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    Psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("Psi"),
        conjugate_symbol=S("Psibar"),
        indices=(SPINOR_INDEX,),
    )
    s = S("s")
    source = _lagrangian(InteractionTerm(coupling=1, fields=(Psi.bar(s),)))
    rule = FieldTransformation(Psi, a * ProjM * psi)

    without = source.transform_fields(rule, repeat=False, real_symbols=())
    with_real = source.transform_fields(rule, repeat=False, real_symbols=(a,))

    coeff_without = without.terms[0].coupling.expand().to_canonical_string()
    coeff_with = with_real.terms[0].coupling.expand().to_canonical_string()
    assert "conj" in coeff_without
    assert "conj" not in coeff_with


def test_matrix_expression_records_replacement_term_dependencies():
    from feynpy import SPINOR_INDEX, ProjM

    B = _dirac("B", (SPINOR_INDEX,))
    C = _dirac("C", (SPINOR_INDEX,))
    A = _dirac("A", (SPINOR_INDEX,))

    rule = FieldTransformation(A, (ProjM * B, replacement(1, C)))
    assert B in rule.dependencies
    assert C in rule.dependencies


def test_matrix_expression_cycle_detection_includes_replacement_terms():
    from feynpy import SPINOR_INDEX, ProjM

    B = _dirac("B", (SPINOR_INDEX,))
    C = _dirac("C", (SPINOR_INDEX,))
    A = _dirac("A", (SPINOR_INDEX,))
    rules = (
        FieldTransformation(A, (ProjM * B, replacement(1, C))),
        FieldTransformation(C, replacement(1, A)),
    )

    with pytest.raises(CyclicTransformationError):
        apply_field_transformations(
            _lagrangian(InteractionTerm(coupling=1, fields=(A(S("s")),))),
            rules,
            repeat=True,
            max_passes=8,
        )
