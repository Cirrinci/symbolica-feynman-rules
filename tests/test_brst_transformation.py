"""Tests for ``lagrangian.operator_action.brst_transformation``.

The BRST operator is a genuinely odd derivation: it acts on gauge fields,
ghosts, antighosts, and auxiliary fields with graded Leibniz signs, and
its non-abelian nilpotency relies on both ghost anticommutation and the
Jacobi identity in the unreduced ``f`` basis.
"""

from __future__ import annotations

import pytest

from symbolica import Expression, S

from lagrangian.operator_action import (
    FieldOperator,
    apply_field_operator_to_term,
    brst_transformation,
)
from model import (
    COLOR_ADJ_INDEX,
    Field,
    GaugeGroup,
    GhostField,
    LORENTZ_INDEX,
    Lagrangian,
)
from model.interactions import InteractionTerm
from symbolic.spenso_structures import structure_constant
from symbolic.tensor_canonicalization import canonize_full


def _canon(expr):
    if hasattr(expr, "expand"):
        return expr.expand().to_canonical_string()
    return str(expr)


def _single_slot_term(occurrence):
    return InteractionTerm(coupling=Expression.num(1), fields=(occurrence,))


def _apply_single(operator, occurrence):
    return apply_field_operator_to_term(_single_slot_term(occurrence), operator)


def _single_slot_lagrangian(occurrence):
    return Lagrangian(terms=(_single_slot_term(occurrence),))


def _zero(expr):
    canonical = canonize_full(
        expr,
        run_gamma=False,
        run_color=False,
        infer_indices=True,
    )
    return canonical.to_canonical_string() == "0"


def _su3_brst_setup(*, explicit_antighost: bool = False):
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "c",
        ghost_of=gluon,
        self_conjugate=False,
        conjugate_symbol=S("cbar"),
        indices=(COLOR_ADJ_INDEX,),
    )
    if explicit_antighost:
        antighost = GhostField(
            "cbar",
            ghost_of=gluon,
            self_conjugate=False,
            indices=(COLOR_ADJ_INDEX,),
            quantum_numbers={"GhostNumber": -1},
        )
    else:
        antighost = ghost
    auxiliary = Field(
        "Baux",
        spin=0,
        self_conjugate=True,
        indices=(COLOR_ADJ_INDEX,),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
    )
    return gluon, ghost, antighost, auxiliary, su3


def _u1_brst_setup():
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX,),
    )
    ghost = GhostField(
        "cA",
        ghost_of=photon,
        self_conjugate=False,
        conjugate_symbol=S("cAbar"),
    )
    auxiliary = Field("BAux", spin=0, self_conjugate=True)
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=S("e"),
        gauge_boson=photon,
        charge="Q",
    )
    return photon, ghost, auxiliary, u1


def test_brst_transformation_returns_odd_field_operator():
    gluon, ghost, antighost, auxiliary, su3 = _su3_brst_setup()

    brst = brst_transformation(
        group=su3,
        ghost=ghost,
        auxiliary=auxiliary,
    )

    assert isinstance(brst, FieldOperator)
    assert brst.parity == 1
    assert brst._brst_antighost_field is ghost
    assert brst._brst_antighost_inferred_from_ghost is True


def test_brst_field_metadata_parities_are_explicit():
    gluon, ghost, _antighost_same, auxiliary, _su3 = _su3_brst_setup()
    antighost = GhostField(
        "cbar_explicit",
        ghost_of=gluon,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
        quantum_numbers={"GhostNumber": -1},
    )

    assert ghost.statistics == "fermion"
    assert antighost.statistics == "fermion"
    assert auxiliary.statistics == "boson"


def test_nonabelian_brst_elementary_rules_match_expected_terms():
    gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    mu, a = S("mu", "a")

    brst_A = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    A_terms = _apply_single(brst_A, gluon(mu, a))
    assert len(A_terms) == 2
    assert tuple((occ.field.name, occ.labels) for occ in A_terms[0].fields) == (
        ("c", {"color_adj": a}),
    )
    assert len(A_terms[0].derivatives) == 1
    assert A_terms[0].derivatives[0].target == 0
    assert A_terms[0].derivatives[0].lorentz_index == mu
    assert _canon(A_terms[0].coupling) == _canon(Expression.num(1))

    b, cidx = S("a_brst_SU3_1", "a_brst_SU3_2")
    assert tuple((occ.field.name, occ.labels) for occ in A_terms[1].fields) == (
        ("G", {"lorentz": mu, "color_adj": b}),
        ("c", {"color_adj": cidx}),
    )
    assert _canon(A_terms[1].coupling) == _canon(S("gS") * structure_constant(a, b, cidx))

    brst_c = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    c_terms = _apply_single(brst_c, ghost(a))
    assert len(c_terms) == 1
    assert tuple((occ.field.name, occ.labels) for occ in c_terms[0].fields) == (
        ("c", {"color_adj": b}),
        ("c", {"color_adj": cidx}),
    )
    assert _canon(c_terms[0].coupling) == _canon(
        -(Expression.num(1) / Expression.num(2)) * S("gS") * structure_constant(a, b, cidx)
    )

    brst_cbar = brst_transformation(
        group=su3,
        ghost=ghost,
        auxiliary=auxiliary,
    )
    cbar_terms = _apply_single(brst_cbar, ghost.bar(a))
    assert len(cbar_terms) == 1
    assert tuple((occ.field.name, occ.labels) for occ in cbar_terms[0].fields) == (
        ("Baux", {"color_adj": a}),
    )
    assert _canon(cbar_terms[0].coupling) == _canon(Expression.num(1))

    brst_B = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    assert _apply_single(brst_B, auxiliary(a)) == ()


def test_brst_supports_explicit_distinct_antighost_field():
    _gluon, ghost, antighost, auxiliary, su3 = _su3_brst_setup(explicit_antighost=True)
    a = S("a")

    brst = brst_transformation(
        group=su3,
        ghost=ghost,
        antighost=antighost,
        auxiliary=auxiliary,
    )
    terms = _apply_single(brst, antighost(a))

    assert len(terms) == 1
    assert tuple((occ.field.name, occ.labels) for occ in terms[0].fields) == (
        ("Baux", {"color_adj": a}),
    )


def test_brst_graded_leibniz_gives_minus_sign_for_ghost_antighost_product():
    _gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    a, b = S("a", "b")

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(ghost(a), ghost.bar(b)),
    )
    results = apply_field_operator_to_term(term, brst)

    assert len(results) == 2
    assert tuple((occ.field.name, occ.labels) for occ in results[0].fields) == (
        ("c", {"color_adj": S("a_brst_SU3_1")}),
        ("c", {"color_adj": S("a_brst_SU3_2")}),
        ("c", {"color_adj": b}),
    )
    assert _canon(results[0].coupling) == _canon(
        -(Expression.num(1) / Expression.num(2))
        * S("gS")
        * structure_constant(a, S("a_brst_SU3_1"), S("a_brst_SU3_2"))
    )

    assert tuple((occ.field.name, occ.labels) for occ in results[1].fields) == (
        ("c", {"color_adj": a}),
        ("Baux", {"color_adj": b}),
    )
    assert _canon(results[1].coupling) == _canon(-Expression.num(1))


def test_brst_nilpotency_on_su3_gauge_field():
    gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    mu, a = S("mu", "a")

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    s2_A = _single_slot_lagrangian(gluon(mu, a)).apply_operator(brst).apply_operator(brst)

    assert _zero(s2_A.to_symbolica().expand())


def test_brst_nilpotency_on_su3_ghost():
    _gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    a = S("a")

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    s2_c = _single_slot_lagrangian(ghost(a)).apply_operator(brst).apply_operator(brst)

    assert _zero(s2_c.to_symbolica().expand())


def test_brst_nilpotency_on_antighost_and_auxiliary():
    _gluon, ghost, antighost, auxiliary, su3 = _su3_brst_setup(explicit_antighost=True)
    a = S("a")

    brst = brst_transformation(
        group=su3,
        ghost=ghost,
        antighost=antighost,
        auxiliary=auxiliary,
    )

    s2_cbar = _single_slot_lagrangian(antighost(a)).apply_operator(brst).apply_operator(brst)
    s2_B = _single_slot_lagrangian(auxiliary(a)).apply_operator(brst).apply_operator(brst)

    assert s2_cbar.to_symbolica().expand().to_canonical_string() == "0"
    assert s2_B.to_symbolica().expand().to_canonical_string() == "0"


def test_abelian_brst_has_derivative_gauge_rule_and_zero_ghost_rule():
    photon, ghost, auxiliary, u1 = _u1_brst_setup()
    mu = S("mu")

    brst_A = brst_transformation(group=u1, ghost=ghost, auxiliary=auxiliary)
    A_terms = _apply_single(brst_A, photon(mu))
    assert len(A_terms) == 1
    assert tuple((occ.field.name, occ.labels) for occ in A_terms[0].fields) == (("cA", {}),)
    assert len(A_terms[0].derivatives) == 1
    assert A_terms[0].derivatives[0].target == 0
    assert A_terms[0].derivatives[0].lorentz_index == mu

    brst_c = brst_transformation(group=u1, ghost=ghost, auxiliary=auxiliary)
    assert _apply_single(brst_c, ghost()) == ()

    s2_A = _single_slot_lagrangian(photon(mu)).apply_operator(brst_A).apply_operator(brst_A)
    assert s2_A.to_symbolica().expand().to_canonical_string() == "0"


def test_brst_rejects_non_odd_ghost_field():
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    bad_ghost = Field(
        "eta",
        spin=0,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
    )
    auxiliary = Field(
        "Baux",
        spin=0,
        self_conjugate=True,
        indices=(COLOR_ADJ_INDEX,),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
    )

    with pytest.raises(ValueError, match="Grassmann-odd"):
        brst_transformation(
            group=su3,
            ghost=bad_ghost,
            auxiliary=auxiliary,
        )


def test_brst_rejects_nonabelian_group_without_structure_constant():
    gluon, ghost, _antighost, auxiliary, _su3 = _su3_brst_setup()
    bad_group = GaugeGroup(
        name="SU3bad",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon,
    )

    with pytest.raises(ValueError, match="structure_constant"):
        brst_transformation(
            group=bad_group,
            ghost=ghost,
            auxiliary=auxiliary,
        )


def test_brst_inferred_antighost_requires_explicit_ghost_conjugate_symbol():
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "c_plain",
        ghost_of=gluon,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
    )
    auxiliary = Field(
        "Baux",
        spin=0,
        self_conjugate=True,
        indices=(COLOR_ADJ_INDEX,),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
    )

    with pytest.raises(ValueError, match="declare conjugate_symbol"):
        brst_transformation(
            group=su3,
            ghost=ghost,
            auxiliary=auxiliary,
        )


def test_brst_antighost_without_auxiliary_is_rejected():
    _gluon, ghost, antighost, _auxiliary, su3 = _su3_brst_setup(explicit_antighost=True)

    with pytest.raises(ValueError, match="antighost supplied without an auxiliary field"):
        brst_transformation(
            group=su3,
            ghost=ghost,
            antighost=antighost,
        )


def test_brst_leaves_unrelated_fields_untouched():
    _gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    singlet = Field("Phi", spin=0, self_conjugate=True)

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)

    assert _apply_single(brst, singlet()) == ()
