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
from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    Field,
    FieldStrength,
    GaugeGroup,
    GaugeRepresentation,
    GhostField,
    LORENTZ_INDEX,
    CompiledLagrangian,
    Model,
    PartialD,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    dirac_field,
    scalar_field,
)
from feynpy.interactions import InteractionTerm
from symbolic.spenso_structures import (
    gauge_generator,
    lorentz_metric,
    structure_constant,
    weak_gauge_generator,
    weak_structure_constant,
)
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
    return CompiledLagrangian(terms=(_single_slot_term(occurrence),))


def _zero(expr, *, field_heads=(), run_gamma=False, run_color=False):
    canonical = canonize_full(
        expr,
        run_gamma=run_gamma,
        run_color=run_color,
        infer_indices=True,
        field_heads=tuple(field_heads),
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


def _su3_brst_matter_setup(*, explicit_antighost: bool = False):
    gluon, ghost, antighost, auxiliary, _su3 = _su3_brst_setup(
        explicit_antighost=explicit_antighost
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fundamental",
            ),
        ),
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


def _su2_brst_matter_setup():
    weak_boson = Field(
        "W",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )
    ghost = GhostField(
        "cW",
        ghost_of=weak_boson,
        self_conjugate=False,
        indices=(WEAK_ADJ_INDEX,),
    )
    auxiliary = Field(
        "BW",
        spin=0,
        self_conjugate=True,
        indices=(WEAK_ADJ_INDEX,),
    )
    su2 = GaugeGroup(
        name="SU2L",
        abelian=False,
        coupling=S("g2"),
        gauge_boson=weak_boson,
        structure_constant=weak_structure_constant,
        representations=(
            GaugeRepresentation(
                index=WEAK_FUND_INDEX,
                generator_builder=weak_gauge_generator,
                name="doublet",
            ),
        ),
    )
    return weak_boson, ghost, auxiliary, su2


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


def test_brst_exact_gauge_fixing_fermion_and_yang_mills_sum_are_invariant():
    gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    mu, nu, a = S("mu", "nu", "a")
    xi = S("xi")

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    psi = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost, auxiliary),
        lagrangian_decl=(
            ghost.bar(a) * PartialD(gluon(mu, a), mu)
            + xi / Expression.num(2) * ghost.bar(a) * auxiliary(a)
        ),
    ).lagrangian()

    s_psi = psi.apply_operator(brst)
    assert _zero(s_psi.apply_operator(brst).to_symbolica().expand())

    yang_mills = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=(
            -(Expression.num(1) / Expression.num(4))
            * FieldStrength(su3, mu, nu, S("a_decl"))
            * FieldStrength(su3, mu, nu, S("a_decl"))
        ),
    ).lagrangian()

    total = yang_mills + s_psi
    assert _zero(total.apply_operator(brst).to_symbolica().expand())


def test_brst_gauge_fixing_fermion_ghost_block_has_expected_negative_sign():
    """The ghost block of ``s(Psi)`` carries the sign from odd Leibniz action.

    This regression inspects the ordered ``InteractionTerm`` output rather than
    the commutative Symbolica export. In the authoritative ordered form

    ``Psi = cbar * partial.G + xi/2 * cbar * B``

    the BRST variation must contain

    * ``- cbar * partial.partial c``
    * ``- g f cbar * (partial G) * c``
    * ``- g f cbar * G * (partial c)``

    with the antighost still on the left. If any future change flips the odd
    Leibniz sign, treats ghosts as even, or silently reorders the factors
    without the Grassmann minus, this test should fail.
    """

    gluon, ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    mu, a = S("mu", "a")
    xi = S("xi")

    brst = brst_transformation(group=su3, ghost=ghost, auxiliary=auxiliary)
    psi = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost, auxiliary),
        lagrangian_decl=(
            ghost.bar(a) * PartialD(gluon(mu, a), mu)
            + xi / Expression.num(2) * ghost.bar(a) * auxiliary(a)
        ),
    ).lagrangian()

    s_psi = psi.apply_operator(brst)

    ghost_kinetic = None
    ghost_covariant = []

    for term in s_psi.terms:
        field_signature = tuple((occ.field.name, occ.conjugated) for occ in term.fields)
        derivative_targets = tuple(action.target for action in term.derivatives)

        if field_signature == (("c", True), ("c", False)):
            ghost_kinetic = term
        elif field_signature == (("c", True), ("G", False), ("c", False)):
            ghost_covariant.append(term)

    assert ghost_kinetic is not None
    assert len(ghost_kinetic.derivatives) == 2
    assert tuple(action.target for action in ghost_kinetic.derivatives) == (1, 1)
    assert ghost_kinetic.fields[0].labels == {"color_adj": a}
    assert ghost_kinetic.fields[1].labels == {"color_adj": a}
    inner_mu = ghost_kinetic.derivatives[0].lorentz_index
    assert ghost_kinetic.derivatives[1].lorentz_index == mu
    assert _canon(ghost_kinetic.coupling) == _canon(-lorentz_metric(mu, inner_mu))

    assert len(ghost_covariant) == 2
    assert {tuple(action.target for action in term.derivatives) for term in ghost_covariant} == {
        (1,),
        (2,),
    }

    for term in ghost_covariant:
        cbar_occ, gluon_occ, ghost_occ = term.fields
        assert cbar_occ.labels == {"color_adj": a}
        assert gluon_occ.labels["lorentz"] == mu
        deriv_mu = term.derivatives[0].lorentz_index
        expected = (
            -S("gS")
            * lorentz_metric(mu, deriv_mu)
            * structure_constant(
                cbar_occ.labels["color_adj"],
                gluon_occ.labels["color_adj"],
                ghost_occ.labels["color_adj"],
            )
        )
        assert _canon(term.coupling) == _canon(expected)


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


def test_brst_rejects_odd_non_ghost_field_even_if_adjoint_shaped():
    gluon, _ghost, _antighost, auxiliary, su3 = _su3_brst_setup()
    bad_ghost = Field(
        "eta",
        spin=1 / 2,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
    )

    with pytest.raises(ValueError, match="kind='ghost'"):
        brst_transformation(
            group=su3,
            ghost=bad_ghost,
            auxiliary=auxiliary,
        )


def test_brst_rejects_ghost_from_the_wrong_nonabelian_group():
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    other_gluon = Field(
        "H",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "cG",
        ghost_of=other_gluon,
        self_conjugate=False,
        conjugate_symbol=S("cGbar"),
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

    with pytest.raises(ValueError, match="associated with the selected gauge boson"):
        brst_transformation(
            group=su3,
            ghost=ghost,
            auxiliary=auxiliary,
        )


def test_brst_rejects_antighost_from_the_wrong_nonabelian_group():
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    other_gluon = Field(
        "H",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "cG",
        ghost_of=gluon,
        self_conjugate=False,
        conjugate_symbol=S("cGbar"),
        indices=(COLOR_ADJ_INDEX,),
    )
    antighost = GhostField(
        "cHbar",
        ghost_of=other_gluon,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
        quantum_numbers={"GhostNumber": -1},
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

    with pytest.raises(ValueError, match="associated with the selected gauge boson"):
        brst_transformation(
            group=su3,
            ghost=ghost,
            antighost=antighost,
            auxiliary=auxiliary,
        )


def test_brst_accepts_matching_ghost_and_antighost_for_the_selected_group():
    gluon, ghost, antighost, auxiliary, su3 = _su3_brst_setup(explicit_antighost=True)

    brst = brst_transformation(
        group=su3,
        ghost=ghost,
        antighost=antighost,
        auxiliary=auxiliary,
    )

    assert brst._brst_ghost_field is ghost
    assert brst._brst_antighost_field is antighost


def test_abelian_brst_scalar_rule_matches_charge_convention_without_auxiliary():
    _photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 2})

    brst = brst_transformation(group=u1, ghost=ghost)

    phi_terms = _apply_single(brst, phi())
    assert len(phi_terms) == 1
    assert tuple((occ.field.name, occ.conjugated) for occ in phi_terms[0].fields) == (
        ("cA", False),
        ("Phi", False),
    )
    assert _canon(phi_terms[0].coupling) == _canon(Expression.I * S("e") * Expression.num(2))

    phibar_terms = _apply_single(brst, phi.bar())
    assert len(phibar_terms) == 1
    assert tuple((occ.field.name, occ.conjugated) for occ in phibar_terms[0].fields) == (
        ("cA", False),
        ("Phi", True),
    )
    assert _canon(phibar_terms[0].coupling) == _canon(-Expression.I * S("e") * Expression.num(2))


def test_abelian_brst_fermion_rule_matches_charge_convention():
    _photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    psi = dirac_field("psi", quantum_numbers={"Q": -1})
    i = S("i")

    brst = brst_transformation(group=u1, ghost=ghost)

    psi_terms = _apply_single(brst, psi(i))
    assert len(psi_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in psi_terms[0].fields) == (
        ("cA", False, {}),
        ("psi", False, {"spinor": i}),
    )
    assert _canon(psi_terms[0].coupling) == _canon(-Expression.I * S("e"))

    psibar_terms = _apply_single(brst, psi.bar(i))
    assert len(psibar_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in psibar_terms[0].fields) == (
        ("cA", False, {}),
        ("psi", True, {"spinor": i}),
    )
    assert _canon(psibar_terms[0].coupling) == _canon(Expression.I * S("e"))


def test_nonabelian_brst_scalar_rule_matches_generator_convention():
    _gluon, ghost, _antighost, _auxiliary, su3 = _su3_brst_matter_setup()
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    c = S("c")

    brst = brst_transformation(group=su3, ghost=ghost)

    phi_terms = _apply_single(brst, phi(c))
    assert len(phi_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in phi_terms[0].fields) == (
        ("c", False, {"color_adj": S("a_brst_SU3_2")}),
        ("Phi", False, {"color_fund": S("c_brst_SU3_1")}),
    )
    assert _canon(phi_terms[0].coupling) == _canon(
        Expression.I
        * S("gS")
        * gauge_generator(S("a_brst_SU3_2"), c, S("c_brst_SU3_1"))
    )

    phibar_terms = _apply_single(brst_transformation(group=su3, ghost=ghost), phi.bar(c))
    assert len(phibar_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in phibar_terms[0].fields) == (
        ("c", False, {"color_adj": S("a_brst_SU3_2")}),
        ("Phi", True, {"color_fund": S("c_brst_SU3_1")}),
    )
    assert _canon(phibar_terms[0].coupling) == _canon(
        -Expression.I
        * S("gS")
        * gauge_generator(S("a_brst_SU3_2"), S("c_brst_SU3_1"), c)
    )


def test_nonabelian_brst_fermion_rule_preserves_spinor_slot_and_conjugation():
    _gluon, ghost, _antighost, _auxiliary, su3 = _su3_brst_matter_setup()
    psi = dirac_field("q", indices=(COLOR_FUND_INDEX,))
    c, i = S("c", "i")

    brst = brst_transformation(group=su3, ghost=ghost)

    psi_terms = _apply_single(brst, psi(c, i))
    assert len(psi_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in psi_terms[0].fields) == (
        ("c", False, {"color_adj": S("a_brst_SU3_2")}),
        ("q", False, {"color_fund": S("c_brst_SU3_1"), "spinor": i}),
    )

    psibar_terms = _apply_single(brst_transformation(group=su3, ghost=ghost), psi.bar(c, i))
    assert len(psibar_terms) == 1
    assert tuple((occ.field.name, occ.conjugated, occ.labels) for occ in psibar_terms[0].fields) == (
        ("c", False, {"color_adj": S("a_brst_SU3_2")}),
        ("q", True, {"color_fund": S("c_brst_SU3_1"), "spinor": i}),
    )


def test_multigroup_matter_brst_contributions_do_not_mix_ghosts():
    gluon, color_ghost, _antighost, _auxiliary, su3 = _su3_brst_matter_setup()
    weak_boson, weak_ghost, _weak_auxiliary, su2 = _su2_brst_matter_setup()
    photon, abelian_ghost, _abelian_auxiliary, u1 = _u1_brst_setup()
    quark = dirac_field(
        "Q",
        indices=(WEAK_FUND_INDEX, COLOR_FUND_INDEX),
        quantum_numbers={"Q": Expression.num(1) / Expression.num(6)},
    )
    w, c, i = S("w", "c", "i")

    su3_terms = _apply_single(
        brst_transformation(group=su3, ghost=color_ghost),
        quark(w, c, i),
    )
    su2_terms = _apply_single(
        brst_transformation(group=su2, ghost=weak_ghost),
        quark(w, c, i),
    )
    u1_terms = _apply_single(
        brst_transformation(group=u1, ghost=abelian_ghost),
        quark(w, c, i),
    )

    assert len(su3_terms) == len(su2_terms) == len(u1_terms) == 1

    su3_ghost, su3_quark = su3_terms[0].fields
    su3_labels = quark.unpack_slot_labels(su3_quark.labels)
    assert su3_ghost.field is color_ghost
    assert su3_labels[0] == w
    assert su3_labels[1] != c
    assert su3_labels[2] == i

    su2_ghost, su2_quark = su2_terms[0].fields
    su2_labels = quark.unpack_slot_labels(su2_quark.labels)
    assert su2_ghost.field is weak_ghost
    assert su2_labels[0] != w
    assert su2_labels[1] == c
    assert su2_labels[2] == i

    u1_ghost, u1_quark = u1_terms[0].fields
    assert u1_ghost.field is abelian_ghost
    assert quark.unpack_slot_labels(u1_quark.labels) == {0: w, 1: c, 2: i}

    assert {term.fields[0].field.name for term in su3_terms + su2_terms + u1_terms} == {
        "c",
        "cA",
        "cW",
    }
    assert all(term.fields[0].field is not weak_ghost for term in su3_terms)
    assert all(term.fields[0].field is not color_ghost for term in su2_terms)
    assert all(term.fields[0].field is not abelian_ghost for term in su3_terms + su2_terms)


def test_brst_graded_leibniz_signs_on_mixed_monomials():
    _photon, ghost, auxiliary, u1 = _u1_brst_setup()
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 2})
    neutral_scalar = scalar_field("S", self_conjugate=True, quantum_numbers={"Q": 0})
    neutral_fermion = dirac_field("eta", quantum_numbers={"Q": 0})
    i = S("i")

    brst = brst_transformation(group=u1, ghost=ghost, auxiliary=auxiliary)

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(ghost(), phi(), ghost.bar()),
    )
    results = apply_field_operator_to_term(term, brst)
    assert len(results) == 2
    assert tuple((occ.field.name, occ.conjugated) for occ in results[0].fields) == (
        ("cA", False),
        ("cA", False),
        ("Phi", False),
        ("cA", True),
    )
    assert _canon(results[0].coupling) == _canon(-Expression.I * S("e") * Expression.num(2))
    assert tuple((occ.field.name, occ.conjugated) for occ in results[1].fields) == (
        ("cA", False),
        ("Phi", False),
        ("BAux", False),
    )
    assert _canon(results[1].coupling) == _canon(-Expression.num(1))

    mixed_term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(ghost(), neutral_scalar(), neutral_fermion(i), ghost.bar()),
    )
    mixed_results = apply_field_operator_to_term(mixed_term, brst)
    assert len(mixed_results) == 1
    assert tuple((occ.field.name, occ.conjugated) for occ in mixed_results[0].fields) == (
        ("cA", False),
        ("S", False),
        ("eta", False),
        ("BAux", False),
    )
    assert _canon(mixed_results[0].coupling) == _canon(Expression.num(1))


def test_brst_nilpotency_on_abelian_matter_fields():
    _photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 2})
    psi = dirac_field("psi", quantum_numbers={"Q": -1})
    i = S("i")

    brst = brst_transformation(group=u1, ghost=ghost)

    s2_phi = _single_slot_lagrangian(phi()).apply_operator(brst).apply_operator(brst)
    s2_psi = _single_slot_lagrangian(psi(i)).apply_operator(brst).apply_operator(brst)

    assert _zero(s2_phi.to_symbolica().expand(), field_heads=(ghost, phi))
    assert _zero(s2_psi.to_symbolica().expand(), field_heads=(ghost, psi))


def test_nonabelian_matter_singlet_mass_term_is_brst_invariant():
    _gluon, ghost, _antighost, _auxiliary, su3 = _su3_brst_matter_setup()
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    c = S("c")
    lagrangian = Model(
        gauge_groups=(su3,),
        fields=(phi, _gluon, ghost),
        lagrangian_decl=S("m2") * phi.bar(c) * phi(c),
    ).lagrangian()

    brst = brst_transformation(group=su3, ghost=ghost)

    assert _zero(
        lagrangian.apply_operator(brst).to_symbolica().expand(),
        field_heads=(ghost, phi),
        run_color=True,
    )


def test_abelian_covariant_scalar_kinetic_term_is_brst_invariant():
    from feynpy import CovD

    photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    mu = S("mu")
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 1})
    lagrangian = Model(
        gauge_groups=(u1,),
        fields=(phi, photon, ghost),
        lagrangian_decl=CovD(phi.bar, mu) * CovD(phi, mu),
    ).lagrangian()

    brst = brst_transformation(group=u1, ghost=ghost)

    assert _zero(
        lagrangian.apply_operator(brst).to_symbolica().expand(),
        field_heads=(photon, ghost, phi),
    )


def test_abelian_fermion_kinetic_term_is_brst_invariant():
    from feynpy import CovD, Gamma

    photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    mu = S("mu")
    psi = dirac_field("psi", quantum_numbers={"Q": -1})
    lagrangian = Model(
        gauge_groups=(u1,),
        fields=(psi, photon, ghost),
        lagrangian_decl=Expression.I * psi.bar() * Gamma(mu) * CovD(psi, mu),
    ).lagrangian()

    brst = brst_transformation(group=u1, ghost=ghost)

    assert _zero(
        lagrangian.apply_operator(brst).to_symbolica().expand(),
        field_heads=(photon, ghost, psi),
        run_gamma=True,
    )


def test_abelian_balanced_yukawa_is_brst_invariant():
    photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 1})
    psi = dirac_field("psi", quantum_numbers={"Q": 0})
    chi = dirac_field("chi", quantum_numbers={"Q": 1})
    lagrangian = Model(
        gauge_groups=(u1,),
        fields=(phi, psi, chi, photon, ghost),
        lagrangian_decl=S("y") * psi.bar() * chi() * phi.bar,
    ).lagrangian()

    brst = brst_transformation(group=u1, ghost=ghost)

    assert _zero(
        lagrangian.apply_operator(brst).to_symbolica().expand(),
        field_heads=(ghost, phi, psi, chi),
    )


def test_abelian_scalar_quartic_is_brst_invariant():
    _photon, ghost, _auxiliary, u1 = _u1_brst_setup()
    phi = scalar_field("Phi", self_conjugate=False, quantum_numbers={"Q": 1})
    lagrangian = Model(
        gauge_groups=(u1,),
        fields=(phi, ghost),
        lagrangian_decl=S("lam") * phi.bar * phi * phi.bar * phi,
    ).lagrangian()

    brst = brst_transformation(group=u1, ghost=ghost)

    assert _zero(
        lagrangian.apply_operator(brst).to_symbolica().expand(),
        field_heads=(ghost, phi),
    )
