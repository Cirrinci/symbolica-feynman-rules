from collections import Counter
from fractions import Fraction

from symbolica import Expression, S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    CovariantDerivativeFactor,
    DC,
    DifferentiatedCovariantFactor,
    DifferentiatedOperatorFactor,
    FS,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    LORENTZ_INDEX,
    Metric,
    Model,
    PartialD,
    SPINOR_INDEX,
    StructureConstant,
)
from symbolic.spenso_structures import gauge_generator, lorentz_metric, structure_constant
from symbolic.tensor_canonicalization import canonize_full
from symbolic.vertex_engine import I, pcomp
from tests.support.builders import canon


def _assert_canon_equal(got, expected):
    assert canon(got - expected) == canon(Expression.num(0))


def _assert_tensor_equal(got, expected):
    assert canon(canonize_full(got - expected, run_color=False)) == canon(
        Expression.num(0)
    )


def _u1_scalar_model(lagrangian_decl):
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    scalar = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=S("Phi"),
        conjugate_symbol=S("Phibar"),
        quantum_numbers={"Q": S("q")},
    )
    u1 = GaugeGroup(
        "U1",
        abelian=True,
        coupling=S("g"),
        gauge_boson=photon,
        charge="Q",
    )
    return Model(
        gauge_groups=(u1,),
        fields=(photon, scalar),
        lagrangian_decl=lagrangian_decl(scalar, photon, u1),
    )


def _su3_gauge_model(lagrangian_decl):
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    su3 = GaugeGroup(
        "SU3",
        abelian=False,
        coupling=S("g"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )
    return Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=lagrangian_decl(gluon, su3),
    )


def _su3_quark_model(lagrangian_decl):
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    su3 = GaugeGroup(
        "SU3",
        abelian=False,
        coupling=S("g"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )
    return Model(
        gauge_groups=(su3,),
        fields=(gluon, quark),
        lagrangian_decl=lagrangian_decl(quark, gluon, su3),
    )


def _signature_names(model):
    return {signature.names for signature in model.lagrangian().vertex_signatures()}


def test_partiald_of_covd_uses_unified_operator_factor_with_legacy_compatibility():
    mu = S("mu")
    probe = Field(
        "probe",
        spin=0,
        self_conjugate=False,
        symbol=S("probe"),
        conjugate_symbol=S("probebar"),
    )

    factor = PartialD(DC(probe, mu), mu)
    assert isinstance(factor, DifferentiatedOperatorFactor)
    assert isinstance(factor.operand, CovariantDerivativeFactor)
    assert factor.lorentz_indices == (mu,)

    current_model = _u1_scalar_model(
        lambda phi, _photon, _u1: phi.bar * PartialD(DC(phi, mu), mu)
    )
    legacy_model = _u1_scalar_model(
        lambda phi, _photon, _u1: phi.bar
        * DifferentiatedCovariantFactor(
            covariant_factor=DC(phi, mu),
            lorentz_indices=(mu,),
        )
    )
    current_photon, current_phi = current_model.fields
    legacy_photon, legacy_phi = legacy_model.fields

    _assert_canon_equal(
        current_model.lagrangian().feynman_rule(
            current_phi.bar,
            current_phi,
            include_delta=False,
        ),
        legacy_model.lagrangian().feynman_rule(
            legacy_phi.bar,
            legacy_phi,
            include_delta=False,
        ),
    )
    _assert_canon_equal(
        current_model.lagrangian().feynman_rule(
            current_phi.bar,
            current_phi,
            current_photon,
            include_delta=False,
        ),
        legacy_model.lagrangian().feynman_rule(
            legacy_phi.bar,
            legacy_phi,
            legacy_photon,
            include_delta=False,
        ),
    )


def test_nested_covariant_derivative_of_charged_scalar_compiles_all_branches():
    mu = S("mu")

    model = _u1_scalar_model(
        lambda phi, _photon, _u1: phi.bar * DC(DC(phi, mu), mu)
    )
    lagrangian = model.lagrangian()

    assert _signature_names(model) == {
        ("Phi.bar", "Phi"),
        ("Phi.bar", "Phi", "A"),
        ("Phi.bar", "Phi", "A", "A"),
    }
    assert Counter(len(term.fields) for term in lagrangian.terms) == {
        2: 1,
        3: 3,
        4: 1,
    }


def test_nested_covariant_derivative_of_charged_scalar_has_golden_rules():
    mu = S("mu")

    model = _u1_scalar_model(
        lambda phi, _photon, _u1: phi.bar * DC(DC(phi, mu), mu)
    )
    photon, phi = model.fields[0], model.fields[1]
    lagrangian = model.lagrangian()

    _assert_canon_equal(
        lagrangian.feynman_rule(phi.bar, phi, include_delta=False),
        -I * pcomp(S("q2"), S("mu1_int")) ** 2,
    )
    _assert_canon_equal(
        lagrangian.feynman_rule(phi.bar, phi, photon, include_delta=False),
        -I
        * S("g")
        * S("q")
        * (pcomp(S("q3"), S("mu3")) + 2 * pcomp(S("q2"), S("mu3"))),
    )
    _assert_canon_equal(
        lagrangian.feynman_rule(phi.bar, phi, photon, photon, include_delta=False),
        -2 * I * S("g") ** 2 * S("q") ** 2 * lorentz_metric(S("mu3"), S("mu4")),
    )


def test_partial_derivative_of_nonabelian_field_strength_uses_product_rule():
    mu, nu, a = S("mu"), S("nu"), S("a")

    model = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * PartialD(FS(su3, mu, nu, a), mu)
    )
    lagrangian = model.lagrangian()

    assert _signature_names(model) == {("G", "G"), ("G", "G", "G")}
    assert Counter(len(term.fields) for term in lagrangian.terms) == {2: 2, 3: 2}

    cubic_terms = [term for term in lagrangian.terms if len(term.fields) == 3]
    derivative_targets = {
        tuple(action.target for action in term.derivatives)
        for term in cubic_terms
    }
    assert derivative_targets == {(1,), (2,)}


def test_partial_derivative_of_nonabelian_field_strength_has_golden_cubic_rule():
    mu, nu, a = S("mu"), S("nu"), S("a")

    model = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * PartialD(FS(su3, mu, nu, a), mu)
    )
    gluon = model.fields[0]
    expected = S("g") * structure_constant(S("a1"), S("a2"), S("a3")) * (
        -pcomp(S("q1"), S("mu2")) * lorentz_metric(S("mu1"), S("mu3"))
        - pcomp(S("q2"), S("mu3")) * lorentz_metric(S("mu1"), S("mu2"))
        - pcomp(S("q3"), S("mu1")) * lorentz_metric(S("mu2"), S("mu3"))
        + pcomp(S("q1"), S("mu3")) * lorentz_metric(S("mu1"), S("mu2"))
        + pcomp(S("q2"), S("mu1")) * lorentz_metric(S("mu2"), S("mu3"))
        + pcomp(S("q3"), S("mu2")) * lorentz_metric(S("mu1"), S("mu3"))
    )

    _assert_canon_equal(
        model.lagrangian().feynman_rule(gluon, gluon, gluon, include_delta=False),
        expected,
    )


def test_covariant_derivative_of_nonabelian_field_strength_adds_adjoint_branch():
    mu, nu, a = S("mu"), S("nu"), S("a")

    model = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * DC(FS(su3, mu, nu, a), mu)
    )
    lagrangian = model.lagrangian()

    assert _signature_names(model) == {
        ("G", "G"),
        ("G", "G", "G"),
        ("G", "G", "G", "G"),
    }
    assert Counter(len(term.fields) for term in lagrangian.terms) == {
        2: 2,
        3: 4,
        4: 1,
    }


def test_covariant_derivative_of_nonabelian_field_strength_has_golden_rules():
    mu, nu, a = S("mu"), S("nu"), S("a")

    partial_model = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * PartialD(FS(su3, mu, nu, a), mu)
    )
    covariant_model = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * DC(FS(su3, mu, nu, a), mu)
    )
    gluon = covariant_model.fields[0]

    _assert_canon_equal(
        covariant_model.lagrangian().feynman_rule(
            gluon,
            gluon,
            gluon,
            include_delta=False,
        ),
        3
        * partial_model.lagrangian().feynman_rule(
            gluon,
            gluon,
            gluon,
            include_delta=False,
        ),
    )

    d = S("d")
    expected_four_gluon = 4 * I * S("g") ** 2 * (
        -structure_constant(S("a1"), S("a2"), d)
        * structure_constant(S("a3"), S("a4"), d)
        * lorentz_metric(S("mu1"), S("mu3"))
        * lorentz_metric(S("mu2"), S("mu4"))
        - structure_constant(S("a1"), S("a3"), d)
        * structure_constant(S("a2"), S("a4"), d)
        * lorentz_metric(S("mu1"), S("mu2"))
        * lorentz_metric(S("mu3"), S("mu4"))
        - structure_constant(S("a1"), S("a4"), d)
        * structure_constant(S("a2"), S("a3"), d)
        * lorentz_metric(S("mu1"), S("mu2"))
        * lorentz_metric(S("mu3"), S("mu4"))
        + structure_constant(S("a1"), S("a2"), d)
        * structure_constant(S("a3"), S("a4"), d)
        * lorentz_metric(S("mu1"), S("mu4"))
        * lorentz_metric(S("mu2"), S("mu3"))
        + structure_constant(S("a1"), S("a3"), d)
        * structure_constant(S("a2"), S("a4"), d)
        * lorentz_metric(S("mu1"), S("mu4"))
        * lorentz_metric(S("mu2"), S("mu3"))
        + structure_constant(S("a1"), S("a4"), d)
        * structure_constant(S("a2"), S("a3"), d)
        * lorentz_metric(S("mu1"), S("mu3"))
        * lorentz_metric(S("mu2"), S("mu4"))
    )
    _assert_tensor_equal(
        covariant_model.lagrangian().feynman_rule(
            gluon,
            gluon,
            gluon,
            gluon,
            include_delta=False,
        ),
        expected_four_gluon,
    )


def test_field_strength_covariant_derivative_matches_manual_adjoint_convention():
    mu, nu = S("mu"), S("nu")
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")

    automatic = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * DC(FS(su3, mu, nu, a), mu)
    )
    manual = _su3_gauge_model(
        lambda gluon, su3: gluon(nu, a) * PartialD(FS(su3, mu, nu, a), mu)
        + S("g")
        * StructureConstant(a, b, c)
        * gluon(nu, a)
        * gluon(mu, b)
        * PartialD(gluon(nu, c), mu)
        - S("g")
        * StructureConstant(a, b, c)
        * gluon(nu, a)
        * gluon(mu, b)
        * PartialD(gluon(mu, c), nu)
        + S("g") ** 2
        * StructureConstant(a, b, c)
        * StructureConstant(c, d, e)
        * gluon(nu, a)
        * gluon(mu, b)
        * gluon(mu, d)
        * gluon(nu, e)
    )
    gluon = automatic.fields[0]

    for legs in (
        (gluon, gluon),
        (gluon, gluon, gluon),
        (gluon, gluon, gluon, gluon),
    ):
        _assert_tensor_equal(
            automatic.lagrangian().feynman_rule(
                *legs,
                include_delta=False,
                simplify=False,
            ),
            manual.lagrangian().feynman_rule(
                *legs,
                include_delta=False,
                simplify=False,
            ),
        )


def test_su3_fundamental_nested_covariant_derivative_has_golden_rules():
    mu = S("mu")

    model = _su3_quark_model(
        lambda q, _gluon, _su3: q.bar * DC(DC(q, mu), mu)
    )
    gluon, quark = model.fields[0], model.fields[1]
    lagrangian = model.lagrangian()
    spin_delta = SPINOR_INDEX.representation.g(S("i1"), S("i2")).to_expression()
    color_delta = COLOR_FUND_INDEX.representation.g(S("c1"), S("c2")).to_expression()

    _assert_tensor_equal(
        lagrangian.feynman_rule(quark.bar, quark, include_delta=False),
        -I * pcomp(S("q2"), S("mu1_int")) ** 2 * spin_delta * color_delta,
    )
    _assert_tensor_equal(
        lagrangian.feynman_rule(quark.bar, quark, gluon, include_delta=False),
        -I
        * S("g")
        * (pcomp(S("q3"), S("mu3")) + 2 * pcomp(S("q2"), S("mu3")))
        * gauge_generator(S("a3"), S("c1"), S("c2"))
        * spin_delta,
    )

    mid = S("m")
    expected_four_point = (
        -I
        * S("g") ** 2
        * lorentz_metric(S("mu3"), S("mu4"))
        * spin_delta
        * (
            gauge_generator(S("a3"), S("c1"), mid)
            * gauge_generator(S("a4"), mid, S("c2"))
            + gauge_generator(S("a4"), S("c1"), mid)
            * gauge_generator(S("a3"), mid, S("c2"))
        )
    )
    _assert_tensor_equal(
        lagrangian.feynman_rule(quark.bar, quark, gluon, gluon, include_delta=False),
        expected_four_point,
    )


def test_deeper_mixed_fs_dc_partiald_combinations_compile():
    mu, nu, rho, a = S("mu"), S("nu"), S("rho"), S("a")

    models = [
        _su3_gauge_model(
            lambda gluon, su3: gluon(nu, a)
            * PartialD(DC(FS(su3, mu, nu, a), rho), mu)
        ),
        _su3_gauge_model(
            lambda gluon, su3: gluon(nu, a)
            * DC(PartialD(FS(su3, mu, nu, a), rho), rho)
        ),
        _su3_gauge_model(
            lambda gluon, su3: gluon(nu, a)
            * DC(DC(FS(su3, mu, nu, a), rho), rho)
        ),
    ]

    assert _signature_names(models[0]) == {
        ("G", "G"),
        ("G", "G", "G"),
        ("G", "G", "G", "G"),
    }
    assert _signature_names(models[1]) == _signature_names(models[0])
    assert _signature_names(models[2]) == {
        ("G", "G"),
        ("G", "G", "G"),
        ("G", "G", "G", "G"),
        ("G", "G", "G", "G", "G"),
    }


def test_triple_nested_field_strength_covariant_derivative_term_count_regression():
    mu, nu, rho, sigma, a = S("mu"), S("nu"), S("rho"), S("sigma"), S("a")

    model = _su3_gauge_model(
        lambda gluon, su3: Metric(rho, sigma)
        * gluon(nu, a)
        * DC(DC(DC(FS(su3, mu, nu, a), rho), sigma), mu)
    )
    lagrangian = model.lagrangian()

    assert len(lagrangian.terms) == 67
    assert Counter(len(term.fields) for term in lagrangian.terms) == {
        2: 2,
        3: 22,
        4: 31,
        5: 11,
        6: 1,
    }


def test_covariant_derivative_of_partially_differentiated_covd_is_field_like():
    mu, nu, rho = S("mu"), S("nu"), S("rho")

    model = _u1_scalar_model(
        lambda phi, _photon, _u1: phi.bar
        * DC(PartialD(DC(phi, nu), rho), mu)
    )

    assert _signature_names(model) == {
        ("Phi.bar", "Phi"),
        ("Phi.bar", "Phi", "A"),
        ("Phi.bar", "Phi", "A", "A"),
    }
