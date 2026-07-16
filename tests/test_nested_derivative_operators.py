from collections import Counter

from symbolica import S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    DC,
    FS,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    LORENTZ_INDEX,
    Model,
    PartialD,
)
from symbolic.spenso_structures import gauge_generator, structure_constant


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


def _signature_names(model):
    return {signature.names for signature in model.lagrangian().vertex_signatures()}


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
