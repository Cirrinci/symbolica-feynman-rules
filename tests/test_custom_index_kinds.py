from fractions import Fraction

import pytest

from symbolica import S
from symbolica.community.spenso import Representation

from compiler.spectators import _spectator_identity_factor
from feynpy import (
    CovD,
    Field,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    IndexType,
    LORENTZ_INDEX,
    Model,
    SPINOR_INDEX,
)
from feynpy.lagrangian import DiracKineticTerm
from symbolic.spenso_structures import gauge_generator, structure_constant
from symbolic.vertex_engine import I


def _canon(expr):
    return expr.expand().to_canonical_string()


def _custom_indices():
    spinor = IndexType("Spinor", Representation.bis(4), "spinoR", prefix="i")
    lorentz = IndexType("Lorentz", Representation.mink(4), "lorentzz", prefix="mu")
    color_fund = IndexType("ColorFund", Representation.cof(3), "color_fund", prefix="c")
    color_adj = IndexType("ColorAdj", Representation.coad(8), "color_adj", prefix="a")
    return spinor, lorentz, color_fund, color_adj


def _yukawa_rule(spinor_index):
    alpha = S("alpha")
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(spinor_index,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    model = Model(S("y") * psi.bar(alpha) * psi(alpha) * phi)
    return model.feynman_rule(psi.bar, psi, phi, simplify=True)


def test_custom_spinor_kind_matches_default_stripped_fermion_externals():
    spinor, _, _, _ = _custom_indices()
    default_expr = _yukawa_rule(SPINOR_INDEX)
    custom_expr = _yukawa_rule(spinor)

    assert _canon(custom_expr) == _canon(default_expr)


def test_direct_vertex_factor_path_is_removed():
    with pytest.raises(TypeError, match="unexpected keyword argument 'coupling'"):
        from symbolic.vertex_engine import vertex_factor

        vertex_factor(
            coupling=S("y"),
            alphas=[S("psibar0"), S("psi0"), S("phi0")],
            betas=[S("b1"), S("b2"), S("b3")],
            ps=[S("p1"), S("p2"), S("p3")],
            x=S("x"),
        )


def test_custom_lorentz_kind_is_not_contracted_as_spectator():
    _, lorentz, _, color_adj = _custom_indices()
    custom_gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(lorentz, color_adj),
    )
    default_like_gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, color_adj),
    )

    custom_factor, custom_left, custom_right = _spectator_identity_factor(custom_gluon)
    default_factor, default_left, default_right = _spectator_identity_factor(default_like_gluon)

    assert _canon(custom_factor) == _canon(default_factor)
    assert set(custom_left) == set(default_left) == {1}
    assert set(custom_right) == set(default_right) == {1}


def test_custom_lorentz_kind_covariant_current_compiles():
    spinor, lorentz, color_fund, color_adj = _custom_indices()
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(lorentz, color_adj),
    )
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(spinor, color_fund),
    )
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gs"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=color_fund,
                generator_builder=gauge_generator,
                name="fundamental",
            ),
        ),
    )
    mu, nu = S("mu"), S("nu")
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=I
        * quark.bar
        * Gamma(mu)
        * CovD(quark, mu)
        * quark.bar
        * Gamma(nu)
        * CovD(quark, nu),
    )

    compiled = model.lagrangian()
    assert len(compiled.terms) > 0


def test_custom_spinor_kind_dirac_kinetic_term_compiles():
    spinor, _, color_fund, color_adj = _custom_indices()
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(spinor, color_fund),
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, color_adj),
    )
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gs"),
        gauge_boson=gluon,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=color_fund,
                generator_builder=gauge_generator,
                name="fundamental",
            ),
        ),
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=DiracKineticTerm(field=quark),
    )

    compiled = model.lagrangian()
    assert len(compiled.terms) > 0


def test_custom_kinds_yukawa_feynman_rule_matches_default():
    spinor, _, _, _ = _custom_indices()
    psi_custom = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(spinor,),
    )
    psi_default = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))

    custom_rule = Model(S("y") * psi_custom.bar * psi_custom * phi).feynman_rule(
        psi_custom.bar, psi_custom, phi, simplify=True
    )
    default_rule = Model(S("y") * psi_default.bar * psi_default * phi).feynman_rule(
        psi_default.bar, psi_default, phi, simplify=True
    )

    assert _canon(custom_rule) == _canon(default_rule)
