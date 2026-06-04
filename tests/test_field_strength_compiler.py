"""Tests for the general ``FieldStrength(...)`` compiler.

These exercise the single expansion path that lowers every ``FieldStrength``
factor into its explicit local pieces

``F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + g f^{abc} A^b_mu A^c_nu``

(abelian groups drop the structure-constant piece) and then reuses the ordinary
local-monomial machinery, so arbitrary ``F^n`` operators compile.
"""

from __future__ import annotations

import pytest
from symbolica import Expression, S

from compiler.gauge import compile_covariant_terms
from model import (
    COLOR_FUND_INDEX,
    Field,
    FieldStrength,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    LORENTZ_INDEX,
    Model,
    SPINOR_INDEX,
    StructureConstant,
    T,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    dirac_field,
)
from symbolic.spenso_structures import (
    gamma_matrix,
    gauge_generator,
    structure_constant,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.tensor_canonicalization import canonize_full
from tests.support.builders import make_gluon, make_photon, make_su3, make_u1


def _quarter():
    return Expression.num(1) / Expression.num(4)


def _field_counts(compiled):
    return sorted(len(term.fields) for term in compiled)


def _canon(expr):
    return expr.expand().to_canonical_string()


def _canon_field_strength_rule_diff(expr):
    return _canon(
        canonize_full(
            expr,
            spinor_indices=(S("i1"), S("i2"), S("s2"), S("spinor_decl_1")),
            lorentz_indices=(S("mu3"), S("mu4"), S("mu1_int"), S("mu2_int")),
            color_fund_indices=(S("c1"), S("c2")),
            adjoint_indices=(S("a3"), S("a4")),
            run_color=False,
        )
    )


def _make_weak_group(coupling, gauge_boson_sym, *, name="SU2L"):
    return GaugeGroup(
        name=name,
        abelian=False,
        coupling=coupling,
        gauge_boson=gauge_boson_sym,
        structure_constant=weak_structure_constant,
        representations=(
            GaugeRepresentation(
                index=WEAK_FUND_INDEX,
                generator_builder=weak_gauge_generator,
                name="doublet",
            ),
        ),
    )


def _make_w_field():
    return Field(
        "W",
        spin=1,
        self_conjugate=True,
        symbol=S("W0"),
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )


# ---------------------------------------------------------------------------
# Abelian F^2: only the gauge-boson two-point bilinears, no self-interactions.
# ---------------------------------------------------------------------------
def test_abelian_field_strength_square_emits_only_bilinears():
    mu, nu = S("mu"), S("nu")
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(S("eQED"), photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=-_quarter()
        * FieldStrength(u1, mu, nu) * FieldStrength(u1, mu, nu),
    )

    compiled = compile_covariant_terms(model)
    # d_mu A_nu - d_nu A_mu distributes 2 x 2 = 4 derivative bilinears.
    assert _field_counts(compiled) == [2, 2, 2, 2]
    # No coupling-powered self-interactions appear in an abelian field strength.
    for term in compiled:
        assert "eQED" not in str(term.coupling)


def test_abelian_field_strength_with_adjoint_index_raises():
    mu, nu, a = S("mu"), S("nu"), S("a")
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(S("eQED"), photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=-_quarter()
        * FieldStrength(u1, mu, nu, a) * FieldStrength(u1, mu, nu, a),
    )
    with pytest.raises(ValueError, match="does not take an adjoint index"):
        compile_covariant_terms(model)


# ---------------------------------------------------------------------------
# Non-abelian F^2: 2G + 3G + 4G, with coupling powers g^0 / g^1 / g^2.
# ---------------------------------------------------------------------------
def test_non_abelian_field_strength_square_emits_2g_3g_4g():
    mu, nu, a = S("mu"), S("nu"), S("aC")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=-_quarter()
        * FieldStrength(su3, mu, nu, a) * FieldStrength(su3, mu, nu, a),
    )

    compiled = compile_covariant_terms(model)
    assert _field_counts(compiled) == [2, 2, 2, 2, 3, 3, 3, 3, 4]

    for term in compiled:
        coupling_text = str(term.coupling)
        if len(term.fields) == 2:
            assert "gS" not in coupling_text
        elif len(term.fields) == 3:
            assert "gS" in coupling_text and "gS^2" not in coupling_text
        else:
            assert "gS^2" in coupling_text


def test_non_abelian_field_strength_without_adjoint_index_raises():
    mu, nu = S("mu"), S("nu")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=-_quarter()
        * FieldStrength(su3, mu, nu) * FieldStrength(su3, mu, nu),
    )
    with pytest.raises(ValueError, match="requires an explicit adjoint index"):
        compile_covariant_terms(model)


def test_single_field_strength_open_adjoint_index_raises():
    mu, nu, a = S("mu"), S("nu"), S("a")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=S("c") * FieldStrength(su3, mu, nu, a),
    )
    with pytest.raises(ValueError, match="open .*adjoint"):
        compile_covariant_terms(model)


def test_uncontracted_field_strength_product_raises():
    mu, nu, a, b = S("mu"), S("nu"), S("a"), S("b")
    rho, sigma = S("rho"), S("sigma")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        # a and b are each open (appear once): not a color singlet.
        lagrangian_decl=S("c")
        * FieldStrength(su3, mu, nu, a) * FieldStrength(su3, rho, sigma, b),
    )
    with pytest.raises(ValueError, match="open .*adjoint"):
        compile_covariant_terms(model)


# ---------------------------------------------------------------------------
# Higher operators contracted with explicit color tensors.
# ---------------------------------------------------------------------------
def test_structure_constant_f_cubed_emits_3g_through_6g():
    mu, nu, rho = S("mu"), S("nu"), S("rho")
    a, b, c = S("a"), S("b"), S("c")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=S("c3")
        * StructureConstant(a, b, c)
        * FieldStrength(su3, mu, nu, a)
        * FieldStrength(su3, nu, rho, b)
        * FieldStrength(su3, rho, mu, c),
    )

    compiled = compile_covariant_terms(model)
    # 3 field strengths, each expanding into (2 derivative + 1 g f A A) pieces,
    # so a monomial with k structure-constant pieces has 3 + k gluons (k = 0..3).
    assert len(compiled) == 27
    assert set(_field_counts(compiled)) == {3, 4, 5, 6}


def test_raw_spenso_structure_constant_matches_declared_f_cubed():
    mu, nu, rho = S("mu"), S("nu"), S("rho")
    a, b, c = S("a"), S("b"), S("c")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")

    declared = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=S("c3")
        * StructureConstant(a, b, c)
        * FieldStrength(su3, mu, nu, a)
        * FieldStrength(su3, nu, rho, b)
        * FieldStrength(su3, rho, mu, c),
    )
    raw = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=S("c3")
        * structure_constant(a, b, c)
        * FieldStrength(su3, mu, nu, a)
        * FieldStrength(su3, nu, rho, b)
        * FieldStrength(su3, rho, mu, c),
    )

    declared_compiled = compile_covariant_terms(declared)
    raw_compiled = compile_covariant_terms(raw)

    assert len(raw_compiled) == len(declared_compiled) == 27
    assert _field_counts(raw_compiled) == _field_counts(declared_compiled)
    assert _canon(
        raw.lagrangian().feynman_rule(gluon, gluon, gluon, include_delta=False)
    ) == _canon(
        declared.lagrangian().feynman_rule(gluon, gluon, gluon, include_delta=False)
    )


def test_color_contracted_f_to_the_fourth_emits_4g_through_8g():
    mu, nu, rho, sigma = S("mu"), S("nu"), S("rho"), S("sigma")
    a, b = S("a"), S("b")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=S("lam")
        * FieldStrength(su3, mu, nu, a) * FieldStrength(su3, mu, nu, a)
        * FieldStrength(su3, rho, sigma, b) * FieldStrength(su3, rho, sigma, b),
    )

    compiled = compile_covariant_terms(model)
    # 4 field strengths -> 3^4 = 81 distributed monomials, 4..8 gluons.
    assert len(compiled) == 81
    assert set(_field_counts(compiled)) == {4, 5, 6, 7, 8}


def test_raw_spenso_gamma_and_generator_match_declared_chain_with_field_strength():
    mu, nu, a = S("mu"), S("nu"), S("a")
    s1, s2, s3 = S("s1"), S("s2"), S("s3")
    i, j = S("i"), S("j")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    quark = dirac_field(
        "qFS",
        indices=(COLOR_FUND_INDEX,),
        symbol=S("qFS"),
        conjugate_symbol=S("qFSbar"),
    )
    qbar = quark.bar(index_labels={SPINOR_INDEX.kind: s1, COLOR_FUND_INDEX.kind: i})
    q = quark(index_labels={SPINOR_INDEX.kind: s3, COLOR_FUND_INDEX.kind: j})

    declared = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * qbar
        * Gamma(mu)
        * Gamma(nu)
        * T(a)
        * q,
    )
    raw = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * qbar
        * gamma_matrix(s1, s2, mu)
        * gamma_matrix(s2, s3, nu)
        * gauge_generator(a, i, j)
        * q,
    )

    declared_compiled = compile_covariant_terms(declared)
    raw_compiled = compile_covariant_terms(raw)

    assert len(raw_compiled) == len(declared_compiled) == 3
    assert raw.lagrangian().vertex_signatures() == declared.lagrangian().vertex_signatures()
    assert _canon_field_strength_rule_diff(
        raw.lagrangian().feynman_rule(quark.bar, quark, gluon, include_delta=False)
        - declared.lagrangian().feynman_rule(quark.bar, quark, gluon, include_delta=False)
    ) == _canon(Expression.num(0))
    assert _canon_field_strength_rule_diff(
        raw.lagrangian().feynman_rule(quark.bar, quark, gluon, gluon, include_delta=False)
        - declared.lagrangian().feynman_rule(quark.bar, quark, gluon, gluon, include_delta=False)
    ) == _canon(Expression.num(0))


# ---------------------------------------------------------------------------
# Mixed gauge groups factorize per group with the correct structure constants.
# ---------------------------------------------------------------------------
def test_mixed_group_field_strength_product_compiles():
    mu3, nu3 = S("mu3"), S("nu3")
    mu2, nu2 = S("mu2"), S("nu2")
    aC, aW = S("aC"), S("aW")
    gluon = make_gluon(name="G", symbol=S("G0"))
    wfield = _make_w_field()
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    su2 = _make_weak_group(S("g2"), wfield.symbol, name="SU2L")
    model = Model(
        gauge_groups=(su3, su2),
        fields=(gluon, wfield),
        lagrangian_decl=S("kappa")
        * FieldStrength(su3, mu3, nu3, aC) * FieldStrength(su3, mu3, nu3, aC)
        * FieldStrength(su2, mu2, nu2, aW) * FieldStrength(su2, mu2, nu2, aW),
    )

    compiled = compile_covariant_terms(model)
    # Two SU(3) and two SU(2) field strengths -> 3^4 = 81 distributed monomials.
    assert len(compiled) == 81
    assert set(_field_counts(compiled)) == {4, 5, 6, 7, 8}

    # Each compiled monomial mixes only gluons (G) and weak bosons (W).
    for term in compiled:
        names = {occ.field.name for occ in term.fields}
        assert names <= {"G", "W"}
