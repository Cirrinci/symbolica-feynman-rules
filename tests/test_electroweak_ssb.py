from __future__ import annotations

from fractions import Fraction

from symbolica import Expression, S

from model import (
    DiagonalYukawaAssignment,
    Field,
    SPINOR_INDEX,
    build_broken_electroweak_sector,
    electroweak_mw,
    electroweak_mz,
    standard_model_higgs_doublet,
)
from symbolic.vertex_engine import Delta, I, bis, pcomp, pi
from symbolic.spenso_structures import lorentz_metric

d = S("d")
g1 = S("g1")
g2 = S("g2")
v = S("v")
ye = S("y_e")

q1, q2, q3 = S("q1", "q2", "q3")
D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)


def _canon(expr):
    return expr.expand().to_canonical_string()


def _relation_coefficients(relation):
    return {
        term.display_name(): _canon(term.coefficient)
        for term in relation.terms
    }


electron = Field(
    "e",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("e0"),
    conjugate_symbol=S("ebar0"),
    indices=(SPINOR_INDEX,),
)

BROKEN = build_broken_electroweak_sector(
    g1=g1,
    g2=g2,
    vev=v,
    higgs_doublet=standard_model_higgs_doublet(),
    yukawas=(DiagonalYukawaAssignment(electron, ye, label="electron Yukawa"),),
)
L = BROKEN.model.lagrangian()


def test_standard_higgs_doublet_uses_su2_doublet_hypercharge_half():
    higgs = standard_model_higgs_doublet()

    assert higgs.indices == (BROKEN.fields.higgs_doublet.indices[0],)
    assert _canon(higgs.quantum_numbers["Y"]) == _canon(Expression.num(1) / Expression.num(2))


def test_higgs_vev_expansion_tracks_higgs_and_goldstones():
    relations = {relation.target: relation for relation in BROKEN.higgs_expansion}

    assert _relation_coefficients(relations["H[1]"]) == {
        "Gp": _canon(Expression.num(1)),
    }
    assert _relation_coefficients(relations["Hdag[1]"]) == {
        "Gp.bar": _canon(Expression.num(1)),
    }
    assert _relation_coefficients(relations["H[2]"]) == {
        "1": _canon(v * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
        "h": _canon((Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
        "G0": _canon(I * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
    }


def test_charged_gauge_mixing_structure():
    relations = {relation.target: relation for relation in BROKEN.charged_mixing}

    assert _relation_coefficients(relations["W+"]) == {
        "W1": _canon((Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
        "W2": _canon(-I * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
    }
    assert _relation_coefficients(relations["W-"]) == {
        "W1": _canon((Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
        "W2": _canon(I * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))),
    }


def test_neutral_gauge_mixing_structure():
    relations = {relation.target: relation for relation in BROKEN.neutral_mixing}
    root = (g1**2 + g2**2) ** (Expression.num(1) / Expression.num(2))

    assert _relation_coefficients(relations["Z"]) == {
        "W3": _canon(g2 / root),
        "B": _canon(-g1 / root),
    }
    assert _relation_coefficients(relations["A"]) == {
        "W3": _canon(g1 / root),
        "B": _canon(g2 / root),
    }


def test_broken_higgs_sector_generates_w_and_z_masses_but_no_photon_mass():
    wp = BROKEN.fields.charged_w
    z = BROKEN.fields.z_boson
    photon = BROKEN.fields.photon
    mw_sq = electroweak_mw(g2, v) ** 2
    mz_sq = electroweak_mz(g1, g2, v) ** 2

    got_w = L.feynman_rule(wp.bar, wp, simplify=True)
    expected_w = I * mw_sq * lorentz_metric(S("i1"), S("i2")) * D2
    assert _canon(got_w) == _canon(expected_w)

    got_z = L.feynman_rule(z, z, simplify=True)
    expected_z = I * mz_sq * lorentz_metric(S("i1"), S("i2")) * D2
    assert _canon(got_z) == _canon(expected_z)

    photon_mass_terms = [
        term
        for term in L.terms
        if not term.derivatives
        and tuple((occ.field, occ.conjugated) for occ in term.fields) == ((photon, False), (photon, False))
    ]
    assert photon_mass_terms == []


def test_goldstone_vector_mixing_terms_are_present():
    g0 = BROKEN.fields.goldstone_neutral
    gp = BROKEN.fields.goldstone_charged
    wp = BROKEN.fields.charged_w
    z = BROKEN.fields.z_boson

    got_g0z = L.feynman_rule(g0, z, simplify=True)
    expected_g0z = electroweak_mz(g1, g2, v) * pcomp(q1, S("i1")) * D2
    assert _canon(got_g0z) == _canon(expected_g0z)

    got_gpwm = L.feynman_rule(gp, wp.bar, simplify=True)
    expected_gpwm = electroweak_mw(g2, v) * pcomp(q1, S("i1")) * D2
    assert _canon(got_gpwm) == _canon(expected_gpwm)


def test_broken_phase_hvv_and_hff_vertices_match_expected_structure():
    h = BROKEN.fields.higgs
    wp = BROKEN.fields.charged_w
    z = BROKEN.fields.z_boson
    me = ye * v * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2))

    got_hww = L.feynman_rule(h, wp.bar, wp, simplify=True)
    expected_hww = I * (g2**2) * v * (Expression.num(1) / Expression.num(2)) * lorentz_metric(S("i1"), S("i2")) * D3
    assert _canon(got_hww) == _canon(expected_hww)

    got_hzz = L.feynman_rule(h, z, z, simplify=True)
    expected_hzz = 2 * I * (electroweak_mz(g1, g2, v) ** 2 / v) * lorentz_metric(S("i1"), S("i2")) * D3
    assert _canon(got_hzz) == _canon(expected_hzz)

    got_hff = L.feynman_rule(electron.bar, electron, h, simplify=True)
    expected_hff = -I * ye * (Expression.num(1) / Expression.num(2)) ** (Expression.num(1) / Expression.num(2)) * bis.g(S("i1"), S("i2")).to_expression() * D3
    assert _canon(got_hff) == _canon(expected_hff)

    got_ff = L.feynman_rule(electron.bar, electron, simplify=True)
    expected_ff = -I * me * bis.g(S("i1"), S("i2")).to_expression() * D2
    assert _canon(got_ff) == _canon(expected_ff)
