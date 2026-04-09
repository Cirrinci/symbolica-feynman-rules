import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from gauge_compiler import compile_covariant_terms, with_compiled_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    LORENTZ_INDEX,
    COLOR_ADJ_KIND,
    LORENTZ_KIND,
    Field,
    GaugeFixingTerm,
    GaugeGroup,
    GhostTerm,
    Model,
)
from model_symbolica import Delta, I, pi, simplify_deltas, vertex_factor  # noqa: E402
from operators import gauge_fixing_bilinear_raw, ghost_gauge_raw, ghost_kinetic_raw  # noqa: E402
from spenso_structures import structure_constant  # noqa: E402


def _model_vertex(*, interaction, external_legs, species_map):
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=S("x"),
        d=S("d"),
    )
    return simplify_deltas(expr, species_map=species_map)


def _make_photon(symbol):
    return Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=symbol,
        indices=(LORENTZ_INDEX,),
    )


def _make_gluon(symbol):
    return Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=symbol,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )


def _make_ghost(symbol, conjugate_symbol):
    return Field(
        "ghG",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=(COLOR_ADJ_INDEX,),
    )


def test_compile_abelian_gauge_fixing_term():
    d = S("d")
    p1, p2 = S("p1", "p2")
    b1, b2 = S("b1", "b2")
    mu3, mu4 = S("mu3", "mu4")
    xi_qed = S("xiQED")
    photon_symbol = S("A")

    photon = _make_photon(photon_symbol)
    u1 = GaugeGroup(
        name="U1QED",
        abelian=True,
        coupling=S("eQED"),
        gauge_boson=photon.symbol,
        charge="Q",
    )
    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        gauge_fixing_terms=(GaugeFixingTerm(gauge_group=u1, xi=xi_qed),),
    )

    compiled = compile_covariant_terms(model)
    assert with_compiled_covariant_terms(model).interactions == compiled
    assert len(compiled) == 1

    legs = (
        photon.leg(p1, species=b1, labels={LORENTZ_KIND: mu3}),
        photon.leg(p2, species=b2, labels={LORENTZ_KIND: mu4}),
    )
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map={b1: photon_symbol, b2: photon_symbol},
    )
    rho_left = compiled[0].derivatives[0].lorentz_index
    rho_right = compiled[0].derivatives[1].lorentz_index
    expected = (
        (I / xi_qed)
        * gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, rho_left, rho_right)
        * (2 * pi) ** d
        * Delta(p1 + p2)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_compile_nonabelian_gauge_fixing_term():
    d = S("d")
    p1, p2 = S("p1", "p2")
    b1, b2 = S("b1", "b2")
    mu3, mu4 = S("mu3", "mu4")
    a3, a4 = S("a3", "a4")
    xi_qcd = S("xiQCD")
    gluon_symbol = S("G")

    gluon = _make_gluon(gluon_symbol)
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        gauge_fixing_terms=(GaugeFixingTerm(gauge_group=su3, xi=xi_qcd),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1

    legs = (
        gluon.leg(p1, species=b1, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        gluon.leg(p2, species=b2, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map={b1: gluon_symbol, b2: gluon_symbol},
    )
    rho_left = compiled[0].derivatives[0].lorentz_index
    rho_right = compiled[0].derivatives[1].lorentz_index
    expected = (
        (I / xi_qcd)
        * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        * gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, rho_left, rho_right)
        * (2 * pi) ** d
        * Delta(p1 + p2)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_compile_nonabelian_ghost_terms():
    d = S("d")
    p1, p2, p3 = S("p1", "p2", "p3")
    b1, b2, b3 = S("b1", "b2", "b3")
    mu3 = S("mu3")
    a1, a2, a3 = S("a1", "a2", "a3")
    gS = S("gS")
    gluon_symbol = S("G")
    ghost_symbol = S("ghG")
    antighost_symbol = S("ghGbar")

    gluon = _make_gluon(gluon_symbol)
    ghost = _make_ghost(ghost_symbol, antighost_symbol)
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        ghost_field=ghost.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        ghost_terms=(GhostTerm(gauge_group=su3),),
    )

    compiled = compile_covariant_terms(model)
    assert with_compiled_covariant_terms(model).interactions == compiled
    assert len(compiled) == 2
    bilinear, cubic = compiled

    bilinear_legs = (
        ghost.leg(p1, conjugated=True, species=b1, labels={COLOR_ADJ_KIND: a1}),
        ghost.leg(p2, species=b2, labels={COLOR_ADJ_KIND: a2}),
    )
    got_bilinear = _model_vertex(
        interaction=bilinear,
        external_legs=bilinear_legs,
        species_map={b1: antighost_symbol, b2: ghost_symbol},
    )
    bilinear_mu = bilinear.derivatives[0].lorentz_index
    bilinear_nu = bilinear.derivatives[1].lorentz_index
    expected_bilinear = (
        -I
        * ghost_kinetic_raw(a1, a2, p1, p2, bilinear_mu, bilinear_nu)
        * (2 * pi) ** d
        * Delta(p1 + p2)
    )
    assert got_bilinear.expand().to_canonical_string() == expected_bilinear.expand().to_canonical_string()

    cubic_legs = (
        ghost.leg(p1, conjugated=True, species=b1, labels={COLOR_ADJ_KIND: a1}),
        gluon.leg(p2, species=b2, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a2}),
        ghost.leg(p3, species=b3, labels={COLOR_ADJ_KIND: a3}),
    )
    got_cubic = _model_vertex(
        interaction=cubic,
        external_legs=cubic_legs,
        species_map={b1: antighost_symbol, b2: gluon_symbol, b3: ghost_symbol},
    )
    cubic_rho = cubic.derivatives[0].lorentz_index
    expected_cubic = (
        -gS
        * ghost_gauge_raw(a1, a2, a3, mu3, cubic_rho, p1)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_cubic.expand().to_canonical_string() == expected_cubic.expand().to_canonical_string()


def test_compile_abelian_ghost_term_is_rejected():
    photon = _make_photon(S("A"))
    ghost = Field(
        "ghA",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        symbol=S("ghA"),
        conjugate_symbol=S("ghAbar"),
    )
    u1 = GaugeGroup(
        name="U1QED",
        abelian=True,
        coupling=S("eQED"),
        gauge_boson=photon.symbol,
        ghost_field=ghost.symbol,
        charge="Q",
    )
    model = Model(
        gauge_groups=(u1,),
        fields=(photon, ghost),
        ghost_terms=(GhostTerm(gauge_group=u1),),
    )

    with pytest.raises(ValueError, match=r"only supported for non-abelian gauge groups"):
        compile_covariant_terms(model)


def test_compile_self_conjugate_ghost_field_is_rejected():
    gluon = _make_gluon(S("G"))
    ghost = Field(
        "ghBad",
        spin=0,
        kind="ghost",
        self_conjugate=True,
        symbol=S("ghBad"),
        indices=(COLOR_ADJ_INDEX,),
    )
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        ghost_field=ghost.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        ghost_terms=(GhostTerm(gauge_group=su3),),
    )

    with pytest.raises(ValueError, match=r"non-self-conjugate"):
        compile_covariant_terms(model)
