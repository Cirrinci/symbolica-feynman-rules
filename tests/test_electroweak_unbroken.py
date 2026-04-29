from __future__ import annotations

from fractions import Fraction

from symbolica import S

from compiler.gauge import compile_covariant_terms, expand_cov_der
from lagrangian.operators import scalar_gauge_contact
from model import (
    CovD,
    Field,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    Lagrangian,
    LORENTZ_INDEX,
    Model,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
)
from symbolic.spenso_structures import (
    gamma_matrix,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import Delta, I, pi, pcomp

d = S("d")
mu = S("mu")

q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)

g1_sym = S("g1")
g2_sym = S("g2")
yL = S("yL")
yR = S("yR")
yH = S("yH")

WField = Field(
    "W",
    spin=1,
    self_conjugate=True,
    symbol=S("W0"),
    indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
)

BField = Field(
    "B",
    spin=1,
    self_conjugate=True,
    symbol=S("B0"),
    indices=(LORENTZ_INDEX,),
)

LDoublet = Field(
    "L",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("L0"),
    conjugate_symbol=S("Lbar0"),
    indices=(SPINOR_INDEX, WEAK_FUND_INDEX),
    quantum_numbers={"Y": yL},
)

ERight = Field(
    "ER",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("ER0"),
    conjugate_symbol=S("ERbar0"),
    indices=(SPINOR_INDEX,),
    quantum_numbers={"Y": yR},
)

HDoublet = Field(
    "H",
    spin=0,
    self_conjugate=False,
    symbol=S("H0"),
    conjugate_symbol=S("Hdag0"),
    indices=(WEAK_FUND_INDEX,),
    quantum_numbers={"Y": yH},
)

WEAK_DOUBLET_REP = GaugeRepresentation(
    index=WEAK_FUND_INDEX,
    generator_builder=weak_gauge_generator,
    name="doublet",
)

SU2L_GROUP = GaugeGroup(
    name="SU2L",
    abelian=False,
    coupling=g2_sym,
    gauge_boson="W",
    structure_constant=weak_structure_constant,
    representations=(WEAK_DOUBLET_REP,),
)

U1Y_GROUP = GaugeGroup(
    name="U1Y",
    abelian=True,
    coupling=g1_sym,
    gauge_boson="B",
    charge="Y",
)

MODEL_EW_DOUBLET_FERMION = Model(
    name="EW-unbroken-left-doublet",
    gauge_groups=(SU2L_GROUP, U1Y_GROUP),
    fields=(LDoublet, WField, BField),
    lagrangian_decl=I * LDoublet.bar * Gamma(mu) * CovD(LDoublet, mu),
)

MODEL_EW_SINGLET_FERMION = Model(
    name="EW-unbroken-right-singlet",
    gauge_groups=(SU2L_GROUP, U1Y_GROUP),
    fields=(ERight, WField, BField),
    lagrangian_decl=I * ERight.bar * Gamma(mu) * CovD(ERight, mu),
)

MODEL_EW_HIGGS = Model(
    name="EW-unbroken-higgs",
    gauge_groups=(SU2L_GROUP, U1Y_GROUP),
    fields=(HDoublet, WField, BField),
    lagrangian_decl=CovD(HDoublet.bar, mu) * CovD(HDoublet, mu),
)


def _canonical(expr):
    return expr.expand().to_canonical_string()


def _assert_equal(got, expected):
    assert _canonical(got) == _canonical(expected)


def _compiled_lagrangian(model: Model) -> Lagrangian:
    return Lagrangian(terms=compile_covariant_terms(model))


def _assert_model_matches_compiled(model: Model, *fields):
    got = model.lagrangian().feynman_rule(*fields)
    ref = _compiled_lagrangian(model).feynman_rule(*fields)
    _assert_equal(got, ref)


def test_expand_cov_der_resolves_both_electroweak_groups():
    expanded = expand_cov_der(MODEL_EW_DOUBLET_FERMION, CovD(LDoublet, S("mu_decl")))

    assert expanded.field is LDoublet
    assert tuple(piece.metadata.gauge_group.name for piece in expanded.gauge_current_pieces) == (
        "SU2L",
        "U1Y",
    )
    assert tuple(piece.active_slot for piece in expanded.gauge_current_pieces) == (1, None)


def test_unbroken_electroweak_doublet_w_current():
    compiled = compile_covariant_terms(MODEL_EW_DOUBLET_FERMION)
    assert len(compiled) == 3

    got_w = MODEL_EW_DOUBLET_FERMION.lagrangian().feynman_rule(
        LDoublet.bar,
        LDoublet,
        WField,
    )
    expected_w = (
        -I
        * g2_sym
        * gamma_matrix(S("i1"), S("i3"), S("i5"))
        * weak_gauge_generator(S("i6"), S("i2"), S("i4"))
        * D3
    )
    _assert_equal(got_w, expected_w)
    _assert_model_matches_compiled(MODEL_EW_DOUBLET_FERMION, LDoublet.bar, LDoublet, WField)


def test_unbroken_electroweak_doublet_b_current():
    got_b = MODEL_EW_DOUBLET_FERMION.lagrangian().feynman_rule(
        LDoublet.bar,
        LDoublet,
        BField,
    )
    expected_b = (
        -I
        * g1_sym
        * yL
        * gamma_matrix(S("i1"), S("i3"), S("i5"))
        * WEAK_FUND_INDEX.representation.g(S("i2"), S("i4")).to_expression()
        * D3
    )
    _assert_equal(got_b, expected_b)
    _assert_model_matches_compiled(MODEL_EW_DOUBLET_FERMION, LDoublet.bar, LDoublet, BField)


def test_unbroken_electroweak_singlet_b_current():
    compiled = compile_covariant_terms(MODEL_EW_SINGLET_FERMION)
    assert len(compiled) == 2

    got_b = MODEL_EW_SINGLET_FERMION.lagrangian().feynman_rule(
        ERight.bar,
        ERight,
        BField,
    )
    expected_b = (
        -I
        * g1_sym
        * yR
        * gamma_matrix(S("i1"), S("i2"), S("i3"))
        * D3
    )
    _assert_equal(got_b, expected_b)
    _assert_model_matches_compiled(MODEL_EW_SINGLET_FERMION, ERight.bar, ERight, BField)


def test_unbroken_electroweak_higgs_w_current():
    compiled = compile_covariant_terms(MODEL_EW_HIGGS)
    assert len(compiled) == 9

    got = MODEL_EW_HIGGS.lagrangian().feynman_rule(
        HDoublet.bar,
        HDoublet,
        WField,
    )
    expected = (
        I
        * g2_sym
        * weak_gauge_generator(S("i4"), S("i1"), S("i2"))
        * (pcomp(q2, mu) - pcomp(q1, mu))
        * D3
    )
    _assert_equal(got, expected)
    _assert_model_matches_compiled(MODEL_EW_HIGGS, HDoublet.bar, HDoublet, WField)


def test_unbroken_electroweak_higgs_b_current():
    got = MODEL_EW_HIGGS.lagrangian().feynman_rule(
        HDoublet.bar,
        HDoublet,
        BField,
    )
    expected = (
        I
        * g1_sym
        * yH
        * WEAK_FUND_INDEX.representation.g(S("i1"), S("i2")).to_expression()
        * (pcomp(q2, mu) - pcomp(q1, mu))
        * D3
    )
    _assert_equal(got, expected)
    _assert_model_matches_compiled(MODEL_EW_HIGGS, HDoublet.bar, HDoublet, BField)


def test_unbroken_electroweak_higgs_ww_contact():
    got = MODEL_EW_HIGGS.lagrangian().feynman_rule(
        HDoublet.bar,
        HDoublet,
        WField,
        WField,
    )
    c_mid = S("w_mid_H_SU2L")
    contact_struct = (
        weak_gauge_generator(S("i4"), S("i1"), c_mid)
        * weak_gauge_generator(S("i6"), c_mid, S("i2"))
        + weak_gauge_generator(S("i6"), S("i1"), c_mid)
        * weak_gauge_generator(S("i4"), c_mid, S("i2"))
    )
    expected = (
        I
        * (g2_sym ** 2)
        * scalar_gauge_contact(S("i3"), S("i5"))
        * contact_struct
        * D4
    )
    _assert_equal(got, expected)
    _assert_model_matches_compiled(MODEL_EW_HIGGS, HDoublet.bar, HDoublet, WField, WField)


def test_unbroken_electroweak_higgs_bb_contact():
    got = MODEL_EW_HIGGS.lagrangian().feynman_rule(
        HDoublet.bar,
        HDoublet,
        BField,
        BField,
    )
    expected = (
        2
        * I
        * (g1_sym ** 2)
        * (yH ** 2)
        * WEAK_FUND_INDEX.representation.g(S("i1"), S("i2")).to_expression()
        * scalar_gauge_contact(S("i3"), S("i4"))
        * D4
    )
    _assert_equal(got, expected)
    _assert_model_matches_compiled(MODEL_EW_HIGGS, HDoublet.bar, HDoublet, BField, BField)


def test_unbroken_electroweak_higgs_wb_contact():
    got = MODEL_EW_HIGGS.lagrangian().feynman_rule(
        HDoublet.bar,
        HDoublet,
        WField,
        BField,
    )
    expected = (
        2
        * I
        * g2_sym
        * g1_sym
        * yH
        * weak_gauge_generator(S("i4"), S("i1"), S("i2"))
        * scalar_gauge_contact(S("i3"), S("i5"))
        * D4
    )
    _assert_equal(got, expected)
    _assert_model_matches_compiled(MODEL_EW_HIGGS, HDoublet.bar, HDoublet, WField, BField)
