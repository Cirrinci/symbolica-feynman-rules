from __future__ import annotations

from fractions import Fraction

from symbolica import Expression, S

from compiler.gauge import compile_covariant_terms
from lagrangian.operators import scalar_gauge_contact
from model import (
    CovD,
    Field,
    FieldStrength,
    GaugeFixing,
    GaugeGroup,
    GaugeRepresentation,
    GhostLagrangian,
    Gamma,
    LORENTZ_INDEX,
    Lagrangian,
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
mu, nu = S("mu", "nu")

q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)

g2_sym = S("g2")
g1_sym = S("g1")
xi_W = S("xiW")

yL = S("yL")
yH = S("yH")

WField = Field(
    "W",
    spin=1,
    self_conjugate=True,
    symbol=S("W0"),
    indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
)

GhostWField = Field(
    "ghW",
    spin=0,
    kind="ghost",
    self_conjugate=False,
    symbol=S("ghW0"),
    conjugate_symbol=S("ghWbar0"),
    indices=(WEAK_ADJ_INDEX,),
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
)

HDoublet = Field(
    "H",
    spin=0,
    self_conjugate=False,
    symbol=S("H0"),
    conjugate_symbol=S("Hdag0"),
    indices=(WEAK_FUND_INDEX,),
)

LYDoublet = Field(
    "LY",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("LY0"),
    conjugate_symbol=S("LYbar0"),
    indices=(SPINOR_INDEX, WEAK_FUND_INDEX),
    quantum_numbers={"Y": yL},
)

HYDoublet = Field(
    "HY",
    spin=0,
    self_conjugate=False,
    symbol=S("HY0"),
    conjugate_symbol=S("HYdag0"),
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

SU2L_GROUP_GHOST = GaugeGroup(
    name="SU2L",
    abelian=False,
    coupling=g2_sym,
    gauge_boson="W",
    ghost_field="ghW",
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

MODEL_SU2_FERMION = Model(
    name="SU2L-fermion-doublet",
    gauge_groups=(SU2L_GROUP,),
    fields=(LDoublet, WField),
    lagrangian_decl=I * LDoublet.bar * Gamma(mu) * CovD(LDoublet, mu),
)

MODEL_SU2_SCALAR = Model(
    name="SU2L-scalar-doublet",
    gauge_groups=(SU2L_GROUP,),
    fields=(HDoublet, WField),
    lagrangian_decl=CovD(HDoublet.bar, mu) * CovD(HDoublet, mu),
)

MODEL_SU2_YM = Model(
    name="SU2L-Yang-Mills",
    gauge_groups=(SU2L_GROUP,),
    fields=(WField,),
    lagrangian_decl=-(Expression.num(1) / Expression.num(4))
    * FieldStrength(SU2L_GROUP, mu, nu) * FieldStrength(SU2L_GROUP, mu, nu),
)

MODEL_SU2_GAUGE_FIXED = Model(
    name="SU2L-gauge-fixed",
    gauge_groups=(SU2L_GROUP_GHOST,),
    fields=(WField, GhostWField),
    lagrangian_decl=(
        -(Expression.num(1) / Expression.num(4))
        * FieldStrength(SU2L_GROUP_GHOST, mu, nu) * FieldStrength(SU2L_GROUP_GHOST, mu, nu)
        + GaugeFixing(SU2L_GROUP_GHOST, xi=xi_W)
        + GhostLagrangian(SU2L_GROUP_GHOST)
    ),
)

MODEL_SU2_U1_FERMION = Model(
    name="SU2LxU1Y-fermion-doublet",
    gauge_groups=(SU2L_GROUP, U1Y_GROUP),
    fields=(LYDoublet, WField, BField),
    lagrangian_decl=I * LYDoublet.bar * Gamma(mu) * CovD(LYDoublet, mu),
)

MODEL_SU2_U1_SCALAR = Model(
    name="SU2LxU1Y-scalar-doublet",
    gauge_groups=(SU2L_GROUP, U1Y_GROUP),
    fields=(HYDoublet, WField, BField),
    lagrangian_decl=CovD(HYDoublet.bar, mu) * CovD(HYDoublet, mu),
)


def _canonical(expr):
    return expr.expand().to_canonical_string()


def _assert_equal(got, expected):
    assert _canonical(got) == _canonical(expected)


def _assert_nonzero(expr):
    assert _canonical(expr) != Expression.num(0).to_canonical_string()


def _lagrangian_vertex(compiled_terms, *fields):
    return Lagrangian(terms=compiled_terms).feynman_rule(*fields)


def _cross_check(model, *fields):
    got = model.lagrangian().feynman_rule(*fields)
    ref = _lagrangian_vertex(compile_covariant_terms(model), *fields)
    _assert_equal(got, ref)


def test_su2_fermion_doublet_vertex():
    got = MODEL_SU2_FERMION.lagrangian().feynman_rule(
        LDoublet.bar, LDoublet, WField,
    )
    expected = (
        -I * g2_sym
        * gamma_matrix(S("i1"), S("i3"), S("i5"))
        * weak_gauge_generator(S("i6"), S("i2"), S("i4"))
        * D3
    )
    _assert_equal(got, expected)
    _cross_check(MODEL_SU2_FERMION, LDoublet.bar, LDoublet, WField)


def test_su2_scalar_doublet_vertices():
    got_3pt = MODEL_SU2_SCALAR.lagrangian().feynman_rule(
        HDoublet.bar, HDoublet, WField,
    )
    expected_3pt = (
        I * g2_sym
        * weak_gauge_generator(S("i4"), S("i1"), S("i2"))
        * (pcomp(q2, mu) - pcomp(q1, mu))
        * D3
    )
    _assert_equal(got_3pt, expected_3pt)

    got_4pt = MODEL_SU2_SCALAR.lagrangian().feynman_rule(
        HDoublet.bar, HDoublet, WField, WField,
    )
    c_mid = S("w_mid_H_SU2L")
    contact_struct = (
        weak_gauge_generator(S("i4"), S("i1"), c_mid)
        * weak_gauge_generator(S("i6"), c_mid, S("i2"))
        + weak_gauge_generator(S("i6"), S("i1"), c_mid)
        * weak_gauge_generator(S("i4"), c_mid, S("i2"))
    )
    expected_4pt = (
        I * (g2_sym ** 2)
        * scalar_gauge_contact(S("i3"), S("i5"))
        * contact_struct
        * D4
    )
    _assert_equal(got_4pt, expected_4pt)

    _cross_check(MODEL_SU2_SCALAR, HDoublet.bar, HDoublet, WField)
    _cross_check(MODEL_SU2_SCALAR, HDoublet.bar, HDoublet, WField, WField)


def test_su2_yang_mills_vertices():
    _cross_check(MODEL_SU2_YM, WField, WField)
    _assert_nonzero(MODEL_SU2_YM.lagrangian().feynman_rule(WField, WField))

    _cross_check(MODEL_SU2_YM, WField, WField, WField)
    _assert_nonzero(MODEL_SU2_YM.lagrangian().feynman_rule(WField, WField, WField))

    _cross_check(MODEL_SU2_YM, WField, WField, WField, WField)
    _assert_nonzero(MODEL_SU2_YM.lagrangian().feynman_rule(WField, WField, WField, WField))


def test_su2_ghost_vertices():
    _cross_check(MODEL_SU2_GAUGE_FIXED, WField, WField)
    _cross_check(MODEL_SU2_GAUGE_FIXED, WField, WField, WField)
    _cross_check(MODEL_SU2_GAUGE_FIXED, GhostWField.bar, GhostWField)
    _cross_check(MODEL_SU2_GAUGE_FIXED, GhostWField.bar, WField, GhostWField)
    _assert_nonzero(
        MODEL_SU2_GAUGE_FIXED.lagrangian().feynman_rule(
            GhostWField.bar, WField, GhostWField,
        ),
    )


def test_su2_u1_fermion_doublet_vertices():
    _cross_check(MODEL_SU2_U1_FERMION, LYDoublet.bar, LYDoublet, WField)
    got_w = MODEL_SU2_U1_FERMION.lagrangian().feynman_rule(
        LYDoublet.bar, LYDoublet, WField,
    )
    expected_w = (
        -I * g2_sym
        * gamma_matrix(S("i1"), S("i3"), S("i5"))
        * weak_gauge_generator(S("i6"), S("i2"), S("i4"))
        * D3
    )
    _assert_equal(got_w, expected_w)

    _cross_check(MODEL_SU2_U1_FERMION, LYDoublet.bar, LYDoublet, BField)
    got_b = MODEL_SU2_U1_FERMION.lagrangian().feynman_rule(
        LYDoublet.bar, LYDoublet, BField,
    )
    weak_fund_id = WEAK_FUND_INDEX.representation.g(S("i2"), S("i4")).to_expression()
    expected_b = (
        -I * g1_sym * yL
        * gamma_matrix(S("i1"), S("i3"), S("i5"))
        * weak_fund_id
        * D3
    )
    _assert_equal(got_b, expected_b)


def test_su2_u1_scalar_doublet_vertices():
    _cross_check(MODEL_SU2_U1_SCALAR, HYDoublet.bar, HYDoublet, WField)
    _cross_check(MODEL_SU2_U1_SCALAR, HYDoublet.bar, HYDoublet, BField)
    _cross_check(MODEL_SU2_U1_SCALAR, HYDoublet.bar, HYDoublet, WField, WField)
    _cross_check(MODEL_SU2_U1_SCALAR, HYDoublet.bar, HYDoublet, BField, BField)
    _cross_check(MODEL_SU2_U1_SCALAR, HYDoublet.bar, HYDoublet, WField, BField)

    _assert_nonzero(
        MODEL_SU2_U1_SCALAR.lagrangian().feynman_rule(
            HYDoublet.bar, HYDoublet, WField, BField,
        ),
    )
