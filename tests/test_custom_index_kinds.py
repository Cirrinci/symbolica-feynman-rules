from fractions import Fraction

from symbolica import S
from symbolica.community.spenso import Representation

from compiler.spectators import _spectator_identity_factor
from model import (
    CovD,
    Field,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    IndexType,
    Lagrangian,
    LORENTZ_INDEX,
    Model,
    SPINOR_INDEX,
)
from model.lagrangian import DiracKineticTerm
from symbolic.spenso_structures import gauge_generator, structure_constant
from symbolic.vertex_engine import I, simplify_deltas, vertex_factor


def _canon(expr):
    return expr.expand().to_canonical_string()


def _custom_indices():
    spinor = IndexType("Spinor", Representation.bis(4), "spinoR", prefix="i")
    lorentz = IndexType("Lorentz", Representation.mink(4), "lorentzz", prefix="mu")
    color_fund = IndexType("ColorFund", Representation.cof(3), "color_fund", prefix="c")
    color_adj = IndexType("ColorAdj", Representation.coad(8), "color_adj", prefix="a")
    return spinor, lorentz, color_fund, color_adj


def _direct_yukawa_vertex(
    *,
    field_index_types,
    leg_index_types,
    field_index_labels=None,
    leg_index_labels=None,
    field_spinor_indices=None,
    leg_spinor_indices=None,
):
    psibar0, psi0, phi0 = S("psibar0", "psi0", "phi0")
    b1, b2, b3 = S("b1", "b2", "b3")
    p1, p2, p3 = S("p1", "p2", "p3")
    s1, s2, s3 = S("s1", "s2", "s3")
    x = S("x")

    expr = vertex_factor(
        coupling=S("y"),
        alphas=[psibar0, psi0, phi0],
        betas=[b1, b2, b3],
        ps=[p1, p2, p3],
        x=x,
        statistics="fermion",
        field_roles=["psibar", "psi", "scalar"],
        leg_roles=["psibar", "psi", "scalar"],
        field_index_types=field_index_types,
        leg_index_types=leg_index_types,
        field_index_labels=field_index_labels,
        leg_index_labels=leg_index_labels,
        field_spinor_indices=field_spinor_indices,
        leg_spinor_indices=leg_spinor_indices,
        leg_spins=[s1, s2, s3],
        strip_externals=True,
        include_delta=False,
    )
    return simplify_deltas(
        expr,
        species_map={b1: psibar0, b2: psi0, b3: phi0},
    )


def test_custom_spinor_kind_matches_default_stripped_fermion_externals():
    spinor, _, _, _ = _custom_indices()
    alpha, i1, i2 = S("alpha", "i1", "i2")

    default_expr = _direct_yukawa_vertex(
        field_index_types=[(SPINOR_INDEX,), (SPINOR_INDEX,), ()],
        leg_index_types=[(SPINOR_INDEX,), (SPINOR_INDEX,), ()],
        field_index_labels=[
            {SPINOR_INDEX.kind: alpha},
            {SPINOR_INDEX.kind: alpha},
            {},
        ],
        leg_index_labels=[
            {SPINOR_INDEX.kind: i1},
            {SPINOR_INDEX.kind: i2},
            {},
        ],
    )
    custom_expr = _direct_yukawa_vertex(
        field_index_types=[(spinor,), (spinor,), ()],
        leg_index_types=[(spinor,), (spinor,), ()],
        field_index_labels=[
            {spinor.kind: alpha},
            {spinor.kind: alpha},
            {},
        ],
        leg_index_labels=[
            {spinor.kind: i1},
            {spinor.kind: i2},
            {},
        ],
    )

    assert _canon(custom_expr) == _canon(default_expr)


def test_legacy_vertex_factor_spinor_path_matches_explicit_custom_labels():
    spinor, _, _, _ = _custom_indices()
    alpha, i1, i2 = S("alpha", "i1", "i2")

    legacy_expr = _direct_yukawa_vertex(
        field_index_types=[(spinor,), (spinor,), ()],
        leg_index_types=[(spinor,), (spinor,), ()],
        field_spinor_indices=[alpha, alpha, None],
        leg_spinor_indices=[i1, i2, None],
    )
    explicit_expr = _direct_yukawa_vertex(
        field_index_types=[(spinor,), (spinor,), ()],
        leg_index_types=[(spinor,), (spinor,), ()],
        field_index_labels=[
            {spinor.kind: alpha},
            {spinor.kind: alpha},
            {},
        ],
        leg_index_labels=[
            {spinor.kind: i1},
            {spinor.kind: i2},
            {},
        ],
    )

    assert _canon(legacy_expr) == _canon(explicit_expr)


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

    custom_rule = Lagrangian(S("y") * psi_custom.bar * psi_custom * phi).feynman_rule(
        psi_custom.bar, psi_custom, phi, simplify=True
    )
    default_rule = Lagrangian(S("y") * psi_default.bar * psi_default * phi).feynman_rule(
        psi_default.bar, psi_default, phi, simplify=True
    )

    assert _canon(custom_rule) == _canon(default_rule)
