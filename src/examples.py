"""
Vertex-factor examples and tests.

Covers both the direct parallel-list API and the FeynRules-style model layer.
"""

import argparse
from fractions import Fraction

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
    UF,
    UbarF,
    delta,
    bis,
    Delta,
    Dot,
    pcomp,
    vertex_factor,
    simplify_deltas,
    simplify_spinor_indices,
    infer_derivative_targets,
    compact_vertex_sum_form,
    compact_sum_notation,
)
from spenso_structures import (
    SPINOR_KIND,
    LORENTZ_KIND,
    COLOR_FUND_KIND,
    COLOR_ADJ_KIND,
    gamma_lowered_matrix,
    gamma_matrix,
    gamma5_matrix,
    gauge_generator,
    lorentz_metric,
    simplify_gamma_chain,
)
from model import (
    Field,
    InteractionTerm,
    DerivativeAction,
    SPINOR_INDEX,
    LORENTZ_INDEX,
    COLOR_FUND_INDEX,
    COLOR_ADJ_INDEX,
)


# ---------------------------------------------------------------------------
# Common symbols
# ---------------------------------------------------------------------------

x = S("x")
d = S("d")

p1, p2, p3, p4, p5, p6 = S("p1", "p2", "p3", "p4", "p5", "p6")
b1, b2, b3, b4, b5, b6 = S("b1", "b2", "b3", "b4", "b5", "b6")

phi0 = S("phi0")
chi0 = S("chi0")
phiC0 = S("phiC0")
phiCdag0 = S("phiCdag0")
psibar0, psi0 = S("psibar0", "psi0")
A0 = S("A0")
G0 = S("G0")

mu, nu = S("mu", "nu")
mu3, mu4 = S("mu3", "mu4")

lam4 = S("lam4")
lam6 = S("lam6")
g_sym = S("g")
gD = S("gD")
gD2 = S("gD2")
gijk = S("gijk")
g1 = S("g1")
g2 = S("g2")
yF = S("yF")
gV = S("gV")
gS = S("gS")
gPhiA = S("gPhiA")
gPhiAA = S("gPhiAA")
g4F = S("g4F")
g_psi4 = S("g_psi4")
gJJ = S("gJJ")
lamC = S("lamC")

alpha_s, beta_s = S("alpha_s", "beta_s")
a_bar, a_psi, b_bar, b_psi = S("a_bar", "a_psi", "b_bar", "b_psi")
i_psi_bar, i_psi = S("i_psi_bar", "i_psi")
i1, i2, i3, i4 = S("i1", "i2", "i3", "i4")
s1, s2, s3, s4 = S("s1", "s2", "s3", "s4")
idx_i, idx_j, idx_k = S("i", "j", "k")

i_bar_q, i_psi_q = S("i_bar_q", "i_psi_q")
c_bar_q, c_psi_q, a_g = S("c_bar_q", "c_psi_q", "a_g")
c1, c2, a3 = S("c1", "c2", "a3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(got, expected, label):
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


def _print_demo_header(title):
    print(f"# === {title} ===\n")


def _print_vertex_block(
    title,
    *,
    description=None,
    vertex=None,
    compact_override=None,
    sum_notation=None,
    interpretation=None,
    error=None,
):
    _print_demo_header(title)
    if description:
        print(description)
    print()

    if error is not None:
        print(error)
    elif vertex is not None:
        print("Vertex:")
        print(vertex)

    if compact_override is not None:
        print()
        print("Compact override:")
        print(compact_override)

    if sum_notation is not None:
        print("Sum notation:")
        print(sum_notation)

    if interpretation is not None:
        print()
        print(f"Interpretation: {interpretation}")

    print()


def _direct_vertex(*, species_map=None, simplify_gamma=False, **kwargs):
    if "include_delta" not in kwargs:
        kwargs["include_delta"] = False
    expr = vertex_factor(x=x, d=d, **kwargs)
    expr = simplify_deltas(expr, species_map=species_map)
    q_ = S("q_")
    x_ = S("x_")
    expr = expr.replace(Expression.EXP(-I * Dot(q_, x_)), 1)
    if simplify_gamma:
        expr = simplify_gamma_chain(expr)
    return expr


def _model_demo_vertex(*, interaction, external_legs, species_map=None, simplify_gamma=False, strip_externals=True):
    expr = _model_vertex(
        interaction=interaction,
        external_legs=external_legs,
        strip_externals=strip_externals,
        include_delta=False,
        species_map=species_map,
    )
    q_ = S("q_")
    x_ = S("x_")
    expr = expr.replace(Expression.EXP(-I * Dot(q_, x_)), 1)
    if simplify_gamma:
        expr = simplify_gamma_chain(expr)
    return expr


# ===================================================================
# DIRECT-API interaction definitions (parallel lists, old style)
# ===================================================================

L_phi4 = dict(
    coupling=lam4,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

L_phi6 = dict(
    coupling=lam6,
    alphas=[phi0] * 6,
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

L_phi2chi2 = dict(
    coupling=g_sym,
    alphas=[phi0, phi0, chi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

L_phiCdag_phiC = dict(
    coupling=lamC,
    alphas=[phiCdag0, phiC0],
    betas=[b1, b2],
    ps=[p1, p2],
)

deriv_indices, deriv_targets = infer_derivative_targets([(0, [mu]), (1, [nu])])
L_deriv = dict(
    coupling=gD,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices,
    derivative_targets=deriv_targets,
)

deriv_indices2, deriv_targets2 = infer_derivative_targets([(0, [mu]), (1, [mu])])
L_deriv2 = dict(
    coupling=gD2,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2,
    derivative_targets=deriv_targets2,
)

L_multi = dict(
    coupling=gijk(idx_i, idx_j, idx_k),
    alphas=[idx_i, idx_i, idx_j, idx_j, idx_k, idx_k],
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

L_yukawa = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None],
    leg_spins=[s1, s2, s3],
)

L_vec_current = dict(
    coupling=gV * gamma_matrix(i_psi_bar, i_psi, mu),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

L_axial_current = dict(
    coupling=gV * gamma_matrix(i_psi_bar, alpha_s, mu) * gamma5_matrix(alpha_s, i_psi),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

L_4fermion = dict(
    coupling=g4F,
    alphas=[psi0, psibar0, psi0, psibar0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psi", "psibar", "psi", "psibar"],
    leg_roles=["psi", "psibar", "psi", "psibar"],
)

L_psibar_psi_sq = dict(
    coupling=-g_psi4 / Expression.num(2),
    alphas=[psibar0, psi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "psibar", "psi"],
    leg_roles=["psibar", "psi", "psibar", "psi"],
    field_spinor_indices=[alpha_s, alpha_s, beta_s, beta_s],
    leg_spins=[s1, s2, s3, s4],
)

L_psibar_psi_sq_spinor = dict(
    **L_psibar_psi_sq,
    leg_spinor_indices=[i1, i2, i3, i4],
)

L_current_current = dict(
    coupling=gJJ * gamma_matrix(a_bar, a_psi, mu) * gamma_lowered_matrix(b_bar, b_psi, mu),
    alphas=[psibar0, psi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "psibar", "psi"],
    leg_roles=["psibar", "psi", "psibar", "psi"],
    field_spinor_indices=[a_bar, a_psi, b_bar, b_psi],
    leg_spins=[s1, s2, s3, s4],
)

L_quark_gluon = dict(
    coupling=gS * gamma_matrix(i_bar_q, i_psi_q, mu) * gauge_generator(a_g, c_bar_q, c_psi_q),
    alphas=[psibar0, psi0, G0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_index_labels=[
        {SPINOR_KIND: i_bar_q, COLOR_FUND_KIND: c_bar_q},
        {SPINOR_KIND: i_psi_q, COLOR_FUND_KIND: c_psi_q},
        {LORENTZ_KIND: mu, COLOR_ADJ_KIND: a_g},
    ],
    leg_index_labels=[
        {SPINOR_KIND: i1, COLOR_FUND_KIND: c1},
        {SPINOR_KIND: i2, COLOR_FUND_KIND: c2},
        {LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3},
    ],
    leg_spins=[s1, s2, s3],
)

L_complex_scalar_current_phi = dict(
    coupling=gPhiA,
    alphas=[phiCdag0, phiC0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    derivative_indices=[mu],
    derivative_targets=[1],
)

L_complex_scalar_current_phidag = dict(
    coupling=-gPhiA,
    alphas=[phiCdag0, phiC0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    derivative_indices=[mu],
    derivative_targets=[0],
)

L_complex_scalar_contact = dict(
    coupling=gPhiAA * lorentz_metric(mu, nu),
    alphas=[phiCdag0, phiC0, A0, A0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

_MIX_BASE = dict(
    alphas=[psibar0, psi0, phi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None, None],
    leg_spins=[s1, s2, s3, s4],
)

L_mix_dpsibar = dict(**_MIX_BASE, coupling=yF, derivative_indices=[mu], derivative_targets=[0])
L_mix_dpsi = dict(**_MIX_BASE, coupling=yF, derivative_indices=[nu], derivative_targets=[1])
L_mix_dphi_dchi = dict(**_MIX_BASE, coupling=yF, derivative_indices=[mu, nu], derivative_targets=[2, 3])

deriv_indices_l1, deriv_targets_l1 = infer_derivative_targets([(2, [mu, mu])])
L_double_deriv_phi_chi = dict(
    **_MIX_BASE,
    coupling=g1,
    derivative_indices=deriv_indices_l1,
    derivative_targets=deriv_targets_l1,
)

deriv_indices_l2, deriv_targets_l2 = infer_derivative_targets([(2, [mu, nu]), (3, [mu, nu])])
L_double_deriv_phi_phi = dict(
    coupling=g2,
    alphas=[psibar0, psi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices_l2,
    derivative_targets=deriv_targets_l2,
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None, None],
    leg_spins=[s1, s2, s3, s4],
)

COMPACT_DERIV = compact_vertex_sum_form(
    coupling=gD, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices, derivative_targets=deriv_targets, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
)
COMPACT_DERIV2 = compact_vertex_sum_form(
    coupling=gD2, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2, derivative_targets=deriv_targets2, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
)


# ===================================================================
# MODEL-LAYER field declarations (FeynRules style)
# ===================================================================

PhiField = Field("Phi", spin=0, self_conjugate=True, symbol=phi0)
ChiField = Field("Chi", spin=0, self_conjugate=True, symbol=chi0)
PhiCField = Field("PhiC", spin=0, self_conjugate=False, symbol=phiC0, conjugate_symbol=phiCdag0)
PsiField = Field("Psi", spin=Fraction(1, 2), self_conjugate=False, symbol=psi0, conjugate_symbol=psibar0, indices=(SPINOR_INDEX,))
GaugeField = Field("A", spin=1, self_conjugate=True, symbol=A0, indices=(LORENTZ_INDEX,))
QuarkField = Field("q", spin=Fraction(1, 2), self_conjugate=False, symbol=psi0, conjugate_symbol=psibar0, indices=(SPINOR_INDEX, COLOR_FUND_INDEX))
GluonField = Field("G", spin=1, self_conjugate=True, symbol=G0, indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX))


# ===================================================================
# MODEL-LAYER interaction terms
# ===================================================================

TERM_phi4 = InteractionTerm(
    coupling=lam4,
    fields=tuple(PhiField.occurrence() for _ in range(4)),
    label="lam4 * phi^4",
)
LEGS_phi4 = tuple(PhiField.leg(p, species=b) for p, b in [(p1, b1), (p2, b2), (p3, b3), (p4, b4)])

TERM_phi2chi2 = InteractionTerm(
    coupling=g_sym,
    fields=(
        PhiField.occurrence(), PhiField.occurrence(),
        ChiField.occurrence(), ChiField.occurrence(),
    ),
    label="g * phi^2 chi^2",
)
LEGS_phi2chi2 = (
    PhiField.leg(p1, species=b1), PhiField.leg(p2, species=b2),
    ChiField.leg(p3, species=b3), ChiField.leg(p4, species=b4),
)

TERM_phiCdag_phiC = InteractionTerm(
    coupling=lamC,
    fields=(PhiCField.occurrence(conjugated=True), PhiCField.occurrence()),
    label="lamC * phi^dag phi",
)
LEGS_phiCdag_phiC = (
    PhiCField.leg(p1, conjugated=True, species=b1),
    PhiCField.leg(p2, species=b2),
)

TERM_yukawa = InteractionTerm(
    coupling=yF,
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: alpha_s}),
        PsiField.occurrence(labels={SPINOR_KIND: alpha_s}),
        PhiField.occurrence(),
    ),
    label="yF * psibar psi phi",
)
LEGS_yukawa = (
    PsiField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1}),
    PsiField.leg(p2, spin=s2, labels={SPINOR_KIND: i2}),
    PhiField.leg(p3),
)
LEGS_yukawa_matrix = (
    PsiField.leg(p1, conjugated=True, species=b1, spin=s1),
    PsiField.leg(p2, species=b2, spin=s2),
    PhiField.leg(p3, species=b3),
)

TERM_vec_current = InteractionTerm(
    coupling=gV * gamma_matrix(i_psi_bar, i_psi, mu),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: i_psi_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: i_psi}),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
    ),
    label="gV * psibar gamma^mu psi A_mu",
)

TERM_axial_current = InteractionTerm(
    coupling=gV * gamma_matrix(i_psi_bar, alpha_s, mu) * gamma5_matrix(alpha_s, i_psi),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: i_psi_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: i_psi}),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
    ),
    label="gV * psibar gamma^mu gamma5 psi A_mu",
)

LEGS_vec_current = (
    PsiField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1}),
    PsiField.leg(p2, spin=s2, labels={SPINOR_KIND: i2}),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}),
)

TERM_psibar_psi_sq = InteractionTerm(
    coupling=-g_psi4 / Expression.num(2),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: alpha_s}),
        PsiField.occurrence(labels={SPINOR_KIND: alpha_s}),
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: beta_s}),
        PsiField.occurrence(labels={SPINOR_KIND: beta_s}),
    ),
    label="-(g/2)(psibar psi)^2",
)
LEGS_fermion4 = (
    PsiField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1}),
    PsiField.leg(p2, spin=s2, labels={SPINOR_KIND: i2}),
    PsiField.leg(p3, conjugated=True, spin=s3, labels={SPINOR_KIND: i3}),
    PsiField.leg(p4, spin=s4, labels={SPINOR_KIND: i4}),
)
LEGS_fermion4_matrix = (
    PsiField.leg(p1, conjugated=True, species=b1, spin=s1),
    PsiField.leg(p2, species=b2, spin=s2),
    PsiField.leg(p3, conjugated=True, species=b3, spin=s3),
    PsiField.leg(p4, species=b4, spin=s4),
)

TERM_current_current = InteractionTerm(
    coupling=gJJ * gamma_matrix(a_bar, a_psi, mu) * gamma_lowered_matrix(b_bar, b_psi, mu),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: a_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: a_psi}),
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: b_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: b_psi}),
    ),
    label="gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)",
)

TERM_quark_gluon = InteractionTerm(
    coupling=gS * gamma_matrix(i_bar_q, i_psi_q, mu) * gauge_generator(a_g, c_bar_q, c_psi_q),
    fields=(
        QuarkField.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar_q, COLOR_FUND_KIND: c_bar_q}),
        QuarkField.occurrence(labels={SPINOR_KIND: i_psi_q, COLOR_FUND_KIND: c_psi_q}),
        GluonField.occurrence(labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a_g}),
    ),
    label="gS * qbar gamma^mu T^a q G^a_mu",
)
LEGS_quark_gluon = (
    QuarkField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1, COLOR_FUND_KIND: c1}),
    QuarkField.leg(p2, spin=s2, labels={SPINOR_KIND: i2, COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
)

TERM_complex_scalar_current_phi = InteractionTerm(
    coupling=gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
    ),
    derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
    label="gPhiA * A_mu * phi^dag * d^mu phi",
)
TERM_complex_scalar_current_phidag = InteractionTerm(
    coupling=-gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
    ),
    derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
    label="-gPhiA * A_mu * d^mu phi^dag * phi",
)
LEGS_complex_scalar_current = (
    PhiCField.leg(p1, conjugated=True, species=b1),
    PhiCField.leg(p2, species=b2),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}, species=b3),
)

TERM_complex_scalar_contact = InteractionTerm(
    coupling=gPhiAA * lorentz_metric(mu, nu),
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
        GaugeField.occurrence(labels={LORENTZ_KIND: nu}),
    ),
    label="gPhiAA * A_mu A^mu phi^dag phi",
)
LEGS_complex_scalar_contact = (
    PhiCField.leg(p1, conjugated=True, species=b1),
    PhiCField.leg(p2, species=b2),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}, species=b3),
    GaugeField.leg(p4, labels={LORENTZ_KIND: mu4}, species=b4),
)


# ===================================================================
# Helper: vertex from model-layer objects
# ===================================================================

def _model_vertex(
    *,
    interaction,
    external_legs,
    strip_externals=True,
    include_delta=True,
    species_map=None,
):
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=x, d=d,
        strip_externals=strip_externals,
        include_delta=include_delta,
    )
    return simplify_deltas(expr, species_map=species_map)


# ===================================================================
# Demo output (human-readable vertex blocks)
# ===================================================================

def _run_scalar_demo():
    print("# " + "=" * 79)
    print("Demo: scalar\n")
    print("# " + "=" * 79)

    _print_vertex_block(
        "scalar: phi^4",
        description="lam4 * phi^4",
        vertex=_direct_vertex(**L_phi4, species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0}),
    )
    _print_vertex_block(
        "scalar: phi^2 chi^2",
        description="g * phi^2 * chi^2",
        vertex=_direct_vertex(**L_phi2chi2, species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0}),
    )
    _print_vertex_block(
        "scalar: complex scalar bilinear",
        description="lamC * phi^dagger * phi",
        vertex=_direct_vertex(**L_phiCdag_phiC, species_map={b1: phiCdag0, b2: phiC0}),
    )
    _print_vertex_block(
        "scalar: derivative (mu,nu) * phi^4",
        description="gD * (d_mu phi)(d_nu phi) phi phi",
        compact_override=COMPACT_DERIV,
        sum_notation=compact_sum_notation(
            derivative_indices=deriv_indices,
            derivative_targets=deriv_targets,
            n_legs=len(L_deriv["ps"]),
        ),
    )
    _print_vertex_block(
        "scalar: derivative (mu,mu) * phi^4",
        description="gD2 * (d_mu phi)(d_mu phi) phi phi",
        compact_override=COMPACT_DERIV2,
        sum_notation=compact_sum_notation(
            derivative_indices=deriv_indices2,
            derivative_targets=deriv_targets2,
            n_legs=len(L_deriv2["ps"]),
        ),
    )
    _print_vertex_block(
        "scalar: multi-species phi_i^2 phi_j^2 phi_k^2",
        description="gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2",
        vertex=_direct_vertex(
            **L_multi,
            species_map={b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k},
        ),
    )


def _run_fermion_demo():
    print("# " + "=" * 79)
    print("Demo: fermion\n")

    _print_vertex_block(
        "fermion: Yukawa [amputated]",
        description="yF * psibar * psi * phi",
        vertex=_direct_vertex(**L_yukawa, species_map={b1: psibar0, b2: psi0, b3: phi0}),
    )
    _print_vertex_block(
        "fermion: Yukawa [matrix element]",
        description="yF * psibar * psi * phi  [matrix element]",
        vertex=_direct_vertex(
            **L_yukawa,
            species_map={b1: psibar0, b2: psi0, b3: phi0},
            strip_externals=False,
        ),
    )
    _print_vertex_block(
        "fermion: vector current",
        description="gV * psibar gamma^mu psi A_mu",
        vertex=_direct_vertex(**L_vec_current, species_map={b1: psibar0, b2: psi0, b3: A0}),
    )
    _print_vertex_block(
        "fermion: axial current",
        description="gV * psibar gamma^mu gamma5 psi A_mu",
        vertex=_direct_vertex(**L_axial_current, species_map={b1: psibar0, b2: psi0, b3: A0}),
    )

    try:
        vertex_factor(**L_4fermion, x=x, d=d)
    except ValueError:
        _print_vertex_block(
            "fermion: underspecified product diagnostic",
            description="g4F * psi * psibar * psi * psibar  [no spinor contractions]",
            error="rejected: multi-fermion operators need explicit spinor contractions",
        )

    _print_vertex_block(
        "fermion: -(g/2)(psibar psi)^2 [amputated]",
        description="-(g/2)(psibar psi)^2 [amputated]",
        vertex=_direct_vertex(**L_psibar_psi_sq, species_map={b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}),
    )
    _print_vertex_block(
        "fermion: -(g/2)(psibar psi)^2 [matrix element]",
        description="-(g/2)(psibar psi)^2 [matrix element]",
        vertex=_direct_vertex(
            **L_psibar_psi_sq,
            species_map={b1: psibar0, b2: psi0, b3: psibar0, b4: psi0},
            strip_externals=False,
        ),
    )
    _print_vertex_block(
        "fermion: current-current operator",
        description="gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]",
        vertex=_direct_vertex(
            **L_current_current,
            species_map={b1: psibar0, b2: psi0, b3: psibar0, b4: psi0},
            simplify_gamma=True,
        ),
    )


def _run_mixed_demo():
    print("# " + "=" * 79)
    print("Demo: fermion+scalar\n")

    _print_vertex_block(
        "fermion+scalar: mixed derivatives",
        description="yF * (d_mu psibar) * psi * phi * chi",
        vertex=_direct_vertex(**L_mix_dpsibar, species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0}),
    )
    _print_vertex_block(
        "fermion+scalar: mixed derivatives",
        description="yF * psibar * (d_nu psi) * phi * chi",
        vertex=_direct_vertex(**L_mix_dpsi, species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0}),
    )
    _print_vertex_block(
        "fermion+scalar: mixed derivatives",
        description="yF * psibar * psi * (d_mu phi) * (d_nu chi)",
        vertex=_direct_vertex(**L_mix_dphi_dchi, species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0}),
    )
    _print_vertex_block(
        "fermion+scalar: higher derivatives",
        description="g1 * psibar * psi * (d^2 phi) * chi",
        vertex=_direct_vertex(**L_double_deriv_phi_chi, species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0}),
    )
    _print_vertex_block(
        "fermion+scalar: higher derivatives",
        description="g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)",
        vertex=_direct_vertex(**L_double_deriv_phi_phi, species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0}),
    )


def _run_gauge_demo():
    print("# " + "=" * 79)
    print("Demo: gauge-ready\n")

    _print_vertex_block(
        "gauge-ready: non-abelian fermion current",
        description="gS * psibar gamma^mu T^a psi G^a_mu",
        vertex=_direct_vertex(**L_quark_gluon, species_map={b1: psibar0, b2: psi0, b3: G0}),
    )
    _print_vertex_block(
        "gauge-ready: complex scalar current",
        description="gPhiA * A_mu * phi^dagger <-> d^mu phi",
        vertex=(
            _direct_vertex(**L_complex_scalar_current_phi, species_map={b1: phiCdag0, b2: phiC0, b3: A0})
            + _direct_vertex(**L_complex_scalar_current_phidag, species_map={b1: phiCdag0, b2: phiC0, b3: A0})
        ).expand(),
    )
    _print_vertex_block(
        "gauge-ready: complex scalar contact",
        description="gPhiAA * A_mu A^mu phi^dagger phi",
        vertex=_direct_vertex(**L_complex_scalar_contact, species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0}),
    )


def _run_demo_output(suite):
    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()
        _run_mixed_demo()
    if suite in ("gauge", "all"):
        _run_gauge_demo()
    if suite == "model":
        print("# " + "=" * 79)
        print("Demo: model-layer\n")
        _print_vertex_block(
            "model: Yukawa [amputated]",
            description=TERM_yukawa.label,
            vertex=_model_demo_vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        )
        _print_vertex_block(
            "model: quark-gluon",
            description=TERM_quark_gluon.label,
            vertex=_model_demo_vertex(
                interaction=TERM_quark_gluon,
                external_legs=LEGS_quark_gluon,
            ),
        )
    if suite == "cross":
        print("# " + "=" * 79)
        print("Demo: cross-checks\n")
        _print_vertex_block(
            "cross: current-current",
            description="model-layer and direct API should agree after gamma simplification",
            vertex=simplify_gamma_chain(
                _model_demo_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)
            ),
        )
    if suite == "role":
        print("# " + "=" * 79)
        print("Demo: role regressions\n")
        _print_vertex_block(
            "role: complex scalar conjugation filtering",
            description="ROLE_SCALAR_DAG / ROLE_SCALAR should eliminate wrong bosonic permutations",
            vertex=simplify_deltas(
                vertex_factor(
                    coupling=lamC,
                    alphas=[phiCdag0, phiC0],
                    betas=[b1, b2],
                    ps=[p1, p2],
                    field_roles=["scalar_dag", "scalar"],
                    leg_roles=["scalar_dag", "scalar"],
                    x=x,
                    d=d,
                ),
                species_map={b1: phiCdag0, b2: phiC0},
            ),
        )

# ===================================================================
# Direct-API tests (validate the engine is correct after refactor)
# ===================================================================

def _run_scalar_tests():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)

    V = simplify_deltas(vertex_factor(**L_phi4, x=x, d=d), species_map=sm_phi)
    _check(V, 24 * I * lam4 * D4, "phi^4")

    V = simplify_deltas(vertex_factor(**L_phi2chi2, x=x, d=d),
                        species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0})
    _check(V, 4 * I * g_sym * D4, "phi^2 chi^2")

    V = simplify_deltas(vertex_factor(**L_phiCdag_phiC, x=x, d=d),
                        species_map={b1: phiCdag0, b2: phiC0})
    _check(V, I * lamC * (2 * pi) ** d * Delta(p1 + p2), "phi^dag phi")

    V = simplify_deltas(vertex_factor(**L_deriv, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV, "Derivative (mu,nu)")

    V = simplify_deltas(vertex_factor(**L_deriv2, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV2, "Derivative (mu,mu)")

    D6 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5 + p6)
    V_multi = vertex_factor(**L_multi, x=x, d=d)
    expected_multi = 8 * I * gijk(idx_i, idx_j, idx_k) * D6
    sm_base = {b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k}
    _check(simplify_deltas(V_multi, species_map=sm_base), expected_multi, "Multi-species (base)")

    print("\n  Scalar+derivative tests passed.\n")


def _run_fermion_tests():
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    G12 = bis.g(i1, i2).to_expression()

    V = simplify_deltas(vertex_factor(**L_yukawa, x=x, d=d), species_map=sm3)
    _check(V, I * yF * G12 * D3, "Yukawa (amputated)")

    V_full = simplify_deltas(
        vertex_factor(**L_yukawa, x=x, d=d, strip_externals=False), species_map=sm3
    )
    s = V_full.to_canonical_string()
    assert "UbarF" in s and "UF" in s, f"Yukawa unstripped missing UF/UbarF: {V_full}"
    print("  Yukawa (unstripped, has UF/UbarF): PASS")

    V = simplify_deltas(vertex_factor(**L_vec_current, x=x, d=d),
                        species_map={b1: psibar0, b2: psi0, b3: A0})
    _check(V, I * gV * gamma_matrix(i1, i2, mu) * D3, "Vector current")

    V = simplify_deltas(vertex_factor(**L_axial_current, x=x, d=d),
                        species_map={b1: psibar0, b2: psi0, b3: A0})
    _check(
        V,
        I * gV * gamma_matrix(i1, alpha_s, mu) * gamma5_matrix(alpha_s, i2) * D3,
        "Axial current",
    )

    try:
        vertex_factor(**L_4fermion, x=x, d=d)
    except ValueError as exc:
        assert "Multi-fermion" in str(exc)
        print("  Underspecified multi-fermion rejected: PASS")
    else:
        raise AssertionError("Bare psi*psibar*psi*psibar should be rejected")

    V = simplify_deltas(vertex_factor(**L_psibar_psi_sq, x=x, d=d), species_map=sm4)
    expected_sp = (
        -I * g_psi4 * D4
        * (bis.g(i1, i2).to_expression() * bis.g(i3, i4).to_expression()
           - bis.g(i1, i4).to_expression() * bis.g(i3, i2).to_expression())
    )
    _check(V, expected_sp, "(psibar psi)^2 amputated")

    V_full = simplify_deltas(
        vertex_factor(**L_psibar_psi_sq, x=x, d=d, strip_externals=False), species_map=sm4
    )
    s = V_full.to_canonical_string()
    assert s != Expression.num(0).to_canonical_string(), "(psibar psi)^2 unstripped should be non-zero"
    print("  (psibar psi)^2 matrix element (non-zero): PASS")

    V = simplify_deltas(vertex_factor(**L_current_current, x=x, d=d), species_map=sm4)
    expected_jj = (
        2 * I * gJJ * D4
        * (gamma_matrix(i1, i2, mu) * gamma_matrix(i3, i4, mu)
           - gamma_matrix(i1, i4, mu) * gamma_matrix(i3, i2, mu))
    )
    _check(simplify_gamma_chain(V), expected_jj, "Current-current stripped")

    print("\n  Fermion tests passed.\n")


def _run_fermion_derivative_mixed_tests():
    sm4 = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    G12 = bis.g(i1, i2).to_expression()

    V = simplify_deltas(vertex_factor(**L_mix_dpsibar, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p1, mu) * G12 * D4, "d_mu psibar")

    V = simplify_deltas(vertex_factor(**L_mix_dpsi, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p2, nu) * G12 * D4, "d_nu psi")

    V = simplify_deltas(vertex_factor(**L_mix_dphi_dchi, x=x, d=d), species_map=sm4)
    _check(V, -I * yF * pcomp(p3, mu) * pcomp(p4, nu) * G12 * D4, "(d_mu phi)(d_nu chi)")

    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_chi, x=x, d=d), species_map=sm4)
    _check(V, -I * g1 * G12 * D4 * pcomp(p3, mu) * pcomp(p3, mu), "g1 * psibar psi (d^2 phi) chi")

    sm_phi2 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0}
    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_phi, x=x, d=d), species_map=sm_phi2)
    _check(
        V,
        2 * I * g2 * G12 * D4
        * pcomp(p3, mu) * pcomp(p3, nu)
        * pcomp(p4, mu) * pcomp(p4, nu),
        "g2 * psibar psi (d_mu d_nu phi)^2",
    )

    print("\n  Mixed fermion+scalar derivative tests passed.\n")


def _run_gauge_ready_tests():
    sm_gauge = {b1: psibar0, b2: psi0, b3: G0}
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)

    V = simplify_deltas(vertex_factor(**L_quark_gluon, x=x, d=d), species_map=sm_gauge)
    expected = I * gS * gamma_matrix(i1, i2, mu3) * gauge_generator(a3, c1, c2) * D3
    _check(V, expected, "Quark-gluon (direct API)")

    sm_scalar_gauge = {b1: phiCdag0, b2: phiC0, b3: A0}
    V_phi = simplify_deltas(
        vertex_factor(**L_complex_scalar_current_phi, x=x, d=d), species_map=sm_scalar_gauge
    )
    V_phidag = simplify_deltas(
        vertex_factor(**L_complex_scalar_current_phidag, x=x, d=d), species_map=sm_scalar_gauge
    )
    V_total = V_phi + V_phidag
    assert V_total.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    print("  Complex scalar current (non-zero): PASS")

    sm_contact = {b1: phiCdag0, b2: phiC0, b3: A0, b4: A0}
    V = simplify_deltas(
        vertex_factor(**L_complex_scalar_contact, x=x, d=d), species_map=sm_contact
    )
    assert V.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    print("  Complex scalar contact (non-zero): PASS")

    print("\n  Gauge-ready tests passed.\n")


# ===================================================================
# Model-layer tests (FeynRules-style API)
# ===================================================================

def _run_model_tests():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    sm_gauge = {b1: psibar0, b2: psi0, b3: G0}
    sm_scalar_gauge = {b1: phiCdag0, b2: phiC0, b3: A0}
    sm_contact = {b1: phiCdag0, b2: phiC0, b3: A0, b4: A0}

    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    G12 = bis.g(i1, i2).to_expression()

    # Scalar
    _check(
        _model_vertex(interaction=TERM_phi4, external_legs=LEGS_phi4, species_map=sm_phi),
        24 * I * lam4 * D4,
        "Model: phi^4",
    )
    _check(
        _model_vertex(interaction=TERM_phi2chi2, external_legs=LEGS_phi2chi2,
                      species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0}),
        4 * I * g_sym * D4,
        "Model: phi^2 chi^2",
    )
    _check(
        _model_vertex(interaction=TERM_phiCdag_phiC, external_legs=LEGS_phiCdag_phiC,
                      species_map={b1: phiCdag0, b2: phiC0}),
        I * lamC * (2 * pi) ** d * Delta(p1 + p2),
        "Model: phi^dag phi",
    )

    # Fermion
    _check(
        _model_vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        I * yF * G12 * D3,
        "Model: Yukawa amputated",
    )
    _check(
        _model_vertex(interaction=TERM_vec_current, external_legs=LEGS_vec_current),
        I * gV * gamma_matrix(i1, i2, mu3) * D3,
        "Model: Vector current",
    )
    _check(
        _model_vertex(interaction=TERM_axial_current, external_legs=LEGS_vec_current),
        I * gV * gamma_matrix(i1, alpha_s, mu3) * gamma5_matrix(alpha_s, i2) * D3,
        "Model: Axial current",
    )

    V_sp = _model_vertex(interaction=TERM_psibar_psi_sq, external_legs=LEGS_fermion4)
    expected_sp = (
        -I * g_psi4 * D4
        * (bis.g(i1, i2).to_expression() * bis.g(i3, i4).to_expression()
           - bis.g(i1, i4).to_expression() * bis.g(i3, i2).to_expression())
    )
    _check(V_sp, expected_sp, "Model: (psibar psi)^2 amputated")

    V_jj = _model_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)
    expected_jj = (
        2 * I * gJJ * D4
        * (gamma_matrix(i1, i2, mu) * gamma_matrix(i3, i4, mu)
           - gamma_matrix(i1, i4, mu) * gamma_matrix(i3, i2, mu))
    )
    _check(simplify_gamma_chain(V_jj), expected_jj, "Model: Current-current")

    # Gauge-ready
    V_qg = _model_vertex(interaction=TERM_quark_gluon, external_legs=LEGS_quark_gluon)
    expected_qg = I * gS * gamma_matrix(i1, i2, mu3) * gauge_generator(a3, c1, c2) * D3
    _check(V_qg, expected_qg, "Model: Quark-gluon")

    V_sc = (
        _model_vertex(interaction=TERM_complex_scalar_current_phi,
                      external_legs=LEGS_complex_scalar_current, species_map=sm_scalar_gauge)
        + _model_vertex(interaction=TERM_complex_scalar_current_phidag,
                        external_legs=LEGS_complex_scalar_current, species_map=sm_scalar_gauge)
    )
    assert V_sc.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    print("  Model: Complex scalar current (non-zero): PASS")

    V_ct = _model_vertex(interaction=TERM_complex_scalar_contact,
                         external_legs=LEGS_complex_scalar_contact, species_map=sm_contact)
    assert V_ct.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    print("  Model: Complex scalar contact (non-zero): PASS")

    print("\n  Model-layer tests passed.\n")


# ===================================================================
# Cross-check: model-layer vs direct API agreement
# ===================================================================

def _run_cross_checks():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    sm_gauge = {b1: psibar0, b2: psi0, b3: G0}

    _check(
        _model_vertex(interaction=TERM_phi4, external_legs=LEGS_phi4, species_map=sm_phi),
        simplify_deltas(vertex_factor(**L_phi4, x=x, d=d), species_map=sm_phi),
        "Cross: phi^4",
    )
    _check(
        _model_vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        simplify_deltas(vertex_factor(**L_yukawa, x=x, d=d), species_map=sm3),
        "Cross: Yukawa",
    )
    _check(
        simplify_gamma_chain(
            _model_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)
        ),
        simplify_gamma_chain(
            simplify_deltas(vertex_factor(**L_current_current, x=x, d=d), species_map=sm4)
        ),
        "Cross: Current-current",
    )
    _check(
        _model_vertex(interaction=TERM_quark_gluon, external_legs=LEGS_quark_gluon),
        simplify_deltas(vertex_factor(**L_quark_gluon, x=x, d=d), species_map=sm_gauge),
        "Cross: Quark-gluon",
    )

    print("\n  Cross-checks passed.\n")


# ===================================================================
# Regression tests: role-based filtering
# ===================================================================

def _run_role_regression_tests():
    from model import FieldRole, ROLE_SCALAR, ROLE_SCALAR_DAG, ROLE_VECTOR, ROLE_PSI, ROLE_PSIBAR

    # 1. Complex boson: role filtering eliminates bad contractions without species_map
    D2 = (2 * pi) ** d * Delta(p1 + p2)
    V_complex = vertex_factor(
        coupling=lamC,
        alphas=[phiCdag0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        x=x, d=d,
    )
    raw_str = V_complex.expand().to_canonical_string()
    assert "delta" in raw_str, "Should have species delta before simplification"
    simplified_no_map = simplify_deltas(V_complex)
    simplified_with_map = simplify_deltas(V_complex, species_map={b1: phiCdag0, b2: phiC0})
    _check(simplified_with_map, I * lamC * D2, "Regression: complex boson with species_map")
    no_map_str = simplified_no_map.expand().to_canonical_string()
    assert "0" != no_map_str, "complex boson without species_map should be non-zero"
    print("  Regression: complex boson filtered by role (no extra term): PASS")

    # 2. Verify that ROLE_SCALAR_DAG won't match ROLE_SCALAR legs (1 perm, not 2)
    V_wrong_order = vertex_factor(
        coupling=lamC,
        alphas=[phiCdag0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR, ROLE_SCALAR_DAG],
        x=x, d=d,
    )
    V_wrong_simplified = simplify_deltas(V_wrong_order, species_map={b1: phiC0, b2: phiCdag0})
    _check(V_wrong_simplified, I * lamC * D2, "Regression: reversed legs still works")

    # 3. Vector role doesn't match scalar role
    V_mixed = vertex_factor(
        coupling=lamC,
        alphas=[A0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_VECTOR, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR, ROLE_VECTOR],
        x=x, d=d,
    )
    V_mixed_simplified = simplify_deltas(V_mixed, species_map={b1: phiC0, b2: A0})
    _check(V_mixed_simplified, I * lamC * D2, "Regression: vector/scalar non-mixing")

    V_no_match = vertex_factor(
        coupling=lamC,
        alphas=[A0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_VECTOR, ROLE_SCALAR],
        leg_roles=[ROLE_VECTOR, ROLE_VECTOR],
        x=x, d=d,
    )
    _check(
        simplify_deltas(V_no_match, species_map={b1: A0, b2: A0}),
        Expression.num(0),
        "Regression: scalar field can't match vector-only legs",
    )

    # 4. FieldRole object semantics
    assert ROLE_PSI.is_fermion
    assert ROLE_PSIBAR.is_fermion
    assert not ROLE_SCALAR.is_fermion
    assert not ROLE_VECTOR.is_fermion
    assert ROLE_PSI.compatible_with(ROLE_PSI)
    assert not ROLE_PSI.compatible_with(ROLE_PSIBAR)
    assert not ROLE_SCALAR.compatible_with(ROLE_SCALAR_DAG)
    assert not ROLE_SCALAR.compatible_with(ROLE_VECTOR)
    assert ROLE_PSI.compatible_with("psi")
    assert not ROLE_PSI.compatible_with("psibar")
    print("  Regression: FieldRole object semantics: PASS")

    print("\n  Role regression tests passed.\n")


# ===================================================================
# Test runner
# ===================================================================

def _run_all_tests():
    print("=" * 80)
    print("  Direct-API tests")
    print("=" * 80)
    _run_scalar_tests()
    _run_fermion_tests()
    _run_fermion_derivative_mixed_tests()
    _run_gauge_ready_tests()

    print("=" * 80)
    print("  Model-layer tests")
    print("=" * 80)
    _run_model_tests()

    print("=" * 80)
    print("  Cross-checks (model vs direct)")
    print("=" * 80)
    _run_cross_checks()

    print("=" * 80)
    print("  Role regression tests")
    print("=" * 80)
    _run_role_regression_tests()

    print("All tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vertex examples and tests.")
    parser.add_argument(
        "--suite",
        choices=("scalar", "fermion", "gauge", "model", "cross", "role", "all"),
        default="all",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Suppress the detailed human-readable vertex output blocks.",
    )
    args = parser.parse_args()

    if not args.no_demo:
        _run_demo_output(args.suite)

    if not args.skip_tests:
        if args.suite == "all":
            _run_all_tests()
        elif args.suite == "scalar":
            _run_scalar_tests()
        elif args.suite == "fermion":
            _run_fermion_tests()
            _run_fermion_derivative_mixed_tests()
        elif args.suite == "gauge":
            _run_gauge_ready_tests()
        elif args.suite == "model":
            _run_model_tests()
        elif args.suite == "cross":
            _run_cross_checks()
        elif args.suite == "role":
            _run_role_regression_tests()
