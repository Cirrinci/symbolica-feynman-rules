"""
Vertex-factor examples and tests using model_symbolica.py.
"""

import argparse
from fractions import Fraction

from model_schema import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    ConcreteIndexSlot,
    IndexType,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    DerivativeAction,
    Field,
    InteractionTerm,
    bind_indices,
    default_external_legs_for_interaction,
)
from model_symbolica import (
    S,
    Expression,
    I,
    UF,
    UbarF,
    delta,
    bis,
    pcomp,
    vertex_factor,
    simplify_deltas,
    simplify_spinor_indices,
    infer_derivative_targets,
    compact_vertex_sum_form,
    compact_sum_notation,
)
from spenso_structures import (
    gauge_generator,
    gamma_lowered_matrix,
    gamma_matrix,
    gamma5_matrix,
    lorentz_metric,
    simplify_gamma_chain,
    slot_labels,
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

mu, nu = S("mu", "nu")

yF = S("yF")
psibar0, psi0 = S("psibar0", "psi0")
i_psi_bar, i_psi = S("i_psi_bar", "i_psi")
s1, s2, s3, s4 = S("s1", "s2", "s3", "s4")
A0 = S("A0")
G0 = S("G0")
gV = S("gV")
gS = S("gS")
gPhiA = S("gPhiA")
gPhiAA = S("gPhiAA")
g4F = S("g4F")
g_psi4 = S("g_psi4")
gJJ = S("gJJ")
alpha_s, beta_s = S("alpha_s", "beta_s")
a_bar, a_psi, b_bar, b_psi = S("a_bar", "a_psi", "b_bar", "b_psi")
i_bar_q, i_psi_q = S("i_bar_q", "i_psi_q")
c_bar_q, c_psi_q, a_g = S("c_bar_q", "c_psi_q", "a_g")
i1, i2, i3, i4 = S("i1", "i2", "i3", "i4")
c1, c2, c3, a1, a2, a3 = S("c1", "c2", "c3", "a1", "a2", "a3")
mu1, mu2, mu3, mu4 = S("mu1", "mu2", "mu3", "mu4")
idx_i, idx_j, idx_k = S("i", "j", "k")

lam4 = S("lam4")
lam6 = S("lam6")
lamC = S("lamC")
g_sym = S("g")
gD = S("gD")
gD2 = S("gD2")
gijk = S("gijk")
g1 = S("g1")
g2 = S("g2")

PsiField = Field(
    "Psi",
    spin=Fraction(1, 2),
    self_conjugate=False,
    kind="fermion",
    symbol=psi0,
    conjugate_symbol=psibar0,
    indices=(SPINOR_INDEX,),
)
PhiCField = Field(
    "PhiC",
    spin=0,
    self_conjugate=False,
    kind="scalar",
    symbol=phiC0,
    conjugate_symbol=phiCdag0,
)
GaugeField = Field(
    "A",
    spin=1,
    self_conjugate=True,
    kind="vector",
    symbol=A0,
    indices=(LORENTZ_INDEX,),
)
QuarkField = Field(
    "q",
    spin=Fraction(1, 2),
    self_conjugate=False,
    kind="fermion",
    symbol=psi0,
    conjugate_symbol=psibar0,
    indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
)
GluonField = Field(
    "G",
    spin=1,
    self_conjugate=True,
    kind="vector",
    symbol=G0,
    indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(got, expected, label):
    """Assert two expressions are equal after expand + canonicalize, then print PASS."""
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


def _display_vertex_inputs(*, interaction, external_legs, alphas, betas, ps):
    if interaction is None:
        return alphas, betas, ps

    shown_alphas = [occ.species for occ in interaction.fields]
    if external_legs is None:
        external_legs = default_external_legs_for_interaction(
            interaction,
            momenta=ps if ps is not None else None,
        )

    shown_betas = [leg.species for leg in external_legs]
    shown_ps = [leg.momentum for leg in external_legs]
    return shown_alphas, shown_betas, shown_ps


def show_vertex(
    title,
    *,
    interaction=None,
    external_legs=None,
    coupling=None,
    alphas=None,
    betas=None,
    ps=None,
    derivative_indices=(),
    derivative_targets=None,
    statistics="boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices=None,
    field_slot_labels=None,
    leg_spins=None,
    leg_spinor_indices=None,
    leg_slot_labels=None,
    strip_externals=True,
    species_map=None,
    compact_override=None,
    show_sum_notation=False,
):
    V = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        coupling=coupling,
        alphas=alphas,
        betas=betas,
        ps=ps,
        x=x,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        field_slot_labels=field_slot_labels,
        leg_spins=leg_spins,
        leg_spinor_indices=leg_spinor_indices,
        leg_slot_labels=leg_slot_labels,
        strip_externals=strip_externals,
        include_delta=True,
        d=d,
    )

    if compact_override is not None:
        compact = compact_override
        result_label = "Compact override"
    else:
        compact = simplify_deltas(V, species_map=species_map)
        result_label = "Vertex"

    shown_alphas, shown_betas, shown_ps = _display_vertex_inputs(
        interaction=interaction,
        external_legs=external_legs,
        alphas=alphas,
        betas=betas,
        ps=ps,
    )

    print("=" * 80)
    print(f"  {title}")
    print(f"  alphas = {shown_alphas}")
    print(f"  betas  = {shown_betas}")
    print(f"  ps     = {shown_ps}")
    print()
    print(f"  {result_label}:")
    print(f"  {compact}")
    if show_sum_notation and derivative_indices:
        print("  Sum notation:")
        print(
            " ",
            compact_sum_notation(
                derivative_indices=derivative_indices,
                derivative_targets=derivative_targets,
                n_legs=len(shown_ps),
            ),
        )
    print()
    return V, compact


def show_expression(title, *, expr, alphas, betas, ps):
    print("=" * 80)
    print(f"  {title}")
    print(f"  alphas = {alphas}")
    print(f"  betas  = {betas}")
    print(f"  ps     = {ps}")
    print()
    print("  Vertex:")
    print(f"  {expr}")
    print()
    return expr


# ---------------------------------------------------------------------------
# Interaction definitions
# ---------------------------------------------------------------------------

# 1) phi^4
L_phi4 = dict(
    coupling=lam4,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# 2) phi^6
L_phi6 = dict(
    coupling=lam6,
    alphas=[phi0] * 6,
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# 3) phi^2 chi^2
L_phi2chi2 = dict(
    coupling=g_sym,
    alphas=[phi0, phi0, chi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# 3b) complex scalar bilinear
L_phiCdag_phiC = dict(
    coupling=lamC,
    alphas=[phiCdag0, phiC0],
    betas=[b1, b2],
    ps=[p1, p2],
    statistics="boson",
    field_roles=["scalar_dag", "scalar"],
    leg_roles=["scalar_dag", "scalar"],
)

# 4) gD * (d_mu phi)(d_nu phi) phi phi
deriv_indices, deriv_targets = infer_derivative_targets([(0, [mu]), (1, [nu])])
L_deriv = dict(
    coupling=gD,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices,
    derivative_targets=deriv_targets,
)

# 5) gD2 * (d_mu phi)(d_mu phi) phi phi
deriv_indices2, deriv_targets2 = infer_derivative_targets([(0, [mu]), (1, [mu])])
L_deriv2 = dict(
    coupling=gD2,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2,
    derivative_targets=deriv_targets2,
)

# 6) gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2
L_multi = dict(
    coupling=gijk(idx_i, idx_j, idx_k),
    alphas=[idx_i, idx_i, idx_j, idx_j, idx_k, idx_k],
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# 7) Yukawa: yF * psibar * psi * phi
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

# 8) Vector current: gV * psibar gamma^mu psi A_mu
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

# 9) Underspecified multi-fermion product kept as a validation diagnostic
L_4fermion = dict(
    coupling=g4F,
    alphas=[psi0, psibar0, psi0, psibar0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psi", "psibar", "psi", "psibar"],
    leg_roles=["psi", "psibar", "psi", "psibar"],
)

# 10) -(g/2)(psibar psi)^2  (amputated / matrix-element / explicit-spinor modes)
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

# 10b) Current-current four-fermion operator:
#     gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)
# Use distinct open endpoint labels for the explicit gamma chains.
# Repeating a label across psibar/psi slots is reserved for inferred scalar
# bilinears such as (psibar psi), not tensor structures with explicit gammas.
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

# 10c) Gauge-ready current with spinor + Lorentz + color slot labels:
#      gS * psibar gamma^mu T^a psi G^a_mu
L_quark_gluon = dict(
    coupling=gS * gamma_matrix(i_bar_q, i_psi_q, mu) * gauge_generator(a_g, c_bar_q, c_psi_q),
    alphas=[psibar0, psi0, G0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_bar_q, i_psi_q, None],
    field_slot_labels=[
        slot_labels(spinor=i_bar_q, color_fund=c_bar_q),
        slot_labels(spinor=i_psi_q, color_fund=c_psi_q),
        slot_labels(lorentz=mu, color_adj=a_g),
    ],
    leg_spins=[s1, s2, s3],
)

# 10d) Complex scalar gauge structures:
#      gPhiA * A_mu * phi^dagger <-> d^mu phi
#      gPhiAA * A_mu A^mu phi^dagger phi
_COMPLEX_SCALAR_GAUGE_BASE = dict(
    alphas=[phiCdag0, phiC0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="boson",
    field_roles=["scalar_dag", "scalar", "vector"],
    leg_roles=["scalar_dag", "scalar", "vector"],
    field_slot_labels=[None, None, slot_labels(lorentz=mu)],
)

L_complex_scalar_current_phi = dict(
    **_COMPLEX_SCALAR_GAUGE_BASE,
    coupling=gPhiA,
    derivative_indices=[mu],
    derivative_targets=[1],
)

L_complex_scalar_current_phidag = dict(
    **_COMPLEX_SCALAR_GAUGE_BASE,
    coupling=-gPhiA,
    derivative_indices=[mu],
    derivative_targets=[0],
)

L_complex_scalar_contact = dict(
    coupling=gPhiAA * lorentz_metric(mu, nu),
    alphas=[phiCdag0, phiC0, A0, A0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="boson",
    field_roles=["scalar_dag", "scalar", "vector", "vector"],
    leg_roles=["scalar_dag", "scalar", "vector", "vector"],
    field_slot_labels=[
        None,
        None,
        slot_labels(lorentz=mu),
        slot_labels(lorentz=nu),
    ],
)

# Metadata-layer versions of representative interactions. These are the
# FeynRules-style declarations we ultimately want to scale out.
TERM_meta_phiCdag_phiC = InteractionTerm(
    coupling=lamC,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
    ),
    label="lamC * phi^dagger * phi",
)

TERM_meta_quark_gluon = InteractionTerm(
    coupling=gS * gamma_matrix(i_bar_q, i_psi_q, mu) * gauge_generator(a_g, c_bar_q, c_psi_q),
    fields=(
        QuarkField.occurrence(
            conjugated=True,
            slot_labels=bind_indices(
                (SPINOR_INDEX, i_bar_q),
                (COLOR_FUND_INDEX, c_bar_q),
            ),
        ),
        QuarkField.occurrence(
            slot_labels=bind_indices(
                (SPINOR_INDEX, i_psi_q),
                (COLOR_FUND_INDEX, c_psi_q),
            ),
        ),
        GluonField.occurrence(
            slot_labels=bind_indices(
                (LORENTZ_INDEX, mu),
                (COLOR_ADJ_INDEX, a_g),
            ),
        ),
    ),
    label="gS * psibar gamma^mu T^a psi G^a_mu",
)

TERM_meta_complex_scalar_current_phi = InteractionTerm(
    coupling=gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(slot_labels=bind_indices((LORENTZ_INDEX, mu))),
    ),
    derivatives=(DerivativeAction(target=1, indices=(mu,)),),
    label="gPhiA * A_mu * phi^dagger * d^mu phi",
)

TERM_meta_complex_scalar_current_phidag = InteractionTerm(
    coupling=-gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(slot_labels=bind_indices((LORENTZ_INDEX, mu))),
    ),
    derivatives=(DerivativeAction(target=0, indices=(mu,)),),
    label="-gPhiA * A_mu * d^mu phi^dagger * phi",
)

TERM_meta_complex_scalar_contact = InteractionTerm(
    coupling=gPhiAA * lorentz_metric(mu, nu),
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        GaugeField.occurrence(slot_labels=bind_indices((LORENTZ_INDEX, mu))),
        GaugeField.occurrence(slot_labels=bind_indices((LORENTZ_INDEX, nu))),
    ),
    label="gPhiAA * A_mu A^mu phi^dagger phi",
)

# 11-14) Mixed derivative fermion+scalar interactions
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
L_mix_5pt = dict(
    coupling=g_sym,
    alphas=[psibar0, psi0, phi0, phi0, chi0],
    betas=[b1, b2, b3, b4, b5],
    ps=[p1, p2, p3, p4, p5],
    derivative_indices=[mu, nu],
    derivative_targets=[0, 1],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None, None, None],
    leg_spins=[s1, s2, s3, s4, s1],
)

# 15-16) Double-derivative scalars with fermion bilinear
# L1 = g1 * psibar * psi * (d^2 phi) * chi   [d^2 modeled as d_mu d_mu on the phi slot]
deriv_indices_l1, deriv_targets_l1 = infer_derivative_targets([(2, [mu, mu])])
L_double_deriv_phi_chi = dict(
    **_MIX_BASE,
    coupling=g1,
    derivative_indices=deriv_indices_l1,
    derivative_targets=deriv_targets_l1,
)

# L2 = g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)
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

# Pre-built compact forms for derivative tests
COMPACT_DERIV = compact_vertex_sum_form(
    coupling=gD, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices, derivative_targets=deriv_targets, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
    include_delta=False,
)
COMPACT_DERIV2 = compact_vertex_sum_form(
    coupling=gD2, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2, derivative_targets=deriv_targets2, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
    include_delta=False,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_scalar_tests():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    D4 = Expression.num(1)

    V = simplify_deltas(vertex_factor(**L_phi4, x=x, d=d), species_map=sm_phi)
    _check(V, 24 * I * lam4 * D4, "phi^4")

    V = simplify_deltas(vertex_factor(**L_phi2chi2, x=x, d=d),
                        species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0})
    _check(V, 4 * I * g_sym * D4, "phi^2 chi^2")

    V = simplify_deltas(
        vertex_factor(**L_phiCdag_phiC, x=x, d=d),
        species_map={b1: phiCdag0, b2: phiC0},
    )
    _check(V, I * lamC, "phi^dagger phi")

    V_meta = simplify_deltas(vertex_factor(interaction=TERM_meta_phiCdag_phiC, x=x, d=d))
    _check(V_meta, I * lamC, "phi^dagger phi [metadata]")

    V = simplify_deltas(vertex_factor(**L_deriv, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV, "Derivative (mu,nu)")

    V = simplify_deltas(vertex_factor(**L_deriv2, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV2, "Derivative (mu,mu)")

    # Multi-species: gijk * phi_i^2 phi_j^2 phi_k^2
    D6 = Expression.num(1)
    V_multi = vertex_factor(**L_multi, x=x, d=d)
    expected_multi = 8 * I * gijk(idx_i, idx_j, idx_k) * D6

    sm_base = {b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k}
    _check(simplify_deltas(V_multi, species_map=sm_base), expected_multi, "Multi-species (base)")

    sm_perm = {b1: idx_j, b2: idx_k, b3: idx_i, b4: idx_j, b5: idx_k, b6: idx_i}
    _check(simplify_deltas(V_multi, species_map=sm_perm), expected_multi, "Multi-species (perm)")

    sm_bad = {b1: idx_i, b2: idx_i, b3: idx_i, b4: idx_j, b5: idx_j, b6: idx_k}
    _check(simplify_deltas(V_multi, species_map=sm_bad), Expression.num(0), "Multi-species (bad mult -> 0)")

    print("\n  Scalar+derivative tests passed.\n")


def _run_fermion_tests():
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    D3 = Expression.num(1)
    D4 = Expression.num(1)
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
        assert "Multi-fermion operators require explicit spinor-contraction data" in str(exc)
        print("  underspecified multi-fermion operator rejected: PASS")
    else:
        raise AssertionError("Bare psi*psibar*psi*psibar should be rejected as underspecified")

    # (psibar psi)^2 amputated -> open spinor structure
    V = simplify_deltas(vertex_factor(**L_psibar_psi_sq, x=x, d=d), species_map=sm4)
    expected_sp = (
        -I * g_psi4 * D4
        * (bis.g(i1, i2).to_expression() * bis.g(i3, i4).to_expression()
           - bis.g(i1, i4).to_expression() * bis.g(i3, i2).to_expression())
    )
    _check(V, expected_sp, "(psibar psi)^2 amputated")

    # (psibar psi)^2 unstripped -> non-zero with UF/UbarF
    V_full = simplify_deltas(
        vertex_factor(**L_psibar_psi_sq, x=x, d=d, strip_externals=False), species_map=sm4
    )
    s = V_full.to_canonical_string()
    assert s != Expression.num(0).to_canonical_string(), "(psibar psi)^2 unstripped should be non-zero"
    assert "UbarF" in s and "UF" in s, f"Missing UF/UbarF: {V_full}"
    print("  (psibar psi)^2 matrix element (non-zero direct/exchange): PASS")

    # Spinor-delta form: -ig * D4 * [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]
    V_sp = simplify_deltas(vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d), species_map=sm4)
    _check(V_sp, expected_sp, "(psibar psi)^2 spinor deltas")

    # Current-current stripped -> direct minus exchange open-index structure
    V = simplify_deltas(vertex_factor(**L_current_current, x=x, d=d), species_map=sm4)
    expected_jj = (
        2 * I * gJJ * D4
        * (
            gamma_matrix(i1, i2, mu) * gamma_matrix(i3, i4, mu)
            - gamma_matrix(i1, i4, mu) * gamma_matrix(i3, i2, mu)
        )
    )
    _check(
        simplify_gamma_chain(V),
        expected_jj,
        "Current-current stripped",
    )

    # Current-current unstripped -> non-zero; contraction topologies are still distinguishable
    V_full = simplify_deltas(
        vertex_factor(**L_current_current, x=x, d=d, strip_externals=False), species_map=sm4
    )
    s = V_full.to_canonical_string()
    assert s != Expression.num(0).to_canonical_string(), "Current-current unstripped should be non-zero"
    assert "UbarF" in s and "UF" in s, f"Current-current unstripped missing UF/UbarF: {V_full}"
    print("  Current-current unstripped (non-zero): PASS")

    invalid_spinor_legs = dict(L_psibar_psi_sq_spinor)
    invalid_spinor_legs["leg_spinor_indices"] = [i1, None, i3, i4]
    try:
        vertex_factor(
            **invalid_spinor_legs,
            x=x,
            d=d,
        )
        raise AssertionError("Expected ValueError for missing fermion leg spinor index")
    except ValueError as exc:
        assert "Fermion legs must carry a spinor index" in str(exc), exc
    print("  Missing fermion leg spinor index -> ValueError: PASS")

    print("\n  Fermion tests passed.\n")


def _run_fermion_derivative_mixed_tests():
    sm4 = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}
    D4 = Expression.num(1)
    D5 = Expression.num(1)
    G12 = bis.g(i1, i2).to_expression()

    V = simplify_deltas(vertex_factor(**L_mix_dpsibar, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p1, mu) * G12 * D4, "d_mu psibar")

    V = simplify_deltas(vertex_factor(**L_mix_dpsi, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p2, nu) * G12 * D4, "d_nu psi")

    V = simplify_deltas(vertex_factor(**L_mix_dphi_dchi, x=x, d=d), species_map=sm4)
    _check(V, -I * yF * pcomp(p3, mu) * pcomp(p4, nu) * G12 * D4, "(d_mu phi)(d_nu chi)")

    sm5 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0}
    V = simplify_deltas(vertex_factor(**L_mix_5pt, x=x, d=d), species_map=sm5)
    _check(V, -2 * I * g_sym * pcomp(p1, mu) * pcomp(p2, nu) * G12 * D5, "5pt mixed")

    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_chi, x=x, d=d), species_map=sm4)
    _check(V, -I * g1 * G12 * D4 * pcomp(p3, mu) * pcomp(p3, mu), "g1 * psibar psi (d^2 phi) chi")

    sm_phi2 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0}
    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_phi, x=x, d=d), species_map=sm_phi2)
    _check(
        V,
        2
        * I
        * g2
        * G12
        * D4
        * pcomp(p3, mu)
        * pcomp(p3, nu)
        * pcomp(p4, mu)
        * pcomp(p4, nu),
        "g2 * psibar psi (d_mu d_nu phi)^2",
    )

    print("\n  Mixed fermion+scalar derivative tests passed.\n")


def _run_gauge_ready_tests():
    sm3 = {b1: psibar0, b2: psi0, b3: G0}
    D3 = Expression.num(1)

    V = simplify_deltas(vertex_factor(**L_quark_gluon, x=x, d=d), species_map=sm3)
    expected = I * gS * gamma_matrix(i1, i2, mu3) * gauge_generator(a3, c1, c2) * D3
    _check(V, expected, "Gauge-ready quark-gluon current")

    V_meta = simplify_deltas(vertex_factor(interaction=TERM_meta_quark_gluon, x=x, d=d))
    _check(V_meta, expected, "Gauge-ready quark-gluon current [metadata]")

    sm_scalar = {b1: phiCdag0, b2: phiC0, b3: A0}
    V_scalar_current = simplify_deltas(
        vertex_factor(**L_complex_scalar_current_phi, x=x, d=d)
        + vertex_factor(**L_complex_scalar_current_phidag, x=x, d=d),
        species_map=sm_scalar,
    )
    _check(
        V_scalar_current,
        gPhiA * (pcomp(p2, mu3) - pcomp(p1, mu3)),
        "Complex scalar gauge current",
    )

    V_scalar_current_meta = simplify_deltas(
        vertex_factor(interaction=TERM_meta_complex_scalar_current_phi, x=x, d=d)
        + vertex_factor(interaction=TERM_meta_complex_scalar_current_phidag, x=x, d=d)
    )
    _check(
        V_scalar_current_meta,
        gPhiA * (pcomp(p2, mu3) - pcomp(p1, mu3)),
        "Complex scalar gauge current [metadata]",
    )

    V_scalar_contact = simplify_deltas(
        vertex_factor(**L_complex_scalar_contact, x=x, d=d),
        species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
    )
    _check(
        V_scalar_contact,
        2 * I * gPhiAA * lorentz_metric(mu3, mu4),
        "Complex scalar gauge contact",
    )

    V_scalar_contact_meta = simplify_deltas(
        vertex_factor(interaction=TERM_meta_complex_scalar_contact, x=x, d=d)
    )
    _check(
        V_scalar_contact_meta,
        2 * I * gPhiAA * lorentz_metric(mu3, mu4),
        "Complex scalar gauge contact [metadata]",
    )

    V_index_slots = simplify_deltas(
        vertex_factor(
            coupling=L_quark_gluon["coupling"],
            alphas=L_quark_gluon["alphas"],
            betas=L_quark_gluon["betas"],
            ps=L_quark_gluon["ps"],
            x=x,
            d=d,
            statistics=L_quark_gluon["statistics"],
            field_roles=L_quark_gluon["field_roles"],
            leg_roles=L_quark_gluon["leg_roles"],
            field_index_slots=[
                QuarkField.bound_index_slots(
                    bind_indices((SPINOR_INDEX, i_bar_q), (COLOR_FUND_INDEX, c_bar_q)),
                    conjugated=True,
                ),
                QuarkField.bound_index_slots(
                    bind_indices((SPINOR_INDEX, i_psi_q), (COLOR_FUND_INDEX, c_psi_q)),
                ),
                GluonField.bound_index_slots(
                    bind_indices((LORENTZ_INDEX, mu), (COLOR_ADJ_INDEX, a_g)),
                ),
            ],
            leg_index_slots=[
                QuarkField.bound_index_slots(
                    bind_indices((SPINOR_INDEX, i1), (COLOR_FUND_INDEX, c1)),
                    conjugated=True,
                ),
                QuarkField.bound_index_slots(
                    bind_indices((SPINOR_INDEX, i2), (COLOR_FUND_INDEX, c2)),
                ),
                GluonField.bound_index_slots(
                    bind_indices((LORENTZ_INDEX, mu3), (COLOR_ADJ_INDEX, a3)),
                ),
            ],
        ),
        species_map=sm3,
    )
    _check(V_index_slots, expected, "Gauge-ready quark-gluon current [concrete slots]")

    OrderedIndexField = Field(
        name="OrderedIndexField",
        spin=0,
        self_conjugate=True,
        kind="scalar",
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX, LORENTZ_INDEX),
    )
    ordered_occurrence = OrderedIndexField.occurrence(
        slot_labels=bind_indices(
            (LORENTZ_INDEX, (mu, nu)),
            (COLOR_ADJ_INDEX, a3),
        )
    )
    assert tuple(slot.index_type for slot in ordered_occurrence.index_slots) == (
        LORENTZ_INDEX,
        COLOR_ADJ_INDEX,
        LORENTZ_INDEX,
    )
    assert tuple(slot.label for slot in ordered_occurrence.index_slots) == (
        mu,
        a3,
        nu,
    )
    print("  Ordered concrete index slots: PASS")

    try:
        PhiCField.occurrence(role="psi")
        raise AssertionError("Expected ValueError for scalar field with fermion role")
    except ValueError as exc:
        assert "incompatible" in str(exc), exc
    print("  Invalid scalar role: PASS")

    try:
        GaugeField.occurrence(role="vector_dag")
        raise AssertionError("Expected ValueError for self-conjugate field with dag role")
    except ValueError as exc:
        assert "self-conjugate" in str(exc), exc
    print("  Invalid self-conjugate dag role: PASS")

    AntiColorFundIndex = IndexType(
        "ColorFundBar",
        COLOR_FUND_INDEX.representation,
        kind="color_fund_bar",
        aliases=("color_fund_bar",),
    )
    ColoredScalarField = Field(
        "ColoredScalar",
        spin=0,
        self_conjugate=False,
        kind="scalar",
        indices=(COLOR_FUND_INDEX,),
        conjugate_indices=(AntiColorFundIndex,),
    )
    colored_conjugate = ColoredScalarField.occurrence(
        conjugated=True,
        slot_labels=bind_indices((AntiColorFundIndex, c3)),
    )
    assert colored_conjugate.index_slots[0].index_type == AntiColorFundIndex
    try:
        ColoredScalarField.occurrence(
            conjugated=True,
            slot_labels=bind_indices((COLOR_FUND_INDEX, c3)),
        )
        raise AssertionError("Expected ValueError for wrong conjugate index signature")
    except ValueError as exc:
        assert "does not declare slot kinds" in str(exc) or "expects" in str(exc), exc
    print("  Conjugate index signature: PASS")

    print("\n  Gauge-ready tests passed.\n")


def _relation_to_original(candidate, original):
    cand = candidate.expand().to_canonical_string()
    if cand == original.expand().to_canonical_string():
        return "== V_orig"
    if cand == (-original).expand().to_canonical_string():
        return "== -V_orig"
    return "different (not +/-V_orig)"


def _run_swap_diagnostics():
    """Diagnostic: operator-order sensitivity for (psibar psi)^2."""
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    V_orig = simplify_deltas(vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d), species_map=sm4)

    L_swap_both = dict(L_psibar_psi_sq_spinor)
    L_swap_both.update(
        alphas=[psi0, psibar0, psi0, psibar0],
        field_roles=["psi", "psibar", "psi", "psibar"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_both = simplify_deltas(vertex_factor(**L_swap_both, x=x, d=d), species_map=sm4)
    print("  swap-both bilinears: ", _relation_to_original(V_swap_both, V_orig))

    L_swap_one = dict(L_psibar_psi_sq_spinor)
    L_swap_one.update(
        alphas=[psi0, psibar0, psibar0, psi0],
        field_roles=["psi", "psibar", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_one = simplify_deltas(vertex_factor(**L_swap_one, x=x, d=d), species_map=sm4)
    print("  swap-one bilinear:  ", _relation_to_original(V_swap_one, V_orig))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_suite_tests(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_tests()
    if suite in ("fermion", "all"):
        _run_fermion_tests()
        _run_fermion_derivative_mixed_tests()
    if suite in ("gauge", "all"):
        _run_gauge_ready_tests()
    print("All selected tests passed.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_scalar_demo():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}

    print("\n=== scalar: phi^4 ===")
    show_vertex("lam4 * phi^4", **L_phi4, species_map=sm_phi)

    print("\n=== scalar: phi^6 ===")
    show_vertex("lam6 * phi^6", **L_phi6,
                species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0, b5: phi0, b6: phi0})

    print("\n=== scalar: phi^2 chi^2 ===")
    show_vertex("g * phi^2 * chi^2", **L_phi2chi2,
                species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0})

    print("\n=== scalar: complex scalar bilinear ===")
    show_vertex(
        "lamC * phi^dagger * phi",
        **L_phiCdag_phiC,
        species_map={b1: phiCdag0, b2: phiC0},
    )

    print("\n=== scalar: metadata layer bilinear ===")
    show_vertex(
        "Field/PhiC metadata -> lamC * phi^dagger * phi",
        interaction=TERM_meta_phiCdag_phiC,
    )

    print("\n=== scalar: derivative (mu,nu) * phi^4 ===")
    show_vertex("gD * (d_mu phi)(d_nu phi) phi phi", **L_deriv,
                species_map=sm_phi, compact_override=COMPACT_DERIV, show_sum_notation=True)

    print("\n=== scalar: derivative (mu,mu) * phi^4 ===")
    show_vertex("gD2 * (d_mu phi)(d_mu phi) phi phi", **L_deriv2,
                species_map=sm_phi, compact_override=COMPACT_DERIV2, show_sum_notation=True)

    print("\n=== scalar: multi-species phi_i^2 phi_j^2 phi_k^2 ===")
    show_vertex("gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2", **L_multi,
                species_map={b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k})


def _run_fermion_demo():
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}

    print("\n=== fermion: Yukawa [amputated] ===")
    show_vertex("yF * psibar * psi * phi", **L_yukawa, species_map=sm3)

    print("\n=== fermion: Yukawa [matrix element] ===")
    show_vertex("yF * psibar * psi * phi  [matrix element]", **L_yukawa,
                strip_externals=False, species_map=sm3)

    print("\n=== fermion: vector current ===")
    show_vertex("gV * psibar gamma^mu psi A_mu", **L_vec_current,
                species_map={b1: psibar0, b2: psi0, b3: A0})

    print("\n=== fermion: axial current ===")
    show_vertex("gV * psibar gamma^mu gamma5 psi A_mu", **L_axial_current,
                species_map={b1: psibar0, b2: psi0, b3: A0})

    print("\n=== fermion: underspecified product diagnostic ===")
    print("=" * 80)
    print("  g4F * psi * psibar * psi * psibar  [no spinor contractions]")
    print("  rejected: multi-fermion operators need explicit spinor contractions")

    print("\n=== fermion: -(g/2)(psibar psi)^2  [amputated] ===")
    show_vertex("-(g/2)(psibar psi)^2  [amputated]", **L_psibar_psi_sq, species_map=sm4)

    print("\n=== fermion: -(g/2)(psibar psi)^2  [matrix element] ===")
    show_vertex("-(g/2)(psibar psi)^2  [matrix element]", **L_psibar_psi_sq,
                strip_externals=False, species_map=sm4)

    print("\n=== fermion: -(g/2)(psibar psi)^2  [explicit spinor labels] ===")
    show_vertex("-(g/2)(psibar psi)^2  [spinor delta]", **L_psibar_psi_sq_spinor, species_map=sm4)
    print("=" * 80)
    print(
        "  -(g/2)(psibar psi)(psibar psi)  →  "
        "V = (-ig)[g(i1,i2)*g(i3,i4) - g(i1,i4)*g(i3,i2)]"
    )
    print()

    print("\n=== fermion: current-current operator ===")
    V_jj = simplify_deltas(vertex_factor(**L_current_current, x=x, d=d), species_map=sm4)
    show_vertex(
        "gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]",
        **L_current_current,
        species_map=sm4,
        compact_override=simplify_gamma_chain(V_jj),
    )
    print("  Interpretation: stripped output keeps the direct minus exchange gamma structure visible.")
    print()

    print("\n=== fermion: operator-order diagnostics (psibar psi)^2 ===")
    _run_swap_diagnostics()
    print()

    sm_mix = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}
    print("\n=== fermion+scalar: mixed derivatives ===")
    show_vertex("yF * (d_mu psibar) * psi * phi * chi", **L_mix_dpsibar, species_map=sm_mix)
    show_vertex("yF * psibar * (d_nu psi) * phi * chi", **L_mix_dpsi, species_map=sm_mix)
    show_vertex("yF * psibar * psi * (d_mu phi) * (d_nu chi)", **L_mix_dphi_dchi, species_map=sm_mix)
    show_vertex("g * (d_mu psibar)(d_nu psi) phi phi chi", **L_mix_5pt,
                species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0})
    show_vertex("g1 * psibar * psi * (d^2 phi) * chi", **L_double_deriv_phi_chi, species_map=sm_mix)
    show_vertex(
        "g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)",
        **L_double_deriv_phi_phi,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0},
    )


def _run_gauge_demo():
    print("\n=== gauge-ready: non-abelian fermion current ===")
    show_vertex(
        "gS * psibar gamma^mu T^a psi G^a_mu",
        **L_quark_gluon,
        species_map={b1: psibar0, b2: psi0, b3: G0},
    )
    print("  Interpretation: the coupling now remaps spinor, Lorentz, and color labels through one slot-label path.")
    print()

    print("\n=== metadata layer: non-abelian fermion current ===")
    show_vertex(
        "Field metadata -> gS * psibar gamma^mu T^a psi G^a_mu",
        interaction=TERM_meta_quark_gluon,
    )
    print("  Interpretation: the same vertex is now declared through Field/InteractionTerm metadata instead of aligned parallel lists.")
    print()

    V_scalar_current = simplify_deltas(
        vertex_factor(**L_complex_scalar_current_phi, x=x, d=d)
        + vertex_factor(**L_complex_scalar_current_phidag, x=x, d=d),
        species_map={b1: phiCdag0, b2: phiC0, b3: A0},
    )
    print("\n=== gauge-ready: complex scalar current ===")
    show_expression(
        "gPhiA * A_mu * phi^dagger <-> d^mu phi",
        expr=V_scalar_current,
        alphas=L_complex_scalar_current_phi["alphas"],
        betas=L_complex_scalar_current_phi["betas"],
        ps=L_complex_scalar_current_phi["ps"],
    )
    print("  Interpretation: the gauge-field Lorentz slot now remaps into the derivative index as well.")
    print()

    V_scalar_current_meta = simplify_deltas(
        vertex_factor(interaction=TERM_meta_complex_scalar_current_phi, x=x, d=d)
        + vertex_factor(interaction=TERM_meta_complex_scalar_current_phidag, x=x, d=d)
    )
    print("\n=== metadata layer: complex scalar current ===")
    show_expression(
        "InteractionTerm -> gPhiA * A_mu * phi^dagger <-> d^mu phi",
        expr=V_scalar_current_meta,
        alphas=[occ.species for occ in TERM_meta_complex_scalar_current_phi.fields],
        betas=[phiCdag0, phiC0, A0],
        ps=[p1, p2, p3],
    )
    print("  Interpretation: the metadata version auto-generates the external legs but keeps the same Lorentz-slot remapping.")
    print()

    print("\n=== gauge-ready: complex scalar contact ===")
    show_vertex(
        "gPhiAA * A_mu A^mu phi^dagger phi",
        **L_complex_scalar_contact,
        species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
    )
    print("  Interpretation: repeated gauge legs stay bosonic, while distinct scalar/scalar_dag roles keep the matter flow explicit.")
    print()

    print("\n=== metadata layer: complex scalar contact ===")
    show_vertex(
        "InteractionTerm -> gPhiAA * A_mu A^mu phi^dagger phi",
        interaction=TERM_meta_complex_scalar_contact,
    )
    print("  Interpretation: the same contact term now comes from Field metadata plus slot-labelled occurrences.")
    print()


def _run_suite_demo(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()
    if suite in ("gauge", "all"):
        _run_gauge_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Symbolica vertex examples.")
    parser.add_argument("--suite", choices=("scalar", "fermion", "gauge", "all"), default="all")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  Demo: {args.suite}")
    print(f"{'='*80}\n")
    _run_suite_demo(args.suite)

    if not args.skip_tests:
        print(f"\n{'='*80}")
        print(f"  Tests: {args.suite}")
        print(f"{'='*80}\n")
        _run_suite_tests(args.suite)
