"""
Vertex-factor examples and tests.

Covers both the direct parallel-list API and the FeynRules-style model layer.
This file currently plays two roles:

- readable showcase of the implemented physics structures
- integration-style regression script for the live source tree
"""

import argparse
import re
from fractions import Fraction

from gauge_compiler import (
    compile_covariant_terms,
    compile_minimal_gauge_interactions,
    with_compiled_covariant_terms,
    with_minimal_gauge_interactions,
)
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
from operators import (
    current_current,
    gauge_fixing_bilinear,
    gauge_fixing_bilinear_raw,
    gauge_kinetic_bilinear,
    gauge_kinetic_bilinear_raw,
    ghost_gauge,
    ghost_gauge_raw,
    ghost_kinetic,
    ghost_kinetic_raw,
    psi_bar_gamma5_psi,
    psi_bar_gamma_psi,
    psi_bar_psi,
    quark_gluon_current,
    scalar_gauge_contact,
    scalar_gauge_current_term,
    yang_mills_four_vertex_raw,
    yang_mills_three_vertex_metric_raw,
    yang_mills_three_vertex_raw,
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
    structure_constant,
    simplify_gamma_chain,
)
from tensor_canonicalization import canonize_spenso_tensors
from model import (
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    Field,
    GhostTerm,
    GaugeFixingTerm,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    InteractionTerm,
    DerivativeAction,
    Model,
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
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

p1, p2, p3, p4, p5, p6 = S("p1", "p2", "p3", "p4", "p5", "p6")
b1, b2, b3, b4, b5, b6 = S("b1", "b2", "b3", "b4", "b5", "b6")

phi0 = S("phi0")
chi0 = S("chi0")
phiC0 = S("phiC0")
phiCdag0 = S("phiCdag0")
phiQCD0 = S("phiQCD0")
phiQCDdag0 = S("phiQCDdag0")
phiMix0 = S("phiMix0")
phiMixdag0 = S("phiMixdag0")
phiBi0 = S("phiBi0")
phiBidag0 = S("phiBidag0")
ghG0 = S("ghG0")
ghGbar0 = S("ghGbar0")
psibar0, psi0 = S("psibar0", "psi0")
psibarQED0, psiQED0 = S("psibarQED0", "psiQED0")
psibarMix0, psiMix0 = S("psibarMix0", "psiMix0")
A0 = S("A0")
G0 = S("G0")

mu, nu = S("mu", "nu")
rho, sigma = S("rho", "sigma")
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
eQED = S("eQED")
xiQED = S("xiQED")
xiQCD = S("xiQCD")
qPhi = S("qPhi")
qPsi = S("qPsi")
qMix = S("qMix")
qPhiMix = S("qPhiMix")
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
c1, c2, c3, c4, c_mid = S("c1", "c2", "c3", "c4", "c_mid")
a3, a4, a5, a6 = S("a3", "a4", "a5", "a6")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(got, expected, label, *, show_vertex=False, description=None, display_vertex=None):
    """Assert symbolic equality and optionally print the checked vertex block."""
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    if show_vertex:
        _print_vertex_block(
            label,
            description=description,
            vertex=got,
            compact_override=display_vertex,
        )
    print(f"  {label}: PASS")


def _print_demo_header(title):
    """Print a short section header for one example block."""
    print(f"# === {title} ===\n")


def _print_section(title, content):
    """Print one labeled section, preserving multiline content."""
    if content is None:
        return
    print(f"{title}:")
    text = _ANSI_ESCAPE_RE.sub("", str(content))
    for line in text.splitlines():
        print(f"  {line}")
    print()


def _print_interaction_terms(terms):
    """Print one or more compiled interaction-term labels."""
    if not terms:
        return
    heading = "Compiled interaction term" if len(terms) == 1 else "Compiled interaction terms"
    print(f"{heading}:")
    for index, term in enumerate(terms, start=1):
        prefix = f"{index}. " if len(terms) > 1 else ""
        if isinstance(term, InteractionTerm):
            label = _ANSI_ESCAPE_RE.sub("", term.label or "<no label>")
            print(f"  {prefix}{label}")
        else:
            print(f"  {prefix}{_ANSI_ESCAPE_RE.sub('', str(term))}")
    print()


def _print_interaction_term_objects(terms):
    """Print the concrete compiled InteractionTerm objects."""
    if not terms:
        return
    heading = "Compiled interaction object" if len(terms) == 1 else "Compiled interaction objects"
    print(f"{heading}:")
    for index, term in enumerate(terms, start=1):
        prefix = f"{index}. " if len(terms) > 1 else ""
        text = _ANSI_ESCAPE_RE.sub("", repr(term))
        first, *rest = text.splitlines() or [text]
        print(f"  {prefix}{first}")
        for line in rest:
            print(f"     {line}")
    print()


def _print_vertex_block(
    title,
    *,
    description=None,
    interaction_terms=None,
    show_term_objects=False,
    vertex=None,
    canonical_vertex=None,
    compact_override=None,
    sum_notation=None,
    interpretation=None,
    error=None,
):
    """Render one example/result block in a notebook-friendly text format."""
    _print_demo_header(title)
    if interaction_terms:
        interaction_terms = tuple(interaction_terms)
        _print_interaction_terms(interaction_terms)
        if show_term_objects:
            _print_interaction_term_objects(interaction_terms)
        if (
            description
            and not (
                len(interaction_terms) == 1
                and isinstance(interaction_terms[0], InteractionTerm)
                and str(description) == interaction_terms[0].label
            )
        ):
            _print_section("Context", description)
    elif description:
        _print_section("Interaction / source", description)

    if vertex is not None:
        _print_section("Vertex", vertex)
    if canonical_vertex is not None:
        _print_section("Canonical vertex", canonical_vertex)
    if compact_override is not None:
        _print_section("Compact form", compact_override)
    if sum_notation is not None:
        _print_section("Sum notation", sum_notation)
    if interpretation is not None:
        _print_section("Interpretation", interpretation)
    if error is not None:
        _print_section("Status", error)
    print()


def _symmetrized_generator_contact(adj_left, adj_right, color_left, color_right, color_middle):
    return (
        gauge_generator(adj_left, color_left, color_middle)
        * gauge_generator(adj_right, color_middle, color_right)
        + gauge_generator(adj_right, color_left, color_middle)
        * gauge_generator(adj_left, color_middle, color_right)
    )


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


def _canonized_gauge_vertex(
    expr,
    *,
    lorentz_indices=(),
    adjoint_indices=(),
    color_fund_indices=(),
    spinor_indices=(),
):
    canonical_expr, _, _ = canonize_spenso_tensors(
        expr,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
    )
    return canonical_expr


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
    coupling=gJJ * current_current(a_bar, a_psi, b_bar, b_psi, mu),
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
    coupling=gS * quark_gluon_current(i_bar_q, i_psi_q, mu, a_g, c_bar_q, c_psi_q),
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
    **scalar_gauge_current_term(gPhiA, mu, 1),
    alphas=[phiCdag0, phiC0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
)

L_complex_scalar_current_phidag = dict(
    **scalar_gauge_current_term(-gPhiA, mu, 0),
    alphas=[phiCdag0, phiC0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
)

L_complex_scalar_contact = dict(
    coupling=gPhiAA * scalar_gauge_contact(mu, nu),
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
PhiQEDField = Field(
    "PhiQED",
    spin=0,
    self_conjugate=False,
    symbol=phiC0,
    conjugate_symbol=phiCdag0,
    quantum_numbers={"Q": qPhi},
)
PhiQCDField = Field(
    "PhiQCD",
    spin=0,
    self_conjugate=False,
    symbol=phiQCD0,
    conjugate_symbol=phiQCDdag0,
    indices=(COLOR_FUND_INDEX,),
)
PhiMixField = Field(
    "PhiMix",
    spin=0,
    self_conjugate=False,
    symbol=phiMix0,
    conjugate_symbol=phiMixdag0,
    indices=(COLOR_FUND_INDEX,),
    quantum_numbers={"Q": qPhiMix},
)
PhiBiField = Field(
    "PhiBi",
    spin=0,
    self_conjugate=False,
    symbol=phiBi0,
    conjugate_symbol=phiBidag0,
    indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
)
PsiQEDField = Field(
    "PsiQED",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=psiQED0,
    conjugate_symbol=psibarQED0,
    indices=(SPINOR_INDEX,),
    quantum_numbers={"Q": qPsi},
)
PsiMixField = Field(
    "PsiMix",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=psiMix0,
    conjugate_symbol=psibarMix0,
    indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    quantum_numbers={"Q": qMix},
)
PsiField = Field("Psi", spin=Fraction(1, 2), self_conjugate=False, symbol=psi0, conjugate_symbol=psibar0, indices=(SPINOR_INDEX,))
GaugeField = Field("A", spin=1, self_conjugate=True, symbol=A0, indices=(LORENTZ_INDEX,))
QuarkField = Field("q", spin=Fraction(1, 2), self_conjugate=False, symbol=psi0, conjugate_symbol=psibar0, indices=(SPINOR_INDEX, COLOR_FUND_INDEX))
GluonField = Field("G", spin=1, self_conjugate=True, symbol=G0, indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX))
GhostGluonField = Field(
    "ghG",
    spin=0,
    kind="ghost",
    self_conjugate=False,
    symbol=ghG0,
    conjugate_symbol=ghGbar0,
    indices=(COLOR_ADJ_INDEX,),
)

COLOR_FUND_REP = GaugeRepresentation(
    index=COLOR_FUND_INDEX,
    generator_builder=gauge_generator,
    name="fundamental",
)
COLOR_FUND_REP_SUM = GaugeRepresentation(
    index=COLOR_FUND_INDEX,
    generator_builder=gauge_generator,
    name="fundamental_sum",
    slot_policy="sum",
)
COLOR_FUND_REP_SLOT0 = GaugeRepresentation(
    index=COLOR_FUND_INDEX,
    generator_builder=gauge_generator,
    name="fundamental_slot0",
    slot=0,
)
QCD_GROUP = GaugeGroup(
    name="SU3C",
    abelian=False,
    coupling=gS,
    gauge_boson=G0,
    structure_constant=structure_constant,
    representations=(COLOR_FUND_REP,),
)
QCD_GROUP_GHOST = GaugeGroup(
    name="SU3C",
    abelian=False,
    coupling=gS,
    gauge_boson=G0,
    ghost_field=ghG0,
    structure_constant=structure_constant,
    representations=(COLOR_FUND_REP,),
)
QCD_GROUP_BISLOT = GaugeGroup(
    name="SU3CBi",
    abelian=False,
    coupling=gS,
    gauge_boson=G0,
    structure_constant=structure_constant,
    representations=(COLOR_FUND_REP_SLOT0,),
)
QCD_GROUP_BISLOT_SUM = GaugeGroup(
    name="SU3CBiSum",
    abelian=False,
    coupling=gS,
    gauge_boson=G0,
    structure_constant=structure_constant,
    representations=(COLOR_FUND_REP_SUM,),
)
QCD_GROUP_AMBIGUOUS = GaugeGroup(
    name="SU3CAmbiguous",
    abelian=False,
    coupling=gS,
    gauge_boson=G0,
    structure_constant=structure_constant,
    representations=(COLOR_FUND_REP,),
)
QED_GROUP = GaugeGroup(
    name="U1QED",
    abelian=True,
    coupling=eQED,
    gauge_boson="A",
    charge="Q",
)
MODEL_QCD_BASE = Model(
    name="QCD-minimal",
    gauge_groups=(QCD_GROUP,),
    fields=(QuarkField, GluonField),
)
MODEL_SCALAR_QED_BASE = Model(
    name="ScalarQED-minimal",
    gauge_groups=(QED_GROUP,),
    fields=(PhiQEDField, GaugeField),
)
MODEL_SCALAR_QCD_BASE = Model(
    name="ScalarQCD-minimal",
    gauge_groups=(QCD_GROUP,),
    fields=(PhiQCDField, GluonField),
)
MODEL_SCALAR_QCD_BISLOT_BASE = Model(
    name="ScalarQCD-bislot-minimal",
    gauge_groups=(QCD_GROUP_BISLOT,),
    fields=(PhiBiField, GluonField),
)
MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM = Model(
    name="ScalarQCD-bislot-covariant-sum",
    gauge_groups=(QCD_GROUP_BISLOT_SUM,),
    fields=(PhiBiField, GluonField),
    covariant_terms=(ComplexScalarKineticTerm(field=PhiBiField),),
)
MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS = Model(
    name="ScalarQCD-bislot-ambiguous",
    gauge_groups=(QCD_GROUP_AMBIGUOUS,),
    fields=(PhiBiField, GluonField),
)
MODEL_QED_FERMION_BASE = Model(
    name="FermionQED-minimal",
    gauge_groups=(QED_GROUP,),
    fields=(PsiQEDField, GaugeField),
)
MODEL_QCD_COVARIANT = Model(
    name="QCD-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(QuarkField, GluonField),
    covariant_terms=(DiracKineticTerm(field=QuarkField),),
)
MODEL_SCALAR_QED_COVARIANT = Model(
    name="ScalarQED-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(PhiQEDField, GaugeField),
    covariant_terms=(ComplexScalarKineticTerm(field=PhiQEDField),),
)
MODEL_SCALAR_QCD_COVARIANT = Model(
    name="ScalarQCD-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(PhiQCDField, GluonField),
    covariant_terms=(ComplexScalarKineticTerm(field=PhiQCDField),),
)
MODEL_QED_FERMION_COVARIANT = Model(
    name="FermionQED-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(PsiQEDField, GaugeField),
    covariant_terms=(DiracKineticTerm(field=PsiQEDField),),
)
MODEL_MIXED_FERMION_COVARIANT = Model(
    name="MixedQCDQED-covariant",
    gauge_groups=(QCD_GROUP, QED_GROUP),
    fields=(PsiMixField, GluonField, GaugeField),
    covariant_terms=(DiracKineticTerm(field=PsiMixField),),
)
MODEL_MIXED_SCALAR_COVARIANT = Model(
    name="MixedScalarQCDQED-covariant",
    gauge_groups=(QCD_GROUP, QED_GROUP),
    fields=(PhiMixField, GluonField, GaugeField),
    covariant_terms=(ComplexScalarKineticTerm(field=PhiMixField),),
)
MODEL_QED_GAUGE_COVARIANT = Model(
    name="QEDGauge-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=QED_GROUP),),
)
MODEL_QCD_GAUGE_COVARIANT = Model(
    name="QCDGauge-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=QCD_GROUP),),
)
MODEL_QED_GAUGE_FIXING_COVARIANT = Model(
    name="QEDGaugeFixing-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    gauge_fixing_terms=(GaugeFixingTerm(gauge_group=QED_GROUP, xi=xiQED),),
)
MODEL_QCD_GAUGE_FIXING_COVARIANT = Model(
    name="QCDGaugeFixing-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    gauge_fixing_terms=(GaugeFixingTerm(gauge_group=QCD_GROUP, xi=xiQCD),),
)
MODEL_QED_ORDINARY_GAUGE_FIXED = Model(
    name="QEDGaugeFixed-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=QED_GROUP),),
    gauge_fixing_terms=(GaugeFixingTerm(gauge_group=QED_GROUP, xi=xiQED),),
)
MODEL_QCD_GHOST_COVARIANT = Model(
    name="QCDGhost-covariant",
    gauge_groups=(QCD_GROUP_GHOST,),
    fields=(GluonField, GhostGluonField),
    ghost_terms=(GhostTerm(gauge_group=QCD_GROUP_GHOST),),
)
MODEL_QCD_ORDINARY_GAUGE_FIXED = Model(
    name="QCDGaugeFixed-covariant",
    gauge_groups=(QCD_GROUP_GHOST,),
    fields=(GluonField, GhostGluonField),
    gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=QCD_GROUP_GHOST),),
    gauge_fixing_terms=(GaugeFixingTerm(gauge_group=QCD_GROUP_GHOST, xi=xiQCD),),
    ghost_terms=(GhostTerm(gauge_group=QCD_GROUP_GHOST),),
)


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
    coupling=gV * psi_bar_gamma_psi(i_psi_bar, i_psi, mu),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: i_psi_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: i_psi}),
        GaugeField.occurrence(labels={LORENTZ_KIND: mu}),
    ),
    label="gV * psibar gamma^mu psi A_mu",
)

TERM_axial_current = InteractionTerm(
    coupling=gV * psi_bar_gamma_psi(i_psi_bar, alpha_s, mu) * psi_bar_gamma5_psi(alpha_s, i_psi),
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
    coupling=gJJ * current_current(a_bar, a_psi, b_bar, b_psi, mu),
    fields=(
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: a_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: a_psi}),
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: b_bar}),
        PsiField.occurrence(labels={SPINOR_KIND: b_psi}),
    ),
    label="gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)",
)

TERM_quark_gluon = InteractionTerm(
    coupling=gS * quark_gluon_current(i_bar_q, i_psi_q, mu, a_g, c_bar_q, c_psi_q),
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
LEGS_qed_fermion = (
    PsiQEDField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1}),
    PsiQEDField.leg(p2, spin=s2, labels={SPINOR_KIND: i2}),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}),
)
LEGS_mixed_fermion_gluon = (
    PsiMixField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1, COLOR_FUND_KIND: c1}),
    PsiMixField.leg(p2, spin=s2, labels={SPINOR_KIND: i2, COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
)
LEGS_mixed_fermion_qed = (
    PsiMixField.leg(p1, conjugated=True, spin=s1, labels={SPINOR_KIND: i1, COLOR_FUND_KIND: c1}),
    PsiMixField.leg(p2, spin=s2, labels={SPINOR_KIND: i2, COLOR_FUND_KIND: c2}),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}),
)
LEGS_mixed_scalar_gluon = (
    PhiMixField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
    PhiMixField.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
)
LEGS_mixed_scalar_qed = (
    PhiMixField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
    PhiMixField.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}, species=b3),
)
LEGS_mixed_scalar_contact = (
    PhiMixField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
    PhiMixField.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
    GaugeField.leg(p4, labels={LORENTZ_KIND: mu4}, species=b4),
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
LEGS_compiled_scalar_current = (
    PhiQEDField.leg(p1, conjugated=True, species=b1),
    PhiQEDField.leg(p2, species=b2),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}, species=b3),
)
LEGS_compiled_scalar_qcd_current = (
    PhiQCDField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
    PhiQCDField.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
)
LEGS_compiled_scalar_bislot_current = (
    PhiBiField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: (c1, c3)}),
    PhiBiField.leg(p2, species=b2, labels={COLOR_FUND_KIND: (c2, c4)}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
)
LEGS_compiled_scalar_bislot_contact = (
    PhiBiField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: (c1, c3)}),
    PhiBiField.leg(p2, species=b2, labels={COLOR_FUND_KIND: (c2, c4)}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
    GluonField.leg(p4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}, species=b4),
)

TERM_complex_scalar_contact = InteractionTerm(
    coupling=gPhiAA * scalar_gauge_contact(mu, nu),
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
LEGS_compiled_scalar_contact = (
    PhiQEDField.leg(p1, conjugated=True, species=b1),
    PhiQEDField.leg(p2, species=b2),
    GaugeField.leg(p3, labels={LORENTZ_KIND: mu3}, species=b3),
    GaugeField.leg(p4, labels={LORENTZ_KIND: mu4}, species=b4),
)
LEGS_compiled_scalar_qcd_contact = (
    PhiQCDField.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
    PhiQCDField.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
    GluonField.leg(p3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}, species=b3),
    GluonField.leg(p4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}, species=b4),
)
LEGS_photon_kinetic = (
    GaugeField.leg(p1, species=b1, labels={LORENTZ_KIND: mu3}),
    GaugeField.leg(p2, species=b2, labels={LORENTZ_KIND: mu4}),
)
LEGS_gluon_kinetic = (
    GluonField.leg(p1, species=b1, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    GluonField.leg(p2, species=b2, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
)
LEGS_three_gluon = (
    GluonField.leg(p1, species=b1, labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a3}),
    GluonField.leg(p2, species=b2, labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: a4}),
    GluonField.leg(p3, species=b3, labels={LORENTZ_KIND: rho, COLOR_ADJ_KIND: a5}),
)
LEGS_four_gluon = (
    GluonField.leg(p1, species=b1, labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a3}),
    GluonField.leg(p2, species=b2, labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: a4}),
    GluonField.leg(p3, species=b3, labels={LORENTZ_KIND: rho, COLOR_ADJ_KIND: a5}),
    GluonField.leg(p4, species=b4, labels={LORENTZ_KIND: sigma, COLOR_ADJ_KIND: a6}),
)
LEGS_ghost_bilinear = (
    GhostGluonField.leg(p1, conjugated=True, species=b1, labels={COLOR_ADJ_KIND: a3}),
    GhostGluonField.leg(p2, species=b2, labels={COLOR_ADJ_KIND: a4}),
)
LEGS_ghost_gluon = (
    GhostGluonField.leg(p1, conjugated=True, species=b1, labels={COLOR_ADJ_KIND: a3}),
    GluonField.leg(p2, species=b2, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a4}),
    GhostGluonField.leg(p3, species=b3, labels={COLOR_ADJ_KIND: a5}),
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
        "scalar: derivative-contracted phi^4",
        description="gD2 * (d_mu phi)(d^mu phi) phi phi",
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


def _run_covariant_demo():
    compiled_qcd = compile_covariant_terms(MODEL_QCD_COVARIANT)
    compiled_qed = compile_covariant_terms(MODEL_QED_FERMION_COVARIANT)
    compiled_mixed = compile_covariant_terms(MODEL_MIXED_FERMION_COVARIANT)
    compiled_mixed_scalar = compile_covariant_terms(MODEL_MIXED_SCALAR_COVARIANT)
    compiled_scalar_qed = compile_covariant_terms(MODEL_SCALAR_QED_COVARIANT)
    compiled_scalar_qcd = compile_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)
    compiled_bislot_sum = compile_covariant_terms(MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM)
    compiled_photon = compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    compiled_yang_mills = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)
    photon_rho = compiled_photon[0].derivatives[0].lorentz_index
    ym_rho = compiled_yang_mills[0].derivatives[0].lorentz_index
    ym_internal = S("color_adj_mid_G_SU3C")

    print("# " + "=" * 79)
    print("Demo: covariant compiler\n")

    _print_vertex_block(
        "covariant: qbar i gamma^mu D_mu q",
        description=MODEL_QCD_COVARIANT.covariant_terms[0].label or "Dirac kinetic term expanded through the gauge compiler",
        interaction_terms=(compiled_qcd[0],),
        vertex=_model_demo_vertex(
            interaction=compiled_qcd[0],
            external_legs=LEGS_quark_gluon,
        ),
    )
    _print_vertex_block(
        "covariant: PsiQEDbar i gamma^mu D_mu PsiQED",
        description=MODEL_QED_FERMION_COVARIANT.covariant_terms[0].label or "Abelian Dirac kinetic term expanded through the gauge compiler",
        interaction_terms=(compiled_qed[0],),
        vertex=_model_demo_vertex(
            interaction=compiled_qed[0],
            external_legs=LEGS_qed_fermion,
        ),
    )
    _print_vertex_block(
        "covariant: one Dirac term over QCD+QED [gluon piece]",
        description="Single kinetic term expanded over all matching gauge groups",
        interaction_terms=(compiled_mixed[0],),
        vertex=_model_demo_vertex(
            interaction=compiled_mixed[0],
            external_legs=LEGS_mixed_fermion_gluon,
        ),
    )
    _print_vertex_block(
        "covariant: one Dirac term over QCD+QED [photon piece]",
        description="Same kinetic term, second gauge-group contribution",
        interaction_terms=(compiled_mixed[1],),
        vertex=_model_demo_vertex(
            interaction=compiled_mixed[1],
            external_legs=LEGS_mixed_fermion_qed,
        ),
    )
    mixed_scalar_qcd_terms = [
        term for term in compiled_mixed_scalar
        if "SU3C: scalar current" in term.label
    ]
    mixed_scalar_qed_terms = [
        term for term in compiled_mixed_scalar
        if "U1QED: scalar current" in term.label
    ]
    mixed_scalar_contact_terms = [
        term for term in compiled_mixed_scalar
        if "mixed contact" in term.label
    ]
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [gluon current]",
        description="Single complex-scalar kinetic term expanded over all matching gauge groups",
        interaction_terms=mixed_scalar_qcd_terms,
        vertex=sum(
            (
                _model_demo_vertex(
                    interaction=term,
                    external_legs=LEGS_mixed_scalar_gluon,
                    species_map={b1: phiMixdag0, b2: phiMix0, b3: G0},
                )
                for term in mixed_scalar_qcd_terms
            ),
            Expression.num(0),
        ).expand(),
    )
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [photon current]",
        description="Same kinetic term, abelian current with the color slot left as a spectator identity",
        interaction_terms=mixed_scalar_qed_terms,
        vertex=sum(
            (
                _model_demo_vertex(
                    interaction=term,
                    external_legs=LEGS_mixed_scalar_qed,
                    species_map={b1: phiMixdag0, b2: phiMix0, b3: A0},
                )
                for term in mixed_scalar_qed_terms
            ),
            Expression.num(0),
        ).expand(),
    )
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [mixed contact]",
        description="Ordered cross-group contact pieces from the same kinetic term, summed into one physical vertex",
        interaction_terms=mixed_scalar_contact_terms,
        vertex=sum(
            (
                _model_demo_vertex(
                    interaction=term,
                    external_legs=LEGS_mixed_scalar_contact,
                    species_map={b1: phiMixdag0, b2: phiMix0, b3: G0, b4: A0},
                )
                for term in mixed_scalar_contact_terms
            ),
            Expression.num(0),
        ).expand(),
    )
    _print_vertex_block(
        "covariant: (D_mu phi)^dagger (D^mu phi) current",
        description=MODEL_SCALAR_QED_COVARIANT.covariant_terms[0].label or "Complex-scalar kinetic term expanded through the gauge compiler",
        interaction_terms=compiled_scalar_qed[:2],
        vertex=(
            _model_demo_vertex(
                interaction=compiled_scalar_qed[0],
                external_legs=LEGS_compiled_scalar_current,
                species_map={b1: phiCdag0, b2: phiC0, b3: A0},
            )
            + _model_demo_vertex(
                interaction=compiled_scalar_qed[1],
                external_legs=LEGS_compiled_scalar_current,
                species_map={b1: phiCdag0, b2: phiC0, b3: A0},
            )
        ).expand(),
    )
    _print_vertex_block(
        "covariant: (D_mu phi)^dagger (D^mu phi) contact",
        description="Two-gauge contact contribution from the same complex-scalar kinetic term.",
        interaction_terms=(compiled_scalar_qed[2],),
        vertex=_model_demo_vertex(
            interaction=compiled_scalar_qed[2],
            external_legs=LEGS_compiled_scalar_contact,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiQCD)^dagger (D^mu PhiQCD) current",
        description=MODEL_SCALAR_QCD_COVARIANT.covariant_terms[0].label or "Non-abelian complex-scalar kinetic term expanded through the gauge compiler",
        interaction_terms=compiled_scalar_qcd[:2],
        vertex=(
            _model_demo_vertex(
                interaction=compiled_scalar_qcd[0],
                external_legs=LEGS_compiled_scalar_qcd_current,
                species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
            )
            + _model_demo_vertex(
                interaction=compiled_scalar_qcd[1],
                external_legs=LEGS_compiled_scalar_qcd_current,
                species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
            )
        ).expand(),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiQCD)^dagger (D^mu PhiQCD) contact",
        description="Two-gluon contact contribution with explicit generator ordering.",
        interaction_terms=(compiled_scalar_qcd[2],),
        vertex=_model_demo_vertex(
            interaction=compiled_scalar_qcd[2],
            external_legs=LEGS_compiled_scalar_qcd_contact,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
        ),
        canonical_vertex=_canonized_gauge_vertex(
            _model_demo_vertex(
                interaction=compiled_scalar_qcd[2],
                external_legs=LEGS_compiled_scalar_qcd_contact,
                species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
            ),
            lorentz_indices=(mu3, mu4),
            adjoint_indices=(a3, a4),
            color_fund_indices=(c1, c2, S("c_mid_PhiQCD_SU3C")),
        ),
        compact_override=I * (gS ** 2) * scalar_gauge_contact(mu3, mu4) * _symmetrized_generator_contact(a3, a4, c1, c2, S("c_mid_PhiQCD_SU3C")),
        interpretation="Raw vertex keeps both generator-order terms explicit; the compact override is the symmetrized color structure.",
    )
    _print_vertex_block(
        "covariant: (D_mu PhiBi)^dagger (D^mu PhiBi) [bislot, slot_policy='sum']",
        description="Bislotted scalar kinetic term expanded by summing over both identical color-fundamental slots.",
        interaction_terms=tuple(
            term for term in compiled_bislot_sum if "current" in term.label
        ),
        vertex=(
            sum(
                (
                    _model_demo_vertex(
                        interaction=term,
                        external_legs=LEGS_compiled_scalar_bislot_current,
                        species_map={b1: phiBidag0, b2: phiBi0, b3: G0},
                    )
                    for term in compiled_bislot_sum
                    if "current" in term.label
                ),
                Expression.num(0),
            )
        ).expand(),
        interpretation=(
            "Vertex shown here is the SUM of all compiled current terms (both slots). "
            "The compiled current terms are listed above, and the summed contact vertex follows below."
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiBi)^dagger (D^mu PhiBi) contact [bislot sum]",
        description="Sum of all ordered slot-pair contact contributions.",
        interaction_terms=tuple(
            term for term in compiled_bislot_sum if "contact" in term.label
        ),
        vertex=(
            sum(
                (
                    _model_demo_vertex(
                        interaction=term,
                        external_legs=LEGS_compiled_scalar_bislot_contact,
                        species_map={b1: phiBidag0, b2: phiBi0, b3: G0, b4: G0},
                    )
                    for term in compiled_bislot_sum
                    if "contact" in term.label
                ),
                Expression.num(0),
            )
        ).expand(),
    )
    _print_vertex_block(
        "covariant: -1/4 F_mu nu F^mu nu [abelian bilinear]",
        description=MODEL_QED_GAUGE_COVARIANT.gauge_kinetic_terms[0].label or "-1/4 U1QED field strength squared",
        interaction_terms=compiled_photon[:2],
        vertex=simplify_gamma_chain((
            _model_demo_vertex(
                interaction=compiled_photon[0],
                external_legs=LEGS_photon_kinetic,
                species_map={b1: A0, b2: A0},
            )
            + _model_demo_vertex(
                interaction=compiled_photon[1],
                external_legs=LEGS_photon_kinetic,
                species_map={b1: A0, b2: A0},
            )
        )).expand(),
        compact_override=I * gauge_kinetic_bilinear(mu3, mu4, p1, p2, photon_rho),
        interpretation="Compact override shows the convention-fixed bilinear after metric contraction.",
    )
    _print_vertex_block(
        "covariant: -1/4 G^a_mu nu G^{a mu nu} [bilinear]",
        description=MODEL_QCD_GAUGE_COVARIANT.gauge_kinetic_terms[0].label or "-1/4 SU3C field strength squared",
        interaction_terms=compiled_yang_mills[:2],
        vertex=simplify_gamma_chain((
            _model_demo_vertex(
                interaction=compiled_yang_mills[0],
                external_legs=LEGS_gluon_kinetic,
                species_map={b1: G0, b2: G0},
            )
            + _model_demo_vertex(
                interaction=compiled_yang_mills[1],
                external_legs=LEGS_gluon_kinetic,
                species_map={b1: G0, b2: G0},
            )
        )).expand(),
        compact_override=(
            I
            * gauge_kinetic_bilinear(mu3, mu4, p1, p2, ym_rho)
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        ),
        interpretation="Compact override keeps the adjoint delta explicit and contracts the derivative metrics.",
    )
    _print_vertex_block(
        "covariant: Yang-Mills 3-gauge vertex",
        description="Cubic self-interaction term from the non-abelian field strength.",
        interaction_terms=(compiled_yang_mills[2],),
        vertex=_model_demo_vertex(
            interaction=compiled_yang_mills[2],
            external_legs=LEGS_three_gluon,
            species_map={b1: G0, b2: G0, b3: G0},
            simplify_gamma=True,
        ).expand(),
        canonical_vertex=_canonized_gauge_vertex(
            _model_demo_vertex(
                interaction=compiled_yang_mills[2],
                external_legs=LEGS_three_gluon,
                species_map={b1: G0, b2: G0, b3: G0},
                simplify_gamma=True,
            ).expand(),
            lorentz_indices=(mu, nu, rho, S("rho_G_SU3C_cubic")),
            adjoint_indices=(a3, a4, a5),
        ),
        compact_override=gS * yang_mills_three_vertex_raw(a3, a4, a5, mu, nu, rho, p1, p2, p3),
        interpretation="Compact override is the convention-fixed Yang-Mills 3-gauge structure.",
    )
    _print_vertex_block(
        "covariant: Yang-Mills 4-gauge vertex",
        description="Quartic self-interaction term from the non-abelian field strength.",
        interaction_terms=(compiled_yang_mills[3],),
        vertex=_model_demo_vertex(
            interaction=compiled_yang_mills[3],
            external_legs=LEGS_four_gluon,
            species_map={b1: G0, b2: G0, b3: G0, b4: G0},
        ),
        canonical_vertex=_canonized_gauge_vertex(
            _model_demo_vertex(
                interaction=compiled_yang_mills[3],
                external_legs=LEGS_four_gluon,
                species_map={b1: G0, b2: G0, b3: G0, b4: G0},
            ),
            lorentz_indices=(mu, nu, rho, sigma),
            adjoint_indices=(a3, a4, a5, a6, S("color_adj_mid_G_SU3C")),
        ),
        compact_override=(
            -I
            * Expression.num(1)
            / Expression.num(2)
            * (gS ** 2)
            * yang_mills_four_vertex_raw(a3, a4, a5, a6, mu, nu, rho, sigma, ym_internal)
        ),
        interpretation="Compact override groups the quartic color structures by Lorentz-metric channel.",
    )


def _run_gauge_fixed_demo():
    compiled_qed_gf = compile_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT)
    compiled_qcd_gf = compile_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT)
    compiled_qed_gauge_fixed = compile_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)
    compiled_qcd_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    compiled_qcd_gauge_fixed = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)

    photon_metric, photon_cross, photon_gf = compiled_qed_gauge_fixed
    photon_rho = photon_metric.derivatives[0].lorentz_index
    qcd_ghost_bilinear, qcd_ghost_gauge = compiled_qcd_ghost
    gluon_metric, gluon_cross, _, _, gluon_gf, _, _ = compiled_qcd_gauge_fixed
    gluon_rho = gluon_metric.derivatives[0].lorentz_index
    ghost_rho = S("rho_ghost")

    print("# " + "=" * 79)
    print("Demo: ordinary gauge fixing and ghosts\n")

    _print_vertex_block(
        "gauge-fixed: -(1/2 xi) (partial.A)^2 [abelian]",
        description="Ordinary linear-covariant gauge fixing for the abelian gauge field.",
        interaction_terms=(compiled_qed_gf[0],),
        show_term_objects=True,
        vertex=_model_demo_vertex(
            interaction=compiled_qed_gf[0],
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        ),
        compact_override=(I / xiQED) * gauge_fixing_bilinear(mu3, mu4, p1, p2),
        interpretation=(
            "The raw vertex is the symmetrized bosonic contraction of the two differentiated gauge fields. "
            "The compact form is the ordinary gauge-fixing bilinear."
        ),
    )
    _print_vertex_block(
        "gauge-fixed: -(1/2 xi) (partial.G)^2 [non-abelian]",
        description="The same ordinary linear-covariant gauge fixing for the gluon field, with the adjoint identity explicit.",
        interaction_terms=(compiled_qcd_gf[0],),
        show_term_objects=True,
        vertex=_model_demo_vertex(
            interaction=compiled_qcd_gf[0],
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        ),
        compact_override=(
            (I / xiQCD)
            * gauge_fixing_bilinear(mu3, mu4, p1, p2)
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        ),
        interpretation="Compared to the abelian case, the only extra structure is the adjoint delta.",
    )
    _print_vertex_block(
        "gauge-fixed: ordinary photon bilinear",
        description="Gauge kinetic plus ordinary gauge fixing combined into the full two-point photon vertex.",
        interaction_terms=(photon_metric, photon_cross, photon_gf),
        show_term_objects=True,
        vertex=simplify_gamma_chain((
            _model_demo_vertex(
                interaction=photon_metric,
                external_legs=LEGS_photon_kinetic,
                species_map={b1: A0, b2: A0},
            )
            + _model_demo_vertex(
                interaction=photon_cross,
                external_legs=LEGS_photon_kinetic,
                species_map={b1: A0, b2: A0},
            )
            + _model_demo_vertex(
                interaction=photon_gf,
                external_legs=LEGS_photon_kinetic,
                species_map={b1: A0, b2: A0},
            )
        )).expand(),
        compact_override=I * (
            gauge_kinetic_bilinear(mu3, mu4, p1, p2, photon_rho)
            + gauge_fixing_bilinear(mu3, mu4, p1, p2) / xiQED
        ),
        interpretation="This is the ordinary gauge-fixed abelian two-point structure in the current compiler conventions.",
    )
    _print_vertex_block(
        "gauge-fixed: Faddeev-Popov ghost bilinear",
        description="Ordinary non-abelian ghost kinetic term compiled from -cbar partial.D c in integrated form.",
        interaction_terms=(qcd_ghost_bilinear,),
        show_term_objects=True,
        vertex=_model_demo_vertex(
            interaction=qcd_ghost_bilinear,
            external_legs=LEGS_ghost_bilinear,
            species_map={b1: ghGbar0, b2: ghG0},
        ),
        compact_override=-I * ghost_kinetic(a3, a4, p1, p2, ghost_rho),
        interpretation="The compact form makes the adjoint delta and the p_bar.p_ghost contraction explicit.",
    )
    _print_vertex_block(
        "gauge-fixed: ghost-gluon interaction",
        description="Ordinary non-abelian ghost coupling. The derivative acts on the antighost, so the cubic vertex carries the antighost momentum.",
        interaction_terms=(qcd_ghost_gauge,),
        show_term_objects=True,
        vertex=_model_demo_vertex(
            interaction=qcd_ghost_gauge,
            external_legs=LEGS_ghost_gluon,
            species_map={b1: ghGbar0, b2: G0, b3: ghG0},
        ),
        compact_override=-gS * ghost_gauge(a3, a4, a5, mu3, p1),
        interpretation="The raw compiler output keeps the Lorentz metric explicit; the compact form contracts it to p_antighost,mu.",
    )
    _print_vertex_block(
        "gauge-fixed: ordinary gluon bilinear",
        description="Yang-Mills bilinear plus ordinary gauge fixing combined into the full two-point gluon vertex.",
        interaction_terms=(gluon_metric, gluon_cross, gluon_gf),
        show_term_objects=True,
        vertex=simplify_gamma_chain((
            _model_demo_vertex(
                interaction=gluon_metric,
                external_legs=LEGS_gluon_kinetic,
                species_map={b1: G0, b2: G0},
            )
            + _model_demo_vertex(
                interaction=gluon_cross,
                external_legs=LEGS_gluon_kinetic,
                species_map={b1: G0, b2: G0},
            )
            + _model_demo_vertex(
                interaction=gluon_gf,
                external_legs=LEGS_gluon_kinetic,
                species_map={b1: G0, b2: G0},
            )
        )).expand(),
        compact_override=(
            I
            * (
                gauge_kinetic_bilinear(mu3, mu4, p1, p2, gluon_rho)
                + gauge_fixing_bilinear(mu3, mu4, p1, p2) / xiQCD
            )
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        ),
        interpretation=(
            "This is the ordinary gauge-fixed non-abelian two-point structure. "
            "The ghost sector is separate and starts at the antighost-ghost and antighost-gluon-ghost vertices above."
        ),
    )


def _run_demo_output(suite):
    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()
        _run_mixed_demo()
    if suite in ("gauge", "all"):
        _run_gauge_demo()
    if suite == "covariant":
        _run_covariant_demo()
    if suite == "gaugefix":
        _run_gauge_fixed_demo()
    if suite == "model":
        compiled_qcd = compile_minimal_gauge_interactions(MODEL_QCD_BASE)
        compiled_qed = compile_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
        compiled_scalar_qed = compile_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
        compiled_scalar_qcd = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)

        print("# " + "=" * 79)
        print("Demo: model-layer\n")
        _print_vertex_block(
            "model: Yukawa [amputated]",
            description=TERM_yukawa.label,
            interaction_terms=(TERM_yukawa,),
            vertex=_model_demo_vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        )
        _print_vertex_block(
            "model: quark-gluon",
            description=TERM_quark_gluon.label,
            interaction_terms=(TERM_quark_gluon,),
            vertex=_model_demo_vertex(
                interaction=TERM_quark_gluon,
                external_legs=LEGS_quark_gluon,
            ),
        )
        _print_vertex_block(
            "minimal gauge compiler: quark-gluon",
            description="Compiled non-abelian fermion current from model metadata.",
            interaction_terms=(compiled_qcd[0],),
            vertex=_model_demo_vertex(
                interaction=compiled_qcd[0],
                external_legs=LEGS_quark_gluon,
            ),
        )
        _print_vertex_block(
            "minimal gauge compiler: fermion QED",
            description="Compiled abelian fermion current from charge metadata.",
            interaction_terms=(compiled_qed[0],),
            vertex=_model_demo_vertex(
                interaction=compiled_qed[0],
                external_legs=LEGS_qed_fermion,
            ),
        )
        _print_vertex_block(
            "minimal gauge compiler: scalar QED current",
            description="compiled U(1) current from gauge group + field charge",
            interaction_terms=compiled_scalar_qed[:2],
            vertex=(
                _model_demo_vertex(
                    interaction=compiled_scalar_qed[0],
                    external_legs=LEGS_compiled_scalar_current,
                    species_map={b1: phiCdag0, b2: phiC0, b3: A0},
                )
                + _model_demo_vertex(
                    interaction=compiled_scalar_qed[1],
                    external_legs=LEGS_compiled_scalar_current,
                    species_map={b1: phiCdag0, b2: phiC0, b3: A0},
                )
            ).expand(),
        )
        _print_vertex_block(
            "minimal gauge compiler: scalar QED contact",
            description="Two-photon contact compiled from the same scalar kinetic term.",
            interaction_terms=(compiled_scalar_qed[2],),
            vertex=_model_demo_vertex(
                interaction=compiled_scalar_qed[2],
                external_legs=LEGS_compiled_scalar_contact,
                species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
            ),
        )
        _print_vertex_block(
            "minimal gauge compiler: scalar QCD current",
            description="compiled SU(3) scalar current from representation metadata",
            interaction_terms=compiled_scalar_qcd[:2],
            vertex=(
                _model_demo_vertex(
                    interaction=compiled_scalar_qcd[0],
                    external_legs=LEGS_compiled_scalar_qcd_current,
                    species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
                )
                + _model_demo_vertex(
                    interaction=compiled_scalar_qcd[1],
                    external_legs=LEGS_compiled_scalar_qcd_current,
                    species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
                )
            ).expand(),
        )
        _print_vertex_block(
            "minimal gauge compiler: scalar QCD contact",
            description="Two-gluon scalar contact compiled from representation metadata.",
            interaction_terms=(compiled_scalar_qcd[2],),
            vertex=_model_demo_vertex(
                interaction=compiled_scalar_qcd[2],
                external_legs=LEGS_compiled_scalar_qcd_contact,
                species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
            ),
        )
    if suite == "cross":
        print("# " + "=" * 79)
        print("Demo: cross-checks\n")
        _print_vertex_block(
            "cross: current-current",
            description="model-layer and direct API should agree after gamma simplification",
            interaction_terms=(TERM_current_current,),
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
    _check(V, COMPACT_DERIV, "Generic derivative combinatorics (mu,nu)")

    V = simplify_deltas(vertex_factor(**L_deriv2, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV2, "Derivative-contracted (mu,mu)")

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

    try:
        vertex_factor(
            **L_yukawa,
            x=x,
            d=d,
            leg_index_labels=[
                {SPINOR_KIND: i1},
                {},
                {},
            ],
        )
    except ValueError as exc:
        assert "all fermion external legs" in str(exc)
        print("  Missing fermion leg spinor index -> ValueError: PASS")
    else:
        raise AssertionError("Partial fermion leg spinor labels should be rejected")

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
        show_vertex=True,
        description=TERM_phi4.label,
    )
    _check(
        _model_vertex(interaction=TERM_phi2chi2, external_legs=LEGS_phi2chi2,
                      species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0}),
        4 * I * g_sym * D4,
        "Model: phi^2 chi^2",
        show_vertex=True,
        description=TERM_phi2chi2.label,
    )
    _check(
        _model_vertex(interaction=TERM_phiCdag_phiC, external_legs=LEGS_phiCdag_phiC,
                      species_map={b1: phiCdag0, b2: phiC0}),
        I * lamC * (2 * pi) ** d * Delta(p1 + p2),
        "Model: phi^dag phi",
        show_vertex=True,
        description=TERM_phiCdag_phiC.label,
    )

    # Fermion
    _check(
        _model_vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        I * yF * G12 * D3,
        "Model: Yukawa amputated",
        show_vertex=True,
        description=TERM_yukawa.label,
    )
    _check(
        _model_vertex(interaction=TERM_vec_current, external_legs=LEGS_vec_current),
        I * gV * psi_bar_gamma_psi(i1, i2, mu3) * D3,
        "Model: Vector current",
        show_vertex=True,
        description=TERM_vec_current.label,
    )
    _check(
        _model_vertex(interaction=TERM_axial_current, external_legs=LEGS_vec_current),
        I * gV * psi_bar_gamma_psi(i1, alpha_s, mu3) * psi_bar_gamma5_psi(alpha_s, i2) * D3,
        "Model: Axial current",
        show_vertex=True,
        description=TERM_axial_current.label,
    )

    V_sp = _model_vertex(interaction=TERM_psibar_psi_sq, external_legs=LEGS_fermion4)
    expected_sp = (
        -I * g_psi4 * D4
        * (psi_bar_psi(i1, i2) * psi_bar_psi(i3, i4)
           - psi_bar_psi(i1, i4) * psi_bar_psi(i3, i2))
    )
    _check(
        V_sp,
        expected_sp,
        "Model: (psibar psi)^2 amputated",
        show_vertex=True,
        description=TERM_psibar_psi_sq.label,
    )

    V_jj = _model_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)
    expected_jj = (
        2 * I * gJJ * D4
        * (psi_bar_gamma_psi(i1, i2, mu) * psi_bar_gamma_psi(i3, i4, mu)
           - psi_bar_gamma_psi(i1, i4, mu) * psi_bar_gamma_psi(i3, i2, mu))
    )
    _check(
        simplify_gamma_chain(V_jj),
        expected_jj,
        "Model: Current-current",
        show_vertex=True,
        description=TERM_current_current.label,
    )

    # Gauge-ready
    V_qg = _model_vertex(interaction=TERM_quark_gluon, external_legs=LEGS_quark_gluon)
    expected_qg = I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * D3
    _check(
        V_qg,
        expected_qg,
        "Model: Quark-gluon",
        show_vertex=True,
        description=TERM_quark_gluon.label,
    )

    V_sc = (
        _model_vertex(interaction=TERM_complex_scalar_current_phi,
                      external_legs=LEGS_complex_scalar_current, species_map=sm_scalar_gauge)
        + _model_vertex(interaction=TERM_complex_scalar_current_phidag,
                        external_legs=LEGS_complex_scalar_current, species_map=sm_scalar_gauge)
    )
    assert V_sc.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    _print_vertex_block(
        "Model: Complex scalar current",
        description="model-layer complex scalar current combination",
        vertex=V_sc,
    )
    print("  Model: Complex scalar current (non-zero): PASS")

    V_ct = _model_vertex(interaction=TERM_complex_scalar_contact,
                         external_legs=LEGS_complex_scalar_contact, species_map=sm_contact)
    assert V_ct.expand().to_canonical_string() != Expression.num(0).to_canonical_string()
    _print_vertex_block(
        "Model: Complex scalar contact",
        description=TERM_complex_scalar_contact.label,
        vertex=V_ct,
    )
    print("  Model: Complex scalar contact (non-zero): PASS")

    print("\n  Model-layer tests passed.\n")


# ===================================================================
# Compiled gauge-model tests
# ===================================================================

def _run_compiled_gauge_tests():
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)

    compiled_qcd = compile_minimal_gauge_interactions(MODEL_QCD_BASE)
    model_qcd = with_minimal_gauge_interactions(MODEL_QCD_BASE)
    assert model_qcd.interactions == compiled_qcd
    assert len(compiled_qcd) == 1

    term_qcd = compiled_qcd[0]
    _check(
        _model_vertex(interaction=term_qcd, external_legs=LEGS_quark_gluon),
        I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * D3,
        "Compiled model: quark-gluon",
        show_vertex=True,
        description=term_qcd.label,
    )

    compiled_qed = compile_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
    model_qed = with_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
    assert model_qed.interactions == compiled_qed
    assert len(compiled_qed) == 1

    term_qed = compiled_qed[0]
    _check(
        _model_vertex(interaction=term_qed, external_legs=LEGS_qed_fermion),
        I * eQED * qPsi * psi_bar_gamma_psi(i1, i2, mu3) * D3,
        "Compiled model: fermion QED",
        show_vertex=True,
        description=term_qed.label,
    )

    compiled_scalar_qed = compile_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
    model_scalar_qed = with_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
    assert model_scalar_qed.interactions == compiled_scalar_qed
    assert len(compiled_scalar_qed) == 3

    term_sc_phi, term_sc_phidag, term_sc_contact = compiled_scalar_qed
    scalar_current_index = term_sc_phi.derivatives[0].lorentz_index

    V_sc = (
        _model_vertex(
            interaction=term_sc_phi,
            external_legs=LEGS_compiled_scalar_current,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0},
        )
        + _model_vertex(
            interaction=term_sc_phidag,
            external_legs=LEGS_compiled_scalar_current,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0},
        )
    )
    expected_sc = eQED * qPhi * (pcomp(p2, scalar_current_index) - pcomp(p1, scalar_current_index)) * D3
    _check(
        V_sc,
        expected_sc,
        "Compiled model: scalar QED current",
        show_vertex=True,
        description="U(1) current pair compiled from field charge and gauge group",
    )

    _check(
        _model_vertex(
            interaction=term_sc_contact,
            external_legs=LEGS_compiled_scalar_contact,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
        ),
        2 * I * (eQED ** 2) * (qPhi ** 2) * scalar_gauge_contact(mu3, mu4) * D4,
        "Compiled model: scalar QED contact",
        show_vertex=True,
        description=term_sc_contact.label,
    )

    compiled_scalar_qcd = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)
    model_scalar_qcd = with_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)
    assert model_scalar_qcd.interactions == compiled_scalar_qcd
    assert len(compiled_scalar_qcd) == 3

    term_sqcd_phi, term_sqcd_phidag, term_sqcd_contact = compiled_scalar_qcd
    scalar_qcd_index = term_sqcd_phi.derivatives[0].lorentz_index
    sqcd_internal = S("c_mid_PhiQCD_SU3C")

    V_sqcd = (
        _model_vertex(
            interaction=term_sqcd_phi,
            external_legs=LEGS_compiled_scalar_qcd_current,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
        )
        + _model_vertex(
            interaction=term_sqcd_phidag,
            external_legs=LEGS_compiled_scalar_qcd_current,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
        )
    )
    expected_sqcd = (
        gS
        * gauge_generator(a3, c1, c2)
        * (pcomp(p2, scalar_qcd_index) - pcomp(p1, scalar_qcd_index))
        * D3
    )
    _check(
        V_sqcd,
        expected_sqcd,
        "Compiled model: scalar QCD current",
        show_vertex=True,
        description="SU(3) current pair compiled from scalar representation metadata",
    )

    expected_sqcd_contact = (
        I
        * (gS ** 2)
        * scalar_gauge_contact(mu3, mu4)
        * _symmetrized_generator_contact(a3, a4, c1, c2, sqcd_internal)
        * D4
    )
    _check(
        _model_vertex(
            interaction=term_sqcd_contact,
            external_legs=LEGS_compiled_scalar_qcd_contact,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
        ),
        expected_sqcd_contact,
        "Compiled model: scalar QCD contact",
        show_vertex=True,
        description=term_sqcd_contact.label,
        display_vertex=expected_sqcd_contact,
    )

    try:
        compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS)
    except ValueError as exc:
        assert "GaugeRepresentation(slot" in str(exc)
        print("  Compiled model: repeated-slot ambiguity rejected: PASS")
    else:
        raise AssertionError("Repeated same-kind representation slot should require GaugeRepresentation(slot=...)")

    compiled_bislot = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_BASE)
    assert len(compiled_bislot) == 3
    term_bislot_phi, term_bislot_phidag, _ = compiled_bislot
    bislot_current_index = term_bislot_phi.derivatives[0].lorentz_index
    spectator_identity = COLOR_FUND_INDEX.representation.g(c3, c4).to_expression()
    V_bislot = (
        _model_vertex(
            interaction=term_bislot_phi,
            external_legs=LEGS_compiled_scalar_bislot_current,
            species_map={b1: phiBidag0, b2: phiBi0, b3: G0},
        )
        + _model_vertex(
            interaction=term_bislot_phidag,
            external_legs=LEGS_compiled_scalar_bislot_current,
            species_map={b1: phiBidag0, b2: phiBi0, b3: G0},
        )
    )
    _check(
        V_bislot,
        gS
        * gauge_generator(a3, c1, c2)
        * spectator_identity
        * (pcomp(p2, bislot_current_index) - pcomp(p1, bislot_current_index))
        * D3,
        "Compiled model: repeated-slot scalar QCD current",
        show_vertex=True,
        description="Active color slot is explicit; the second identical color slot remains a spectator identity.",
    )

    print("\n  Compiled gauge-model tests passed.\n")


# ===================================================================
# Covariant-derivative compiler tests
# ===================================================================

def _run_covariant_compiler_tests():
    D2 = (2 * pi) ** d * Delta(p1 + p2)
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)

    compiled_qcd = compile_covariant_terms(MODEL_QCD_COVARIANT)
    model_qcd = with_compiled_covariant_terms(MODEL_QCD_COVARIANT)
    assert model_qcd.interactions == compiled_qcd
    assert len(compiled_qcd) == 1

    term_qcd = compiled_qcd[0]
    _check(
        _model_vertex(interaction=term_qcd, external_legs=LEGS_quark_gluon),
        -I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * D3,
        "Covariant compiler: quark-gluon",
        show_vertex=True,
        description=term_qcd.label,
    )

    compiled_qed = compile_covariant_terms(MODEL_QED_FERMION_COVARIANT)
    model_qed = with_compiled_covariant_terms(MODEL_QED_FERMION_COVARIANT)
    assert model_qed.interactions == compiled_qed
    assert len(compiled_qed) == 1

    term_qed = compiled_qed[0]
    _check(
        _model_vertex(interaction=term_qed, external_legs=LEGS_qed_fermion),
        -I * eQED * qPsi * psi_bar_gamma_psi(i1, i2, mu3) * D3,
        "Covariant compiler: fermion QED",
        show_vertex=True,
        description=term_qed.label,
    )

    rogue_photon = Field(
        "RogueA",
        spin=1,
        self_conjugate=True,
        symbol=S("RogueA"),
        indices=(LORENTZ_INDEX,),
    )
    rogue_qed_group = GaugeGroup(
        name="RogueU1",
        abelian=True,
        coupling=eQED,
        gauge_boson=rogue_photon,
        charge="Q",
    )
    rogue_qed_model = Model(
        name="rogue-qed-gauge-boson",
        gauge_groups=(rogue_qed_group,),
        fields=(PsiQEDField,),
        covariant_terms=(DiracKineticTerm(field=PsiQEDField),),
    )
    try:
        compile_covariant_terms(rogue_qed_model)
    except ValueError as exc:
        assert "declared in model.fields" in str(exc)
        print("  Undeclared gauge boson metadata rejected: PASS")
    else:
        raise AssertionError("Undeclared gauge boson metadata should be rejected")

    multi_rep_fermion = Field(
        "PsiMultiRep",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psiMultiRep"),
        conjugate_symbol=S("psibarMultiRep"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX, COLOR_ADJ_INDEX),
    )
    multi_rep_gluon = Field(
        "GMultiRep",
        spin=1,
        self_conjugate=True,
        symbol=S("GMultiRep"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    multi_rep_group = GaugeGroup(
        name="SU3MultiRep",
        abelian=False,
        coupling=gS,
        gauge_boson=multi_rep_gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
            GaugeRepresentation(
                index=COLOR_ADJ_INDEX,
                generator_builder=structure_constant,
                name="adjoint",
            ),
        ),
    )
    multi_rep_model = Model(
        name="multi-representation-rejection",
        gauge_groups=(multi_rep_group,),
        fields=(multi_rep_fermion, multi_rep_gluon),
        covariant_terms=(DiracKineticTerm(field=multi_rep_fermion),),
    )
    try:
        compile_covariant_terms(multi_rep_model)
    except ValueError as exc:
        assert "matches multiple representations" in str(exc)
        print("  Multiple non-abelian representation matches rejected: PASS")
    else:
        raise AssertionError("Multiple same-group representation matches should be rejected")

    compiled_mixed = compile_covariant_terms(MODEL_MIXED_FERMION_COVARIANT)
    model_mixed = with_compiled_covariant_terms(MODEL_MIXED_FERMION_COVARIANT)
    assert model_mixed.interactions == compiled_mixed
    assert len(compiled_mixed) == 2

    _check(
        _model_vertex(interaction=compiled_mixed[0], external_legs=LEGS_mixed_fermion_gluon),
        -I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * D3,
        "Covariant compiler: mixed fermion QCD piece",
        show_vertex=True,
        description="One kinetic term automatically expanded over all matching gauge groups",
    )
    _check(
        _model_vertex(interaction=compiled_mixed[1], external_legs=LEGS_mixed_fermion_qed),
        -I
        * eQED
        * qMix
        * psi_bar_gamma_psi(i1, i2, mu3)
        * COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
        * D3,
        "Covariant compiler: mixed fermion QED piece",
        show_vertex=True,
        description="Second contribution from the same mixed-group kinetic term",
    )

    compiled_mixed_scalar = compile_covariant_terms(MODEL_MIXED_SCALAR_COVARIANT)
    model_mixed_scalar = with_compiled_covariant_terms(MODEL_MIXED_SCALAR_COVARIANT)
    assert model_mixed_scalar.interactions == compiled_mixed_scalar
    assert len(compiled_mixed_scalar) == 8

    mixed_scalar_qcd_terms = [
        term for term in compiled_mixed_scalar
        if "SU3C: scalar current" in term.label
    ]
    mixed_scalar_qed_terms = [
        term for term in compiled_mixed_scalar
        if "U1QED: scalar current" in term.label
    ]
    mixed_scalar_contact_terms = [
        term for term in compiled_mixed_scalar
        if "mixed contact" in term.label
    ]
    assert len(mixed_scalar_qcd_terms) == 2
    assert len(mixed_scalar_qed_terms) == 2
    assert len(mixed_scalar_contact_terms) == 2

    mixed_scalar_qcd_index = mixed_scalar_qcd_terms[0].derivatives[0].lorentz_index
    V_mixed_scalar_qcd = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=LEGS_mixed_scalar_gluon,
                species_map={b1: phiMixdag0, b2: phiMix0, b3: G0},
            )
            for term in mixed_scalar_qcd_terms
        ),
        Expression.num(0),
    )
    _check(
        V_mixed_scalar_qcd,
        I
        * gS
        * gauge_generator(a3, c1, c2)
        * (pcomp(p2, mixed_scalar_qcd_index) - pcomp(p1, mixed_scalar_qcd_index))
        * D3,
        "Covariant compiler: mixed scalar QCD current",
        show_vertex=True,
        description="Single mixed-group scalar kinetic term, non-abelian current piece",
    )

    mixed_scalar_qed_index = mixed_scalar_qed_terms[0].derivatives[0].lorentz_index
    V_mixed_scalar_qed = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=LEGS_mixed_scalar_qed,
                species_map={b1: phiMixdag0, b2: phiMix0, b3: A0},
            )
            for term in mixed_scalar_qed_terms
        ),
        Expression.num(0),
    )
    _check(
        V_mixed_scalar_qed,
        I
        * eQED
        * qPhiMix
        * COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
        * (pcomp(p2, mixed_scalar_qed_index) - pcomp(p1, mixed_scalar_qed_index))
        * D3,
        "Covariant compiler: mixed scalar QED current",
        show_vertex=True,
        description="Same mixed-group scalar kinetic term, abelian current piece",
    )

    V_mixed_scalar_contact = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=LEGS_mixed_scalar_contact,
                species_map={b1: phiMixdag0, b2: phiMix0, b3: G0, b4: A0},
            )
            for term in mixed_scalar_contact_terms
        ),
        Expression.num(0),
    )
    _check(
        V_mixed_scalar_contact,
        2
        * I
        * gS
        * eQED
        * qPhiMix
        * gauge_generator(a3, c1, c2)
        * scalar_gauge_contact(mu3, mu4)
        * D4,
        "Covariant compiler: mixed scalar contact",
        show_vertex=True,
        description="Cross-group contact terms from one complex-scalar kinetic term",
    )

    compiled_scalar_qed = compile_covariant_terms(MODEL_SCALAR_QED_COVARIANT)
    model_scalar_qed = with_compiled_covariant_terms(MODEL_SCALAR_QED_COVARIANT)
    assert model_scalar_qed.interactions == compiled_scalar_qed
    assert len(compiled_scalar_qed) == 3

    term_sc_phi, term_sc_phidag, term_sc_contact = compiled_scalar_qed
    scalar_current_index = term_sc_phi.derivatives[0].lorentz_index

    V_sc = (
        _model_vertex(
            interaction=term_sc_phi,
            external_legs=LEGS_compiled_scalar_current,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0},
        )
        + _model_vertex(
            interaction=term_sc_phidag,
            external_legs=LEGS_compiled_scalar_current,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0},
        )
    )
    expected_sc = I * eQED * qPhi * (pcomp(p2, scalar_current_index) - pcomp(p1, scalar_current_index)) * D3
    _check(
        V_sc,
        expected_sc,
        "Covariant compiler: scalar QED current",
        show_vertex=True,
        description=term_sc_phi.label,
    )

    _check(
        _model_vertex(
            interaction=term_sc_contact,
            external_legs=LEGS_compiled_scalar_contact,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
        ),
        2 * I * (eQED ** 2) * (qPhi ** 2) * scalar_gauge_contact(mu3, mu4) * D4,
        "Covariant compiler: scalar QED contact",
        show_vertex=True,
        description=term_sc_contact.label,
    )

    compiled_scalar_qcd = compile_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)
    model_scalar_qcd = with_compiled_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)
    assert model_scalar_qcd.interactions == compiled_scalar_qcd
    assert len(compiled_scalar_qcd) == 3

    term_sqcd_phi, term_sqcd_phidag, term_sqcd_contact = compiled_scalar_qcd
    scalar_qcd_index = term_sqcd_phi.derivatives[0].lorentz_index
    sqcd_internal = S("c_mid_PhiQCD_SU3C")

    V_sqcd = (
        _model_vertex(
            interaction=term_sqcd_phi,
            external_legs=LEGS_compiled_scalar_qcd_current,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
        )
        + _model_vertex(
            interaction=term_sqcd_phidag,
            external_legs=LEGS_compiled_scalar_qcd_current,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0},
        )
    )
    expected_sqcd = (
        I
        * gS
        * gauge_generator(a3, c1, c2)
        * (pcomp(p2, scalar_qcd_index) - pcomp(p1, scalar_qcd_index))
        * D3
    )
    _check(
        V_sqcd,
        expected_sqcd,
        "Covariant compiler: scalar QCD current",
        show_vertex=True,
        description=term_sqcd_phi.label,
    )

    expected_sqcd_contact = (
        I
        * (gS ** 2)
        * scalar_gauge_contact(mu3, mu4)
        * _symmetrized_generator_contact(a3, a4, c1, c2, sqcd_internal)
        * D4
    )
    _check(
        _model_vertex(
            interaction=term_sqcd_contact,
            external_legs=LEGS_compiled_scalar_qcd_contact,
            species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
        ),
        expected_sqcd_contact,
        "Covariant compiler: scalar QCD contact",
        show_vertex=True,
        description=term_sqcd_contact.label,
        display_vertex=expected_sqcd_contact,
    )

    compiled_photon = compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    model_photon = with_compiled_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    assert model_photon.interactions == compiled_photon
    assert len(compiled_photon) == 2

    photon_metric, photon_cross = compiled_photon
    photon_rho = photon_metric.derivatives[0].lorentz_index
    photon_left = photon_cross.derivatives[0].lorentz_index
    photon_right = photon_cross.derivatives[1].lorentz_index
    V_photon = (
        _model_vertex(
            interaction=photon_metric,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        )
        + _model_vertex(
            interaction=photon_cross,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        )
    )
    V_photon = simplify_gamma_chain(V_photon)
    _check(
        V_photon,
        I * gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, photon_rho, photon_left, photon_right) * D2,
        "Covariant compiler: abelian gauge bilinear",
        show_vertex=True,
        description=MODEL_QED_GAUGE_COVARIANT.gauge_kinetic_terms[0].label or "-1/4 U1QED field strength squared",
        display_vertex=I * gauge_kinetic_bilinear(mu3, mu4, p1, p2, photon_rho) * D2,
    )

    compiled_yang_mills = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)
    model_yang_mills = with_compiled_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)
    assert model_yang_mills.interactions == compiled_yang_mills
    assert len(compiled_yang_mills) == 4

    ym_metric, ym_cross, ym_cubic, ym_quartic = compiled_yang_mills
    ym_rho = ym_metric.derivatives[0].lorentz_index
    ym_left = ym_cross.derivatives[0].lorentz_index
    ym_right = ym_cross.derivatives[1].lorentz_index
    ym_cubic_rho = ym_cubic.derivatives[0].lorentz_index
    ym_internal = S("color_adj_mid_G_SU3C")

    V_ym_bilinear = (
        _model_vertex(
            interaction=ym_metric,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        )
        + _model_vertex(
            interaction=ym_cross,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        )
    )
    V_ym_bilinear = simplify_gamma_chain(V_ym_bilinear)
    _check(
        V_ym_bilinear,
        I
        * gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, ym_rho, ym_left, ym_right)
        * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        * D2,
        "Covariant compiler: non-abelian gauge bilinear",
        show_vertex=True,
        description=MODEL_QCD_GAUGE_COVARIANT.gauge_kinetic_terms[0].label or "-1/4 SU3C field strength squared",
        display_vertex=(
            I
            * gauge_kinetic_bilinear(mu3, mu4, p1, p2, ym_rho)
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
            * D2
        ),
    )

    _check(
        simplify_gamma_chain(_model_vertex(
            interaction=ym_cubic,
            external_legs=LEGS_three_gluon,
            species_map={b1: G0, b2: G0, b3: G0},
        )),
        simplify_gamma_chain(
            gS
            * yang_mills_three_vertex_metric_raw(a3, a4, a5, mu, nu, rho, p1, p2, p3, ym_cubic_rho)
            * D3
        ),
        "Covariant compiler: Yang-Mills cubic",
        show_vertex=True,
        description=ym_cubic.label,
        display_vertex=gS * yang_mills_three_vertex_raw(a3, a4, a5, mu, nu, rho, p1, p2, p3) * D3,
    )

    _check(
        _model_vertex(
            interaction=ym_quartic,
            external_legs=LEGS_four_gluon,
            species_map={b1: G0, b2: G0, b3: G0, b4: G0},
        ),
        -I
        * (gS ** 2)
        * Expression.num(1)
        / Expression.num(2)
        * yang_mills_four_vertex_raw(a3, a4, a5, a6, mu, nu, rho, sigma, ym_internal)
        * D4,
        "Covariant compiler: Yang-Mills quartic",
        show_vertex=True,
        description=ym_quartic.label,
        display_vertex=(
            -I
            * Expression.num(1)
            / Expression.num(2)
            * (gS ** 2)
            * yang_mills_four_vertex_raw(a3, a4, a5, a6, mu, nu, rho, sigma, ym_internal)
            * D4
        ),
    )

    print("\n  Covariant / pure-gauge compiler tests passed.\n")


# ===================================================================
# Ordinary gauge-fixing / ghost compiler tests
# ===================================================================

def _run_gauge_fixed_compiler_tests():
    D2 = (2 * pi) ** d * Delta(p1 + p2)
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)

    compiled_qed_gf = compile_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT)
    model_qed_gf = with_compiled_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT)
    assert model_qed_gf.interactions == compiled_qed_gf
    assert len(compiled_qed_gf) == 1

    qed_gf_term = compiled_qed_gf[0]
    qed_gf_left = qed_gf_term.derivatives[0].lorentz_index
    qed_gf_right = qed_gf_term.derivatives[1].lorentz_index
    _check(
        _model_vertex(
            interaction=qed_gf_term,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        ),
        (I / xiQED) * gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, qed_gf_left, qed_gf_right) * D2,
        "Gauge-fixed compiler: abelian gauge fixing",
        show_vertex=True,
        description=qed_gf_term.label,
        display_vertex=(I / xiQED) * gauge_fixing_bilinear(mu3, mu4, p1, p2) * D2,
    )

    compiled_qcd_gf = compile_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT)
    model_qcd_gf = with_compiled_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT)
    assert model_qcd_gf.interactions == compiled_qcd_gf
    assert len(compiled_qcd_gf) == 1

    qcd_gf_term = compiled_qcd_gf[0]
    qcd_gf_left = qcd_gf_term.derivatives[0].lorentz_index
    qcd_gf_right = qcd_gf_term.derivatives[1].lorentz_index
    _check(
        _model_vertex(
            interaction=qcd_gf_term,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        ),
        (
            (I / xiQCD)
            * gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, qcd_gf_left, qcd_gf_right)
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
            * D2
        ),
        "Gauge-fixed compiler: non-abelian gauge fixing",
        show_vertex=True,
        description=qcd_gf_term.label,
        display_vertex=(
            (I / xiQCD)
            * gauge_fixing_bilinear(mu3, mu4, p1, p2)
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
            * D2
        ),
    )

    compiled_qed_gauge_fixed = compile_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)
    model_qed_gauge_fixed = with_compiled_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)
    assert model_qed_gauge_fixed.interactions == compiled_qed_gauge_fixed
    assert len(compiled_qed_gauge_fixed) == 3

    photon_metric, photon_cross, photon_gf = compiled_qed_gauge_fixed
    photon_rho = photon_metric.derivatives[0].lorentz_index
    photon_left = photon_cross.derivatives[0].lorentz_index
    photon_right = photon_cross.derivatives[1].lorentz_index
    photon_gf_left = photon_gf.derivatives[0].lorentz_index
    photon_gf_right = photon_gf.derivatives[1].lorentz_index
    V_photon_gauge_fixed = simplify_gamma_chain((
        _model_vertex(
            interaction=photon_metric,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        )
        + _model_vertex(
            interaction=photon_cross,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        )
        + _model_vertex(
            interaction=photon_gf,
            external_legs=LEGS_photon_kinetic,
            species_map={b1: A0, b2: A0},
        )
    ))
    _check(
        V_photon_gauge_fixed,
        simplify_gamma_chain(
            I
            * (
                gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, photon_rho, photon_left, photon_right)
                + gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, photon_gf_left, photon_gf_right) / xiQED
            )
            * D2
        ),
        "Gauge-fixed compiler: ordinary photon bilinear",
        show_vertex=True,
        description="Gauge kinetic plus ordinary gauge fixing",
        display_vertex=I * (
            gauge_kinetic_bilinear(mu3, mu4, p1, p2, photon_rho)
            + gauge_fixing_bilinear(mu3, mu4, p1, p2) / xiQED
        ) * D2,
    )

    compiled_qcd_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    model_qcd_ghost = with_compiled_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    assert model_qcd_ghost.interactions == compiled_qcd_ghost
    assert len(compiled_qcd_ghost) == 2

    qcd_ghost_bilinear, qcd_ghost_gauge = compiled_qcd_ghost
    ghost_mu = qcd_ghost_bilinear.derivatives[0].lorentz_index
    ghost_nu = qcd_ghost_bilinear.derivatives[1].lorentz_index
    _check(
        _model_vertex(
            interaction=qcd_ghost_bilinear,
            external_legs=LEGS_ghost_bilinear,
            species_map={b1: ghGbar0, b2: ghG0},
        ),
        -I * ghost_kinetic_raw(a3, a4, p1, p2, ghost_mu, ghost_nu) * D2,
        "Gauge-fixed compiler: ghost bilinear",
        show_vertex=True,
        description=qcd_ghost_bilinear.label,
        display_vertex=-I * ghost_kinetic(a3, a4, p1, p2, S("rho_ghost")) * D2,
    )

    ghost_rho = qcd_ghost_gauge.derivatives[0].lorentz_index
    _check(
        _model_vertex(
            interaction=qcd_ghost_gauge,
            external_legs=LEGS_ghost_gluon,
            species_map={b1: ghGbar0, b2: G0, b3: ghG0},
        ),
        -gS * ghost_gauge_raw(a3, a4, a5, mu3, ghost_rho, p1) * D3,
        "Gauge-fixed compiler: ghost-gluon interaction",
        show_vertex=True,
        description=qcd_ghost_gauge.label,
        display_vertex=-gS * ghost_gauge(a3, a4, a5, mu3, p1) * D3,
    )

    compiled_qcd_gauge_fixed = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)
    model_qcd_gauge_fixed = with_compiled_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)
    assert model_qcd_gauge_fixed.interactions == compiled_qcd_gauge_fixed
    assert len(compiled_qcd_gauge_fixed) == 7

    gluon_metric, gluon_cross, _, _, gluon_gf, _, _ = compiled_qcd_gauge_fixed
    gluon_rho = gluon_metric.derivatives[0].lorentz_index
    gluon_left = gluon_cross.derivatives[0].lorentz_index
    gluon_right = gluon_cross.derivatives[1].lorentz_index
    gluon_gf_left = gluon_gf.derivatives[0].lorentz_index
    gluon_gf_right = gluon_gf.derivatives[1].lorentz_index
    V_gluon_gauge_fixed = simplify_gamma_chain((
        _model_vertex(
            interaction=gluon_metric,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        )
        + _model_vertex(
            interaction=gluon_cross,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        )
        + _model_vertex(
            interaction=gluon_gf,
            external_legs=LEGS_gluon_kinetic,
            species_map={b1: G0, b2: G0},
        )
    ))
    _check(
        V_gluon_gauge_fixed,
        simplify_gamma_chain(
            I
            * (
                gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, gluon_rho, gluon_left, gluon_right)
                + gauge_fixing_bilinear_raw(mu3, mu4, p1, p2, gluon_gf_left, gluon_gf_right) / xiQCD
            )
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
            * D2
        ),
        "Gauge-fixed compiler: ordinary gluon bilinear",
        show_vertex=True,
        description="Yang-Mills bilinear plus ordinary gauge fixing",
        display_vertex=(
            I
            * (
                gauge_kinetic_bilinear(mu3, mu4, p1, p2, gluon_rho)
                + gauge_fixing_bilinear(mu3, mu4, p1, p2) / xiQCD
            )
            * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
            * D2
        ),
    )

    print("\n  Ordinary gauge-fixing / ghost compiler tests passed.\n")


# ===================================================================
# Tensor canonicalization tests
# ===================================================================

def _run_tensor_canonicalization_tests():
    antisym_expr = structure_constant(a3, a4, a5) + structure_constant(a4, a3, a5)
    canon_antisym, _, _ = canonize_spenso_tensors(
        antisym_expr,
        adjoint_indices=(a3, a4, a5),
    )
    _check(
        canon_antisym,
        Expression.num(0),
        "Tensor canon: structure constant antisymmetry",
    )

    compiled_scalar_qcd = compile_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)
    raw_contact = _model_demo_vertex(
        interaction=compiled_scalar_qcd[2],
        external_legs=LEGS_compiled_scalar_qcd_contact,
        species_map={b1: phiQCDdag0, b2: phiQCD0, b3: G0, b4: G0},
    )
    alt_dummy = S("c_mid_alt")
    renamed_contact = raw_contact.replace(S("c_mid_PhiQCD_SU3C"), alt_dummy)

    canon_contact = _canonized_gauge_vertex(
        raw_contact,
        lorentz_indices=(mu3, mu4),
        adjoint_indices=(a3, a4),
        color_fund_indices=(c1, c2, S("c_mid_PhiQCD_SU3C")),
    )
    canon_contact_renamed = _canonized_gauge_vertex(
        renamed_contact,
        lorentz_indices=(mu3, mu4),
        adjoint_indices=(a3, a4),
        color_fund_indices=(c1, c2, alt_dummy),
    )
    _check(
        canon_contact,
        canon_contact_renamed,
        "Tensor canon: scalar QCD contact dummy-label invariance",
    )

    print("\n  Tensor canonicalization tests passed.\n")


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
    print("  Compiled gauge-model tests")
    print("=" * 80)
    _run_compiled_gauge_tests()

    print("=" * 80)
    print("  Covariant / pure-gauge compiler tests")
    print("=" * 80)
    _run_covariant_compiler_tests()

    print("=" * 80)
    print("  Ordinary gauge-fixing / ghost compiler tests")
    print("=" * 80)
    _run_gauge_fixed_compiler_tests()

    print("=" * 80)
    print("  Tensor canonicalization tests")
    print("=" * 80)
    _run_tensor_canonicalization_tests()

    print("=" * 80)
    print("  Role regression tests")
    print("=" * 80)
    _run_role_regression_tests()

    print("All tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vertex examples and tests.")
    parser.add_argument(
        "--suite",
        choices=("scalar", "fermion", "gauge", "gaugefix", "model", "covariant", "cross", "role", "all"),
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
        elif args.suite == "gaugefix":
            _run_gauge_fixed_compiler_tests()
        elif args.suite == "model":
            _run_model_tests()
            _run_compiled_gauge_tests()
        elif args.suite == "covariant":
            _run_covariant_compiler_tests()
            _run_tensor_canonicalization_tests()
        elif args.suite == "cross":
            _run_cross_checks()
        elif args.suite == "role":
            _run_role_regression_tests()
