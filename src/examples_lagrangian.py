"""
Declarative Lagrangian API examples and regression tests.

The demo path in this file is written around the new front end:

  1. define fields / gauge groups / models
  2. declare the Lagrangian with ``lagrangian_decl=...``
  3. call ``Model.lagrangian().feynman_rule(...)``

Some regression checks still use low-level objects internally as parity
references, but the human-facing examples are meant to read as declarative
model definitions rather than backend ``InteractionTerm`` construction.

Conventions (automatic):
  - Momenta: q1, q2, q3, ...
  - Indices: i1, i2, i3, ... (sequential across legs)
  - Conjugated fields: Field.bar
"""

import argparse
import re

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
    bis,
    Delta,
    pcomp,
    compact_vertex_sum_form,
)
from spenso_structures import (
    SPINOR_KIND,
    gauge_generator,
    gamma_matrix,
    gamma5_matrix,
    simplify_gamma_chain,
    structure_constant,
)
from model import (
    COLOR_FUND_INDEX,
    InteractionTerm,
    DerivativeAction,
    Field,
    Lagrangian,
    Model,
    Gamma,
    PartialD,
)
from operators import (
    psi_bar_psi,
    quark_gluon_current,
    scalar_gauge_contact,
)
from gauge_compiler import (
    compile_covariant_terms,
    compile_minimal_gauge_interactions,
)
from tensor_canonicalization import canonize_spenso_tensors

from examples import (
    # symbols
    d,
    lam4, lam6, g_sym, lamC, yF, gV, gS, eQED, xiQED, xiQCD,
    gD, gD2, g1, g2,
    qPhi, qPsi, qMix, qPhiMix,
    gPhiA, gPhiAA, g_psi4, gJJ,
    alpha_s,
    mu, nu,
    # fields
    PhiField, ChiField, PhiCField, PhiQEDField,
    PhiQCDField, PhiMixField, PhiBiField,
    PsiField, PsiQEDField, PsiMixField,
    QuarkField, GluonField, GhostGluonField,
    GaugeField,
    # interaction terms
    TERM_phi4, TERM_phi2chi2, TERM_phiCdag_phiC,
    TERM_yukawa, TERM_vec_current, TERM_axial_current,
    TERM_psibar_psi_sq, TERM_current_current,
    TERM_quark_gluon,
    TERM_complex_scalar_current_phi, TERM_complex_scalar_current_phidag,
    TERM_complex_scalar_contact,
    # models
    MODEL_QCD_BASE, MODEL_QED_FERMION_BASE,
    MODEL_SCALAR_QED_BASE, MODEL_SCALAR_QCD_BASE,
    MODEL_SCALAR_QCD_BISLOT_BASE,
    MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS,
    MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM,
    MODEL_QCD_COVARIANT, MODEL_QED_FERMION_COVARIANT,
    MODEL_MIXED_FERMION_COVARIANT,
    MODEL_SCALAR_QED_COVARIANT, MODEL_SCALAR_QCD_COVARIANT,
    MODEL_MIXED_SCALAR_COVARIANT,
    MODEL_QED_GAUGE_COVARIANT, MODEL_QCD_GAUGE_COVARIANT,
    MODEL_QED_GAUGE_FIXING_COVARIANT, MODEL_QCD_GAUGE_FIXING_COVARIANT,
    MODEL_QCD_GHOST_COVARIANT,
    MODEL_QCD_ORDINARY_GAUGE_FIXED, MODEL_QED_ORDINARY_GAUGE_FIXED,
)


# ---------------------------------------------------------------------------
# Auto-index symbols
# ---------------------------------------------------------------------------

q1, q2, q3, q4, q5, q6 = S("q1", "q2", "q3", "q4", "q5", "q6")
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_DEFAULT_LAGRANGIAN_TERM_LIMIT = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(got, expected, label):
    """Assert symbolic equality."""
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


def _check_nonzero(expr, label):
    """Assert the expression is not identically zero."""
    assert (
        expr.expand().to_canonical_string()
        != Expression.num(0).to_canonical_string()
    ), f"{label} FAILED: got zero"
    print(f"  {label}: PASS (non-zero)")


def _lagrangian_vertex(compiled_terms, *fields, filter_fn=None):
    """Compute vertex from explicit compiled terms using the Lagrangian API.

    This is the reference path: bypasses Model.lagrangian() and builds
    a Lagrangian directly from compiled InteractionTerm objects, so the
    result uses the same auto-conventions (q1, q2, ...; i1, i2, ...) as
    the Model.lagrangian() path.
    """
    terms = compiled_terms
    if filter_fn is not None:
        terms = tuple(t for t in terms if filter_fn(t))
    L = Lagrangian(terms=terms)
    return L.feynman_rule(*fields)


def _lagrangian_from_terms(terms):
    """Build a Lagrangian from a sequence of InteractionTerm objects."""
    return Lagrangian(terms=tuple(terms))


def _print_demo_header(title):
    print(f"# === {title} ===\n")


def _print_section(title, content):
    if content is None:
        return
    print(f"{title}:")
    text = _ANSI_ESCAPE_RE.sub("", str(content))
    for line in text.splitlines():
        print(f"  {line}")
    print()


def _term_label(term):
    if hasattr(term, "label") and term.label:
        return term.label
    return str(term)


def _model_decl_label(model, attr_name, fallback):
    terms = getattr(model, attr_name, ())
    if not terms:
        terms = model.lagrangian_decl.source_terms
    if terms and getattr(terms[0], "label", ""):
        return terms[0].label
    return fallback


def _demo_lagrangian_terms(model, compiled_terms, *fields):
    source_terms = model.source_lagrangian_terms()
    if source_terms:
        return source_terms
    return _matching_lagrangian_terms(compiled_terms, *fields)


def _format_lagrangian_terms(terms, *, max_terms=_DEFAULT_LAGRANGIAN_TERM_LIMIT):
    cleaned_terms = tuple(terms or ())
    if not cleaned_terms:
        return None
    lines = []
    shown_terms = cleaned_terms[:max_terms] if max_terms is not None else cleaned_terms
    for index, term in enumerate(shown_terms):
        prefix = "" if index == 0 else "+ "
        text = _ANSI_ESCAPE_RE.sub("", _term_label(term))
        term_lines = text.splitlines() or [text]
        lines.append(f"{prefix}{term_lines[0]}")
        continuation_prefix = "  " if index == 0 else "  "
        for line in term_lines[1:]:
            lines.append(f"{continuation_prefix}{line}")
    omitted = len(cleaned_terms) - len(shown_terms)
    if omitted > 0:
        lines.append(f"... (+ {omitted} more matching terms omitted)")
    return "\n".join(lines)


def _print_vertex_block(
    title,
    *,
    lagrangian_terms=None,
    lagrangian_term_limit=_DEFAULT_LAGRANGIAN_TERM_LIMIT,
    description=None,
    vertex=None,
    canonical_vertex=None,
    compact_override=None,
    sum_notation=None,
    interpretation=None,
    error=None,
):
    _print_demo_header(title)
    if lagrangian_terms:
        _print_section(
            "Lagrangian",
            _format_lagrangian_terms(lagrangian_terms, max_terms=lagrangian_term_limit),
        )
    if description:
        _print_section("Context", description)
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


def _print_model_vertex_case(
    title,
    model,
    *fields,
    description=None,
    lagrangian_term_limit=_DEFAULT_LAGRANGIAN_TERM_LIMIT,
    vertex_transform=None,
):
    compiled_terms = compile_covariant_terms(model)
    vertex = model.lagrangian().feynman_rule(*fields)
    if vertex_transform is not None:
        vertex = vertex_transform(vertex)
    _print_vertex_block(
        title,
        lagrangian_terms=_demo_lagrangian_terms(model, compiled_terms, *fields),
        lagrangian_term_limit=lagrangian_term_limit,
        description=description,
        vertex=vertex,
    )


def _matching_lagrangian_terms(terms, *fields):
    matches = []
    for term in terms:
        try:
            Lagrangian(terms=(term,)).feynman_rule(*fields)
        except ValueError:
            continue
        matches.append(term)
    return tuple(matches)


def _symmetrized_generator_contact(adj_left, adj_right, color_left, color_right, color_middle):
    return (
        gauge_generator(adj_left, color_left, color_middle)
        * gauge_generator(adj_right, color_middle, color_right)
        + gauge_generator(adj_right, color_left, color_middle)
        * gauge_generator(adj_left, color_middle, color_right)
    )


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


# ---------------------------------------------------------------------------
# Momentum-conservation delta helpers
# ---------------------------------------------------------------------------

D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
HALF = Expression.num(1) / Expression.num(2)


# ---------------------------------------------------------------------------
# Declarative demo models
# ---------------------------------------------------------------------------

MODEL_DECL_SCALAR_PHI4 = Model(
    name="DeclScalarPhi4",
    fields=(PhiField,),
    lagrangian_decl=lam4 * PhiField * PhiField * PhiField * PhiField,
)

MODEL_DECL_SCALAR_PHI2CHI2 = Model(
    name="DeclScalarPhi2Chi2",
    fields=(PhiField, ChiField),
    lagrangian_decl=g_sym * PhiField * PhiField * ChiField * ChiField,
)

MODEL_DECL_COMPLEX_SCALAR_BILINEAR = Model(
    name="DeclComplexScalarBilinear",
    fields=(PhiCField,),
    lagrangian_decl=lamC * PhiCField.bar * PhiCField,
)

MODEL_DECL_SCALAR_DERIVATIVE = Model(
    name="DeclScalarDerivative",
    fields=(PhiField,),
    lagrangian_decl=gD2 * PartialD(PhiField, mu) * PartialD(PhiField, mu) * PhiField * PhiField,
)

MODEL_DECL_SCALAR_PHI6 = Model(
    name="DeclScalarPhi6",
    fields=(PhiField,),
    lagrangian_decl=lam6 * PhiField * PhiField * PhiField * PhiField * PhiField * PhiField,
)

MODEL_DECL_YUKAWA = Model(
    name="DeclYukawa",
    fields=(PsiField, PhiField),
    lagrangian_decl=yF * PsiField.bar * PsiField * PhiField,
)

MODEL_DECL_VECTOR_CURRENT = Model(
    name="DeclVectorCurrent",
    fields=(PsiField, GaugeField),
    lagrangian_decl=gV * PsiField.bar * Gamma(mu) * PsiField * GaugeField,
)

MODEL_DECL_FOUR_FERMION = Model(
    name="DeclFourFermion",
    fields=(PsiField,),
    lagrangian_decl=-(HALF * g_psi4) * PsiField.bar * PsiField * PsiField.bar * PsiField,
)

MODEL_DECL_MIXED_FERMION_DERIVATIVE = Model(
    name="DeclMixedFermionDerivative",
    fields=(PsiField, PhiField, ChiField),
    lagrangian_decl=yF * PsiField.bar * Gamma(mu) * PartialD(PsiField, mu) * PhiField * ChiField,
)

MODEL_DECL_MIXED_SCALAR_DERIVATIVES = Model(
    name="DeclMixedScalarDerivatives",
    fields=(PsiField, PhiField, ChiField),
    lagrangian_decl=yF * PsiField.bar * PsiField * PartialD(PhiField, mu) * PartialD(ChiField, mu),
)

MODEL_DECL_MIXED_HIGHER_DERIVATIVE = Model(
    name="DeclMixedHigherDerivative",
    fields=(PsiField, PhiField, ChiField),
    lagrangian_decl=g1 * PsiField.bar * PsiField * PartialD(PartialD(PhiField, mu), mu) * ChiField,
)


# ===================================================================
# 1. Scalar vertices
# ===================================================================

def _run_scalar_tests():
    print("\n  --- Scalar vertices (Lagrangian API) ---")

    # phi^4
    L = Lagrangian(terms=(TERM_phi4,))
    got = L.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    _check(got, 24 * I * lam4 * D4, "L-API: phi^4")

    # phi^2 chi^2
    L = Lagrangian(terms=(TERM_phi2chi2,))
    got = L.feynman_rule(PhiField, PhiField, ChiField, ChiField)
    _check(got, 4 * I * g_sym * D4, "L-API: phi^2 chi^2")

    # phi^dag phi
    L = Lagrangian(terms=(TERM_phiCdag_phiC,))
    got = L.feynman_rule(PhiCField.bar, PhiCField)
    _check(got, I * lamC * D2, "L-API: phi^dag phi")

    # phi^4 with derivatives: just test non-zero
    TERM_deriv = InteractionTerm(
        coupling=gD,
        fields=tuple(PhiField.occurrence() for _ in range(4)),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu), DerivativeAction(target=1, lorentz_index=nu)),
        label="gD * (d_mu phi)(d_nu phi) phi phi",
    )
    L_d = Lagrangian(terms=(TERM_deriv,))
    got_d = L_d.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    expected_d = compact_vertex_sum_form(
        coupling=gD,
        ps=[q1, q2, q3, q4],
        derivative_indices=[mu, nu],
        derivative_targets=[0, 1],
        d=d,
        field_species=[PhiField.symbol] * 4,
        leg_species=[PhiField.symbol] * 4,
    )
    _check(got_d, expected_d, "L-API: phi^4 derivative (mu,nu)")

    TERM_deriv2 = InteractionTerm(
        coupling=gD2,
        fields=tuple(PhiField.occurrence() for _ in range(4)),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu), DerivativeAction(target=1, lorentz_index=mu)),
        label="gD2 * (d_mu phi)(d^mu phi) phi phi",
    )
    L_d2 = Lagrangian(terms=(TERM_deriv2,))
    got_d2 = L_d2.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    expected_d2 = compact_vertex_sum_form(
        coupling=gD2,
        ps=[q1, q2, q3, q4],
        derivative_indices=[mu, mu],
        derivative_targets=[0, 1],
        d=d,
        field_species=[PhiField.symbol] * 4,
        leg_species=[PhiField.symbol] * 4,
    )
    _check(got_d2, expected_d2, "L-API: phi^4 derivative (mu,mu)")

    # phi^6: 6-point scalar vertex
    D6 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4 + q5 + q6)
    TERM_phi6 = InteractionTerm(
        coupling=lam6,
        fields=tuple(PhiField.occurrence() for _ in range(6)),
        label="lam6 * phi^6",
    )
    L6 = Lagrangian(terms=(TERM_phi6,))
    got6 = L6.feynman_rule(PhiField, PhiField, PhiField, PhiField, PhiField, PhiField)
    _check(got6, 720 * I * lam6 * D6, "L-API: phi^6")

    # Composition: InteractionTerm + InteractionTerm → Lagrangian
    L_composed = TERM_phi4 + TERM_phi2chi2
    assert isinstance(L_composed, Lagrangian), "InteractionTerm + InteractionTerm should produce Lagrangian"
    got_phi4 = L_composed.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    _check(got_phi4, 24 * I * lam4 * D4, "L-API: composed L phi^4")
    got_mixed = L_composed.feynman_rule(PhiField, PhiField, ChiField, ChiField)
    _check(got_mixed, 4 * I * g_sym * D4, "L-API: composed L phi^2 chi^2")

    print("\n  Scalar tests passed.\n")


# ===================================================================
# 2. Fermion vertices
# ===================================================================

def _run_fermion_tests():
    print("\n  --- Fermion vertices (Lagrangian API) ---")

    # Auto indices for PsiField.bar → i1=spinor, PsiField → i2=spinor

    # Yukawa: psibar psi phi → I*yF*bis(i1,i2)*D3
    L = Lagrangian(terms=(TERM_yukawa,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField)
    G12 = bis.g(S("i1"), S("i2")).to_expression()
    _check(got, I * yF * G12 * D3, "L-API: Yukawa")

    # Vector current: psibar psi A → I*gV*gamma(i1,i2,i3)*D3
    L = Lagrangian(terms=(TERM_vec_current,))
    got = L.feynman_rule(PsiField.bar, PsiField, GaugeField)
    expected_vec = I * gV * gamma_matrix(S("i1"), S("i2"), S("i3")) * D3
    _check(got, expected_vec, "L-API: vector current")

    # Axial current: psibar gamma^mu gamma5 psi A
    L = Lagrangian(terms=(TERM_axial_current,))
    got = L.feynman_rule(PsiField.bar, PsiField, GaugeField)
    expected_axial = I * gV * gamma_matrix(S("i1"), alpha_s, S("i3")) * gamma5_matrix(alpha_s, S("i2")) * D3
    _check(got, expected_axial, "L-API: axial current")

    # (psibar psi)^2: 4 fermion legs
    # PsiField.bar → i1=spinor, PsiField → i2=spinor,
    # PsiField.bar → i3=spinor, PsiField → i4=spinor
    L = Lagrangian(terms=(TERM_psibar_psi_sq,))
    got = L.feynman_rule(PsiField.bar, PsiField, PsiField.bar, PsiField)
    expected_sp = (
        -I * g_psi4 * D4
        * (psi_bar_psi(S("i1"), S("i2")) * psi_bar_psi(S("i3"), S("i4"))
           - psi_bar_psi(S("i1"), S("i4")) * psi_bar_psi(S("i3"), S("i2")))
    )
    _check(got, expected_sp, "L-API: (psibar psi)^2")

    # Current-current: test after gamma chain simplification
    L = Lagrangian(terms=(TERM_current_current,))
    got = simplify_gamma_chain(
        L.feynman_rule(PsiField.bar, PsiField, PsiField.bar, PsiField)
    )
    expected_jj = (
        2 * I * gJJ * D4
        * (gamma_matrix(S("i1"), S("i2"), mu) * gamma_matrix(S("i3"), S("i4"), mu)
           - gamma_matrix(S("i1"), S("i4"), mu) * gamma_matrix(S("i3"), S("i2"), mu))
    )
    _check(got, expected_jj, "L-API: current-current")

    print("\n  Fermion tests passed.\n")


# ===================================================================
# 2b. Mixed fermion+scalar derivative vertices
# ===================================================================

def _run_mixed_derivative_tests():
    print("\n  --- Mixed fermion+scalar derivatives (Lagrangian API) ---")

    # Build interaction terms with derivatives on different fields
    _MIX_FIELDS = (
        PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: alpha_s}),
        PsiField.occurrence(labels={SPINOR_KIND: alpha_s}),
        PhiField.occurrence(),
        ChiField.occurrence(),
    )

    TERM_dpsibar = InteractionTerm(
        coupling=yF, fields=_MIX_FIELDS,
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        label="yF * (d_mu psibar) psi phi chi",
    )
    TERM_dpsi = InteractionTerm(
        coupling=yF, fields=_MIX_FIELDS,
        derivatives=(DerivativeAction(target=1, lorentz_index=nu),),
        label="yF * psibar (d_nu psi) phi chi",
    )
    TERM_dphi_dchi = InteractionTerm(
        coupling=yF, fields=_MIX_FIELDS,
        derivatives=(DerivativeAction(target=2, lorentz_index=mu), DerivativeAction(target=3, lorentz_index=nu)),
        label="yF * psibar psi (d_mu phi)(d_nu chi)",
    )
    TERM_d2phi_chi = InteractionTerm(
        coupling=g1, fields=_MIX_FIELDS,
        derivatives=(DerivativeAction(target=2, lorentz_index=mu), DerivativeAction(target=2, lorentz_index=mu)),
        label="g1 * psibar psi (d^2 phi) chi",
    )
    TERM_d2phi2 = InteractionTerm(
        coupling=g2,
        fields=(
            PsiField.occurrence(conjugated=True, labels={SPINOR_KIND: alpha_s}),
            PsiField.occurrence(labels={SPINOR_KIND: alpha_s}),
            PhiField.occurrence(),
            PhiField.occurrence(),
        ),
        derivatives=(
            DerivativeAction(target=2, lorentz_index=mu),
            DerivativeAction(target=2, lorentz_index=nu),
            DerivativeAction(target=3, lorentz_index=mu),
            DerivativeAction(target=3, lorentz_index=nu),
        ),
        label="g2 * psibar psi (d_mu d_nu phi)^2",
    )
    G12 = bis.g(S("i1"), S("i2")).to_expression()
    D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)

    # d_mu psibar
    L = Lagrangian(terms=(TERM_dpsibar,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check(got, yF * pcomp(q1, mu) * G12 * D4, "L-API: d_mu psibar * psi * phi * chi")

    # d_nu psi
    L = Lagrangian(terms=(TERM_dpsi,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check(got, yF * pcomp(q2, nu) * G12 * D4, "L-API: psibar * d_nu psi * phi * chi")

    # (d_mu phi)(d_nu chi)
    L = Lagrangian(terms=(TERM_dphi_dchi,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check(
        got,
        -I * yF * pcomp(q3, mu) * pcomp(q4, nu) * G12 * D4,
        "L-API: psibar * psi * (d_mu phi)(d_nu chi)",
    )

    # g1 * psibar psi (d^2 phi) chi
    L = Lagrangian(terms=(TERM_d2phi_chi,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check(
        got,
        -I * g1 * G12 * D4 * pcomp(q3, mu) * pcomp(q3, mu),
        "L-API: g1 * psibar psi (d^2 phi) chi",
    )

    # g2 * psibar psi (d_mu d_nu phi)^2
    L = Lagrangian(terms=(TERM_d2phi2,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, PhiField)
    _check(
        got,
        2 * I * g2 * G12 * D4
        * pcomp(q3, mu) * pcomp(q3, nu)
        * pcomp(q4, mu) * pcomp(q4, nu),
        "L-API: g2 * psibar psi (d_mu d_nu phi)^2",
    )

    print("\n  Mixed derivative tests passed.\n")


# ===================================================================
# 3. Gauge-ready vertices
# ===================================================================

def _run_gauge_ready_tests():
    print("\n  --- Gauge-ready vertices (Lagrangian API) ---")

    # Quark-gluon:
    # QuarkField.bar → i1=spinor, i2=color_fund
    # QuarkField     → i3=spinor, i4=color_fund
    # GluonField     → i5=lorentz, i6=color_adj
    L = Lagrangian(terms=(TERM_quark_gluon,))
    got = L.feynman_rule(QuarkField.bar, QuarkField, GluonField)
    expected_qg = I * gS * quark_gluon_current(S("i1"), S("i3"), S("i5"), S("i6"), S("i2"), S("i4")) * D3
    _check(got, expected_qg, "L-API: quark-gluon")

    # Complex scalar current: phiC^dag, phiC, A → non-zero
    L_sc = Lagrangian(terms=(TERM_complex_scalar_current_phi, TERM_complex_scalar_current_phidag))
    got_sc = L_sc.feynman_rule(PhiCField.bar, PhiCField, GaugeField)
    expected_sc = gPhiA * (pcomp(q2, mu) - pcomp(q1, mu)) * D3
    _check(got_sc, expected_sc, "L-API: complex scalar current")

    # Complex scalar contact: phiC^dag, phiC, A, A
    L_ct = Lagrangian(terms=(TERM_complex_scalar_contact,))
    got_ct = L_ct.feynman_rule(PhiCField.bar, PhiCField, GaugeField, GaugeField)
    expected_ct = 2 * I * gPhiAA * scalar_gauge_contact(S("i1"), S("i2")) * D4
    _check(got_ct, expected_ct, "L-API: complex scalar contact")

    print("\n  Gauge-ready tests passed.\n")


# ===================================================================
# 4. Compiled minimal gauge models
# ===================================================================

def _run_compiled_minimal_tests():
    print("\n  --- Compiled minimal gauge models (Lagrangian API) ---")

    # QCD quark-gluon
    compiled_qcd = compile_minimal_gauge_interactions(MODEL_QCD_BASE)
    L_qcd = Lagrangian(terms=compiled_qcd)
    got = L_qcd.feynman_rule(QuarkField.bar, QuarkField, GluonField)
    expected_qcd = (
        -I * gS
        * quark_gluon_current(S("i1"), S("i3"), S("i5"), S("i6"), S("i2"), S("i4"))
        * D3
    )
    _check(got, expected_qcd, "L-API minimal: QCD quark-gluon")

    # QED fermion
    compiled_qed = compile_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
    L_qed = Lagrangian(terms=compiled_qed)
    got = L_qed.feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField)
    expected_qed = -I * eQED * qPsi * gamma_matrix(S("i1"), S("i2"), S("i3")) * D3
    _check(got, expected_qed, "L-API minimal: QED fermion")

    # Scalar QED current (3pt) and contact (4pt)
    compiled_sc_qed = compile_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
    L_sc_qed = Lagrangian(terms=compiled_sc_qed)
    got_3pt = L_sc_qed.feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField)
    expected_sc_qed_3pt = I * eQED * qPhi * (pcomp(q2, mu) - pcomp(q1, mu)) * D3
    _check(got_3pt, expected_sc_qed_3pt, "L-API minimal: scalar QED current")
    got_4pt = L_sc_qed.feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField)
    expected_sc_qed_4pt = 2 * I * (eQED ** 2) * (qPhi ** 2) * scalar_gauge_contact(S("i1"), S("i2")) * D4
    _check(got_4pt, expected_sc_qed_4pt, "L-API minimal: scalar QED contact")

    # Scalar QCD current (3pt) and contact (4pt)
    compiled_sc_qcd = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)
    L_sc_qcd = Lagrangian(terms=compiled_sc_qcd)
    got_3pt = L_sc_qcd.feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField)
    expected_sc_qcd_3pt = (
        I * gS * gauge_generator(S("i4"), S("i1"), S("i2"))
        * (pcomp(q2, mu) - pcomp(q1, mu))
        * D3
    )
    _check(got_3pt, expected_sc_qcd_3pt, "L-API minimal: scalar QCD current")
    got_4pt = L_sc_qcd.feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField, GluonField)
    expected_sc_qcd_4pt = (
        I * (gS ** 2)
        * scalar_gauge_contact(S("i3"), S("i5"))
        * _symmetrized_generator_contact(
            S("i4"), S("i6"), S("i1"), S("i2"), S("c_mid_PhiQCD_SU3C"),
        )
        * D4
    )
    _check(got_4pt, expected_sc_qcd_4pt, "L-API minimal: scalar QCD contact")

    try:
        compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS)
    except ValueError as exc:
        assert "GaugeRepresentation(slot" in str(exc)
        print("  L-API minimal: repeated-slot ambiguity rejected: PASS")
    else:
        raise AssertionError("Repeated same-kind representation slot should require GaugeRepresentation(slot=...)")

    compiled_bislot = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_BASE)
    L_bislot = Lagrangian(terms=compiled_bislot)
    spectator_identity = COLOR_FUND_INDEX.representation.g(S("i2"), S("i4")).to_expression()
    got_bislot = L_bislot.feynman_rule(PhiBiField.bar, PhiBiField, GluonField)
    expected_bislot = (
        I * gS
        * gauge_generator(S("i6"), S("i1"), S("i3"))
        * spectator_identity
        * (pcomp(q2, mu) - pcomp(q1, mu))
        * D3
    )
    _check(got_bislot, expected_bislot, "L-API minimal: repeated-slot scalar QCD current")

    print("\n  Compiled minimal gauge model tests passed.\n")


# ===================================================================
# 5. Covariant compiler via Model.lagrangian()
# ===================================================================

def _run_covariant_tests():
    print("\n  --- Covariant compiler via Model.lagrangian() ---")

    # QCD covariant: quark-gluon
    got = MODEL_QCD_COVARIANT.lagrangian().feynman_rule(QuarkField.bar, QuarkField, GluonField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QCD_COVARIANT),
                             QuarkField.bar, QuarkField, GluonField)
    _check(got, ref, "L-API covariant: QCD quark-gluon")

    # QED fermion covariant
    got = MODEL_QED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_FERMION_COVARIANT),
                             PsiQEDField.bar, PsiQEDField, GaugeField)
    _check(got, ref, "L-API covariant: QED fermion")

    # Mixed fermion covariant: gluon piece + photon piece
    compiled_mixed = compile_covariant_terms(MODEL_MIXED_FERMION_COVARIANT)

    got_gluon = MODEL_MIXED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiMixField.bar, PsiMixField, GluonField)
    ref_gluon = _lagrangian_vertex(compiled_mixed, PsiMixField.bar, PsiMixField, GluonField)
    _check(got_gluon, ref_gluon, "L-API covariant: mixed fermion gluon piece")

    got_photon = MODEL_MIXED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiMixField.bar, PsiMixField, GaugeField)
    ref_photon = _lagrangian_vertex(compiled_mixed, PsiMixField.bar, PsiMixField, GaugeField)
    _check(got_photon, ref_photon, "L-API covariant: mixed fermion photon piece")

    # Scalar QED covariant: current (3pt) + contact (4pt)
    compiled_sc_qed = compile_covariant_terms(MODEL_SCALAR_QED_COVARIANT)

    got_3pt = MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField)
    ref_3pt = _lagrangian_vertex(compiled_sc_qed, PhiQEDField.bar, PhiQEDField, GaugeField)
    _check(got_3pt, ref_3pt, "L-API covariant: scalar QED current")

    got_4pt = MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField)
    ref_4pt = _lagrangian_vertex(compiled_sc_qed, PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField)
    _check(got_4pt, ref_4pt, "L-API covariant: scalar QED contact")

    # Scalar QCD covariant: current (3pt) + contact (4pt)
    compiled_sc_qcd = compile_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)

    got_3pt = MODEL_SCALAR_QCD_COVARIANT.lagrangian().feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField)
    ref_3pt = _lagrangian_vertex(compiled_sc_qcd, PhiQCDField.bar, PhiQCDField, GluonField)
    _check(got_3pt, ref_3pt, "L-API covariant: scalar QCD current")

    got_4pt = MODEL_SCALAR_QCD_COVARIANT.lagrangian().feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField, GluonField)
    ref_4pt = _lagrangian_vertex(compiled_sc_qcd, PhiQCDField.bar, PhiQCDField, GluonField, GluonField)
    _check(got_4pt, ref_4pt, "L-API covariant: scalar QCD contact")

    # Mixed scalar covariant: QCD current + QED current + mixed contact
    compiled_mix_sc = compile_covariant_terms(MODEL_MIXED_SCALAR_COVARIANT)

    got_qcd = MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(PhiMixField.bar, PhiMixField, GluonField)
    ref_qcd = _lagrangian_vertex(compiled_mix_sc, PhiMixField.bar, PhiMixField, GluonField)
    _check(got_qcd, ref_qcd, "L-API covariant: mixed scalar QCD current")

    got_qed = MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(PhiMixField.bar, PhiMixField, GaugeField)
    ref_qed = _lagrangian_vertex(compiled_mix_sc, PhiMixField.bar, PhiMixField, GaugeField)
    _check(got_qed, ref_qed, "L-API covariant: mixed scalar QED current")

    got_contact = MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(PhiMixField.bar, PhiMixField, GluonField, GaugeField)
    ref_contact = _lagrangian_vertex(compiled_mix_sc, PhiMixField.bar, PhiMixField, GluonField, GaugeField)
    _check(got_contact, ref_contact, "L-API covariant: mixed scalar contact")

    # Bislot scalar (two identical COLOR_FUND slots, slot_policy='sum')
    compiled_bislot = compile_covariant_terms(MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM)
    got_bislot_3pt = MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM.lagrangian().feynman_rule(
        PhiBiField.bar, PhiBiField, GluonField,
    )
    ref_bislot_3pt = _lagrangian_vertex(compiled_bislot, PhiBiField.bar, PhiBiField, GluonField)
    _check(got_bislot_3pt, ref_bislot_3pt, "L-API covariant: bislot scalar current")

    got_bislot_4pt = MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM.lagrangian().feynman_rule(
        PhiBiField.bar, PhiBiField, GluonField, GluonField,
    )
    ref_bislot_4pt = _lagrangian_vertex(compiled_bislot, PhiBiField.bar, PhiBiField, GluonField, GluonField)
    _check(got_bislot_4pt, ref_bislot_4pt, "L-API covariant: bislot scalar contact")

    print("\n  Covariant compiler tests passed.\n")


# ===================================================================
# 6. Pure-gauge sector via Model.lagrangian()
# ===================================================================

def _run_pure_gauge_tests():
    print("\n  --- Pure-gauge sector (Lagrangian API) ---")

    # QED photon bilinear (2pt)
    got = MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT),
                             GaugeField, GaugeField)
    _check(got, ref, "L-API gauge: QED photon bilinear")

    # QCD gluon bilinear (2pt), 3-gluon cubic, 4-gluon quartic
    compiled_ym = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)

    got_2pt = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField)
    ref_2pt = _lagrangian_vertex(compiled_ym, GluonField, GluonField)
    _check(got_2pt, ref_2pt, "L-API gauge: QCD gluon bilinear")

    got_3pt = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField, GluonField)
    ref_3pt = _lagrangian_vertex(compiled_ym, GluonField, GluonField, GluonField)
    _check(got_3pt, ref_3pt, "L-API gauge: QCD 3-gluon cubic")

    got_4pt = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField, GluonField, GluonField)
    ref_4pt = _lagrangian_vertex(compiled_ym, GluonField, GluonField, GluonField, GluonField)
    _check(got_4pt, ref_4pt, "L-API gauge: QCD 4-gluon quartic")

    print("\n  Pure-gauge sector tests passed.\n")


# ===================================================================
# 7. Gauge fixing
# ===================================================================

def _run_gauge_fixing_tests():
    print("\n  --- Gauge fixing (Lagrangian API) ---")

    # QED gauge-fixing bilinear
    got = MODEL_QED_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT),
                             GaugeField, GaugeField)
    _check(got, ref, "L-API GF: QED photon gauge-fixing bilinear")

    # QCD gauge-fixing bilinear
    got = MODEL_QCD_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT),
                             GluonField, GluonField)
    _check(got, ref, "L-API GF: QCD gluon gauge-fixing bilinear")

    print("\n  Gauge fixing tests passed.\n")


# ===================================================================
# 8. Ghosts
# ===================================================================

def _run_ghost_tests():
    print("\n  --- Ghost sector (Lagrangian API) ---")

    compiled_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)

    # Ghost bilinear (2pt)
    got_2pt = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(GhostGluonField.bar, GhostGluonField)
    ref_2pt = _lagrangian_vertex(compiled_ghost, GhostGluonField.bar, GhostGluonField)
    _check(got_2pt, ref_2pt, "L-API ghost: ghost bilinear")

    # Ghost-gluon (3pt)
    got_3pt = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField)
    ref_3pt = _lagrangian_vertex(compiled_ghost, GhostGluonField.bar, GluonField, GhostGluonField)
    _check(got_3pt, ref_3pt, "L-API ghost: ghost-gluon 3pt")

    print("\n  Ghost sector tests passed.\n")


# ===================================================================
# 9. Full gauge-fixed QCD and QED
# ===================================================================

def _run_full_gauge_fixed_tests():
    print("\n  --- Full gauge-fixed models (Lagrangian API) ---")

    compiled_qcd = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)
    L_qcd = MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian()

    # Gluon bilinear (kinetic + GF)
    got_gluon_2pt = L_qcd.feynman_rule(GluonField, GluonField)
    ref_gluon_2pt = _lagrangian_vertex(compiled_qcd, GluonField, GluonField)
    _check(got_gluon_2pt, ref_gluon_2pt, "L-API full QCD: gluon bilinear")

    # 3-gluon
    got_3g = L_qcd.feynman_rule(GluonField, GluonField, GluonField)
    ref_3g = _lagrangian_vertex(compiled_qcd, GluonField, GluonField, GluonField)
    _check(got_3g, ref_3g, "L-API full QCD: 3-gluon")

    # 4-gluon
    got_4g = L_qcd.feynman_rule(GluonField, GluonField, GluonField, GluonField)
    ref_4g = _lagrangian_vertex(compiled_qcd, GluonField, GluonField, GluonField, GluonField)
    _check(got_4g, ref_4g, "L-API full QCD: 4-gluon")

    # Ghost bilinear
    got_ghost_2pt = L_qcd.feynman_rule(GhostGluonField.bar, GhostGluonField)
    ref_ghost_2pt = _lagrangian_vertex(compiled_qcd, GhostGluonField.bar, GhostGluonField)
    _check(got_ghost_2pt, ref_ghost_2pt, "L-API full QCD: ghost bilinear")

    # Ghost-gluon
    got_ghost_gluon = L_qcd.feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField)
    ref_ghost_gluon = _lagrangian_vertex(compiled_qcd, GhostGluonField.bar, GluonField, GhostGluonField)
    _check(got_ghost_gluon, ref_ghost_gluon, "L-API full QCD: ghost-gluon")

    # Full QED
    compiled_qed = compile_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)

    got_photon = MODEL_QED_ORDINARY_GAUGE_FIXED.lagrangian().feynman_rule(GaugeField, GaugeField)
    ref_photon = _lagrangian_vertex(compiled_qed, GaugeField, GaugeField)
    _check(got_photon, ref_photon, "L-API full QED: photon bilinear")

    print("\n  Full gauge-fixed model tests passed.\n")


# ===================================================================
# 10. Cross-checks: Model.lagrangian() vs explicit compiled Lagrangian
# ===================================================================

def _run_cross_checks():
    """Cross-check: Model.lagrangian() vs Lagrangian(terms=compile(...)).

    Both sides use the Lagrangian API auto-conventions, so the comparison
    is a pure consistency check of the Model.lagrangian() compilation path.
    """
    print("\n  --- Cross-checks: Model.lagrangian() vs explicit Lagrangian ---")

    # Covariant QCD quark-gluon
    got = MODEL_QCD_COVARIANT.lagrangian().feynman_rule(QuarkField.bar, QuarkField, GluonField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QCD_COVARIANT),
                             QuarkField.bar, QuarkField, GluonField)
    _check(got, ref, "Cross: QCD quark-gluon")

    # Covariant QED fermion
    got = MODEL_QED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_FERMION_COVARIANT),
                             PsiQEDField.bar, PsiQEDField, GaugeField)
    _check(got, ref, "Cross: QED fermion")

    # Pure gauge QED bilinear
    got = MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT),
                             GaugeField, GaugeField)
    _check(got, ref, "Cross: QED photon bilinear")

    # Pure gauge QCD — 2pt, 3pt, 4pt
    compiled_ym = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)
    L_ym = MODEL_QCD_GAUGE_COVARIANT.lagrangian()
    for n, label in [(2, "gluon bilinear"), (3, "3-gluon"), (4, "4-gluon")]:
        fields = [GluonField] * n
        got = L_ym.feynman_rule(*fields)
        ref = _lagrangian_vertex(compiled_ym, *fields)
        _check(got, ref, f"Cross: QCD {label}")

    # Ghost sector
    compiled_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    L_ghost = MODEL_QCD_GHOST_COVARIANT.lagrangian()
    got = L_ghost.feynman_rule(GhostGluonField.bar, GhostGluonField)
    ref = _lagrangian_vertex(compiled_ghost, GhostGluonField.bar, GhostGluonField)
    _check(got, ref, "Cross: ghost bilinear")

    got = L_ghost.feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField)
    ref = _lagrangian_vertex(compiled_ghost, GhostGluonField.bar, GluonField, GhostGluonField)
    _check(got, ref, "Cross: ghost-gluon")

    # Gauge-fixing
    got = MODEL_QED_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT),
                             GaugeField, GaugeField)
    _check(got, ref, "Cross: QED gauge-fixing")

    got = MODEL_QCD_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField)
    ref = _lagrangian_vertex(compile_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT),
                             GluonField, GluonField)
    _check(got, ref, "Cross: QCD gauge-fixing")

    # Full QCD (all vertex types in one model)
    compiled_full_qcd = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)
    L_full = MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian()
    for fields, label in [
        ([GluonField, GluonField], "full QCD gluon bilinear"),
        ([GluonField] * 3, "full QCD 3-gluon"),
        ([GluonField] * 4, "full QCD 4-gluon"),
        ([GhostGluonField.bar, GhostGluonField], "full QCD ghost bilinear"),
        ([GhostGluonField.bar, GluonField, GhostGluonField], "full QCD ghost-gluon"),
    ]:
        got = L_full.feynman_rule(*fields)
        ref = _lagrangian_vertex(compiled_full_qcd, *fields)
        _check(got, ref, f"Cross: {label}")

    print("\n  Cross-checks passed.\n")


# ===================================================================
# 11. Tensor canonicalization
# ===================================================================

def _run_tensor_canonicalization_tests():
    print("\n  --- Tensor canonicalization (Lagrangian API) ---")

    antisym_expr = structure_constant(S("a3"), S("a4"), S("a5")) + structure_constant(S("a4"), S("a3"), S("a5"))
    canon_antisym, _, _ = canonize_spenso_tensors(
        antisym_expr,
        adjoint_indices=(S("a3"), S("a4"), S("a5")),
    )
    _check(canon_antisym, Expression.num(0), "Tensor canon: structure constant antisymmetry")

    raw_contact = MODEL_SCALAR_QCD_COVARIANT.lagrangian().feynman_rule(
        PhiQCDField.bar, PhiQCDField, GluonField, GluonField, simplify=False,
    )
    alt_dummy = S("c_mid_alt")
    renamed_contact = raw_contact.replace(S("c_mid_PhiQCD_SU3C"), alt_dummy)

    canon_contact = _canonized_gauge_vertex(
        raw_contact,
        lorentz_indices=(S("i3"), S("i5")),
        adjoint_indices=(S("i4"), S("i6")),
        color_fund_indices=(S("i1"), S("i2"), S("c_mid_PhiQCD_SU3C")),
    )
    canon_contact_renamed = _canonized_gauge_vertex(
        renamed_contact,
        lorentz_indices=(S("i3"), S("i5")),
        adjoint_indices=(S("i4"), S("i6")),
        color_fund_indices=(S("i1"), S("i2"), alt_dummy),
    )
    _check(canon_contact, canon_contact_renamed, "Tensor canon: scalar QCD contact dummy-label invariance")

    print("\n  Tensor canonicalization tests passed.\n")


# ===================================================================
# 12. Role / matcher regressions
# ===================================================================

def _run_role_regression_tests():
    print("\n  --- Role / matcher regressions (Lagrangian API) ---")

    L_complex = Lagrangian(terms=(TERM_phiCdag_phiC,))
    expected_complex = I * lamC * D2
    _check(L_complex.feynman_rule(PhiCField.bar, PhiCField), expected_complex, "Regression: complex boson exact")
    _check(L_complex.feynman_rule(PhiCField, PhiCField.bar), expected_complex, "Regression: reversed legs still works")
    _check(L_complex.feynman_rule((PhiCField, True), PhiCField), expected_complex, "Regression: tuple field syntax")

    shared_symbol = S("X_shared")
    scalar = Field("ScalarShared", spin=0, self_conjugate=True, symbol=shared_symbol)
    vector = Field("VectorShared", spin=1, self_conjugate=True, symbol=shared_symbol)
    L_scalar = Lagrangian(terms=(InteractionTerm(coupling=lamC, fields=(scalar.occurrence(),)),))
    try:
        L_scalar.feynman_rule(vector)
    except ValueError as exc:
        assert "No matching interaction terms" in str(exc)
        print("  Regression: vector/scalar non-mixing: PASS")
    else:
        raise AssertionError("Vector field should not match scalar-only interaction even if symbols coincide")

    phi_alias = Field("PhiAlias", spin=0, self_conjugate=True, symbol=S("Y_shared"))
    chi_alias = Field("ChiAlias", spin=0, self_conjugate=True, symbol=S("Y_shared"))
    L_alias = Lagrangian(terms=(InteractionTerm(coupling=lamC, fields=(phi_alias.occurrence(),)),))
    try:
        L_alias.feynman_rule(chi_alias)
    except ValueError as exc:
        assert "No matching interaction terms" in str(exc)
        print("  Regression: same-symbol distinct fields rejected: PASS")
    else:
        raise AssertionError("Distinct declared fields sharing a symbol should not silently match")

    print("\n  Role / matcher regressions passed.\n")


# ===================================================================
# Demo output
# ===================================================================

def _run_scalar_demo():
    print("# " + "=" * 79)
    print("Demo: scalar (Lagrangian API)\n")

    _print_model_vertex_case(
        "scalar: phi^4",
        MODEL_DECL_SCALAR_PHI4,
        PhiField, PhiField, PhiField, PhiField,
    )
    _print_model_vertex_case(
        "scalar: phi^2 chi^2",
        MODEL_DECL_SCALAR_PHI2CHI2,
        PhiField, PhiField, ChiField, ChiField,
    )
    _print_model_vertex_case(
        "scalar: complex scalar bilinear",
        MODEL_DECL_COMPLEX_SCALAR_BILINEAR,
        PhiCField.bar, PhiCField,
    )
    _print_model_vertex_case(
        "scalar: derivative-contracted phi^4",
        MODEL_DECL_SCALAR_DERIVATIVE,
        PhiField, PhiField, PhiField, PhiField,
        description="Lorentz-contracted derivative interaction.",
    )
    _print_model_vertex_case(
        "scalar: phi^6",
        MODEL_DECL_SCALAR_PHI6,
        PhiField, PhiField, PhiField, PhiField, PhiField, PhiField,
    )


def _run_fermion_demo():
    print("# " + "=" * 79)
    print("Demo: fermion (Lagrangian API)\n")

    _print_model_vertex_case(
        "fermion: Yukawa",
        MODEL_DECL_YUKAWA,
        PsiField.bar, PsiField, PhiField,
    )
    _print_model_vertex_case(
        "fermion: vector current",
        MODEL_DECL_VECTOR_CURRENT,
        PsiField.bar, PsiField, GaugeField,
    )
    _print_model_vertex_case(
        "fermion: scalar four-fermion operator",
        MODEL_DECL_FOUR_FERMION,
        PsiField.bar, PsiField, PsiField.bar, PsiField,
        description="Local four-fermion operator written directly as a field product.",
    )


def _run_mixed_demo():
    print("# " + "=" * 79)
    print("Demo: fermion+scalar (Lagrangian API)\n")

    _print_model_vertex_case(
        "fermion+scalar: psibar gamma^mu partial_mu psi phi chi",
        MODEL_DECL_MIXED_FERMION_DERIVATIVE,
        PsiField.bar, PsiField, PhiField, ChiField,
        description="One local fermion derivative operator written directly with PartialD(...).",
    )
    _print_model_vertex_case(
        "fermion+scalar: psibar psi (partial_mu phi)(partial^mu chi)",
        MODEL_DECL_MIXED_SCALAR_DERIVATIVES,
        PsiField.bar, PsiField, PhiField, ChiField,
        description="Ordinary scalar derivatives inside the same local source term.",
    )
    _print_model_vertex_case(
        "fermion+scalar: psibar psi (partial^2 phi) chi",
        MODEL_DECL_MIXED_HIGHER_DERIVATIVE,
        PsiField.bar, PsiField, PhiField, ChiField,
        description="Nested PartialD(...) gives higher local derivatives without InteractionTerm(...).",
    )


def _run_gauge_demo():
    print("# " + "=" * 79)
    print("Demo: gauge interactions from declarative models\n")

    _print_model_vertex_case(
        "gauge: abelian fermion current from CovD",
        MODEL_QED_FERMION_COVARIANT,
        PsiQEDField.bar, PsiQEDField, GaugeField,
        description="Canonical Dirac kinetic term expanded into the photon current.",
    )
    _print_model_vertex_case(
        "gauge: abelian scalar current from CovD",
        MODEL_SCALAR_QED_COVARIANT,
        PhiQEDField.bar, PhiQEDField, GaugeField,
        description="Complex-scalar kinetic term expanded into the one-photon current.",
    )
    _print_model_vertex_case(
        "gauge: abelian scalar contact from CovD",
        MODEL_SCALAR_QED_COVARIANT,
        PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField,
        description="The same scalar kinetic term also generates the two-photon contact.",
    )
    _print_model_vertex_case(
        "gauge: non-abelian fermion current from CovD",
        MODEL_QCD_COVARIANT,
        QuarkField.bar, QuarkField, GluonField,
        description="Non-abelian fermion current generated from the same declarative pattern.",
    )


def _run_minimal_demo():
    print("# " + "=" * 79)
    print("Demo: minimal gauge compiler (Lagrangian API)\n")

    compiled_qcd = compile_minimal_gauge_interactions(MODEL_QCD_BASE)
    _print_vertex_block(
        "minimal gauge compiler: quark-gluon",
        lagrangian_terms=_matching_lagrangian_terms(compiled_qcd, QuarkField.bar, QuarkField, GluonField),
        description="Compiled non-abelian fermion current from model metadata.",
        vertex=Lagrangian(terms=compiled_qcd).feynman_rule(QuarkField.bar, QuarkField, GluonField),
    )

    compiled_qed = compile_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
    _print_vertex_block(
        "minimal gauge compiler: fermion QED",
        lagrangian_terms=_matching_lagrangian_terms(compiled_qed, PsiQEDField.bar, PsiQEDField, GaugeField),
        description="Compiled abelian fermion current from charge metadata.",
        vertex=Lagrangian(terms=compiled_qed).feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField),
    )

    compiled_scalar_qed = compile_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
    _print_vertex_block(
        "minimal gauge compiler: scalar QED current",
        lagrangian_terms=_matching_lagrangian_terms(compiled_scalar_qed, PhiQEDField.bar, PhiQEDField, GaugeField),
        description="Current terms compiled from one complex-scalar kinetic term.",
        vertex=Lagrangian(terms=compiled_scalar_qed).feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField),
    )
    _print_vertex_block(
        "minimal gauge compiler: scalar QED contact",
        lagrangian_terms=_matching_lagrangian_terms(
            compiled_scalar_qed, PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField,
        ),
        description="Two-gauge contact compiled from the same scalar kinetic term.",
        vertex=Lagrangian(terms=compiled_scalar_qed).feynman_rule(
            PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField,
        ),
    )

    compiled_scalar_qcd = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)
    _print_vertex_block(
        "minimal gauge compiler: scalar QCD current",
        lagrangian_terms=_matching_lagrangian_terms(compiled_scalar_qcd, PhiQCDField.bar, PhiQCDField, GluonField),
        description="Current terms compiled from non-abelian scalar representation metadata.",
        vertex=Lagrangian(terms=compiled_scalar_qcd).feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField),
    )
    _print_vertex_block(
        "minimal gauge compiler: scalar QCD contact",
        lagrangian_terms=_matching_lagrangian_terms(
            compiled_scalar_qcd, PhiQCDField.bar, PhiQCDField, GluonField, GluonField,
        ),
        description="Two-gluon contact with explicit generator ordering.",
        vertex=Lagrangian(terms=compiled_scalar_qcd).feynman_rule(
            PhiQCDField.bar, PhiQCDField, GluonField, GluonField,
        ),
    )

    try:
        compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS)
    except ValueError as exc:
        _print_vertex_block(
            "minimal gauge compiler: repeated-slot ambiguity",
            description="Repeated same-kind representation slots require an explicit slot selection.",
            error=f"rejected: {exc}",
        )

    compiled_bislot = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BISLOT_BASE)
    _print_vertex_block(
        "minimal gauge compiler: repeated-slot scalar QCD current",
        lagrangian_terms=_matching_lagrangian_terms(compiled_bislot, PhiBiField.bar, PhiBiField, GluonField),
        description="One active color slot plus one spectator identity.",
        vertex=Lagrangian(terms=compiled_bislot).feynman_rule(PhiBiField.bar, PhiBiField, GluonField),
    )


def _run_covariant_demo():
    compiled_qcd = compile_covariant_terms(MODEL_QCD_COVARIANT)
    compiled_qed = compile_covariant_terms(MODEL_QED_FERMION_COVARIANT)
    compiled_mixed = compile_covariant_terms(MODEL_MIXED_FERMION_COVARIANT)
    compiled_scalar_qed = compile_covariant_terms(MODEL_SCALAR_QED_COVARIANT)
    compiled_scalar_qcd = compile_covariant_terms(MODEL_SCALAR_QCD_COVARIANT)
    compiled_mixed_scalar = compile_covariant_terms(MODEL_MIXED_SCALAR_COVARIANT)
    compiled_bislot = compile_covariant_terms(MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM)
    compiled_photon = compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    compiled_yang_mills = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)

    print("# " + "=" * 79)
    print("Demo: covariant compiler (Lagrangian API)\n")

    _print_vertex_block(
        "covariant: qbar i gamma^mu D_mu q",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_COVARIANT, compiled_qcd, QuarkField.bar, QuarkField, GluonField),
        description=_model_decl_label(
            MODEL_QCD_COVARIANT,
            "covariant_terms",
            "Dirac kinetic term expanded through the gauge compiler",
        ),
        vertex=MODEL_QCD_COVARIANT.lagrangian().feynman_rule(QuarkField.bar, QuarkField, GluonField),
    )
    _print_vertex_block(
        "covariant: PsiQEDbar i gamma^mu D_mu PsiQED",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_FERMION_COVARIANT, compiled_qed, PsiQEDField.bar, PsiQEDField, GaugeField),
        description=_model_decl_label(
            MODEL_QED_FERMION_COVARIANT,
            "covariant_terms",
            "Abelian Dirac kinetic term expanded through the gauge compiler",
        ),
        vertex=MODEL_QED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField),
    )
    _print_vertex_block(
        "covariant: one Dirac term over QCD+QED [gluon piece]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_MIXED_FERMION_COVARIANT, compiled_mixed, PsiMixField.bar, PsiMixField, GluonField),
        description="Single kinetic term expanded over all matching gauge groups.",
        vertex=MODEL_MIXED_FERMION_COVARIANT.lagrangian().feynman_rule(
            PsiMixField.bar, PsiMixField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: one Dirac term over QCD+QED [photon piece]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_MIXED_FERMION_COVARIANT, compiled_mixed, PsiMixField.bar, PsiMixField, GaugeField),
        description="Same kinetic term, second gauge-group contribution.",
        vertex=MODEL_MIXED_FERMION_COVARIANT.lagrangian().feynman_rule(
            PsiMixField.bar, PsiMixField, GaugeField,
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu phi)^dagger (D^mu phi) current",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_SCALAR_QED_COVARIANT, compiled_scalar_qed, PhiQEDField.bar, PhiQEDField, GaugeField),
        description=_model_decl_label(
            MODEL_SCALAR_QED_COVARIANT,
            "covariant_terms",
            "Complex-scalar kinetic term expanded through the gauge compiler",
        ),
        vertex=MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField),
    )
    _print_vertex_block(
        "covariant: (D_mu phi)^dagger (D^mu phi) contact",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_SCALAR_QED_COVARIANT, compiled_scalar_qed, PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField,
        ),
        description="Two-gauge contact contribution from the same complex-scalar kinetic term.",
        vertex=MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(
            PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField,
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiQCD)^dagger (D^mu PhiQCD) current",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_SCALAR_QCD_COVARIANT, compiled_scalar_qcd, PhiQCDField.bar, PhiQCDField, GluonField),
        description=_model_decl_label(
            MODEL_SCALAR_QCD_COVARIANT,
            "covariant_terms",
            "Non-abelian complex-scalar kinetic term expanded through the gauge compiler",
        ),
        vertex=MODEL_SCALAR_QCD_COVARIANT.lagrangian().feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiQCD)^dagger (D^mu PhiQCD) contact",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_SCALAR_QCD_COVARIANT, compiled_scalar_qcd, PhiQCDField.bar, PhiQCDField, GluonField, GluonField,
        ),
        description="Two-gluon contact contribution with explicit generator ordering.",
        vertex=MODEL_SCALAR_QCD_COVARIANT.lagrangian().feynman_rule(
            PhiQCDField.bar, PhiQCDField, GluonField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [gluon current]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_MIXED_SCALAR_COVARIANT, compiled_mixed_scalar, PhiMixField.bar, PhiMixField, GluonField),
        description="Single complex-scalar kinetic term expanded over all matching gauge groups.",
        vertex=MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(
            PhiMixField.bar, PhiMixField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [photon current]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_MIXED_SCALAR_COVARIANT, compiled_mixed_scalar, PhiMixField.bar, PhiMixField, GaugeField),
        description="Same kinetic term, abelian current with the color slot left as a spectator identity.",
        vertex=MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(
            PhiMixField.bar, PhiMixField, GaugeField,
        ),
    )
    _print_vertex_block(
        "covariant: one scalar term over QCD+QED [mixed contact]",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_MIXED_SCALAR_COVARIANT, compiled_mixed_scalar, PhiMixField.bar, PhiMixField, GluonField, GaugeField,
        ),
        description="Ordered cross-group contact pieces from the same kinetic term.",
        vertex=MODEL_MIXED_SCALAR_COVARIANT.lagrangian().feynman_rule(
            PhiMixField.bar, PhiMixField, GluonField, GaugeField,
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiBi)^dagger (D^mu PhiBi) [bislot, slot_policy='sum']",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM, compiled_bislot, PhiBiField.bar, PhiBiField, GluonField),
        description="Bislotted scalar kinetic term expanded by summing over both identical color-fundamental slots.",
        vertex=MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM.lagrangian().feynman_rule(
            PhiBiField.bar, PhiBiField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: (D_mu PhiBi)^dagger (D^mu PhiBi) contact [bislot sum]",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM, compiled_bislot, PhiBiField.bar, PhiBiField, GluonField, GluonField,
        ),
        description="Sum of all ordered slot-pair contact contributions.",
        vertex=MODEL_SCALAR_QCD_BISLOT_COVARIANT_SUM.lagrangian().feynman_rule(
            PhiBiField.bar, PhiBiField, GluonField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: -1/4 F_mu nu F^mu nu [abelian bilinear]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_GAUGE_COVARIANT, compiled_photon, GaugeField, GaugeField),
        description=_model_decl_label(
            MODEL_QED_GAUGE_COVARIANT,
            "gauge_kinetic_terms",
            "-1/4 U1QED field strength squared",
        ),
        vertex=MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField),
    )
    _print_vertex_block(
        "covariant: -1/4 G^a_mu nu G^{a mu nu} [bilinear]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField),
        description=_model_decl_label(
            MODEL_QCD_GAUGE_COVARIANT,
            "gauge_kinetic_terms",
            "-1/4 SU3C field strength squared",
        ),
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField),
    )
    _print_vertex_block(
        "covariant: Yang-Mills 3-gauge vertex",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField, GluonField),
        description="Cubic self-interaction term from the non-abelian field strength.",
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
            GluonField, GluonField, GluonField,
        ),
    )
    _print_vertex_block(
        "covariant: Yang-Mills 4-gauge vertex",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField, GluonField, GluonField,
        ),
        description="Quartic self-interaction term from the non-abelian field strength.",
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
            GluonField, GluonField, GluonField, GluonField,
        ),
    )


def _run_pure_gauge_demo():
    compiled_photon = compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    compiled_yang_mills = compile_covariant_terms(MODEL_QCD_GAUGE_COVARIANT)

    print("# " + "=" * 79)
    print("Demo: pure gauge (Lagrangian API)\n")

    _print_vertex_block(
        "pure gauge: QED photon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_GAUGE_COVARIANT, compiled_photon, GaugeField, GaugeField),
        description=_model_decl_label(
            MODEL_QED_GAUGE_COVARIANT,
            "gauge_kinetic_terms",
            "-1/4 U1QED field strength squared",
        ),
        vertex=MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField),
    )
    _print_vertex_block(
        "pure gauge: QCD gluon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField),
        description=_model_decl_label(
            MODEL_QCD_GAUGE_COVARIANT,
            "gauge_kinetic_terms",
            "-1/4 SU3C field strength squared",
        ),
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField),
    )
    _print_vertex_block(
        "pure gauge: QCD 3-gluon",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField, GluonField),
        description="Cubic Yang-Mills self interaction.",
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField, GluonField),
    )
    _print_vertex_block(
        "pure gauge: QCD 4-gluon",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_QCD_GAUGE_COVARIANT, compiled_yang_mills, GluonField, GluonField, GluonField, GluonField,
        ),
        description="Quartic Yang-Mills self interaction.",
        vertex=MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
            GluonField, GluonField, GluonField, GluonField,
        ),
    )


def _run_gauge_fixed_demo():
    compiled_qed_gf = compile_covariant_terms(MODEL_QED_GAUGE_FIXING_COVARIANT)
    compiled_qcd_gf = compile_covariant_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT)
    compiled_qcd_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    compiled_qed_full = compile_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)
    compiled_qcd_full = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)

    print("# " + "=" * 79)
    print("Demo: ordinary gauge fixing and ghosts (Lagrangian API)\n")

    _print_vertex_block(
        "gauge-fixed: -(1/2 xi) (partial.A)^2 [abelian]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_GAUGE_FIXING_COVARIANT, compiled_qed_gf, GaugeField, GaugeField),
        description=_model_decl_label(
            MODEL_QED_GAUGE_FIXING_COVARIANT,
            "gauge_fixing_terms",
            "-(1/2 xiQED) (U1QED gauge fixing)",
        ),
        vertex=MODEL_QED_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField),
    )
    _print_vertex_block(
        "gauge-fixed: -(1/2 xi) (partial.G)^2 [non-abelian]",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GAUGE_FIXING_COVARIANT, compiled_qcd_gf, GluonField, GluonField),
        description=_model_decl_label(
            MODEL_QCD_GAUGE_FIXING_COVARIANT,
            "gauge_fixing_terms",
            "-(1/2 xiQCD) (SU3C gauge fixing)",
        ),
        vertex=MODEL_QCD_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField),
    )
    _print_vertex_block(
        "gauge-fixed: ordinary photon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_ORDINARY_GAUGE_FIXED, compiled_qed_full, GaugeField, GaugeField),
        description="Gauge kinetic plus ordinary gauge fixing combined into the full two-point photon vertex.",
        vertex=MODEL_QED_ORDINARY_GAUGE_FIXED.lagrangian().feynman_rule(GaugeField, GaugeField),
    )
    _print_vertex_block(
        "gauge-fixed: Faddeev-Popov ghost bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GHOST_COVARIANT, compiled_qcd_ghost, GhostGluonField.bar, GhostGluonField),
        description=_model_decl_label(
            MODEL_QCD_GHOST_COVARIANT,
            "ghost_terms",
            "SU3C Faddeev-Popov ghosts bilinear",
        ),
        vertex=MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
            GhostGluonField.bar, GhostGluonField,
        ),
    )
    _print_vertex_block(
        "gauge-fixed: ghost-gluon interaction",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GHOST_COVARIANT, compiled_qcd_ghost, GhostGluonField.bar, GluonField, GhostGluonField),
        description="Ordinary non-abelian ghost coupling.",
        vertex=MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
            GhostGluonField.bar, GluonField, GhostGluonField,
        ),
    )
    _print_vertex_block(
        "gauge-fixed: ordinary gluon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GluonField, GluonField),
        description="Yang-Mills bilinear plus ordinary gauge fixing combined into the full two-point gluon vertex.",
        vertex=MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian().feynman_rule(GluonField, GluonField),
    )


def _run_ghost_demo():
    compiled_qcd_ghost = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)

    print("# " + "=" * 79)
    print("Demo: ghost sector (Lagrangian API)\n")

    _print_vertex_block(
        "ghost: bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GHOST_COVARIANT, compiled_qcd_ghost, GhostGluonField.bar, GhostGluonField),
        description=_model_decl_label(
            MODEL_QCD_GHOST_COVARIANT,
            "ghost_terms",
            "SU3C Faddeev-Popov ghosts bilinear",
        ),
        vertex=MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
            GhostGluonField.bar, GhostGluonField,
        ),
    )
    _print_vertex_block(
        "ghost: ghost-gluon interaction",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_GHOST_COVARIANT, compiled_qcd_ghost, GhostGluonField.bar, GluonField, GhostGluonField),
        description="Derivative on the antighost gives the antighost momentum in the cubic vertex.",
        vertex=MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
            GhostGluonField.bar, GluonField, GhostGluonField,
        ),
    )


def _run_full_demo():
    compiled_qcd_full = compile_covariant_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED)
    compiled_qed_full = compile_covariant_terms(MODEL_QED_ORDINARY_GAUGE_FIXED)
    lagrangian_qcd = MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian()
    lagrangian_qed = MODEL_QED_ORDINARY_GAUGE_FIXED.lagrangian()

    print("# " + "=" * 79)
    print("Demo: full gauge-fixed models (Lagrangian API)\n")

    _print_vertex_block(
        "full QCD: gluon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GluonField, GluonField),
        description="Gauge kinetic plus ordinary gauge fixing in one model.",
        vertex=lagrangian_qcd.feynman_rule(GluonField, GluonField),
    )
    _print_vertex_block(
        "full QCD: 3-gluon",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GluonField, GluonField, GluonField),
        description="Non-abelian cubic self interaction inside the fully gauge-fixed model.",
        vertex=lagrangian_qcd.feynman_rule(GluonField, GluonField, GluonField),
    )
    _print_vertex_block(
        "full QCD: 4-gluon",
        lagrangian_terms=_demo_lagrangian_terms(
            MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GluonField, GluonField, GluonField, GluonField,
        ),
        description="Quartic Yang-Mills self interaction inside the fully gauge-fixed model.",
        vertex=lagrangian_qcd.feynman_rule(GluonField, GluonField, GluonField, GluonField),
    )
    _print_vertex_block(
        "full QCD: ghost bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GhostGluonField.bar, GhostGluonField),
        description="Ghost kinetic term inside the fully gauge-fixed model.",
        vertex=lagrangian_qcd.feynman_rule(GhostGluonField.bar, GhostGluonField),
    )
    _print_vertex_block(
        "full QCD: ghost-gluon",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_ORDINARY_GAUGE_FIXED, compiled_qcd_full, GhostGluonField.bar, GluonField, GhostGluonField),
        description="Ghost-gauge coupling inside the fully gauge-fixed model.",
        vertex=lagrangian_qcd.feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField),
    )
    _print_vertex_block(
        "full QED: photon bilinear",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QED_ORDINARY_GAUGE_FIXED, compiled_qed_full, GaugeField, GaugeField),
        description="Gauge kinetic plus ordinary gauge fixing for the abelian model.",
        vertex=lagrangian_qed.feynman_rule(GaugeField, GaugeField),
    )


def _run_cross_demo():
    compiled_qcd = compile_covariant_terms(MODEL_QCD_COVARIANT)
    model_vertex = MODEL_QCD_COVARIANT.lagrangian().feynman_rule(QuarkField.bar, QuarkField, GluonField)
    explicit_vertex = _lagrangian_vertex(compiled_qcd, QuarkField.bar, QuarkField, GluonField)

    print("# " + "=" * 79)
    print("Demo: cross-checks (Lagrangian API)\n")

    _print_vertex_block(
        "cross: qbar i gamma^mu D_mu q",
        lagrangian_terms=_demo_lagrangian_terms(MODEL_QCD_COVARIANT, compiled_qcd, QuarkField.bar, QuarkField, GluonField),
        description="Model.lagrangian() and an explicit Lagrangian built from compiled terms should agree.",
        vertex=model_vertex,
        compact_override=explicit_vertex,
        interpretation="Here the compact form is the explicit compiled-Lagrangian reference vertex.",
    )


def _run_role_demo():
    print("# " + "=" * 79)
    print("Demo: role regressions (Lagrangian API)\n")

    _print_vertex_block(
        "role: complex scalar conjugation filtering",
        lagrangian_terms=(TERM_phiCdag_phiC,),
        description="The same term should match both Phi.bar,Phi and reversed external order.",
        vertex=Lagrangian(terms=(TERM_phiCdag_phiC,)).feynman_rule(PhiCField.bar, PhiCField),
        compact_override=Lagrangian(terms=(TERM_phiCdag_phiC,)).feynman_rule(PhiCField, PhiCField.bar),
        interpretation="Compact form shows the reversed-leg query, which must agree with the primary vertex.",
    )


def _run_demo(suite):
    """Print human-readable vertex output for inspection."""

    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()
        _run_mixed_demo()
    if suite in ("gauge", "all"):
        _run_gauge_demo()
    if suite in ("minimal", "all"):
        _run_minimal_demo()
    if suite in ("covariant", "all"):
        _run_covariant_demo()
    if suite in ("puregauge", "all"):
        _run_pure_gauge_demo()
    if suite in ("gaugefix", "all"):
        _run_gauge_fixed_demo()
    if suite in ("ghost", "all"):
        _run_ghost_demo()
    if suite in ("full", "all"):
        _run_full_demo()
    if suite == "cross":
        _run_cross_demo()
    if suite == "role":
        _run_role_demo()


# ===================================================================
# Test runner
# ===================================================================

def _run_all_tests():
    print("=" * 80)
    print("  Lagrangian API: scalar vertices")
    print("=" * 80)
    _run_scalar_tests()

    print("=" * 80)
    print("  Lagrangian API: fermion vertices")
    print("=" * 80)
    _run_fermion_tests()

    print("=" * 80)
    print("  Lagrangian API: mixed fermion+scalar derivatives")
    print("=" * 80)
    _run_mixed_derivative_tests()

    print("=" * 80)
    print("  Lagrangian API: gauge-ready vertices")
    print("=" * 80)
    _run_gauge_ready_tests()

    print("=" * 80)
    print("  Lagrangian API: compiled minimal gauge models")
    print("=" * 80)
    _run_compiled_minimal_tests()

    print("=" * 80)
    print("  Lagrangian API: covariant compiler")
    print("=" * 80)
    _run_covariant_tests()

    print("=" * 80)
    print("  Lagrangian API: pure-gauge sector")
    print("=" * 80)
    _run_pure_gauge_tests()

    print("=" * 80)
    print("  Lagrangian API: gauge fixing")
    print("=" * 80)
    _run_gauge_fixing_tests()

    print("=" * 80)
    print("  Lagrangian API: ghost sector")
    print("=" * 80)
    _run_ghost_tests()

    print("=" * 80)
    print("  Lagrangian API: full gauge-fixed models")
    print("=" * 80)
    _run_full_gauge_fixed_tests()

    print("=" * 80)
    print("  Lagrangian API: cross-checks vs explicit compiled Lagrangian")
    print("=" * 80)
    _run_cross_checks()

    print("=" * 80)
    print("  Lagrangian API: tensor canonicalization")
    print("=" * 80)
    _run_tensor_canonicalization_tests()

    print("=" * 80)
    print("  Lagrangian API: role / matcher regressions")
    print("=" * 80)
    _run_role_regression_tests()

    print("All Lagrangian API tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lagrangian API examples and tests.")
    parser.add_argument(
        "--suite",
        choices=(
            "scalar", "fermion", "mixed", "gauge", "covariant", "gaugefix",
            "full", "cross", "minimal", "puregauge", "ghost", "tensor", "role", "all",
        ),
        default="all",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Suppress the human-readable vertex output blocks.",
    )
    args = parser.parse_args()

    if not args.no_demo:
        _run_demo(args.suite)

    if not args.skip_tests:
        if args.suite == "all":
            _run_all_tests()
        elif args.suite == "scalar":
            _run_scalar_tests()
        elif args.suite == "fermion":
            _run_fermion_tests()
        elif args.suite == "mixed":
            _run_mixed_derivative_tests()
        elif args.suite == "gauge":
            _run_gauge_ready_tests()
        elif args.suite == "minimal":
            _run_compiled_minimal_tests()
        elif args.suite == "covariant":
            _run_covariant_tests()
        elif args.suite == "puregauge":
            _run_pure_gauge_tests()
        elif args.suite == "gaugefix":
            _run_gauge_fixing_tests()
        elif args.suite == "ghost":
            _run_ghost_tests()
        elif args.suite == "full":
            _run_full_gauge_fixed_tests()
        elif args.suite == "cross":
            _run_cross_checks()
        elif args.suite == "tensor":
            _run_tensor_canonicalization_tests()
        elif args.suite == "role":
            _run_role_regression_tests()
