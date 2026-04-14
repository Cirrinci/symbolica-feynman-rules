"""
Lagrangian API examples and regression tests.

This file mirrors src/examples.py but uses exclusively the FeynRules-style
Lagrangian API (Model.lagrangian().feynman_rule(...)).

Conventions (automatic):
  - Momenta: q1, q2, q3, ...
  - Indices: i1, i2, i3, ... (sequential across legs)
  - Conjugated fields: Field.bar
"""

import argparse

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
    bis,
    Delta,
    pcomp,
)
from spenso_structures import (
    SPINOR_KIND,
    gamma_matrix,
    gamma5_matrix,
    simplify_gamma_chain,
)
from model import (
    InteractionTerm,
    DerivativeAction,
    Lagrangian,
)
from operators import (
    psi_bar_psi,
    quark_gluon_current,
)
from gauge_compiler import (
    compile_covariant_terms,
    compile_minimal_gauge_interactions,
)

from examples import (
    # symbols
    d,
    lam4, g_sym, lamC, yF, gV, gS, eQED, xiQED, xiQCD,
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


# ---------------------------------------------------------------------------
# Momentum-conservation delta helpers
# ---------------------------------------------------------------------------

D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)


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
        coupling=S("gD"),
        fields=tuple(PhiField.occurrence() for _ in range(4)),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu), DerivativeAction(target=1, lorentz_index=nu)),
        label="gD * (d_mu phi)(d_nu phi) phi phi",
    )
    L_d = Lagrangian(terms=(TERM_deriv,))
    got_d = L_d.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    _check_nonzero(got_d, "L-API: phi^4 derivative (mu,nu)")

    TERM_deriv2 = InteractionTerm(
        coupling=S("gD2"),
        fields=tuple(PhiField.occurrence() for _ in range(4)),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu), DerivativeAction(target=1, lorentz_index=mu)),
        label="gD2 * (d_mu phi)(d^mu phi) phi phi",
    )
    L_d2 = Lagrangian(terms=(TERM_deriv2,))
    got_d2 = L_d2.feynman_rule(PhiField, PhiField, PhiField, PhiField)
    _check_nonzero(got_d2, "L-API: phi^4 derivative (mu,mu)")

    # phi^6: 6-point scalar vertex
    D6 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4 + q5 + q6)
    TERM_phi6 = InteractionTerm(
        coupling=S("lam6"),
        fields=tuple(PhiField.occurrence() for _ in range(6)),
        label="lam6 * phi^6",
    )
    L6 = Lagrangian(terms=(TERM_phi6,))
    got6 = L6.feynman_rule(PhiField, PhiField, PhiField, PhiField, PhiField, PhiField)
    _check(got6, 720 * I * S("lam6") * D6, "L-API: phi^6")

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

    # d_mu psibar: non-zero
    L = Lagrangian(terms=(TERM_dpsibar,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check_nonzero(got, "L-API: d_mu psibar * psi * phi * chi")

    # d_nu psi: non-zero
    L = Lagrangian(terms=(TERM_dpsi,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check_nonzero(got, "L-API: psibar * d_nu psi * phi * chi")

    # (d_mu phi)(d_nu chi): non-zero
    L = Lagrangian(terms=(TERM_dphi_dchi,))
    got = L.feynman_rule(PsiField.bar, PsiField, PhiField, ChiField)
    _check_nonzero(got, "L-API: psibar * psi * (d_mu phi)(d_nu chi)")

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
    _check_nonzero(got_sc, "L-API: complex scalar current")

    # Complex scalar contact: phiC^dag, phiC, A, A → non-zero
    L_ct = Lagrangian(terms=(TERM_complex_scalar_contact,))
    got_ct = L_ct.feynman_rule(PhiCField.bar, PhiCField, GaugeField, GaugeField)
    _check_nonzero(got_ct, "L-API: complex scalar contact")

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
    _check_nonzero(got, "L-API minimal: QCD quark-gluon")

    # QED fermion
    compiled_qed = compile_minimal_gauge_interactions(MODEL_QED_FERMION_BASE)
    L_qed = Lagrangian(terms=compiled_qed)
    got = L_qed.feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField)
    _check_nonzero(got, "L-API minimal: QED fermion")

    # Scalar QED current (3pt) and contact (4pt)
    compiled_sc_qed = compile_minimal_gauge_interactions(MODEL_SCALAR_QED_BASE)
    L_sc_qed = Lagrangian(terms=compiled_sc_qed)
    got_3pt = L_sc_qed.feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField)
    _check_nonzero(got_3pt, "L-API minimal: scalar QED current")
    got_4pt = L_sc_qed.feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField)
    _check_nonzero(got_4pt, "L-API minimal: scalar QED contact")

    # Scalar QCD current (3pt) and contact (4pt)
    compiled_sc_qcd = compile_minimal_gauge_interactions(MODEL_SCALAR_QCD_BASE)
    L_sc_qcd = Lagrangian(terms=compiled_sc_qcd)
    got_3pt = L_sc_qcd.feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField)
    _check_nonzero(got_3pt, "L-API minimal: scalar QCD current")
    got_4pt = L_sc_qcd.feynman_rule(PhiQCDField.bar, PhiQCDField, GluonField, GluonField)
    _check_nonzero(got_4pt, "L-API minimal: scalar QCD contact")

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
# 10. Cross-checks: Lagrangian API vs old _model_vertex() pipeline
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
# Demo output
# ===================================================================

def _run_demo(suite):
    """Print human-readable vertex output for inspection."""

    if suite in ("scalar", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): scalar\n")
        L = Lagrangian(terms=(TERM_phi4,))
        print(f"  phi^4:        {L.feynman_rule(PhiField, PhiField, PhiField, PhiField)}")
        L = Lagrangian(terms=(TERM_phi2chi2,))
        print(f"  phi^2 chi^2:  {L.feynman_rule(PhiField, PhiField, ChiField, ChiField)}")
        L = Lagrangian(terms=(TERM_phiCdag_phiC,))
        print(f"  phi^dag phi:  {L.feynman_rule(PhiCField.bar, PhiCField)}")
        print()

    if suite in ("fermion", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): fermion\n")
        L = Lagrangian(terms=(TERM_yukawa,))
        print(f"  Yukawa:         {L.feynman_rule(PsiField.bar, PsiField, PhiField)}")
        L = Lagrangian(terms=(TERM_vec_current,))
        print(f"  Vector current: {L.feynman_rule(PsiField.bar, PsiField, GaugeField)}")
        L = Lagrangian(terms=(TERM_axial_current,))
        print(f"  Axial current:  {L.feynman_rule(PsiField.bar, PsiField, GaugeField)}")
        L = Lagrangian(terms=(TERM_psibar_psi_sq,))
        print(f"  (psibar psi)^2: {L.feynman_rule(PsiField.bar, PsiField, PsiField.bar, PsiField)}")
        L = Lagrangian(terms=(TERM_current_current,))
        print(f"  Current-current: {simplify_gamma_chain(L.feynman_rule(PsiField.bar, PsiField, PsiField.bar, PsiField))}")
        print()

    if suite in ("gauge", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): gauge-ready\n")
        L = Lagrangian(terms=(TERM_quark_gluon,))
        print(f"  Quark-gluon: {L.feynman_rule(QuarkField.bar, QuarkField, GluonField)}")
        L = Lagrangian(terms=(TERM_complex_scalar_current_phi, TERM_complex_scalar_current_phidag))
        print(f"  Scalar current: {L.feynman_rule(PhiCField.bar, PhiCField, GaugeField)}")
        L = Lagrangian(terms=(TERM_complex_scalar_contact,))
        print(f"  Scalar contact: {L.feynman_rule(PhiCField.bar, PhiCField, GaugeField, GaugeField)}")
        print()

    if suite in ("covariant", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): covariant\n")
        print(f"  QCD quark-gluon:  {MODEL_QCD_COVARIANT.lagrangian().feynman_rule(QuarkField.bar, QuarkField, GluonField)}")
        print(f"  QED fermion:      {MODEL_QED_FERMION_COVARIANT.lagrangian().feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField)}")
        print(f"  Scalar QED 3pt:   {MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField)}")
        print(f"  Scalar QED 4pt:   {MODEL_SCALAR_QED_COVARIANT.lagrangian().feynman_rule(PhiQEDField.bar, PhiQEDField, GaugeField, GaugeField)}")
        print()

    if suite in ("gaugefix", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): gauge fixing + ghosts\n")
        print(f"  QED GF:      {MODEL_QED_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)}")
        print(f"  QCD GF:      {MODEL_QCD_GAUGE_FIXING_COVARIANT.lagrangian().feynman_rule(GluonField, GluonField)}")
        print(f"  Ghost bilin: {MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(GhostGluonField.bar, GhostGluonField)}")
        print(f"  Ghost-gluon: {MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField)}")
        print()

    if suite in ("full", "all"):
        print("# " + "=" * 79)
        print("Demo (Lagrangian API): full gauge-fixed\n")
        L_qcd = MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian()
        print(f"  Full QCD gluon bilinear: {L_qcd.feynman_rule(GluonField, GluonField)}")
        print(f"  Full QCD 3-gluon:        {L_qcd.feynman_rule(GluonField, GluonField, GluonField)}")
        print(f"  Full QCD 4-gluon:        {L_qcd.feynman_rule(GluonField, GluonField, GluonField, GluonField)}")
        print(f"  Full QCD ghost bilinear: {L_qcd.feynman_rule(GhostGluonField.bar, GhostGluonField)}")
        print(f"  Full QCD ghost-gluon:    {L_qcd.feynman_rule(GhostGluonField.bar, GluonField, GhostGluonField)}")
        L_qed = MODEL_QED_ORDINARY_GAUGE_FIXED.lagrangian()
        print(f"  Full QED photon bilinear: {L_qed.feynman_rule(GaugeField, GaugeField)}")
        print()


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
    print("  Lagrangian API: cross-checks vs old pipeline")
    print("=" * 80)
    _run_cross_checks()

    print("All Lagrangian API tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lagrangian API examples and tests.")
    parser.add_argument(
        "--suite",
        choices=(
            "scalar", "fermion", "mixed", "gauge", "covariant", "gaugefix",
            "full", "cross", "minimal", "puregauge", "ghost", "all",
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
