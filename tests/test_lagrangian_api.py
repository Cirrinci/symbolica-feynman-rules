"""Tests for the FeynRules-style Lagrangian API.

Validates that ``Lagrangian.feynman_rule()`` produces the same vertex factors
as the existing ``vertex_factor()`` pipeline, using automatic index and
momentum conventions.

Covers:
- Field.bar / ConjugateField basics
- Lagrangian composition (Lagrangian + Lagrangian, + InteractionTerm, sum)
- InteractionTerm + InteractionTerm composition
- Scalar, fermion, and gauge vertices via the Lagrangian API
- Model.lagrangian() idempotency (no double-counting on precompiled models)
- Compiled sectors: scalar covariant, gauge kinetic, gauge-fixing, ghosts
"""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from symbolica import S, Expression  # noqa: E402

from compiler.gauge import (  # noqa: E402
    compile_covariant_terms,
    with_compiled_covariant_terms,
)
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_INDEX,
    COLOR_FUND_KIND,
    CompiledLagrangian,
    CovD,
    DeclaredLagrangian,
    FieldStrength,
    Gamma,
    Gamma5,
    LORENTZ_INDEX,
    LORENTZ_KIND,
    Metric,
    SPINOR_INDEX,
    SPINOR_KIND,
    T,
    ConjugateField,
    ComplexScalarKineticTerm,
    DerivativeAction,
    DiracKineticTerm,
    Field,
    GaugeFixing,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    GhostLagrangian,
    InteractionTerm,
    Lagrangian,
    Model,
    PartialD,
)
from symbolic.vertex_engine import (  # noqa: E402
    Delta,
    I,
    pi,
    pcomp,
    simplify_deltas,
    simplify_vertex,
    vertex_factor,
)
from symbolic.spenso_structures import (  # noqa: E402
    gauge_generator,
    gamma5_matrix,
    gamma_matrix,
    lorentz_metric,
    simplify_gamma_chain,
    structure_constant,
)
from lagrangian.operators import (  # noqa: E402
    current_current,
    psi_bar_gamma_psi,
    psi_bar_psi,
    quark_gluon_current,
    scalar_gauge_contact,
)
from tests.support.builders import (  # noqa: E402
    canon as _canon,
    dirac_covd_decl as _dirac_decl,
    gauge_kinetic_decl as _gauge_decl,
    make_complex_scalar,
    make_dirac_fermion as _make_dirac_fermion,
    make_ghost as _make_ghost,
    make_gluon as _make_gluon,
    make_photon as _make_photon,
    make_su3 as _make_su3,
    make_u1 as _make_u1,
    scalar_covd_decl as _scalar_decl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_vertex(interaction, legs, d=None):
    """Compute a reference vertex via the existing low-level pipeline."""
    if d is None:
        d = S("d")
    return simplify_vertex(vertex_factor(
        interaction=interaction,
        external_legs=legs,
        x=S("x_"),
        d=d,
        strip_externals=True,
        include_delta=True,
    ), external_legs=legs)

# ---------------------------------------------------------------------------
# Shared sample model cases
# ---------------------------------------------------------------------------

gS, eQED, xiQED, xiQCD = S("gS", "eQED", "xiQED", "xiQCD")
qPhi, qPsi, qMix, qPhiMix = S("qPhi", "qPsi", "qMix", "qPhiMix")
phiC0, phiCdag0 = S("phiC0", "phiCdag0")
phiQCD0, phiQCDdag0 = S("phiQCD0", "phiQCDdag0")
phiMix0, phiMixdag0 = S("phiMix0", "phiMixdag0")
ghG0, ghGbar0 = S("ghG0", "ghGbar0")
psibar0, psi0 = S("psibar0", "psi0")
psibarQED0, psiQED0 = S("psibarQED0", "psiQED0")
psibarMix0, psiMix0 = S("psibarMix0", "psiMix0")
A0, G0 = S("A0", "G0")
mu_sample, nu_sample = S("mu", "nu")

PhiQEDField = make_complex_scalar(
    "PhiQED",
    symbol=phiC0,
    conjugate_symbol=phiCdag0,
    charge=qPhi,
)
PhiQCDField = make_complex_scalar(
    "PhiQCD",
    symbol=phiQCD0,
    conjugate_symbol=phiQCDdag0,
    color=True,
)
PhiMixField = make_complex_scalar(
    "PhiMix",
    symbol=phiMix0,
    conjugate_symbol=phiMixdag0,
    color=True,
    charge=qPhiMix,
)
PsiQEDField = _make_dirac_fermion(
    "PsiQED",
    symbol=psiQED0,
    conjugate_symbol=psibarQED0,
    charge=qPsi,
)
PsiMixField = _make_dirac_fermion(
    "PsiMix",
    symbol=psiMix0,
    conjugate_symbol=psibarMix0,
    color=True,
    charge=qMix,
)
GaugeField = _make_photon(name="A", symbol=A0)
QuarkField = _make_dirac_fermion(
    "q",
    symbol=psi0,
    conjugate_symbol=psibar0,
    color=True,
)
GluonField = _make_gluon(name="G", symbol=G0)
GhostGluonField = _make_ghost(
    name="ghG",
    symbol=ghG0,
    conjugate_symbol=ghGbar0,
)

QCD_GROUP = _make_su3(gS, GluonField.symbol, ghost_sym=GhostGluonField.symbol, name="SU3C")
QED_GROUP = _make_u1(eQED, GaugeField.symbol, name="U1QED")

MODEL_QCD_COVARIANT = Model(
    name="QCD-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(QuarkField, GluonField),
    lagrangian_decl=_dirac_decl(QuarkField, mu=mu_sample),
)
MODEL_SCALAR_QED_COVARIANT = Model(
    name="ScalarQED-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(PhiQEDField, GaugeField),
    lagrangian_decl=_scalar_decl(PhiQEDField, mu=mu_sample),
)
MODEL_SCALAR_QCD_COVARIANT = Model(
    name="ScalarQCD-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(PhiQCDField, GluonField),
    lagrangian_decl=_scalar_decl(PhiQCDField, mu=mu_sample),
)
MODEL_QED_FERMION_COVARIANT = Model(
    name="FermionQED-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(PsiQEDField, GaugeField),
    lagrangian_decl=_dirac_decl(PsiQEDField, mu=mu_sample),
)
MODEL_MIXED_FERMION_COVARIANT = Model(
    name="MixedQCDQED-covariant",
    gauge_groups=(QCD_GROUP, QED_GROUP),
    fields=(PsiMixField, GluonField, GaugeField),
    lagrangian_decl=_dirac_decl(PsiMixField, mu=mu_sample),
)
MODEL_MIXED_SCALAR_COVARIANT = Model(
    name="MixedScalarQCDQED-covariant",
    gauge_groups=(QCD_GROUP, QED_GROUP),
    fields=(PhiMixField, GluonField, GaugeField),
    lagrangian_decl=_scalar_decl(PhiMixField, mu=mu_sample),
)
MODEL_QED_GAUGE_COVARIANT = Model(
    name="QEDGauge-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    lagrangian_decl=_gauge_decl(QED_GROUP, mu=mu_sample, nu=nu_sample),
)
MODEL_QCD_GAUGE_COVARIANT = Model(
    name="QCDGauge-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    lagrangian_decl=_gauge_decl(QCD_GROUP, mu=mu_sample, nu=nu_sample),
)
MODEL_QED_GAUGE_FIXING_COVARIANT = Model(
    name="QEDGaugeFixing-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    lagrangian_decl=GaugeFixing(QED_GROUP, xi=xiQED),
)
MODEL_QCD_GAUGE_FIXING_COVARIANT = Model(
    name="QCDGaugeFixing-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    lagrangian_decl=GaugeFixing(QCD_GROUP, xi=xiQCD),
)
MODEL_QED_ORDINARY_GAUGE_FIXED = Model(
    name="QEDGaugeFixed-covariant",
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    lagrangian_decl=(
        _gauge_decl(QED_GROUP, mu=mu_sample, nu=nu_sample)
        + GaugeFixing(QED_GROUP, xi=xiQED)
    ),
)
MODEL_QCD_GHOST_COVARIANT = Model(
    name="QCDGhost-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField, GhostGluonField),
    lagrangian_decl=GhostLagrangian(QCD_GROUP),
)
MODEL_QCD_ORDINARY_GAUGE_FIXED = Model(
    name="QCDGaugeFixed-covariant",
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField, GhostGluonField),
    lagrangian_decl=(
        _gauge_decl(QCD_GROUP, mu=mu_sample, nu=nu_sample)
        + GaugeFixing(QCD_GROUP, xi=xiQCD)
        + GhostLagrangian(QCD_GROUP)
    ),
)


# ---------------------------------------------------------------------------
# Field.bar / ConjugateField basics
# ---------------------------------------------------------------------------

def test_field_bar_returns_conjugate_field():
    phi = Field(name="phi", spin=0, self_conjugate=False)
    cf = phi.bar
    assert isinstance(cf, ConjugateField)
    assert cf.field is phi


def test_field_bar_self_conjugate():
    phi = Field(name="phi", spin=0, self_conjugate=True)
    cf = phi.bar
    assert isinstance(cf, ConjugateField)
    assert cf.field is phi


# ---------------------------------------------------------------------------
# Lagrangian composition
# ---------------------------------------------------------------------------

def test_lagrangian_add():
    t1 = InteractionTerm(coupling=S("g"), fields=())
    t2 = InteractionTerm(coupling=S("h"), fields=())
    L1 = Lagrangian(terms=(t1,))
    L2 = Lagrangian(terms=(t2,))
    L3 = L1 + L2
    assert len(L3.terms) == 2
    assert L3.terms[0] is t1
    assert L3.terms[1] is t2


def test_lagrangian_radd_zero():
    L = Lagrangian(terms=(InteractionTerm(coupling=S("g"), fields=()),))
    result = 0 + L
    assert isinstance(result, Lagrangian)
    assert len(result.terms) == 1


def test_lagrangian_sum():
    L1 = Lagrangian(terms=(InteractionTerm(coupling=S("g"), fields=()),))
    L2 = Lagrangian(terms=(InteractionTerm(coupling=S("h"), fields=()),))
    total = sum([L1, L2], Lagrangian())
    assert isinstance(total, Lagrangian)
    assert len(total.terms) == 2


def test_lagrangian_add_interaction_term():
    t1 = InteractionTerm(coupling=S("g"), fields=())
    t2 = InteractionTerm(coupling=S("h"), fields=())
    L = Lagrangian(terms=(t1,))
    result = L + t2
    assert isinstance(result, Lagrangian)
    assert len(result.terms) == 2
    assert result.terms[1] is t2


def test_interaction_term_add():
    """InteractionTerm + InteractionTerm produces a Lagrangian."""
    t1 = InteractionTerm(coupling=S("g"), fields=())
    t2 = InteractionTerm(coupling=S("h"), fields=())
    result = t1 + t2
    assert isinstance(result, Lagrangian)
    assert len(result.terms) == 2
    assert result.terms[0] is t1
    assert result.terms[1] is t2


def test_interaction_term_add_to_lagrangian():
    """InteractionTerm + Lagrangian produces a Lagrangian (radd path)."""
    t = InteractionTerm(coupling=S("g"), fields=())
    L = Lagrangian(terms=(InteractionTerm(coupling=S("h"), fields=()),))
    result = t + L
    assert isinstance(result, Lagrangian)
    assert len(result.terms) == 2
    assert result.terms[0] is t


def test_interaction_term_feynman_rule_matches_lagrangian_wrapper():
    """InteractionTerm.feynman_rule() is a thin convenience wrapper."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    term = InteractionTerm(
        coupling=S("lam4"),
        fields=tuple(phi.occurrence() for _ in range(4)),
    )

    assert _canon(term.feynman_rule(phi, phi, phi, phi, simplify=True)) == _canon(
        Lagrangian(terms=(term,)).feynman_rule(phi, phi, phi, phi, simplify=True)
    )


# ---------------------------------------------------------------------------
# Scalar phi^4: self-conjugate bosonic vertex
# ---------------------------------------------------------------------------

def test_phi4_vertex():
    """phi^4 via Lagrangian.feynman_rule matches the direct API result."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam4 = S("lam4")

    term = InteractionTerm(
        coupling=lam4,
        fields=tuple(phi.occurrence() for _ in range(4)),
    )
    L = Lagrangian(terms=(term,))
    got = L.feynman_rule(phi, phi, phi, phi, simplify=True)

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 24 * I * lam4 * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_phi4_field_product():
    """Standalone Lagrangian(...) lowers a pure local field product directly."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam4 = S("lam4")

    got = Lagrangian(lam4 * phi * phi * phi * phi).feynman_rule(
        phi, phi, phi, phi, simplify=True
    )

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 24 * I * lam4 * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_partiald_phi4_product():
    """Standalone Lagrangian(...) lowers local PartialD monomials directly."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    gD2 = S("gD2")
    mu = S("mu")

    got = Lagrangian(
        gD2 * PartialD(phi, mu) * PartialD(phi, mu) * phi * phi
    ).feynman_rule(phi, phi, phi, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=gD2,
            fields=(phi.occurrence(), phi.occurrence(), phi.occurrence(), phi.occurrence()),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=mu),
                DerivativeAction(target=1, lorentz_index=mu),
            ),
        ),
    ))
    expected = ref.feynman_rule(phi, phi, phi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_partiald_accepts_labels_and_conjugated_keywords():
    """PartialD(..., labels=..., conjugated=...) matches the explicit FieldOccurrence form."""
    ghost = _make_ghost()
    mu = S("mu")
    a_bar = S("a_bar")
    a_ghost = S("a_ghost")

    keyword_form = Lagrangian(
        COLOR_ADJ_INDEX.representation.g(a_bar, a_ghost).to_expression()
        * PartialD(
            ghost,
            mu,
            conjugated=True,
            labels={COLOR_ADJ_KIND: a_bar},
        )
        * PartialD(
            ghost,
            mu,
            labels={COLOR_ADJ_KIND: a_ghost},
        )
    )
    occurrence_form = Lagrangian(
        COLOR_ADJ_INDEX.representation.g(a_bar, a_ghost).to_expression()
        * PartialD(
            ghost.occurrence(
                conjugated=True,
                labels={COLOR_ADJ_KIND: a_bar},
            ),
            mu,
        )
        * PartialD(
            ghost.occurrence(labels={COLOR_ADJ_KIND: a_ghost}),
            mu,
        )
    )

    assert _canon(keyword_form.feynman_rule(ghost.bar, ghost, simplify=True)) == _canon(
        occurrence_form.feynman_rule(ghost.bar, ghost, simplify=True)
    )


def test_lagrangian_accepts_declared_yukawa_product():
    """Two-fermion scalar bilinears infer the needed spinor labels automatically."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    y = S("y")

    got = Lagrangian(y * psi.bar * psi * phi).feynman_rule(
        psi.bar, psi, phi, simplify=True
    )

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=y,
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: S("alpha")}),
                psi.occurrence(labels={SPINOR_KIND: S("alpha")}),
                phi.occurrence(),
            ),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_colored_fermion_bilinear_uses_explicit_color_identity():
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )

    got = Lagrangian(quark.bar * quark).feynman_rule(
        quark.bar, quark, simplify=True
    )
    canon = _canon(got)
    spin_identity = _canon(SPINOR_INDEX.representation.g(S("i1"), S("i2")).to_expression())
    color_identity = _canon(COLOR_FUND_INDEX.representation.g(S("c1"), S("c2")).to_expression())

    assert spin_identity in canon
    assert color_identity in canon
    assert "delta(q,q)" not in canon
    assert "delta(qbar,qbar)" not in canon


def test_lagrangian_accepts_declared_vector_current():
    """A unique vector slot inherits the declared Lorentz label from Gamma(mu)."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    A = _make_photon()
    g = S("g")
    mu = S("mu")

    got = Lagrangian(g * psi.bar * Gamma(mu) * psi * A).feynman_rule(
        psi.bar, psi, A, simplify=True
    )

    expected = (
        I
        * g
        * gamma_matrix(S("i1"), S("i2"), S("mu3"))
        * (2 * pi) ** S("d")
        * Delta(S("q1") + S("q2") + S("q3"))
    )
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_axial_current():
    """Ordered spinor chains support Gamma(mu) * Gamma5() locally."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    A = _make_photon()
    g = S("g")
    mu = S("mu")
    i_bar = S("i_bar")
    i_psi = S("i_psi")
    alpha = S("spinor_decl_3")

    got = Lagrangian(
        g * psi.bar * Gamma(mu) * Gamma5() * psi * A
    ).feynman_rule(psi.bar, psi, A, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g * gamma_matrix(i_bar, alpha, mu) * gamma5_matrix(alpha, i_psi),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                A.occurrence(labels={LORENTZ_KIND: mu}),
            ),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, A, simplify=True)
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_current_current():
    """Multiple spinor chains in one monomial lower without InteractionTerm."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    g = S("g")
    mu = S("mu")
    a_bar, a_psi, b_bar, b_psi = S("a_bar", "a_psi", "b_bar", "b_psi")

    got = Lagrangian(
        g * psi.bar * Gamma(mu) * psi * psi.bar * Gamma(mu) * psi
    ).feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g * current_current(a_bar, a_psi, b_bar, b_psi, mu),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: a_bar}),
                psi.occurrence(labels={SPINOR_KIND: a_psi}),
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: b_bar}),
                psi.occurrence(labels={SPINOR_KIND: b_psi}),
            ),
            closed_dirac_bilinears=((0, 1), (2, 3)),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_declared_psibar_psi_sq_uses_closed_bilinear_signs():
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    g = S("g")
    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")

    got = Lagrangian(
        g * psi.bar * psi * psi.bar * psi
    ).feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)

    expected = (
        2
        * I
        * g
        * (2 * pi) ** d
        * Delta(q1 + q2 + q3 + q4)
        * (
            psi_bar_psi(S("i1"), S("i2")) * psi_bar_psi(S("i3"), S("i4"))
            + psi_bar_psi(S("i1"), S("i4")) * psi_bar_psi(S("i3"), S("i2"))
        )
    )
    assert _canon(got) == _canon(expected)


def test_declared_current_current_uses_closed_bilinear_signs():
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    g = S("g")
    mu = S("mu")
    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")

    got = simplify_gamma_chain(
        Lagrangian(
        g * psi.bar * Gamma(mu) * psi * psi.bar * Gamma(mu) * psi
        ).feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)
    )

    expected = (
        2
        * I
        * g
        * (2 * pi) ** d
        * Delta(q1 + q2 + q3 + q4)
        * (
            gamma_matrix(S("i1"), S("i2"), mu) * gamma_matrix(S("i3"), S("i4"), mu)
            + gamma_matrix(S("i1"), S("i4"), mu) * gamma_matrix(S("i3"), S("i2"), mu)
        )
    )
    assert _canon(got) == _canon(expected)


def test_distinct_species_closed_bilinear_product_is_stable_and_nonzero():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    g4 = S("g4")

    L = Lagrangian(g4 * psi.bar * psi * chi.bar * chi)
    assert len(L.terms) == 1
    assert L.terms[0].closed_dirac_bilinears == ((0, 1), (2, 3))

    got_1 = L.feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)
    got_2 = L.feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)

    assert _canon(got_1) == _canon(got_2)
    assert _canon(got_1) != _canon(Expression.num(0))


def test_distinct_species_closed_bilinear_order_is_bosonic():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    g4 = S("g4")

    got_1 = Lagrangian(
        g4 * psi.bar * psi * chi.bar * chi
    ).feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)
    got_2 = Lagrangian(
        g4 * chi.bar * chi * psi.bar * psi
    ).feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)

    assert _canon(got_1) == _canon(got_2)


def test_reversed_fields_inside_bilinear_are_rejected():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    g4 = S("g4")

    with pytest.raises(ValueError, match="Unsupported fermion ordering in local monomial"):
        Lagrangian(
            g4 * psi * psi.bar * chi.bar * chi
        )


def test_partially_recognized_multi_fermion_chain_is_rejected():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    g4 = S("g4")

    with pytest.raises(ValueError, match="Unsupported fermion ordering in local monomial"):
        Lagrangian(g4 * psi.bar * psi * chi * chi.bar)


def test_identical_closed_bilinear_square_is_deterministic_and_nonzero():
    psi = _make_dirac_fermion("Psi")
    g4 = S("g4")

    L = Lagrangian(g4 * psi.bar * psi * psi.bar * psi)
    assert len(L.terms) == 1
    assert L.terms[0].closed_dirac_bilinears == ((0, 1), (2, 3))

    got_1 = L.feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)
    got_2 = L.feynman_rule(psi.bar, psi, psi.bar, psi, simplify=True)

    assert _canon(got_1) == _canon(got_2)
    assert _canon(got_1) != _canon(Expression.num(0))


def test_distinct_species_vector_bilinear_order_is_stable():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    gV = S("gV")
    mu = S("mu")

    L = Lagrangian(gV * psi.bar * Gamma(mu) * psi * chi.bar * Gamma(mu) * chi)
    assert len(L.terms) == 1
    assert L.terms[0].closed_dirac_bilinears == ((0, 1), (2, 3))

    got_1 = simplify_gamma_chain(
        L.feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)
    )
    got_2 = simplify_gamma_chain(
        Lagrangian(
            gV * chi.bar * Gamma(mu) * chi * psi.bar * Gamma(mu) * psi
        ).feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True)
    )

    assert _canon(got_1) == _canon(got_2)
    assert _canon(got_1) != _canon(Expression.num(0))


def test_lagrangian_accepts_declared_local_quark_gluon_current():
    """A color-chain T(a) can sit in the same local monomial as Gamma(mu)."""
    g = S("g")
    a = S("a")
    mu = S("mu")
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    gluon = _make_gluon()
    i_bar, i_psi, c_bar, c_psi = S("i_bar", "i_psi", "c_bar", "c_psi")

    got = Lagrangian(
        g * quark.bar * Gamma(mu) * T(a) * quark * gluon
    ).feynman_rule(quark.bar, quark, gluon, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g * quark_gluon_current(i_bar, i_psi, mu, a, c_bar, c_psi),
            fields=(
                quark.occurrence(
                    conjugated=True,
                    labels={SPINOR_KIND: i_bar, COLOR_FUND_KIND: c_bar},
                ),
                quark.occurrence(
                    labels={SPINOR_KIND: i_psi, COLOR_FUND_KIND: c_psi},
                ),
                gluon.occurrence(labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a}),
            ),
        ),
    ))
    expected = ref.feynman_rule(quark.bar, quark, gluon, simplify=True)
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_local_scalar_contact():
    """Metric(mu, nu) binds a two-vector contact term declaratively."""
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phibar"),
    )
    A = _make_photon()
    g = S("g")
    mu = S("mu")
    nu = S("nu")

    got = Lagrangian(
        g * Metric(mu, nu) * phi.bar * phi * A * A
    ).feynman_rule(phi.bar, phi, A, A, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g * scalar_gauge_contact(mu, nu),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                A.occurrence(labels={LORENTZ_KIND: nu}),
            ),
        ),
    ))
    expected = ref.feynman_rule(phi.bar, phi, A, A, simplify=True)
    assert _canon(got) == _canon(expected)


def test_lagrangian_rejects_ambiguous_local_metric_attachment():
    """Metric(mu, rho) must not guess which vector fields carry its endpoints."""
    A = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    B = Field(
        "B",
        spin=1,
        self_conjugate=True,
        symbol=S("B"),
        indices=(LORENTZ_INDEX,),
    )
    C = Field(
        "C",
        spin=1,
        self_conjugate=True,
        symbol=S("C"),
        indices=(LORENTZ_INDEX,),
    )
    mu = S("mu")
    rho = S("rho")

    with pytest.raises(ValueError, match="Ambiguous local tensor attachment"):
        Lagrangian(A * B * C * Metric(mu, rho))


def test_lagrangian_accepts_declared_two_fermion_partiald_operator():
    """Two-fermion local derivative operators lower without explicit InteractionTerm."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    y = S("y")
    mu = S("mu")

    got = Lagrangian(
        y * PartialD(psi.bar, mu) * psi * phi * chi
    ).feynman_rule(psi.bar, psi, phi, chi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=y,
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: S("alpha")}),
                psi.occurrence(labels={SPINOR_KIND: S("alpha")}),
                phi.occurrence(),
                chi.occurrence(),
            ),
            derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, phi, chi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_declared_complex_scalar_current_pair():
    """A unique vector slot also inherits the shared derivative Lorentz index."""
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phibar"),
    )
    A = _make_photon()
    g = S("g")
    mu = S("mu")

    got = Lagrangian(
        g * phi.bar * PartialD(phi, mu) * A,
        -g * PartialD(phi.bar, mu) * phi * A,
    ).feynman_rule(phi.bar, phi, A, simplify=True)

    expected = (
        g
        * (pcomp(S("q2"), mu) - pcomp(S("q1"), mu))
        * (2 * pi) ** S("d")
        * Delta(S("q1") + S("q2") + S("q3"))
    )
    assert _canon(got) == _canon(expected)


def test_lagrangian_accepts_lagrangian_decl_keyword():
    """The standalone API accepts a Model-style lagrangian_decl= alias."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam4 = S("lam4")

    got = Lagrangian(
        lagrangian_decl=lam4 * phi * phi * phi * phi
    ).feynman_rule(phi, phi, phi, phi, simplify=True)

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 24 * I * lam4 * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


def test_compiled_lagrangian_rejects_source_declarations():
    """Compiled extraction objects reject source declarations directly."""
    qPsi = S("qPsi")
    mu = S("mu")
    fermion = Field(
        "PsiQED",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": qPsi},
    )

    with pytest.raises(ValueError, match="Use Model\\(lagrangian_decl="):
        CompiledLagrangian(I * fermion.bar * Gamma(mu) * CovD(fermion, mu))


def test_lagrangian_rejects_covariant_decl_without_model():
    """Model-dependent declarations still require Model metadata."""
    qPsi = S("qPsi")
    mu = S("mu")
    fermion = Field(
        "PsiQED",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": qPsi},
    )

    with pytest.raises(ValueError, match="Use Model\\(lagrangian_decl="):
        Lagrangian(I * fermion.bar * Gamma(mu) * CovD(fermion, mu))


def test_compiled_lagrangian_add_rejects_source_declarations():
    """Compiled extraction objects reject source declarations in composition too."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))

    with pytest.raises(ValueError, match="Use Model\\(lagrangian_decl="):
        CompiledLagrangian() + (S("lam4") * phi * phi * phi * phi)


def test_model_accepts_declared_phi4_field_product():
    """Model.lagrangian_decl accepts a pure field-product local operator."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam4 = S("lam4")

    model = Model(fields=(phi,), lagrangian_decl=lam4 * phi * phi * phi * phi)
    got = model.lagrangian().feynman_rule(phi, phi, phi, phi, simplify=True)

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 24 * I * lam4 * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_psibar_psi_sq_phi4_field_product():
    """Canonical psibar psi chains in a field product get inferred spinor labels."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    g = S("g")
    alpha = S("alpha")
    beta = S("beta")

    explicit = InteractionTerm(
        coupling=g,
        fields=(
            psi.occurrence(conjugated=True, labels={SPINOR_KIND: alpha}),
            psi.occurrence(labels={SPINOR_KIND: alpha}),
            psi.occurrence(conjugated=True, labels={SPINOR_KIND: beta}),
            psi.occurrence(labels={SPINOR_KIND: beta}),
            phi.occurrence(),
            phi.occurrence(),
            phi.occurrence(),
            phi.occurrence(),
        ),
    )

    declared_model = Model(
        fields=(psi, phi),
        lagrangian_decl=g * psi.bar * psi * psi.bar * psi * phi * phi * phi * phi,
    )
    explicit_model = Model(fields=(psi, phi), lagrangian_decl=explicit)
    field_args = (psi.bar, psi, psi.bar, psi, phi, phi, phi, phi)

    got = declared_model.lagrangian().feynman_rule(*field_args, simplify=True)
    expected = explicit_model.lagrangian().feynman_rule(*field_args, simplify=True)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_local_phi4_product():
    """Pure local field monomials lower directly from lagrangian_decl."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam_phi4 = S("lam_phi4")

    model = Model(fields=(phi,), lagrangian_decl=lam_phi4 * phi * phi * phi * phi)
    got = model.lagrangian().feynman_rule(phi, phi, phi, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=lam_phi4,
            fields=(phi.occurrence(), phi.occurrence(), phi.occurrence(), phi.occurrence()),
        ),
    ))
    expected = ref.feynman_rule(phi, phi, phi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_local_yukawa_product():
    """Local Yukawa monomials lower directly from lagrangian_decl."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    y = S("y")

    model = Model(fields=(psi, phi), lagrangian_decl=y * psi.bar * psi * phi)
    got = model.lagrangian().feynman_rule(psi.bar, psi, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=y,
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: S("alpha_decl_1")}),
                psi.occurrence(labels={SPINOR_KIND: S("alpha_decl_1")}),
                phi.occurrence(),
            ),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_partiald_phi4_product():
    """PartialD(...) lowers local scalar derivative monomials without InteractionTerm."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    gD2 = S("gD2")
    mu = S("mu")

    model = Model(
        fields=(phi,),
        lagrangian_decl=gD2 * PartialD(phi, mu) * PartialD(phi, mu) * phi * phi,
    )
    got = model.lagrangian().feynman_rule(phi, phi, phi, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=gD2,
            fields=(phi.occurrence(), phi.occurrence(), phi.occurrence(), phi.occurrence()),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=mu),
                DerivativeAction(target=1, lorentz_index=mu),
            ),
        ),
    ))
    expected = ref.feynman_rule(phi, phi, phi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_nested_partiald():
    """Nested PartialD(...) maps to repeated DerivativeAction entries on one slot."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    g2 = S("g2")
    mu = S("mu")
    nu = S("nu")

    model = Model(fields=(phi,), lagrangian_decl=g2 * PartialD(PartialD(phi, mu), nu) * phi)
    got = model.lagrangian().feynman_rule(phi, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g2,
            fields=(phi.occurrence(), phi.occurrence()),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=mu),
                DerivativeAction(target=0, lorentz_index=nu),
            ),
        ),
    ))
    expected = ref.feynman_rule(phi, phi, simplify=True)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_gamma_partiald_fermion_operator():
    """Gamma(...) * PartialD(...) lowers to a local fermion derivative operator."""
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    g = S("g")
    mu = S("mu")
    nu = S("nu")
    i_bar = S("i_bar")
    i_psi = S("i_psi")

    model = Model(
        fields=(psi, phi, chi),
        lagrangian_decl=g * psi.bar * Gamma(mu) * PartialD(psi, nu) * phi * chi,
    )
    got = model.lagrangian().feynman_rule(psi.bar, psi, phi, chi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=g * psi_bar_gamma_psi(i_bar, i_psi, mu),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                phi.occurrence(),
                chi.occurrence(),
            ),
            derivatives=(DerivativeAction(target=1, lorentz_index=nu),),
        ),
    ))
    expected = ref.feynman_rule(psi.bar, psi, phi, chi, simplify=True)
    assert _canon(got) == _canon(expected)


# ---------------------------------------------------------------------------
# Complex scalar phi^dag phi mass term
# ---------------------------------------------------------------------------

def test_complex_scalar_mass_term():
    """lamC * phiC^dag phiC via Lagrangian.feynman_rule."""
    phiC = Field("PhiC", spin=0, self_conjugate=False, symbol=S("phiC"), conjugate_symbol=S("phiCdag"))
    lamC = S("lamC")

    term = InteractionTerm(
        coupling=lamC,
        fields=(phiC.occurrence(conjugated=True), phiC.occurrence()),
    )
    L = Lagrangian(terms=(term,))
    got = L.feynman_rule(phiC.bar, phiC, simplify=True)

    q1, q2 = S("q1", "q2")
    d = S("d")
    expected = I * lamC * (2 * pi) ** d * Delta(q1 + q2)
    assert _canon(got) == _canon(expected)


def test_model_accepts_declared_complex_scalar_mass_product():
    """Conjugated field products lower to InteractionTerm correctly."""
    phiC = Field("PhiC", spin=0, self_conjugate=False, symbol=S("phiC"), conjugate_symbol=S("phiCdag"))
    lamC = S("lamC")

    model = Model(fields=(phiC,), lagrangian_decl=lamC * phiC.bar * phiC)
    got = model.lagrangian().feynman_rule(phiC.bar, phiC, simplify=True)

    q1, q2 = S("q1", "q2")
    d = S("d")
    expected = I * lamC * (2 * pi) ** d * Delta(q1 + q2)
    assert _canon(got) == _canon(expected)


def test_covd_accepts_conjugated_keyword():
    """CovD(..., conjugated=True) matches the existing Field.bar form."""
    eQED, qPhi = S("eQED", "qPhi")
    mu = S("mu")
    phi = Field(
        "PhiQED",
        spin=0,
        self_conjugate=False,
        symbol=S("phiQED"),
        conjugate_symbol=S("phiQEDbar"),
        quantum_numbers={"Q": qPhi},
    )
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    keyword_model = Model(
        gauge_groups=(u1,),
        fields=(phi, photon),
        lagrangian_decl=CovD(phi, mu, conjugated=True) * CovD(phi, mu),
    )
    bar_model = Model(
        gauge_groups=(u1,),
        fields=(phi, photon),
        lagrangian_decl=CovD(phi.bar, mu) * CovD(phi, mu),
    )

    assert _canon(
        keyword_model.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    ) == _canon(
        bar_model.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    )


def test_complex_scalar_mass_term_tuple_input():
    """(Field, bool) input works the same as Field.bar."""
    phiC = Field("PhiC", spin=0, self_conjugate=False, symbol=S("phiC"), conjugate_symbol=S("phiCdag"))
    lamC = S("lamC")

    term = InteractionTerm(
        coupling=lamC,
        fields=(phiC.occurrence(conjugated=True), phiC.occurrence()),
    )
    L = Lagrangian(terms=(term,))
    got = L.feynman_rule((phiC, True), phiC, simplify=True)

    q1, q2 = S("q1", "q2")
    d = S("d")
    expected = I * lamC * (2 * pi) ** d * Delta(q1 + q2)
    assert _canon(got) == _canon(expected)


# ---------------------------------------------------------------------------
# phi^2 chi^2: mixed species
# ---------------------------------------------------------------------------

def test_phi2_chi2_vertex():
    """g * phi^2 chi^2 via Lagrangian.feynman_rule."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    g = S("g")

    term = InteractionTerm(
        coupling=g,
        fields=(phi.occurrence(), phi.occurrence(), chi.occurrence(), chi.occurrence()),
    )
    L = Lagrangian(terms=(term,))
    got = L.feynman_rule(phi, phi, chi, chi, simplify=True)

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 4 * I * g * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


# ---------------------------------------------------------------------------
# Custom momenta override
# ---------------------------------------------------------------------------

def test_custom_momenta():
    """feynman_rule with explicit momenta overrides q1, q2, ..."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam = S("lam")

    term = InteractionTerm(
        coupling=lam,
        fields=tuple(phi.occurrence() for _ in range(4)),
    )
    L = Lagrangian(terms=(term,))
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    got = L.feynman_rule(phi, phi, phi, phi, momenta=[p1, p2, p3, p4], simplify=True)

    d = S("d")
    expected = 24 * I * lam * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert _canon(got) == _canon(expected)


# ---------------------------------------------------------------------------
# No matching term -> ValueError
# ---------------------------------------------------------------------------

def test_no_match_raises():
    """feynman_rule raises ValueError when no interaction matches."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))

    term = InteractionTerm(
        coupling=S("g"),
        fields=tuple(phi.occurrence() for _ in range(4)),
    )
    L = Lagrangian(terms=(term,))
    with pytest.raises(ValueError, match="No matching"):
        L.feynman_rule(phi, phi, chi, chi)


def test_same_symbol_different_kind_does_not_match():
    """Distinct field kinds sharing one symbol must not silently match."""
    scalar = Field("S", spin=0, self_conjugate=True, symbol=S("X"))
    vector = Field("V", spin=1, self_conjugate=True, symbol=S("X"))
    term = InteractionTerm(coupling=S("g"), fields=(scalar.occurrence(),))
    L = Lagrangian(terms=(term,))

    with pytest.raises(ValueError, match="No matching"):
        L.feynman_rule(vector)


def test_same_symbol_different_fields_do_not_match():
    """Distinct declared fields sharing a symbol must not silently match."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("X"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("X"))
    term = InteractionTerm(coupling=S("g"), fields=(phi.occurrence(),))
    L = Lagrangian(terms=(term,))

    with pytest.raises(ValueError, match="No matching"):
        L.feynman_rule(chi)


# ---------------------------------------------------------------------------
# Lagrangian sums multiple matching terms
# ---------------------------------------------------------------------------

def test_multiple_matching_terms_are_summed():
    """Two phi^4 terms in a Lagrangian are summed by feynman_rule."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    g1 = S("g1")
    g2 = S("g2")

    t1 = InteractionTerm(coupling=g1, fields=tuple(phi.occurrence() for _ in range(4)))
    t2 = InteractionTerm(coupling=g2, fields=tuple(phi.occurrence() for _ in range(4)))
    L = Lagrangian(terms=(t1, t2))
    got = L.feynman_rule(phi, phi, phi, phi, simplify=True)

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    d = S("d")
    expected = 24 * I * (g1 + g2) * (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)
    assert _canon(got) == _canon(expected)


def test_feynman_rule_no_args_returns_all_vertices():
    """A zero-argument call returns every available vertex keyed by names."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    lam4 = S("lam4")
    g = S("g")

    L = Lagrangian(terms=(
        InteractionTerm(coupling=lam4, fields=tuple(phi.occurrence() for _ in range(4))),
        InteractionTerm(
            coupling=g,
            fields=(phi.occurrence(), phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
    ))

    rules = L.feynman_rule(simplify=True)

    phi4_key = ("Phi", "Phi", "Phi", "Phi")
    mixed_key = ("Phi", "Phi", "Chi", "Chi")
    assert set(rules) == {phi4_key, mixed_key}
    assert _canon(rules[phi4_key]) == _canon(
        L.feynman_rule(phi, phi, phi, phi, simplify=True)
    )
    assert _canon(rules[mixed_key]) == _canon(
        L.feynman_rule(phi, phi, chi, chi, simplify=True)
    )


def test_feynman_rule_no_args_can_return_field_keys():
    """The previous object-keyed zero-argument mapping is still available."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lam4 = S("lam4")
    L = Lagrangian(lam4 * phi * phi * phi * phi)

    rules = L.feynman_rule(simplify=True, key_format="fields")
    key = (phi, phi, phi, phi)

    assert tuple(rules) == (key,)
    assert _canon(rules[key]) == _canon(
        L.feynman_rule(phi, phi, phi, phi, simplify=True)
    )


def test_feynman_rules_returns_all_vertices():
    """feynman_rules() exposes the same grouped mapping as zero-arg feynman_rule()."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    L = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
        InteractionTerm(
            coupling=S("lam"),
            fields=tuple(phi.occurrence() for _ in range(4)),
        ),
    ))

    assert {
        key: _canon(value) for key, value in L.feynman_rules(simplify=True).items()
    } == {
        key: _canon(value) for key, value in L.feynman_rule(simplify=True).items()
    }


def test_feynman_rules_filters_by_arity():
    """arity=... keeps only the requested vertex size."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    L = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
        InteractionTerm(
            coupling=S("lam"),
            fields=tuple(phi.occurrence() for _ in range(4)),
        ),
    ))

    rules = L.feynman_rules(arity=3, simplify=True, key_format="fields")

    assert tuple(rules) == ((phi, chi, chi),)
    assert _canon(rules[(phi, chi, chi)]) == _canon(
        L.feynman_rule(phi, chi, chi, simplify=True)
    )


def test_feynman_rules_select_uses_requested_field_tuples():
    """select=[...] reuses the single-vertex path for exactly the requested tuples."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    L = Lagrangian(terms=(
        InteractionTerm(coupling=S("g"), fields=(phi.occurrence(), chi.occurrence(), chi.occurrence())),
    ))

    rules = L.feynman_rules(
        select=[(chi, phi, chi)],
        simplify=True,
        key_format="fields",
    )

    assert tuple(rules) == ((chi, phi, chi),)
    assert _canon(rules[(chi, phi, chi)]) == _canon(
        L.feynman_rule(chi, phi, chi, simplify=True)
    )


def test_feynman_rule_no_args_merges_repeated_signature():
    """Repeated terms with the same field content are grouped and summed."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    g1 = S("g1")
    g2 = S("g2")

    t1 = InteractionTerm(coupling=g1, fields=tuple(phi.occurrence() for _ in range(4)))
    t2 = InteractionTerm(coupling=g2, fields=tuple(phi.occurrence() for _ in range(4)))
    L = Lagrangian(terms=(t1, t2))

    rules = L.feynman_rule(simplify=True)
    raw_rules = L.feynman_rule(simplify=False)
    key = ("Phi", "Phi", "Phi", "Phi")

    assert tuple(rules) == (key,)
    assert _canon(rules[key]) == _canon(
        L.feynman_rule(phi, phi, phi, phi, simplify=True)
    )
    assert _canon(raw_rules[key]) == _canon(
        L.feynman_rule(phi, phi, phi, phi, simplify=False)
    )


def test_feynman_rule_no_args_normalizes_conjugated_field_keys():
    """Non-self-conjugate fields use .bar suffixes in default name keys."""
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phibar"),
    )
    lam = S("lam")
    L = Lagrangian(lam * phi.bar * phi)

    rules = L.feynman_rule(simplify=True)
    key = next(iter(rules))

    assert len(rules) == 1
    assert key == ("PhiC.bar", "PhiC")
    assert _canon(rules[key]) == _canon(L.feynman_rule(phi.bar, phi, simplify=True))


def test_feynman_rule_no_args_field_keys_preserve_conjugated_objects():
    """key_format='fields' keeps Field.bar objects for programmatic lookup."""
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phibar"),
    )
    lam = S("lam")
    L = Lagrangian(lam * phi.bar * phi)

    rules = L.feynman_rule(simplify=True, key_format="fields")
    key = next(iter(rules))

    assert len(rules) == 1
    assert isinstance(key[0], ConjugateField)
    assert key[0].field is phi
    assert key[1] is phi
    assert _canon(rules[key]) == _canon(L.feynman_rule(phi.bar, phi, simplify=True))


def test_feynman_rule_no_args_rejects_momenta_override():
    """A shared momenta= override is ambiguous for mixed-arity enumeration."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    L = Lagrangian(S("g") * phi * phi)

    with pytest.raises(ValueError, match="momenta"):
        L.feynman_rule(momenta=[S("p")])


def test_feynman_rule_rejects_unknown_key_format():
    """Typos in key_format fail loudly."""
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    L = Lagrangian(S("g") * phi * phi)

    with pytest.raises(ValueError, match="key_format"):
        L.feynman_rule(key_format="objects")


def test_feynman_rule_no_args_name_keys_reject_ambiguous_names():
    """Default name keys do not silently merge different same-name fields."""
    phi1 = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi1"))
    phi2 = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi2"))
    L = Lagrangian(terms=(
        InteractionTerm(coupling=S("g1"), fields=(phi1.occurrence(), phi1.occurrence())),
        InteractionTerm(coupling=S("g2"), fields=(phi2.occurrence(), phi2.occurrence())),
    ))

    with pytest.raises(ValueError, match="key_format='fields'"):
        L.feynman_rule()

    rules = L.feynman_rule(key_format="fields")
    assert (phi1, phi1) in rules
    assert (phi2, phi2) in rules


def test_feynman_rule_no_args_nontrivial_model_summary_is_usable():
    """A larger compiled model returns a usable mapping without API changes."""
    rules = MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian().feynman_rule(simplify=False)

    assert ("G", "G") in rules
    assert ("G", "G", "G") in rules
    assert ("G", "G", "G", "G") in rules
    assert ("ghG.bar", "ghG") in rules
    assert ("ghG.bar", "G", "ghG") in rules
    assert _canon(rules[("ghG.bar", "G", "ghG")]) == _canon(
        MODEL_QCD_ORDINARY_GAUGE_FIXED.lagrangian().feynman_rule(
            GhostGluonField.bar,
            GluonField,
            GhostGluonField,
            simplify=False,
        )
    )


# ---------------------------------------------------------------------------
# Model.lagrangian() integration: QED Dirac kinetic term
# ---------------------------------------------------------------------------

def test_model_lagrangian_qed_fermion():
    """Model.lagrangian().feynman_rule for a QED fermion-photon vertex."""
    eQED, qPsi = S("eQED", "qPsi")
    fermion = Field("PsiQED", spin=Fraction(1, 2), self_conjugate=False,
                     symbol=S("psi"), conjugate_symbol=S("psibar"),
                     indices=(SPINOR_INDEX,), quantum_numbers={"Q": qPsi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)
    model = Model(gauge_groups=(u1,), fields=(fermion, photon),
                  lagrangian_decl=_dirac_decl(fermion))

    L = model.lagrangian()
    got = L.feynman_rule(fermion.bar, fermion, photon, simplify=True)

    q1, q2, q3 = S("q1", "q2", "q3")
    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2
    gauge_term = next(term for term in compiled if len(term.fields) == 3)
    legs = (
        fermion.leg(q1, conjugated=True, labels={SPINOR_KIND: S("i1")}),
        fermion.leg(q2, labels={SPINOR_KIND: S("i2")}),
        photon.leg(q3, labels={LORENTZ_KIND: S("mu3")}),
    )
    ref = _ref_vertex(gauge_term, legs)
    assert _canon(got) == _canon(ref)


def test_declared_lagrangian_qed_fermion():
    """A declarative CovD-based Dirac term preserves the legacy QED gauge vertex."""
    eQED, qPsi = S("eQED", "qPsi")
    mu = S("mu")
    fermion = Field("PsiQED", spin=Fraction(1, 2), self_conjugate=False,
                     symbol=S("psi"), conjugate_symbol=S("psibar"),
                     indices=(SPINOR_INDEX,), quantum_numbers={"Q": qPsi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    decl = I * fermion.bar * Gamma(mu) * CovD(fermion, mu)
    model = Model(gauge_groups=(u1,), fields=(fermion, photon), lagrangian_decl=decl)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = Model(gauge_groups=(u1,), fields=(fermion, photon),
                       covariant_terms=(DiracKineticTerm(field=fermion),))

    assert isinstance(model.lagrangian_decl, DeclaredLagrangian)
    assert len(model.lagrangian_decl.source_terms) == 1
    assert "CovD(" in str(model.lagrangian_decl.source_terms[0])
    assert len(model.all_covariant_terms()) == 1
    assert isinstance(model.all_covariant_terms()[0], DiracKineticTerm)

    got = model.lagrangian().feynman_rule(fermion.bar, fermion, photon, simplify=True)
    ref = legacy.lagrangian().feynman_rule(fermion.bar, fermion, photon, simplify=True)
    assert _canon(got) == _canon(ref)


def test_declared_lagrangian_qed_fermion_includes_free_bilinear():
    eQED, qPsi = S("eQED", "qPsi")
    mu = S("mu")
    fermion = Field("PsiQED", spin=Fraction(1, 2), self_conjugate=False,
                     symbol=S("psi"), conjugate_symbol=S("psibar"),
                     indices=(SPINOR_INDEX,), quantum_numbers={"Q": qPsi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)
    model = Model(
        gauge_groups=(u1,),
        fields=(fermion, photon),
        lagrangian_decl=I * fermion.bar * Gamma(mu) * CovD(fermion, mu),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2
    assert any(len(term.fields) == 2 for term in compiled)

    L = model.lagrangian()
    got = L.feynman_rule(fermion.bar, fermion, simplify=True)

    i_bar, i_psi = S("i_bar"), S("i_psi")
    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=I * psi_bar_gamma_psi(i_bar, i_psi, mu),
            fields=(
                fermion.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                fermion.occurrence(labels={SPINOR_KIND: i_psi}),
            ),
            derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        ),
    ))

    assert _canon(got) == _canon(ref.feynman_rule(fermion.bar, fermion, simplify=True))


def test_legacy_qed_fermion_kinetic_term_is_gauge_only():
    """Legacy DiracKineticTerm keeps only gauge-generated interactions."""
    eQED, qPsi = S("eQED", "qPsi")
    fermion = Field("PsiQED", spin=Fraction(1, 2), self_conjugate=False,
                     symbol=S("psi"), conjugate_symbol=S("psibar"),
                     indices=(SPINOR_INDEX,), quantum_numbers={"Q": qPsi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = Model(
            gauge_groups=(u1,),
            fields=(fermion, photon),
            covariant_terms=(DiracKineticTerm(field=fermion),),
        )

    compiled = compile_covariant_terms(legacy)
    assert compiled
    assert all(len(term.fields) != 2 for term in compiled)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        legacy.lagrangian().feynman_rule(fermion.bar, fermion, simplify=True)


# ---------------------------------------------------------------------------
# Model.lagrangian() integration: QCD quark-gluon vertex
# ---------------------------------------------------------------------------

def test_model_lagrangian_qcd_fermion():
    """Model.lagrangian().feynman_rule for a QCD quark-gluon vertex."""
    gS = S("gS")
    quark = Field("q", spin=Fraction(1, 2), self_conjugate=False,
                   symbol=S("psi"), conjugate_symbol=S("psibar"),
                   indices=(SPINOR_INDEX, COLOR_FUND_INDEX))
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)
    model = Model(gauge_groups=(su3,), fields=(quark, gluon),
                  lagrangian_decl=_dirac_decl(quark))

    L = model.lagrangian()
    got = L.feynman_rule(quark.bar, quark, gluon, simplify=True)

    q1, q2, q3 = S("q1", "q2", "q3")
    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2
    gauge_term = next(term for term in compiled if len(term.fields) == 3)
    legs = (
        quark.leg(q1, conjugated=True, labels={SPINOR_KIND: S("i1"), COLOR_FUND_KIND: S("c1")}),
        quark.leg(q2, labels={SPINOR_KIND: S("i2"), COLOR_FUND_KIND: S("c2")}),
        gluon.leg(q3, labels={LORENTZ_KIND: S("mu3"), COLOR_ADJ_KIND: S("a3")}),
    )
    ref = _ref_vertex(gauge_term, legs)
    assert _canon(got) == _canon(ref)


def test_declared_lagrangian_scalar_qed_matches_legacy():
    """A declarative scalar CovD term preserves the legacy QED gauge vertices."""
    eQED, qPhi = S("eQED", "qPhi")
    mu = S("mu")
    phi = Field("PhiQED", spin=0, self_conjugate=False,
                symbol=S("phiQED"), conjugate_symbol=S("phiQEDbar"),
                quantum_numbers={"Q": qPhi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    decl = CovD(phi.bar, mu) * CovD(phi, mu)
    model = Model(gauge_groups=(u1,), fields=(phi, photon), lagrangian_decl=decl)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = Model(gauge_groups=(u1,), fields=(phi, photon),
                       covariant_terms=(ComplexScalarKineticTerm(field=phi),))

    got_3pt = model.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    ref_3pt = legacy.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    assert _canon(got_3pt) == _canon(ref_3pt)

    got_4pt = model.lagrangian().feynman_rule(phi.bar, phi, photon, photon, simplify=True)
    ref_4pt = legacy.lagrangian().feynman_rule(phi.bar, phi, photon, photon, simplify=True)
    assert _canon(got_4pt) == _canon(ref_4pt)


def test_declared_lagrangian_scalar_qed_includes_free_bilinear():
    eQED, qPhi = S("eQED", "qPhi")
    mu = S("mu")
    phi = Field("PhiQED", spin=0, self_conjugate=False,
                symbol=S("phiQED"), conjugate_symbol=S("phiQEDbar"),
                quantum_numbers={"Q": qPhi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)
    model = Model(
        gauge_groups=(u1,),
        fields=(phi, photon),
        lagrangian_decl=CovD(phi.bar, mu) * CovD(phi, mu),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 4
    assert any(len(term.fields) == 2 for term in compiled)

    L = model.lagrangian()
    got = L.feynman_rule(phi.bar, phi, simplify=True)

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
            ),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=mu),
                DerivativeAction(target=1, lorentz_index=mu),
            ),
        ),
    ))

    assert _canon(got) == _canon(ref.feynman_rule(phi.bar, phi, simplify=True))


def test_legacy_scalar_qed_kinetic_term_is_gauge_only():
    """Legacy ComplexScalarKineticTerm keeps only gauge-generated interactions."""
    eQED, qPhi = S("eQED", "qPhi")
    phi = Field("PhiQED", spin=0, self_conjugate=False,
                symbol=S("phiQED"), conjugate_symbol=S("phiQEDbar"),
                quantum_numbers={"Q": qPhi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = Model(
            gauge_groups=(u1,),
            fields=(phi, photon),
            covariant_terms=(ComplexScalarKineticTerm(field=phi),),
        )

    compiled = compile_covariant_terms(legacy)
    assert compiled
    assert all(len(term.fields) != 2 for term in compiled)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        legacy.lagrangian().feynman_rule(phi.bar, phi, simplify=True)


def test_declared_lagrangian_field_strength_matches_legacy():
    """FieldStrength declarations lower to the legacy gauge-kinetic term."""
    gS = S("gS")
    mu, nu = S("mu", "nu")
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    decl = -(Expression.num(1) / Expression.num(4)) * FieldStrength(su3, mu, nu) * FieldStrength(su3, mu, nu)
    model = Model(gauge_groups=(su3,), fields=(gluon,), lagrangian_decl=decl)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = Model(gauge_groups=(su3,), fields=(gluon,),
                       gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=su3),))

    got_3pt = model.lagrangian().feynman_rule(gluon, gluon, gluon, simplify=True)
    ref_3pt = legacy.lagrangian().feynman_rule(gluon, gluon, gluon, simplify=True)
    assert _canon(got_3pt) == _canon(ref_3pt)

    got_4pt = model.lagrangian().feynman_rule(gluon, gluon, gluon, gluon, simplify=True)
    ref_4pt = legacy.lagrangian().feynman_rule(gluon, gluon, gluon, gluon, simplify=True)
    assert _canon(got_4pt) == _canon(ref_4pt)


def test_declared_lagrangian_rejects_scalar_covd_without_conjugate_pair():
    """Scalar declarative form requires one barred and one unbarred CovD factor."""
    eQED, qPhi = S("eQED", "qPhi")
    mu = S("mu")
    phi = Field(
        "PhiQED",
        spin=0,
        self_conjugate=False,
        symbol=S("phiQED"),
        conjugate_symbol=S("phiQEDbar"),
        quantum_numbers={"Q": qPhi},
    )
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    with pytest.raises(ValueError, match="Unsupported declarative Lagrangian term"):
        Model(
            gauge_groups=(u1,),
            fields=(phi, photon),
            lagrangian_decl=CovD(phi, mu) * CovD(phi, mu),
        )


def test_declared_lagrangian_rejects_gamma_covd_index_mismatch():
    """Dirac declarative form requires Gamma and CovD to share Lorentz index."""
    eQED, qPsi = S("eQED", "qPsi")
    mu, nu = S("mu", "nu")
    fermion = Field(
        "PsiQED",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": qPsi},
    )
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    with pytest.raises(ValueError, match="Unsupported declarative Lagrangian term"):
        Model(
            gauge_groups=(u1,),
            fields=(fermion, photon),
            lagrangian_decl=I * fermion.bar * Gamma(mu) * CovD(fermion, nu),
        )


def test_declared_lagrangian_rejects_field_strength_index_mismatch():
    """Field-strength declarative form requires matching (mu,nu) pair."""
    gS = S("gS")
    mu, nu = S("mu", "nu")
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    with pytest.raises(ValueError, match="Unsupported declarative Lagrangian term"):
        Model(
            gauge_groups=(su3,),
            fields=(gluon,),
            lagrangian_decl=-(Expression.num(1) / Expression.num(4))
            * FieldStrength(su3, mu, nu)
            * FieldStrength(su3, nu, mu),
        )


def test_declared_lagrangian_rejects_field_strength_repeated_lorentz_index():
    """FieldStrength(G, mu, mu) must be rejected before gauge compilation."""
    e = S("e")
    mu = S("mu")
    photon = _make_photon()
    u1 = _make_u1(e, photon.symbol)

    with pytest.raises(ValueError, match="FieldStrength indices must be distinct"):
        Model(
            gauge_groups=(u1,),
            fields=(photon,),
            lagrangian_decl=-(Expression.num(1) / Expression.num(4))
            * FieldStrength(u1, mu, mu)
            * FieldStrength(u1, mu, mu),
        )


# ===========================================================================
# Precompiled-model idempotency tests
# ===========================================================================

def test_precompiled_model_lagrangian_no_double_count():
    """with_compiled_covariant_terms(model).lagrangian() must not double-count."""
    gS = S("gS")
    quark = Field("q", spin=Fraction(1, 2), self_conjugate=False,
                   symbol=S("psi"), conjugate_symbol=S("psibar"),
                   indices=(SPINOR_INDEX, COLOR_FUND_INDEX))
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)
    model = Model(gauge_groups=(su3,), fields=(quark, gluon),
                  lagrangian_decl=_dirac_decl(quark))

    L_fresh = model.lagrangian()
    precompiled = with_compiled_covariant_terms(model)
    L_pre = precompiled.lagrangian()

    assert len(L_fresh.terms) == len(L_pre.terms), (
        f"Term count mismatch: fresh={len(L_fresh.terms)}, precompiled={len(L_pre.terms)}"
    )

    got_fresh = L_fresh.feynman_rule(quark.bar, quark, gluon, simplify=True)
    got_pre = L_pre.feynman_rule(quark.bar, quark, gluon, simplify=True)
    assert _canon(got_fresh) == _canon(got_pre)


def test_precompiled_clears_declaration_slots():
    """with_compiled_covariant_terms clears declaration slots."""
    gS = S("gS")
    quark = Field("q", spin=Fraction(1, 2), self_conjugate=False,
                   symbol=S("psi"), conjugate_symbol=S("psibar"),
                   indices=(SPINOR_INDEX, COLOR_FUND_INDEX))
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)
    model = Model(gauge_groups=(su3,), fields=(quark, gluon),
                  lagrangian_decl=_dirac_decl(quark))

    precompiled = with_compiled_covariant_terms(model)
    assert precompiled.covariant_terms == ()
    assert precompiled.gauge_kinetic_terms == ()
    assert precompiled.gauge_fixing_terms == ()
    assert precompiled.ghost_terms == ()
    assert len(precompiled.interactions) == 2


def test_precompiled_keeps_manual_declared_interactions():
    """Precompilation clears only physical declarative terms, not manual ones."""
    eQED, qPhi, lam4 = S("eQED", "qPhi", "lam4")
    mu = S("mu")
    phi = Field("PhiQED", spin=0, self_conjugate=False,
                symbol=S("phiQED"), conjugate_symbol=S("phiQEDbar"),
                quantum_numbers={"Q": qPhi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)
    quartic = InteractionTerm(
        coupling=lam4,
        fields=tuple(phi.occurrence() for _ in range(4)),
    )
    decl = quartic + (CovD(phi.bar, mu) * CovD(phi, mu))

    model = Model(gauge_groups=(u1,), fields=(phi, photon), lagrangian_decl=decl)
    precompiled = with_compiled_covariant_terms(model)

    assert precompiled.covariant_terms == ()
    assert precompiled.gauge_kinetic_terms == ()
    assert precompiled.gauge_fixing_terms == ()
    assert precompiled.ghost_terms == ()
    assert len(precompiled.lagrangian_decl.source_terms) == 1
    assert isinstance(precompiled.lagrangian_decl.source_terms[0], InteractionTerm)

    fresh_phi4 = model.lagrangian().feynman_rule(phi, phi, phi, phi, simplify=True)
    pre_phi4 = precompiled.lagrangian().feynman_rule(phi, phi, phi, phi, simplify=True)
    assert _canon(fresh_phi4) == _canon(pre_phi4)

    fresh_gauge = model.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    pre_gauge = precompiled.lagrangian().feynman_rule(phi.bar, phi, photon, simplify=True)
    assert _canon(fresh_gauge) == _canon(pre_gauge)


def test_precompiled_full_stack_no_double_count():
    """Full gauge-fixed + ghost model: precompiled vs fresh lagrangian()."""
    gS, xiQCD = S("gS", "xiQCD")
    gluon = _make_gluon()
    ghost = _make_ghost()
    su3 = _make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=(
            _gauge_decl(su3)
            + GaugeFixing(su3, xi=xiQCD)
            + GhostLagrangian(su3)
        ),
    )

    L_fresh = model.lagrangian()
    L_pre = with_compiled_covariant_terms(model).lagrangian()
    assert len(L_fresh.terms) == len(L_pre.terms)


# ===========================================================================
# Compiled sector: scalar QED covariant term
# ===========================================================================

def test_lagrangian_scalar_qed_covariant():
    """Scalar QED current terms via Lagrangian API match low-level pipeline."""
    eQED, qPhi = S("eQED", "qPhi")
    phi = Field("PhiQED", spin=0, self_conjugate=False,
                symbol=S("phiQED"), conjugate_symbol=S("phiQEDbar"),
                quantum_numbers={"Q": qPhi})
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    model = Model(
        gauge_groups=(u1,),
        fields=(phi, photon),
        lagrangian_decl=_scalar_decl(phi),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 4  # two current terms + one contact term + one free bilinear

    L = model.lagrangian()

    q1, q2, q3 = S("q1", "q2", "q3")
    got_3pt = L.feynman_rule(phi.bar, phi, photon, simplify=True)

    ref_sum = Expression.num(0)
    for term in compiled:
        if len(term.fields) == 3:
            legs = (
                phi.leg(q1, conjugated=True, labels={}),
                phi.leg(q2, labels={}),
                photon.leg(q3, labels={LORENTZ_KIND: S("i3")}),
            )
            ref_sum += _ref_vertex(term, legs)
    assert _canon(got_3pt) != _canon(Expression.num(0)), "3-point vertex should be non-zero"


# ===========================================================================
# Compiled sector: gauge kinetic (abelian and non-abelian)
# ===========================================================================

def test_lagrangian_abelian_gauge_kinetic():
    """Abelian gauge kinetic bilinear via Lagrangian API."""
    eQED = S("eQED")
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=_gauge_decl(u1),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2  # metric + cross bilinears

    L = model.lagrangian()
    got = L.feynman_rule(photon, photon, simplify=True)

    q1, q2 = S("q1", "q2")
    ref_sum = Expression.num(0)
    for term in compiled:
        legs = (
            photon.leg(q1, labels={LORENTZ_KIND: S("mu1")}),
            photon.leg(q2, labels={LORENTZ_KIND: S("mu2")}),
        )
        ref_sum += _ref_vertex(term, legs)
    assert _canon(got) == _canon(ref_sum)


def test_lagrangian_yang_mills_cubic():
    """Non-abelian Yang-Mills 3-gluon vertex via Lagrangian API."""
    gS = S("gS")
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=_gauge_decl(su3),
    )

    L = model.lagrangian()
    got = L.feynman_rule(gluon, gluon, gluon, simplify=True)
    assert _canon(got) != _canon(Expression.num(0)), "3-gluon vertex should be non-zero"


def test_lagrangian_yang_mills_quartic():
    """Non-abelian Yang-Mills 4-gluon vertex via Lagrangian API."""
    gS = S("gS")
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=_gauge_decl(su3),
    )

    L = model.lagrangian()
    got = L.feynman_rule(gluon, gluon, gluon, gluon, simplify=True)
    assert _canon(got) != _canon(Expression.num(0)), "4-gluon vertex should be non-zero"


# ===========================================================================
# Compiled sector: gauge fixing
# ===========================================================================

def test_lagrangian_abelian_gauge_fixing():
    """Abelian gauge-fixing bilinear via Lagrangian API matches low-level."""
    eQED, xiQED = S("eQED", "xiQED")
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=GaugeFixing(u1, xi=xiQED),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1

    L = model.lagrangian()
    got = L.feynman_rule(photon, photon, simplify=True)

    q1, q2 = S("q1", "q2")
    legs = (
        photon.leg(q1, labels={LORENTZ_KIND: S("mu1")}),
        photon.leg(q2, labels={LORENTZ_KIND: S("mu2")}),
    )
    ref = _ref_vertex(compiled[0], legs)
    assert _canon(got) == _canon(ref)


def test_lagrangian_nonabelian_gauge_fixing():
    """Non-abelian gauge-fixing bilinear via Lagrangian API matches low-level."""
    gS, xiQCD = S("gS", "xiQCD")
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=GaugeFixing(su3, xi=xiQCD),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1

    L = model.lagrangian()
    got = L.feynman_rule(gluon, gluon, simplify=True)

    q1, q2 = S("q1", "q2")
    legs = (
        gluon.leg(q1, labels={LORENTZ_KIND: S("mu1"), COLOR_ADJ_KIND: S("a1")}),
        gluon.leg(q2, labels={LORENTZ_KIND: S("mu2"), COLOR_ADJ_KIND: S("a2")}),
    )
    ref = _ref_vertex(compiled[0], legs)
    assert _canon(got) == _canon(ref)


def test_gauge_fixing_wrapper_preserves_source_and_scalar_prefactor():
    """GaugeFixing(...) stays visible in the source declaration and scales naturally."""
    eQED, xiQED = S("eQED", "xiQED")
    photon = _make_photon()
    u1 = _make_u1(eQED, photon.symbol)

    base = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=GaugeFixing(u1, xi=xiQED),
    )
    scaled = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=-(Expression.num(2) * GaugeFixing(u1, xi=xiQED)),
    )

    assert str(scaled.lagrangian_decl.source_terms[0]) == f"-2 * GaugeFixing(U1, xi={xiQED})"

    got = scaled.lagrangian().feynman_rule(photon, photon, simplify=True)
    ref = -(Expression.num(2) * base.lagrangian().feynman_rule(photon, photon, simplify=True))
    assert _canon(got) == _canon(ref)


# ===========================================================================
# Compiled sector: ghosts
# ===========================================================================

def test_lagrangian_ghost_bilinear():
    """Ghost bilinear via Lagrangian API matches low-level pipeline."""
    gS = S("gS")
    gluon = _make_gluon()
    ghost = _make_ghost()
    su3 = _make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GhostLagrangian(su3),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2  # bilinear + gauge interaction

    L = model.lagrangian()
    got = L.feynman_rule(ghost.bar, ghost, simplify=True)

    q1, q2 = S("q1", "q2")
    bilinear = compiled[0]
    legs = (
        ghost.leg(q1, conjugated=True, labels={COLOR_ADJ_KIND: S("a1")}),
        ghost.leg(q2, labels={COLOR_ADJ_KIND: S("a2")}),
    )
    ref = _ref_vertex(bilinear, legs)
    assert _canon(got) == _canon(ref)


def test_lagrangian_ghost_gauge_interaction():
    """Ghost-gluon 3-point vertex via Lagrangian API matches low-level."""
    gS = S("gS")
    gluon = _make_gluon()
    ghost = _make_ghost()
    su3 = _make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GhostLagrangian(su3),
    )

    compiled = compile_covariant_terms(model)
    ghost_gauge_term = compiled[1]

    L = model.lagrangian()
    got = L.feynman_rule(ghost.bar, gluon, ghost, simplify=True)

    q1, q2, q3 = S("q1", "q2", "q3")
    legs = (
        ghost.leg(q1, conjugated=True, labels={COLOR_ADJ_KIND: S("a1")}),
        gluon.leg(q2, labels={LORENTZ_KIND: S("mu2"), COLOR_ADJ_KIND: S("a2")}),
        ghost.leg(q3, labels={COLOR_ADJ_KIND: S("a3")}),
    )
    ref = _ref_vertex(ghost_gauge_term, legs)
    assert _canon(got) == _canon(ref)


def test_manual_raw_spenso_gauge_fixing_and_ghost_sources():
    """Raw Spenso tensors plus labeled FieldOccurrence factors match built-in wrappers."""
    gS, xiQCD = S("gS", "xiQCD")
    gluon = _make_gluon()
    ghost = _make_ghost()
    su3 = _make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol)

    mu_left = S("mu_left")
    mu_right = S("mu_right")
    rho_left = S("rho_left")
    rho_right = S("rho_right")
    mu = S("mu")
    rho_ghost = S("rho_ghost")

    a_left = S("a_left")
    a_right = S("a_right")
    a_bar = S("a_bar")
    a_gluon = S("a_gluon")
    a_ghost = S("a_ghost")

    manual = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=(
            -(Expression.num(1) / (Expression.num(2) * xiQCD))
            * COLOR_ADJ_INDEX.representation.g(a_left, a_right).to_expression()
            * LORENTZ_INDEX.representation.g(mu_left, rho_left).to_expression()
            * LORENTZ_INDEX.representation.g(mu_right, rho_right).to_expression()
            * PartialD(
                gluon.occurrence(
                    labels={LORENTZ_KIND: mu_left, COLOR_ADJ_KIND: a_left}
                ),
                rho_left,
            )
            * PartialD(
                gluon.occurrence(
                    labels={LORENTZ_KIND: mu_right, COLOR_ADJ_KIND: a_right}
                ),
                rho_right,
            )
            + COLOR_ADJ_INDEX.representation.g(a_bar, a_ghost).to_expression()
            * PartialD(
                ghost.occurrence(
                    conjugated=True,
                    labels={COLOR_ADJ_KIND: a_bar},
                ),
                mu,
            )
            * PartialD(
                ghost.occurrence(labels={COLOR_ADJ_KIND: a_ghost}),
                mu,
            )
            + (
                -gS
                * structure_constant(a_bar, a_gluon, a_ghost)
                * LORENTZ_INDEX.representation.g(rho_ghost, mu_left).to_expression()
                * PartialD(
                    ghost.occurrence(
                        conjugated=True,
                        labels={COLOR_ADJ_KIND: a_bar},
                    ),
                    rho_ghost,
                )
                * gluon.occurrence(
                    labels={LORENTZ_KIND: mu_left, COLOR_ADJ_KIND: a_gluon}
                )
                * ghost.occurrence(labels={COLOR_ADJ_KIND: a_ghost})
            )
        ),
    )
    wrapped = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GaugeFixing(su3, xi=xiQCD) + GhostLagrangian(su3),
    )

    manual_lagrangian = manual.lagrangian()
    wrapped_lagrangian = wrapped.lagrangian()

    assert len(manual_lagrangian.terms) == 3
    assert _canon(manual_lagrangian.feynman_rule(gluon, gluon, simplify=True)) == _canon(
        wrapped_lagrangian.feynman_rule(gluon, gluon, simplify=True)
    )
    assert _canon(
        manual_lagrangian.feynman_rule(ghost.bar, ghost, simplify=True)
    ) == _canon(
        wrapped_lagrangian.feynman_rule(ghost.bar, ghost, simplify=True)
    )
    assert _canon(
        manual_lagrangian.feynman_rule(ghost.bar, gluon, ghost, simplify=True)
    ) == _canon(
        wrapped_lagrangian.feynman_rule(ghost.bar, gluon, ghost, simplify=True)
    )


def test_ghost_lagrangian_wrapper_preserves_source_and_scalar_prefactor():
    """GhostLagrangian(...) stays visible in the source declaration and scales naturally."""
    gS = S("gS")
    gluon = _make_gluon()
    ghost = _make_ghost()
    su3 = _make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol)

    base = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GhostLagrangian(su3),
    )
    scaled = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=Expression.num(3) * GhostLagrangian(su3),
    )

    assert str(scaled.lagrangian_decl.source_terms[0]) == "3 * GhostLagrangian(SU3)"

    got = scaled.lagrangian().feynman_rule(ghost.bar, gluon, ghost, simplify=True)
    ref = Expression.num(3) * base.lagrangian().feynman_rule(ghost.bar, gluon, ghost, simplify=True)
    assert _canon(got) == _canon(ref)


# ===========================================================================
# Precompiled-model idempotency on *actual* examples.py models
#
# These use the same model objects that examples.py feeds through
# with_compiled_covariant_terms(), so they guard against the double-counting
# regression on the real repo workflow.
# ===========================================================================

def _assert_precompiled_idempotent(model, label):
    """Fresh lagrangian() and precompiled lagrangian() must have equal term count."""
    L_fresh = model.lagrangian()
    precompiled = with_compiled_covariant_terms(model)

    assert precompiled.covariant_terms == (), f"{label}: covariant_terms not cleared"
    assert precompiled.gauge_kinetic_terms == (), f"{label}: gauge_kinetic_terms not cleared"
    assert precompiled.gauge_fixing_terms == (), f"{label}: gauge_fixing_terms not cleared"
    assert precompiled.ghost_terms == (), f"{label}: ghost_terms not cleared"

    L_pre = precompiled.lagrangian()
    assert len(L_fresh.terms) == len(L_pre.terms), (
        f"{label}: term count mismatch: fresh={len(L_fresh.terms)}, "
        f"precompiled={len(L_pre.terms)}"
    )
    return L_fresh, L_pre


def test_precompiled_examples_qcd_fermion():
    """MODEL_QCD_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QCD_COVARIANT, "QCD-covariant")
    got = L_fresh.feynman_rule(QuarkField.bar, QuarkField, GluonField, simplify=True)
    got_pre = L_pre.feynman_rule(QuarkField.bar, QuarkField, GluonField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qed_fermion():
    """MODEL_QED_FERMION_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QED_FERMION_COVARIANT, "FermionQED-covariant")
    got = L_fresh.feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField, simplify=True)
    got_pre = L_pre.feynman_rule(PsiQEDField.bar, PsiQEDField, GaugeField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_scalar_qed():
    """MODEL_SCALAR_QED_COVARIANT: precompiled vs fresh, term count."""
    _assert_precompiled_idempotent(MODEL_SCALAR_QED_COVARIANT, "ScalarQED-covariant")


def test_precompiled_examples_scalar_qcd():
    """MODEL_SCALAR_QCD_COVARIANT: precompiled vs fresh, term count."""
    _assert_precompiled_idempotent(MODEL_SCALAR_QCD_COVARIANT, "ScalarQCD-covariant")


def test_precompiled_examples_mixed_fermion():
    """MODEL_MIXED_FERMION_COVARIANT: precompiled vs fresh, term count."""
    _assert_precompiled_idempotent(
        MODEL_MIXED_FERMION_COVARIANT, "MixedQCDQED-covariant")


def test_precompiled_examples_mixed_scalar():
    """MODEL_MIXED_SCALAR_COVARIANT: precompiled vs fresh, term count."""
    _assert_precompiled_idempotent(
        MODEL_MIXED_SCALAR_COVARIANT, "MixedScalarQCDQED-covariant")


def test_precompiled_examples_qed_gauge_kinetic():
    """MODEL_QED_GAUGE_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QED_GAUGE_COVARIANT, "QEDGauge-covariant")
    got = L_fresh.feynman_rule(GaugeField, GaugeField, simplify=True)
    got_pre = L_pre.feynman_rule(GaugeField, GaugeField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qcd_gauge_kinetic():
    """MODEL_QCD_GAUGE_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QCD_GAUGE_COVARIANT, "QCDGauge-covariant")
    got = L_fresh.feynman_rule(GluonField, GluonField, simplify=True)
    got_pre = L_pre.feynman_rule(GluonField, GluonField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qed_gauge_fixing():
    """MODEL_QED_GAUGE_FIXING_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QED_GAUGE_FIXING_COVARIANT, "QEDGaugeFixing-covariant")
    got = L_fresh.feynman_rule(GaugeField, GaugeField, simplify=True)
    got_pre = L_pre.feynman_rule(GaugeField, GaugeField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qcd_gauge_fixing():
    """MODEL_QCD_GAUGE_FIXING_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QCD_GAUGE_FIXING_COVARIANT, "QCDGaugeFixing-covariant")
    got = L_fresh.feynman_rule(GluonField, GluonField, simplify=True)
    got_pre = L_pre.feynman_rule(GluonField, GluonField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qed_ordinary_gauge_fixed():
    """MODEL_QED_ORDINARY_GAUGE_FIXED: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QED_ORDINARY_GAUGE_FIXED, "QEDGaugeFixed-covariant")
    got = L_fresh.feynman_rule(GaugeField, GaugeField, simplify=True)
    got_pre = L_pre.feynman_rule(GaugeField, GaugeField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qcd_ghost():
    """MODEL_QCD_GHOST_COVARIANT: precompiled vs fresh, vertex-level."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QCD_GHOST_COVARIANT, "QCDGhost-covariant")
    got = L_fresh.feynman_rule(GhostGluonField.bar, GhostGluonField, simplify=True)
    got_pre = L_pre.feynman_rule(GhostGluonField.bar, GhostGluonField, simplify=True)
    assert _canon(got) == _canon(got_pre)


def test_precompiled_examples_qcd_ordinary_gauge_fixed():
    """MODEL_QCD_ORDINARY_GAUGE_FIXED: precompiled vs fresh, term count + vertex."""
    L_fresh, L_pre = _assert_precompiled_idempotent(
        MODEL_QCD_ORDINARY_GAUGE_FIXED, "QCDGaugeFixed-covariant")
    got = L_fresh.feynman_rule(GluonField, GluonField, simplify=True)
    got_pre = L_pre.feynman_rule(GluonField, GluonField, simplify=True)
    assert _canon(got) == _canon(got_pre)
    got_3g = L_fresh.feynman_rule(GluonField, GluonField, GluonField, simplify=True)
    got_3g_pre = L_pre.feynman_rule(GluonField, GluonField, GluonField, simplify=True)
    assert _canon(got_3g) == _canon(got_3g_pre)
    got_ghost = L_fresh.feynman_rule(
        GhostGluonField.bar, GhostGluonField, simplify=True)
    got_ghost_pre = L_pre.feynman_rule(
        GhostGluonField.bar, GhostGluonField, simplify=True)
    assert _canon(got_ghost) == _canon(got_ghost_pre)


def test_dressed_dirac_covd_with_scalar_spectators():
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": S("Qpsi")},
    )
    A = _make_photon()
    U1 = _make_u1(S("e"), A.symbol)
    mu = S("mu")

    model = Model(
        gauge_groups=(U1,),
        fields=(psi, phi, A),
        lagrangian_decl=I * psi.bar * Gamma(mu) * CovD(psi, mu) * phi * phi,
    )
    L = model.lagrangian()

    i_bar, i_psi = S("i_bar"), S("i_psi")
    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=I * psi_bar_gamma_psi(i_bar, i_psi, mu),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                phi.occurrence(),
                phi.occurrence(),
            ),
            derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        ),
        InteractionTerm(
            coupling=-(S("e") * S("Qpsi")) * psi_bar_gamma_psi(i_bar, i_psi, mu),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                phi.occurrence(),
                phi.occurrence(),
            ),
        ),
    ))

    assert _canon(L.feynman_rule(psi.bar, psi, phi, phi)) == _canon(
        ref.feynman_rule(psi.bar, psi, phi, phi)
    )
    assert _canon(L.feynman_rule(psi.bar, psi, A, phi, phi)) == _canon(
        ref.feynman_rule(psi.bar, psi, A, phi, phi)
    )


def test_dressed_scalar_covd_with_complex_scalar_spectators():
    phi = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phibar"),
        quantum_numbers={"Q": S("Qphi")},
    )
    chi = Field(
        "Chi",
        spin=0,
        self_conjugate=False,
        symbol=S("chi"),
        conjugate_symbol=S("chibar"),
    )
    A = _make_photon()
    U1 = _make_u1(S("e"), A.symbol)
    mu = S("mu")
    nu = S("nu")

    model = Model(
        gauge_groups=(U1,),
        fields=(phi, chi, A),
        lagrangian_decl=CovD(phi.bar, mu) * CovD(phi, mu) * chi.bar * chi,
    )
    L = model.lagrangian()

    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
                chi.occurrence(conjugated=True),
                chi.occurrence(),
            ),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=mu),
                DerivativeAction(target=1, lorentz_index=mu),
            ),
        ),
        InteractionTerm(
            coupling=I * S("e") * S("Qphi"),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                chi.occurrence(conjugated=True),
                chi.occurrence(),
            ),
            derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        ),
        InteractionTerm(
            coupling=-(I * S("e") * S("Qphi")),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                chi.occurrence(conjugated=True),
                chi.occurrence(),
            ),
            derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        ),
        InteractionTerm(
            coupling=(S("e") * S("Qphi")) ** 2 * lorentz_metric(mu, nu),
            fields=(
                phi.occurrence(conjugated=True),
                phi.occurrence(),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                A.occurrence(labels={LORENTZ_KIND: nu}),
                chi.occurrence(conjugated=True),
                chi.occurrence(),
            ),
        ),
    ))

    assert _canon(L.feynman_rule(phi.bar, phi, chi.bar, chi)) == _canon(
        ref.feynman_rule(phi.bar, phi, chi.bar, chi)
    )
    assert _canon(L.feynman_rule(phi.bar, phi, A, chi.bar, chi)) == _canon(
        ref.feynman_rule(phi.bar, phi, A, chi.bar, chi)
    )
    assert _canon(L.feynman_rule(phi.bar, phi, A, A, chi.bar, chi)) == _canon(
        ref.feynman_rule(phi.bar, phi, A, A, chi.bar, chi)
    )


def test_dressed_dirac_covd_with_fermion_spectator_bilinear():
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": S("Qpsi")},
    )
    xi = Field(
        "Xi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("xi"),
        conjugate_symbol=S("xibar"),
        indices=(SPINOR_INDEX,),
    )
    A = _make_photon()
    U1 = _make_u1(S("e"), A.symbol)
    mu = S("mu")

    model = Model(
        gauge_groups=(U1,),
        fields=(psi, xi, A),
        lagrangian_decl=I * psi.bar * Gamma(mu) * CovD(psi, mu) * xi.bar * xi,
    )
    L = model.lagrangian()

    i_bar, i_psi = S("i_bar"), S("i_psi")
    j_bar, j_psi = S("j_bar"), S("j_psi")
    ref = Lagrangian(terms=(
        InteractionTerm(
            coupling=I * psi_bar_gamma_psi(i_bar, i_psi, mu) * psi_bar_psi(j_bar, j_psi),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                xi.occurrence(conjugated=True, labels={SPINOR_KIND: j_bar}),
                xi.occurrence(labels={SPINOR_KIND: j_psi}),
            ),
            derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        ),
        InteractionTerm(
            coupling=-(S("e") * S("Qpsi")) * psi_bar_gamma_psi(i_bar, i_psi, mu) * psi_bar_psi(j_bar, j_psi),
            fields=(
                psi.occurrence(conjugated=True, labels={SPINOR_KIND: i_bar}),
                psi.occurrence(labels={SPINOR_KIND: i_psi}),
                A.occurrence(labels={LORENTZ_KIND: mu}),
                xi.occurrence(conjugated=True, labels={SPINOR_KIND: j_bar}),
                xi.occurrence(labels={SPINOR_KIND: j_psi}),
            ),
        ),
    ))

    assert _canon(L.feynman_rule(psi.bar, psi, xi.bar, xi)) == _canon(
        ref.feynman_rule(psi.bar, psi, xi.bar, xi)
    )
    assert _canon(L.feynman_rule(psi.bar, psi, A, xi.bar, xi)) == _canon(
        ref.feynman_rule(psi.bar, psi, A, xi.bar, xi)
    )


def test_generic_declared_monomial_supports_multiple_dirac_covd_chains():
    gS = S("gS")
    mu = S("mu")
    nu = S("nu")
    a = S("a")
    b = S("b")
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    chi = Field(
        "Chi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("chi"),
        conjugate_symbol=S("chibar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(psi, chi, gluon),
        lagrangian_decl=(
            I
            * psi.bar
            * Gamma(mu)
            * CovD(psi, mu)
            * chi.bar
            * Gamma(nu)
            * CovD(chi, nu)
        ),
    )
    L = model.lagrangian()

    gluon_mu = gluon.occurrence(labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a})
    gluon_nu = gluon.occurrence(labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: a})
    ref_single_gluon = Lagrangian(
        (-gS) * psi.bar * Gamma(mu) * T(a) * psi * chi.bar * Gamma(nu) * PartialD(chi, nu) * gluon_mu
        + (-gS) * psi.bar * Gamma(mu) * PartialD(psi, mu) * chi.bar * Gamma(nu) * T(a) * chi * gluon_nu
    )
    assert _canon(
        L.feynman_rule(psi.bar, psi, chi.bar, chi, gluon, simplify=True)
    ) == _canon(
        ref_single_gluon.feynman_rule(
            psi.bar, psi, chi.bar, chi, gluon, simplify=True
        )
    )

    gluon_mu_a = gluon.occurrence(labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a})
    gluon_nu_b = gluon.occurrence(labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: b})
    ref_double_gluon = Lagrangian(
        -(I * gS * gS)
        * psi.bar
        * Gamma(mu)
        * T(a)
        * psi
        * chi.bar
        * Gamma(nu)
        * T(b)
        * chi
        * gluon_mu_a
        * gluon_nu_b
    )
    assert _canon(
        L.feynman_rule(psi.bar, psi, chi.bar, chi, gluon, gluon, simplify=True)
    ) == _canon(
        ref_double_gluon.feynman_rule(
            psi.bar, psi, chi.bar, chi, gluon, gluon, simplify=True
        )
    )


def test_generic_declared_monomial_supports_same_species_quark_covd_product():
    gS = S("gS")
    mu = S("mu")
    nu = S("nu")
    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    gluon = _make_gluon()
    su3 = _make_su3(gS, gluon.symbol)

    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=(
            I
            * quark.bar
            * Gamma(mu)
            * CovD(quark, mu)
            * quark.bar
            * Gamma(nu)
            * CovD(quark, nu)
        ),
    )
    L = model.lagrangian()

    got_single_gluon = L.feynman_rule(
        quark.bar, quark, quark.bar, quark, gluon, simplify=True
    )

    mu_int = S("mu1_int")
    q1, q2, q3, q4, q5 = S("q1", "q2", "q3", "q4", "q5")
    c1, c2, c3, c4 = S("c1", "c2", "c3", "c4")
    i1, i2, i3, i4 = S("i1", "i2", "i3", "i4")
    mu5 = S("mu5")
    a5 = S("a5")
    color_metric = COLOR_FUND_INDEX.representation.g
    expected_relative_signs = (
        color_metric(c1, c2).to_expression()
        * gamma_matrix(i1, i2, mu_int)
        * gamma_matrix(i3, i4, mu5)
        * gauge_generator(a5, c3, c4)
        * pcomp(q2, mu_int)
        + color_metric(c1, c4).to_expression()
        * gamma_matrix(i1, i4, mu_int)
        * gamma_matrix(i3, i2, mu5)
        * gauge_generator(a5, c3, c2)
        * pcomp(q4, mu_int)
        + color_metric(c2, c3).to_expression()
        * gamma_matrix(i1, i4, mu5)
        * gamma_matrix(i3, i2, mu_int)
        * gauge_generator(a5, c1, c4)
        * pcomp(q2, mu_int)
        + color_metric(c3, c4).to_expression()
        * gamma_matrix(i1, i2, mu5)
        * gamma_matrix(i3, i4, mu_int)
        * gauge_generator(a5, c1, c2)
        * pcomp(q4, mu_int)
    )
    expected_single_gluon = (
        2
        * gS
        * (2 * pi) ** S("d")
        * Delta(q1 + q2 + q3 + q4 + q5)
        * expected_relative_signs
    )

    got_single_gluon_canon = _canon(got_single_gluon)
    assert got_single_gluon_canon in {
        _canon(expected_single_gluon),
        _canon(-expected_single_gluon),
    }
    assert L.feynman_rule(quark.bar, quark, quark.bar, quark, gluon, gluon, simplify=True) != 0
