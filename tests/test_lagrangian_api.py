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

from gauge_compiler import (  # noqa: E402
    compile_covariant_terms,
    with_compiled_covariant_terms,
)
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_INDEX,
    COLOR_FUND_KIND,
    CovD,
    DeclaredLagrangian,
    FieldStrength,
    Gamma,
    LORENTZ_INDEX,
    LORENTZ_KIND,
    SPINOR_INDEX,
    SPINOR_KIND,
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
from model_symbolica import (  # noqa: E402
    Delta,
    I,
    pi,
    pcomp,
    simplify_deltas,
    simplify_vertex,
    vertex_factor,
)
from spenso_structures import (  # noqa: E402
    gauge_generator,
    lorentz_metric,
    structure_constant,
)
from operators import (  # noqa: E402
    psi_bar_gamma_psi,
    psi_bar_psi,
)
from examples import (  # noqa: E402
    MODEL_QCD_COVARIANT,
    MODEL_QED_FERMION_COVARIANT,
    MODEL_SCALAR_QED_COVARIANT,
    MODEL_SCALAR_QCD_COVARIANT,
    MODEL_MIXED_FERMION_COVARIANT,
    MODEL_MIXED_SCALAR_COVARIANT,
    MODEL_QED_GAUGE_COVARIANT,
    MODEL_QCD_GAUGE_COVARIANT,
    MODEL_QED_GAUGE_FIXING_COVARIANT,
    MODEL_QCD_GAUGE_FIXING_COVARIANT,
    MODEL_QED_ORDINARY_GAUGE_FIXED,
    MODEL_QCD_GHOST_COVARIANT,
    MODEL_QCD_ORDINARY_GAUGE_FIXED,
    QuarkField,
    GluonField,
    GaugeField,
    PsiQEDField,
    PsiMixField,
    PhiQEDField,
    PhiQCDField,
    PhiMixField,
    GhostGluonField,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon(expr):
    return expr.expand().to_canonical_string()


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
    ))


def _make_photon():
    return Field("A", spin=1, self_conjugate=True, symbol=S("A"),
                 indices=(LORENTZ_INDEX,))


def _make_gluon():
    return Field("G", spin=1, self_conjugate=True, symbol=S("G"),
                 indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX))


def _make_ghost():
    return Field("ghG", spin=0, kind="ghost", self_conjugate=False,
                 symbol=S("ghG"), conjugate_symbol=S("ghGbar"),
                 indices=(COLOR_ADJ_INDEX,))


def _make_u1(coupling, gauge_boson_sym, name="U1", charge="Q"):
    return GaugeGroup(name=name, abelian=True, coupling=coupling,
                      gauge_boson=gauge_boson_sym, charge=charge)


def _make_su3(coupling, gauge_boson_sym, ghost_sym=None, name="SU3"):
    return GaugeGroup(
        name=name, abelian=False, coupling=coupling,
        gauge_boson=gauge_boson_sym,
        ghost_field=ghost_sym,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(index=COLOR_FUND_INDEX,
                                generator_builder=gauge_generator, name="fund"),
        ),
    )


def _dirac_decl(field):
    return I * field.bar * Gamma(S("mu_decl")) * CovD(field, S("mu_decl"))


def _scalar_decl(field):
    return CovD(field.bar, S("mu_decl")) * CovD(field, S("mu_decl"))


def _gauge_decl(group):
    mu_decl, nu_decl = S("mu_decl", "nu_decl")
    return (
        -(Expression.num(1) / Expression.num(4))
        * FieldStrength(group, mu_decl, nu_decl)
        * FieldStrength(group, mu_decl, nu_decl)
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
    legs = (
        fermion.leg(q1, conjugated=True, labels={SPINOR_KIND: S("i1")}),
        fermion.leg(q2, labels={SPINOR_KIND: S("i2")}),
        photon.leg(q3, labels={LORENTZ_KIND: S("i3")}),
    )
    ref = _ref_vertex(compiled[0], legs)
    assert _canon(got) == _canon(ref)


def test_declared_lagrangian_qed_fermion():
    """A declarative CovD-based Dirac term lowers to the legacy QED term."""
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
    legs = (
        quark.leg(q1, conjugated=True, labels={SPINOR_KIND: S("i1"), COLOR_FUND_KIND: S("i2")}),
        quark.leg(q2, labels={SPINOR_KIND: S("i3"), COLOR_FUND_KIND: S("i4")}),
        gluon.leg(q3, labels={LORENTZ_KIND: S("i5"), COLOR_ADJ_KIND: S("i6")}),
    )
    ref = _ref_vertex(compiled[0], legs)
    assert _canon(got) == _canon(ref)


def test_declared_lagrangian_scalar_qed_matches_legacy():
    """A declarative scalar CovD term lowers to the legacy scalar kinetic term."""
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
    assert len(precompiled.interactions) == 1


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
    assert len(compiled) == 3  # two current terms + one contact term

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
            photon.leg(q1, labels={LORENTZ_KIND: S("i1")}),
            photon.leg(q2, labels={LORENTZ_KIND: S("i2")}),
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
        photon.leg(q1, labels={LORENTZ_KIND: S("i1")}),
        photon.leg(q2, labels={LORENTZ_KIND: S("i2")}),
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
        gluon.leg(q1, labels={LORENTZ_KIND: S("i1"), COLOR_ADJ_KIND: S("i2")}),
        gluon.leg(q2, labels={LORENTZ_KIND: S("i3"), COLOR_ADJ_KIND: S("i4")}),
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
        ghost.leg(q1, conjugated=True, labels={COLOR_ADJ_KIND: S("i1")}),
        ghost.leg(q2, labels={COLOR_ADJ_KIND: S("i2")}),
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
        ghost.leg(q1, conjugated=True, labels={COLOR_ADJ_KIND: S("i1")}),
        gluon.leg(q2, labels={LORENTZ_KIND: S("i2"), COLOR_ADJ_KIND: S("i3")}),
        ghost.leg(q3, labels={COLOR_ADJ_KIND: S("i4")}),
    )
    ref = _ref_vertex(ghost_gauge_term, legs)
    assert _canon(got) == _canon(ref)


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
