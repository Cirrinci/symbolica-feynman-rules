"""Tests for the FeynRules-style Lagrangian API.

Validates that ``Lagrangian.feynman_rule()`` produces the same vertex factors
as the existing ``vertex_factor()`` pipeline, using automatic index and
momentum conventions.
"""

import sys
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from symbolica import S, Expression  # noqa: E402

from gauge_compiler import compile_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_INDEX,
    COLOR_FUND_KIND,
    LORENTZ_INDEX,
    LORENTZ_KIND,
    SPINOR_INDEX,
    SPINOR_KIND,
    ConjugateField,
    ComplexScalarKineticTerm,
    DerivativeAction,
    DiracKineticTerm,
    Field,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    InteractionTerm,
    Lagrangian,
    Model,
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
from spenso_structures import gauge_generator, structure_constant  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canon(expr):
    return expr.expand().to_canonical_string()


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
    try:
        L.feynman_rule(phi, phi, chi, chi)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


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
    eQED = S("eQED")
    qPsi = S("qPsi")
    psi_sym = S("psi")
    psibar_sym = S("psibar")
    photon_sym = S("A")

    fermion = Field(
        "PsiQED",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=psi_sym,
        conjugate_symbol=psibar_sym,
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": qPsi},
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=photon_sym,
        indices=(LORENTZ_INDEX,),
    )
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=eQED,
        gauge_boson=photon.symbol,
        charge="Q",
    )
    model = Model(
        gauge_groups=(u1,),
        fields=(fermion, photon),
        covariant_terms=(DiracKineticTerm(field=fermion),),
    )

    L = model.lagrangian()
    got = L.feynman_rule(fermion.bar, fermion, photon, simplify=True)

    q1, q2, q3 = S("q1", "q2", "q3")
    i1, i2 = S("i1", "i2")
    i3 = S("i3")
    d = S("d")

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1
    legs = (
        fermion.leg(q1, conjugated=True, labels={SPINOR_KIND: i1}),
        fermion.leg(q2, labels={SPINOR_KIND: i2}),
        photon.leg(q3, labels={LORENTZ_KIND: i3}),
    )
    ref = vertex_factor(
        interaction=compiled[0],
        external_legs=legs,
        x=S("x_"),
        d=d,
        strip_externals=True,
        include_delta=True,
    )
    ref = simplify_vertex(ref)

    assert _canon(got) == _canon(ref)


# ---------------------------------------------------------------------------
# Model.lagrangian() integration: QCD quark-gluon vertex
# ---------------------------------------------------------------------------

def test_model_lagrangian_qcd_fermion():
    """Model.lagrangian().feynman_rule for a QCD quark-gluon vertex."""
    gS = S("gS")
    psi_sym = S("psi")
    psibar_sym = S("psibar")
    gluon_sym = S("G")

    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=psi_sym,
        conjugate_symbol=psibar_sym,
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=gluon_sym,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        covariant_terms=(DiracKineticTerm(field=quark),),
    )

    L = model.lagrangian()
    got = L.feynman_rule(quark.bar, quark, gluon, simplify=True)

    q1, q2, q3 = S("q1", "q2", "q3")
    d = S("d")

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1
    legs = (
        quark.leg(q1, conjugated=True, labels={SPINOR_KIND: S("i1"), COLOR_FUND_KIND: S("i2")}),
        quark.leg(q2, labels={SPINOR_KIND: S("i3"), COLOR_FUND_KIND: S("i4")}),
        gluon.leg(q3, labels={LORENTZ_KIND: S("i5"), COLOR_ADJ_KIND: S("i6")}),
    )
    ref = vertex_factor(
        interaction=compiled[0],
        external_legs=legs,
        x=S("x_"),
        d=d,
        strip_externals=True,
        include_delta=True,
    )
    ref = simplify_vertex(ref)

    assert _canon(got) == _canon(ref)
