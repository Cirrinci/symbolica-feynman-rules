"""Tests for the unified, source-agnostic typed-index handling.

These cover the cases the older lowering could not handle without manual slot
labels: raw Spenso tensors (``gamma(...)``, ``t(...)``, ``f(...)``) and indexed
symbols drive field-slot attachment through the *same* pipeline as the
declarative helpers (``Gamma(...)``, ``T(...)``, ``StructureConstant(...)``).

The guiding properties are:

* a typed index written on any tensor claims its matching field slot, whether
  declarative or raw, and whether the slot was labelled by hand or not,
* the adjoint representation follows the gauge group (color vs weak), and
* genuinely ambiguous attachments raise instead of silently inventing labels.
"""

from __future__ import annotations

from fractions import Fraction
import re

import pytest
from symbolica import S

from model import (
    COLOR_FUND_INDEX,
    Field,
    FieldStrength,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    LORENTZ_INDEX,
    Model,
    Parameter,
    SPINOR_INDEX,
    T,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    dirac_field,
)
from symbolic.spenso_structures import (
    gamma_matrix,
    gauge_generator,
    weak_gauge_generator,
    weak_structure_constant,
)
from tests.support.builders import make_gluon, make_su3

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _compiled_terms(model):
    return model.lagrangian().terms


def _plain(text) -> str:
    return _ANSI.sub("", str(text))


def _canon(expr):
    return expr.expand().to_canonical_string()


def _make_w_field(name="W"):
    return Field(
        name,
        spin=1,
        self_conjugate=True,
        symbol=S("W0"),
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )


def _make_weak_group(gauge_boson_field, *, coupling=None, name="SU2L"):
    """SU(2) group whose ``gauge_boson`` is the real Field (not a symbol).

    Passing the Field lets the adjoint representation be resolved from the boson
    itself, so the weak adjoint typing is exercised end to end.
    """
    return GaugeGroup(
        name=name,
        abelian=False,
        coupling=coupling or S("g2"),
        gauge_boson=gauge_boson_field,
        structure_constant=weak_structure_constant,
        representations=(
            GaugeRepresentation(
                index=WEAK_FUND_INDEX,
                generator_builder=weak_gauge_generator,
                name="doublet",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Implicit (unlabelled) raw gamma * gauge_generator * psi.
# ---------------------------------------------------------------------------
def test_implicit_raw_gamma_generator_attaches_color_slots():
    mu, nu, a = S("mu"), S("nu"), S("a")
    s1, s2, s3, i, j = S("s1"), S("s2"), S("s3"), S("i"), S("j")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    quark = dirac_field(
        "qFS",
        indices=(COLOR_FUND_INDEX,),
        symbol=S("qFS"),
        conjugate_symbol=S("qFSbar"),
    )

    # No color (or spinor) slot labels written on the fermions: the raw tensors
    # alone must drive the full attachment.
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * quark.bar
        * gamma_matrix(s1, s2, mu)
        * gamma_matrix(s2, s3, nu)
        * gauge_generator(a, i, j)
        * quark,
    )

    compiled = _compiled_terms(model)
    assert len(compiled) == 3
    for term in compiled:
        qbar = next(o for o in term.fields if o.field.name == "qFS" and o.conjugated)
        q = next(o for o in term.fields if o.field.name == "qFS" and not o.conjugated)
        qbar_labels = qbar.field.unpack_slot_labels(qbar.labels)
        q_labels = q.field.unpack_slot_labels(q.labels)
        color_slot = qbar.field.index_positions(index=COLOR_FUND_INDEX)[0]
        # The raw generator indices i and j claimed the antiquark / quark color
        # slots; no fresh internal color label was invented.
        assert _plain(qbar_labels[color_slot]) == "i"
        assert _plain(q_labels[color_slot]) == "j"


def test_implicit_raw_matches_declarative_and_explicit_labels():
    mu, nu, a = S("mu"), S("nu"), S("a")
    s1, s2, s3, i, j = S("s1"), S("s2"), S("s3"), S("i"), S("j")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    quark = dirac_field(
        "qFS",
        indices=(COLOR_FUND_INDEX,),
        symbol=S("qFS"),
        conjugate_symbol=S("qFSbar"),
    )

    declared = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * quark.bar
        * Gamma(mu)
        * Gamma(nu)
        * T(a)
        * quark,
    )
    raw_implicit = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * quark.bar
        * gamma_matrix(s1, s2, mu)
        * gamma_matrix(s2, s3, nu)
        * gauge_generator(a, i, j)
        * quark,
    )
    # The same operator written with the slots labelled by hand.
    qbar = quark.bar(index_labels={SPINOR_INDEX.kind: s1, COLOR_FUND_INDEX.kind: i})
    q = quark(index_labels={SPINOR_INDEX.kind: s3, COLOR_FUND_INDEX.kind: j})
    raw_explicit = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * qbar
        * gamma_matrix(s1, s2, mu)
        * gamma_matrix(s2, s3, nu)
        * gauge_generator(a, i, j)
        * q,
    )

    # Same physical vertices as the declarative chain ...
    assert (
        raw_implicit.lagrangian().vertex_signatures()
        == declared.lagrangian().vertex_signatures()
    )
    # ... and byte-identical to writing the slot labels by hand: auto-attachment
    # reproduces the manual labelling exactly.
    for legs in (
        (quark.bar, quark, gluon),
        (quark.bar, quark, gluon, gluon),
    ):
        assert _canon(
            raw_implicit.lagrangian().feynman_rule(*legs, include_delta=False)
        ) == _canon(
            raw_explicit.lagrangian().feynman_rule(*legs, include_delta=False)
        )


def test_explicit_gamma_and_t_api_matches_raw_tensor_form():
    mu, nu, a = S("mu"), S("nu"), S("a")
    s1, s2, s3, i, j = S("s1"), S("s2"), S("s3"), S("i"), S("j")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    psi = Field(
        "PsiC",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psiC"),
        conjugate_symbol=S("psibarC"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )

    explicit_api = Model(
        gauge_groups=(su3,),
        fields=(psi, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * psi.bar(s1, i)
        * Gamma(s1, s2, mu)
        * Gamma(s2, s3, nu)
        * T(a, i, j)
        * psi(s3, j),
    )
    raw = Model(
        gauge_groups=(su3,),
        fields=(psi, gluon),
        lagrangian_decl=FieldStrength(su3, mu, nu, a)
        * psi.bar(s1, i)
        * gamma_matrix(s1, s2, mu)
        * gamma_matrix(s2, s3, nu)
        * gauge_generator(a, i, j)
        * psi(s3, j),
    )

    explicit_compiled = _compiled_terms(explicit_api)
    raw_compiled = _compiled_terms(raw)

    assert len(explicit_compiled) == len(raw_compiled) == 3
    assert explicit_api.lagrangian().vertex_signatures() == raw.lagrangian().vertex_signatures()
    for legs in (
        (psi.bar, psi, gluon),
        (psi.bar, psi, gluon, gluon),
    ):
        assert _canon(
            explicit_api.lagrangian().feynman_rule(*legs, include_delta=False)
        ) == _canon(
            raw.lagrangian().feynman_rule(*legs, include_delta=False)
        )


# ---------------------------------------------------------------------------
# 2. Raw weak structure constant with SU(2) field strengths.
# ---------------------------------------------------------------------------
def test_raw_weak_structure_constant_uses_weak_adjoint():
    mu, nu, rho = S("mu"), S("nu"), S("rho")
    a, b, c = S("a"), S("b"), S("c")
    wfield = _make_w_field()
    su2 = _make_weak_group(wfield)
    model = Model(
        gauge_groups=(su2,),
        fields=(wfield,),
        lagrangian_decl=S("c3")
        * weak_structure_constant(a, b, c)
        * FieldStrength(su2, mu, nu, a)
        * FieldStrength(su2, nu, rho, b)
        * FieldStrength(su2, rho, mu, c),
    )

    compiled = _compiled_terms(model)
    assert len(compiled) == 27
    assert set(sorted(len(t.fields) for t in compiled)) == {3, 4, 5, 6}

    # Every adjoint slot must be the SU(2) weak adjoint coad(3); a color coad(8)
    # would mean the field strength adjoint was mis-typed.
    color_adjoint = 0
    weak_adjoint = 0
    for term in compiled:
        text = _plain(term.coupling)
        color_adjoint += text.count("coad(8")
        weak_adjoint += text.count("coad(3")
    assert color_adjoint == 0
    assert weak_adjoint > 0


# ---------------------------------------------------------------------------
# 3. Mixed SU(3)/SU(2) raw tensors in one operator.
# ---------------------------------------------------------------------------
def test_mixed_su3_su2_raw_generators_attach_per_group():
    mu, nu = S("mu"), S("nu")
    rho, sigma = S("rho"), S("sigma")
    aC, aW = S("aC"), S("aW")
    sc1, sc2, sc3 = S("sc1"), S("sc2"), S("sc3")
    ic, jc = S("ic"), S("jc")
    sw1, sw2, sw3 = S("sw1"), S("sw2"), S("sw3")
    iw, jw = S("iw"), S("jw")

    gluon = make_gluon(name="G", symbol=S("G0"))
    wfield = _make_w_field()
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    su2 = _make_weak_group(wfield)
    quark = dirac_field(
        "qC",
        indices=(COLOR_FUND_INDEX,),
        symbol=S("qC"),
        conjugate_symbol=S("qCbar"),
    )
    lepton = dirac_field(
        "lW",
        indices=(WEAK_FUND_INDEX,),
        symbol=S("lW"),
        conjugate_symbol=S("lWbar"),
    )

    model = Model(
        gauge_groups=(su3, su2),
        fields=(quark, lepton, gluon, wfield),
        lagrangian_decl=FieldStrength(su3, mu, nu, aC)
        * FieldStrength(su2, rho, sigma, aW)
        * quark.bar
        * gamma_matrix(sc1, sc2, mu)
        * gamma_matrix(sc2, sc3, nu)
        * gauge_generator(aC, ic, jc)
        * quark
        * lepton.bar
        * gamma_matrix(sw1, sw2, rho)
        * gamma_matrix(sw2, sw3, sigma)
        * weak_gauge_generator(aW, iw, jw)
        * lepton,
    )

    compiled = _compiled_terms(model)
    assert compiled  # it lowers at all

    # The color generator must carry the color adjoint (coad 8) with fundamental
    # color legs (cof 3); the weak generator must carry the weak adjoint (coad 3)
    # with weak-doublet legs (cof 2). A crossed assignment would surface as a
    # generator whose adjoint dimension does not match its fundamental legs.
    generator_re = re.compile(r"t\(coad\((\d+),[^)]*\),cof\((\d+),[^)]*\),cof\((\d+),[^)]*\)\)")
    seen = set()
    for term in compiled:
        for adj_dim, l_dim, r_dim in generator_re.findall(_plain(term.coupling)):
            seen.add((adj_dim, l_dim, r_dim))
            if adj_dim == "8":
                assert (l_dim, r_dim) == ("3", "3")  # color generator
            elif adj_dim == "3":
                assert (l_dim, r_dim) == ("2", "2")  # weak-doublet generator
    # Both group generators actually appeared.
    assert ("8", "3", "3") in seen
    assert ("3", "2", "2") in seen


# ---------------------------------------------------------------------------
# 4. Ambiguity must raise, never silently invent or mis-attach.
# ---------------------------------------------------------------------------
def test_repeated_slot_with_single_raw_label_is_ambiguous():
    f = S("f")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS"), gluon.symbol, name="SU3C")
    # A scalar carrying TWO fundamental color slots; an indexed-symbol parameter
    # offers only ONE free fundamental label, so which of the two slots it should
    # claim is genuinely ambiguous and must raise rather than guess.
    bi_color = Field(
        "BiC",
        spin=0,
        self_conjugate=True,
        symbol=S("biC"),
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )
    y = Parameter("Yc", indices=(COLOR_FUND_INDEX,))

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        _compiled_terms(
            Model(
                gauge_groups=(su3,),
                fields=(bi_color, gluon),
                parameters=(y,),
                lagrangian_decl=y(f) * bi_color,
            )
        )
