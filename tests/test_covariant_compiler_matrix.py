import sys
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from gauge_compiler import compile_covariant_terms, with_compiled_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    Field,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    Model,
)
from model_symbolica import Delta, I, pi, pcomp, simplify_deltas, vertex_factor  # noqa: E402
from operators import (  # noqa: E402
    gauge_kinetic_bilinear_raw,
    psi_bar_gamma_psi,
    quark_gluon_current,
    scalar_gauge_contact,
    yang_mills_four_vertex_raw,
    yang_mills_three_vertex_metric_raw,
)
from spenso_structures import gauge_generator, structure_constant, simplify_gamma_chain  # noqa: E402


def _model_vertex(*, interaction, external_legs, species_map):
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=S("x"),
        d=S("d"),
    )
    return simplify_deltas(expr, species_map=species_map)


def _symmetrized_generator_contact(adj_left, adj_right, color_left, color_right, color_middle):
    return (
        gauge_generator(adj_left, color_left, color_middle)
        * gauge_generator(adj_right, color_middle, color_right)
        + gauge_generator(adj_right, color_left, color_middle)
        * gauge_generator(adj_left, color_middle, color_right)
    )


def _make_photon(symbol):
    return Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=symbol,
        indices=(LORENTZ_INDEX,),
    )


def _make_gluon(symbol):
    return Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=symbol,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )


def _make_u1_group(*, coupling, gauge_boson, name="U1", charge="Q"):
    return GaugeGroup(
        name=name,
        abelian=True,
        coupling=coupling,
        gauge_boson=gauge_boson,
        charge=charge,
    )


def _make_su3_group(*, coupling, gauge_boson, name="SU3"):
    return GaugeGroup(
        name=name,
        abelian=False,
        coupling=coupling,
        gauge_boson=gauge_boson,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )


def test_covariant_dirac_qcd_current():
    d = S("d")
    p1, p2, p3 = S("p1", "p2", "p3")
    b1, b2, b3 = S("b1", "b2", "b3")
    i1, i2 = S("i1", "i2")
    mu3 = S("mu3")
    c1, c2 = S("c1", "c2")
    a3 = S("a3")
    gS = S("gS")
    psi = S("psi")
    psibar = S("psibar")
    gluon_symbol = S("G")

    quark = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=psi,
        conjugate_symbol=psibar,
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    gluon = _make_gluon(gluon_symbol)
    su3 = _make_su3_group(coupling=gS, gauge_boson=gluon.symbol)
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon),
        covariant_terms=(DiracKineticTerm(field=quark),),
    )

    compiled = compile_covariant_terms(model)
    assert with_compiled_covariant_terms(model).interactions == compiled
    assert len(compiled) == 1

    legs = (
        quark.leg(p1, conjugated=True, species=b1, labels={"spinor": i1, COLOR_FUND_KIND: c1}),
        quark.leg(p2, species=b2, labels={"spinor": i2, COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    )
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map={b1: psibar, b2: psi, b3: gluon_symbol},
    )
    expected = -I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * (2 * pi) ** d * Delta(p1 + p2 + p3)
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_covariant_dirac_qed_current():
    d = S("d")
    p1, p2, p3 = S("p1", "p2", "p3")
    b1, b2, b3 = S("b1", "b2", "b3")
    i1, i2 = S("i1", "i2")
    mu3 = S("mu3")
    eQED = S("eQED")
    qPsi = S("qPsi")
    psi = S("psi")
    psibar = S("psibar")
    photon_symbol = S("A")

    fermion = Field(
        "PsiQED",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=psi,
        conjugate_symbol=psibar,
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": qPsi},
    )
    photon = _make_photon(photon_symbol)
    u1 = _make_u1_group(coupling=eQED, gauge_boson=photon.symbol)
    model = Model(
        gauge_groups=(u1,),
        fields=(fermion, photon),
        covariant_terms=(DiracKineticTerm(field=fermion),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 1

    legs = (
        fermion.leg(p1, conjugated=True, species=b1, labels={"spinor": i1}),
        fermion.leg(p2, species=b2, labels={"spinor": i2}),
        photon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3}),
    )
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map={b1: psibar, b2: psi, b3: photon_symbol},
    )
    expected = -I * eQED * qPsi * psi_bar_gamma_psi(i1, i2, mu3) * (2 * pi) ** d * Delta(p1 + p2 + p3)
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_covariant_mixed_fermion_expands_over_qcd_and_qed():
    d = S("d")
    p1, p2, p3 = S("p1", "p2", "p3")
    b1, b2, b3 = S("b1", "b2", "b3")
    i1, i2 = S("i1", "i2")
    mu3 = S("mu3")
    c1, c2 = S("c1", "c2")
    a3 = S("a3")
    gS = S("gS")
    eQED = S("eQED")
    qMix = S("qMix")
    psi = S("psiMix")
    psibar = S("psibarMix")
    gluon_symbol = S("G")
    photon_symbol = S("A")

    fermion = Field(
        "PsiMix",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=psi,
        conjugate_symbol=psibar,
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
        quantum_numbers={"Q": qMix},
    )
    gluon = _make_gluon(gluon_symbol)
    photon = _make_photon(photon_symbol)
    su3 = _make_su3_group(coupling=gS, gauge_boson=gluon.symbol, name="SU3C")
    u1 = _make_u1_group(coupling=eQED, gauge_boson=photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(su3, u1),
        fields=(fermion, gluon, photon),
        covariant_terms=(DiracKineticTerm(field=fermion),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2

    # The compiler emits contributions in model.gauge_groups order.
    qcd_term, qed_term = compiled

    qcd_legs = (
        fermion.leg(p1, conjugated=True, species=b1, labels={"spinor": i1, COLOR_FUND_KIND: c1}),
        fermion.leg(p2, species=b2, labels={"spinor": i2, COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    )
    got_qcd = _model_vertex(
        interaction=qcd_term,
        external_legs=qcd_legs,
        species_map={b1: psibar, b2: psi, b3: gluon_symbol},
    )
    expected_qcd = -I * gS * quark_gluon_current(i1, i2, mu3, a3, c1, c2) * (2 * pi) ** d * Delta(p1 + p2 + p3)
    assert got_qcd.expand().to_canonical_string() == expected_qcd.expand().to_canonical_string()

    qed_legs = (
        fermion.leg(p1, conjugated=True, species=b1, labels={"spinor": i1, COLOR_FUND_KIND: c1}),
        fermion.leg(p2, species=b2, labels={"spinor": i2, COLOR_FUND_KIND: c2}),
        photon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3}),
    )
    got_qed = _model_vertex(
        interaction=qed_term,
        external_legs=qed_legs,
        species_map={b1: psibar, b2: psi, b3: photon_symbol},
    )
    expected_qed = (
        -I
        * eQED
        * qMix
        * psi_bar_gamma_psi(i1, i2, mu3)
        * COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_qed.expand().to_canonical_string() == expected_qed.expand().to_canonical_string()


def test_covariant_scalar_qed_current_and_contact():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    eQED = S("eQED")
    qPhi = S("qPhi")
    phi = S("phi")
    phidag = S("phidag")
    photon_symbol = S("A")

    scalar = Field(
        "PhiQED",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        quantum_numbers={"Q": qPhi},
    )
    photon = _make_photon(photon_symbol)
    u1 = _make_u1_group(coupling=eQED, gauge_boson=photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(scalar, photon),
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 3
    current_plus, current_minus, contact = compiled
    current_index = current_plus.derivatives[0].lorentz_index

    current_legs = (
        scalar.leg(p1, conjugated=True, species=b1),
        scalar.leg(p2, species=b2),
        photon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3}),
    )
    got_current = (
        _model_vertex(
            interaction=current_plus,
            external_legs=current_legs,
            species_map={b1: phidag, b2: phi, b3: photon_symbol},
        )
        + _model_vertex(
            interaction=current_minus,
            external_legs=current_legs,
            species_map={b1: phidag, b2: phi, b3: photon_symbol},
        )
    )
    expected_current = I * eQED * qPhi * (pcomp(p2, current_index) - pcomp(p1, current_index)) * (2 * pi) ** d * Delta(p1 + p2 + p3)
    assert got_current.expand().to_canonical_string() == expected_current.expand().to_canonical_string()

    contact_legs = (
        scalar.leg(p1, conjugated=True, species=b1),
        scalar.leg(p2, species=b2),
        photon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3}),
        photon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4}),
    )
    got_contact = _model_vertex(
        interaction=contact,
        external_legs=contact_legs,
        species_map={b1: phidag, b2: phi, b3: photon_symbol, b4: photon_symbol},
    )
    expected_contact = 2 * I * (eQED ** 2) * (qPhi ** 2) * scalar_gauge_contact(mu3, mu4) * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert got_contact.expand().to_canonical_string() == expected_contact.expand().to_canonical_string()


def test_covariant_scalar_qcd_current_and_contact():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2 = S("c1", "c2")
    a3, a4 = S("a3", "a4")
    gS = S("gS")
    phi = S("phi")
    phidag = S("phidag")
    gluon_symbol = S("G")

    scalar = Field(
        "PhiQCD",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX,),
    )
    gluon = _make_gluon(gluon_symbol)
    su3 = _make_su3_group(coupling=gS, gauge_boson=gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(scalar, gluon),
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 3
    current_plus, current_minus, contact = compiled
    current_index = current_plus.derivatives[0].lorentz_index

    current_legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    )
    got_current = (
        _model_vertex(
            interaction=current_plus,
            external_legs=current_legs,
            species_map={b1: phidag, b2: phi, b3: gluon_symbol},
        )
        + _model_vertex(
            interaction=current_minus,
            external_legs=current_legs,
            species_map={b1: phidag, b2: phi, b3: gluon_symbol},
        )
    )
    expected_current = (
        I
        * gS
        * gauge_generator(a3, c1, c2)
        * (pcomp(p2, current_index) - pcomp(p1, current_index))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_current.expand().to_canonical_string() == expected_current.expand().to_canonical_string()

    contact_legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        gluon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    got_contact = _model_vertex(
        interaction=contact,
        external_legs=contact_legs,
        species_map={b1: phidag, b2: phi, b3: gluon_symbol, b4: gluon_symbol},
    )
    expected_contact = (
        I
        * (gS ** 2)
        * scalar_gauge_contact(mu3, mu4)
        * _symmetrized_generator_contact(a3, a4, c1, c2, S("c_mid_PhiQCD_SU3C"))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got_contact.expand().to_canonical_string() == expected_contact.expand().to_canonical_string()


def test_covariant_mixed_scalar_currents_and_contact():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2 = S("c1", "c2")
    a3 = S("a3")
    gS = S("gS")
    eQED = S("eQED")
    qPhiMix = S("qPhiMix")
    phi = S("phiMix")
    phidag = S("phidagMix")
    gluon_symbol = S("G")
    photon_symbol = S("A")

    scalar = Field(
        "PhiMix",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX,),
        quantum_numbers={"Q": qPhiMix},
    )
    gluon = _make_gluon(gluon_symbol)
    photon = _make_photon(photon_symbol)
    su3 = _make_su3_group(coupling=gS, gauge_boson=gluon.symbol, name="SU3C")
    u1 = _make_u1_group(coupling=eQED, gauge_boson=photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(su3, u1),
        fields=(scalar, gluon, photon),
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 8

    qcd_terms = [term for term in compiled if "SU3C: scalar current" in term.label]
    qed_terms = [term for term in compiled if "U1QED: scalar current" in term.label]
    mixed_contact_terms = [term for term in compiled if "mixed contact" in term.label]
    assert len(qcd_terms) == 2
    assert len(qed_terms) == 2
    assert len(mixed_contact_terms) == 2

    qcd_index = qcd_terms[0].derivatives[0].lorentz_index
    qcd_legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    )
    got_qcd = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=qcd_legs,
                species_map={b1: phidag, b2: phi, b3: gluon_symbol},
            )
            for term in qcd_terms
        ),
        Expression.num(0),
    )
    expected_qcd = (
        I
        * gS
        * gauge_generator(a3, c1, c2)
        * (pcomp(p2, qcd_index) - pcomp(p1, qcd_index))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_qcd.expand().to_canonical_string() == expected_qcd.expand().to_canonical_string()

    qed_index = qed_terms[0].derivatives[0].lorentz_index
    qed_legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        photon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3}),
    )
    got_qed = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=qed_legs,
                species_map={b1: phidag, b2: phi, b3: photon_symbol},
            )
            for term in qed_terms
        ),
        Expression.num(0),
    )
    expected_qed = (
        I
        * eQED
        * qPhiMix
        * COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
        * (pcomp(p2, qed_index) - pcomp(p1, qed_index))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_qed.expand().to_canonical_string() == expected_qed.expand().to_canonical_string()

    contact_legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        photon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4}),
    )
    got_contact = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=contact_legs,
                species_map={b1: phidag, b2: phi, b3: gluon_symbol, b4: photon_symbol},
            )
            for term in mixed_contact_terms
        ),
        Expression.num(0),
    )
    expected_contact = (
        2
        * I
        * gS
        * eQED
        * qPhiMix
        * gauge_generator(a3, c1, c2)
        * scalar_gauge_contact(mu3, mu4)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got_contact.expand().to_canonical_string() == expected_contact.expand().to_canonical_string()


def test_covariant_abelian_gauge_bilinear():
    d = S("d")
    p1, p2 = S("p1", "p2")
    b1, b2 = S("b1", "b2")
    mu3, mu4 = S("mu3", "mu4")
    photon_symbol = S("A")

    photon = _make_photon(photon_symbol)
    u1 = _make_u1_group(coupling=S("eQED"), gauge_boson=photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=u1),),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 2
    metric_term, cross_term = compiled
    rho = metric_term.derivatives[0].lorentz_index
    left = cross_term.derivatives[0].lorentz_index
    right = cross_term.derivatives[1].lorentz_index

    legs = (
        photon.leg(p1, species=b1, labels={LORENTZ_KIND: mu3}),
        photon.leg(p2, species=b2, labels={LORENTZ_KIND: mu4}),
    )
    got = simplify_gamma_chain(
        _model_vertex(
            interaction=metric_term,
            external_legs=legs,
            species_map={b1: photon_symbol, b2: photon_symbol},
        )
        + _model_vertex(
            interaction=cross_term,
            external_legs=legs,
            species_map={b1: photon_symbol, b2: photon_symbol},
        )
    )
    expected = I * gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, rho, left, right) * (2 * pi) ** d * Delta(p1 + p2)
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_covariant_yang_mills_bilinear_cubic_and_quartic():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu, nu, rho, sigma = S("mu", "nu", "rho", "sigma")
    mu3, mu4 = S("mu3", "mu4")
    a3, a4, a5, a6 = S("a3", "a4", "a5", "a6")
    gS = S("gS")
    gluon_symbol = S("G")

    gluon = _make_gluon(gluon_symbol)
    su3 = _make_su3_group(coupling=gS, gauge_boson=gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=su3),),
    )

    compiled = compile_covariant_terms(model)
    assert with_compiled_covariant_terms(model).interactions == compiled
    assert len(compiled) == 4
    metric_term, cross_term, cubic_term, quartic_term = compiled

    bilinear_legs = (
        gluon.leg(p1, species=b1, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        gluon.leg(p2, species=b2, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    bilinear_rho = metric_term.derivatives[0].lorentz_index
    bilinear_left = cross_term.derivatives[0].lorentz_index
    bilinear_right = cross_term.derivatives[1].lorentz_index
    got_bilinear = simplify_gamma_chain(
        _model_vertex(
            interaction=metric_term,
            external_legs=bilinear_legs,
            species_map={b1: gluon_symbol, b2: gluon_symbol},
        )
        + _model_vertex(
            interaction=cross_term,
            external_legs=bilinear_legs,
            species_map={b1: gluon_symbol, b2: gluon_symbol},
        )
    )
    expected_bilinear = (
        I
        * gauge_kinetic_bilinear_raw(mu3, mu4, p1, p2, bilinear_rho, bilinear_left, bilinear_right)
        * COLOR_ADJ_INDEX.representation.g(a3, a4).to_expression()
        * (2 * pi) ** d
        * Delta(p1 + p2)
    )
    assert got_bilinear.expand().to_canonical_string() == expected_bilinear.expand().to_canonical_string()

    cubic_rho = cubic_term.derivatives[0].lorentz_index
    cubic_legs = (
        gluon.leg(p1, species=b1, labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a3}),
        gluon.leg(p2, species=b2, labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: a4}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: rho, COLOR_ADJ_KIND: a5}),
    )
    got_cubic = simplify_gamma_chain(
        _model_vertex(
            interaction=cubic_term,
            external_legs=cubic_legs,
            species_map={b1: gluon_symbol, b2: gluon_symbol, b3: gluon_symbol},
        )
    )
    expected_cubic = simplify_gamma_chain(
        gS
        * yang_mills_three_vertex_metric_raw(a3, a4, a5, mu, nu, rho, p1, p2, p3, cubic_rho)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got_cubic.expand().to_canonical_string() == expected_cubic.expand().to_canonical_string()

    quartic_legs = (
        gluon.leg(p1, species=b1, labels={LORENTZ_KIND: mu, COLOR_ADJ_KIND: a3}),
        gluon.leg(p2, species=b2, labels={LORENTZ_KIND: nu, COLOR_ADJ_KIND: a4}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: rho, COLOR_ADJ_KIND: a5}),
        gluon.leg(p4, species=b4, labels={LORENTZ_KIND: sigma, COLOR_ADJ_KIND: a6}),
    )
    got_quartic = _model_vertex(
        interaction=quartic_term,
        external_legs=quartic_legs,
        species_map={b1: gluon_symbol, b2: gluon_symbol, b3: gluon_symbol, b4: gluon_symbol},
    )
    expected_quartic = (
        -I
        * Expression.num(1)
        / Expression.num(2)
        * (gS ** 2)
        * yang_mills_four_vertex_raw(a3, a4, a5, a6, mu, nu, rho, sigma, S("color_adj_mid_G_SU3C"))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got_quartic.expand().to_canonical_string() == expected_quartic.expand().to_canonical_string()
