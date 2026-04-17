import sys
from pathlib import Path


# Allow importing from repo `src/` without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from compiler.covariant import (  # noqa: E402
    compile_covariant_terms,
    compile_mixed_complex_scalar_contact_terms,
)
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    CovD,
    LORENTZ_KIND,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    Model,
)
from symbolic.vertex_engine import Delta, I, pi, simplify_deltas, vertex_factor  # noqa: E402
from lagrangian.operators import scalar_gauge_contact  # noqa: E402
from symbolic.spenso_structures import gauge_generator, structure_constant  # noqa: E402


def _model_vertex(*, interaction, external_legs, species_map):
    x = S("x")
    d = S("d")
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=x,
        d=d,
    )
    return simplify_deltas(expr, species_map=species_map)


def _scalar_decl(field):
    return CovD(field.bar, S("mu_decl")) * CovD(field, S("mu_decl"))


def test_compile_mixed_complex_scalar_contact_terms_abelian_nonabelian():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2 = S("c1", "c2")
    a3 = S("a3")
    gS = S("gS")
    eQED = S("eQED")
    qPhi = S("qPhi")
    phi = S("phi")
    phidag = S("phidag")
    G = S("G")
    A = S("A")

    scalar = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX,),
        quantum_numbers={"Q": qPhi},
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=G,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=A,
        indices=(LORENTZ_INDEX,),
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
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=eQED,
        gauge_boson=photon.symbol,
        charge="Q",
    )

    compiled = compile_mixed_complex_scalar_contact_terms(
        scalar=scalar,
        left_gauge_group=su3,
        left_gauge_field=gluon,
        right_gauge_group=u1,
        right_gauge_field=photon,
    )
    assert len(compiled) == 1
    assert compiled[0].label == "SU3 x U1: scalar mixed contact [slot 1]"

    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        photon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4}),
    )
    species_map = {b1: phidag, b2: phi, b3: G, b4: A}
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map=species_map,
    )

    expected = (
        I
        * gS
        * eQED
        * qPhi
        * gauge_generator(a3, c1, c2)
        * scalar_gauge_contact(mu3, mu4)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_compile_mixed_complex_scalar_contact_terms_nonabelian_nonabelian_same_slot():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2 = S("c1", "c2")
    a3, a4 = S("a3", "a4")
    gL = S("gL")
    gR = S("gR")
    phi = S("phi")
    phidag = S("phidag")
    L = S("L")
    R = S("R")

    scalar = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX,),
    )
    left_boson = Field(
        "L",
        spin=1,
        self_conjugate=True,
        symbol=L,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    right_boson = Field(
        "R",
        spin=1,
        self_conjugate=True,
        symbol=R,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    left_group = GaugeGroup(
        name="GL",
        abelian=False,
        coupling=gL,
        gauge_boson=left_boson.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fundL",
            ),
        ),
    )
    right_group = GaugeGroup(
        name="GR",
        abelian=False,
        coupling=gR,
        gauge_boson=right_boson.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fundR",
            ),
        ),
    )

    compiled = compile_mixed_complex_scalar_contact_terms(
        scalar=scalar,
        left_gauge_group=left_group,
        left_gauge_field=left_boson,
        right_gauge_group=right_group,
        right_gauge_field=right_boson,
    )
    assert len(compiled) == 1
    assert compiled[0].label == "GL x GR: scalar mixed contact [slot 1]"

    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        left_boson.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        right_boson.leg(p4, species=b4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    species_map = {b1: phidag, b2: phi, b3: L, b4: R}
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map=species_map,
    )

    expected = (
        I
        * gL
        * gR
        * gauge_generator(a3, c1, S("c_mid_Phi_GL_GR"))
        * gauge_generator(a4, S("c_mid_Phi_GL_GR"), c2)
        * scalar_gauge_contact(mu3, mu4)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_compile_mixed_complex_scalar_contact_terms_nonabelian_nonabelian_cross_slot():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2, c3, c4 = S("c1", "c2", "c3", "c4")
    a3, a4 = S("a3", "a4")
    gL = S("gL")
    gR = S("gR")
    phi = S("phi")
    phidag = S("phidag")
    L = S("L")
    R = S("R")

    scalar = Field(
        "PhiBi",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )
    left_boson = Field(
        "L",
        spin=1,
        self_conjugate=True,
        symbol=L,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    right_boson = Field(
        "R",
        spin=1,
        self_conjugate=True,
        symbol=R,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    left_group = GaugeGroup(
        name="GL",
        abelian=False,
        coupling=gL,
        gauge_boson=left_boson.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fundL",
                slot=0,
            ),
        ),
    )
    right_group = GaugeGroup(
        name="GR",
        abelian=False,
        coupling=gR,
        gauge_boson=right_boson.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fundR",
                slot=1,
            ),
        ),
    )

    compiled = compile_mixed_complex_scalar_contact_terms(
        scalar=scalar,
        left_gauge_group=left_group,
        left_gauge_field=left_boson,
        right_gauge_group=right_group,
        right_gauge_field=right_boson,
    )
    assert len(compiled) == 1
    assert compiled[0].label == "GL x GR: scalar mixed contact [slots 1, 2]"

    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: (c1, c3)}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: (c2, c4)}),
        left_boson.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        right_boson.leg(p4, species=b4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    species_map = {b1: phidag, b2: phi, b3: L, b4: R}
    got = _model_vertex(
        interaction=compiled[0],
        external_legs=legs,
        species_map=species_map,
    )

    expected = (
        I
        * gL
        * gR
        * gauge_generator(a3, c1, c2)
        * gauge_generator(a4, c3, c4)
        * scalar_gauge_contact(mu3, mu4)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_mixed_scalar_covariant_term_includes_cross_group_contact():
    x = S("x")
    del x  # kept to mirror the local symbolic conventions
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    mu3, mu4 = S("mu3", "mu4")
    c1, c2 = S("c1", "c2")
    a3 = S("a3")
    gS = S("gS")
    eQED = S("eQED")
    qPhi = S("qPhi")
    phi = S("phiMix")
    phidag = S("phiMixdag")
    G = S("G")
    A = S("A")

    scalar = Field(
        "PhiMix",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX,),
        quantum_numbers={"Q": qPhi},
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=G,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=A,
        indices=(LORENTZ_INDEX,),
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
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=eQED,
        gauge_boson=photon.symbol,
        charge="Q",
    )
    model = Model(
        name="mixed-scalar",
        gauge_groups=(su3, u1),
        fields=(scalar, gluon, photon),
        lagrangian_decl=_scalar_decl(scalar),
    )

    compiled = compile_covariant_terms(model)
    assert len(compiled) == 8

    qcd_currents = [term for term in compiled if "SU3: scalar current" in term.label]
    qed_currents = [term for term in compiled if "U1: scalar current" in term.label]
    mixed_contacts = [term for term in compiled if "mixed contact" in term.label]

    assert len(qcd_currents) == 2
    assert len(qed_currents) == 2
    assert len(mixed_contacts) == 2

    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: c1}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: c2}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        photon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4}),
    )
    species_map = {b1: phidag, b2: phi, b3: G, b4: A}
    got = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=legs,
                species_map=species_map,
            )
            for term in mixed_contacts
        ),
        Expression.num(0),
    )

    expected = (
        2
        * I
        * gS
        * eQED
        * qPhi
        * gauge_generator(a3, c1, c2)
        * scalar_gauge_contact(mu3, mu4)
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()
