import sys
from pathlib import Path


# Allow importing from repo `src/` without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from gauge_compiler import compile_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    ComplexScalarKineticTerm,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    Model,
)
from model_symbolica import Delta, I, pi, simplify_deltas, vertex_factor  # noqa: E402
from operators import scalar_gauge_contact  # noqa: E402
from spenso_structures import gauge_generator, structure_constant  # noqa: E402


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
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
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
