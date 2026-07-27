"""Core regression tests for labelled covariant-derivative lowering.

The compact Dirac kinetic-core path regenerates identity labels. When a
monomial already carries explicit flavor (or other representation) labels that
are tied to an indexed coefficient, that path would disconnect the coefficient
from the fields. Labelled Dirac CovD monomials must therefore take the generic
CovD lowering path so the user-written labels survive compilation.
"""

from fractions import Fraction

from symbolica import S

from feynpy import (
    LORENTZ_INDEX,
    SPINOR_INDEX,
    CovD,
    Field,
    Gamma,
    GaugeGroup,
    Model,
    flavor_index,
)
from symbolic.vertex_engine import I


def _u1_model(*, fermion, photon, coupling=S("e")):
    group = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=coupling,
        gauge_boson=photon.symbol,
        charge="Q",
    )
    return group


def test_labelled_dirac_covd_preserves_distinct_generation_labels():
    generation = flavor_index("Generation", 3, prefix="f")
    f1, f2, mu = S("f1"), S("f2"), S("mu")
    alpha = S("alphaK")

    fermion = Field(
        "psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX, generation),
        quantum_numbers={"Q": S("qPsi")},
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    model = Model(
        gauge_groups=(_u1_model(fermion=fermion, photon=photon),),
        fields=(fermion, photon),
        lagrangian_decl=(
            I
            * alpha(f1, f2)
            * fermion.bar(index_labels={generation.kind: f1})
            * Gamma(mu)
            * CovD(fermion(index_labels={generation.kind: f2}), mu)
        ),
    )

    terms = model.lagrangian().terms
    assert len(terms) == 2
    for term in terms:
        generations = [
            occurrence.labels.get("generation")
            for occurrence in term.fields
            if occurrence.field.name == "psi"
        ]
        assert generations == [f1, f2]
        coupling_text = str(term.coupling)
        assert "alphaK" in coupling_text
        assert "f1" in coupling_text and "f2" in coupling_text


def test_unlabelled_dirac_covd_still_compiles_via_compact_core():
    generation = flavor_index("Generation", 3, prefix="f")
    mu = S("mu")

    fermion = Field(
        "psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX, generation),
        quantum_numbers={"Q": S("qPsi")},
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    model = Model(
        gauge_groups=(_u1_model(fermion=fermion, photon=photon),),
        fields=(fermion, photon),
        lagrangian_decl=I * fermion.bar * Gamma(mu) * CovD(fermion, mu),
    )

    terms = model.lagrangian().terms
    assert len(terms) == 2
    # Compact-core path invents distinct identity generation labels rather than
    # preserving user-written ones (there are none).
    for term in terms:
        generations = [
            occurrence.labels.get("generation")
            for occurrence in term.fields
            if occurrence.field.name == "psi"
        ]
        assert len(generations) == 2
        assert generations[0] != generations[1]
