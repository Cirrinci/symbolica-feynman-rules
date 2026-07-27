"""Core regression tests for labelled covariant-derivative lowering."""

from fractions import Fraction

from symbolica import S

from feynpy import (
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    CovD,
    Field,
    Gamma,
    GaugeGroup,
    Model,
    flavor_index,
)
from feynpy.lowering import _analyze_declared_source_term
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
    declaration = (
        I
        * alpha(f1, f2)
        * fermion.bar(index_labels={generation.kind: f1})
        * Gamma(mu)
        * CovD(fermion(index_labels={generation.kind: f2}), mu)
    )
    analyzed = _analyze_declared_source_term(declaration)
    assert analyzed is not None
    assert analyzed.covariant_core is not None
    assert analyzed.generic_covariant_monomial is None
    assert analyzed.covariant_core.left_labels == {generation.kind: f1}
    assert analyzed.covariant_core.right_labels == {generation.kind: f2}

    model = Model(
        gauge_groups=(_u1_model(fermion=fermion, photon=photon),),
        fields=(fermion, photon),
        lagrangian_decl=declaration,
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


def test_one_sided_labelled_dirac_covd_stays_on_generic_path():
    generation = flavor_index("Generation", 3, prefix="f")
    f1, mu = S("f1"), S("mu")
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
    declaration = (
        I
        * alpha(f1)
        * fermion.bar(index_labels={generation.kind: f1})
        * Gamma(mu)
        * CovD(fermion, mu)
    )

    analyzed = _analyze_declared_source_term(declaration)
    assert analyzed is not None
    assert analyzed.covariant_core is None
    assert analyzed.generic_covariant_monomial is declaration


def test_partly_labelled_dirac_covd_with_unlabelled_matter_slot_stays_generic():
    generation = flavor_index("Generation", 3, prefix="f")
    f1, f2, mu = S("f1"), S("f2"), S("mu")
    alpha = S("alphaK")

    fermion = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX, generation),
        quantum_numbers={"Q": S("qPsi")},
    )
    declaration = (
        I
        * alpha(f1, f2)
        * fermion.bar(index_labels={generation.kind: f1})
        * Gamma(mu)
        * CovD(fermion(index_labels={generation.kind: f2}), mu)
    )

    analyzed = _analyze_declared_source_term(declaration)
    assert analyzed is not None
    assert analyzed.covariant_core is None
    assert analyzed.generic_covariant_monomial is declaration


def test_labelled_dirac_covd_with_explicit_nonflavor_slot_stays_generic():
    generation = flavor_index("Generation", 3, prefix="f")
    f1, f2, c1, mu = S("f1"), S("f2"), S("c1"), S("mu")
    alpha = S("alphaK")

    fermion = Field(
        "q",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("q"),
        conjugate_symbol=S("qbar"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX, generation),
        quantum_numbers={"Q": S("qPsi")},
    )
    declaration = (
        I
        * alpha(f1, f2)
        * fermion.bar(index_labels={COLOR_FUND_INDEX.kind: c1, generation.kind: f1})
        * Gamma(mu)
        * CovD(
            fermion(index_labels={COLOR_FUND_INDEX.kind: c1, generation.kind: f2}),
            mu,
        )
    )

    analyzed = _analyze_declared_source_term(declaration)
    assert analyzed is not None
    assert analyzed.covariant_core is None
    assert analyzed.generic_covariant_monomial is declaration


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
