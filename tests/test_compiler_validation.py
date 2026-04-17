import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from compiler.covariant import compile_covariant_terms, expand_cov_der  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    CovD,
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    FieldStrength,
    Gamma,
    GaugeKineticTerm,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    Model,
)
from symbolic.vertex_engine import Delta, I, pi, simplify_deltas, vertex_factor  # noqa: E402
from lagrangian.operators import psi_bar_gamma_psi  # noqa: E402
from symbolic.spenso_structures import gauge_generator, structure_constant  # noqa: E402


def _dirac_decl(field, *, gauge_group=None):
    if gauge_group is None:
        return I * field.bar * Gamma(S("mu_decl")) * CovD(field, S("mu_decl"))
    return DiracKineticTerm(field=field, gauge_group=gauge_group)


def _scalar_decl(field, *, gauge_group=None):
    if gauge_group is None:
        return CovD(field.bar, S("mu_decl")) * CovD(field, S("mu_decl"))
    return ComplexScalarKineticTerm(field=field, gauge_group=gauge_group)


def _gauge_decl(group):
    mu_decl, nu_decl = S("mu_decl", "nu_decl")
    return (
        -(Expression.num(1) / Expression.num(4))
        * FieldStrength(group, mu_decl, nu_decl)
        * FieldStrength(group, mu_decl, nu_decl)
    )


def _set_decl(model, item):
    model.lagrangian_decl = type(model.lagrangian_decl).from_item(item)
    model.covariant_terms = ()
    model.gauge_kinetic_terms = ()
    model.gauge_fixing_terms = ()
    model.ghost_terms = ()


def _make_u1_model(*, field_charge):
    psi = Field(
        "Psi",
        spin=1 / 2,
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": field_charge},
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=S("e"),
        gauge_boson=photon.symbol,
        charge="Q",
    )
    return Model(
        gauge_groups=(u1,),
        fields=(psi, photon),
    ), psi, photon, u1


def _make_scalar_u1_model(*, field_charge, self_conjugate=False):
    scalar = Field(
        "PhiU1",
        spin=0,
        self_conjugate=self_conjugate,
        symbol=S("phi_u1"),
        conjugate_symbol=None if self_conjugate else S("phidag_u1"),
        quantum_numbers={"Q": field_charge},
    )
    photon = Field(
        "A",
        spin=1,
        self_conjugate=True,
        symbol=S("A"),
        indices=(LORENTZ_INDEX,),
    )
    u1 = GaugeGroup(
        name="U1",
        abelian=True,
        coupling=S("e"),
        gauge_boson=photon.symbol,
        charge="Q",
    )
    return Model(
        gauge_groups=(u1,),
        fields=(scalar, photon),
    ), scalar, photon, u1


def _make_su3_model():
    scalar = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=S("phi"),
        conjugate_symbol=S("phidag"),
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
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
    return Model(
        gauge_groups=(su3,),
        fields=(scalar, gluon),
    ), scalar, gluon, su3


def test_covariant_dirac_term_rejects_undeclared_field_object():
    model, _, _, _ = _make_u1_model(field_charge=S("qPsi"))
    rogue = Field(
        "Rogue",
        spin=1 / 2,
        self_conjugate=False,
        symbol=S("rogue"),
        conjugate_symbol=S("roguebar"),
        indices=(SPINOR_INDEX,),
        quantum_numbers={"Q": S("qRogue")},
    )
    _set_decl(model, _dirac_decl(rogue))

    with pytest.raises(ValueError, match=r"declared in model\.fields"):
        compile_covariant_terms(model)


def test_gauge_kinetic_term_rejects_undeclared_gauge_group_object():
    model, _, photon, _ = _make_u1_model(field_charge=S("qPsi"))
    rogue_group = GaugeGroup(
        name="RogueU1",
        abelian=True,
        coupling=S("eRogue"),
        gauge_boson=photon.symbol,
        charge="Q",
    )
    _set_decl(model, _gauge_decl(rogue_group))

    with pytest.raises(ValueError, match=r"declared in model\.gauge_groups"):
        compile_covariant_terms(model)


def test_covariant_dirac_term_rejects_undeclared_gauge_boson_field_object():
    model, psi, _, _ = _make_u1_model(field_charge=S("qPsi"))
    rogue_photon = Field(
        "RogueA",
        spin=1,
        self_conjugate=True,
        symbol=S("rogueA"),
        indices=(LORENTZ_INDEX,),
    )
    rogue_group = GaugeGroup(
        name="RogueU1",
        abelian=True,
        coupling=S("eRogue"),
        gauge_boson=rogue_photon,
        charge="Q",
    )
    model.gauge_groups = (rogue_group,)
    model.fields = (psi,)
    _set_decl(model, _dirac_decl(psi))

    with pytest.raises(ValueError, match=r"declared in model\.fields"):
        compile_covariant_terms(model)


def test_explicit_abelian_group_selection_rejects_neutral_field():
    model, psi, _, u1 = _make_u1_model(field_charge=0)
    _set_decl(model, _dirac_decl(psi, gauge_group=u1))

    with pytest.raises(ValueError, match=r"non-zero charge 'Q' under gauge group 'U1'"):
        compile_covariant_terms(model)


def test_expand_cov_der_exposes_group_representation_and_conjugation_metadata():
    model, scalar, _, u1 = _make_scalar_u1_model(field_charge=S("qPhi"))

    expanded = expand_cov_der(model, CovD(scalar.bar, S("mu_decl")))

    assert expanded.field is scalar
    assert expanded.conjugated is True
    assert expanded.derivative_piece.field is scalar
    assert expanded.derivative_piece.conjugated is True
    assert len(expanded.gauge_current_pieces) == 1

    piece = expanded.gauge_current_pieces[0]
    assert piece.metadata.gauge_group is u1
    assert piece.metadata.gauge_field.symbol == S("A")
    assert piece.metadata.representation is None
    assert piece.metadata.representation_name == "Q"
    assert piece.metadata.representation_slots == ()
    assert piece.metadata.repeated_index is False
    assert piece.metadata.conjugated is True
    assert piece.metadata.conjugation_supported is True
    assert piece.metadata.self_conjugate_field is False
    assert expanded.contact_ready_data == expanded.gauge_current_pieces


def test_expand_cov_der_rejects_self_conjugate_field():
    model, scalar, _, _ = _make_scalar_u1_model(field_charge=S("qPhi"), self_conjugate=True)

    with pytest.raises(ValueError, match=r"non-self-conjugate matter fields"):
        expand_cov_der(model, CovD(scalar, S("mu_decl")))


def test_fermion_vertex_rejects_partial_spinor_leg_labels():
    model, psi, photon, _ = _make_u1_model(field_charge=S("qPsi"))
    _set_decl(model, _dirac_decl(psi))
    compiled = compile_covariant_terms(model)

    legs = (
        psi.leg(S("p1"), conjugated=True, labels={"spinor": S("i1")}),
        psi.leg(S("p2")),
        photon.leg(S("p3"), labels={"lorentz": S("mu3")}),
    )

    with pytest.raises(ValueError, match=r"all fermion external legs"):
        vertex_factor(
            interaction=compiled[0],
            external_legs=legs,
            x=S("x"),
            d=S("d"),
        )


def test_fermion_vertex_still_amputates_when_only_bosonic_leg_labels_are_given():
    model, psi, photon, _ = _make_u1_model(field_charge=S("qPsi"))
    _set_decl(model, _dirac_decl(psi))
    compiled = compile_covariant_terms(model)

    p1, p2, p3 = S("p1", "p2", "p3")
    mu3 = S("mu3")
    expr = simplify_deltas(
        vertex_factor(
            interaction=compiled[0],
            external_legs=(
                psi.leg(p1, conjugated=True),
                psi.leg(p2),
                photon.leg(p3, labels={"lorentz": mu3}),
            ),
            x=S("x"),
            d=S("d"),
        ),
        species_map={
            psi.conjugate_symbol: psi.conjugate_symbol,
            psi.symbol: psi.symbol,
            photon.symbol: photon.symbol,
        },
    )

    expected = (
        -I
        * S("e")
        * S("qPsi")
        * psi_bar_gamma_psi(S("i1"), S("i2"), mu3)
        * (2 * pi) ** S("d")
        * Delta(p1 + p2 + p3)
    )
    assert expr.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_explicit_nonabelian_group_selection_rejects_singlet_field():
    model, scalar, _, su3 = _make_su3_model()
    _set_decl(model, _scalar_decl(scalar, gauge_group=su3))

    with pytest.raises(ValueError, match=r"declared representation under gauge group 'SU3'"):
        compile_covariant_terms(model)


def test_nonabelian_field_matching_multiple_representations_is_rejected():
    fermion = Field(
        "PsiMultiRep",
        spin=1 / 2,
        self_conjugate=False,
        symbol=S("psi_multi"),
        conjugate_symbol=S("psibar_multi"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX, COLOR_ADJ_INDEX),
    )
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
            GaugeRepresentation(
                index=COLOR_ADJ_INDEX,
                generator_builder=structure_constant,
                name="adjoint",
            ),
        ),
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(fermion, gluon),
        lagrangian_decl=_dirac_decl(fermion),
    )

    with pytest.raises(ValueError, match=r"matches multiple representations"):
        compile_covariant_terms(model)
