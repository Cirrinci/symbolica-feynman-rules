import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from gauge_compiler import compile_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    Field,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    Model,
)
from spenso_structures import gauge_generator, structure_constant  # noqa: E402


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
    model.covariant_terms = (DiracKineticTerm(field=rogue),)

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
    model.gauge_kinetic_terms = (GaugeKineticTerm(gauge_group=rogue_group),)

    with pytest.raises(ValueError, match=r"declared in model\.gauge_groups"):
        compile_covariant_terms(model)


def test_explicit_abelian_group_selection_rejects_neutral_field():
    model, psi, _, u1 = _make_u1_model(field_charge=0)
    model.covariant_terms = (DiracKineticTerm(field=psi, gauge_group=u1),)

    with pytest.raises(ValueError, match=r"non-zero charge 'Q' under gauge group 'U1'"):
        compile_covariant_terms(model)


def test_explicit_nonabelian_group_selection_rejects_singlet_field():
    model, scalar, _, su3 = _make_su3_model()
    model.covariant_terms = (ComplexScalarKineticTerm(field=scalar, gauge_group=su3),)

    with pytest.raises(ValueError, match=r"declared representation under gauge group 'SU3'"):
        compile_covariant_terms(model)
