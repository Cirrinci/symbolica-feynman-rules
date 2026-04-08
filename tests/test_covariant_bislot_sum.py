import sys
from pathlib import Path

import pytest


# Allow importing from repo `src/` without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from gauge_compiler import compile_covariant_terms  # noqa: E402
from model import (  # noqa: E402
    COLOR_FUND_INDEX,
    COLOR_ADJ_INDEX,
    LORENTZ_INDEX,
    ComplexScalarKineticTerm,
    Field,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    Model,
)
from spenso_structures import gauge_generator, structure_constant  # noqa: E402


def _make_bislot_scalar():
    phi = S("phi")
    phidag = S("phidag")
    return Field(
        "PhiBiTest",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )


def _make_gluon():
    G = S("G")
    return Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=G,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )


def test_ambiguity_is_error_by_default_for_repeated_slots():
    gS = S("gS")
    scalar = _make_bislot_scalar()
    gluon = _make_gluon()

    rep_unique_default = GaugeRepresentation(
        index=COLOR_FUND_INDEX,
        generator_builder=gauge_generator,
        name="fund",
        # slot_policy defaults to "unique"
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(rep_unique_default,),
    )
    model = Model(
        name="bislot-ambiguous",
        gauge_groups=(su3,),
        fields=(scalar, gluon),
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
    )

    with pytest.raises(ValueError, match=r"repeated index type|slot_policy='sum'|slot=\\.+"):
        compile_covariant_terms(model)


def test_slot_policy_sum_expands_currents_and_contacts_over_slots():
    gS = S("gS")
    scalar = _make_bislot_scalar()
    gluon = _make_gluon()

    rep_sum = GaugeRepresentation(
        index=COLOR_FUND_INDEX,
        generator_builder=gauge_generator,
        name="fund_sum",
        slot_policy="sum",
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(rep_sum,),
    )
    model = Model(
        name="bislot-sum",
        gauge_groups=(su3,),
        fields=(scalar, gluon),
        covariant_terms=(ComplexScalarKineticTerm(field=scalar),),
        gauge_kinetic_terms=(GaugeKineticTerm(gauge_group=su3),),
    )

    compiled = compile_covariant_terms(model)

    # Complex-scalar kinetic term, bislot with slot_policy='sum':
    # - currents: 2 slots * 2 (phi vs phidag derivative placement) = 4 terms
    # - contact: ordered slot pairs (0,0), (0,1), (1,0), (1,1) = 4 terms
    # plus gauge kinetic terms (2 abelian-like bilinear terms + YM cubic + quartic) in this model.
    # We only assert the kinetic-term expansion part here by filtering labels.
    kinetic_terms = [t for t in compiled if "scalar" in t.label]
    assert len(kinetic_terms) == 8

    contact_terms = [t for t in kinetic_terms if "contact" in t.label]
    assert len(contact_terms) == 4

    current_terms = [t for t in kinetic_terms if "current" in t.label]
    assert len(current_terms) == 4

    # Sanity: each contact term must contain two gauge fields in its InteractionTerm fields.
    for term in contact_terms:
        assert len(term.fields) == 4

