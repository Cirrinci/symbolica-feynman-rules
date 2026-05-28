from __future__ import annotations

import pytest
from symbolica import Expression, S

from lagrangian.ibp import _occurrence_key
from lagrangian.operator_action import _occurrence_cache_key
from model import (
    COLOR_FUND_INDEX,
    DerivativeAction,
    DerivativeRef,
    DiracBilinear,
    Field,
    InteractionTerm,
    LORENTZ_INDEX,
    SlotRef,
    dirac_field,
    scalar_field,
)


def test_field_occurrence_slot_labels_roundtrip_repeated_kind():
    phi = scalar_field(
        "Phi",
        self_conjugate=True,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )
    i, j, k = S("i", "j", "k")

    occurrence = phi.occurrence(labels={"color_fund": (i, j)})
    assert occurrence.slot_labels.values == (i, j)
    assert occurrence.labels == {"color_fund": (i, j)}

    updated = occurrence.with_slot_label(1, k)
    assert updated.slot_labels.values == (i, k)
    assert updated.labels == {"color_fund": (i, k)}


def test_ibp_occurrence_key_is_invariant_under_input_label_dict_order():
    field = Field(
        "X",
        spin=0,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_FUND_INDEX),
    )
    mu, i = S("mu"), S("i")

    left = field.occurrence(labels={"lorentz": mu, "color_fund": i})
    right = field.occurrence(labels={"color_fund": i, "lorentz": mu})

    assert left.slot_labels.values == (mu, i)
    assert right.slot_labels.values == (mu, i)
    assert _occurrence_key(left) == _occurrence_key(right)


def test_occurrence_cache_key_is_invariant_under_partial_label_normalization():
    field = Field(
        "X",
        spin=0,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_FUND_INDEX),
    )
    i = S("i")

    left = field.occurrence(labels={"color_fund": i})
    right = field.occurrence(labels={"lorentz": None, "color_fund": i})

    assert left.slot_labels.values == (None, i)
    assert right.slot_labels.values == (None, i)
    assert _occurrence_cache_key(left) == _occurrence_cache_key(right)


def test_interaction_term_index_bindings_track_field_slots_and_derivatives():
    A = Field("A", spin=1, indices=(LORENTZ_INDEX,), self_conjugate=True)
    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(A(mu),),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
    )

    bindings = term.index_bindings
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.label == mu
    assert binding.field_slots == (SlotRef(occurrence=0, slot=0),)
    assert binding.derivatives == (DerivativeRef(ordinal=0, target=0),)
    assert binding.multiplicity == 2
    assert binding.is_open is False


def test_structural_dirac_bilinears_bridge_to_legacy_vertex_kwargs():
    psi = dirac_field("psi")
    alpha = S("alpha")
    p_in, p_out = S("p_in", "p_out")
    spinor_slot = 0

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True, labels={"spinor": alpha}),
            psi.occurrence(labels={"spinor": alpha}),
        ),
        dirac_bilinears=(
            DiracBilinear(
                psibar=SlotRef(occurrence=0, slot=spinor_slot),
                psi=SlotRef(occurrence=1, slot=spinor_slot),
            ),
        ),
    )

    assert term.closed_dirac_bilinears == ((0, 1),)

    kwargs = term.to_vertex_kwargs(
        (
            psi.leg(p_in, conjugated=True, labels={"spinor": S("i1")}),
            psi.leg(p_out, labels={"spinor": S("i2")}),
        )
    )

    assert kwargs["closed_dirac_bilinears"] == ((0, 1),)
    assert kwargs["field_index_labels"] == [{"spinor": alpha}, {"spinor": alpha}]


def test_interaction_term_rejects_out_of_bounds_structural_dirac_bilinear():
    psi = dirac_field("psi")

    with pytest.raises(ValueError, match="outside the interaction arity"):
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(psi.occurrence(conjugated=True), psi.occurrence()),
            dirac_bilinears=(
                DiracBilinear(
                    psibar=SlotRef(occurrence=2, slot=0),
                    psi=SlotRef(occurrence=1, slot=0),
                ),
            ),
        )


def test_interaction_term_rejects_structural_dirac_bilinear_with_wrong_endpoint():
    psi = dirac_field("psi")

    with pytest.raises(ValueError, match="psibar endpoint must point to a conjugated"):
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(psi.occurrence(), psi.occurrence(conjugated=True)),
            dirac_bilinears=(
                DiracBilinear(
                    psibar=SlotRef(occurrence=0, slot=0),
                    psi=SlotRef(occurrence=1, slot=0),
                ),
            ),
        )


def test_interaction_term_rejects_structural_dirac_bilinear_with_wrong_slot():
    psi = dirac_field("psi")

    with pytest.raises(ValueError, match="must target spinor slot 0, got 1"):
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(psi.occurrence(conjugated=True), psi.occurrence()),
            dirac_bilinears=(
                DiracBilinear(
                    psibar=SlotRef(occurrence=0, slot=1),
                    psi=SlotRef(occurrence=1, slot=0),
                ),
            ),
        )
