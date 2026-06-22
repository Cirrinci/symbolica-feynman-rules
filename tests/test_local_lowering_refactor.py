from __future__ import annotations

import pytest
from symbolica import Expression, S

from feynpy import COLOR_FUND_INDEX, Gamma, Model, PartialD, T, dirac_field, flavor_index, scalar_field
from feynpy.lowering import (
    _LocalChainBinding,
    _LocalFieldEntry,
    _LocalLoweringState,
    _LocalSlotRef,
    _ParsedLocalMonomial,
    _build_local_resolved_bindings,
    _identify_local_contraction_pairs,
)
from tests.support.builders import make_photon


def _label_key(label) -> str:
    return label.to_canonical_string() if hasattr(label, "to_canonical_string") else str(label)


def _normalized_lowering_signature(interaction):
    replacements: dict[str, object] = {}
    ordered_symbolic_replacements: list[tuple[object, object]] = []

    def normalize_label(label):
        key = _label_key(label)
        replacement = replacements.get(key)
        if replacement is None:
            replacement = S(f"norm_label_{len(replacements) + 1}")
            replacements[key] = replacement
            if hasattr(label, "to_atom_tree"):
                ordered_symbolic_replacements.append((label, replacement))
        return replacement

    fields = tuple(
        (
            occurrence.field.name,
            occurrence.conjugated,
            tuple(normalize_label(label) for label in occurrence.slot_labels.values),
        )
        for occurrence in interaction.fields
    )
    derivatives = tuple(
        (action.target, normalize_label(action.lorentz_index))
        for action in interaction.derivatives
    )

    coupling = interaction.coupling
    for original, replacement in ordered_symbolic_replacements:
        coupling = coupling.replace(original, replacement)

    return (
        coupling.expand().to_canonical_string(),
        fields,
        derivatives,
        interaction.closed_dirac_bilinears,
        interaction.sector,
        interaction.origin,
    )


def _lower_local(expr):
    lagrangian = Model(expr).lagrangian()
    assert len(lagrangian.terms) == 1
    return lagrangian.terms[0]


def test_local_lowering_bilinear_is_dummy_rename_invariant():
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    i, j = S("i"), S("j")

    first = _lower_local(S("m") * phi.bar(i) * phi(i))
    second = _lower_local(S("m") * phi.bar(j) * phi(j))

    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)


def test_local_lowering_quartic_is_dummy_rename_invariant():
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    i, j, a, b = S("i", "j", "a", "b")

    first = _lower_local(
        -S("lam") * phi.bar(i) * phi(i) * phi.bar(j) * phi(j)
    )
    second = _lower_local(
        -S("lam") * phi.bar(a) * phi(a) * phi.bar(b) * phi(b)
    )

    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)


def test_local_lowering_repeated_index_kind_pairs_slots_positionally():
    phi = scalar_field(
        "Phi",
        self_conjugate=False,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )

    interaction = _lower_local(S("m") * phi.bar * phi)

    left_labels = interaction.fields[0].slot_labels.values
    right_labels = interaction.fields[1].slot_labels.values

    assert left_labels[0] == right_labels[0]
    assert left_labels[1] == right_labels[1]
    assert left_labels[0] != left_labels[1]


def test_local_lowering_manual_gauge_fixing_is_dummy_rename_invariant():
    photon = make_photon(name="A", symbol=S("A0"))
    xi = S("xi_local")

    first = _lower_local(
        -(Expression.num(1) / (Expression.num(2) * xi))
        * PartialD(photon(S("alpha_label")), S("alpha_label"))
        * PartialD(photon(S("beta_label")), S("beta_label"))
    )
    second = _lower_local(
        -(Expression.num(1) / (Expression.num(2) * xi))
        * PartialD(photon(S("mu_anything")), S("mu_anything"))
        * PartialD(photon(S("nu_anything")), S("nu_anything"))
    )

    assert first.sector == second.sector == "gauge_fixing"
    assert first.origin == second.origin == "manual_gauge_fixing"
    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)


def test_local_lowering_explicit_and_implicit_yukawa_match():
    phi = scalar_field("PhiY", self_conjugate=True)
    psi = dirac_field("psiY", indices=(COLOR_FUND_INDEX,))
    eta = dirac_field("etaY", indices=(COLOR_FUND_INDEX,))
    i = S("i")

    explicit = _lower_local(
        S("y") * phi * psi.bar(i) * eta(i)
    )
    implicit = _lower_local(
        S("y") * phi * psi.bar * eta
    )

    assert _normalized_lowering_signature(explicit) == _normalized_lowering_signature(implicit)


def test_local_lowering_preserves_raw_indexed_yukawa_function():
    generation = flavor_index("GenerationY", dimension=3, prefix="f")
    phi = scalar_field("PhiYraw", self_conjugate=True)
    psi = dirac_field("psiYraw", indices=(generation,))
    eta = dirac_field("etaYraw", indices=(generation,))
    f1, f2 = S("f1"), S("f2")

    interaction = _lower_local(
        S("Y")(f1, f2)
        * phi
        * psi.bar(index_labels={generation.kind: f1})
        * eta(index_labels={generation.kind: f2})
    )

    assert "Y" in interaction.coupling.to_canonical_string()
    assert len(interaction.closed_dirac_bilinears) == 1


def test_local_lowering_gamma_chain_bilinear_is_dummy_rename_invariant():
    psi = dirac_field("psiChain")
    mu, nu = S("mu"), S("nu")

    first = _lower_local(
        S("g") * psi.bar * Gamma(mu) * psi
    )
    second = _lower_local(
        S("g") * psi.bar * Gamma(nu) * psi
    )

    assert first.closed_dirac_bilinears == second.closed_dirac_bilinears == ((0, 1),)
    assert _normalized_lowering_signature(first)[1:] == _normalized_lowering_signature(second)[1:]


def test_local_lowering_repeated_kind_chain_attachment_requires_explicit_labels():
    psi = dirac_field(
        "psiRep",
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )

    with pytest.raises(ValueError, match="repeated color_fund slots"):
        _lower_local(
            S("g") * psi.bar * T(S("a")) * psi
        )


def test_local_lowering_rejects_conflicting_chain_and_resolved_fermion_pairings():
    psi_bar = dirac_field("psiBarConflict")
    psi = dirac_field("psiConflict")
    chi = dirac_field("chiConflict")
    alpha, beta = S("alpha"), S("beta")

    parsed = _ParsedLocalMonomial(
        field_entries=(
            _LocalFieldEntry(
                field=psi_bar,
                conjugated=True,
                derivative_indices=(),
                labels={"spinor": alpha},
            ),
            _LocalFieldEntry(
                field=psi,
                conjugated=False,
                derivative_indices=(),
                labels={"spinor": beta},
            ),
            _LocalFieldEntry(
                field=chi,
                conjugated=False,
                derivative_indices=(),
                labels={"spinor": alpha},
            ),
        ),
        declared_factors=(),
        free_tensor_factors=(),
        interval_chain_factors=((), ()),
    )
    state = _LocalLoweringState(
        parsed=parsed,
        typed_index_labels=(),
        coupling=Expression.num(1),
        slot_labels=[{0: alpha}, {0: beta}, {0: alpha}],
        explicit_slot_labels=[{0: alpha}, {0: beta}, {0: alpha}],
        counters={},
        chain_bindings=[
            _LocalChainBinding(
                kind="spinor",
                left=_LocalSlotRef(field_idx=0, slot=0),
                right=_LocalSlotRef(field_idx=1, slot=0),
                factors=(),
            )
        ],
    )
    state.resolved_bindings = _build_local_resolved_bindings(
        state,
        slot_labels=state.slot_labels,
    )

    with pytest.raises(ValueError, match="Inconsistent local fermion pairing"):
        _identify_local_contraction_pairs(state)
