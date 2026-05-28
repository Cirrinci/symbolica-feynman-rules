from __future__ import annotations

from symbolica import Expression, S

from model import COLOR_FUND_INDEX, PartialD, scalar_field
from model.lowering import _lower_standalone_lagrangian_source_term
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


def test_local_lowering_bilinear_is_dummy_rename_invariant():
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    i, j = S("i"), S("j")

    first = _lower_standalone_lagrangian_source_term(S("m") * phi.bar(i) * phi(i))
    second = _lower_standalone_lagrangian_source_term(S("m") * phi.bar(j) * phi(j))

    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)


def test_local_lowering_quartic_is_dummy_rename_invariant():
    phi = scalar_field("Phi", self_conjugate=False, indices=(COLOR_FUND_INDEX,))
    i, j, a, b = S("i", "j", "a", "b")

    first = _lower_standalone_lagrangian_source_term(
        -S("lam") * phi.bar(i) * phi(i) * phi.bar(j) * phi(j)
    )
    second = _lower_standalone_lagrangian_source_term(
        -S("lam") * phi.bar(a) * phi(a) * phi.bar(b) * phi(b)
    )

    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)


def test_local_lowering_repeated_index_kind_pairs_slots_positionally():
    phi = scalar_field(
        "Phi",
        self_conjugate=False,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )

    interaction = _lower_standalone_lagrangian_source_term(S("m") * phi.bar * phi)

    left_labels = interaction.fields[0].slot_labels.values
    right_labels = interaction.fields[1].slot_labels.values

    assert left_labels[0] == right_labels[0]
    assert left_labels[1] == right_labels[1]
    assert left_labels[0] != left_labels[1]


def test_local_lowering_manual_gauge_fixing_is_dummy_rename_invariant():
    photon = make_photon(name="A", symbol=S("A0"))
    xi = S("xi_local")

    first = _lower_standalone_lagrangian_source_term(
        -(Expression.num(1) / (Expression.num(2) * xi))
        * PartialD(photon(S("alpha_label")), S("alpha_label"))
        * PartialD(photon(S("beta_label")), S("beta_label"))
    )
    second = _lower_standalone_lagrangian_source_term(
        -(Expression.num(1) / (Expression.num(2) * xi))
        * PartialD(photon(S("mu_anything")), S("mu_anything"))
        * PartialD(photon(S("nu_anything")), S("nu_anything"))
    )

    assert first.sector == second.sector == "gauge_fixing"
    assert first.origin == second.origin == "manual_gauge_fixing"
    assert _normalized_lowering_signature(first) == _normalized_lowering_signature(second)
