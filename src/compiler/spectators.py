"""Internal helpers for spectator labels and spectator-decorated interactions."""

from __future__ import annotations

from symbolica import Expression, S

from model import Field
from model.interactions import InteractionTerm
from model.metadata import is_lorentz_index
from typing import Optional


def _default_index_labels(field: Field, index, qualifier: str = "id", slot: Optional[int] = None):
    stem = f"{field.name}_{index.kind}"
    if slot is not None and field.index_kind_count(index.kind) > 1:
        stem += f"_{slot + 1}"
    stem += f"_{qualifier}"
    return S(f"{index.prefix}_bar_{stem}"), S(f"{index.prefix}_{stem}")


def _spectator_identity_factor(field: Field, *, exclude_slots=()):
    factor = Expression.num(1)
    left_slot_labels = {}
    right_slot_labels = {}

    for slot, index in enumerate(field.indices):
        if slot in exclude_slots or is_lorentz_index(index):
            continue
        left_label, right_label = _default_index_labels(field, index, slot=slot)
        factor *= index.representation.g(left_label, right_label).to_expression()
        left_slot_labels[slot] = left_label
        right_slot_labels[slot] = right_label

    return factor, left_slot_labels, right_slot_labels


def _default_open_index_label(field: Field, index, position: int, slot: int, *, conjugated: bool):
    stem = f"{field.name}_{index.kind}_spect_{position + 1}"
    if field.index_kind_count(index.kind) > 1:
        stem += f"_{slot + 1}"
    if conjugated and not field.self_conjugate:
        return S(f"{index.prefix}_bar_{stem}")
    return S(f"{index.prefix}_{stem}")


def _materialize_spectator_occurrences(spectators: tuple[tuple[Field, bool], ...]):
    """Build spectator occurrences plus their internal contraction factor."""
    if not spectators:
        return Expression.num(1), (), ()

    factor = Expression.num(1)
    slot_labels = [dict() for _ in spectators]
    fermion_bilinears: list[tuple[int, int]] = []
    by_field: dict[int, tuple[Field, dict[bool, list[int]]]] = {}

    for pos, (field, conjugated) in enumerate(spectators):
        field_id = id(field)
        if field_id not in by_field:
            by_field[field_id] = (field, {False: [], True: []})
        by_field[field_id][1][bool(conjugated)].append(pos)

    for field, positions in by_field.values():
        plain_positions = positions[False]
        conj_positions = positions[True]

        if field.kind == "fermion":
            if len(plain_positions) != len(conj_positions):
                raise ValueError(
                    f"Declarative spectator fermions for field {field.name!r} must appear in "
                    "explicit bar/psi pairs. Unpaired or more complicated spinor structures "
                    "are not covered by the modern declarative Model(...) API."
                )
            pair_count = len(plain_positions)
        elif field.self_conjugate:
            pair_count = 0
        else:
            pair_count = min(len(plain_positions), len(conj_positions))

        for pair_idx in range(pair_count):
            left_pos = conj_positions[pair_idx]
            right_pos = plain_positions[pair_idx]
            if field.kind == "fermion":
                fermion_bilinears.append((left_pos, right_pos))
            for slot, index in enumerate(field.indices):
                if is_lorentz_index(index):
                    continue
                left_label, right_label = _default_index_labels(
                    field,
                    index,
                    qualifier=f"spect_{pair_idx + 1}",
                    slot=slot,
                )
                factor *= index.representation.g(left_label, right_label).to_expression()
                slot_labels[left_pos][slot] = left_label
                slot_labels[right_pos][slot] = right_label

    occurrences = []
    for pos, (field, conjugated) in enumerate(spectators):
        pos_slot_labels = slot_labels[pos]
        for slot, index in enumerate(field.indices):
            if slot in pos_slot_labels:
                continue
            pos_slot_labels[slot] = _default_open_index_label(
                field,
                index,
                pos,
                slot,
                conjugated=conjugated,
            )
        occurrences.append(
            field.occurrence(
                conjugated=conjugated,
                labels=field.pack_slot_labels(pos_slot_labels),
            )
        )

    return factor, tuple(occurrences), tuple(fermion_bilinears)


def _decorate_interactions_with_spectators(
    interactions: tuple[InteractionTerm, ...],
    *,
    spectator_factor,
    spectator_occurrences: tuple,
    spectator_bilinears: tuple[tuple[int, int], ...] = (),
) -> tuple[InteractionTerm, ...]:
    return tuple(
        InteractionTerm(
            coupling=interaction.coupling * spectator_factor,
            fields=interaction.fields + spectator_occurrences,
            derivatives=interaction.derivatives,
            closed_dirac_bilinears=interaction.closed_dirac_bilinears
            + tuple(
                (len(interaction.fields) + left, len(interaction.fields) + right)
                for left, right in spectator_bilinears
            ),
            label=interaction.label,
        )
        for interaction in interactions
    )
