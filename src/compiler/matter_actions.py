"""Internal helpers for matter bilinear gauge actions and scalar contacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from lagrangian.operators import psi_bar_gamma_psi, scalar_gauge_contact
from model import DerivativeAction, Field, InteractionTerm

from .spectators import _spectator_identity_factor


@dataclass(frozen=True)
class _BilinearGaugeActionData:
    coupling: object
    left_slot_labels: dict[int, object]
    right_slot_labels: dict[int, object]
    gauge_labels: dict


@dataclass(frozen=True)
class _ScalarContactActionData:
    coupling: object
    scalar_bar_labels: dict
    scalar_labels: dict
    left_gauge_labels: dict
    right_gauge_labels: dict
    active_slots: tuple[int, ...]


def _build_bilinear_gauge_action_data(
    field: Field,
    piece: Any,
    *,
    gauge_action_from_piece: Callable,
    lorentz_label=None,
    matter_labels=None,
    adjoint_label=None,
    spectator_exclude_slots=(),
):
    action = gauge_action_from_piece(
        field,
        piece,
        lorentz_label=lorentz_label,
        adjoint_label=adjoint_label,
        purpose="Covariant gauge-action compilation",
    )
    left_slot_labels: dict[int, object] = {}
    right_slot_labels: dict[int, object] = {}
    exclude_slots: set[int] = set(spectator_exclude_slots)
    coupling = action.coupling

    if action.representation is not None:
        if action.representation_slot is None:
            raise ValueError("Non-abelian gauge actions require an active matter slot.")
        left_label, right_label = matter_labels or action.default_matter_labels()
        coupling *= action.generator(left_label, right_label)
        left_slot_labels[action.representation_slot] = left_label
        right_slot_labels[action.representation_slot] = right_label
        exclude_slots.add(action.representation_slot)

    spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
        field,
        exclude_slots=exclude_slots,
    )
    coupling *= spectator_factor
    left_slot_labels.update(spectator_left_slots)
    right_slot_labels.update(spectator_right_slots)

    return _BilinearGaugeActionData(
        coupling=coupling,
        left_slot_labels=left_slot_labels,
        right_slot_labels=right_slot_labels,
        gauge_labels=action.gauge_labels(),
    )


def _compile_scalar_current_from_piece(
    *,
    scalar: Field,
    piece: Any,
    derivative_target: int,
    gauge_action_from_piece: Callable,
    coupling_prefactor=1,
    label: str = "",
    lorentz_label=None,
    matter_labels=None,
    adjoint_label=None,
) -> InteractionTerm:
    action = _build_bilinear_gauge_action_data(
        scalar,
        piece,
        gauge_action_from_piece=gauge_action_from_piece,
        lorentz_label=lorentz_label,
        matter_labels=matter_labels,
        adjoint_label=adjoint_label,
    )
    derivative_label = lorentz_label or piece.lorentz_index
    sign = -1 if piece.metadata.conjugated else 1
    return InteractionTerm(
        coupling=sign * coupling_prefactor * action.coupling,
        fields=(
            scalar.occurrence(
                conjugated=True,
                labels=scalar.pack_slot_labels(action.left_slot_labels),
            ),
            scalar.occurrence(labels=scalar.pack_slot_labels(action.right_slot_labels)),
            piece.metadata.gauge_field.occurrence(labels=action.gauge_labels),
        ),
        derivatives=(DerivativeAction(target=derivative_target, lorentz_index=derivative_label),),
        label=label,
    )


def _build_fermion_current_interaction(
    *,
    fermion: Field,
    piece: Any,
    spinor_slot: int,
    i_bar,
    i_psi,
    gauge_action_from_piece: Callable,
    slot_suffix: Callable[[Field, Optional[int]], str],
    prefactor=1,
    label: str = "",
    lorentz_label=None,
    matter_labels=None,
    adjoint_label=None,
    spectator_exclude_slots=(),
) -> InteractionTerm:
    action = _build_bilinear_gauge_action_data(
        fermion,
        piece,
        gauge_action_from_piece=gauge_action_from_piece,
        lorentz_label=lorentz_label,
        matter_labels=matter_labels,
        adjoint_label=adjoint_label,
        spectator_exclude_slots=spectator_exclude_slots,
    )
    bar_labels = fermion.pack_slot_labels({
        spinor_slot: i_bar,
        **action.left_slot_labels,
    })
    psi_labels = fermion.pack_slot_labels({
        spinor_slot: i_psi,
        **action.right_slot_labels,
    })

    rep = piece.metadata.representation
    slot_label = slot_suffix(fermion, piece.active_slot)
    label_suffix = f"{label} [{rep.index.name}{slot_label}]" if label and rep is not None else label
    return InteractionTerm(
        coupling=prefactor
        * action.coupling
        * psi_bar_gamma_psi(i_bar, i_psi, lorentz_label or piece.lorentz_index),
        fields=(
            fermion.occurrence(conjugated=True, labels=bar_labels),
            fermion.occurrence(labels=psi_labels),
            piece.metadata.gauge_field.occurrence(labels=action.gauge_labels),
        ),
        closed_dirac_bilinears=((0, 1),),
        label=label_suffix or f"{piece.metadata.gauge_group.name}: {fermion.name} gauge current",
    )


def _default_scalar_contact_internal_label(
    scalar: Field,
    left_piece: Any,
    right_piece: Any,
    *,
    symbol: Callable[[str], object],
    slot_suffix: Callable[[Field, Optional[int]], str],
) -> object:
    left_group = left_piece.metadata.gauge_group
    right_group = right_piece.metadata.gauge_group
    slot = left_piece.active_slot
    if slot is None or left_piece.metadata.representation is None:
        raise ValueError("Internal scalar-contact labels require a non-abelian active slot.")

    if left_group is right_group:
        return symbol(
            f"{left_piece.metadata.representation.index.prefix}_mid_{scalar.name}_{left_group.name}"
            f"{slot_suffix(scalar, slot)}"
        )
    return symbol(
        f"{left_piece.metadata.representation.index.prefix}_mid_{scalar.name}_{left_group.name}_{right_group.name}"
        f"{slot_suffix(scalar, slot)}"
    )


def _build_scalar_contact_action_data(
    *,
    scalar: Field,
    left_piece: Any,
    right_piece: Any,
    gauge_action_from_piece: Callable,
    symbol: Callable[[str], object],
    slot_suffix: Callable[[Field, Optional[int]], str],
    left_lorentz_label=None,
    right_lorentz_label=None,
    matter_labels=None,
    left_adjoint_label=None,
    right_adjoint_label=None,
    internal_label=None,
    contact_prefactor=1,
) -> _ScalarContactActionData:
    left_action = gauge_action_from_piece(
        scalar,
        left_piece,
        lorentz_label=left_lorentz_label,
        adjoint_label=left_adjoint_label,
        default_adjoint_qualifier="mix",
        purpose="Scalar contact compilation (left gauge field)",
    )
    right_action = gauge_action_from_piece(
        scalar,
        right_piece,
        lorentz_label=right_lorentz_label,
        adjoint_label=right_adjoint_label,
        default_adjoint_qualifier="mix",
        purpose="Scalar contact compilation (right gauge field)",
    )

    coupling = contact_prefactor * left_action.coupling * right_action.coupling
    scalar_bar_slot_labels: dict[int, object] = {}
    scalar_slot_labels: dict[int, object] = {}
    active_slots: list[int] = []
    exclude_slots: set[int] = set()

    if left_action.representation_slot is not None:
        active_slots.append(left_action.representation_slot)
        exclude_slots.add(left_action.representation_slot)
    if right_action.representation_slot is not None:
        active_slots.append(right_action.representation_slot)
        exclude_slots.add(right_action.representation_slot)

    if (
        left_action.representation is not None
        and right_action.representation is not None
        and left_action.representation_slot == right_action.representation_slot
    ):
        active_slot = left_action.representation_slot
        if active_slot is None:
            raise ValueError("Same-slot scalar contacts require an active matter slot.")
        left_label, right_label = matter_labels or left_action.default_matter_labels()
        middle = internal_label or _default_scalar_contact_internal_label(
            scalar,
            left_piece,
            right_piece,
            symbol=symbol,
            slot_suffix=slot_suffix,
        )
        coupling *= (
            left_action.generator(left_label, middle)
            * right_action.generator(middle, right_label)
        )
        scalar_bar_slot_labels[active_slot] = left_label
        scalar_slot_labels[active_slot] = right_label
    else:
        for action in (left_action, right_action):
            if action.representation is None:
                continue
            if action.representation_slot is None:
                raise ValueError("Non-abelian scalar contacts require an active matter slot.")
            left_label, right_label = matter_labels or action.default_matter_labels()
            coupling *= action.generator(left_label, right_label)
            scalar_bar_slot_labels[action.representation_slot] = left_label
            scalar_slot_labels[action.representation_slot] = right_label

    spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
        scalar,
        exclude_slots=exclude_slots,
    )
    coupling *= spectator_factor
    scalar_bar_slot_labels.update(spectator_left_slots)
    scalar_slot_labels.update(spectator_right_slots)

    return _ScalarContactActionData(
        coupling=coupling * scalar_gauge_contact(
            left_action.lorentz_label,
            right_action.lorentz_label,
        ),
        scalar_bar_labels=scalar.pack_slot_labels(scalar_bar_slot_labels),
        scalar_labels=scalar.pack_slot_labels(scalar_slot_labels),
        left_gauge_labels=left_action.gauge_labels(),
        right_gauge_labels=right_action.gauge_labels(),
        active_slots=tuple(dict.fromkeys(active_slots)),
    )


def _compile_scalar_contact_terms(
    *,
    scalar: Field,
    left_pieces: tuple[Any, ...],
    right_pieces: tuple[Any, ...],
    gauge_action_from_piece: Callable,
    symbol: Callable[[str], object],
    slot_suffix: Callable[[Field, Optional[int]], str],
    mixed_scalar_contact_slot_suffix: Callable[[tuple[int, ...]], str],
    left_lorentz_label,
    right_lorentz_label,
    matter_labels=None,
    left_adjoint_label=None,
    right_adjoint_label=None,
    internal_label=None,
    contact_prefactor=1,
    label_prefix: str = "",
    label_kind: str,
) -> tuple[InteractionTerm, ...]:
    """Build scalar two-gauge contacts from resolved gauge actions."""

    prefix = label_prefix + " " if label_prefix else ""
    contact_terms: list[InteractionTerm] = []
    for left_piece in left_pieces:
        for right_piece in right_pieces:
            contact_data = _build_scalar_contact_action_data(
                scalar=scalar,
                left_piece=left_piece,
                right_piece=right_piece,
                gauge_action_from_piece=gauge_action_from_piece,
                symbol=symbol,
                slot_suffix=slot_suffix,
                left_lorentz_label=left_lorentz_label,
                right_lorentz_label=right_lorentz_label,
                matter_labels=matter_labels,
                left_adjoint_label=left_adjoint_label,
                right_adjoint_label=right_adjoint_label,
                internal_label=internal_label,
                contact_prefactor=contact_prefactor,
            )

            if label_kind == "same_group":
                gauge_group = left_piece.metadata.gauge_group
                slot_label = ""
                if not gauge_group.abelian:
                    slot_label = (
                        f" [slots {left_piece.active_slot + 1},"
                        f"{right_piece.active_slot + 1}]"
                    )
                label = prefix + f"{gauge_group.name}: scalar contact{slot_label}"
            elif label_kind == "mixed_group":
                left_group = left_piece.metadata.gauge_group
                right_group = right_piece.metadata.gauge_group
                label = (
                    prefix
                    + f"{left_group.name} x {right_group.name}: scalar mixed contact"
                    + mixed_scalar_contact_slot_suffix(contact_data.active_slots)
                )
            else:
                raise ValueError(f"Unknown scalar contact label kind: {label_kind!r}")

            contact_terms.append(
                InteractionTerm(
                    coupling=contact_data.coupling,
                    fields=(
                        scalar.occurrence(
                            conjugated=True,
                            labels=contact_data.scalar_bar_labels,
                        ),
                        scalar.occurrence(labels=contact_data.scalar_labels),
                        left_piece.metadata.gauge_field.occurrence(
                            labels=contact_data.left_gauge_labels
                        ),
                        right_piece.metadata.gauge_field.occurrence(
                            labels=contact_data.right_gauge_labels
                        ),
                    ),
                    label=label,
                )
            )

    return tuple(contact_terms)
