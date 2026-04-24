"""
Gauge-structure compilers for the Symbolica/Spenso prototype.

This module exposes two layers on purpose:

- the minimal compiler: structural gauge interactions from metadata
- the covariant/physical compiler: convention-fixed kinetic-term expansion

Frozen conventions for the physical path:

- Fourier transform derivatives act as ``-i p_mu``
- ``vertex_factor(...)`` contributes the universal overall ``+i``
- matter covariant derivatives use ``D_mu = partial_mu + i g A_mu``
- non-abelian field strengths use
  ``F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu``

With these choices, matter currents carry the signs already locked down in the
covariant tests, the Yang-Mills 3-gauge vertex is real, and the 4-gauge vertex
keeps an explicit overall ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from symbolica import S, Expression

from model import (
    CovD,
    CovariantDerivativeFactor,
    ComplexScalarKineticTerm,
    DerivativeAction,
    DiracKineticTerm,
    Field,
    GhostTerm,
    GaugeKineticTerm,
    GaugeFixingTerm,
    GaugeGroup,
    GaugeRepresentation,
    InteractionTerm,
    Model,
)
from lagrangian.operators import psi_bar_gamma_psi, scalar_gauge_contact
from symbolic.spenso_structures import LORENTZ_KIND, SPINOR_KIND, lorentz_metric


_HALF = Expression.num(1) / Expression.num(2)
_QUARTER = Expression.num(1) / Expression.num(4)


def _symbol(name: str):
    return S(name)


def _is_explicit_zero(value) -> bool:
    """Whether one coefficient is manifestly zero after local normalization."""
    try:
        normalized = value.expand() if hasattr(value, "expand") else value
    except Exception:
        normalized = value
    try:
        return bool(normalized == 0)
    except Exception:
        return False


def _default_spinor_labels(field: Field, gauge_group: GaugeGroup):
    stem = f"{field.name}_{gauge_group.name}"
    return _symbol(f"i_bar_{stem}"), _symbol(f"i_{stem}")


def _slot_suffix(field: Field, slot: Optional[int]) -> str:
    if slot is None:
        return ""
    kind = field.indices[slot].kind
    if field.index_kind_count(kind) <= 1:
        return ""
    return f"_slot{slot + 1}"


def _default_vector_label(field: Field, gauge_group: GaugeGroup, suffix: str = "mu"):
    del field, gauge_group
    return _symbol(suffix)


def _default_matter_labels(field: Field, rep_prefix: str, slot: Optional[int] = None):
    stem = f"{field.name}{_slot_suffix(field, slot)}"
    return _symbol(f"{rep_prefix}_bar_{stem}"), _symbol(f"{rep_prefix}_{stem}")


def _default_index_labels(field: Field, index, qualifier: str = "id", slot: Optional[int] = None):
    stem = f"{field.name}_{index.kind}"
    if slot is not None and field.index_kind_count(index.kind) > 1:
        stem += f"_{slot + 1}"
    stem += f"_{qualifier}"
    return _symbol(f"{index.prefix}_bar_{stem}"), _symbol(f"{index.prefix}_{stem}")


def _first_non_lorentz_index_slot(field: Field) -> Optional[int]:
    for slot, index in enumerate(field.indices):
        if index.kind != LORENTZ_KIND:
            return slot
    return None


def _adjoint_index_kind(gauge_field: Field) -> Optional[str]:
    slot = _first_non_lorentz_index_slot(gauge_field)
    if slot is None:
        return None
    return gauge_field.indices[slot].kind


def _adjoint_index_slot(gauge_field: Field) -> Optional[int]:
    return _first_non_lorentz_index_slot(gauge_field)


def _unique_slot(field: Field, kind: str, *, purpose: str) -> int:
    slots = field.index_positions(kind=kind)
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to expose exactly one {kind!r} slot; "
            f"found {len(slots)}."
        )
    return slots[0]


def _default_gauge_lorentz_labels(field: Field, gauge_group: GaugeGroup, count: int):
    stem = f"{field.name}_{gauge_group.name}"
    return tuple(_symbol(f"mu_{stem}_{slot}") for slot in range(1, count + 1))


def _default_adjoint_labels(gauge_field: Field, gauge_group: GaugeGroup, count: int):
    adj_kind = _adjoint_index_kind(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz adjoint index."
        )
    stem = f"{gauge_field.name}_{gauge_group.name}"
    return tuple(_symbol(f"{adj_kind}_{stem}_{slot}") for slot in range(1, count + 1))


def _default_internal_adjoint_label(gauge_field: Field, gauge_group: GaugeGroup):
    adj_kind = _adjoint_index_kind(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz adjoint index."
        )
    return _symbol(f"{adj_kind}_mid_{gauge_field.name}_{gauge_group.name}")


def _default_ghost_labels(field: Field, index, slot: Optional[int] = None):
    return _default_index_labels(field, index, qualifier="ghost", slot=slot)


def _build_structure_constant(gauge_group: GaugeGroup, left_label, middle_label, right_label):
    if gauge_group.structure_constant is None or not callable(gauge_group.structure_constant):
        raise ValueError(
            f"Gauge group {gauge_group.name!r} needs a callable structure_constant "
            "builder for pure-gauge compilation."
        )
    return gauge_group.structure_constant(left_label, middle_label, right_label)


def _field_charge(field: Field, gauge_group: GaugeGroup):
    if gauge_group.charge is None:
        raise ValueError(f"Gauge group {gauge_group.name!r} has no abelian charge label.")
    if gauge_group.charge not in field.quantum_numbers:
        raise ValueError(
            f"Field {field.name!r} has no quantum number {gauge_group.charge!r} "
            f"required by gauge group {gauge_group.name!r}."
        )
    return field.quantum_numbers[gauge_group.charge]


def _field_transforms_under_gauge_group(field: Field, gauge_group: GaugeGroup) -> bool:
    if gauge_group.abelian:
        if gauge_group.charge is None:
            return False
        return field.quantum_numbers.get(gauge_group.charge, 0) != 0
    return gauge_group.matter_representation(field) is not None


def _require_declared_field(model: Model, target, *, purpose: str) -> Field:
    """Resolve one field strictly from the parent model declarations."""
    if isinstance(target, Field):
        for field in model.fields:
            if field is target:
                return field
        raise ValueError(
            f"{purpose} requires field {target.name!r} to be declared in model.fields."
        )

    resolved = model.find_field(target)
    if resolved is None:
        raise ValueError(f"{purpose} could not resolve field {target!r} in model.fields.")
    return resolved


def _require_declared_gauge_group(model: Model, target, *, purpose: str) -> GaugeGroup:
    """Resolve one gauge group strictly from the parent model declarations."""
    if isinstance(target, GaugeGroup):
        for gauge_group in model.gauge_groups:
            if gauge_group is target:
                return gauge_group
        raise ValueError(
            f"{purpose} requires gauge group {target.name!r} to be declared in model.gauge_groups."
        )

    resolved = model.find_gauge_group(target)
    if resolved is None:
        raise ValueError(
            f"{purpose} could not resolve gauge group {target!r} in model.gauge_groups."
        )
    return resolved


def _require_field_transforms_under_gauge_group(
    field: Field,
    gauge_group: GaugeGroup,
    *,
    purpose: str,
):
    """Reject explicit group selections that do not act on the chosen field."""
    if gauge_group.abelian:
        if gauge_group.charge is None:
            raise ValueError(
                f"{purpose} cannot use abelian gauge group {gauge_group.name!r} "
                "without a declared charge label."
            )
        charge = field.quantum_numbers.get(gauge_group.charge, 0)
        if charge == 0:
            raise ValueError(
                f"{purpose} requires field {field.name!r} to carry non-zero "
                f"charge {gauge_group.charge!r} under gauge group {gauge_group.name!r}."
            )
        return

    if gauge_group.matter_representation(field) is None:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to carry a declared "
            f"representation under gauge group {gauge_group.name!r}."
        )


def _resolve_covariant_gauge_groups(model: Model, *, field: Field, gauge_group=None) -> tuple[GaugeGroup, ...]:
    purpose = f"Covariant compilation for field {field.name!r}"
    if gauge_group is not None:
        if isinstance(gauge_group, (tuple, list)):
            resolved = []
            for item in gauge_group:
                group = _require_declared_gauge_group(model, item, purpose=purpose)
                _require_field_transforms_under_gauge_group(field, group, purpose=purpose)
                resolved.append(group)
            return tuple(resolved)

        resolved = _require_declared_gauge_group(model, gauge_group, purpose=purpose)
        _require_field_transforms_under_gauge_group(field, resolved, purpose=purpose)
        return (resolved,)

    matches = tuple(group for group in model.gauge_groups if _field_transforms_under_gauge_group(field, group))
    if not matches:
        raise ValueError(f"Field {field.name!r} does not transform under any declared gauge group.")
    return matches


def _spectator_identity_factor(field: Field, *, exclude_slots=()):
    factor = Expression.num(1)
    left_slot_labels = {}
    right_slot_labels = {}

    for slot, index in enumerate(field.indices):
        if slot in exclude_slots or index.kind == LORENTZ_KIND:
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
        return _symbol(f"{index.prefix}_bar_{stem}")
    return _symbol(f"{index.prefix}_{stem}")


def _materialize_spectator_occurrences(spectators: tuple[tuple[Field, bool], ...]):
    """Build spectator occurrences plus their internal contraction factor.

    Rules:
    - non-self-conjugate scalar spectators automatically contract in obvious
      ``phi.bar * phi`` pairs
    - fermion spectators must appear in explicit ``psibar * psi`` pairs
    - any remaining spectator indices stay open on the final vertex
    """
    if not spectators:
        return Expression.num(1), ()

    factor = Expression.num(1)
    slot_labels = [dict() for _ in spectators]
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
                    "explicit bar/psi pairs. Use InteractionTerm(...) for unpaired or more "
                    "complicated spinor structures."
                )
            pair_count = len(plain_positions)
        elif field.self_conjugate:
            pair_count = 0
        else:
            pair_count = min(len(plain_positions), len(conj_positions))

        for pair_idx in range(pair_count):
            left_pos = conj_positions[pair_idx]
            right_pos = plain_positions[pair_idx]
            for slot, index in enumerate(field.indices):
                if index.kind == LORENTZ_KIND:
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

    return factor, tuple(occurrences)


def _decorate_interactions_with_spectators(
    interactions: tuple[InteractionTerm, ...],
    *,
    spectator_factor,
    spectator_occurrences: tuple,
) -> tuple[InteractionTerm, ...]:
    return tuple(
        InteractionTerm(
            coupling=interaction.coupling * spectator_factor,
            fields=interaction.fields + spectator_occurrences,
            derivatives=interaction.derivatives,
            label=interaction.label,
        )
        for interaction in interactions
    )


def _nonabelian_rep_and_slots(field: Field, gauge_group: GaugeGroup):
    rep_info = gauge_group.matter_representation_and_slots(field)
    if rep_info is None:
        raise ValueError(
            f"Field {field.name!r} carries no representation declared for "
            f"gauge group {gauge_group.name!r}."
        )
    return rep_info


def _adjoint_slot_info(gauge_field: Field, *, purpose: str):
    adj_kind = _adjoint_index_kind(gauge_field)
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz "
            f"adjoint index for {purpose}."
        )
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot "
            f"for {purpose}."
        )
    return adj_kind, adj_slot


def _mixed_scalar_contact_slot_suffix(active_slots: tuple[int, ...]) -> str:
    if not active_slots:
        return ""
    if len(active_slots) == 1:
        return f" [slot {active_slots[0] + 1}]"
    return f" [slots {', '.join(str(slot + 1) for slot in active_slots)}]"


def _ghost_field_for_group(model: Model, gauge_group: GaugeGroup) -> Field:
    """Resolve and validate the ghost field declared for one gauge group."""
    if gauge_group.ghost_field is None:
        raise ValueError(
            f"Ghost compilation requires gauge group {gauge_group.name!r} to declare ghost_field."
        )
    ghost_field = _require_declared_field(
        model,
        gauge_group.ghost_field,
        purpose="Ghost compilation",
    )
    if ghost_field.kind != "ghost":
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to have kind='ghost'."
        )
    if ghost_field.self_conjugate:
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to be non-self-conjugate."
        )
    return ghost_field


@dataclass(frozen=True)
class CovariantDerivativePartialPiece:
    field: Field
    lorentz_index: object
    conjugated: bool = False


@dataclass(frozen=True)
class CovariantGaugeMetadata:
    gauge_group: GaugeGroup
    gauge_field: Field
    representation: GaugeRepresentation | None
    representation_slots: tuple[int, ...]
    repeated_index: bool
    conjugated: bool
    conjugation_supported: bool
    self_conjugate_field: bool

    @property
    def representation_name(self) -> str:
        if self.representation is None:
            return self.gauge_group.charge or ""
        return self.representation.name or self.representation.index.name


@dataclass(frozen=True)
class CovariantGaugePiece:
    metadata: CovariantGaugeMetadata
    lorentz_index: object
    active_slot: Optional[int] = None


@dataclass(frozen=True)
class ExpandedCovariantDerivative:
    factor: CovariantDerivativeFactor
    field: Field
    conjugated: bool
    derivative_piece: CovariantDerivativePartialPiece
    gauge_current_pieces: tuple[CovariantGaugePiece, ...]
    contact_ready_data: tuple[CovariantGaugePiece, ...]


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


@dataclass(frozen=True)
class _GaugeAction:
    """One resolved gauge action on one matter-field occurrence.

    Abelian and non-abelian actions intentionally share this shape.  Abelian
    actions have no active matter slot and carry the charge in ``coupling``.
    Non-abelian actions carry the coupling constant here and insert the concrete
    representation generator only when the caller has chosen matter-slot labels.
    """

    field: Field
    piece: CovariantGaugePiece
    lorentz_label: object
    gauge_lorentz_slot: int
    gauge_slot_labels: dict[int, object]
    coupling: object
    representation: GaugeRepresentation | None = None
    representation_slot: Optional[int] = None
    adjoint_label: object | None = None

    @classmethod
    def from_piece(
        cls,
        field: Field,
        piece: CovariantGaugePiece,
        *,
        lorentz_label=None,
        adjoint_label=None,
        default_adjoint_qualifier: str = "",
        purpose: str,
    ) -> "_GaugeAction":
        gauge_group = piece.metadata.gauge_group
        gauge_field = piece.metadata.gauge_field
        mu = lorentz_label or piece.lorentz_index
        gauge_lorentz_slot = _unique_slot(
            gauge_field,
            LORENTZ_KIND,
            purpose=purpose,
        )
        gauge_slot_labels = {gauge_lorentz_slot: mu}

        if gauge_group.abelian:
            return cls(
                field=field,
                piece=piece,
                lorentz_label=mu,
                gauge_lorentz_slot=gauge_lorentz_slot,
                gauge_slot_labels=gauge_slot_labels,
                coupling=gauge_group.coupling * _field_charge(field, gauge_group),
            )

        rep = piece.metadata.representation
        rep_slot = piece.active_slot
        if rep is None or rep_slot is None:
            raise ValueError(
                f"Non-abelian covariant action for field {field.name!r} is missing "
                "representation metadata."
            )

        adj_kind, adj_slot = _adjoint_slot_info(gauge_field, purpose=purpose)
        qualifier = f"_{default_adjoint_qualifier}" if default_adjoint_qualifier else ""
        adjoint = adjoint_label or _symbol(
            f"{adj_kind}_{gauge_field.name}_{gauge_group.name}{qualifier}"
        )
        gauge_slot_labels[adj_slot] = adjoint
        return cls(
            field=field,
            piece=piece,
            lorentz_label=mu,
            gauge_lorentz_slot=gauge_lorentz_slot,
            gauge_slot_labels=gauge_slot_labels,
            coupling=gauge_group.coupling,
            representation=rep,
            representation_slot=rep_slot,
            adjoint_label=adjoint,
        )

    @property
    def gauge_field(self) -> Field:
        return self.piece.metadata.gauge_field

    def gauge_labels(self) -> dict:
        return self.gauge_field.pack_slot_labels(self.gauge_slot_labels)

    def default_matter_labels(self):
        if self.representation is None or self.representation_slot is None:
            raise ValueError("Abelian gauge actions do not select matter-slot labels.")
        return _default_matter_labels(
            self.field,
            self.representation.index.prefix,
            slot=self.representation_slot,
        )

    def generator(self, left_label, right_label):
        if self.representation is None:
            raise ValueError("Abelian gauge actions do not have representation generators.")
        return self.representation.build_generator(
            self.adjoint_label,
            left_label,
            right_label,
        )


def _validate_expandable_covariant_field(
    field: Field,
    *,
    purpose: str,
):
    if field.kind not in ("fermion", "scalar"):
        raise ValueError(
            f"{purpose} currently supports only fermion or scalar matter fields; "
            f"got kind={field.kind!r} for field {field.name!r}."
        )
    if field.self_conjugate:
        raise ValueError(
            f"{purpose} currently supports only non-self-conjugate matter fields; "
            f"field {field.name!r} is self-conjugate."
        )


def _covariant_gauge_metadata(
    field: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    *,
    conjugated: bool,
) -> CovariantGaugeMetadata:
    if gauge_group.abelian:
        _field_charge(field, gauge_group)
        rep = None
        rep_slots: tuple[int, ...] = ()
    else:
        rep, rep_slots = _nonabelian_rep_and_slots(field, gauge_group)

    return CovariantGaugeMetadata(
        gauge_group=gauge_group,
        gauge_field=gauge_field,
        representation=rep,
        representation_slots=rep_slots,
        repeated_index=len(rep_slots) > 1,
        conjugated=conjugated,
        conjugation_supported=not field.self_conjugate,
        self_conjugate_field=field.self_conjugate,
    )


def _expand_field_gauge_pieces(
    *,
    field: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_index,
    conjugated: bool,
    purpose: str,
) -> tuple[CovariantGaugePiece, ...]:
    _validate_expandable_covariant_field(field, purpose=purpose)
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    metadata = _covariant_gauge_metadata(
        field,
        gauge_group,
        gauge_field,
        conjugated=conjugated,
    )
    if gauge_group.abelian:
        return (CovariantGaugePiece(metadata=metadata, lorentz_index=lorentz_index),)
    return tuple(
        CovariantGaugePiece(
            metadata=metadata,
            lorentz_index=lorentz_index,
            active_slot=slot,
        )
        for slot in metadata.representation_slots
    )


def expand_cov_der(
    model: Model,
    cov_factor: CovariantDerivativeFactor,
    *,
    gauge_group=None,
) -> ExpandedCovariantDerivative:
    """Resolve one ``CovD(...)`` factor into derivative and gauge-action pieces.

    The result is representation-aware and explicit about:
    - which declared gauge groups act on the field
    - which representation was matched
    - which concrete field slots are active
    - whether repeated-slot expansion is in use
    - whether the input occurrence is conjugated
    """
    if not isinstance(cov_factor, CovariantDerivativeFactor):
        raise TypeError(
            "expand_cov_der(...) expects a CovariantDerivativeFactor produced by CovD(...)."
        )

    purpose = "Covariant-derivative expansion"
    field = _require_declared_field(model, cov_factor.field, purpose=purpose)
    effective_conjugated = bool(cov_factor.conjugated and not field.self_conjugate)
    _validate_expandable_covariant_field(field, purpose=purpose)

    normalized_factor = CovariantDerivativeFactor(
        field=field,
        lorentz_index=cov_factor.lorentz_index,
        conjugated=effective_conjugated,
    )
    gauge_groups = _resolve_covariant_gauge_groups(
        model,
        field=field,
        gauge_group=gauge_group,
    )

    gauge_pieces: list[CovariantGaugePiece] = []
    for group in gauge_groups:
        gauge_field = model.gauge_boson_field(group)
        gauge_pieces.extend(
            _expand_field_gauge_pieces(
                field=field,
                gauge_group=group,
                gauge_field=gauge_field,
                lorentz_index=normalized_factor.lorentz_index,
                conjugated=effective_conjugated,
                purpose=purpose,
            )
        )

    derivative_piece = CovariantDerivativePartialPiece(
        field=field,
        lorentz_index=normalized_factor.lorentz_index,
        conjugated=effective_conjugated,
    )
    return ExpandedCovariantDerivative(
        factor=normalized_factor,
        field=field,
        conjugated=effective_conjugated,
        derivative_piece=derivative_piece,
        gauge_current_pieces=tuple(gauge_pieces),
        contact_ready_data=tuple(gauge_pieces),
    )


def _build_bilinear_gauge_action_data(
    field: Field,
    piece: CovariantGaugePiece,
    *,
    lorentz_label=None,
    matter_labels=None,
    adjoint_label=None,
    spectator_exclude_slots=(),
):
    action = _GaugeAction.from_piece(
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
    piece: CovariantGaugePiece,
    derivative_target: int,
    coupling_prefactor=1,
    label: str = "",
    lorentz_label=None,
    matter_labels=None,
    adjoint_label=None,
) -> InteractionTerm:
    action = _build_bilinear_gauge_action_data(
        scalar,
        piece,
        lorentz_label=lorentz_label,
        matter_labels=matter_labels,
        adjoint_label=adjoint_label,
    )
    derivative_label = lorentz_label or piece.lorentz_index
    sign = -1 if piece.metadata.conjugated else 1
    return InteractionTerm(
        coupling=sign * coupling_prefactor * action.coupling,
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar.pack_slot_labels(action.left_slot_labels)),
            scalar.occurrence(labels=scalar.pack_slot_labels(action.right_slot_labels)),
            piece.metadata.gauge_field.occurrence(labels=action.gauge_labels),
        ),
        derivatives=(DerivativeAction(target=derivative_target, lorentz_index=derivative_label),),
        label=label,
    )


def _build_fermion_current_interaction(
    *,
    fermion: Field,
    piece: CovariantGaugePiece,
    spinor_slot: int,
    i_bar,
    i_psi,
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
    slot_suffix = _slot_suffix(fermion, piece.active_slot)
    slot_label = f"{label} [{rep.index.name}{slot_suffix}]" if label and rep is not None else label
    return InteractionTerm(
        coupling=prefactor * action.coupling * psi_bar_gamma_psi(i_bar, i_psi, lorentz_label or piece.lorentz_index),
        fields=(
            fermion.occurrence(conjugated=True, labels=bar_labels),
            fermion.occurrence(labels=psi_labels),
            piece.metadata.gauge_field.occurrence(labels=action.gauge_labels),
        ),
        label=slot_label or f"{piece.metadata.gauge_group.name}: {fermion.name} gauge current",
    )


def _default_scalar_contact_internal_label(
    scalar: Field,
    left_piece: CovariantGaugePiece,
    right_piece: CovariantGaugePiece,
) -> object:
    left_group = left_piece.metadata.gauge_group
    right_group = right_piece.metadata.gauge_group
    slot = left_piece.active_slot
    if slot is None or left_piece.metadata.representation is None:
        raise ValueError("Internal scalar-contact labels require a non-abelian active slot.")

    if left_group is right_group:
        return _symbol(
            f"{left_piece.metadata.representation.index.prefix}_mid_{scalar.name}_{left_group.name}"
            f"{_slot_suffix(scalar, slot)}"
        )
    return _symbol(
        f"{left_piece.metadata.representation.index.prefix}_mid_{scalar.name}_{left_group.name}_{right_group.name}"
        f"{_slot_suffix(scalar, slot)}"
    )


def _build_scalar_contact_action_data(
    *,
    scalar: Field,
    left_piece: CovariantGaugePiece,
    right_piece: CovariantGaugePiece,
    left_lorentz_label=None,
    right_lorentz_label=None,
    matter_labels=None,
    left_adjoint_label=None,
    right_adjoint_label=None,
    internal_label=None,
    contact_prefactor=1,
) -> _ScalarContactActionData:
    left_action = _GaugeAction.from_piece(
        scalar,
        left_piece,
        lorentz_label=left_lorentz_label,
        adjoint_label=left_adjoint_label,
        default_adjoint_qualifier="mix",
        purpose="Scalar contact compilation (left gauge field)",
    )
    right_action = _GaugeAction.from_piece(
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
    left_pieces: tuple[CovariantGaugePiece, ...],
    right_pieces: tuple[CovariantGaugePiece, ...],
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
    """Build scalar two-gauge contacts from resolved gauge actions.

    The same ordered product is used for both same-group contacts and mixed
    cross-group contacts.  Only the human-facing label differs between the two
    cases.
    """

    prefix = label_prefix + " " if label_prefix else ""
    contact_terms: list[InteractionTerm] = []
    for left_piece in left_pieces:
        for right_piece in right_pieces:
            contact_data = _build_scalar_contact_action_data(
                scalar=scalar,
                left_piece=left_piece,
                right_piece=right_piece,
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
                    + _mixed_scalar_contact_slot_suffix(contact_data.active_slots)
                )
            else:
                raise ValueError(f"Unknown scalar contact label kind: {label_kind!r}")

            contact_terms.append(
                InteractionTerm(
                    coupling=contact_data.coupling,
                    fields=(
                        scalar.occurrence(conjugated=True, labels=contact_data.scalar_bar_labels),
                        scalar.occurrence(labels=contact_data.scalar_labels),
                        left_piece.metadata.gauge_field.occurrence(labels=contact_data.left_gauge_labels),
                        right_piece.metadata.gauge_field.occurrence(labels=contact_data.right_gauge_labels),
                    ),
                    label=label,
                )
            )

    return tuple(contact_terms)


def compile_fermion_gauge_current(
    *,
    fermion: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_label=None,
    spinor_labels=None,
    matter_labels=None,
    adjoint_label=None,
    prefactor=1,
    label: str = "",
) -> tuple[InteractionTerm, ...]:
    """Compile one or more fermion-gauge current interactions from model metadata.

    If the relevant gauge representation resolves to multiple index slots and the
    representation has ``slot_policy='sum'``, this returns one interaction term
    per active slot.
    """
    if fermion.kind != "fermion":
        raise ValueError(f"Expected a fermion field, got kind={fermion.kind!r}.")

    mu = lorentz_label or _default_vector_label(gauge_field, gauge_group, suffix="mu")
    i_bar, i_psi = spinor_labels or _default_spinor_labels(fermion, gauge_group)
    fermion_spinor_slot = _unique_slot(
        fermion,
        SPINOR_KIND,
        purpose="Fermion gauge-current compilation",
    )

    pieces = _expand_field_gauge_pieces(
        field=fermion,
        gauge_group=gauge_group,
        gauge_field=gauge_field,
        lorentz_index=mu,
        conjugated=False,
        purpose="Fermion gauge-current compilation",
    )

    interactions: list[InteractionTerm] = []
    for piece in pieces:
        interactions.append(
            _build_fermion_current_interaction(
                fermion=fermion,
                piece=piece,
                spinor_slot=fermion_spinor_slot,
                i_bar=i_bar,
                i_psi=i_psi,
                prefactor=prefactor,
                label=label,
                lorentz_label=mu,
                matter_labels=matter_labels,
                adjoint_label=adjoint_label,
                spectator_exclude_slots={fermion_spinor_slot},
            )
        )

    return tuple(interactions)


def compile_complex_scalar_gauge_terms(
    *,
    scalar: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_labels=None,
    matter_labels=None,
    adjoint_labels=None,
    internal_label=None,
    current_prefactor=1,
    contact_prefactor=1,
    label_prefix: str = "",
):
    """Compile the complex-scalar gauge current/contact terms.

    If the relevant gauge representation resolves to multiple index slots and the
    representation has ``slot_policy='sum'``, the output sums over active slots:
    - current terms: one pair (phi / phi^dagger derivative placement) per slot
    - contact terms: one term per ordered slot pair (slot_i, slot_j)
    """
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")

    mu, nu = lorentz_labels or (
        _default_vector_label(gauge_field, gauge_group, suffix="mu"),
        _default_vector_label(gauge_field, gauge_group, suffix="nu"),
    )
    prefix = label_prefix + " " if label_prefix else ""

    adjoint_mu = None
    adjoint_nu = None
    if not gauge_group.abelian:
        adj_kind, _ = _adjoint_slot_info(
            gauge_field,
            purpose="Complex-scalar gauge-term compilation",
        )
        adjoint_mu, adjoint_nu = adjoint_labels or (
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_1"),
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_2"),
        )

    right_pieces = _expand_field_gauge_pieces(
        field=scalar,
        gauge_group=gauge_group,
        gauge_field=gauge_field,
        lorentz_index=mu,
        conjugated=False,
        purpose="Complex-scalar gauge-term compilation",
    )
    left_pieces = _expand_field_gauge_pieces(
        field=scalar,
        gauge_group=gauge_group,
        gauge_field=gauge_field,
        lorentz_index=mu,
        conjugated=True,
        purpose="Complex-scalar gauge-term compilation",
    )

    current_terms: list[InteractionTerm] = []
    for right_piece, left_piece in zip(right_pieces, left_pieces):
        current_terms.append(
            _compile_scalar_current_from_piece(
                scalar=scalar,
                piece=right_piece,
                derivative_target=1,
                coupling_prefactor=current_prefactor,
                label=prefix + f"{gauge_group.name}: scalar current (+){_slot_suffix(scalar, right_piece.active_slot)}",
                lorentz_label=mu,
                matter_labels=matter_labels,
                adjoint_label=adjoint_mu,
            )
        )
        current_terms.append(
            _compile_scalar_current_from_piece(
                scalar=scalar,
                piece=left_piece,
                derivative_target=0,
                coupling_prefactor=current_prefactor,
                label=prefix + f"{gauge_group.name}: scalar current (-){_slot_suffix(scalar, left_piece.active_slot)}",
                lorentz_label=mu,
                matter_labels=matter_labels,
                adjoint_label=adjoint_mu,
            )
        )

    contact_terms = _compile_scalar_contact_terms(
        scalar=scalar,
        left_pieces=left_pieces,
        right_pieces=right_pieces,
        left_lorentz_label=mu,
        right_lorentz_label=nu,
        matter_labels=matter_labels,
        left_adjoint_label=adjoint_mu,
        right_adjoint_label=adjoint_nu,
        internal_label=internal_label,
        contact_prefactor=contact_prefactor,
        label_prefix=label_prefix,
        label_kind="same_group",
    )

    return tuple(current_terms) + contact_terms


def compile_mixed_complex_scalar_contact_terms(
    *,
    scalar: Field,
    left_gauge_group: GaugeGroup,
    left_gauge_field: Field,
    right_gauge_group: GaugeGroup,
    right_gauge_field: Field,
    lorentz_labels=None,
    left_adjoint_label=None,
    right_adjoint_label=None,
    contact_prefactor=1,
    label_prefix: str = "",
):
    """Compile ordered cross-group contact terms from ``(D_mu phi)^dagger (D^mu phi)``.

    This covers the mixed products that arise when a complex scalar transforms
    under more than one gauge group:

    ``(g_r A^r_mu action_r(phi))^dagger (g_s A^{s,mu} action_s(phi))``

    The two group actions are representation-dependent:
    abelian groups contribute a charge factor, while non-abelian groups
    contribute one generator per active representation slot.
    """
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")

    mu, nu = lorentz_labels or (
        _default_vector_label(left_gauge_field, left_gauge_group, suffix="mu"),
        _default_vector_label(right_gauge_field, right_gauge_group, suffix="nu"),
    )
    left_pieces = _expand_field_gauge_pieces(
        field=scalar,
        gauge_group=left_gauge_group,
        gauge_field=left_gauge_field,
        lorentz_index=mu,
        conjugated=True,
        purpose="Mixed scalar contact compilation",
    )
    right_pieces = _expand_field_gauge_pieces(
        field=scalar,
        gauge_group=right_gauge_group,
        gauge_field=right_gauge_field,
        lorentz_index=nu,
        conjugated=False,
        purpose="Mixed scalar contact compilation",
    )

    return _compile_scalar_contact_terms(
        scalar=scalar,
        left_pieces=left_pieces,
        right_pieces=right_pieces,
        left_lorentz_label=mu,
        right_lorentz_label=nu,
        left_adjoint_label=left_adjoint_label,
        right_adjoint_label=right_adjoint_label,
        contact_prefactor=contact_prefactor,
        label_prefix=label_prefix,
        label_kind="mixed_group",
    )


def compile_gauge_kinetic_bilinear_terms(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the two-point part of ``-1/4 F_{mu nu} F^{mu nu}``."""
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}")
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}")
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Gauge kinetic compilation",
    )
    identity_factor, left_slots, right_slots = _spectator_identity_factor(
        gauge_field,
        exclude_slots={gauge_lorentz_slot},
    )

    left_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, **left_slots})
    right_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, **right_slots})
    shared_fields = (
        gauge_field.occurrence(labels=left_field_labels),
        gauge_field.occurrence(labels=right_field_labels),
    )
    prefix = label_prefix + " " if label_prefix else ""

    return (
        InteractionTerm(
            coupling=-coefficient * _HALF * identity_factor * lorentz_metric(alpha, beta),
            fields=shared_fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho),
                DerivativeAction(target=1, lorentz_index=rho),
            ),
            label=prefix + f"{gauge_group.name}: gauge kinetic bilinear (metric)",
        ),
        InteractionTerm(
            coupling=(
                coefficient
                * _HALF
                * identity_factor
                * lorentz_metric(rho_left, beta)
                * lorentz_metric(rho_right, alpha)
            ),
            fields=shared_fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=prefix + f"{gauge_group.name}: gauge kinetic bilinear (cross)",
        ),
    )


def compile_yang_mills_cubic_term(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the cubic Yang-Mills term from ``-1/4 F^a_{mu nu} F^{a mu nu}``."""
    if gauge_group.abelian:
        raise ValueError("Abelian gauge groups do not have Yang-Mills cubic self-interactions.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta, gamma = _default_gauge_lorentz_labels(gauge_field, gauge_group, 3)
    adj_left, adj_middle, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 3)
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Yang-Mills cubic compilation",
    )
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot."
        )
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}_cubic")
    coupling = (
        coefficient
        * gauge_group.coupling
        * _build_structure_constant(gauge_group, adj_left, adj_middle, adj_right)
        * lorentz_metric(alpha, gamma)
        * lorentz_metric(rho, beta)
    )

    return InteractionTerm(
        coupling=coupling,
        fields=(
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, adj_slot: adj_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, adj_slot: adj_middle})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: gamma, adj_slot: adj_right})
            ),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: Yang-Mills cubic",
    )


def compile_yang_mills_quartic_term(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the quartic Yang-Mills term from ``-1/4 F^a_{mu nu} F^{a mu nu}``."""
    if gauge_group.abelian:
        raise ValueError("Abelian gauge groups do not have Yang-Mills quartic self-interactions.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta, gamma, delta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 4)
    adj_left, adj_mid_left, adj_mid_right, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 4)
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Yang-Mills quartic compilation",
    )
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot."
        )
    internal = _default_internal_adjoint_label(gauge_field, gauge_group)
    coupling = (
        -coefficient
        * _QUARTER
        * (gauge_group.coupling ** 2)
        * _build_structure_constant(gauge_group, adj_left, adj_mid_left, internal)
        * _build_structure_constant(gauge_group, adj_mid_right, adj_right, internal)
        * lorentz_metric(alpha, gamma)
        * lorentz_metric(beta, delta)
    )

    return InteractionTerm(
        coupling=coupling,
        fields=(
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, adj_slot: adj_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, adj_slot: adj_mid_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: gamma, adj_slot: adj_mid_right})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: delta, adj_slot: adj_right})
            ),
        ),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: Yang-Mills quartic",
    )


def compile_gauge_kinetic_term(model: Model, term: GaugeKineticTerm) -> tuple[InteractionTerm, ...]:
    """Compile ``-1/4 F_{mu nu} F^{mu nu}`` for one declared gauge group.

    For abelian groups this yields only the gauge-boson bilinear.
    For non-abelian groups it also appends the Yang-Mills cubic and quartic
    self-interaction terms.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Gauge-kinetic compilation",
    )

    gauge_field = model.gauge_boson_field(gauge_group)
    label_prefix = term.label or f"-1/4 {gauge_group.name} field strength squared"

    interactions = list(
        compile_gauge_kinetic_bilinear_terms(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    if gauge_group.abelian:
        return tuple(interactions)

    interactions.append(
        compile_yang_mills_cubic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    interactions.append(
        compile_yang_mills_quartic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    return tuple(interactions)


def compile_gauge_fixing_term(model: Model, term: GaugeFixingTerm) -> tuple[InteractionTerm, ...]:
    """Compile the ordinary linear-covariant gauge-fixing bilinear.

    The supported form is ``-(coefficient / 2 xi) (partial.A)^2`` for one
    declared gauge group.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Gauge-fixing compilation",
    )
    if _is_explicit_zero(term.xi):
        raise ValueError("Gauge-fixing compilation requires xi to be non-zero.")
    gauge_field = model.gauge_boson_field(gauge_group)
    if gauge_field.kind != "vector":
        raise ValueError(f"Gauge-fixing compilation requires a vector field, got kind={gauge_field.kind!r}.")
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Gauge-fixing compilation",
    )

    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}_gf")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}_gf")
    identity_factor, left_slots, right_slots = _spectator_identity_factor(
        gauge_field,
        exclude_slots={gauge_lorentz_slot},
    )

    coupling = (
        -term.coefficient
        * _HALF
        / term.xi
        * identity_factor
        * lorentz_metric(alpha, rho_left)
        * lorentz_metric(beta, rho_right)
    )
    left_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, **left_slots})
    right_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, **right_slots})
    label = term.label or f"-(1/2 {term.xi}) ({gauge_group.name} gauge fixing)"

    return (
        InteractionTerm(
            coupling=coupling,
            fields=(
                gauge_field.occurrence(labels=left_field_labels),
                gauge_field.occurrence(labels=right_field_labels),
            ),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=label,
        ),
    )


def compile_ghost_term(model: Model, term: GhostTerm) -> tuple[InteractionTerm, ...]:
    """Compile the ordinary unbroken Faddeev-Popov ghost sector.

    For the current conventions this corresponds to:
    ``-cbar^a partial^mu(D_mu c)^a = (partial cbar)(partial c) - g f (partial cbar) A c``.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Ghost compilation",
    )
    if gauge_group.abelian:
        raise ValueError(
            f"Ghost compilation is only supported for non-abelian gauge groups; got {gauge_group.name!r}."
        )

    gauge_field = model.gauge_boson_field(gauge_group)
    ghost_field = _ghost_field_for_group(model, gauge_group)

    adj_kind, gauge_adj_slot = _adjoint_slot_info(
        gauge_field,
        purpose="Ghost compilation",
    )
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Ghost compilation",
    )
    ghost_adj_slot = _unique_slot(
        ghost_field,
        adj_kind,
        purpose="Ghost compilation",
    )

    mu = _default_vector_label(gauge_field, gauge_group, suffix="mu")
    nu = _default_vector_label(gauge_field, gauge_group, suffix="nu_ghost")
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}_ghost")
    a_bar, a_ghost = _default_ghost_labels(
        ghost_field,
        ghost_field.indices[ghost_adj_slot],
        slot=ghost_adj_slot,
    )
    a_gauge = _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_ghost")

    ghost_bar_labels = ghost_field.pack_slot_labels({ghost_adj_slot: a_bar})
    ghost_labels = ghost_field.pack_slot_labels({ghost_adj_slot: a_ghost})
    gauge_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: mu, gauge_adj_slot: a_gauge})
    label_prefix = term.label or f"{gauge_group.name} Faddeev-Popov ghosts"

    ghost_bilinear = InteractionTerm(
        coupling=(
            term.coefficient
            * ghost_field.indices[ghost_adj_slot].representation.g(a_bar, a_ghost).to_expression()
            * lorentz_metric(mu, nu)
        ),
        fields=(
            ghost_field.occurrence(conjugated=True, labels=ghost_bar_labels),
            ghost_field.occurrence(labels=ghost_labels),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=1, lorentz_index=nu),
        ),
        label=label_prefix + " bilinear",
    )

    ghost_gauge = InteractionTerm(
        coupling=(
            -term.coefficient
            * gauge_group.coupling
            * _build_structure_constant(gauge_group, a_bar, a_gauge, a_ghost)
            * lorentz_metric(rho, mu)
        ),
        fields=(
            ghost_field.occurrence(conjugated=True, labels=ghost_bar_labels),
            gauge_field.occurrence(labels=gauge_labels),
            ghost_field.occurrence(labels=ghost_labels),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=label_prefix + " gauge interaction",
    )

    return (ghost_bilinear, ghost_gauge)


def compile_minimal_gauge_interactions(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile minimal gauge interactions using kinetic-term conventions.

    The generated interactions are the gauge pieces implied by the standard
    covariant kinetic terms:
    - fermions: ``i psibar gamma^mu D_mu psi``
    - complex scalars: ``(D_mu phi)^dagger (D^mu phi)``

    This keeps the standalone minimal compiler consistent with the declarative
    ``CovD(...)`` path exposed by ``Model.lagrangian()``.
    """
    interactions: list[InteractionTerm] = []

    for gauge_group in model.gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)

        for field in model.fields:
            if field == gauge_field:
                continue

            if field.kind == "fermion":
                if gauge_group.abelian:
                    if gauge_group.charge is None or field.quantum_numbers.get(gauge_group.charge, 0) == 0:
                        continue
                else:
                    if gauge_group.matter_representation(field) is None:
                        continue

                interactions.extend(
                    compile_fermion_gauge_current(
                        fermion=field,
                        gauge_group=gauge_group,
                        gauge_field=gauge_field,
                        prefactor=-1,
                    )
                )
                continue

            if field.kind == "scalar" and not field.self_conjugate:
                if gauge_group.abelian:
                    if gauge_group.charge is None or field.quantum_numbers.get(gauge_group.charge, 0) == 0:
                        continue
                else:
                    if gauge_group.matter_representation(field) is None:
                        continue
                interactions.extend(
                    compile_complex_scalar_gauge_terms(
                        scalar=field,
                        gauge_group=gauge_group,
                        gauge_field=gauge_field,
                        current_prefactor=Expression.I,
                        contact_prefactor=1,
                    )
                )

    return tuple(interactions)


def with_minimal_gauge_interactions(model: Model) -> Model:
    """Return a copy of a model with compiled gauge interactions appended."""
    compiled = compile_minimal_gauge_interactions(model)
    return replace(model, interactions=model.interactions + compiled)


def compile_dirac_kinetic_term(model: Model, term: DiracKineticTerm) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``psibar i gamma^mu D_mu psi``.

    One model declaration can expand into several interactions when the same
    fermion transforms under multiple declared gauge groups.
    """
    fermion = _require_declared_field(
        model,
        term.field,
        purpose="Dirac kinetic compilation",
    )
    if fermion.kind != "fermion":
        raise ValueError(f"Dirac kinetic term requires a fermion field, got kind={fermion.kind!r}.")

    label = term.label or f"i {fermion.name}bar gamma^mu D_mu {fermion.name}"
    mu = _symbol("mu")
    expanded = expand_cov_der(
        model,
        CovD(fermion, mu),
        gauge_group=term.gauge_group,
    )
    fermion_spinor_slot = _unique_slot(
        fermion,
        SPINOR_KIND,
        purpose="Dirac kinetic compilation",
    )

    interactions: list[InteractionTerm] = []
    for piece in expanded.gauge_current_pieces:
        gauge_group = piece.metadata.gauge_group
        i_bar, i_psi = _default_spinor_labels(fermion, gauge_group)
        interactions.append(
            _build_fermion_current_interaction(
                fermion=fermion,
                piece=piece,
                spinor_slot=fermion_spinor_slot,
                i_bar=i_bar,
                i_psi=i_psi,
                prefactor=-term.coefficient,
                label=label,
                lorentz_label=mu,
                spectator_exclude_slots={fermion_spinor_slot},
            )
        )

    return tuple(interactions)


def _compile_dirac_partial_term(fermion: Field, *, coefficient=1, label: str = "") -> InteractionTerm:
    mu = _symbol("mu")
    i_bar = _symbol(f"i_bar_{fermion.name}_covd")
    i_psi = _symbol(f"i_{fermion.name}_covd")
    fermion_spinor_slot = _unique_slot(
        fermion,
        SPINOR_KIND,
        purpose="Dirac kinetic partial-term compilation",
    )
    bar_slot_labels = {fermion_spinor_slot: i_bar}
    psi_slot_labels = {fermion_spinor_slot: i_psi}
    core_factor, core_bar_slots, core_psi_slots = _spectator_identity_factor(
        fermion,
        exclude_slots={fermion_spinor_slot},
    )
    bar_slot_labels.update(core_bar_slots)
    psi_slot_labels.update(core_psi_slots)
    bar_labels = fermion.pack_slot_labels(bar_slot_labels)
    psi_labels = fermion.pack_slot_labels(psi_slot_labels)
    return InteractionTerm(
        coupling=Expression.I * coefficient * core_factor * psi_bar_gamma_psi(i_bar, i_psi, mu),
        fields=(
            fermion.occurrence(conjugated=True, labels=bar_labels),
            fermion.occurrence(labels=psi_labels),
        ),
        derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        label=label or f"i {fermion.name}bar gamma^mu d_mu {fermion.name}",
    )


def compile_complex_scalar_kinetic_term(
    model: Model,
    term: ComplexScalarKineticTerm,
) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``(D_mu phi)^dagger (D^mu phi)``.

    The output contains both the current term and the two-gauge contact term
    for each applicable gauge group.
    """
    scalar = _require_declared_field(
        model,
        term.field,
        purpose="Complex-scalar kinetic compilation",
    )
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError(
            "Complex-scalar kinetic terms require a non-self-conjugate scalar field."
        )

    label_prefix = term.label or f"(D_mu {scalar.name})^dagger (D^mu {scalar.name})"
    gauge_groups = _resolve_covariant_gauge_groups(
        model,
        field=scalar,
        gauge_group=term.gauge_group,
    )

    interactions = []
    resolved_gauge_fields = []
    for gauge_group in gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)
        resolved_gauge_fields.append((gauge_group, gauge_field))
        interactions.extend(
            compile_complex_scalar_gauge_terms(
                scalar=scalar,
                gauge_group=gauge_group,
                gauge_field=gauge_field,
                current_prefactor=Expression.I * term.coefficient,
                contact_prefactor=term.coefficient,
                label_prefix=label_prefix,
            )
        )

    for left_idx, (left_gauge_group, left_gauge_field) in enumerate(resolved_gauge_fields):
        for right_idx, (right_gauge_group, right_gauge_field) in enumerate(resolved_gauge_fields):
            if left_idx == right_idx:
                continue
            interactions.extend(
                compile_mixed_complex_scalar_contact_terms(
                    scalar=scalar,
                    left_gauge_group=left_gauge_group,
                    left_gauge_field=left_gauge_field,
                    right_gauge_group=right_gauge_group,
                    right_gauge_field=right_gauge_field,
                    contact_prefactor=term.coefficient,
                    label_prefix=label_prefix,
                )
            )
    return tuple(interactions)


def _compile_complex_scalar_partial_term(scalar: Field, *, coefficient=1, label: str = "") -> InteractionTerm:
    mu = _symbol("mu")
    core_factor, scalar_bar_slots, scalar_slots = _spectator_identity_factor(scalar, exclude_slots=())
    scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slots)
    scalar_labels = scalar.pack_slot_labels(scalar_slots)
    return InteractionTerm(
        coupling=coefficient * core_factor,
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
            scalar.occurrence(labels=scalar_labels),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=1, lorentz_index=mu),
        ),
        label=label or f"(d_mu {scalar.name})^dagger (d^mu {scalar.name})",
    )

def _assemble_full_covariant_operator(
    gauge_terms: tuple[InteractionTerm, ...],
    partial_term: InteractionTerm,
    spectator_factor: Expression,
    spectator_occurrences: tuple[tuple[int, int], ...],
) -> tuple[InteractionTerm, ...]:
    """Assemble the full covariant operator from gauge and partial-derivative terms.

    Both gauge and partial terms are decorated with spectators (if any) and
    combined to form the complete declared CovD monomial semantics.
    """
    gauge_decorated = _decorate_interactions_with_spectators(
        gauge_terms,
        spectator_factor=spectator_factor,
        spectator_occurrences=spectator_occurrences,
    )
    partial_decorated = _decorate_interactions_with_spectators(
        (partial_term,),
        spectator_factor=spectator_factor,
        spectator_occurrences=spectator_occurrences,
    )
    return gauge_decorated + partial_decorated

def _compile_declared_covariant_core(
    model: Model,
    core: DiracKineticTerm | ComplexScalarKineticTerm,
    spectators: tuple[tuple[Field, bool], ...] = (),
) -> tuple[InteractionTerm, ...]:
    """Compile one declared ``CovD`` monomial as the full kinetic operator.

    Declarative covariant-derivative monomials represent the full source
    operator, not only the gauge-current part.  The lowering therefore always
    emits both pieces:
    - the free partial-derivative bilinear
    - the gauge-interaction terms generated from the same core

    Any spectator fields are attached uniformly to both pieces so that plain and
    spectator-decorated declarative ``CovD`` monomials share the same
    semantics.
    """
    spectator_factor, spectator_occurrences = _materialize_spectator_occurrences(spectators)

    if isinstance(core, DiracKineticTerm):
        fermion = _require_declared_field(
            model,
            core.field,
            purpose="Covariant monomial compilation",
        )
        if fermion.kind != "fermion":
            raise ValueError(
                f"Covariant Dirac monomial requires a fermion field, got kind={fermion.kind!r}."
            )
        gauge_terms = compile_dirac_kinetic_term(model, core)
        partial_term = _compile_dirac_partial_term(
            fermion,
            coefficient=core.coefficient,
            label=core.label or f"i {fermion.name}bar gamma^mu D_mu {fermion.name} partial",
        )
        return _assemble_full_covariant_operator(
            gauge_terms,
            partial_term,
            spectator_factor,
            spectator_occurrences,
        )

    if isinstance(core, ComplexScalarKineticTerm):
        scalar = _require_declared_field(
            model,
            core.field,
            purpose="Covariant monomial compilation",
        )
        if scalar.kind != "scalar" or scalar.self_conjugate:
            raise ValueError(
                "Covariant complex-scalar monomials require a non-self-conjugate scalar field."
            )
        gauge_terms = compile_complex_scalar_kinetic_term(model, core)
        partial_term = _compile_complex_scalar_partial_term(
            scalar,
            coefficient=core.coefficient,
            label=core.label or f"(D_mu {scalar.name})^dagger (D^mu {scalar.name}) derivative",
        )
        return _assemble_full_covariant_operator(
            gauge_terms,
            partial_term,
            spectator_factor,
            spectator_occurrences,
        )

    raise TypeError(f"Unsupported covariant monomial core type: {type(core)!r}")


def compile_covariant_terms(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile all declared non-local / structured source terms in a model.

    Local interaction monomials are handled separately by ``Model.all_interactions()``.
    This function expands only the declared terms that require compilation:
    covariant-derivative monomials, field-strength terms, gauge fixing, ghosts,
    and the legacy physical declaration slots.

    Declarative ``CovD(...)`` monomials are compiled as full operators, so their
    output includes the free bilinear partial-derivative contribution alongside
    the gauge-interaction pieces.  Legacy ``DiracKineticTerm`` and
    ``ComplexScalarKineticTerm`` declarations keep their existing gauge-only
    behavior.
    """
    interactions: list[InteractionTerm] = []

    for analyzed in model.analyzed_source_terms():
        if analyzed.interaction is not None:
            continue

        if analyzed.covariant_core is not None:
            interactions.extend(
                _compile_declared_covariant_core(
                    model,
                    analyzed.covariant_core,
                    analyzed.covariant_spectators,
                )
            )
            continue

        if analyzed.gauge_kinetic is not None:
            interactions.extend(compile_gauge_kinetic_term(model, analyzed.gauge_kinetic))
            continue

        if analyzed.gauge_fixing is not None:
            interactions.extend(compile_gauge_fixing_term(model, analyzed.gauge_fixing))
            continue

        if analyzed.ghost is not None:
            interactions.extend(compile_ghost_term(model, analyzed.ghost))

    for term in model.covariant_terms:
        if isinstance(term, DiracKineticTerm):
            interactions.extend(compile_dirac_kinetic_term(model, term))
            continue
        if isinstance(term, ComplexScalarKineticTerm):
            interactions.extend(compile_complex_scalar_kinetic_term(model, term))
            continue
        raise TypeError(f"Unsupported covariant term type: {type(term)!r}")

    for term in model.gauge_kinetic_terms:
        interactions.extend(compile_gauge_kinetic_term(model, term))

    for term in model.gauge_fixing_terms:
        interactions.extend(compile_gauge_fixing_term(model, term))

    for term in model.ghost_terms:
        interactions.extend(compile_ghost_term(model, term))

    return tuple(interactions)


def with_compiled_covariant_terms(model: Model) -> Model:
    """Return a copy of a model with compiled physical kinetic terms appended.

    The returned model has empty declaration slots (covariant_terms,
    gauge_kinetic_terms, gauge_fixing_terms, ghost_terms) so that a
    subsequent call to ``Model.lagrangian()`` does not re-compile and
    double-count the same terms.
    """
    compiled = compile_covariant_terms(model)
    preserved_source_terms = tuple(
        analyzed.term
        for analyzed in model.analyzed_source_terms()
        if not analyzed.needs_compilation
    )
    return replace(
        model,
        interactions=model.interactions + compiled,
        lagrangian_decl=type(model.lagrangian_decl)(source_terms=preserved_source_terms),
        covariant_terms=(),
        gauge_kinetic_terms=(),
        gauge_fixing_terms=(),
        ghost_terms=(),
    )
