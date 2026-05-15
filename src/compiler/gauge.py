"""
Gauge-structure compilers for the Symbolica/Spenso prototype.

This module exposes two layers on purpose:

- the minimal compiler: structural gauge interactions from metadata
- the covariant/physical compiler: convention-fixed kinetic-term expansion

Frozen conventions for the physical path:

- Fourier transform derivatives act as ``-i p_mu``
- ``vertex_factor(...)`` contributes the universal overall ``+i``
- matter covariant derivatives use ``D_mu = partial_mu - i g A_mu``
- non-abelian field strengths use
  ``F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu + g f^{abc} A^b_mu A^c_nu``

With these choices, matter currents follow the same sign convention as the
local gauge-ready examples, the Yang-Mills 3-gauge vertex is real, and the
4-gauge vertex keeps an explicit overall ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

from symbolica import S, Expression

from .covariant_core import (
    _compile_covariant_core as _compile_covariant_core_impl,
    _compile_declared_covariant_core as _compile_declared_covariant_core_impl,
)
from .matter_actions import (
    _build_fermion_current_interaction,
    _compile_scalar_contact_terms,
    _compile_scalar_current_from_piece,
)
from .spectators import (
    _decorate_interactions_with_spectators,
    _default_index_labels,
    _materialize_spectator_occurrences,
    _spectator_identity_factor,
)
from model import (
    CovD,
    CovariantDerivativeFactor,
    DerivativeAction,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    InteractionTerm,
    Model,
    PartialD,
)
from model.declared import (
    _DeclaredMonomial,
    _FieldFactor,
    DifferentiatedCovariantFactor,
    GeneratorFactor,
    PartialDerivativeFactor,
)
from model.lagrangian import (
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    GaugeFixingTerm,
    GaugeKineticTerm,
    GhostTerm,
)
from model.lowering import _analyze_declared_source_term, _lower_local_interaction_monomial
from symbolic.spenso_structures import lorentz_metric
from model.metadata import (
    is_lorentz_index,
    is_spinor_index,
    lorentz_kind_for,
    lorentz_slots_for,
    spinor_kind_for,
    spinor_slots_for,
    unique_lorentz_slot,
    unique_spinor_slot,
)


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


def _first_non_lorentz_index_slot(field: Field) -> Optional[int]:
    for slot, index in enumerate(field.indices):
        if not is_lorentz_index(index):
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
    if adj_kind == "color_adj":
        stem = "a_mid"
    elif adj_kind == "weak_adj":
        stem = "aw_mid"
    else:
        stem = f"{adj_kind}_mid"
    return _symbol(f"{stem}_{gauge_field.name}_{gauge_group.name}")


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


def _ghost_associated_gauge_boson_matches(
    model: Model,
    field: Field,
    gauge_group: GaugeGroup,
) -> bool:
    if not field.is_ghost or field.ghost_of is None:
        return True

    gauge_boson = model.gauge_boson_field(gauge_group)
    associated = model.find_field(field.ghost_of)
    if associated is not None:
        return gauge_boson is associated

    target_text = str(field.ghost_of)
    return target_text in (
        gauge_boson.name,
        str(gauge_boson.symbol),
        str(gauge_group.gauge_boson),
    )


def _maybe_nonabelian_rep_and_slots(
    field: Field,
    gauge_group: GaugeGroup,
    *,
    model: Optional[Model] = None,
    gauge_field: Optional[Field] = None,
) -> Optional[tuple[GaugeRepresentation, tuple[int, ...]]]:
    rep_info = gauge_group.matter_representation_and_slots(field)
    if rep_info is not None:
        return rep_info

    if model is None or gauge_field is None:
        return None
    if gauge_group.structure_constant is None or not callable(gauge_group.structure_constant):
        return None

    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_slot is None:
        return None
    adjoint_index = gauge_field.indices[adj_slot]
    inferred = GaugeRepresentation(
        index=adjoint_index,
        generator_builder=gauge_group.structure_constant,
        name="adjoint",
    )
    slots = inferred.slots_for(field)
    if not slots:
        return None
    return inferred, tuple(slots)


def _covd_field_transforms_under_gauge_group(
    model: Model,
    field: Field,
    gauge_group: GaugeGroup,
) -> bool:
    if gauge_group.abelian:
        if gauge_group.charge is None:
            return False
        return field.quantum_numbers.get(gauge_group.charge, 0) != 0

    if not _ghost_associated_gauge_boson_matches(model, field, gauge_group):
        return False

    gauge_field = model.gauge_boson_field(gauge_group)
    return _maybe_nonabelian_rep_and_slots(
        field,
        gauge_group,
        model=model,
        gauge_field=gauge_field,
    ) is not None


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


def _require_covd_field_transforms_under_gauge_group(
    model: Model,
    field: Field,
    gauge_group: GaugeGroup,
    *,
    purpose: str,
):
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

    if not _ghost_associated_gauge_boson_matches(model, field, gauge_group):
        raise ValueError(
            f"{purpose} requires ghost field {field.name!r} to be associated with "
            f"the gauge boson of gauge group {gauge_group.name!r}."
        )

    gauge_field = model.gauge_boson_field(gauge_group)
    if _maybe_nonabelian_rep_and_slots(
        field,
        gauge_group,
        model=model,
        gauge_field=gauge_field,
    ) is None:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to carry a declared "
            f"representation under gauge group {gauge_group.name!r}, or the "
            "group adjoint index so an adjoint action can be inferred."
        )


def _resolve_covariant_gauge_groups(model: Model, *, field: Field, gauge_group=None) -> tuple[GaugeGroup, ...]:
    purpose = f"Covariant compilation for field {field.name!r}"
    if gauge_group is not None:
        if isinstance(gauge_group, (tuple, list)):
            resolved = []
            for item in gauge_group:
                group = _require_declared_gauge_group(model, item, purpose=purpose)
                _require_covd_field_transforms_under_gauge_group(
                    model,
                    field,
                    group,
                    purpose=purpose,
                )
                resolved.append(group)
            return tuple(resolved)

        resolved = _require_declared_gauge_group(model, gauge_group, purpose=purpose)
        _require_covd_field_transforms_under_gauge_group(
            model,
            field,
            resolved,
            purpose=purpose,
        )
        return (resolved,)

    matches = tuple(
        group
        for group in model.gauge_groups
        if _covd_field_transforms_under_gauge_group(model, field, group)
    )
    if not matches:
        raise ValueError(f"Field {field.name!r} does not transform under any declared gauge group.")
    return matches


def _nonabelian_rep_and_slots(
    field: Field,
    gauge_group: GaugeGroup,
    *,
    model: Optional[Model] = None,
    gauge_field: Optional[Field] = None,
):
    rep_info = _maybe_nonabelian_rep_and_slots(
        field,
        gauge_group,
        model=model,
        gauge_field=gauge_field,
    )
    if rep_info is None:
        raise ValueError(
            f"Field {field.name!r} carries no representation declared or inferred for "
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
    if not ghost_field.is_ghost:
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to have kind='ghost'."
        )
    if ghost_field.self_conjugate:
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to be non-self-conjugate."
        )
    if not _ghost_associated_gauge_boson_matches(model, ghost_field, gauge_group):
        raise ValueError(
            f"Ghost compilation requires ghost field {ghost_field.name!r} to be associated "
            f"with the gauge boson of gauge group {gauge_group.name!r}."
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
    representation: Optional[GaugeRepresentation]
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
    representation: Optional[GaugeRepresentation] = None
    representation_slot: Optional[int] = None
    adjoint_label: Optional[object] = None

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
        gauge_lorentz_slot = unique_lorentz_slot(
            gauge_field,
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


@dataclass(frozen=True)
class _GaugeFieldLayout:
    """Slot layout for pure-gauge field occurrences."""

    field: Field
    lorentz_slot: int
    adjoint_kind: Optional[str] = None
    adjoint_slot: Optional[int] = None

    @classmethod
    def from_field(
        cls,
        gauge_field: Field,
        *,
        purpose: str,
        require_adjoint: bool = False,
    ) -> "_GaugeFieldLayout":
        if gauge_field.kind != "vector":
            raise ValueError(f"{purpose} requires a vector field, got kind={gauge_field.kind!r}.")

        lorentz_slot = unique_lorentz_slot(gauge_field, purpose=purpose)
        adjoint_kind = None
        adjoint_slot = None
        if require_adjoint:
            adjoint_kind, adjoint_slot = _adjoint_slot_info(
                gauge_field,
                purpose=purpose,
            )

        return cls(
            field=gauge_field,
            lorentz_slot=lorentz_slot,
            adjoint_kind=adjoint_kind,
            adjoint_slot=adjoint_slot,
        )

    def occurrence(self, lorentz_label, adjoint_label=None):
        slot_labels = {self.lorentz_slot: lorentz_label}
        if adjoint_label is not None:
            if self.adjoint_slot is None:
                raise ValueError(f"Gauge field {self.field.name!r} has no resolved adjoint slot.")
            slot_labels[self.adjoint_slot] = adjoint_label
        return self.field.occurrence(labels=self.field.pack_slot_labels(slot_labels))

    def bilinear_occurrences(self, left_lorentz_label, right_lorentz_label):
        identity_factor, left_slots, right_slots = _spectator_identity_factor(
            self.field,
            exclude_slots={self.lorentz_slot},
        )
        return identity_factor, (
            self.field.occurrence(
                labels=self.field.pack_slot_labels({
                    self.lorentz_slot: left_lorentz_label,
                    **left_slots,
                })
            ),
            self.field.occurrence(
                labels=self.field.pack_slot_labels({
                    self.lorentz_slot: right_lorentz_label,
                    **right_slots,
                })
            ),
        )


@dataclass(frozen=True)
class _GenericCovariantBranch:
    """One branch in a generic ``CovD(...)`` monomial expansion."""

    coefficient: object = 1
    inline_factors: tuple[object, ...] = ()
    tail_factors: tuple[object, ...] = ()


def _gauge_piece_action_key(piece: CovariantGaugePiece) -> tuple[str, str, str, Optional[int]]:
    """Stable key for matching the same gauge action across conjugate expansions."""
    return (
        piece.metadata.gauge_group.name,
        piece.metadata.gauge_field.name,
        piece.metadata.representation_name,
        piece.active_slot,
    )


def _format_gauge_piece_action_key(key: tuple[str, str, str, Optional[int]]) -> str:
    group_name, gauge_field_name, representation_name, active_slot = key
    slot = "abelian" if active_slot is None else f"slot {active_slot + 1}"
    return f"{group_name}/{gauge_field_name}/{representation_name}/{slot}"


def _pair_gauge_pieces_by_action(
    *,
    right_pieces: tuple[CovariantGaugePiece, ...],
    left_pieces: tuple[CovariantGaugePiece, ...],
    purpose: str,
) -> tuple[tuple[CovariantGaugePiece, CovariantGaugePiece], ...]:
    """Pair ordinary and conjugated gauge actions by meaning, not tuple order."""
    left_by_key: dict[tuple[str, str, str, Optional[int]], CovariantGaugePiece] = {}
    for piece in left_pieces:
        key = _gauge_piece_action_key(piece)
        if key in left_by_key:
            raise ValueError(
                f"{purpose} found duplicate conjugated gauge action "
                f"{_format_gauge_piece_action_key(key)}."
            )
        left_by_key[key] = piece

    paired: list[tuple[CovariantGaugePiece, CovariantGaugePiece]] = []
    missing_left: list[tuple[str, str, str, Optional[int]]] = []
    for right_piece in right_pieces:
        key = _gauge_piece_action_key(right_piece)
        left_piece = left_by_key.pop(key, None)
        if left_piece is None:
            missing_left.append(key)
            continue
        paired.append((right_piece, left_piece))

    if missing_left or left_by_key:
        missing = ", ".join(_format_gauge_piece_action_key(key) for key in missing_left)
        extra = ", ".join(_format_gauge_piece_action_key(key) for key in left_by_key)
        details = []
        if missing:
            details.append(f"missing conjugated actions for: {missing}")
        if extra:
            details.append(f"unmatched conjugated actions: {extra}")
        raise ValueError(f"{purpose} could not pair gauge actions; " + "; ".join(details))

    return tuple(paired)


def _active_representation_slots(
    pieces: tuple[CovariantGaugePiece, ...],
) -> tuple[int, ...]:
    """Return active non-abelian matter slots in first-seen order."""
    return tuple(
        dict.fromkeys(piece.active_slot for piece in pieces if piece.active_slot is not None)
    )


def _reject_ambiguous_manual_label_overrides(
    *,
    field: Field,
    pieces: tuple[CovariantGaugePiece, ...],
    overrides: tuple[tuple[str, object], ...],
    purpose: str,
) -> None:
    """Reject global manual labels when one declaration expands over many slots.

    The public override parameters name one set of labels.  When a
    slot_policy='sum' representation expands into several active slots, applying
    that same override to every slot is ambiguous, so callers should either omit
    the override or declare a single explicit GaugeRepresentation(slot=...).
    """
    provided = tuple(name for name, value in overrides if value is not None)
    if not provided:
        return

    active_slots = _active_representation_slots(pieces)
    if len(active_slots) <= 1:
        return

    slots = ", ".join(str(slot + 1) for slot in active_slots)
    raise ValueError(
        f"{purpose} received global manual label override(s) "
        f"{', '.join(provided)} for field {field.name!r}, but the gauge action "
        f"expands over multiple active representation slots ({slots}). Omit the "
        "override, or declare a single explicit GaugeRepresentation(slot=...)."
    )


def _validate_expandable_covariant_field(
    field: Field,
    *,
    purpose: str,
):
    if field.self_conjugate:
        raise ValueError(
            f"{purpose} currently supports only non-self-conjugate matter fields; "
            f"field {field.name!r} is self-conjugate."
        )


def _covariant_gauge_metadata(
    model: Optional[Model],
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
        rep, rep_slots = _nonabelian_rep_and_slots(
            field,
            gauge_group,
            model=model,
            gauge_field=gauge_field,
        )

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
    model: Optional[Model] = None,
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
        model,
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
                model=model,
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


def _fresh_generic_covd_label(prefix: str, counters: dict[str, int], stem: str) -> object:
    counters[prefix] = counters.get(prefix, 0) + 1
    return _symbol(f"{prefix}_{stem}_{counters[prefix]}")


def _generic_covd_piece_prefactor(
    *,
    conjugated: bool,
    action: _GaugeAction,
):
    sign = 1 if conjugated else -1
    if action.representation is not None:
        gauge_adj_slot = _adjoint_index_slot(action.gauge_field)
        if (
            gauge_adj_slot is not None
            and action.representation.index == action.gauge_field.indices[gauge_adj_slot]
        ):
            return sign * action.coupling
    return sign * Expression.I * action.coupling


def _generic_covd_matter_factor(
    field: Field,
    *,
    conjugated: bool,
    action: _GaugeAction,
    counters: dict[str, int],
) -> _FieldFactor:
    labels = {}
    if action.representation is not None and action.representation_slot is not None:
        label = _fresh_generic_covd_label(
            action.representation.index.prefix,
            counters,
            f"{field.name}_slot{action.representation_slot + 1}_covd",
        )
        labels = field.pack_slot_labels({action.representation_slot: label})
    return _FieldFactor(
        field=field,
        conjugated=conjugated,
        labels=labels,
    )


def _expand_generic_covd_factor(
    model: Model,
    cov_factor: CovariantDerivativeFactor,
    *,
    counters: dict[str, int],
) -> tuple[_GenericCovariantBranch, ...]:
    expanded = expand_cov_der(model, cov_factor)
    branches = [
        _GenericCovariantBranch(
            coefficient=Expression.num(1),
            inline_factors=(
                PartialDerivativeFactor(
                    field=expanded.field,
                    lorentz_indices=(expanded.derivative_piece.lorentz_index,),
                    conjugated=expanded.conjugated,
                    labels={},
                ),
            ),
        )
    ]

    for piece in expanded.gauge_current_pieces:
        adjoint_label = None
        if not piece.metadata.gauge_group.abelian:
            adj_kind, _ = _adjoint_slot_info(
                piece.metadata.gauge_field,
                purpose="Generic declared CovD lowering",
            )
            adjoint_label = _fresh_generic_covd_label(
                adj_kind,
                counters,
                f"{piece.metadata.gauge_field.name}_{piece.metadata.gauge_group.name}_covd",
            )

        action = _GaugeAction.from_piece(
            expanded.field,
            piece,
            lorentz_label=piece.lorentz_index,
            adjoint_label=adjoint_label,
            purpose="Generic declared CovD lowering",
        )
        matter_factor = _generic_covd_matter_factor(
            expanded.field,
            conjugated=expanded.conjugated,
            action=action,
            counters=counters,
        )
        inline_factors: list[object] = []
        if expanded.conjugated:
            inline_factors.append(matter_factor)
            if action.representation is not None:
                inline_factors.append(
                    GeneratorFactor(
                        action.adjoint_label,
                        generator_builder=action.representation.generator_builder,
                        index_kind=action.representation.index.kind,
                    )
                )
        else:
            if action.representation is not None:
                inline_factors.append(
                    GeneratorFactor(
                        action.adjoint_label,
                        generator_builder=action.representation.generator_builder,
                        index_kind=action.representation.index.kind,
                    )
                )
            inline_factors.append(matter_factor)

        branches.append(
            _GenericCovariantBranch(
                coefficient=_generic_covd_piece_prefactor(
                    conjugated=expanded.conjugated,
                    action=action,
                ),
                inline_factors=tuple(inline_factors),
                tail_factors=(
                    _FieldFactor(
                        field=action.gauge_field,
                        labels=action.gauge_labels(),
                    ),
                ),
            )
        )

    return tuple(branches)


def _differentiate_declared_branch_field_factor(factor, *, lorentz_index):
    if isinstance(factor, _FieldFactor):
        return PartialDerivativeFactor(
            field=factor.field,
            lorentz_indices=(lorentz_index,),
            conjugated=factor.conjugated,
            labels=factor.labels,
        )
    if isinstance(factor, PartialDerivativeFactor):
        return PartialDerivativeFactor(
            field=factor.field,
            lorentz_indices=factor.lorentz_indices + (lorentz_index,),
            conjugated=factor.conjugated,
            labels=factor.labels,
        )
    return None


def _differentiate_generic_covariant_branch(
    branch: _GenericCovariantBranch,
    *,
    lorentz_index,
) -> tuple[_GenericCovariantBranch, ...]:
    factors = list(branch.inline_factors + branch.tail_factors)
    inline_count = len(branch.inline_factors)
    differentiated: list[_GenericCovariantBranch] = []

    for idx, factor in enumerate(factors):
        differentiated_factor = _differentiate_declared_branch_field_factor(
            factor,
            lorentz_index=lorentz_index,
        )
        if differentiated_factor is None:
            continue
        updated = list(factors)
        updated[idx] = differentiated_factor
        differentiated.append(
            _GenericCovariantBranch(
                coefficient=branch.coefficient,
                inline_factors=tuple(updated[:inline_count]),
                tail_factors=tuple(updated[inline_count:]),
            )
        )

    if differentiated:
        return tuple(differentiated)
    raise ValueError(
        "Cannot apply PartialD(...) to expanded CovD(...) branch with no field factors."
    )


def _expand_differentiated_covd_factor(
    model: Model,
    factor: DifferentiatedCovariantFactor,
    *,
    counters: dict[str, int],
) -> tuple[_GenericCovariantBranch, ...]:
    branches = _expand_generic_covd_factor(
        model,
        factor.covariant_factor,
        counters=counters,
    )
    for lorentz_index in factor.lorentz_indices:
        next_branches: list[_GenericCovariantBranch] = []
        for branch in branches:
            next_branches.extend(
                _differentiate_generic_covariant_branch(
                    branch,
                    lorentz_index=lorentz_index,
                )
            )
        branches = tuple(next_branches)
    return branches


def _expand_generic_declared_covariant_monomial(
    model: Model,
    term: _DeclaredMonomial,
) -> tuple[_DeclaredMonomial, ...]:
    counters: dict[str, int] = {}
    branches = (_GenericCovariantBranch(),)

    for factor in term.factors:
        if isinstance(factor, CovariantDerivativeFactor):
            replacements = _expand_generic_covd_factor(
                model,
                factor,
                counters=counters,
            )
        elif isinstance(factor, DifferentiatedCovariantFactor):
            replacements = _expand_differentiated_covd_factor(
                model,
                factor,
                counters=counters,
            )
        else:
            branches = tuple(
                _GenericCovariantBranch(
                    coefficient=branch.coefficient,
                    inline_factors=branch.inline_factors + (factor,),
                    tail_factors=branch.tail_factors,
                )
                for branch in branches
            )
            continue
        next_branches: list[_GenericCovariantBranch] = []
        for branch in branches:
            for replacement in replacements:
                coefficient = branch.coefficient * replacement.coefficient
                if _is_explicit_zero(coefficient):
                    continue
                next_branches.append(
                    _GenericCovariantBranch(
                        coefficient=coefficient,
                        inline_factors=branch.inline_factors + replacement.inline_factors,
                        tail_factors=branch.tail_factors + replacement.tail_factors,
                    )
                )
        branches = tuple(next_branches)

    return tuple(
        _DeclaredMonomial(
            coefficient=term.coefficient * branch.coefficient,
            factors=branch.inline_factors + branch.tail_factors,
        )
        for branch in branches
        if not _is_explicit_zero(term.coefficient * branch.coefficient)
    )


def _compile_generic_declared_covariant_monomial(
    model: Model,
    term: _DeclaredMonomial,
) -> tuple[InteractionTerm, ...]:
    interactions: list[InteractionTerm] = []
    for expanded_term in _expand_generic_declared_covariant_monomial(model, term):
        interaction = _lower_local_interaction_monomial(expanded_term)
        if interaction is None:
            raise ValueError(
                "Could not lower a declarative monomial after expanding CovD(...). "
                "Use InteractionTerm(...) for unsupported non-local orderings."
            )
        interactions.append(interaction)
    return tuple(interactions)


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
    fermion_spinor_slot = unique_spinor_slot(
        fermion,
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
    _reject_ambiguous_manual_label_overrides(
        field=fermion,
        pieces=pieces,
        overrides=(
            ("matter_labels", matter_labels),
            ("adjoint_label", adjoint_label),
        ),
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
                gauge_action_from_piece=_GaugeAction.from_piece,
                slot_suffix=_slot_suffix,
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
    _reject_ambiguous_manual_label_overrides(
        field=scalar,
        pieces=right_pieces,
        overrides=(
            ("matter_labels", matter_labels),
            ("adjoint_labels", adjoint_labels),
            ("internal_label", internal_label),
        ),
        purpose="Complex-scalar gauge-term compilation",
    )

    current_terms: list[InteractionTerm] = []
    for right_piece, left_piece in _pair_gauge_pieces_by_action(
        right_pieces=right_pieces,
        left_pieces=left_pieces,
        purpose="Complex-scalar gauge-term compilation",
    ):
        current_terms.append(
            _compile_scalar_current_from_piece(
                scalar=scalar,
                piece=right_piece,
                derivative_target=1,
                gauge_action_from_piece=_GaugeAction.from_piece,
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
                gauge_action_from_piece=_GaugeAction.from_piece,
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
        gauge_action_from_piece=_GaugeAction.from_piece,
        symbol=_symbol,
        slot_suffix=_slot_suffix,
        mixed_scalar_contact_slot_suffix=_mixed_scalar_contact_slot_suffix,
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
    _reject_ambiguous_manual_label_overrides(
        field=scalar,
        pieces=left_pieces,
        overrides=(("left_adjoint_label", left_adjoint_label),),
        purpose="Mixed scalar contact compilation",
    )
    _reject_ambiguous_manual_label_overrides(
        field=scalar,
        pieces=right_pieces,
        overrides=(("right_adjoint_label", right_adjoint_label),),
        purpose="Mixed scalar contact compilation",
    )

    return _compile_scalar_contact_terms(
        scalar=scalar,
        left_pieces=left_pieces,
        right_pieces=right_pieces,
        gauge_action_from_piece=_GaugeAction.from_piece,
        symbol=_symbol,
        slot_suffix=_slot_suffix,
        mixed_scalar_contact_slot_suffix=_mixed_scalar_contact_slot_suffix,
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
    layout = _GaugeFieldLayout.from_field(
        gauge_field,
        purpose="Gauge kinetic compilation",
    )
    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}")
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}")
    identity_factor, shared_fields = layout.bilinear_occurrences(alpha, beta)
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
    layout = _GaugeFieldLayout.from_field(
        gauge_field,
        purpose="Yang-Mills cubic compilation",
        require_adjoint=True,
    )
    alpha, beta, gamma = _default_gauge_lorentz_labels(gauge_field, gauge_group, 3)
    adj_left, adj_middle, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 3)
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}_cubic")
    coupling = (
        coefficient
        * (-gauge_group.coupling)
        * _build_structure_constant(gauge_group, adj_left, adj_middle, adj_right)
        * lorentz_metric(alpha, gamma)
        * lorentz_metric(rho, beta)
    )

    return InteractionTerm(
        coupling=coupling,
        fields=(
            layout.occurrence(alpha, adj_left),
            layout.occurrence(beta, adj_middle),
            layout.occurrence(gamma, adj_right),
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
    layout = _GaugeFieldLayout.from_field(
        gauge_field,
        purpose="Yang-Mills quartic compilation",
        require_adjoint=True,
    )
    alpha, beta, gamma, delta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 4)
    adj_left, adj_mid_left, adj_mid_right, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 4)
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
            layout.occurrence(alpha, adj_left),
            layout.occurrence(beta, adj_mid_left),
            layout.occurrence(gamma, adj_mid_right),
            layout.occurrence(delta, adj_right),
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

    bilinear_terms = compile_gauge_kinetic_bilinear_terms(
        gauge_group=gauge_group,
        gauge_field=gauge_field,
        coefficient=term.coefficient,
        label_prefix=label_prefix,
    )
    if gauge_group.abelian:
        return bilinear_terms

    return bilinear_terms + (
        compile_yang_mills_cubic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        ),
        compile_yang_mills_quartic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        ),
    )


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
    layout = _GaugeFieldLayout.from_field(
        gauge_field,
        purpose="Gauge-fixing compilation",
    )

    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}_gf")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}_gf")
    identity_factor, fields = layout.bilinear_occurrences(alpha, beta)
    label = term.label or f"-(1/2 {term.xi}) ({gauge_group.name} gauge fixing)"
    manual_term = (
        -term.coefficient
        * _HALF
        / term.xi
        * identity_factor
        * lorentz_metric(alpha, rho_left)
        * lorentz_metric(beta, rho_right)
        * PartialD(fields[0], rho_left)
        * PartialD(fields[1], rho_right)
    )
    interaction = _lower_local_interaction_monomial(manual_term)
    if interaction is None:
        raise ValueError(
            "Gauge-fixing compilation could not lower the helper form through "
            "the ordinary local interaction path."
        )

    return (
        replace(
            interaction,
            label=label,
            sector="gauge_fixing",
            origin="GaugeFixing",
            origin_group=gauge_group,
        ),
    )


def compile_ghost_term(model: Model, term: GhostTerm) -> tuple[InteractionTerm, ...]:
    """Compile the ordinary unbroken Faddeev-Popov ghost sector.

    For the current conventions this corresponds to:
    ``-cbar^a partial^mu(D_mu c)^a = (partial cbar)(partial c) + g f (partial cbar) A c``.
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
    gauge_layout = _GaugeFieldLayout.from_field(
        gauge_field,
        purpose="Ghost compilation",
        require_adjoint=True,
    )
    adj_kind = gauge_layout.adjoint_kind
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
            term.coefficient
            * gauge_group.coupling
            * _build_structure_constant(gauge_group, a_bar, a_gauge, a_ghost)
            * lorentz_metric(rho, mu)
        ),
        fields=(
            ghost_field.occurrence(conjugated=True, labels=ghost_bar_labels),
            gauge_layout.occurrence(mu, a_gauge),
            ghost_field.occurrence(labels=ghost_labels),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=label_prefix + " gauge interaction",
    )

    return (ghost_bilinear, ghost_gauge)


def compile_dirac_kinetic_term(model: Model, term: DiracKineticTerm) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``psibar i gamma^mu D_mu psi``.

    One model declaration can expand into several interactions when the same
    fermion transforms under multiple declared gauge groups. This intentionally
    omits the free ``psibar i gamma^mu partial_mu psi`` bilinear; declarative
    ``CovD(...)`` monomials add that piece separately.
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
    fermion_spinor_slot = unique_spinor_slot(
        fermion,
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
                gauge_action_from_piece=_GaugeAction.from_piece,
                slot_suffix=_slot_suffix,
                prefactor=term.coefficient,
                label=label,
                lorentz_label=mu,
                spectator_exclude_slots={fermion_spinor_slot},
            )
        )

    return tuple(interactions)


def compile_complex_scalar_kinetic_term(
    model: Model,
    term: ComplexScalarKineticTerm,
) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``(D_mu phi)^dagger (D^mu phi)``.

    The output contains both the current term and the two-gauge contact term
    for each applicable gauge group. This intentionally omits the free
    ``(partial_mu phi)^dagger (partial^mu phi)`` bilinear; declarative
    ``CovD(...)`` monomials add that piece separately.
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


def _compile_covariant_core(
    model: Model,
    core: Union[DiracKineticTerm, ComplexScalarKineticTerm],
    *,
    include_free_bilinear: bool,
    spectators: tuple[tuple[Field, bool], ...] = (),
) -> tuple[InteractionTerm, ...]:
    """Compile one covariant kinetic core with explicit free-bilinear policy."""
    return _compile_covariant_core_impl(
        model,
        core,
        include_free_bilinear=include_free_bilinear,
        spectators=spectators,
        require_declared_field=_require_declared_field,
        compile_dirac_kinetic_term=compile_dirac_kinetic_term,
        compile_complex_scalar_kinetic_term=compile_complex_scalar_kinetic_term,
        symbol=_symbol,
    )


def _compile_declared_covariant_core(
    model: Model,
    core: Union[DiracKineticTerm, ComplexScalarKineticTerm],
    spectators: tuple[tuple[Field, bool], ...] = (),
) -> tuple[InteractionTerm, ...]:
    """Compile one declarative ``CovD`` monomial as the full kinetic operator."""
    return _compile_declared_covariant_core_impl(
        model,
        core,
        spectators=spectators,
        require_declared_field=_require_declared_field,
        compile_dirac_kinetic_term=compile_dirac_kinetic_term,
        compile_complex_scalar_kinetic_term=compile_complex_scalar_kinetic_term,
        symbol=_symbol,
    )


def compile_covariant_terms(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile all declared source terms in ``Model.lagrangian_decl``.

    This expands every declared term, including:

    - covariant-derivative kinetic cores (``CovD(...) CovD(...)``,
      ``i Psi.bar Gamma(mu) CovD(Psi, mu)``)
    - generic ``CovD(...)`` monomials that need gauge-metadata expansion
    - field-strength gauge kinetic terms (``-1/4 F F``)
    - explicit ``GaugeFixing(...)`` and ``GhostLagrangian(...)`` declarations
    - pure local interaction monomials.
    """
    interactions: list[InteractionTerm] = []

    for term in model.lagrangian_decl.source_terms:
        analyzed = _analyze_declared_source_term(term, parameters=model.parameters)
        if analyzed is None:
            raise TypeError(f"Unsupported declarative term: {type(term)!r}")

        if analyzed.interaction is not None:
            interactions.append(analyzed.interaction)
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

        if analyzed.generic_covariant_monomial is not None:
            interactions.extend(
                _compile_generic_declared_covariant_monomial(
                    model,
                    analyzed.generic_covariant_monomial,
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

    return tuple(interactions)
