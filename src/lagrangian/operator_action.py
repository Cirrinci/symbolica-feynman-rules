"""Linear/graded operator action on lowered ``InteractionTerm`` objects.

This module adds a *thin* layer that sits between the lowered/expanded
Lagrangian (``CompiledLagrangian``, made of ``InteractionTerm`` objects) and
``vertex_engine.py``. It does **not** modify the vertex engine itself.

The point of this layer is to let users act on a fully compiled Lagrangian
with linear or graded-derivation-like operators -- ordinary partial
derivatives, gauge transformations, BRST transformations, etc. -- while
preserving the ordered, fermion/ghost-aware structure of each interaction
monomial.

Design choices (see the design plan in the accompanying chat / commit
message for full context):

* The action is defined on lowered ``InteractionTerm`` objects, because that
  is the unit that already carries an ordered tuple of ``FieldOccurrence``
  factors, derivative annotations keyed to field slots, and closed Dirac
  bilinear metadata. Acting at this level avoids re-deriving any of that
  structure from a scalar Symbolica expression.

* The operator is described by a small, declarative ``FieldOperator``
  dataclass holding a ``parity`` (0 = even / bosonic, 1 = odd / fermionic
  or BRST-like) and a callable hook ``on_field(occurrence) -> result | None``
  describing what the operator does to one field occurrence. ``None`` means
  "operator annihilates / leaves this slot alone" so gauge-charge-aware
  actions are easy.

* The result of acting on one slot is an ``OperatorAtomResult``, i.e. a
  collection of ``OperatorSummand`` objects. Each summand carries a scalar
  ``coefficient`` (a Symbolica/numeric expression that multiplies the
  existing term coupling) and a ``replacement`` tuple of ``FieldOccurrence``
  that is spliced *positionally* into the original field slot. Splicing
  positionally is what makes the layer order-preserving.

* The graded Leibniz sign is computed slot by slot from the operator parity
  and the cumulative parity of the field occurrences to the *left* of the
  slot being acted on. For bosonic operators this collapses to ``+1`` at
  every slot.

* The authoritative representation remains the ordered ``InteractionTerm``
  structure. ``symbolica_export.py`` is a separate, display-only view; see
  the notes there for why Symbolica is not safe as a source of truth for
  fermion/ghost ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from itertools import product as iter_product
from typing import Callable, Optional, Sequence, Union

from symbolica import Expression, S

from model.interactions import (
    DerivativeAction,
    DiracBilinear,
    FieldOccurrence,
    InteractionTerm,
    SlotRef,
)
from model.metadata import Field, FieldRole, is_lorentz_index, unique_spinor_slot


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


Parity = int  # 0 = even (bosonic), 1 = odd (fermionic / BRST / ghost-like)


def _validate_operator_parity(name: str, parity: Parity) -> None:
    if parity not in (0, 1):
        raise ValueError(
            f"Operator {name!r} got parity={parity!r}; "
            "expected 0 (even) or 1 (odd)."
        )


@dataclass(frozen=True)
class OperatorSummand:
    """One summand contributed by ``FieldOperator.on_field(occurrence)``.

    ``coefficient`` multiplies the existing ``InteractionTerm.coupling`` of
    the term being acted on (Symbolica expressions, plain numbers and
    fractions all work; in particular ``1`` is the identity coefficient).

    ``replacement`` is the ordered tuple of ``FieldOccurrence`` objects that
    replaces the original slot. Common cases:

    * ``(new_occurrence,)`` -- "replace this field by another single field"
      (e.g. ``O[phi] = X`` where ``X`` is a fresh field occurrence).
    * ``(...,)`` with more than one entry -- e.g. a gauge transformation
      contributing ``i T^a c^a psi`` for a fermion: the ghost slot and the
      fermion slot together replace the original ``psi`` slot.
    * ``()`` -- "this slot is annihilated by the operator". Only allowed
      when the slot is not part of a closed Dirac bilinear and carries no
      derivative actions. Otherwise the engine raises a structured error.

    ``new_derivatives`` is an extra list of ``DerivativeAction`` objects to
    attach to the replacement *as fresh derivatives*. The ``target`` of
    each action is interpreted as a **position inside the replacement
    tuple** (0-indexed) and is translated by the engine to an absolute
    slot index when the splice happens. This is the mechanism by which a
    runtime "create a fresh ``partial_mu``" operator (see ``partial(...)``
    below) lowers to an ``InteractionTerm.derivatives`` entry on the
    correct slot -- something that a pure ``FieldOccurrence`` replacement
    cannot express.
    """

    coefficient: object = 1
    replacement: tuple[FieldOccurrence, ...] = ()
    new_derivatives: tuple[DerivativeAction, ...] = ()


@dataclass(frozen=True)
class OperatorAtomResult:
    """Result of ``FieldOperator.on_field(occurrence)``.

    A bag of ``OperatorSummand``s. Use the convenience helpers below
    (``single_field_result``, ``zero_result``, ``constant_result``) for the
    common cases.
    """

    summands: tuple[OperatorSummand, ...] = ()

    def __iter__(self):
        return iter(self.summands)

    def __len__(self):
        return len(self.summands)


def single_field_result(
    replacement: Union[FieldOccurrence, Sequence[FieldOccurrence]],
    *,
    coefficient: object = 1,
    new_derivatives: Sequence[DerivativeAction] = (),
) -> OperatorAtomResult:
    """Convenience: build a result with one summand.

    ``new_derivatives`` is forwarded to the resulting ``OperatorSummand``;
    each ``DerivativeAction.target`` is interpreted as a position inside
    ``replacement`` (0-indexed). The engine translates that to an
    absolute slot index when the splice happens.
    """

    if isinstance(replacement, FieldOccurrence):
        rep_tuple: tuple[FieldOccurrence, ...] = (replacement,)
    else:
        rep_tuple = tuple(replacement)
    return OperatorAtomResult(
        summands=(
            OperatorSummand(
                coefficient=coefficient,
                replacement=rep_tuple,
                new_derivatives=tuple(new_derivatives),
            ),
        ),
    )


def zero_result() -> OperatorAtomResult:
    """Convenience: operator annihilates this slot (returns no summands).

    Equivalent to ``on_field`` returning an empty bag of summands. Both this
    and returning ``None`` from ``on_field`` are accepted, but they mean
    slightly different things: ``None`` means "operator leaves this slot
    untouched and contributes nothing for this slot in the Leibniz sum",
    while ``zero_result()`` means "operator acts on this slot but the result
    is zero" -- still a valid Leibniz contribution, just numerically zero.
    """

    return OperatorAtomResult(summands=())


def constant_result(coefficient: object) -> OperatorAtomResult:
    """Convenience: scalar replacement of one field by a pure coefficient.

    This drops the original field slot and multiplies the coupling by
    ``coefficient``. Useful for tests and quick algebraic checks; in
    practice you usually want ``single_field_result``.
    """

    return OperatorAtomResult(
        summands=(OperatorSummand(coefficient=coefficient, replacement=()),),
    )


@dataclass(frozen=True)
class FieldOperator:
    """A linear/graded operator acting on lowered interaction terms.

    Parameters
    ----------
    name:
        Human-readable name, used only for diagnostics and ``repr``.
    parity:
        Grassmann parity of the operator. ``0`` (even) for ordinary
        derivations like partial derivatives, gauge transformations of
        bosonic fields, etc.  ``1`` (odd) for fermionic / BRST-like
        operators. The graded Leibniz sign uses this.
    on_field:
        Callable ``(occurrence: FieldOccurrence) -> OperatorAtomResult |
        None``. Returning ``None`` means "leave this slot alone" (no
        contribution to the Leibniz sum for this slot). Returning an
        ``OperatorAtomResult`` (possibly empty) means "act on this slot".
    commute_with_partial_derivative:
        If True (default), the operator is assumed to commute with the
        partial derivatives encoded in ``InteractionTerm.derivatives``.
        For a replacement of length ``N >= 1`` on a slot that carries
        ``M`` derivative actions, the engine then performs the bosonic
        Leibniz expansion of those derivatives across the replacement
        slots: each independent derivative may land on any replacement
        slot, so each summand fans out to ``N ** M`` output terms (with
        ``N ** 0 = 1`` recovering the simple no-derivative case). The
        partial derivative is parity-even, so all such terms enter with
        sign ``+1`` relative to the slot's Leibniz sign.

        If False, the engine refuses to act on any slot that carries a
        derivative action and raises a structured error -- callers that
        need a non-commuting action (BRST through a covariant derivative,
        for instance) should pre-expand the derivative or write a custom
        ``on_field`` that consumes the derivative context explicitly.
    """

    name: str
    parity: Parity = 0
    on_field: Optional[Callable[[FieldOccurrence], Optional[OperatorAtomResult]]] = None
    commute_with_partial_derivative: bool = True

    def __post_init__(self):
        _validate_operator_parity(self.name, self.parity)

    def __call__(self, occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        if self.on_field is None:
            return None
        return self.on_field(occurrence)


@dataclass(frozen=True)
class TermOperator:
    """A whole-term operator acting directly on ``InteractionTerm`` objects.

    This is the escape hatch for operators whose action cannot be expressed
    safely as an independent slot-wise derivation. The returned terms are
    assumed to already preserve any ordered-field, derivative-target, and
    bilinear metadata invariants required by the rest of the pipeline.
    """

    name: str
    parity: Parity = 0
    apply_to_term: Optional[Callable[[InteractionTerm], Sequence[InteractionTerm]]] = None

    def __post_init__(self):
        _validate_operator_parity(self.name, self.parity)

    def __call__(self, term: InteractionTerm) -> tuple[InteractionTerm, ...]:
        if self.apply_to_term is None:
            return ()
        return tuple(self.apply_to_term(term))


@dataclass(frozen=True)
class OperatorExpansionError(ValueError):
    """Raised when one operator application would exceed an output-term cap."""

    operator_name: str
    origin: str
    slot: Optional[int]
    replacement_len: Optional[int]
    derivative_count_on_slot: Optional[int]
    projected_terms: int
    max_generated_terms: int

    def __str__(self) -> str:
        details = [f"operator={self.operator_name!r}"]
        if self.origin:
            details.append(f"origin={self.origin!r}")
        if self.slot is not None:
            details.append(f"slot={self.slot}")
        if self.replacement_len is not None:
            details.append(f"replacement_len={self.replacement_len}")
        if self.derivative_count_on_slot is not None:
            details.append(f"derivative_count_on_slot={self.derivative_count_on_slot}")
        details.append(f"projected_terms={self.projected_terms}")
        details.append(f"max_generated_terms={self.max_generated_terms}")
        return "Operator expansion exceeds configured limit (" + ", ".join(details) + ")."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _OperatorApplicationMemo:
    inherited_replacements: dict[tuple[object, ...], tuple[FieldOccurrence, ...]] = dataclass_field(
        default_factory=dict
    )
    bilinear_remaps: dict[tuple[object, ...], tuple[DiracBilinear, ...]] = dataclass_field(
        default_factory=dict
    )
    derivative_arrangements: dict[tuple[object, ...], tuple[tuple[DerivativeAction, ...], ...]] = dataclass_field(
        default_factory=dict
    )


def _symbolic_cache_key(value):
    if isinstance(value, tuple):
        return tuple(_symbolic_cache_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_symbolic_cache_key(item) for item in value)
    if hasattr(value, "to_canonical_string"):
        return ("expr", value.to_canonical_string())
    return value


def _occurrence_cache_key(occurrence: FieldOccurrence) -> tuple[object, ...]:
    return (
        id(occurrence.field),
        bool(occurrence.conjugated),
        tuple(_symbolic_cache_key(label) for label in occurrence.slot_labels.values),
    )


def _replacement_cache_key(
    replacement: Sequence[FieldOccurrence],
) -> tuple[tuple[object, ...], ...]:
    return tuple(_occurrence_cache_key(occurrence) for occurrence in replacement)


def _derivative_cache_key(
    derivatives: Sequence[DerivativeAction],
) -> tuple[tuple[object, object], ...]:
    return tuple(
        (action.target, _symbolic_cache_key(action.lorentz_index))
        for action in derivatives
    )


def _occurrence_parity(occurrence: FieldOccurrence) -> int:
    """Grassmann parity of a single ``FieldOccurrence``.

    Fermions and ghosts are odd; everything else is even. This duplicates
    the convention already used by ``InteractionTerm.statistics`` but reads
    one occurrence at a time, which is exactly what the graded Leibniz sum
    needs.
    """

    return 1 if occurrence.field.statistics == "fermion" else 0


def _leibniz_sign(operator_parity: int, parity_to_the_left: int) -> int:
    """``(-1) ** (|O| * sum_{j<k} parity(field_j))`` as a plain integer."""

    exponent = (operator_parity * parity_to_the_left) % 2
    return 1 if exponent == 0 else -1


def _validate_replacement_against_existing_features(
    *,
    operator: FieldOperator,
    slot: int,
    replacement: tuple[FieldOccurrence, ...],
    derivatives_on_slot: tuple[DerivativeAction, ...],
    bilinears_containing_slot: tuple[tuple[int, int], ...],
):
    """Reject replacements that the splice helpers cannot safely handle.

    Structural validation of bilinear endpoints lives in
    ``_validate_and_remap_bilinears`` (which inspects the replacement's
    field statistics/conjugation). This helper only handles the
    coarse-grained "is the replacement compatible with the operator's
    options at all?" question.
    """

    if not replacement:
        if derivatives_on_slot:
            raise ValueError(
                f"Operator {operator.name!r} annihilates slot {slot} but the slot "
                "carries derivative actions; supply a non-empty replacement or "
                "extend the operator to act on derivative slots explicitly."
            )
        if bilinears_containing_slot:
            raise ValueError(
                f"Operator {operator.name!r} annihilates slot {slot} but the slot "
                "is part of a closed Dirac bilinear; this would break the bilinear "
                "structure. Provide a non-empty replacement instead."
            )
        return

    if derivatives_on_slot and not operator.commute_with_partial_derivative:
        raise ValueError(
            f"Operator {operator.name!r} does not commute with partial derivatives, "
            f"and slot {slot} carries derivative actions. Implement a custom action "
            "before acting on this slot."
        )


def _inherit_missing_replacement_labels(
    *,
    original_occurrence: FieldOccurrence,
    replacement: tuple[FieldOccurrence, ...],
) -> tuple[FieldOccurrence, ...]:
    """Fill missing replacement labels from the acted occurrence when safe.

    This is intentionally conservative. For one exact ``IndexType`` we only
    inherit labels when:

    * the original occurrence already carries labels for that index type;
    * all replacement slots of that exact index type live on a single
      replacement occurrence; and
    * that occurrence carries the same number of slots of the index type as
      the original occurrence.

    Explicit replacement labels are preserved. Ambiguous cases -- most
    importantly product-valued replacements with multiple compatible target
    occurrences -- are left untouched so callers can specify labels
    explicitly.
    """

    if not replacement:
        return replacement

    source_field = original_occurrence.field
    source_slot_labels = original_occurrence.slot_labels
    if not any(label is not None for label in source_slot_labels.values):
        return replacement

    replacement_list = list(replacement)

    source_slots_by_index: dict[object, list[int]] = {}
    for slot, index in enumerate(source_field.indices):
        source_slots_by_index.setdefault(index, [])
        source_slots_by_index[index].append(slot)
    source_slots_by_index = {
        index: tuple(slots) for index, slots in source_slots_by_index.items()
    }

    replacement_occurrences_by_index: dict[object, set[int]] = {}
    for occurrence_index, occurrence in enumerate(replacement_list):
        for index in occurrence.field.indices:
            replacement_occurrences_by_index.setdefault(index, set()).add(occurrence_index)

    for index, source_slots in source_slots_by_index.items():
        if not any(source_slot_labels.get(slot) is not None for slot in source_slots):
            continue

        target_occurrence_indices = replacement_occurrences_by_index.get(index, set())
        if len(target_occurrence_indices) != 1:
            continue

        target_occurrence_index = next(iter(target_occurrence_indices))
        target_occurrence = replacement_list[target_occurrence_index]
        target_slots = target_occurrence.field.index_positions(index=index)
        if len(target_slots) != len(source_slots):
            continue

        target_slot_labels = target_occurrence.slot_labels
        changed = False
        for source_slot, target_slot in zip(source_slots, target_slots):
            if target_slot_labels.get(target_slot) is not None:
                continue
            source_label = source_slot_labels.get(source_slot)
            if source_label is None:
                continue
            target_slot_labels = target_slot_labels.replace(target_slot, source_label)
            changed = True

        if not changed:
            continue

        replacement_list[target_occurrence_index] = target_occurrence.with_slot_labels(
            target_slot_labels
        )

    return tuple(replacement_list)


def _matching_fermion_positions_in_replacement(
    replacement: Sequence[FieldOccurrence],
    *,
    conjugated_target: bool,
) -> list[SlotRef]:
    """Find positions in ``replacement`` that look like a Dirac-fermion of
    the requested conjugation.

    Uses ``Field.kind == "fermion"`` (not just ``statistics == "fermion"``)
    so that ghost fields -- which are Grassmann-odd but have no Dirac
    spinor index -- are not accidentally picked up as bilinear endpoints.
    """

    matches: list[SlotRef] = []
    for position, occurrence in enumerate(replacement):
        if occurrence.field.kind != "fermion":
            continue
        if bool(occurrence.conjugated) != bool(conjugated_target):
            continue
        matches.append(
            SlotRef(
                occurrence=position,
                slot=unique_spinor_slot(
                    occurrence.field,
                    purpose="Operator bilinear remapping",
                ),
            )
        )
    return matches


def _validate_closed_bilinear_pair_structure(
    bilinears: Sequence[DiracBilinear],
    *,
    error_prefix: str,
) -> tuple[DiracBilinear, ...]:
    intervals = sorted(
        (
            min(bilinear.psibar.occurrence, bilinear.psi.occurrence),
            max(bilinear.psibar.occurrence, bilinear.psi.occurrence),
            bilinear,
        )
        for bilinear in bilinears
    )
    for (_, prev_end, _), (next_start, _, _) in zip(intervals, intervals[1:]):
        if next_start <= prev_end:
            raise ValueError(
                f"{error_prefix} would produce overlapping or interleaved closed "
                "Dirac bilinears, which are not supported."
            )
    return tuple(bilinear for _, _, bilinear in intervals)


def _validate_and_remap_bilinears(
    *,
    operator: FieldOperator,
    slot: int,
    original_occurrence: FieldOccurrence,
    replacement: tuple[FieldOccurrence, ...],
    bilinears: tuple[DiracBilinear, ...],
) -> tuple[DiracBilinear, ...]:
    """Validate and remap ``closed_dirac_bilinears`` across one splice.

    For each structural Dirac bilinear:

    * Bilinears not involving ``slot`` are shifted by ``len(replacement) -
      1`` for endpoints ``> slot``.
    * Bilinears with the ``psibar`` endpoint at ``slot`` require the
      replacement to contain *exactly one* Dirac-fermion occurrence with
      ``conjugated == True``; its position inside the replacement becomes
      the new ``psibar_slot``.
    * Bilinears with the ``psi`` endpoint at ``slot`` require the
      replacement to contain exactly one Dirac-fermion occurrence with
      ``conjugated == False``.

    If the replacement does not preserve the bilinear structure, we raise
    a structured ``ValueError``: silently shifting the indices would
    leave a stale bilinear pointing at a non-Dirac-fermion slot, which
    the vertex engine would later reject with a less informative error.
    """

    shift = len(replacement) - 1

    def shift_slot_ref(ref: SlotRef) -> SlotRef:
        if ref.occurrence > slot:
            return SlotRef(occurrence=ref.occurrence + shift, slot=ref.slot)
        return ref

    remapped: list[DiracBilinear] = []
    for bilinear in bilinears:
        if bilinear.psibar.occurrence == slot:
            matches = _matching_fermion_positions_in_replacement(
                replacement, conjugated_target=True
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Operator {operator.name!r} acts on slot {slot}, the psibar "
                    "endpoint of a closed Dirac bilinear, but the replacement does "
                    f"not contain exactly one conjugated Dirac-fermion factor (found "
                    f"{len(matches)}). Provide a replacement that preserves the "
                    "bilinear structure, or drop the bilinear metadata before "
                    "applying the operator."
                )
            remapped.append(
                DiracBilinear(
                    psibar=SlotRef(
                        occurrence=slot + matches[0].occurrence,
                        slot=matches[0].slot,
                    ),
                    psi=shift_slot_ref(bilinear.psi),
                )
            )
            continue
        if bilinear.psi.occurrence == slot:
            matches = _matching_fermion_positions_in_replacement(
                replacement, conjugated_target=False
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Operator {operator.name!r} acts on slot {slot}, the psi "
                    "endpoint of a closed Dirac bilinear, but the replacement does "
                    f"not contain exactly one unconjugated Dirac-fermion factor "
                    f"(found {len(matches)}). Provide a replacement that preserves "
                    "the bilinear structure, or drop the bilinear metadata before "
                    "applying the operator."
                )
            remapped.append(
                DiracBilinear(
                    psibar=shift_slot_ref(bilinear.psibar),
                    psi=SlotRef(
                        occurrence=slot + matches[0].occurrence,
                        slot=matches[0].slot,
                    ),
                )
            )
            continue
        remapped.append(
            DiracBilinear(
                psibar=shift_slot_ref(bilinear.psibar),
                psi=shift_slot_ref(bilinear.psi),
            )
        )
    return _validate_closed_bilinear_pair_structure(
        remapped,
        error_prefix=(
            f"Operator {operator.name!r} acting on slot {slot}"
        ),
    )


def _enumerate_derivative_arrangements(
    *,
    operator: FieldOperator,
    slot: int,
    replacement_len: int,
    derivatives: tuple[DerivativeAction, ...],
) -> tuple[tuple[DerivativeAction, ...], ...]:
    """Enumerate output ``derivatives`` tuples for the bosonic Leibniz
    expansion of derivatives across one splice.

    Derivatives not on ``slot`` are simply shifted; derivatives on
    ``slot`` itself fan out: each independent derivative may land on any
    of the ``replacement_len`` replacement slots, giving
    ``replacement_len ** (#on_slot)`` independent arrangements. The
    partial derivative is parity-even, so no extra signs appear here.

    The empty-replacement / non-commuting-partial cases are guarded by
    ``_validate_replacement_against_existing_features`` and would also
    raise here defensively if reached.
    """

    shift = replacement_len - 1

    def shift_index(idx: int) -> int:
        if idx > slot:
            return idx + shift
        return idx

    on_slot = tuple(action for action in derivatives if action.target == slot)
    off_slot = tuple(
        DerivativeAction(target=shift_index(action.target), lorentz_index=action.lorentz_index)
        for action in derivatives
        if action.target != slot
    )

    if not on_slot:
        return (off_slot,)

    if replacement_len == 0:
        raise ValueError(
            f"Operator {operator.name!r} cannot annihilate slot {slot} while it "
            "carries derivative actions."
        )
    if not operator.commute_with_partial_derivative:
        raise ValueError(
            f"Operator {operator.name!r} does not commute with partial derivatives, "
            f"and slot {slot} carries derivative actions."
        )

    arrangements: list[tuple[DerivativeAction, ...]] = []
    for assignment in iter_product(range(replacement_len), repeat=len(on_slot)):
        placed = tuple(
            DerivativeAction(target=slot + offset, lorentz_index=action.lorentz_index)
            for action, offset in zip(on_slot, assignment)
        )
        arrangements.append(off_slot + placed)
    return tuple(arrangements)


def _project_derivative_arrangements_count(
    *,
    replacement_len: int,
    derivative_count_on_slot: int,
) -> int:
    if derivative_count_on_slot == 0:
        return 1
    return replacement_len ** derivative_count_on_slot


def _translate_new_derivatives(
    *,
    operator: FieldOperator,
    slot: int,
    replacement_len: int,
    new_derivatives: tuple[DerivativeAction, ...],
) -> tuple[DerivativeAction, ...]:
    """Translate fresh ``new_derivatives`` from positions inside the
    replacement to absolute slot indices.

    Each input ``DerivativeAction.target`` must be a valid position
    ``0 <= target < replacement_len``. The output ``target`` is the
    corresponding absolute slot index ``slot + target`` after splicing.
    A missing ``lorentz_index`` is rejected here rather than later in
    ``InteractionTerm`` consumers.
    """

    if not new_derivatives:
        return ()
    if replacement_len == 0:
        raise ValueError(
            f"Operator {operator.name!r} requested fresh derivatives on slot "
            f"{slot} but the replacement is empty; a derivative needs a slot "
            "to act on."
        )

    translated: list[DerivativeAction] = []
    for action in new_derivatives:
        if action.lorentz_index is None:
            raise ValueError(
                f"Operator {operator.name!r}: fresh derivative on slot {slot} "
                "has lorentz_index=None; pass a concrete Lorentz index "
                "(e.g. S('mu'))."
            )
        if not (0 <= action.target < replacement_len):
            raise ValueError(
                f"Operator {operator.name!r}: fresh derivative target "
                f"{action.target} is out of range for a replacement of length "
                f"{replacement_len}."
            )
        translated.append(
            DerivativeAction(
                target=slot + action.target,
                lorentz_index=action.lorentz_index,
            )
        )
    return tuple(translated)


def _coupling_product(*pieces: object) -> object:
    """Multiply Symbolica expressions / Python scalars conservatively.

    The compiled-term coupling can be a Symbolica ``Expression`` or a plain
    Python number; ``OperatorSummand.coefficient`` can be either too. Plain
    ``*`` works for both, but we centralize the call to keep the engine
    easy to read and easy to instrument.
    """

    result: object = 1
    for piece in pieces:
        result = result * piece
    return result


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def apply_field_operator_to_term(
    term: InteractionTerm,
    operator: FieldOperator,
    *,
    max_generated_terms: Optional[int] = None,
    _memo: Optional[_OperatorApplicationMemo] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply ``operator`` to one lowered ``InteractionTerm``.

    Implements the graded Leibniz rule

        O(F_1 * F_2 * ... * F_n) =
            sum_k (-1)^{|O| * P_{<k}} F_1 * ... * F_{k-1}
                                       * O(F_k)
                                       * F_{k+1} * ... * F_n

    where ``P_{<k}`` is the cumulative parity of the field occurrences to
    the left of slot ``k``. The action on one slot can produce multiple
    summands; each summand becomes a fresh ``InteractionTerm``.
    """

    if _memo is None:
        _memo = _OperatorApplicationMemo()

    parity_to_the_left = 0
    new_terms: list[InteractionTerm] = []

    derivatives_by_slot: dict[int, list[DerivativeAction]] = {}
    for action in term.derivatives:
        derivatives_by_slot.setdefault(action.target, []).append(action)

    bilinears_by_slot: dict[int, list[DiracBilinear]] = {}
    for bilinear in term.dirac_bilinears:
        bilinears_by_slot.setdefault(bilinear.psibar.occurrence, []).append(bilinear)
        bilinears_by_slot.setdefault(bilinear.psi.occurrence, []).append(bilinear)

    for slot, occurrence in enumerate(term.fields):
        result = operator(occurrence)
        if result is None:
            parity_to_the_left = (parity_to_the_left + _occurrence_parity(occurrence)) % 2
            continue

        sign = _leibniz_sign(operator.parity, parity_to_the_left)
        derivatives_on_slot = tuple(derivatives_by_slot.get(slot, ()))
        bilinears_with_slot = tuple(bilinears_by_slot.get(slot, ()))

        for summand in result.summands:
            raw_replacement = tuple(summand.replacement)
            replacement_key = (
                _occurrence_cache_key(occurrence),
                _replacement_cache_key(raw_replacement),
            )
            replacement = _memo.inherited_replacements.get(replacement_key)
            if replacement is None:
                replacement = _inherit_missing_replacement_labels(
                    original_occurrence=occurrence,
                    replacement=raw_replacement,
                )
                _memo.inherited_replacements[replacement_key] = replacement
            _validate_replacement_against_existing_features(
                operator=operator,
                slot=slot,
                replacement=replacement,
                derivatives_on_slot=derivatives_on_slot,
                bilinears_containing_slot=bilinears_with_slot,
            )

            new_fields = term.fields[:slot] + replacement + term.fields[slot + 1 :]
            bilinear_key = (
                operator.name,
                slot,
                _occurrence_cache_key(occurrence),
                _replacement_cache_key(replacement),
                term.dirac_bilinears,
            )
            new_bilinears = _memo.bilinear_remaps.get(bilinear_key)
            if new_bilinears is None:
                new_bilinears = _validate_and_remap_bilinears(
                    operator=operator,
                    slot=slot,
                    original_occurrence=occurrence,
                    replacement=replacement,
                    bilinears=term.dirac_bilinears,
                )
                _memo.bilinear_remaps[bilinear_key] = new_bilinears
            new_coupling = _coupling_product(
                sign,
                summand.coefficient,
                term.coupling,
            )

            fresh_derivatives = _translate_new_derivatives(
                operator=operator,
                slot=slot,
                replacement_len=len(replacement),
                new_derivatives=summand.new_derivatives,
            )

            arrangement_key = (
                operator.name,
                slot,
                len(replacement),
                operator.commute_with_partial_derivative,
                _derivative_cache_key(term.derivatives),
            )
            arrangements = _memo.derivative_arrangements.get(arrangement_key)
            projected_terms = _project_derivative_arrangements_count(
                replacement_len=len(replacement),
                derivative_count_on_slot=len(derivatives_on_slot),
            )
            if (
                max_generated_terms is not None
                and len(new_terms) + projected_terms > max_generated_terms
            ):
                raise OperatorExpansionError(
                    operator_name=operator.name,
                    origin=term.origin,
                    slot=slot,
                    replacement_len=len(replacement),
                    derivative_count_on_slot=len(derivatives_on_slot),
                    projected_terms=projected_terms,
                    max_generated_terms=max_generated_terms,
                )
            if arrangements is None:
                arrangements = _enumerate_derivative_arrangements(
                    operator=operator,
                    slot=slot,
                    replacement_len=len(replacement),
                    derivatives=term.derivatives,
                )
                _memo.derivative_arrangements[arrangement_key] = arrangements
            for arrangement in arrangements:
                new_terms.append(
                    replace(
                        term,
                        coupling=new_coupling,
                        fields=new_fields,
                        derivatives=arrangement + fresh_derivatives,
                        closed_dirac_bilinears=tuple(
                            bilinear.as_legacy() for bilinear in new_bilinears
                        ),
                        dirac_bilinears=new_bilinears,
                        origin=(term.origin + (";" if term.origin else "") + f"op:{operator.name}@slot{slot}"),
                    )
                )

        parity_to_the_left = (parity_to_the_left + _occurrence_parity(occurrence)) % 2

    return tuple(new_terms)


def apply_field_operator(
    terms: Sequence[InteractionTerm],
    operator: FieldOperator,
    *,
    max_generated_terms: Optional[int] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply ``operator`` to a sequence of ``InteractionTerm`` objects."""

    memo = _OperatorApplicationMemo()
    expanded: list[InteractionTerm] = []
    for term in terms:
        remaining = None
        if max_generated_terms is not None:
            remaining = max_generated_terms - len(expanded)
        expanded.extend(
            apply_field_operator_to_term(
                term,
                operator,
                max_generated_terms=remaining,
                _memo=memo,
            )
        )
    return tuple(expanded)


def apply_term_operator_to_term(
    term: InteractionTerm,
    operator: TermOperator,
    *,
    max_generated_terms: Optional[int] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply one whole-term operator to one lowered interaction term."""

    results = tuple(operator(term))
    if max_generated_terms is not None and len(results) > max_generated_terms:
        raise OperatorExpansionError(
            operator_name=operator.name,
            origin=term.origin,
            slot=None,
            replacement_len=None,
            derivative_count_on_slot=None,
            projected_terms=len(results),
            max_generated_terms=max_generated_terms,
        )
    return results


def apply_term_operator(
    terms: Sequence[InteractionTerm],
    operator: TermOperator,
    *,
    max_generated_terms: Optional[int] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply one whole-term operator to a sequence of interaction terms."""

    expanded: list[InteractionTerm] = []
    for term in terms:
        remaining = None
        if max_generated_terms is not None:
            remaining = max_generated_terms - len(expanded)
        expanded.extend(
            apply_term_operator_to_term(
                term,
                operator,
                max_generated_terms=remaining,
            )
        )
    return tuple(expanded)


def apply_operator_to_term(
    term: InteractionTerm,
    operator: Union[FieldOperator, TermOperator],
    *,
    max_generated_terms: Optional[int] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply either a ``FieldOperator`` or a ``TermOperator`` to one term."""

    if isinstance(operator, FieldOperator):
        return apply_field_operator_to_term(
            term,
            operator,
            max_generated_terms=max_generated_terms,
        )
    if isinstance(operator, TermOperator):
        return apply_term_operator_to_term(
            term,
            operator,
            max_generated_terms=max_generated_terms,
        )
    raise TypeError(
        "operator must be a FieldOperator or TermOperator, got "
        f"{type(operator).__name__}."
    )


def apply_operator(
    terms: Sequence[InteractionTerm],
    operator: Union[FieldOperator, TermOperator],
    *,
    max_generated_terms: Optional[int] = None,
) -> tuple[InteractionTerm, ...]:
    """Apply either a ``FieldOperator`` or a ``TermOperator`` to terms."""

    if max_generated_terms is not None and max_generated_terms < 0:
        raise ValueError("max_generated_terms must be >= 0 or None.")
    if isinstance(operator, FieldOperator):
        return apply_field_operator(
            terms,
            operator,
            max_generated_terms=max_generated_terms,
        )
    if isinstance(operator, TermOperator):
        return apply_term_operator(
            terms,
            operator,
            max_generated_terms=max_generated_terms,
        )
    raise TypeError(
        "operator must be a FieldOperator or TermOperator, got "
        f"{type(operator).__name__}."
    )


# ---------------------------------------------------------------------------
# Sugar constructors for common operator kinds
# ---------------------------------------------------------------------------


def replacement_operator(
    name: str,
    mapping: dict[Field, Union[FieldOccurrence, OperatorAtomResult]],
    *,
    parity: Parity = 0,
    commute_with_partial_derivative: bool = True,
) -> FieldOperator:
    """Sugar for ``O[field] = something`` with a per-field dictionary.

    ``mapping`` values may be:

    * a ``FieldOccurrence`` -- shorthand for ``single_field_result(...)``;
    * a fully formed ``OperatorAtomResult`` -- used as-is.

    Fields not present in the mapping are left untouched (the operator
    returns ``None`` for them).
    """

    def on_field(occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        value = mapping.get(occurrence.field)
        if value is None:
            return None
        if isinstance(value, FieldOccurrence):
            return single_field_result(value)
        if isinstance(value, OperatorAtomResult):
            return value
        raise TypeError(
            f"replacement_operator({name!r}): mapping values must be "
            "FieldOccurrence or OperatorAtomResult, got "
            f"{type(value).__name__}."
        )

    return FieldOperator(
        name=name,
        parity=parity,
        on_field=on_field,
        commute_with_partial_derivative=commute_with_partial_derivative,
    )


def partial(
    lorentz_index: object,
    field: Optional[Field] = None,
    *,
    on: Optional[Union[Field, Sequence[Field]]] = None,
    name: Optional[str] = None,
):
    """User-facing partial-derivative factory with two roles.

    * ``partial(mu)`` (one positional argument) returns a runtime
      ``FieldOperator`` that, when applied via
      ``CompiledLagrangian.apply_operator(...)``, implements the spacetime
      partial derivative as a graded Leibniz derivation: it walks the
      ordered slots, attaches a fresh ``DerivativeAction(lorentz_index=mu)``
      on each, and produces one output term per slot. Use the keyword
      ``on=`` to restrict it to one field or to a tuple of fields, e.g.
      ``partial(mu, on=Phi)`` or ``partial(mu, on=(Phi, Chi))``. This is
      the spacetime derivative, **not** a field-space variation.

    * ``partial(mu, Phi)`` (two positional arguments) is sugar for the
      declarative factor ``PartialD(Phi, mu)``: it is suitable for use
      inside ``Model(lagrangian_decl=...)``. The argument order matches
      the natural reading "the partial in direction mu, applied to Phi".

    The two roles are *disjoint*: the runtime operator lives in this
    module, the declarative factor lives in ``model.declared``. Both are
    exposed here so users have one obvious name to reach for, and the
    polymorphism is decided by whether a ``field`` argument is provided.

    Other keyword arguments only apply to the runtime form:

    * ``name``: human-readable name used by diagnostics / ``repr``. If
      not supplied, defaults to ``f"d_{lorentz_index}"``.
    """

    if lorentz_index is None:
        raise TypeError("partial(...) requires a non-None Lorentz index.")

    if field is not None:
        if on is not None or name is not None:
            raise TypeError(
                "partial(mu, field) is a declarative-factor shortcut and "
                "does not accept on=/name= keyword arguments."
            )
        from model.declared import PartialD

        return PartialD(field, lorentz_index)

    if on is None:
        target_fields: Optional[tuple[Field, ...]] = None
    elif isinstance(on, Field):
        target_fields = (on,)
    else:
        target_fields = tuple(on)
        if not target_fields:
            raise ValueError("partial(..., on=...) was given an empty list of fields.")
        for f in target_fields:
            if not isinstance(f, Field):
                raise TypeError(
                    f"partial(..., on=...): expected Field, got {type(f).__name__}."
                )

    op_name = name if name is not None else f"d_{lorentz_index}"

    def on_field(occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        if target_fields is not None and occurrence.field not in target_fields:
            return None
        new_occ = _copy_occurrence(occurrence)
        return single_field_result(
            new_occ,
            new_derivatives=(DerivativeAction(target=0, lorentz_index=lorentz_index),),
        )

    return FieldOperator(name=op_name, parity=0, on_field=on_field)


# ---------------------------------------------------------------------------
# Infinitesimal gauge variation as a graded derivation
# ---------------------------------------------------------------------------


def _first_non_lorentz_index(field: Field):
    """Return the first non-Lorentz ``IndexType`` carried by ``field`` (the
    adjoint slot for a gauge boson), or ``None``.
    """

    for index in field.indices:
        if not is_lorentz_index(index):
            return index
    return None


@dataclass(frozen=True)
class _GaugeOperatorContext:
    group: object
    operator_name: str
    abelian: bool
    gauge_boson: Optional[Field]
    gauge_coupling: object
    adjoint_index: Optional[object]


def _replace_slot_label(field: Field, occurrence: FieldOccurrence, slot: int, new_label):
    """Build a new occurrence of ``field`` with the label on ``slot`` replaced
    by ``new_label`` and everything else preserved.
    """

    del field
    return occurrence.with_slot_label(slot, new_label)


def _copy_occurrence(occurrence: FieldOccurrence) -> FieldOccurrence:
    """Clone one occurrence while preserving conjugation and index labels."""

    return occurrence.field.occurrence(
        conjugated=occurrence.conjugated,
        labels=occurrence.slot_labels.to_legacy(),
    )


def _lorentz_label_of(occurrence: FieldOccurrence):
    """Return the unique Lorentz label of ``occurrence``, or ``None``."""

    field = occurrence.field
    for slot, index in enumerate(field.indices):
        if is_lorentz_index(index):
            label = occurrence.slot_labels.get(slot)
            if label is not None:
                return label
    return None


def _build_alpha_field(parameter, adjoint_index) -> Field:
    """Build the synthetic scalar field used to carry the gauge parameter
    α (and ∂_μ α via the engine's existing derivative machinery).

    ``adjoint_index`` is ``None`` for abelian groups and the adjoint
    ``IndexType`` for non-abelian groups, so that the synthetic field has
    the right index layout and prints as ``alpha(a)`` in Symbolica.
    """

    from model.metadata import scalar_field

    name = str(parameter)
    indices = () if adjoint_index is None else (adjoint_index,)
    return scalar_field(name, self_conjugate=True, indices=indices)


def _single_required_slot(*, field: Field, index, context: str) -> int:
    """Return the unique slot carrying ``index`` or raise a clear error."""

    slots = field.index_positions(index=index)
    if len(slots) != 1:
        raise ValueError(
            f"{context}: field {field.name!r} must carry exactly one "
            f"{index.name!r} index slot, found {len(slots)}."
        )
    return slots[0]


def _validate_field_parity(*, field: Field, expected: str, context: str) -> None:
    """Reject a field whose Grassmann parity does not match ``expected``."""

    is_odd = field.statistics == "fermion"
    if expected == "odd" and not is_odd:
        raise ValueError(
            f"{context}: field {field.name!r} must be Grassmann-odd "
            "(statistics='fermion')."
        )
    if expected == "even" and is_odd:
        raise ValueError(
            f"{context}: field {field.name!r} must be Grassmann-even "
            "(statistics != 'fermion')."
        )


def _gauge_operator_context(
    *,
    group,
    operator_name: str,
    require_structure_constant: bool = False,
) -> _GaugeOperatorContext:
    abelian = bool(group.abelian)
    gauge_boson = group.gauge_boson
    gauge_coupling = group.coupling
    adjoint_index = None

    if not abelian:
        if gauge_boson is None:
            raise ValueError(
                f"{operator_name}({group.name!r}): non-abelian group has no "
                "gauge_boson; cannot infer the adjoint index."
            )
        if require_structure_constant and group.structure_constant is None:
            raise ValueError(
                f"{operator_name}({group.name!r}): non-abelian {operator_name} requires "
                "a structure_constant."
            )
        adjoint_index = _first_non_lorentz_index(gauge_boson)
        if adjoint_index is None:
            raise ValueError(
                f"{operator_name}({group.name!r}): could not find a "
                "non-Lorentz (adjoint) index on the gauge boson "
                f"{gauge_boson.name!r}."
            )

    return _GaugeOperatorContext(
        group=group,
        operator_name=operator_name,
        abelian=abelian,
        gauge_boson=gauge_boson,
        gauge_coupling=gauge_coupling,
        adjoint_index=adjoint_index,
    )


def _gauge_operator_label_factory(
    *,
    operator_name: str,
    adjoint_index,
):
    counter = [0]

    def _next(stem: str):
        counter[0] += 1
        return S(f"{stem}_{operator_name}_{counter[0]}")

    def _fresh_adj_label():
        if adjoint_index is None:
            return None
        return _next(adjoint_index.prefix)

    return _next, _fresh_adj_label


def _field_matches_reference(
    *,
    target,
    field: Field,
) -> bool:
    if target is None:
        return False
    if isinstance(target, Field):
        return target is field
    target_text = str(target)
    return target_text in (
        field.name,
        str(field.symbol),
        str(field.conjugate_symbol) if field.conjugate_symbol is not None else "",
    )


def _ghost_associated_with_gauge_boson(
    *,
    field: Field,
    gauge_boson: Optional[Field],
) -> bool:
    if gauge_boson is None:
        return False
    if not field.is_ghost or field.ghost_of is None:
        return False
    return _field_matches_reference(target=field.ghost_of, field=gauge_boson)


def _validate_brst_ghost_field(
    *,
    field: Field,
    gauge_boson: Optional[Field],
    context: str,
) -> None:
    if not field.is_ghost:
        raise ValueError(
            f"{context}: field {field.name!r} must have kind='ghost' "
            "or equivalent ghost metadata."
        )
    if not _ghost_associated_with_gauge_boson(field=field, gauge_boson=gauge_boson):
        raise ValueError(
            f"{context}: field {field.name!r} must be associated with the "
            f"selected gauge boson {getattr(gauge_boson, 'name', gauge_boson)!r} via ghost_of."
        )


def _gauge_boson_adjoint_slot_and_label(
    *,
    occurrence: FieldOccurrence,
    adjoint_index,
    context: str,
) -> tuple[int, object]:
    adj_slot = _single_required_slot(
        field=occurrence.field,
        index=adjoint_index,
        context=context,
    )
    old_adj_label = occurrence.slot_labels.get(adj_slot)
    if old_adj_label is None:
        raise ValueError(
            f"{context}: gauge boson {occurrence.field.name!r} occurrence has no adjoint label."
        )
    return adj_slot, old_adj_label


def _abelian_matter_gauge_result(
    *,
    occurrence: FieldOccurrence,
    group,
    gauge_coupling,
    carrier_occurrence_builder,
):
    if not group.charge:
        return None
    q = occurrence.field.quantum_numbers.get(group.charge)
    if q is None or q == 0:
        return None
    sign = -1 if occurrence.conjugated else 1
    carrier_occurrence = carrier_occurrence_builder(None)
    new_occurrence = _copy_occurrence(occurrence)
    return single_field_result(
        (carrier_occurrence, new_occurrence),
        coefficient=sign * Expression.I * gauge_coupling * q,
    )


def _nonabelian_matter_gauge_result(
    *,
    occurrence: FieldOccurrence,
    group,
    gauge_coupling,
    carrier_occurrence_builder,
    next_label,
    fresh_adj_label,
    context: str,
):
    rep_and_slots = group.matter_representation_and_slots(occurrence.field)
    if rep_and_slots is None:
        return None
    rep, slots = rep_and_slots
    slot_labels = occurrence.slot_labels
    summands: list[OperatorSummand] = []
    for slot in slots:
        old_rep_label = slot_labels.get(slot)
        if old_rep_label is None:
            raise ValueError(
                f"{context}: field {occurrence.field.name!r} has "
                f"no explicit label on representation slot {slot + 1}; cannot "
                "insert a contracted generator without one."
            )
        new_rep_label = next_label(rep.index.prefix)
        adj_label = fresh_adj_label()

        new_occurrence = _replace_slot_label(
            occurrence.field,
            occurrence,
            slot,
            new_rep_label,
        )
        carrier_occurrence = carrier_occurrence_builder(adj_label)

        if occurrence.conjugated:
            generator = rep.build_generator(adj_label, new_rep_label, old_rep_label)
            sign = -1
        else:
            generator = rep.build_generator(adj_label, old_rep_label, new_rep_label)
            sign = 1

        summands.append(
            OperatorSummand(
                coefficient=sign * Expression.I * gauge_coupling * generator,
                replacement=(carrier_occurrence, new_occurrence),
            )
        )

    return OperatorAtomResult(summands=tuple(summands))


def _set_operator_metadata(operator, /, **attrs):
    for name, value in attrs.items():
        object.__setattr__(operator, name, value)


def gauge_variation(
    *,
    group,
    parameter: object = "alpha",
    name: Optional[str] = None,
) -> FieldOperator:
    """Infinitesimal gauge variation operator for one gauge group.

    Returns a ``FieldOperator`` (parity 0) which, applied through
    ``CompiledLagrangian.apply_operator``, performs the graded Leibniz
    expansion of the **linearized** gauge transformation

    * matter rep R, fundamental indices ``(i, j, ...)``:
        ``δ Φ_i = + i g α^a T^a_{ij} Φ_j``
        ``δ Φ̄_i = - i g α^a Φ̄_j T^a_{ji}``
    * abelian U(1) charge ``q`` (read from ``field.quantum_numbers[group.charge]``):
        ``δ Φ = + i g q α Φ`` ,  ``δ Φ̄ = - i g q α Φ̄``
    * gauge boson:
        ``δ A^a_μ = + ∂_μ α^a - g f^{abc} α^b A^c_μ``  (non-abelian)
        ``δ A_μ   = + ∂_μ α``                          (abelian)

    The signs follow the codebase convention
    ``D_μ = ∂_μ - i g A_μ`` and ``F^a_{μν} = ∂_μ A^a_ν - ∂_ν A^a_μ
    + g f^{abc} A^b_μ A^c_ν`` documented at the top of
    ``compiler/gauge.py``; in particular the inhomogeneous part of
    ``δA^a_μ`` is ``+∂_μ α^a``, not ``-∂_μ α^a``, and the homogeneous
    part comes with a *minus* sign in front of ``g f^{abc} α^b A^c_μ``.

    Implementation note. ``α`` is materialised as a synthetic scalar
    ``Field`` (with the group's adjoint index if non-abelian). This is
    what makes ``δ`` commute correctly with existing partial derivatives
    on the acted slot: each variation hands the engine a *product*
    replacement ``(α, Φ)`` (or ``(α, A)``), and the bosonic Leibniz
    fan-out already in ``apply_field_operator`` distributes any
    pre-existing ``DerivativeAction`` on the slot across both
    replacement factors. There is **no** new derivative bookkeeping
    here; the inhomogeneous gauge-field part uses ``new_derivatives``
    to attach the fresh ``∂_μ`` on top of α.

    Parameters
    ----------
    group:
        The ``GaugeGroup`` whose infinitesimal action is requested.
    parameter:
        Symbolica symbol *or* string used to name α. Plain strings are
        wrapped automatically; pass a Symbolica symbol if you have an
        existing one. The synthetic field is called ``str(parameter)``.
    name:
        Optional human-readable operator name (used by diagnostics).
    """

    op_name = name if name is not None else f"delta_{group.name}"
    context = _gauge_operator_context(
        group=group,
        operator_name="gauge_variation",
    )
    alpha_field = _build_alpha_field(parameter, context.adjoint_index)
    _next, _fresh_adj_label = _gauge_operator_label_factory(
        operator_name=op_name,
        adjoint_index=context.adjoint_index,
    )

    def _alpha_occurrence(adj_label):
        if context.adjoint_index is None:
            return alpha_field.occurrence()
        return alpha_field.occurrence(
            labels=alpha_field.pack_slot_labels({0: adj_label}),
        )

    def _matter_abelian(occurrence: FieldOccurrence):
        return _abelian_matter_gauge_result(
            occurrence=occurrence,
            group=group,
            gauge_coupling=context.gauge_coupling,
            carrier_occurrence_builder=_alpha_occurrence,
        )

    def _matter_non_abelian(occurrence: FieldOccurrence):
        return _nonabelian_matter_gauge_result(
            occurrence=occurrence,
            group=group,
            gauge_coupling=context.gauge_coupling,
            carrier_occurrence_builder=_alpha_occurrence,
            next_label=_next,
            fresh_adj_label=_fresh_adj_label,
            context=f"gauge_variation({group.name!r})",
        )

    def _gauge_boson_variation(occurrence: FieldOccurrence):
        lorentz_label = _lorentz_label_of(occurrence)
        if lorentz_label is None:
            raise ValueError(
                f"gauge_variation({group.name!r}): gauge boson "
                f"{occurrence.field.name!r} occurrence has no Lorentz label."
            )

        # ---- inhomogeneous part: + ∂_μ α^a  (in the codebase convention)
        if context.abelian:
            inhom_alpha_occ = _alpha_occurrence(None)
        else:
            adj_slot, old_adj_label = _gauge_boson_adjoint_slot_and_label(
                occurrence=occurrence,
                adjoint_index=context.adjoint_index,
                context=f"gauge_variation({group.name!r})",
            )
            inhom_alpha_occ = _alpha_occurrence(old_adj_label)

        inhom = OperatorSummand(
            coefficient=Expression.num(1),
            replacement=(inhom_alpha_occ,),
            new_derivatives=(
                DerivativeAction(target=0, lorentz_index=lorentz_label),
            ),
        )

        if context.abelian or group.structure_constant is None:
            return OperatorAtomResult(summands=(inhom,))

        # ---- homogeneous part: - g f^{abc} α^b A^c_μ  (with 'a' = old label)
        b_label = _fresh_adj_label()
        c_label = _fresh_adj_label()
        hom_alpha_occ = _alpha_occurrence(b_label)
        hom_A_occ = _replace_slot_label(
            occurrence.field, occurrence, adj_slot, c_label
        )
        f_abc = group.structure_constant(old_adj_label, b_label, c_label)
        hom = OperatorSummand(
            coefficient=-context.gauge_coupling * f_abc,
            replacement=(hom_alpha_occ, hom_A_occ),
        )
        return OperatorAtomResult(summands=(inhom, hom))

    def on_field(occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        field = occurrence.field
        if context.gauge_boson is not None and field is context.gauge_boson:
            return _gauge_boson_variation(occurrence)
        if context.abelian:
            return _matter_abelian(occurrence)
        return _matter_non_abelian(occurrence)

    op = FieldOperator(name=op_name, parity=0, on_field=on_field)
    _set_operator_metadata(
        op,
        _gauge_parameter_field=alpha_field,
    )
    return op


def brst_transformation(
    *,
    group,
    ghost: Field,
    antighost: Optional[Field] = None,
    auxiliary: Optional[Field] = None,
    name: Optional[str] = None,
) -> FieldOperator:
    """BRST differential for one gauge group.

    The returned runtime operator is Grassmann-odd (``parity = 1``) and
    implements the minimal Yang-Mills + ghost sector rules

    * ``s A^a_mu = partial_mu c^a + g f^{abc} A^b_mu c^c``
    * ``s c^a = -1/2 g f^{abc} c^b c^c``
    * ``s cbar^a = B^a``
    * ``s B^a = 0``

    in the codebase's ordered ``InteractionTerm`` representation, so the
    existing graded Leibniz rule supplies the required signs on products.

    If ``antighost`` is omitted but ``auxiliary`` is supplied, the common
    Faddeev-Popov convention is inferred automatically: the antighost is
    taken to be the conjugated occurrence ``ghost.bar(...)``. To keep that
    inference explicit and readable, the ghost field must declare a concrete
    ``conjugate_symbol``.

    Convention note. The non-abelian gauge-field rule is written in the
    ordered form ``A * c`` (not ``c * A``) because the ghost is
    Grassmann-odd; preserving that order is what makes ``s^2 = 0`` work
    under the existing slot-wise graded derivation engine.
    """

    op_name = name if name is not None else f"brst_{group.name}"
    inferred_antighost_from_ghost = False
    effective_antighost = antighost
    context = _gauge_operator_context(
        group=group,
        operator_name="brst_transformation",
        require_structure_constant=True,
    )

    _validate_field_parity(
        field=ghost,
        expected="odd",
        context=f"brst_transformation({group.name!r}) ghost",
    )
    _validate_brst_ghost_field(
        field=ghost,
        gauge_boson=context.gauge_boson,
        context=f"brst_transformation({group.name!r}) ghost",
    )
    if effective_antighost is None and auxiliary is not None:
        if ghost.conjugate_symbol is None:
            raise ValueError(
                f"brst_transformation({group.name!r}): omitting antighost requires "
                f"ghost {ghost.name!r} to declare conjugate_symbol so ghost.bar(...) "
                "can represent the antighost. Pass antighost=... explicitly "
                "otherwise."
            )
        effective_antighost = ghost
        inferred_antighost_from_ghost = True

    if antighost is not None:
        _validate_field_parity(
            field=antighost,
            expected="odd",
            context=f"brst_transformation({group.name!r}) antighost",
        )
        _validate_brst_ghost_field(
            field=antighost,
            gauge_boson=context.gauge_boson,
            context=f"brst_transformation({group.name!r}) antighost",
        )
    if auxiliary is not None:
        _validate_field_parity(
            field=auxiliary,
            expected="even",
            context=f"brst_transformation({group.name!r}) auxiliary",
        )
    if effective_antighost is not None and auxiliary is None:
        raise ValueError(
            f"brst_transformation({group.name!r}): antighost supplied without "
            "an auxiliary field. Pass auxiliary=... explicitly or omit the "
            "antighost rule."
        )

    if context.abelian:
        adjoint_index = None
        ghost_slot = None
        auxiliary_slot = None
    else:
        adjoint_index = context.adjoint_index
        ghost_slot = _single_required_slot(
            field=ghost,
            index=adjoint_index,
            context=f"brst_transformation({group.name!r}) ghost",
        )
        auxiliary_slot = None
        if auxiliary is not None:
            auxiliary_slot = _single_required_slot(
                field=auxiliary,
                index=adjoint_index,
                context=f"brst_transformation({group.name!r}) auxiliary",
            )
        if effective_antighost is not None and effective_antighost is not ghost:
            _single_required_slot(
                field=effective_antighost,
                index=adjoint_index,
                context=f"brst_transformation({group.name!r}) antighost",
            )

    _next, _fresh_adj_label = _gauge_operator_label_factory(
        operator_name=op_name,
        adjoint_index=adjoint_index,
    )

    def _ghost_occurrence(adj_label):
        if adjoint_index is None:
            return ghost.occurrence()
        return ghost.occurrence(
            labels=ghost.pack_slot_labels({ghost_slot: adj_label}),
        )

    def _auxiliary_occurrence(adj_label):
        if auxiliary is None:
            return None
        if adjoint_index is None:
            return auxiliary.occurrence()
        return auxiliary.occurrence(
            labels=auxiliary.pack_slot_labels({auxiliary_slot: adj_label}),
        )

    def _gauge_boson_variation(occurrence: FieldOccurrence):
        lorentz_label = _lorentz_label_of(occurrence)
        if lorentz_label is None:
            raise ValueError(
                f"brst_transformation({group.name!r}): gauge boson "
                f"{occurrence.field.name!r} occurrence has no Lorentz label."
            )

        if context.abelian:
            inhom_ghost_occ = _ghost_occurrence(None)
        else:
            adj_slot, old_adj_label = _gauge_boson_adjoint_slot_and_label(
                occurrence=occurrence,
                adjoint_index=adjoint_index,
                context=f"brst_transformation({group.name!r})",
            )
            inhom_ghost_occ = _ghost_occurrence(old_adj_label)

        inhom = OperatorSummand(
            coefficient=Expression.num(1),
            replacement=(inhom_ghost_occ,),
            new_derivatives=(
                DerivativeAction(target=0, lorentz_index=lorentz_label),
            ),
        )

        if context.abelian:
            return OperatorAtomResult(summands=(inhom,))

        b_label = _fresh_adj_label()
        c_label = _fresh_adj_label()
        hom_A_occ = _replace_slot_label(occurrence.field, occurrence, adj_slot, b_label)
        hom_ghost_occ = _ghost_occurrence(c_label)
        f_abc = group.structure_constant(old_adj_label, b_label, c_label)
        hom = OperatorSummand(
            coefficient=context.gauge_coupling * f_abc,
            replacement=(hom_A_occ, hom_ghost_occ),
        )
        return OperatorAtomResult(summands=(inhom, hom))

    def _ghost_variation(occurrence: FieldOccurrence):
        if context.abelian:
            return zero_result()

        slot_labels = occurrence.slot_labels
        old_adj_label = slot_labels.get(ghost_slot)
        if old_adj_label is None:
            raise ValueError(
                f"brst_transformation({group.name!r}): ghost occurrence has no "
                "adjoint label."
            )
        b_label = _fresh_adj_label()
        c_label = _fresh_adj_label()
        f_abc = group.structure_constant(old_adj_label, b_label, c_label)
        coefficient = -Expression.num(1) / Expression.num(2) * context.gauge_coupling * f_abc
        return single_field_result(
            (_ghost_occurrence(b_label), _ghost_occurrence(c_label)),
            coefficient=coefficient,
        )

    def _matter_variation(occurrence: FieldOccurrence):
        if context.abelian:
            return _abelian_matter_gauge_result(
                occurrence=occurrence,
                group=group,
                gauge_coupling=context.gauge_coupling,
                carrier_occurrence_builder=_ghost_occurrence,
            )
        return _nonabelian_matter_gauge_result(
            occurrence=occurrence,
            group=group,
            gauge_coupling=context.gauge_coupling,
            carrier_occurrence_builder=_ghost_occurrence,
            next_label=_next,
            fresh_adj_label=_fresh_adj_label,
            context=f"brst_transformation({group.name!r})",
        )

    def _antighost_variation(occurrence: FieldOccurrence):
        aux_occ = _auxiliary_occurrence(None)
        if aux_occ is None:
            return None
        return single_field_result(aux_occ)

    def on_field(occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        field = occurrence.field
        if context.gauge_boson is not None and field is context.gauge_boson:
            return _gauge_boson_variation(occurrence)
        if field is ghost and not occurrence.conjugated:
            return _ghost_variation(occurrence)
        if auxiliary is not None and field is auxiliary:
            return zero_result()
        if effective_antighost is ghost:
            if field is ghost and occurrence.conjugated:
                return _antighost_variation(occurrence)
        elif effective_antighost is not None and field is effective_antighost and not occurrence.conjugated:
            return _antighost_variation(occurrence)
        return _matter_variation(occurrence)

    op = FieldOperator(name=op_name, parity=1, on_field=on_field)
    _set_operator_metadata(
        op,
        _brst_group=group,
        _brst_ghost_field=ghost,
        _brst_antighost_field=effective_antighost,
        _brst_antighost_inferred_from_ghost=inferred_antighost_from_ghost,
        _brst_auxiliary_field=auxiliary,
    )
    return op


__all__ = (
    "OperatorExpansionError",
    "FieldOperator",
    "OperatorAtomResult",
    "OperatorSummand",
    "Parity",
    "TermOperator",
    "apply_operator",
    "apply_operator_to_term",
    "apply_field_operator",
    "apply_field_operator_to_term",
    "apply_term_operator",
    "apply_term_operator_to_term",
    "brst_transformation",
    "constant_result",
    "gauge_variation",
    "partial",
    "replacement_operator",
    "single_field_result",
    "zero_result",
)
