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
from typing import Callable, Optional, Sequence, Union

from symbolica import Expression

from model.interactions import (
    DerivativeAction,
    FieldOccurrence,
    InteractionTerm,
)
from model.metadata import Field, FieldRole


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


Parity = int  # 0 = even (bosonic), 1 = odd (fermionic / BRST / ghost-like)


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
    """

    coefficient: object = 1
    replacement: tuple[FieldOccurrence, ...] = ()


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
) -> OperatorAtomResult:
    """Convenience: build a result with one summand."""

    if isinstance(replacement, FieldOccurrence):
        rep_tuple: tuple[FieldOccurrence, ...] = (replacement,)
    else:
        rep_tuple = tuple(replacement)
    return OperatorAtomResult(
        summands=(OperatorSummand(coefficient=coefficient, replacement=rep_tuple),),
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
        Any ``DerivativeAction(target=k, ...)`` on the slot being acted on
        is re-targeted to the first replacement slot. If False, the engine
        refuses to act on slots that carry derivative actions and raises a
        structured error unless the user provides their own custom logic
        (future extension).
    """

    name: str
    parity: Parity = 0
    on_field: Optional[Callable[[FieldOccurrence], Optional[OperatorAtomResult]]] = None
    commute_with_partial_derivative: bool = True

    def __post_init__(self):
        if self.parity not in (0, 1):
            raise ValueError(
                f"FieldOperator {self.name!r} got parity={self.parity!r}; "
                "expected 0 (even) or 1 (odd)."
            )

    def __call__(self, occurrence: FieldOccurrence) -> Optional[OperatorAtomResult]:
        if self.on_field is None:
            return None
        return self.on_field(occurrence)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _remap_after_splice(
    *,
    slot: int,
    replacement_len: int,
    derivatives: tuple[DerivativeAction, ...],
    bilinears: tuple[tuple[int, int], ...],
) -> tuple[tuple[DerivativeAction, ...], tuple[tuple[int, int], ...]]:
    """Remap derivative ``target``s and bilinear endpoints after splicing.

    Splicing ``replacement_len`` field occurrences in place of one slot at
    position ``slot`` shifts every slot index ``> slot`` by
    ``replacement_len - 1``. Derivative actions on ``slot`` itself are
    retargeted onto the first replacement slot (``slot``).

    The bilinear endpoints are remapped consistently. We do not attempt to
    re-derive bilinear structure here; if the operator changes a slot's
    fermion structure the caller is expected to handle that explicitly.
    """

    shift = replacement_len - 1

    def remap_index(idx: int) -> int:
        if idx == slot:
            return slot
        if idx > slot:
            return idx + shift
        return idx

    new_derivatives = tuple(
        DerivativeAction(target=remap_index(action.target), lorentz_index=action.lorentz_index)
        for action in derivatives
    )
    new_bilinears = tuple(
        (remap_index(psibar_slot), remap_index(psi_slot))
        for psibar_slot, psi_slot in bilinears
    )
    return new_derivatives, new_bilinears


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

    parity_to_the_left = 0
    new_terms: list[InteractionTerm] = []

    derivatives_by_slot: dict[int, list[DerivativeAction]] = {}
    for action in term.derivatives:
        derivatives_by_slot.setdefault(action.target, []).append(action)

    bilinears_by_slot: dict[int, list[tuple[int, int]]] = {}
    for psibar_slot, psi_slot in term.closed_dirac_bilinears:
        bilinears_by_slot.setdefault(psibar_slot, []).append((psibar_slot, psi_slot))
        bilinears_by_slot.setdefault(psi_slot, []).append((psibar_slot, psi_slot))

    for slot, occurrence in enumerate(term.fields):
        result = operator(occurrence)
        if result is None:
            parity_to_the_left = (parity_to_the_left + _occurrence_parity(occurrence)) % 2
            continue

        sign = _leibniz_sign(operator.parity, parity_to_the_left)
        derivatives_on_slot = tuple(derivatives_by_slot.get(slot, ()))
        bilinears_with_slot = tuple(bilinears_by_slot.get(slot, ()))

        for summand in result.summands:
            replacement = tuple(summand.replacement)
            _validate_replacement_against_existing_features(
                operator=operator,
                slot=slot,
                replacement=replacement,
                derivatives_on_slot=derivatives_on_slot,
                bilinears_containing_slot=bilinears_with_slot,
            )

            new_fields = term.fields[:slot] + replacement + term.fields[slot + 1 :]
            new_derivatives, new_bilinears = _remap_after_splice(
                slot=slot,
                replacement_len=len(replacement),
                derivatives=term.derivatives,
                bilinears=term.closed_dirac_bilinears,
            )
            new_coupling = _coupling_product(
                sign,
                summand.coefficient,
                term.coupling,
            )
            new_terms.append(
                replace(
                    term,
                    coupling=new_coupling,
                    fields=new_fields,
                    derivatives=new_derivatives,
                    closed_dirac_bilinears=new_bilinears,
                    origin=(term.origin + (";" if term.origin else "") + f"op:{operator.name}@slot{slot}"),
                )
            )

        parity_to_the_left = (parity_to_the_left + _occurrence_parity(occurrence)) % 2

    return tuple(new_terms)


def apply_field_operator(
    terms: Sequence[InteractionTerm],
    operator: FieldOperator,
) -> tuple[InteractionTerm, ...]:
    """Apply ``operator`` to a sequence of ``InteractionTerm`` objects."""

    expanded: list[InteractionTerm] = []
    for term in terms:
        expanded.extend(apply_field_operator_to_term(term, operator))
    return tuple(expanded)


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


__all__ = (
    "FieldOperator",
    "OperatorAtomResult",
    "OperatorSummand",
    "Parity",
    "apply_field_operator",
    "apply_field_operator_to_term",
    "constant_result",
    "replacement_operator",
    "single_field_result",
    "zero_result",
)
