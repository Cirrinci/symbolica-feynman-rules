"""Display-only Symbolica export for lowered ``InteractionTerm`` objects.

This module exposes a way to view a compiled / lowered / flavor-expanded
Lagrangian as a single Symbolica expression. The resulting expression is
useful for:

* visual inspection / pretty printing,
* coupling-level scalar algebra (``expand``, ``collect``, ``factor``),
* sanity-check comparisons against externally produced expressions
  (e.g. FeynRules-style output) at the level of products of named atoms.

What it is **not**:

* a faithful representation of fermion / ghost product ordering -- plain
  Symbolica multiplication is commutative and canonical-ordered (e.g.
  ``psibar * psi`` and ``psi * psibar`` both canonicalize to the same
  expression). For ordering-sensitive operations always work on the
  authoritative ``InteractionTerm`` structure instead;
* a self-describing format -- the exported expression carries field
  symbols and label names but not their declared ``IndexType``,
  Grassmann parity, flavor structure, or which atoms came from which
  ``Field`` declaration. Reverse reconstruction (Symbolica ->
  ``InteractionTerm``) therefore requires extra metadata; see
  ``SymbolicaFieldRegistry`` below for the shape of such a registry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from symbolica import Expression, S

from model.interactions import DerivativeAction, FieldOccurrence, InteractionTerm
from model.metadata import Field


PARTIAL_DERIVATIVE_HEAD = "PartialD"
"""Symbolica function head used to wrap a derivative action in the export."""


def _occurrence_label_arguments(occurrence: FieldOccurrence) -> tuple[object, ...]:
    """Return slot-ordered label arguments for one ``FieldOccurrence``.

    Missing labels are rendered as ``S('?')`` so that the user-facing
    expression always has the same arity as the field's declared index
    layout, even when an occurrence ships without explicit labels for some
    slot. This keeps the export robust on partial / hand-built terms.
    """

    slot_labels = occurrence.field.unpack_slot_labels(occurrence.labels)
    placeholder = S("?")
    return tuple(slot_labels.get(slot, placeholder) for slot in range(len(occurrence.field.indices)))


def _occurrence_atom(occurrence: FieldOccurrence) -> object:
    """Build the bare ``species(label_1, ..., label_n)`` Symbolica atom."""

    species = occurrence.species
    arguments = _occurrence_label_arguments(occurrence)
    if not arguments:
        return species
    return species(*arguments)


def _wrap_with_derivatives(atom: object, derivative_indices: Sequence[object]) -> object:
    """Wrap an occurrence atom in ``PartialD(..., mu)`` per derivative action."""

    partial = S(PARTIAL_DERIVATIVE_HEAD)
    result = atom
    for lorentz_index in derivative_indices:
        result = partial(result, lorentz_index)
    return result


def interaction_term_to_symbolica(term: InteractionTerm) -> object:
    """Render a single ``InteractionTerm`` as a Symbolica expression.

    The result is the coupling multiplied by one atom per
    ``FieldOccurrence``, with each atom optionally wrapped in
    ``PartialD(..., mu)`` calls for the term's derivative actions.

    Ordering is **not** preserved (Symbolica multiplication is
    commutative); see the module docstring for the implications.
    """

    derivatives_by_slot: dict[int, list[object]] = {}
    for action in term.derivatives:
        derivatives_by_slot.setdefault(action.target, []).append(action.lorentz_index)

    expression: object = term.coupling
    for slot, occurrence in enumerate(term.fields):
        atom = _occurrence_atom(occurrence)
        wrapped = _wrap_with_derivatives(atom, derivatives_by_slot.get(slot, ()))
        expression = expression * wrapped
    return expression


def interaction_terms_to_symbolica(terms: Iterable[InteractionTerm]) -> object:
    """Sum the Symbolica expressions of a sequence of interaction terms."""

    total: object = Expression.num(0)
    for term in terms:
        total = total + interaction_term_to_symbolica(term)
    return total


def lagrangian_to_symbolica(lagrangian, *, flavor_expand=False) -> object:
    """Render any object exposing a ``terms`` attribute as a Symbolica sum.

    Works for ``CompiledLagrangian`` / ``Lagrangian`` instances directly,
    as well as any duck-typed container whose ``terms`` attribute is an
    iterable of ``InteractionTerm``.

    ``flavor_expand`` mirrors the keyword of ``CompiledLagrangian.feynman_rule``.
    When truthy (or when an explicit flavor index / iterable of flavor indices
    is supplied), the export reflects the flavor-expanded compiled view of the
    Lagrangian rather than the flavor-generic source terms. For objects that
    don't expose ``_expanded_terms`` (i.e. plain duck-typed containers), the
    parameter is silently ignored and ``.terms`` is used as-is.
    """

    expanded_terms_method = getattr(lagrangian, "_expanded_terms", None)
    if expanded_terms_method is not None:
        terms = expanded_terms_method(flavor_expand=flavor_expand)
    else:
        terms = lagrangian.terms
    return interaction_terms_to_symbolica(terms)


# ---------------------------------------------------------------------------
# Reverse-direction stub
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolicaFieldRegistry:
    """Registry intended to support a *future* Symbolica -> term reverse pass.

    The reverse direction (Symbolica expression -> ordered ``InteractionTerm``)
    is **not** implemented. This dataclass documents the shape of the
    metadata that any safe reverse pass would need.

    A reverse pass would need at minimum:

    1. A registry mapping atom names (Symbolica function heads) to the
       corresponding ``Field`` / ``Parameter`` declarations, so that every
       atom in the parsed expression can be matched against a known
       declaration and its declared ``IndexType`` / parity / flavor
       structure can be recovered.
    2. An explicit ordering convention or an explicit non-commutative
       product head (e.g. ``OrderedProd(...)``) in the source expression.
       Without that, Symbolica's commutative multiplication will already
       have lost fermion / ghost order during canonicalization.
    3. A label parser that recovers per-slot index labels from each atom's
       call arguments and packs them back into the engine's kind-keyed
       label format (``Field.pack_slot_labels``).

    Until those pieces are in place, ``from_symbolica`` is intentionally
    left as a documented ``NotImplementedError`` so that callers see a
    clear, actionable diagnostic instead of a silently-incorrect result.
    """

    fields: Mapping[str, Field] = ()
    extra_atoms: Mapping[str, str] = ()

    def from_symbolica(self, expression) -> tuple[InteractionTerm, ...]:
        raise NotImplementedError(
            "Symbolica -> InteractionTerm reconstruction is not implemented. "
            "Plain Symbolica multiplication is commutative, so any fermion / "
            "ghost product order has already been lost in the source "
            "expression. A safe reverse pass needs (i) an explicit "
            "non-commutative product head in the source expression, (ii) a "
            "field/parameter registry keyed by atom name, and (iii) a "
            "per-slot label parser. See the SymbolicaFieldRegistry docstring "
            "for the design sketch."
        )


__all__ = (
    "PARTIAL_DERIVATIVE_HEAD",
    "SymbolicaFieldRegistry",
    "interaction_term_to_symbolica",
    "interaction_terms_to_symbolica",
    "lagrangian_to_symbolica",
)
