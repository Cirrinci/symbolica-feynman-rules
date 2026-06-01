"""Symbolica export helpers for lowered ``InteractionTerm`` objects.

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

On top of the raw export, this module also provides a small amount of
ergonomics for structural manipulations that users naturally expect when
working with Symbolica:

* ``pattern_matches(...)`` enumerates top-level wildcard matches together
  with the residual coefficient of each matched factor;
* ``pattern_coefficient(...)`` sums those residual coefficients.

This fills a gap in Symbolica's native ``Expression.coefficient(...)``,
which is literal-only and therefore does not treat wildcards in the
coefficient argument as pattern variables.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from symbolica import AtomType, Expression, S

from model.interactions import DerivativeAction, FieldOccurrence, InteractionTerm
from model.metadata import Field


PARTIAL_DERIVATIVE_HEAD = "PartialD"
"""Symbolica function head used to wrap a derivative action in the export."""

DERIVATIVE_STYLE_PARTIALD = "partiald"
DERIVATIVE_STYLE_COORDINATE = "coordinate"


def _sanitize_symbolica_identifier(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_")
    if not cleaned:
        cleaned = "coord"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def _coordinate_symbol_for_lorentz_index(lorentz_index: object):
    text = str(lorentz_index)
    return S(f"x_{_sanitize_symbolica_identifier(text)}")


def _normalize_derivative_export(
    *,
    derivative_style: str,
    coordinate_map: Optional[Mapping[object, object]],
):
    if derivative_style not in (DERIVATIVE_STYLE_PARTIALD, DERIVATIVE_STYLE_COORDINATE):
        raise ValueError(
            "derivative_style must be 'partiald' or 'coordinate'."
        )
    normalized_coordinate_map = dict(coordinate_map or {})
    return derivative_style, normalized_coordinate_map


def _occurrence_label_arguments(occurrence: FieldOccurrence) -> tuple[object, ...]:
    """Return slot-ordered label arguments for one ``FieldOccurrence``.

    Missing labels are rendered as ``S('?')`` so that the user-facing
    expression always has the same arity as the field's declared index
    layout, even when an occurrence ships without explicit labels for some
    slot. This keeps the export robust on partial / hand-built terms.
    """

    placeholder = S("?")
    return tuple(
        label if label is not None else placeholder
        for label in occurrence.slot_labels.values
    )


def _occurrence_atom(
    occurrence: FieldOccurrence,
    *,
    coordinate_arguments: Sequence[object] = (),
) -> object:
    """Build the bare ``species(label_1, ..., label_n)`` Symbolica atom."""

    species = occurrence.species
    arguments = _occurrence_label_arguments(occurrence) + tuple(coordinate_arguments)
    if not arguments:
        return species
    return species(*arguments)


def _wrap_with_derivatives(
    atom: object,
    derivative_indices: Sequence[object],
    *,
    derivative_style: str,
    coordinate_map: Mapping[object, object],
) -> object:
    """Wrap an occurrence atom in ``PartialD(..., mu)`` per derivative action."""

    result = atom
    for lorentz_index in derivative_indices:
        if derivative_style == DERIVATIVE_STYLE_PARTIALD:
            partial = S(PARTIAL_DERIVATIVE_HEAD)
            result = partial(result, lorentz_index)
            continue
        coordinate = coordinate_map.get(lorentz_index)
        if coordinate is None:
            coordinate = _coordinate_symbol_for_lorentz_index(lorentz_index)
        result = result.derivative(coordinate)
    return result


def _term_derivative_indices(term: InteractionTerm) -> tuple[object, ...]:
    ordered_unique: list[object] = []
    for action in term.derivatives:
        if any(existing == action.lorentz_index for existing in ordered_unique):
            continue
        ordered_unique.append(action.lorentz_index)
    return tuple(ordered_unique)


def _atom_sort_key(atom: object) -> str:
    if hasattr(atom, "to_canonical_string"):
        return atom.to_canonical_string()
    return str(atom)


def _odd_factor_sort_sign(
    factor_data: Sequence[tuple[object, bool]],
) -> int:
    odd_slots = [
        slot
        for slot, (_atom, is_odd) in enumerate(factor_data)
        if is_odd
    ]
    if len(odd_slots) < 2:
        return 1

    target = sorted(
        odd_slots,
        key=lambda slot: _atom_sort_key(factor_data[slot][0]),
    )
    working = list(odd_slots)
    sign = 1
    for slot, desired in enumerate(target):
        current_slot = working.index(desired, slot)
        while current_slot > slot:
            working[current_slot], working[current_slot - 1] = (
                working[current_slot - 1],
                working[current_slot],
            )
            sign *= -1
            current_slot -= 1
    return sign


def _terms_derivative_indices(terms: Iterable[InteractionTerm]) -> tuple[object, ...]:
    ordered_unique: list[object] = []
    for term in terms:
        for lorentz_index in _term_derivative_indices(term):
            if any(existing == lorentz_index for existing in ordered_unique):
                continue
            ordered_unique.append(lorentz_index)
    return tuple(ordered_unique)


def interaction_term_to_symbolica(
    term: InteractionTerm,
    *,
    derivative_style: str = DERIVATIVE_STYLE_PARTIALD,
    coordinate_map: Optional[Mapping[object, object]] = None,
    coordinate_arguments: Optional[Sequence[object]] = None,
) -> object:
    """Render a single ``InteractionTerm`` as a Symbolica expression.

    The result is the coupling multiplied by one atom per
    ``FieldOccurrence``, with each atom optionally wrapped in
    ``PartialD(..., mu)`` calls for the term's derivative actions.

    Ordering is **not** preserved (Symbolica multiplication is
    commutative); see the module docstring for the implications.
    """

    derivative_style, normalized_coordinate_map = _normalize_derivative_export(
        derivative_style=derivative_style,
        coordinate_map=coordinate_map,
    )
    if coordinate_arguments is None:
        term_coordinate_indices = _term_derivative_indices(term)
        coordinate_arguments = tuple(
            normalized_coordinate_map.get(index, _coordinate_symbol_for_lorentz_index(index))
            for index in term_coordinate_indices
        ) if derivative_style == DERIVATIVE_STYLE_COORDINATE else ()
    else:
        coordinate_arguments = tuple(coordinate_arguments)

    derivatives_by_slot: dict[int, list[object]] = {}
    for action in term.derivatives:
        derivatives_by_slot.setdefault(action.target, []).append(action.lorentz_index)

    factor_data: list[tuple[object, bool]] = []
    for slot, occurrence in enumerate(term.fields):
        atom = _occurrence_atom(
            occurrence,
            coordinate_arguments=coordinate_arguments if derivative_style == DERIVATIVE_STYLE_COORDINATE else (),
        )
        wrapped = _wrap_with_derivatives(
            atom,
            derivatives_by_slot.get(slot, ()),
            derivative_style=derivative_style,
            coordinate_map=normalized_coordinate_map,
        )
        factor_data.append((wrapped, occurrence.field.statistics == "fermion"))

    expression: object = _odd_factor_sort_sign(factor_data) * term.coupling
    for wrapped, _is_odd in factor_data:
        expression = expression * wrapped
    return expression


def interaction_terms_to_symbolica(
    terms: Iterable[InteractionTerm],
    *,
    derivative_style: str = DERIVATIVE_STYLE_PARTIALD,
    coordinate_map: Optional[Mapping[object, object]] = None,
) -> object:
    """Sum the Symbolica expressions of a sequence of interaction terms."""

    terms = tuple(terms)
    derivative_style, normalized_coordinate_map = _normalize_derivative_export(
        derivative_style=derivative_style,
        coordinate_map=coordinate_map,
    )
    if derivative_style == DERIVATIVE_STYLE_COORDINATE:
        coordinate_arguments = tuple(
            normalized_coordinate_map.get(index, _coordinate_symbol_for_lorentz_index(index))
            for index in _terms_derivative_indices(terms)
        )
    else:
        coordinate_arguments = ()

    total: object = Expression.num(0)
    for term in terms:
        total = total + interaction_term_to_symbolica(
            term,
            derivative_style=derivative_style,
            coordinate_map=normalized_coordinate_map,
            coordinate_arguments=coordinate_arguments,
        )
    return total


def lagrangian_to_symbolica(
    lagrangian,
    *,
    flavor_expand=False,
    derivative_style: str = DERIVATIVE_STYLE_PARTIALD,
    coordinate_map: Optional[Mapping[object, object]] = None,
) -> object:
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
    return interaction_terms_to_symbolica(
        terms,
        derivative_style=derivative_style,
        coordinate_map=coordinate_map,
    )


# ---------------------------------------------------------------------------
# Pattern-oriented Symbolica helpers
# ---------------------------------------------------------------------------


def _top_level_terms(expr: Expression) -> tuple[Expression, ...]:
    """Return additive top-level terms, or ``(expr,)`` for non-sums."""

    if expr.get_type() == AtomType.Add:
        return tuple(expr)
    return (expr,)


def _canonical_key(expr: Expression) -> str:
    return expr.to_canonical_string()


@dataclass(frozen=True)
class PatternCoefficientMatch:
    """One top-level wildcard match together with its residual coefficient."""

    term: Expression
    matched_factor: Expression
    coefficient: Expression
    bindings: tuple[tuple[Expression, Expression], ...]


def pattern_matches(
    expr: Expression,
    pattern: Expression,
    *,
    expand: bool = False,
    deduplicate: bool = True,
) -> tuple[PatternCoefficientMatch, ...]:
    """Enumerate top-level wildcard matches and their residual coefficients.

    Each match is computed against one additive term of ``expr``. The helper
    only considers matches rooted at the top level of that term
    (``min_level=max_level=0``), which is the useful notion for Lagrangian
    monomials where one wants to identify a matched factor and keep the
    leftover multiplicative coefficient.

    ``deduplicate=True`` removes purely commutative duplicate matches inside
    one additive term, e.g. the two wildcard orderings of ``A(mu)*A(nu)``
    against the same bilinear.
    """

    if expand and hasattr(expr, "expand"):
        expr = expr.expand()

    matches: list[PatternCoefficientMatch] = []
    for term in _top_level_terms(expr):
        seen_factors: set[str] = set()
        for match_map in term.match(
            pattern,
            min_level=0,
            max_level=0,
            partial=True,
        ):
            matched_factor = pattern.replace_wildcards(match_map)
            matched_key = _canonical_key(matched_factor)
            if deduplicate and matched_key in seen_factors:
                continue
            seen_factors.add(matched_key)

            coefficient = term.coefficient(matched_factor)
            bindings = tuple(
                sorted(
                    match_map.items(),
                    key=lambda item: item[0].to_canonical_string(),
                )
            )
            matches.append(
                PatternCoefficientMatch(
                    term=term,
                    matched_factor=matched_factor,
                    coefficient=coefficient,
                    bindings=bindings,
                )
            )
    return tuple(matches)


def pattern_coefficient(
    expr: Expression,
    pattern: Expression,
    *,
    expand: bool = False,
    deduplicate: bool = True,
) -> Expression:
    """Return the summed residual coefficient of a wildcard pattern."""

    total = Expression.num(0)
    for match in pattern_matches(
        expr,
        pattern,
        expand=expand,
        deduplicate=deduplicate,
    ):
        total += match.coefficient
    return total


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
    "DERIVATIVE_STYLE_COORDINATE",
    "DERIVATIVE_STYLE_PARTIALD",
    "PARTIAL_DERIVATIVE_HEAD",
    "PatternCoefficientMatch",
    "SymbolicaFieldRegistry",
    "interaction_term_to_symbolica",
    "interaction_terms_to_symbolica",
    "lagrangian_to_symbolica",
    "pattern_coefficient",
    "pattern_matches",
)
