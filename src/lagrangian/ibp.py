"""Scalar integration-by-parts normal forms on compiled interaction terms."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations as iter_permutations

from model.interactions import DerivativeAction, FieldOccurrence, InteractionTerm


def _symbolic_key(value):
    if isinstance(value, tuple):
        return tuple(_symbolic_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_symbolic_key(item) for item in value)
    if hasattr(value, "to_canonical_string"):
        return ("expr", value.to_canonical_string())
    return value


def _occurrence_key(occurrence: FieldOccurrence) -> tuple[object, ...]:
    labels = tuple(
        sorted((kind, _symbolic_key(value)) for kind, value in occurrence.labels.items())
    ) if occurrence.labels else ()
    return (id(occurrence.field), bool(occurrence.conjugated), labels)


def _derivative_sort_key(action: DerivativeAction) -> tuple[int, object]:
    return action.target, _symbolic_key(action.lorentz_index)


def _term_key(term: InteractionTerm) -> tuple[object, ...]:
    derivatives = tuple(sorted(term.derivatives, key=_derivative_sort_key))
    return (
        tuple(_occurrence_key(occurrence) for occurrence in term.fields),
        tuple((action.target, _symbolic_key(action.lorentz_index)) for action in derivatives),
        term.closed_dirac_bilinears,
    )


def _bundle_sort_key(bundle: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(_symbolic_key(item) for item in bundle)


def _left_normalize_identical_scalar_slots(term: InteractionTerm) -> InteractionTerm:
    if not term.fields:
        return term

    bundles: list[list[object]] = [[] for _ in term.fields]
    for action in term.derivatives:
        bundles[action.target].append(action.lorentz_index)

    normalized_bundles = [
        tuple(sorted(bundle, key=_symbolic_key))
        for bundle in bundles
    ]
    normalized_bundles.sort(key=_bundle_sort_key, reverse=True)

    derivatives = tuple(
        DerivativeAction(target=target, lorentz_index=lorentz_index)
        for target, bundle in enumerate(normalized_bundles)
        for lorentz_index in bundle
    )
    return replace(term, derivatives=derivatives)


def _term_from_bundles(
    term: InteractionTerm,
    bundles: tuple[tuple[object, ...], ...],
    *,
    coupling=None,
) -> InteractionTerm:
    derivatives = tuple(
        DerivativeAction(target=target, lorentz_index=lorentz_index)
        for target, bundle in enumerate(bundles)
        for lorentz_index in bundle
    )
    return replace(
        term,
        coupling=term.coupling if coupling is None else coupling,
        derivatives=derivatives,
    )


def _canonicalize_derivatives(term: InteractionTerm) -> InteractionTerm:
    term = replace(
        term,
        derivatives=tuple(sorted(term.derivatives, key=_derivative_sort_key)),
    )
    return term


def _simplify_coupling(coupling):
    if hasattr(coupling, "expand"):
        return coupling.expand()
    return coupling


def _is_zero_coupling(coupling) -> bool:
    simplified = _simplify_coupling(coupling)
    if hasattr(simplified, "to_canonical_string"):
        return simplified.to_canonical_string() == "0"
    return simplified == 0


def _combine_like_terms(terms: tuple[InteractionTerm, ...]) -> tuple[InteractionTerm, ...]:
    grouped: dict[tuple[object, ...], InteractionTerm] = {}
    for raw_term in terms:
        term = _left_normalize_identical_scalar_slots(
            _canonicalize_derivatives(raw_term)
        )
        key = _term_key(term)
        prior = grouped.get(key)
        if prior is None:
            grouped[key] = replace(term, coupling=_simplify_coupling(term.coupling))
            continue
        grouped[key] = replace(
            prior,
            coupling=_simplify_coupling(prior.coupling + term.coupling),
        )

    reduced = [
        term
        for _, term in sorted(grouped.items(), key=lambda item: repr(item[0]))
        if not _is_zero_coupling(term.coupling)
    ]
    return tuple(reduced)


def _validate_ibp_v1_term(term: InteractionTerm) -> None:
    if term.closed_dirac_bilinears:
        raise ValueError("ibp_normal_form() v1 does not support Dirac bilinears.")
    if not term.fields:
        return

    first = term.fields[0]
    if first.field.kind != "scalar":
        raise ValueError("ibp_normal_form() v1 supports only scalar terms.")
    if first.field.indices:
        raise ValueError(
            "ibp_normal_form() v1 supports only unlabeled scalar species without extra indices."
        )

    reference = (first.field, bool(first.conjugated))
    for occurrence in term.fields:
        if occurrence.field.kind != "scalar":
            raise ValueError("ibp_normal_form() v1 supports only scalar terms.")
        if occurrence.field is not reference[0] or bool(occurrence.conjugated) != reference[1]:
            raise ValueError("ibp_normal_form() v1 rejects mixed scalar species or conjugations.")
        if occurrence.field.indices:
            raise ValueError(
                "ibp_normal_form() v1 supports only scalar species without nontrivial indices."
            )
        if occurrence.labels:
            raise ValueError(
                "ibp_normal_form() v1 supports only unlabeled scalar occurrences."
            )


def _symmetrize_identical_scalar_term(term: InteractionTerm) -> tuple[InteractionTerm, ...]:
    if not term.fields:
        return (term,)

    bundles: list[tuple[object, ...]] = [() for _ in term.fields]
    for action in term.derivatives:
        items = list(bundles[action.target])
        items.append(action.lorentz_index)
        bundles[action.target] = tuple(sorted(items, key=_symbolic_key))

    unique_bundles = tuple(dict.fromkeys(iter_permutations(tuple(bundles))))
    if len(unique_bundles) == 1:
        return (term,)

    factor = len(unique_bundles)
    return tuple(
        _term_from_bundles(
            term,
            bundle_assignment,
            coupling=_simplify_coupling(term.coupling / factor),
        )
        for bundle_assignment in unique_bundles
    )


def _remove_one_last_slot_derivative(
    derivatives: tuple[DerivativeAction, ...],
    *,
    last_slot: int,
) -> tuple[DerivativeAction, tuple[DerivativeAction, ...]]:
    removed = False
    removed_action = None
    remaining: list[DerivativeAction] = []
    for action in derivatives:
        if not removed and action.target == last_slot:
            removed = True
            removed_action = action
            continue
        remaining.append(action)
    if removed_action is None:
        raise ValueError("No derivative found on the requested slot.")
    return removed_action, tuple(remaining)


def _ibp_normalize_term(
    term: InteractionTerm,
    memo: dict[tuple[object, ...], tuple[InteractionTerm, ...]],
) -> tuple[InteractionTerm, ...]:
    term = _canonicalize_derivatives(term)
    key = (
        _term_key(term),
        _symbolic_key(term.coupling),
    )
    cached = memo.get(key)
    if cached is not None:
        return cached

    _validate_ibp_v1_term(term)
    if not term.fields:
        memo[key] = (term,)
        return memo[key]

    last_slot = len(term.fields) - 1
    if len(term.fields) == 1:
        memo[key] = () if term.derivatives else (term,)
        return memo[key]

    if not any(action.target == last_slot for action in term.derivatives):
        memo[key] = (term,)
        return memo[key]

    removed_action, remaining = _remove_one_last_slot_derivative(
        term.derivatives,
        last_slot=last_slot,
    )

    expanded: list[InteractionTerm] = []
    for target in range(last_slot):
        branch = replace(
            term,
            coupling=_simplify_coupling(-term.coupling),
            derivatives=remaining + (
                DerivativeAction(target=target, lorentz_index=removed_action.lorentz_index),
            ),
        )
        expanded.extend(_ibp_normalize_term(branch, memo))

    memo[key] = _combine_like_terms(tuple(expanded))
    return memo[key]


def ibp_normal_form(lagrangian):
    """Return the v1 scalar IBP normal form of a compiled Lagrangian."""

    from model.lagrangian import CompiledLagrangian

    memo: dict[tuple[object, ...], tuple[InteractionTerm, ...]] = {}
    expanded: list[InteractionTerm] = []
    for term in lagrangian.terms:
        _validate_ibp_v1_term(term)
        for sym_term in _symmetrize_identical_scalar_term(term):
            expanded.extend(_ibp_normalize_term(sym_term, memo))

    return CompiledLagrangian(
        terms=_combine_like_terms(tuple(expanded)),
        parameters=lagrangian.parameters,
    )
