"""Post-processing helpers for declarative field transformations.

The transformation engine preserves the ordered interaction-term structure, but
matrix-valued replacements naturally generate expanded coupling expressions.
This module reduces those transformed terms back to a stable canonical form
without changing the authoritative ordered-field representation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from symbolica import AtomType, Expression, S

from lagrangian.symbolica_export import interaction_term_to_symbolica
from symbolic.spenso_structures import (
    gamma_matrix,
    gamma5_matrix,
    projector_left,
    projector_right,
    simplify_invariants,
    spinor_metric,
)
from symbolic.tensor_canonicalization import canonize_full

from .interactions import DerivativeAction, DiracBilinear, FieldOccurrence, InteractionTerm, SlotRef
from .lagrangian import CompiledLagrangian
from .metadata import LORENTZ_INDEX, Parameter, SPINOR_INDEX


_PLAIN_SPINOR_HEADS = frozenset({"PL", "PR"})


def _key(value) -> str:
    if hasattr(value, "to_canonical_string"):
        return value.to_canonical_string()
    return str(value)


def _num(value: int):
    return Expression.num(value)


def _is_zero(value) -> bool:
    if value == 0:
        return True
    return hasattr(value, "expand") and value.expand().to_canonical_string() == "0"


def _factors(expr):
    if isinstance(expr, Expression) and expr.get_type() == AtomType.Mul:
        return tuple(expr)
    return (expr,)


def _terms(expr):
    if isinstance(expr, Expression) and expr.get_type() == AtomType.Add:
        return tuple(expr)
    return (expr,)


def _product(factors: Sequence[object]):
    result = _num(1)
    for factor in factors:
        result *= factor
    return result


def _replace_symbol(value, old, new):
    if not hasattr(value, "replace"):
        return value
    return value.replace(old, new)


def _iter_expression_nodes(expr):
    if not isinstance(expr, Expression):
        return
    yield expr
    atom_type = expr.get_type()
    if atom_type in (AtomType.Add, AtomType.Mul, AtomType.Pow, AtomType.Fn):
        for child in expr:
            yield from _iter_expression_nodes(child)


def _count_symbol_occurrences(expr, label_key: str) -> int:
    count = 0
    for node in _iter_expression_nodes(expr):
        if node.get_type() == AtomType.Var and node.to_canonical_string() == label_key:
            count += 1
    return count


def _symbol_name(node) -> str:
    if isinstance(node, Expression):
        return node.get_name()
    return str(node)


def _slot_ref(expr):
    if not isinstance(expr, Expression) or expr.get_type() != AtomType.Fn:
        return None
    name = expr.get_name().rsplit("::", 1)[-1]
    args = tuple(expr)
    if name not in {"bis", "mink", "cof", "coad"} or len(args) != 2:
        return None
    label = args[1]
    if not isinstance(label, Expression) or label.get_type() != AtomType.Var:
        return None
    return {
        "family": f"{name}({args[0].to_canonical_string()})",
        "prefix": name,
        "label": label,
        "label_key": label.to_canonical_string(),
    }


def _metric_factor_info(factor):
    if not isinstance(factor, Expression) or factor.get_type() != AtomType.Fn:
        return None
    if factor.get_name() != "spenso::g":
        return None
    args = tuple(factor)
    if len(args) != 2:
        return None
    left = _slot_ref(args[0])
    right = _slot_ref(args[1])
    if left is None or right is None:
        return None
    if left["family"] != right["family"]:
        return None
    return left, right


def _gamma5_factor_info(factor):
    if not isinstance(factor, Expression) or factor.get_type() != AtomType.Fn:
        return None
    if factor.get_name() != "spenso::gamma5":
        return None
    args = tuple(factor)
    if len(args) != 2:
        return None
    left = _slot_ref(args[0])
    right = _slot_ref(args[1])
    if left is None or right is None:
        return None
    return left, right


def _projector_factor_info(factor):
    if not isinstance(factor, Expression) or factor.get_type() != AtomType.Fn:
        return None
    name = factor.get_name().rsplit("::", 1)[-1]
    if name not in _PLAIN_SPINOR_HEADS:
        return None
    args = tuple(factor)
    if len(args) != 2:
        return None
    return {
        "kind": name,
        "left": args[0],
        "right": args[1],
    }


def _spinor_basis_factor_info(factor):
    projector = _projector_factor_info(factor)
    if projector is not None:
        return projector

    metric = _metric_factor_info(factor)
    if metric is not None and metric[0]["prefix"] == "bis":
        return {
            "kind": "g",
            "left": metric[0]["label"],
            "right": metric[1]["label"],
        }

    gamma5 = _gamma5_factor_info(factor)
    if gamma5 is not None:
        return {
            "kind": "gamma5",
            "left": gamma5[0]["label"],
            "right": gamma5[1]["label"],
        }
    return None


def _normalize_scalar(expr):
    result = expr.expand() if hasattr(expr, "expand") else expr
    if isinstance(result, Expression):
        result = simplify_invariants(result, run_color=False)
        if hasattr(result, "expand"):
            result = result.expand()
    return result


def _expr_equal(left, right) -> bool:
    return _is_zero(_normalize_scalar(left - right))


def _collapse_spinor_basis_sums(expr):
    total = _num(0)
    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for term in _terms(expr):
        factors = tuple(_factors(term))
        spinor_factors = [
            (position, _spinor_basis_factor_info(factor))
            for position, factor in enumerate(factors)
        ]
        spinor_factors = [
            (position, info)
            for position, info in spinor_factors
            if info is not None
        ]
        if len(spinor_factors) != 1:
            total += term
            continue

        position, info = spinor_factors[0]
        scalar = _product(
            factor
            for factor_position, factor in enumerate(factors)
            if factor_position != position
        )
        key = (_key(info["left"]), _key(info["right"]))
        group = grouped.setdefault(
            key,
            {
                "left": info["left"],
                "right": info["right"],
                "g": _num(0),
                "gamma5": _num(0),
                "PL": _num(0),
                "PR": _num(0),
            },
        )
        group[info["kind"]] = _normalize_scalar(group[info["kind"]] + scalar)

    for key, group in tuple(grouped.items()):
        reverse_key = (key[1], key[0])
        if reverse_key == key:
            continue
        reverse = grouped.get(reverse_key)
        if reverse is None:
            continue
        if _is_zero(group["g"]):
            continue
        if any(not _is_zero(group[kind]) for kind in ("gamma5", "PL", "PR")):
            continue
        if not any(
            not _is_zero(reverse[kind]) for kind in ("gamma5", "PL", "PR")
        ):
            continue
        reverse["g"] = _normalize_scalar(reverse["g"] + group["g"])
        group["g"] = _num(0)

    for group in grouped.values():
        left = group["left"]
        right = group["right"]
        coeff_pr = _normalize_scalar(
            group["PR"] + group["g"] + group["gamma5"]
        )
        coeff_pl = _normalize_scalar(
            group["PL"] + group["g"] - group["gamma5"]
        )

        if _is_zero(coeff_pr) and _is_zero(coeff_pl):
            continue
        if _expr_equal(coeff_pr, coeff_pl):
            total += _normalize_scalar(coeff_pr * spinor_metric(left, right))
            continue
        if _expr_equal(coeff_pr, -coeff_pl):
            total += _normalize_scalar(coeff_pr * gamma5_matrix(left, right))
            continue
        if not _is_zero(coeff_pr):
            total += _normalize_scalar(coeff_pr * projector_right(left, right))
        if not _is_zero(coeff_pl):
            total += _normalize_scalar(coeff_pl * projector_left(left, right))

    return total.expand() if hasattr(total, "expand") else total


def _field_label_usage(term: InteractionTerm) -> dict[str, int]:
    usage: dict[str, int] = defaultdict(int)
    for occurrence in term.fields:
        for label in occurrence.slot_labels.values:
            if label is None:
                continue
            if isinstance(label, Expression) and label.get_type() == AtomType.Num:
                continue
            usage[_key(label)] += 1
    for action in term.derivatives:
        label = action.lorentz_index
        if isinstance(label, Expression) and label.get_type() == AtomType.Num:
            continue
        if label is not None:
            usage[_key(label)] += 1
    return dict(usage)


def _replace_label_in_occurrence(occurrence: FieldOccurrence, old_key: str, new_label):
    slot_labels = occurrence.slot_labels
    changed = False
    for slot, label in enumerate(slot_labels.values):
        if label is None or _key(label) != old_key:
            continue
        slot_labels = slot_labels.replace(slot, new_label)
        changed = True
    return occurrence.with_slot_labels(slot_labels) if changed else occurrence


def _replace_label_in_derivatives(
    derivatives: Sequence[DerivativeAction],
    old_key: str,
    new_label,
) -> tuple[DerivativeAction, ...]:
    updated: list[DerivativeAction] = []
    for action in derivatives:
        lorentz_index = action.lorentz_index
        if lorentz_index is not None and _key(lorentz_index) == old_key:
            updated.append(
                DerivativeAction(
                    target=action.target,
                    lorentz_index=new_label,
                )
            )
            continue
        updated.append(action)
    return tuple(updated)


def _coupling_text(expr) -> str:
    if hasattr(expr, "to_canonical_string"):
        return expr.to_canonical_string()
    return str(expr)


def _has_spinor_postprocess_features(text: str) -> bool:
    return any(
        token in text
        for token in ("gamma5", "bis(4", "PL(", "PR(", "::gamma(")
    )


def _has_metric_postprocess_features(text: str) -> bool:
    return "::g(" in text


def _normalize_coupling(expr):
    result = expr.expand() if hasattr(expr, "expand") else expr
    previous = None
    for _ in range(12):
        current = _coupling_text(result)
        if current == previous:
            break
        previous = current
        has_spinor = _has_spinor_postprocess_features(current)
        has_metric = _has_metric_postprocess_features(current)

        if has_spinor and "gamma5" in current:
            result = _simplify_gamma5_square_chains(result)
            if hasattr(result, "expand"):
                result = result.expand()
            current = _coupling_text(result)
            has_spinor = _has_spinor_postprocess_features(current)
            has_metric = _has_metric_postprocess_features(current)

        if isinstance(result, Expression) and (has_metric or has_spinor):
            result = simplify_invariants(
                result,
                run_gamma=has_spinor,
                run_color=False,
            )
            if hasattr(result, "expand"):
                result = result.expand()

        if has_spinor and isinstance(result, Expression):
            result = _collapse_spinor_basis_sums(result)
            if hasattr(result, "expand"):
                result = result.expand()

        if not (has_metric or has_spinor):
            break
    return result


def _simplify_gamma5_square_chains(expr):
    total = _num(0)
    for term in _terms(expr):
        factors = list(_factors(term))
        changed = True
        while changed:
            changed = False
            gamma5_factors = [
                (position, _gamma5_factor_info(factor))
                for position, factor in enumerate(factors)
            ]
            gamma5_factors = [
                (position, info)
                for position, info in gamma5_factors
                if info is not None
            ]
            for left_pos, left_info in gamma5_factors:
                for right_pos, right_info in gamma5_factors:
                    if right_pos <= left_pos:
                        continue
                    left_labels = {
                        left_info[0]["label_key"]: left_info[0]["label"],
                        left_info[1]["label_key"]: left_info[1]["label"],
                    }
                    right_labels = {
                        right_info[0]["label_key"]: right_info[0]["label"],
                        right_info[1]["label_key"]: right_info[1]["label"],
                    }
                    shared = tuple(
                        key
                        for key in left_labels
                        if key in right_labels
                    )
                    if len(shared) != 1:
                        continue
                    shared_key = shared[0]
                    left_remaining = next(
                        value for key, value in left_labels.items() if key != shared_key
                    )
                    right_remaining = next(
                        value for key, value in right_labels.items() if key != shared_key
                    )
                    factors = [
                        factor
                        for position, factor in enumerate(factors)
                        if position not in {left_pos, right_pos}
                    ]
                    factors.append(spinor_metric(left_remaining, right_remaining))
                    changed = True
                    break
                if changed:
                    break
        total += _product(factors)
    return total.expand() if hasattr(total, "expand") else total


def _contract_metric_identities(term: InteractionTerm) -> InteractionTerm:
    if not isinstance(term.coupling, Expression):
        return term

    coupling = term.coupling.expand()
    if isinstance(coupling, Expression) and coupling.get_type() == AtomType.Add:
        return replace(term, coupling=coupling)

    factors = list(_factors(coupling))
    fields = tuple(term.fields)
    derivatives = tuple(term.derivatives)

    changed = True
    while changed:
        changed = False
        field_usage = _field_label_usage(replace(term, fields=fields, derivatives=derivatives))
        for metric_position, factor in enumerate(factors):
            info = _metric_factor_info(factor)
            if info is None:
                continue
            left, right = info
            left_key = left["label_key"]
            right_key = right["label_key"]
            left_in_fields = field_usage.get(left_key, 0) > 0
            right_in_fields = field_usage.get(right_key, 0) > 0

            if left_in_fields and right_in_fields:
                continue

            left_other = sum(
                _count_symbol_occurrences(other_factor, left_key)
                for position, other_factor in enumerate(factors)
                if position != metric_position
            )
            right_other = sum(
                _count_symbol_occurrences(other_factor, right_key)
                for position, other_factor in enumerate(factors)
                if position != metric_position
            )
            if left_other == 0 and right_other == 0:
                continue

            if left_in_fields and not right_in_fields:
                winner, loser = left["label"], right["label"]
                winner_key, loser_key = left_key, right_key
            elif right_in_fields and not left_in_fields:
                winner, loser = right["label"], left["label"]
                winner_key, loser_key = right_key, left_key
            elif left_other > 0 and right_other == 0:
                winner, loser = left["label"], right["label"]
                winner_key, loser_key = left_key, right_key
            elif right_other > 0 and left_other == 0:
                winner, loser = right["label"], left["label"]
                winner_key, loser_key = right_key, left_key
            else:
                if left_key <= right_key:
                    winner, loser = left["label"], right["label"]
                    winner_key, loser_key = left_key, right_key
                else:
                    winner, loser = right["label"], left["label"]
                    winner_key, loser_key = right_key, left_key

            updated_factors = []
            for position, other_factor in enumerate(factors):
                if position == metric_position:
                    continue
                updated_factors.append(_replace_symbol(other_factor, loser, winner))
            factors = updated_factors
            fields = tuple(
                _replace_label_in_occurrence(occurrence, loser_key, winner)
                for occurrence in fields
            )
            derivatives = _replace_label_in_derivatives(
                derivatives,
                loser_key,
                winner,
            )
            changed = True
            break

    rebuilt = _product(factors)
    rebuilt = _normalize_coupling(rebuilt)
    return replace(
        term,
        coupling=rebuilt,
        fields=fields,
        derivatives=derivatives,
    )


def _occurrence_sort_key(occurrence: FieldOccurrence):
    label_key = tuple(
        _key(label) if label is not None else ""
        for label in occurrence.slot_labels.values
    )
    return (
        occurrence.field.name,
        bool(occurrence.conjugated and not occurrence.field.self_conjugate),
        label_key,
    )


def _canonicalize_field_order(term: InteractionTerm) -> InteractionTerm:
    fermion_positions = tuple(
        position
        for position, occurrence in enumerate(term.fields)
        if occurrence.field.statistics == "fermion"
    )
    if not fermion_positions:
        return term
    boson_positions = tuple(
        position
        for position, occurrence in enumerate(term.fields)
        if occurrence.field.statistics != "fermion"
    )
    order = fermion_positions + boson_positions
    if order == tuple(range(len(term.fields))):
        return term

    index_map = {
        old_position: new_position
        for new_position, old_position in enumerate(order)
    }
    fields = tuple(term.fields[old_position] for old_position in order)
    derivatives = tuple(
        DerivativeAction(
            target=index_map[action.target],
            lorentz_index=action.lorentz_index,
        )
        for action in term.derivatives
    )
    derivatives = tuple(
        sorted(
            derivatives,
            key=lambda action: (action.target, _key(action.lorentz_index)),
        )
    )
    bilinears = tuple(
        DiracBilinear(
            psibar=SlotRef(
                occurrence=index_map[bilinear.psibar.occurrence],
                slot=bilinear.psibar.slot,
            ),
            psi=SlotRef(
                occurrence=index_map[bilinear.psi.occurrence],
                slot=bilinear.psi.slot,
            ),
        )
        for bilinear in term.dirac_bilinears
    )
    bilinears = tuple(
        sorted(
            bilinears,
            key=lambda bilinear: (
                bilinear.psibar.occurrence,
                bilinear.psi.occurrence,
            ),
        )
    )
    return replace(
        term,
        coupling=term.coupling,
        fields=fields,
        derivatives=derivatives,
        closed_dirac_bilinears=tuple(
            bilinear.as_legacy() for bilinear in bilinears
        ),
        dirac_bilinears=bilinears,
    )


def _canonical_label(prefix: str, counters: dict[str, int]) -> Expression:
    counters[prefix] += 1
    return S(f"{prefix}_canon_{counters[prefix]}")


def _coupling_slot_labels(expr) -> list[tuple[str, Expression]]:
    labels: list[tuple[str, Expression]] = []
    for node in _iter_expression_nodes(expr):
        slot = _slot_ref(node)
        if slot is not None:
            labels.append((slot["prefix"], slot["label"]))
            continue
        if node.get_type() != AtomType.Fn:
            continue
        name = node.get_name().rsplit("::", 1)[-1]
        if name not in _PLAIN_SPINOR_HEADS:
            continue
        for argument in tuple(node):
            if not isinstance(argument, Expression) or argument.get_type() != AtomType.Var:
                continue
            labels.append((SPINOR_INDEX.prefix or SPINOR_INDEX.kind, argument))
    return labels


def _parameter_argument_labels(expr, parameters: Sequence[object]) -> list[tuple[str, Expression]]:
    parameter_lookup = {
        parameter.name: parameter
        for parameter in parameters
        if isinstance(parameter, Parameter) and parameter.indices
    }
    labels: list[tuple[str, Expression]] = []
    for node in _iter_expression_nodes(expr):
        if node.get_type() != AtomType.Fn:
            continue
        parameter = parameter_lookup.get(node.get_name().rsplit("::", 1)[-1])
        if parameter is None:
            continue
        arguments = tuple(node)
        if len(arguments) != len(parameter.indices):
            continue
        for argument, index in zip(arguments, parameter.indices):
            if not isinstance(argument, Expression) or argument.get_type() != AtomType.Var:
                continue
            labels.append((index.prefix or index.kind, argument))
    return labels


def _standardize_term_labels(
    term: InteractionTerm,
    *,
    parameters: Sequence[object],
) -> InteractionTerm:
    mapping: dict[str, tuple[object, Expression]] = {}
    counters: dict[str, int] = defaultdict(int)
    usage_counts: dict[str, int] = defaultdict(int)

    for occurrence in term.fields:
        for label in occurrence.slot_labels.values:
            if label is None:
                continue
            if isinstance(label, Expression) and label.get_type() == AtomType.Num:
                continue
            usage_counts[_key(label)] += 1

    for action in term.derivatives:
        label = action.lorentz_index
        if label is None:
            continue
        if isinstance(label, Expression) and label.get_type() == AtomType.Num:
            continue
        usage_counts[_key(label)] += 1

    if isinstance(term.coupling, Expression):
        for _prefix, label in _parameter_argument_labels(term.coupling, parameters):
            usage_counts[_key(label)] += 1
        for _prefix, label in _coupling_slot_labels(term.coupling):
            usage_counts[_key(label)] += 1

    def register(prefix: str, label) -> None:
        if label is None:
            return
        if isinstance(label, Expression) and label.get_type() == AtomType.Num:
            return
        label_key = _key(label)
        if usage_counts.get(label_key, 0) <= 1:
            return
        mapping.setdefault(
            label_key,
            (label, _canonical_label(prefix, counters)),
        )

    for occurrence in term.fields:
        for slot, index in enumerate(occurrence.field.indices):
            register(index.prefix or index.kind, occurrence.slot_labels.get(slot))

    for action in term.derivatives:
        register(LORENTZ_INDEX.prefix or LORENTZ_INDEX.kind, action.lorentz_index)

    if isinstance(term.coupling, Expression):
        for prefix, label in _parameter_argument_labels(term.coupling, parameters):
            register(prefix, label)
        for prefix, label in _coupling_slot_labels(term.coupling):
            register(prefix, label)

    if not mapping:
        return term

    fields = tuple(term.fields)
    derivatives = tuple(term.derivatives)
    coupling = term.coupling
    for old_key, (old_label, new_label) in mapping.items():
        coupling = _replace_symbol(coupling, old_label, new_label)
        fields = tuple(
            _replace_label_in_occurrence(occurrence, old_key, new_label)
            for occurrence in fields
        )
        derivatives = _replace_label_in_derivatives(
            derivatives,
            old_key,
            new_label,
        )
    derivatives = tuple(
        sorted(
            derivatives,
            key=lambda action: (action.target, _key(action.lorentz_index)),
        )
    )
    return replace(
        term,
        coupling=coupling,
        fields=fields,
        derivatives=derivatives,
    )


def _canonical_term_key(term: InteractionTerm):
    return (
        tuple(
            (
                occurrence.field.name,
                bool(occurrence.conjugated),
                tuple(
                    _key(label) if label is not None else None
                    for label in occurrence.slot_labels.values
                ),
            )
            for occurrence in term.fields
        ),
        tuple(
            (action.target, _key(action.lorentz_index))
            for action in term.derivatives
        ),
        tuple(
            (
                bilinear.psibar.occurrence,
                bilinear.psibar.slot,
                bilinear.psi.occurrence,
                bilinear.psi.slot,
            )
            for bilinear in term.dirac_bilinears
        ),
        term.label,
        term.sector,
    )


def _canonicalize_monomial_term(
    term: InteractionTerm,
    *,
    parameters: Sequence[object],
) -> tuple[InteractionTerm, ...]:
    current = term
    if _has_metric_postprocess_features(_coupling_text(current.coupling)):
        current = _contract_metric_identities(current)
        coupling = current.coupling
    else:
        coupling = _normalize_coupling(current.coupling)
    if _is_zero(coupling):
        return ()
    current = replace(current, coupling=coupling)
    current = _canonicalize_field_order(current)
    current = _standardize_term_labels(current, parameters=parameters)
    current = replace(current, coupling=_normalize_coupling(current.coupling))
    if _is_zero(current.coupling):
        return ()
    return (current,)


def canonicalize_transformed_terms(
    terms: Sequence[InteractionTerm],
    *,
    parameters: Sequence[object] = (),
) -> tuple[InteractionTerm, ...]:
    expanded_terms: list[InteractionTerm] = []
    for term in terms:
        coupling = term.coupling.expand() if hasattr(term.coupling, "expand") else term.coupling
        for coupling_term in _terms(coupling):
            expanded_terms.extend(
                _canonicalize_monomial_term(
                    replace(term, coupling=coupling_term),
                    parameters=parameters,
                )
            )

    merged: dict[tuple, InteractionTerm] = {}
    for term in expanded_terms:
        key = _canonical_term_key(term)
        prior = merged.get(key)
        if prior is None:
            merged[key] = term
            continue
        coupling = _normalize_coupling(prior.coupling + term.coupling)
        if _is_zero(coupling):
            del merged[key]
            continue
        merged[key] = replace(prior, coupling=coupling)

    normalized: list[InteractionTerm] = []
    for term in merged.values():
        coupling = _normalize_coupling(term.coupling)
        if _is_zero(coupling):
            continue
        normalized.append(replace(term, coupling=coupling))
    return tuple(normalized)


def canonical_compiled_expression(
    compiled: CompiledLagrangian,
    *,
    field_heads: Iterable[object] = (),
    run_color: bool = False,
):
    expression = _num(0)
    for term in compiled.terms:
        expression += interaction_term_to_symbolica(term)
    return canonize_full(
        expression,
        infer_indices=True,
        field_heads=tuple(field_heads),
        run_color=run_color,
    )


def equivalent_canonical_compiled(
    left,
    right,
    *,
    field_heads: Iterable[object] = (),
    run_color: bool = False,
) -> bool:
    left_expr = (
        canonical_compiled_expression(left, field_heads=field_heads, run_color=run_color)
        if isinstance(left, CompiledLagrangian)
        else canonize_full(left, infer_indices=True, field_heads=tuple(field_heads), run_color=run_color)
    )
    right_expr = (
        canonical_compiled_expression(right, field_heads=field_heads, run_color=run_color)
        if isinstance(right, CompiledLagrangian)
        else canonize_full(right, infer_indices=True, field_heads=tuple(field_heads), run_color=run_color)
    )
    return _key(left_expr) == _key(right_expr)


def find_source_basis_occurrences(
    compiled: CompiledLagrangian,
    *,
    source_fields: Sequence[object],
) -> tuple[tuple[int, int, FieldOccurrence], ...]:
    source_set = frozenset(source_fields)
    hits = []
    for term_index, term in enumerate(compiled.terms):
        for slot, occurrence in enumerate(term.fields):
            if occurrence.field in source_set:
                hits.append((term_index, slot, occurrence))
    return tuple(hits)


def validate_compiled_index_multiplicities(
    compiled: CompiledLagrangian,
) -> tuple[str, ...]:
    issues: list[str] = []
    for term_index, term in enumerate(compiled.terms):
        field_labels = {
            _key(label)
            for occurrence in term.fields
            for label in occurrence.slot_labels.values
            if label is not None
            and not (isinstance(label, Expression) and label.get_type() == AtomType.Num)
        }
        derivative_labels = {
            _key(action.lorentz_index)
            for action in term.derivatives
            if action.lorentz_index is not None
            and not (
                isinstance(action.lorentz_index, Expression)
                and action.lorentz_index.get_type() == AtomType.Num
            )
        }
        connected_labels = field_labels | derivative_labels
        if not isinstance(term.coupling, Expression):
            continue
        for prefix, label in _parameter_argument_labels(term.coupling, compiled.parameters):
            del prefix
            label_key = _key(label)
            if label_key in connected_labels:
                continue
            if _count_symbol_occurrences(term.coupling, label_key) == 1:
                issues.append(
                    f"term {term_index} carries dangling parameter label {label_key!r} "
                    "that is not connected to any field slot."
                )
        for prefix, label in _coupling_slot_labels(term.coupling):
            del prefix
            label_key = _key(label)
            if label_key in connected_labels:
                continue
            if _count_symbol_occurrences(term.coupling, label_key) == 1:
                issues.append(
                    f"term {term_index} carries dangling tensor label {label_key!r} "
                    "that is not connected to any field slot."
                )
    return tuple(dict.fromkeys(issues))


def _conjugate_occurrence(occurrence: FieldOccurrence) -> FieldOccurrence:
    conjugated = False
    if not occurrence.field.self_conjugate:
        conjugated = not occurrence.conjugated
    return occurrence.field.occurrence(
        conjugated=conjugated,
        labels=occurrence.labels,
    )


def _conjugate_coefficient(
    value,
    real_symbols: Sequence[object],
    *,
    parameters: Sequence[object] = (),
):
    if not hasattr(value, "conj"):
        return value
    result = value.conj()
    parameter_lookup = {
        parameter.name: parameter
        for parameter in parameters
        if isinstance(parameter, Parameter)
    }

    def partner_for(parameter: Parameter):
        if parameter.unitary_partner:
            partner = parameter_lookup.get(parameter.unitary_partner)
            if partner is not None:
                return partner
        if parameter.name.endswith("Dag"):
            return parameter_lookup.get(parameter.name[:-3])
        return parameter_lookup.get(f"{parameter.name}Dag")

    for node in tuple(_iter_expression_nodes(value)):
        if node.get_type() != AtomType.Fn:
            continue
        parameter = parameter_lookup.get(node.get_name().rsplit("::", 1)[-1])
        if parameter is None:
            continue
        partner = partner_for(parameter)
        if partner is None:
            continue
        args = tuple(node)
        partner_args = args[::-1] if len(args) == 2 else args
        result = result.replace(node.conj(), partner(*partner_args))

    targets = (
        symbol.symbol if isinstance(symbol, Parameter) else symbol
        for symbol in real_symbols
    )
    for target in sorted(targets, key=lambda item: len(_key(item)), reverse=True):
        if hasattr(target, "conj") and hasattr(result, "replace"):
            result = result.replace(target.conj(), target)
    return result


def _hermitian_conjugate_spinor_operators(expr):
    if not isinstance(expr, Expression):
        return expr

    result = expr
    for node in reversed(tuple(_iter_expression_nodes(expr))):
        replacement = None

        projector = _projector_factor_info(node)
        if projector is not None:
            if projector["kind"] == "PR":
                replacement = projector_left(projector["right"], projector["left"])
            else:
                replacement = projector_right(projector["right"], projector["left"])

        metric = _metric_factor_info(node)
        if replacement is None and metric is not None and metric[0]["prefix"] == "bis":
            replacement = spinor_metric(metric[1]["label"], metric[0]["label"])

        gamma5 = _gamma5_factor_info(node)
        if replacement is None and gamma5 is not None:
            replacement = gamma5_matrix(gamma5[1]["label"], gamma5[0]["label"])

        if (
            replacement is None
            and isinstance(node, Expression)
            and node.get_type() == AtomType.Fn
            and node.get_name() == "spenso::gamma"
        ):
            args = tuple(node)
            if len(args) == 3:
                left = _slot_ref(args[0])
                right = _slot_ref(args[1])
                lorentz = _slot_ref(args[2])
                if (
                    left is not None
                    and right is not None
                    and left["prefix"] == "bis"
                    and right["prefix"] == "bis"
                    and lorentz is not None
                    and lorentz["prefix"] == "mink"
                ):
                    replacement = gamma_matrix(
                        right["label"],
                        left["label"],
                        lorentz["label"],
                    )

        if replacement is not None:
            result = result.replace(node, replacement)
    return result


def _hermitian_conjugate_term(
    term: InteractionTerm,
    *,
    real_symbols: Sequence[object],
    parameters: Sequence[object] = (),
) -> InteractionTerm:
    arity = len(term.fields)
    fields = tuple(_conjugate_occurrence(occurrence) for occurrence in reversed(term.fields))
    derivatives = tuple(
        DerivativeAction(
            target=arity - 1 - action.target,
            lorentz_index=action.lorentz_index,
        )
        for action in term.derivatives
    )
    derivatives = tuple(
        sorted(
            derivatives,
            key=lambda action: (action.target, _key(action.lorentz_index)),
        )
    )
    bilinears = tuple(
        DiracBilinear(
            psibar=SlotRef(
                occurrence=arity - 1 - bilinear.psi.occurrence,
                slot=bilinear.psi.slot,
            ),
            psi=SlotRef(
                occurrence=arity - 1 - bilinear.psibar.occurrence,
                slot=bilinear.psibar.slot,
            ),
        )
        for bilinear in term.dirac_bilinears
    )
    bilinears = tuple(
        sorted(
            bilinears,
            key=lambda bilinear: (
                bilinear.psibar.occurrence,
                bilinear.psi.occurrence,
            ),
        )
    )
    return replace(
        term,
        coupling=_hermitian_conjugate_spinor_operators(
            _conjugate_coefficient(
                term.coupling,
                real_symbols,
                parameters=parameters,
            )
        ),
        fields=fields,
        derivatives=derivatives,
        closed_dirac_bilinears=tuple(
            bilinear.as_legacy() for bilinear in bilinears
        ),
        dirac_bilinears=bilinears,
    )


def compiled_is_hermitian(
    compiled: CompiledLagrangian,
    *,
    real_symbols: Sequence[object] = (),
    field_heads: Iterable[object] = (),
    run_color: bool = False,
) -> bool:
    conjugated = CompiledLagrangian(
        terms=tuple(
            _hermitian_conjugate_term(
                term,
                real_symbols=real_symbols,
                parameters=compiled.parameters,
            )
            for term in compiled.terms
        ),
        parameters=compiled.parameters,
    )
    return equivalent_canonical_compiled(
        compiled,
        conjugated,
        field_heads=field_heads,
        run_color=run_color,
    )


__all__ = (
    "canonical_compiled_expression",
    "canonicalize_transformed_terms",
    "compiled_is_hermitian",
    "equivalent_canonical_compiled",
    "find_source_basis_occurrences",
    "validate_compiled_index_multiplicities",
)
