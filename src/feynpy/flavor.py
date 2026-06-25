"""Flavor-class expansion helpers for compiled interaction terms."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Optional

from symbolica import AtomType, Expression, S

from .interactions import InteractionTerm
from .metadata import IndexType, Parameter


def _canonical_value_key(value):
    if isinstance(value, tuple):
        return tuple(_canonical_value_key(item) for item in value)
    if hasattr(value, "to_canonical_string"):
        return value.to_canonical_string()
    return value


def _label_name(label) -> str:
    if hasattr(label, "to_canonical_string"):
        return str(label)
    return str(label)


def _atom_symbol_name(head: str) -> str:
    return head.rsplit("::", 1)[-1]


def _expression_num(value: int):
    return Expression.num(value)


def _atom_type_name(expr) -> str:
    if hasattr(expr, "get_type"):
        try:
            return str(expr.get_type())
        except Exception:
            return ""
    return ""


def _term_factors(expr):
    if _atom_type_name(expr) == "AtomType.Mul":
        return tuple(expr)
    return (expr,)


def _term_product(factors):
    result = Expression.num(1)
    for factor in factors:
        result *= factor
    return result


def _parameter_name_map(parameters: tuple[Parameter, ...]) -> dict[str, Parameter]:
    return {parameter.name: parameter for parameter in parameters}


def _parameter_symbol_map(parameters: tuple[Parameter, ...]) -> dict[str, Parameter]:
    mapping = {}
    for parameter in parameters:
        symbol = parameter.symbol
        if hasattr(symbol, "to_canonical_string"):
            mapping[symbol.to_canonical_string()] = parameter
        mapping[str(symbol)] = parameter
    return mapping


def _resolve_parameter_reference(reference, parameters: tuple[Parameter, ...]) -> Optional[Parameter]:
    if isinstance(reference, Parameter):
        return reference

    by_name = _parameter_name_map(parameters)
    by_symbol = _parameter_symbol_map(parameters)

    if isinstance(reference, str):
        return by_name.get(reference) or by_symbol.get(reference)
    if hasattr(reference, "to_canonical_string"):
        canonical = reference.to_canonical_string()
        return by_symbol.get(canonical) or by_name.get(canonical)
    return None


def _unitary_partner_lookup(parameters: tuple[Parameter, ...]) -> dict[str, Parameter]:
    lookup: dict[str, Parameter] = {}
    for parameter in parameters:
        if parameter.unitary_partner is None:
            continue
        partner = _resolve_parameter_reference(parameter.unitary_partner, parameters)
        if partner is None:
            continue
        lookup[parameter.name] = partner
    return lookup


def _factor_parameter_call(factor, parameters: tuple[Parameter, ...]):
    if not hasattr(factor, "to_atom_tree"):
        return None
    try:
        node = factor.to_atom_tree()
    except Exception:
        return None
    if str(node.atom_type) != "AtomType.Fn":
        return None

    head = _atom_symbol_name(node.head)
    parameter = _parameter_name_map(parameters).get(head)
    if parameter is None or len(node.tail) != len(parameter.indices):
        return None

    labels = []
    for child in node.tail:
        if str(child.atom_type) != "AtomType.Var":
            return None
        labels.append(S(_atom_symbol_name(child.head)))
    return parameter, tuple(labels)


def _unitary_metric(parameter: Parameter, slot: int, left_label, right_label):
    index = parameter.indices[slot]
    return index.representation.g(left_label, right_label).to_expression()


def _simplify_unitary_parameter_products(expr, parameters: tuple[Parameter, ...]):
    if not parameters:
        return expr

    partner_lookup = _unitary_partner_lookup(parameters)
    if not partner_lookup:
        return expr

    terms = (
        tuple(expr)
        if isinstance(expr, Expression) and expr.get_type() == AtomType.Add
        else (expr,)
    )
    reduced_terms = []

    for term in terms:
        factors = list(_term_factors(term))
        changed = True
        while changed:
            changed = False
            for left_pos, left_factor in enumerate(factors):
                left_call = _factor_parameter_call(left_factor, parameters)
                if left_call is None:
                    continue
                left_parameter, left_labels = left_call
                partner = partner_lookup.get(left_parameter.name)
                if partner is None or len(left_parameter.indices) != 2:
                    continue
                for right_pos in range(left_pos + 1, len(factors)):
                    right_factor = factors[right_pos]
                    right_call = _factor_parameter_call(right_factor, parameters)
                    if right_call is None:
                        continue
                    right_parameter, right_labels = right_call
                    if right_parameter is not partner:
                        continue

                    replacement = None
                    if left_labels[0] == right_labels[1]:
                        replacement = _unitary_metric(
                            left_parameter,
                            1,
                            left_labels[1],
                            right_labels[0],
                        )
                    elif left_labels[1] == right_labels[0]:
                        replacement = _unitary_metric(
                            left_parameter,
                            0,
                            left_labels[0],
                            right_labels[1],
                        )
                    if replacement is None:
                        continue

                    factors = [
                        factor
                        for position, factor in enumerate(factors)
                        if position not in {left_pos, right_pos}
                    ]
                    factors.append(replacement)
                    changed = True
                    break
                if changed:
                    break

        reduced_terms.append(_term_product(factors))

    result = sum(reduced_terms, Expression.num(0))
    if hasattr(result, "expand"):
        result = result.expand()
    return result


def _simplify_expression(expr, parameters: tuple[Parameter, ...] = ()):
    expr = _simplify_unitary_parameter_products(expr, parameters)
    if hasattr(expr, "expand"):
        return expr.expand()
    return expr


def _is_zero_expression(expr) -> bool:
    if expr == 0:
        return True
    if hasattr(expr, "expand"):
        return expr.expand().to_canonical_string() == "0"
    return False


def _parameter_head_map(parameters: tuple[Parameter, ...]) -> dict[str, Parameter]:
    return {
        f"python::{str(parameter.symbol)}": parameter
        for parameter in parameters
    }


def _index_is_selected(
    index: IndexType,
    selected_indices: Optional[frozenset[IndexType]],
) -> bool:
    return selected_indices is None or index in selected_indices


def _register_flavor_label(label_entries, label_counts, label, index):
    label_name = _label_name(label)
    prior = label_entries.get(label_name)
    if prior is None:
        label_entries[label_name] = (label, index)
    else:
        _prior_label, prior_index = prior
        if prior_index != index:
            raise ValueError(
                f"Flavor label {label_name!r} is shared across incompatible flavor "
                f"indices {prior_index.name!r} and {index.name!r}."
            )
    label_counts[label_name] += 1


def _collect_parameter_flavor_labels(
    expr,
    parameters,
    label_entries,
    label_counts,
    selected_indices,
):
    if not parameters or not hasattr(expr, "to_atom_tree"):
        return

    heads = _parameter_head_map(parameters)
    single_index_parameter_labels: list[tuple[Parameter, str]] = []

    def visit(node):
        atom_type = str(node.atom_type)
        if atom_type == "AtomType.Fn":
            parameter = heads.get(node.head)
            if parameter is not None and len(node.tail) == len(parameter.indices):
                for index, arg in zip(parameter.indices, node.tail):
                    if not index.is_flavor or not _index_is_selected(
                        index,
                        selected_indices,
                    ):
                        continue
                    if str(arg.atom_type) != "AtomType.Var":
                        continue
                    label = S(_atom_symbol_name(arg.head))
                    _register_flavor_label(label_entries, label_counts, label, index)
                    if len(parameter.indices) == 1:
                        single_index_parameter_labels.append((parameter, _label_name(label)))
        for child in node.tail:
            visit(child)

    visit(expr.to_atom_tree())

    for parameter, label_name in single_index_parameter_labels:
        if parameter.permits_label_summation():
            continue
        if label_counts[label_name] > 2:
            raise ValueError(
                f"Parameter {parameter.name!r} uses flavor label {label_name!r} more than "
                "twice in one term. Mark the parameter with allow_summation=True to use "
                "this diagonal shorthand."
            )


def _collect_term_flavor_labels(
    term: InteractionTerm,
    parameters: tuple[Parameter, ...],
    selected_indices: Optional[frozenset[IndexType]],
):
    label_entries: dict[str, tuple[object, object]] = {}
    label_counts: Counter = Counter()

    for occurrence in term.fields:
        slot_labels = occurrence.slot_labels
        for slot, index in enumerate(occurrence.field.indices):
            if not index.is_flavor or not _index_is_selected(
                index,
                selected_indices,
            ):
                continue
            label = slot_labels.get(slot)
            if label is None:
                raise ValueError(
                    f"Cannot flavor-expand field {occurrence.field.name!r}: flavor index "
                    f"{index.name!r} has no explicit label in this interaction term."
                )
            _register_flavor_label(label_entries, label_counts, label, index)

    _collect_parameter_flavor_labels(
        term.coupling,
        parameters,
        label_entries,
        label_counts,
        selected_indices,
    )

    ordered = []
    for label_name in sorted(label_entries):
        label, index = label_entries[label_name]
        if index.dimension is None:
            raise ValueError(
                f"Flavor index {index.name!r} has no declared dimension for expansion."
            )
        ordered.append((label_name, label, index))
    return tuple(ordered)


def _apply_parameter_components(expr, parameters: tuple[Parameter, ...]):
    if not hasattr(expr, "replace"):
        return expr
    result = expr
    for parameter in parameters:
        for component_key, component_value in parameter.components.items():
            result = result.replace(parameter(*component_key), component_value)
    return result


def _reduce_concrete_flavor_metrics(expr, indices: tuple[IndexType, ...]):
    if not hasattr(expr, "replace"):
        return expr
    result = expr
    for index in indices:
        if index.dimension is None:
            continue
        for row in range(1, index.dimension + 1):
            for col in range(1, index.dimension + 1):
                result = result.replace(
                    index.representation.g(row, col).to_expression(),
                    _expression_num(1 if row == col else 0),
                )
    return result


def _expand_occurrence(occurrence, assignment_by_name, selected_indices):
    field = occurrence.field
    slot_labels = occurrence.slot_labels
    flavor_slot = field.flavor_index_slot()

    if flavor_slot is None:
        if any(
            index.is_flavor and _index_is_selected(index, selected_indices)
            for index in field.indices
        ):
            raise ValueError(
                f"Cannot flavor-expand field {field.name!r} because no class members are defined."
            )
        return occurrence

    index = field.indices[flavor_slot]
    if not _index_is_selected(index, selected_indices):
        return occurrence
    label = slot_labels.get(flavor_slot)
    if label is None:
        raise ValueError(
            f"Cannot flavor-expand field {field.name!r}: flavor index {index.name!r} "
            "has no assigned label."
        )

    if not field.class_members:
        raise ValueError(
            f"Cannot flavor-expand field {field.name!r} over index {index.name!r} "
            "because no class members are defined."
        )

    member = field.class_member_for(assignment_by_name[_label_name(label)])
    member_slot_labels = {}
    for slot, slot_label in enumerate(slot_labels.values):
        if slot == flavor_slot or slot_label is None:
            continue
        member_slot = slot if slot < flavor_slot else slot - 1
        member_slot_labels[member_slot] = slot_label

    return member.occurrence(
        conjugated=occurrence.conjugated,
        labels=member.pack_slot_labels(member_slot_labels),
    )


def _term_structure_key(term: InteractionTerm):
    return (
        tuple(
            (
                occurrence.field,
                occurrence.conjugated,
                tuple(_canonical_value_key(value) for value in occurrence.slot_labels.values),
            )
            for occurrence in term.fields
        ),
        tuple(
            (action.target, _canonical_value_key(action.lorentz_index))
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
        term.sector,
    )


def _merge_terms(
    terms: list[InteractionTerm],
    *,
    parameters: tuple[Parameter, ...] = (),
) -> tuple[InteractionTerm, ...]:
    merged: dict[tuple, InteractionTerm] = {}
    for term in terms:
        key = _term_structure_key(term)
        prior = merged.get(key)
        if prior is None:
            merged[key] = term
            continue
        coupling = _simplify_expression(
            prior.coupling + term.coupling,
            parameters=parameters,
        )
        if _is_zero_expression(coupling):
            del merged[key]
            continue
        merged[key] = replace(prior, coupling=coupling)
    return tuple(merged.values())


def simplify_parameter_identities(
    terms,
    *,
    parameters: tuple[Parameter, ...] = (),
) -> tuple[InteractionTerm, ...]:
    """Apply generic parameter identities, such as unitary matrix contractions."""

    simplified_terms: list[InteractionTerm] = []
    for term in terms:
        coupling = _simplify_expression(term.coupling, parameters=parameters)
        if _is_zero_expression(coupling):
            continue
        simplified_terms.append(replace(term, coupling=coupling))
    return _canonicalize_projector_terms(
        _merge_terms(simplified_terms, parameters=parameters),
        parameters=parameters,
    )


def expand_flavor_terms(
    terms,
    *,
    parameters: tuple[Parameter, ...] = (),
    selected_indices: Optional[tuple[IndexType, ...]] = None,
) -> tuple[InteractionTerm, ...]:
    """Expand flavor-generic fields in compiled interaction terms."""

    selected = None if selected_indices is None else frozenset(selected_indices)
    expanded_terms: list[InteractionTerm] = []

    additive_terms: list[InteractionTerm] = []
    for term in terms:
        coupling_terms = (
            tuple(term.coupling)
            if isinstance(term.coupling, Expression)
            and term.coupling.get_type() == AtomType.Add
            else (term.coupling,)
        )
        additive_terms.extend(
            replace(term, coupling=coupling)
            for coupling in coupling_terms
        )

    for term in additive_terms:
        assignments = _collect_term_flavor_labels(
            term,
            parameters,
            selected,
        )
        if not assignments:
            expanded_terms.append(term)
            continue

        assignment_ranges = [range(1, index.dimension + 1) for _, _label, index in assignments]
        assigned_indices = tuple(
            index
            for _label_name, _label, index in assignments
        )
        for values in product(*assignment_ranges):
            assignment_by_name = {
                label_name: value
                for (label_name, _label, _index), value in zip(assignments, values)
            }

            coupling = term.coupling
            for label_name, label, _index in assignments:
                if hasattr(coupling, "replace"):
                    coupling = coupling.replace(label, _expression_num(assignment_by_name[label_name]))
            coupling = _apply_parameter_components(coupling, parameters)
            coupling = _reduce_concrete_flavor_metrics(
                coupling,
                assigned_indices,
            )
            coupling = _simplify_expression(coupling, parameters=parameters)
            if _is_zero_expression(coupling):
                continue

            expanded_terms.append(
                replace(
                    term,
                    coupling=coupling,
                    fields=tuple(
                        _expand_occurrence(occurrence, assignment_by_name, selected)
                        for occurrence in term.fields
                    ),
                )
            )

    return _canonicalize_projector_terms(
        _merge_terms(expanded_terms, parameters=parameters),
        parameters=parameters,
    )


def _canonicalize_projector_terms(
    terms,
    *,
    parameters: tuple[Parameter, ...] = (),
) -> tuple[InteractionTerm, ...]:
    from .transformation_postprocess import canonicalize_transformed_terms

    direct: list[InteractionTerm] = []
    projector_terms: list[InteractionTerm] = []
    for term in terms:
        text = (
            term.coupling.to_canonical_string()
            if hasattr(term.coupling, "to_canonical_string")
            else str(term.coupling)
        )
        if "PL(" in text or "PR(" in text:
            projector_terms.append(term)
            continue
        direct.append(term)
    if not projector_terms:
        return tuple(direct)
    return tuple(direct) + canonicalize_transformed_terms(
        tuple(projector_terms),
        parameters=parameters,
    )
