"""Flavor-class expansion helpers for compiled interaction terms."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product

from symbolica import Expression, S

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


def _simplify_expression(expr):
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
    selected_indices: frozenset[IndexType] | None,
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
    selected_indices: frozenset[IndexType] | None,
):
    label_entries: dict[str, tuple[object, object]] = {}
    label_counts: Counter = Counter()

    for occurrence in term.fields:
        slot_labels = occurrence.field.unpack_slot_labels(occurrence.labels)
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


def _expand_occurrence(occurrence, assignment_by_name, selected_indices):
    field = occurrence.field
    slot_labels = field.unpack_slot_labels(occurrence.labels)
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
    for slot, slot_label in slot_labels.items():
        if slot == flavor_slot:
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
                tuple(
                    sorted(
                        (kind, _canonical_value_key(value))
                        for kind, value in occurrence.labels.items()
                    )
                ),
            )
            for occurrence in term.fields
        ),
        tuple(
            (action.target, _canonical_value_key(action.lorentz_index))
            for action in term.derivatives
        ),
        term.closed_dirac_bilinears,
        term.sector,
    )


def _merge_terms(terms: list[InteractionTerm]) -> tuple[InteractionTerm, ...]:
    merged: dict[tuple, InteractionTerm] = {}
    for term in terms:
        key = _term_structure_key(term)
        prior = merged.get(key)
        if prior is None:
            merged[key] = term
            continue
        coupling = _simplify_expression(prior.coupling + term.coupling)
        if _is_zero_expression(coupling):
            del merged[key]
            continue
        merged[key] = replace(prior, coupling=coupling)
    return tuple(merged.values())


def expand_flavor_terms(
    terms,
    *,
    parameters: tuple[Parameter, ...] = (),
    selected_indices: tuple[IndexType, ...] | None = None,
) -> tuple[InteractionTerm, ...]:
    """Expand flavor-generic fields in compiled interaction terms."""

    selected = None if selected_indices is None else frozenset(selected_indices)
    expanded_terms: list[InteractionTerm] = []

    for term in terms:
        assignments = _collect_term_flavor_labels(
            term,
            parameters,
            selected,
        )
        if not assignments:
            expanded_terms.append(term)
            continue

        assignment_ranges = [range(1, index.dimension + 1) for _, _label, index in assignments]
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
            coupling = _simplify_expression(coupling)
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

    return _merge_terms(expanded_terms)
