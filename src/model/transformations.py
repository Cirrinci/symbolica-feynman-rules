"""Declarative simultaneous field transformations on compiled interactions.

Covariant derivatives and field strengths are expanded in the original gauge
basis before this layer runs. At that point fields and partial derivatives
carry structural metadata, so transformations do not need to reparse scalar
expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from itertools import product
from typing import Callable, Iterable, Mapping, Optional, Sequence

from symbolica import AtomType, Expression, S

from .interactions import FieldOccurrence, InteractionTerm
from .lagrangian import CompiledLagrangian
from .metadata import (
    ConjugateField,
    Field,
    IndexType,
    Parameter,
    representation_family,
)


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


def _coerce_occurrence(value) -> FieldOccurrence:
    if isinstance(value, FieldOccurrence):
        return value
    if isinstance(value, ConjugateField):
        return value.field.occurrence(conjugated=True)
    if isinstance(value, Field):
        return value.occurrence()
    raise TypeError(
        "Transformation replacement fields must be Field, Field.bar, or "
        f"FieldOccurrence values, got {type(value).__name__}."
    )


@dataclass(frozen=True)
class ReplacementTerm:
    """One monomial in a field replacement."""

    coefficient: object = 1
    fields: tuple[object, ...] = ()

    def occurrences(self) -> tuple[FieldOccurrence, ...]:
        return tuple(_coerce_occurrence(value) for value in self.fields)


def replacement(coefficient=1, *fields) -> ReplacementTerm:
    """Compact constructor for one replacement monomial."""

    return ReplacementTerm(coefficient=coefficient, fields=tuple(fields))


@dataclass
class TransformationContext:
    """Occurrence-specific context supplied to callable rule builders."""

    occurrence: FieldOccurrence
    term: InteractionTerm
    slot: int
    _used_labels: set[str] = dataclass_field(default_factory=set)
    _counter: int = 0

    def label(self, slot: int):
        return self.occurrence.slot_labels.get(slot)

    def fresh(self, index: IndexType, stem: str = "transform"):
        prefix = index.prefix or index.kind
        while True:
            self._counter += 1
            candidate = S(f"{prefix}_{stem}_{self._counter}")
            if _key(candidate) not in self._used_labels:
                self._used_labels.add(_key(candidate))
                return candidate


ReplacementBuilder = Callable[
    [TransformationContext],
    Sequence[ReplacementTerm],
]


@dataclass(frozen=True)
class FieldTransformation:
    """One FeynRules-like field definition.

    ``components`` maps source index-slot positions to fixed component values.
    Static terms support automatic conjugation. Callable builders can provide
    ``conjugate_builder`` for matrix-index or projector-specific conjugation.
    """

    source: Field
    terms: tuple[ReplacementTerm, ...] = ()
    components: Mapping[int, object] = dataclass_field(default_factory=dict)
    builder: Optional[ReplacementBuilder] = None
    conjugate_terms: Optional[tuple[ReplacementTerm, ...]] = None
    conjugate_builder: Optional[ReplacementBuilder] = None
    dependencies: tuple[Field, ...] = ()
    auto_conjugate: bool = True
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.source, Field):
            raise TypeError("FieldTransformation.source must be a Field.")
        if self.builder is not None and self.terms:
            raise ValueError("Use either terms= or builder= for one transformation.")
        for slot in self.components:
            if not (0 <= slot < len(self.source.indices)):
                raise ValueError(
                    f"Component slot {slot} is outside field {self.source.name!r}."
                )
        if any(not isinstance(field, Field) for field in self.dependencies):
            raise TypeError(
                "FieldTransformation.dependencies must contain Field instances."
            )

    @property
    def display_name(self) -> str:
        return self.name or self.source.name

    def matches(self, occurrence: FieldOccurrence) -> bool:
        if occurrence.field is not self.source:
            return False
        for slot, expected in self.components.items():
            actual = occurrence.slot_labels.get(slot)
            if actual is None or _key(actual) != _key(expected):
                return False
        return True

    def referenced_fields(self) -> frozenset[Field]:
        values: list[ReplacementTerm] = list(self.terms)
        if self.conjugate_terms is not None:
            values.extend(self.conjugate_terms)
        return frozenset(self.dependencies) | frozenset(
            occurrence.field
            for term in values
            for occurrence in term.occurrences()
        )


class CyclicTransformationError(ValueError):
    """Raised when field definitions contain a dependency cycle."""


def _conjugate_coefficient(value, real_symbols: Sequence[object]):
    if not hasattr(value, "conj"):
        return value
    result = value.conj()
    targets = (
        symbol.symbol if isinstance(symbol, Parameter) else symbol
        for symbol in real_symbols
    )
    for target in sorted(targets, key=lambda item: len(_key(item)), reverse=True):
        if hasattr(target, "conj") and hasattr(result, "replace"):
            result = result.replace(target.conj(), target)
    return result


def _conjugate_occurrence(occurrence: FieldOccurrence) -> FieldOccurrence:
    conjugated = False
    if not occurrence.field.self_conjugate:
        conjugated = not occurrence.conjugated
    return occurrence.field.occurrence(
        conjugated=conjugated,
        labels=occurrence.labels,
    )


def _conjugate_terms(
    terms: Sequence[ReplacementTerm],
    *,
    real_symbols: Sequence[object],
) -> tuple[ReplacementTerm, ...]:
    return tuple(
        ReplacementTerm(
            coefficient=_conjugate_coefficient(term.coefficient, real_symbols),
            fields=tuple(
                _conjugate_occurrence(occurrence)
                for occurrence in reversed(term.occurrences())
            ),
        )
        for term in terms
    )


def _term_used_labels(term: InteractionTerm) -> set[str]:
    return {_key(binding.label) for binding in term.index_bindings}


def _rule_terms(
    rule: FieldTransformation,
    *,
    occurrence: FieldOccurrence,
    term: InteractionTerm,
    slot: int,
    real_symbols: Sequence[object],
) -> tuple[ReplacementTerm, ...]:
    context = TransformationContext(
        occurrence=occurrence,
        term=term,
        slot=slot,
        _used_labels=_term_used_labels(term),
    )
    conjugated = bool(occurrence.conjugated and not occurrence.field.self_conjugate)
    if conjugated:
        if rule.conjugate_builder is not None:
            return tuple(rule.conjugate_builder(context))
        if rule.conjugate_terms is not None:
            return tuple(rule.conjugate_terms)
        if rule.builder is not None:
            if not rule.auto_conjugate:
                return ()
            raise ValueError(
                f"Transformation {rule.display_name!r} uses builder=; declare "
                "conjugate_builder= for conjugated occurrences."
            )
        if rule.auto_conjugate:
            return _conjugate_terms(rule.terms, real_symbols=real_symbols)
        return ()

    if rule.builder is not None:
        return tuple(rule.builder(context))
    return tuple(rule.terms)


def _validate_acyclic(rules: Sequence[FieldTransformation]) -> None:
    sources = {rule.source for rule in rules}
    graph = {
        rule.source: tuple(
            field for field in rule.referenced_fields() if field in sources
        )
        for rule in rules
    }
    visiting: set[Field] = set()
    visited: set[Field] = set()

    def visit(field: Field, path: tuple[Field, ...]):
        if field in visiting:
            cycle = path[path.index(field) :] + (field,)
            raise CyclicTransformationError(
                "Cyclic field transformations: "
                + " -> ".join(item.name for item in cycle)
            )
        if field in visited:
            return
        visiting.add(field)
        for target in graph.get(field, ()):
            visit(target, path + (target,))
        visiting.remove(field)
        visited.add(field)

    for source in graph:
        visit(source, (source,))


def _matching_rule(
    occurrence: FieldOccurrence,
    rules: Sequence[FieldTransformation],
) -> Optional[FieldTransformation]:
    matches = [rule for rule in rules if rule.matches(occurrence)]
    if len(matches) > 1:
        names = ", ".join(rule.display_name for rule in matches)
        raise ValueError(
            f"Ambiguous transformations for {occurrence.field.name!r}: {names}."
        )
    return matches[0] if matches else None


def _transform_term_once(
    term: InteractionTerm,
    *,
    rules: Sequence[FieldTransformation],
    real_symbols: Sequence[object],
) -> tuple[InteractionTerm, ...]:
    from lagrangian.operator_action import splice_field_replacement

    selected: list[Optional[tuple[FieldTransformation, tuple[ReplacementTerm, ...]]]] = []
    for slot, occurrence in enumerate(term.fields):
        rule = _matching_rule(occurrence, rules)
        if rule is None:
            selected.append(None)
            continue
        selected.append(
            (
                rule,
                _rule_terms(
                    rule,
                    occurrence=occurrence,
                    term=term,
                    slot=slot,
                    real_symbols=real_symbols,
                ),
            )
        )

    if all(item is None for item in selected):
        return (term,)
    if any(item is not None and item[1] == () for item in selected):
        return ()

    choices = [
        item[1] if item is not None else (ReplacementTerm(fields=(term.fields[slot],)),)
        for slot, item in enumerate(selected)
    ]
    output: list[InteractionTerm] = []
    for selected_terms in product(*choices):
        branches = (term,)
        for slot in reversed(range(len(term.fields))):
            item = selected[slot]
            if item is None:
                continue
            rule, _options = item
            selected_term = selected_terms[slot]
            next_branches: list[InteractionTerm] = []
            for branch in branches:
                replacement_occurrences = selected_term.occurrences()
                if (
                    not replacement_occurrences
                    and any(action.target == slot for action in branch.derivatives)
                ):
                    # Partial derivatives annihilate spacetime-independent
                    # replacement coefficients such as vacuum expectation
                    # values and mixing angles.
                    continue
                next_branches.extend(
                    splice_field_replacement(
                        branch,
                        slot=slot,
                        replacement=replacement_occurrences,
                        coefficient=selected_term.coefficient,
                        name=rule.display_name,
                    )
                )
            branches = tuple(next_branches)
        output.extend(branches)
    return tuple(output)


def _has_transformable_fields(
    terms: Sequence[InteractionTerm],
    rules: Sequence[FieldTransformation],
) -> bool:
    return any(
        _matching_rule(occurrence, rules) is not None
        for term in terms
        for occurrence in term.fields
    )


def apply_field_transformations(
    lagrangian: CompiledLagrangian,
    rules: Sequence[FieldTransformation],
    *,
    repeat: bool = True,
    max_passes: int = 32,
    real_symbols: Sequence[object] = (),
) -> CompiledLagrangian:
    """Apply one simultaneous field-transformation stage.

    Replacements created during one pass are not matched until the next pass.
    With ``repeat=True``, dependent rules are reapplied to a fixed point.
    """

    ordered_rules = tuple(rules)
    _validate_acyclic(ordered_rules)
    current = tuple(lagrangian.terms)
    passes = 0
    while True:
        transformed = tuple(
            output
            for term in current
            for output in _transform_term_once(
                term,
                rules=ordered_rules,
                real_symbols=real_symbols,
            )
        )
        passes += 1
        if not repeat or not _has_transformable_fields(transformed, ordered_rules):
            return CompiledLagrangian(
                terms=transformed,
                parameters=lagrangian.parameters,
            )
        if passes >= max_passes:
            raise RuntimeError(
                f"Field transformations did not reach a fixed point in {max_passes} passes."
            )
        current = transformed


def _replace_occurrence_components(
    occurrence: FieldOccurrence,
    assignments: Mapping[tuple[IndexType, str], int],
) -> FieldOccurrence:
    slot_labels = occurrence.slot_labels
    for slot, index in enumerate(occurrence.field.indices):
        label = slot_labels.get(slot)
        if label is None:
            continue
        value = assignments.get((index, _key(label)))
        if value is not None:
            slot_labels = slot_labels.replace(slot, _num(value))
    return occurrence.with_slot_labels(slot_labels)


def _component_metric_replacements(
    indices: Sequence[IndexType],
) -> tuple[tuple[object, object], ...]:
    replacements = []
    for index in indices:
        if index.dimension is None:
            continue
        left_label = S(f"component_metric_left_{index.kind}")
        right_label = S(f"component_metric_right_{index.kind}")
        symbolic_metric = index.representation.g(
            left_label,
            right_label,
        ).to_expression()
        for left in range(1, index.dimension + 1):
            for right in range(1, index.dimension + 1):
                pattern = symbolic_metric.replace(
                    left_label,
                    _num(left),
                ).replace(
                    right_label,
                    _num(right),
                )
                atom_tree = pattern.to_atom_tree() if hasattr(pattern, "to_atom_tree") else None
                if atom_tree is not None and str(atom_tree.atom_type) == "AtomType.Num":
                    continue
                replacements.append(
                    (
                        pattern,
                        _num(1 if left == right else 0),
                    )
                )
    return tuple(replacements)


def _atom_symbol_name(head: str) -> str:
    return head.rsplit("::", 1)[-1]


def _is_fixed_component(value) -> bool:
    if isinstance(value, int):
        return True
    return (
        isinstance(value, Expression)
        and value.get_type() == AtomType.Num
    )


def _collect_coupling_component_labels(
    coupling,
    *,
    selected: Sequence[IndexType],
    labels: dict[tuple[IndexType, str], object],
) -> None:
    if not hasattr(coupling, "to_atom_tree"):
        return

    index_by_representation = {
        representation_family(index.representation): index
        for index in selected
    }

    def visit(node):
        if str(node.atom_type) == "AtomType.Fn" and len(node.tail) == 2:
            family = node.head.rsplit("::", 1)[-1]
            dimension_node, label_node = node.tail
            representation = f"{family}({dimension_node.head})"
            index = index_by_representation.get(representation)
            if (
                index is not None
                and str(label_node.atom_type) == "AtomType.Var"
            ):
                label = S(_atom_symbol_name(label_node.head))
                labels[(index, _key(label))] = label
        for child in node.tail:
            visit(child)

    visit(coupling.to_atom_tree())


def expand_index_components(
    lagrangian: CompiledLagrangian,
    indices: Iterable[IndexType],
    *,
    tensor_components: Mapping[object, object] = (),
) -> CompiledLagrangian:
    """Expand selected finite index labels into explicit component values."""

    selected = tuple(indices)
    selected_set = frozenset(selected)
    for index in selected:
        if index.dimension is None:
            raise ValueError(
                f"Index {index.name!r} needs a finite dimension for component expansion."
            )

    tensor_items = (
        tuple(tensor_components.items())
        if isinstance(tensor_components, Mapping)
        else tuple(tensor_components)
    )
    replacements = _component_metric_replacements(selected) + tensor_items
    expanded: list[InteractionTerm] = []

    for term in lagrangian.terms:
        labels: dict[tuple[IndexType, str], object] = {}
        for occurrence in term.fields:
            for slot, index in enumerate(occurrence.field.indices):
                if index not in selected_set:
                    continue
                label = occurrence.slot_labels.get(slot)
                if label is None:
                    raise ValueError(
                        f"Cannot component-expand {occurrence.field.name!r}: "
                        f"index {index.name!r} has no label."
                    )
                if _is_fixed_component(label):
                    continue
                labels[(index, _key(label))] = label
        _collect_coupling_component_labels(
            term.coupling,
            selected=selected,
            labels=labels,
        )

        ordered = tuple(
            (index, name, labels[(index, name)])
            for index, name in sorted(
                labels,
                key=lambda item: (item[0].name, item[1]),
            )
        )
        if not ordered:
            coupling = term.coupling
            if hasattr(coupling, "replace"):
                for pattern, value in replacements:
                    coupling = coupling.replace(pattern, value)
            if hasattr(coupling, "expand"):
                coupling = coupling.expand()
            if not _is_zero(coupling):
                expanded.append(replace(term, coupling=coupling))
            continue

        ranges = [range(1, index.dimension + 1) for index, _name, _label in ordered]
        for values in product(*ranges):
            assignments = {
                (index, name): value
                for (index, name, _label), value in zip(ordered, values)
            }
            coupling = term.coupling
            if hasattr(coupling, "replace"):
                for index, name, label in ordered:
                    coupling = coupling.replace(label, _num(assignments[(index, name)]))
                for pattern, value in replacements:
                    coupling = coupling.replace(pattern, value)
            if hasattr(coupling, "expand"):
                coupling = coupling.expand()
            if _is_zero(coupling):
                continue
            expanded.append(
                replace(
                    term,
                    coupling=coupling,
                    fields=tuple(
                        _replace_occurrence_components(occurrence, assignments)
                        for occurrence in term.fields
                    ),
                )
            )

    return CompiledLagrangian(
        terms=tuple(expanded),
        parameters=lagrangian.parameters,
    )


__all__ = (
    "CyclicTransformationError",
    "FieldTransformation",
    "ReplacementTerm",
    "TransformationContext",
    "apply_field_transformations",
    "expand_index_components",
    "replacement",
)
