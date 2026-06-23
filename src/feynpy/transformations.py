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
    is_spinor_index,
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
    real_symbols: Sequence[object] = ()
    _label_pool: "FreshLabelPool" = dataclass_field(
        default_factory=lambda: FreshLabelPool()
    )

    def label(self, slot: int):
        return self.occurrence.slot_labels.get(slot)

    def fresh(self, index: IndexType, stem: str = "transform"):
        return self._label_pool.fresh(index, stem)


@dataclass
class FreshLabelPool:
    used_labels: set[str] = dataclass_field(default_factory=set)
    counter: int = 0

    def fresh(self, index: IndexType, stem: str = "transform"):
        prefix = index.prefix or index.kind
        while True:
            self.counter += 1
            candidate = S(f"{prefix}_{stem}_{self.counter}")
            if _key(candidate) not in self.used_labels:
                self.used_labels.add(_key(candidate))
                return candidate


ReplacementBuilder = Callable[
    [TransformationContext],
    Sequence[ReplacementTerm],
]


def _is_scalar_constant(value) -> bool:
    """True for plain numeric/symbolic scalars used as constant replacements."""
    from fractions import Fraction

    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, complex, Fraction, Expression))


def _occurrence_from_field_factor(factor) -> FieldOccurrence:
    return factor.field.occurrence(
        conjugated=factor.conjugated,
        labels=factor.labels or None,
    )


def _expr_monomials(expr) -> tuple[object, ...]:
    """Parse a replacement expression into ``ReplacementTerm``/``_DeclaredMonomial``.

    Accepts the same field-arithmetic DSL used by ``Model(lagrangian_decl=...)``:
    ``Field``, ``Field.bar``, their scalar-weighted sums and products, matrix
    factors (``ProjM``, ``ProjP``, ``rotation(...)``), bare scalar constants (for
    vacuum shifts), already-built ``ReplacementTerm`` values, or a tuple/list
    mixing those.
    """
    from .declared import _DeclaredMonomial
    from .lowering import _declared_source_terms_from_item

    items = expr if isinstance(expr, (tuple, list)) else (expr,)
    out: list[object] = []
    for item in items:
        if isinstance(item, ReplacementTerm):
            out.append(item)
            continue
        declared = _declared_source_terms_from_item(item)
        if declared is None:
            if _is_scalar_constant(item):
                out.append(ReplacementTerm(coefficient=item, fields=()))
                continue
            raise TypeError(
                "FieldTransformation expression must be built from Fields, "
                "Field.bar, their scalar-weighted sums/products, matrix factors "
                "(ProjM/ProjP/rotation), scalar constants, or ReplacementTerm "
                f"values; got {type(item).__name__}."
            )
        for term in declared:
            if not isinstance(term, _DeclaredMonomial):
                raise TypeError(
                    "FieldTransformation expression only accepts field monomials; "
                    f"got {type(term).__name__}. Use terms=/builder= instead."
                )
            out.append(term)
    return tuple(out)


def _monomial_has_matrix(monomial) -> bool:
    from .declared import _DeclaredMonomial, _MatrixFactor

    return isinstance(monomial, _DeclaredMonomial) and any(
        isinstance(factor, _MatrixFactor) for factor in monomial.factors
    )


def _static_term_from_monomial(monomial) -> ReplacementTerm:
    from .declared import _DeclaredMonomial, _FieldFactor

    if isinstance(monomial, ReplacementTerm):
        return monomial
    assert isinstance(monomial, _DeclaredMonomial)
    fields: list[FieldOccurrence] = []
    for factor in monomial.factors:
        if not isinstance(factor, _FieldFactor):
            raise TypeError(
                "An expression-style FieldTransformation may only contain plain "
                f"field factors; got {type(factor).__name__}. Operators such as "
                "CovD, Gamma, PartialD, or FieldStrength are not valid replacement "
                "targets. Use builder= for index-dependent replacements."
            )
        fields.append(_occurrence_from_field_factor(factor))
    return ReplacementTerm(coefficient=monomial.coefficient, fields=tuple(fields))


def replacement_terms_from_expr(expr) -> tuple[ReplacementTerm, ...]:
    """Lower a matrix-free replacement expression to ``ReplacementTerm`` monomials."""
    return tuple(_static_term_from_monomial(item) for item in _expr_monomials(expr))


def _expr_dependencies(monomials: Sequence[object]) -> tuple[Field, ...]:
    from .declared import _DeclaredMonomial, _FieldFactor

    fields: list[Field] = []
    for monomial in monomials:
        if isinstance(monomial, ReplacementTerm):
            fields.extend(occurrence.field for occurrence in monomial.occurrences())
            continue
        if isinstance(monomial, _DeclaredMonomial):
            for factor in monomial.factors:
                if isinstance(factor, _FieldFactor):
                    fields.append(factor.field)
    return tuple(dict.fromkeys(fields))


def _resolve_matrix_monomial(
    monomial,
    *,
    occurrence: FieldOccurrence,
    label_pool: "FreshLabelPool",
    conjugated: bool,
    real_symbols: Sequence[object],
) -> tuple[ReplacementTerm, ...]:
    """Wire one matrix monomial against a concrete occurrence's indices.

    Matrices act on the source's free index of their family; the contracted leg
    is a fresh label shared with the target field. Conjugated occurrences use the
    factor's conjugate builder with swapped legs and flip the target's bar.
    """
    from .declared import _DeclaredMonomial, _FieldFactor, _MatrixFactor

    if isinstance(monomial, ReplacementTerm):
        terms = (monomial,)
        return (
            _conjugate_terms(terms, real_symbols=real_symbols)
            if conjugated
            else terms
        )

    assert isinstance(monomial, _DeclaredMonomial)
    matrices = [f for f in monomial.factors if isinstance(f, _MatrixFactor)]
    field_factors = [f for f in monomial.factors if isinstance(f, _FieldFactor)]
    extras = [
        f
        for f in monomial.factors
        if not isinstance(f, (_MatrixFactor, _FieldFactor))
    ]
    if extras:
        raise TypeError(
            "A matrix-valued field definition may only contain matrix factors "
            f"(ProjM/ProjP/rotation) and one target field; got {type(extras[0]).__name__}."
        )
    if len(field_factors) != 1:
        raise TypeError(
            "A matrix-valued field definition must contain exactly one target "
            f"field; got {len(field_factors)}."
        )
    target = field_factors[0].field

    source_labels: dict[IndexType, object] = {}
    for slot, index in enumerate(occurrence.field.indices):
        label = occurrence.slot_labels.get(slot)
        if _is_symbolic_label(label):
            source_labels.setdefault(index, label)

    chains: dict[IndexType, list[_MatrixFactor]] = {}
    for matrix in matrices:
        chains.setdefault(matrix.index, []).append(matrix)

    coefficient = (
        _conjugate_coefficient(monomial.coefficient, real_symbols)
        if conjugated
        else monomial.coefficient
    )
    endpoints: dict[IndexType, object] = {}
    for index_type, chain in chains.items():
        source_label = source_labels.get(index_type)
        if source_label is None:
            raise ValueError(
                f"Source {occurrence.field.name!r} has no free {index_type.name!r} "
                "index for the matrix in its field definition."
            )
        points = [source_label] + [
            label_pool.fresh(index_type, "transform") for _ in chain
        ]
        for position, matrix in enumerate(chain):
            left, right = points[position], points[position + 1]
            coefficient = coefficient * (
                matrix.conjugate_build(right, left)
                if conjugated
                else matrix.build(left, right)
            )
        endpoints[index_type] = points[-1]

    target_labels: dict[int, object] = {}
    for slot, index in enumerate(target.indices):
        if index in endpoints:
            target_labels[slot] = endpoints[index]
        elif index in source_labels:
            target_labels[slot] = source_labels[index]

    conjugate_target = field_factors[0].conjugated
    if conjugated and not target.self_conjugate:
        conjugate_target = not conjugate_target
    target_occurrence = target.occurrence(
        conjugated=conjugate_target,
        labels=target.pack_slot_labels(target_labels),
    )
    return (ReplacementTerm(coefficient=coefficient, fields=(target_occurrence,)),)


def _matrix_expr_builders(
    monomials: Sequence[object],
) -> tuple[ReplacementBuilder, ReplacementBuilder]:
    def make(conjugated: bool) -> ReplacementBuilder:
        def build(context: TransformationContext) -> tuple[ReplacementTerm, ...]:
            out: list[ReplacementTerm] = []
            for monomial in monomials:
                out.extend(
                    _resolve_matrix_monomial(
                        monomial,
                        occurrence=context.occurrence,
                        label_pool=context._label_pool,
                        conjugated=conjugated,
                        real_symbols=context.real_symbols,
                    )
                )
            return tuple(out)

        return build

    return make(False), make(True)


@dataclass(frozen=True)
class FieldTransformation:
    """One FeynRules-like field definition.

    The replacement can be given three ways (use exactly one):

    * ``expr=`` (also accepted positionally): a FeynRules-like field expression.
      It supports plain mixings ``FieldTransformation(B, -sw * Z + cw * A)``,
      vacuum shifts ``FieldTransformation(Phi, vev * c + c * H, components={0: 2})``,
      and matrix factors such as chiral projectors and flavor rotations,
      ``FieldTransformation(LL, ProjM * l, components={1: 2})`` or
      ``FieldTransformation(QL, rotation(CKM, CKMDag) * ProjM * dq, components={1: 2})``.
      Matrix expressions are wired against each occurrence's free indices and are
      conjugated automatically for ``field.bar`` occurrences.
    * ``terms=``: explicit ``replacement(...)`` monomials.
    * ``builder=``: a callable for fully custom index-dependent replacements.

    ``components`` maps source index-slot positions to fixed component values.
    Static terms support automatic conjugation. Callable builders can provide
    ``conjugate_builder`` for matrix-index or projector-specific conjugation.
    """

    source: Field
    expr: object = None
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
        provided = [
            label
            for label, is_set in (
                ("expr", self.expr is not None),
                ("terms", bool(self.terms)),
                ("builder", self.builder is not None),
            )
            if is_set
        ]
        if len(provided) > 1:
            raise ValueError(
                "Provide only one of expr=, terms=, builder= for one "
                f"transformation (got {', '.join(provided)})."
            )
        if self.expr is not None:
            monomials = _expr_monomials(self.expr)
            if any(_monomial_has_matrix(monomial) for monomial in monomials):
                builder, conjugate_builder = _matrix_expr_builders(monomials)
                object.__setattr__(self, "builder", builder)
                object.__setattr__(self, "conjugate_builder", conjugate_builder)
                object.__setattr__(
                    self,
                    "dependencies",
                    tuple(
                        dict.fromkeys(
                            self.dependencies + _expr_dependencies(monomials)
                        )
                    ),
                )
            else:
                object.__setattr__(
                    self,
                    "terms",
                    tuple(_static_term_from_monomial(item) for item in monomials),
                )
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


def _is_symbolic_label(value) -> bool:
    return value is not None and not _is_fixed_component(value)


def _is_conjugated_occurrence(occurrence: FieldOccurrence) -> bool:
    return bool(occurrence.conjugated and not occurrence.field.self_conjugate)


def _rule_handles_occurrence(
    rule: FieldTransformation,
    occurrence: FieldOccurrence,
) -> bool:
    if not _is_conjugated_occurrence(occurrence):
        return True
    return (
        rule.conjugate_builder is not None
        or rule.conjugate_terms is not None
        or rule.auto_conjugate
    )


def _rule_terms(
    rule: FieldTransformation,
    *,
    occurrence: FieldOccurrence,
    term: InteractionTerm,
    slot: int,
    real_symbols: Sequence[object],
    label_pool: FreshLabelPool,
) -> tuple[ReplacementTerm, ...]:
    context = TransformationContext(
        occurrence=occurrence,
        term=term,
        slot=slot,
        real_symbols=real_symbols,
        _label_pool=label_pool,
    )
    conjugated = _is_conjugated_occurrence(occurrence)
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
        raise ValueError(
            f"Transformation {rule.display_name!r} does not define a conjugate replacement."
        )

    if rule.builder is not None:
        return tuple(rule.builder(context))
    return tuple(rule.terms)


def _validate_acyclic(rules: Sequence[FieldTransformation]) -> None:
    source_to_rule_ids: dict[Field, tuple[int, ...]] = {}
    for rule_id, rule in enumerate(rules):
        source_to_rule_ids.setdefault(rule.source, [])
        source_to_rule_ids[rule.source].append(rule_id)
    source_to_rule_ids = {
        field: tuple(rule_ids)
        for field, rule_ids in source_to_rule_ids.items()
    }

    def explicit_targets(rule: FieldTransformation) -> tuple[FieldOccurrence, ...]:
        terms: list[ReplacementTerm] = list(rule.terms)
        if rule.conjugate_terms is not None:
            terms.extend(rule.conjugate_terms)
        if rule.expr is not None and rule.builder is not None:
            for monomial in _expr_monomials(rule.expr):
                if isinstance(monomial, ReplacementTerm):
                    terms.append(monomial)
                elif not _monomial_has_matrix(monomial):
                    terms.append(_static_term_from_monomial(monomial))
        return tuple(
            occurrence
            for term in terms
            for occurrence in term.occurrences()
        )

    def target_rule_ids_for_occurrence(
        occurrence: FieldOccurrence,
    ) -> tuple[int, ...]:
        return tuple(
            rule_id
            for rule_id in source_to_rule_ids.get(occurrence.field, ())
            if (
                rules[rule_id].matches(occurrence)
                and _rule_handles_occurrence(rules[rule_id], occurrence)
            )
        )

    graph = {
        rule_id: tuple(
            dict.fromkeys(
                target_rule_id
                for dependency in rules[rule_id].dependencies
                for target_rule_id in source_to_rule_ids.get(dependency, ())
            )
        ) + tuple(
            target_rule_id
            for occurrence in explicit_targets(rules[rule_id])
            for target_rule_id in target_rule_ids_for_occurrence(occurrence)
        )
        for rule_id in range(len(rules))
    }

    visiting: set[int] = set()
    visited: set[int] = set()

    def rule_name(rule: FieldTransformation) -> str:
        if rule.name:
            return rule.name
        if not rule.components:
            return rule.source.name
        qualifiers = ",".join(
            f"{slot + 1}={_key(value)}"
            for slot, value in sorted(rule.components.items())
        )
        return f"{rule.source.name}[{qualifiers}]"

    def visit(rule_id: int, path: tuple[int, ...]):
        if rule_id in visiting:
            cycle = path[path.index(rule_id) :] + (rule_id,)
            raise CyclicTransformationError(
                "Cyclic field transformations: "
                + " -> ".join(rule_name(rules[item]) for item in cycle)
            )
        if rule_id in visited:
            return
        visiting.add(rule_id)
        for target in graph.get(rule_id, ()):
            visit(target, path + (target,))
        visiting.remove(rule_id)
        visited.add(rule_id)

    for rule_id in graph:
        visit(rule_id, (rule_id,))


def _matching_rule(
    occurrence: FieldOccurrence,
    rules: Sequence[FieldTransformation],
) -> Optional[FieldTransformation]:
    matches = [
        rule
        for rule in rules
        if rule.matches(occurrence) and _rule_handles_occurrence(rule, occurrence)
    ]
    if len(matches) > 1:
        names = ", ".join(rule.display_name for rule in matches)
        raise ValueError(
            f"Ambiguous transformations for {occurrence.field.name!r}: {names}."
        )
    return matches[0] if matches else None


def _parameter_lookup(parameters: Sequence[object]) -> dict[str, Parameter]:
    lookup: dict[str, Parameter] = {}
    for parameter in parameters:
        if not isinstance(parameter, Parameter) or not parameter.indices:
            continue
        symbol = parameter.symbol
        if not hasattr(symbol, "to_atom_tree"):
            continue
        lookup[_atom_symbol_name(symbol.to_atom_tree().head)] = parameter
    return lookup


def _field_binding_multiplicities(
    term: InteractionTerm,
) -> tuple[dict[tuple[IndexType, str], object], dict[tuple[IndexType, str], int]]:
    labels: dict[tuple[IndexType, str], object] = {}
    multiplicities: dict[tuple[IndexType, str], int] = {}
    for binding in term.index_bindings:
        if not _is_symbolic_label(binding.label) or is_spinor_index(binding.index):
            continue
        key = (binding.index, _key(binding.label))
        labels[key] = binding.label
        multiplicities[key] = binding.multiplicity
    return labels, multiplicities


def _known_label_types(
    *terms: InteractionTerm,
) -> dict[str, tuple[IndexType, ...]]:
    labels: dict[str, list[IndexType]] = {}
    for term in terms:
        for binding in term.index_bindings:
            if not _is_symbolic_label(binding.label):
                continue
            labels.setdefault(_key(binding.label), [])
            if binding.index not in labels[_key(binding.label)]:
                labels[_key(binding.label)].append(binding.index)
    return {
        label_key: tuple(indices)
        for label_key, indices in labels.items()
    }


def _coefficient_label_multiplicities(
    coupling,
    *,
    known_indices: Sequence[IndexType],
    known_label_types: Mapping[str, Sequence[IndexType]],
    parameters: Sequence[object],
) -> tuple[dict[tuple[IndexType, str], object], dict[tuple[IndexType, str], int]]:
    if not hasattr(coupling, "to_atom_tree"):
        return {}, {}

    indices_by_family: dict[str, tuple[IndexType, ...]] = {}
    for index in dict.fromkeys(known_indices):
        family = representation_family(index.representation)
        indices_by_family.setdefault(family, [])
        if index not in indices_by_family[family]:
            indices_by_family[family].append(index)
    indices_by_family = {
        family: tuple(indices)
        for family, indices in indices_by_family.items()
    }
    parameter_lookup = _parameter_lookup(parameters)
    labels: dict[tuple[IndexType, str], object] = {}
    multiplicities: dict[tuple[IndexType, str], int] = {}

    def count(index: IndexType, label) -> None:
        if is_spinor_index(index):
            return
        key = (index, _key(label))
        labels[key] = label
        multiplicities[key] = multiplicities.get(key, 0) + 1

    def resolve_representation_index(
        representation: str,
        label,
    ) -> Optional[IndexType]:
        candidates = indices_by_family.get(representation, ())
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None

        label_key = _key(label)
        from_known_labels = tuple(
            candidate
            for candidate in candidates
            if candidate in known_label_types.get(label_key, ())
        )
        if len(from_known_labels) == 1:
            return from_known_labels[0]

        prefix_matches = tuple(
            candidate
            for candidate in candidates
            if label_key.startswith(candidate.prefix or candidate.kind)
        )
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        return None

    def visit(node) -> None:
        atom_type = str(node.atom_type)
        if atom_type == "AtomType.Fn":
            parameter = parameter_lookup.get(_atom_symbol_name(node.head))
            if parameter is not None and len(node.tail) == len(parameter.indices):
                for argument, index in zip(node.tail, parameter.indices):
                    if str(argument.atom_type) != "AtomType.Var":
                        continue
                    count(index, S(_atom_symbol_name(argument.head)))

            if len(node.tail) == 2:
                dimension_node, label_node = node.tail
                if str(label_node.atom_type) == "AtomType.Var":
                    label = S(_atom_symbol_name(label_node.head))
                    representation = f"{node.head.rsplit('::', 1)[-1]}({dimension_node.head})"
                    index = resolve_representation_index(representation, label)
                    if index is not None:
                        count(index, label)

        for child in node.tail:
            visit(child)

    visit(coupling.to_atom_tree())
    return labels, multiplicities


def _term_label_multiplicities(
    term: InteractionTerm,
    *,
    reference_terms: Sequence[InteractionTerm],
    parameters: Sequence[object],
) -> dict[tuple[IndexType, str], tuple[object, int]]:
    labels, multiplicities = _field_binding_multiplicities(term)
    known_label_types = _known_label_types(*reference_terms)
    known_indices = tuple(
        dict.fromkeys(
            binding.index
            for reference in reference_terms
            for binding in reference.index_bindings
        )
    )
    coefficient_labels, coefficient_multiplicities = _coefficient_label_multiplicities(
        term.coupling,
        known_indices=known_indices,
        known_label_types=known_label_types,
        parameters=parameters,
    )
    for key, label in coefficient_labels.items():
        labels.setdefault(key, label)
        multiplicities[key] = multiplicities.get(key, 0) + coefficient_multiplicities[key]
    return {
        key: (labels[key], multiplicity)
        for key, multiplicity in multiplicities.items()
    }


def _validate_index_structure(
    *,
    original: InteractionTerm,
    transformed: InteractionTerm,
    transformation_names: Sequence[str],
    parameters: Sequence[object],
) -> None:
    reference_terms = (original, transformed)
    original_bindings = _term_label_multiplicities(
        original,
        reference_terms=reference_terms,
        parameters=parameters,
    )
    transformed_bindings = _term_label_multiplicities(
        transformed,
        reference_terms=reference_terms,
        parameters=parameters,
    )
    stage_name = ", ".join(dict.fromkeys(transformation_names)) or "field transformation"

    for key, (label, original_multiplicity) in original_bindings.items():
        transformed_multiplicity = transformed_bindings.get(key, (label, 0))[1]
        if transformed_multiplicity == original_multiplicity:
            continue
        if original_multiplicity == 1 or transformed_multiplicity == 1:
            raise ValueError(
                f"{stage_name} changes free index {label!r} of type "
                f"{key[0].name!r} from multiplicity {original_multiplicity} to "
                f"{transformed_multiplicity}. Preserve the index explicitly or use "
                "a fixed component."
            )
        raise ValueError(
            f"{stage_name} changes index {label!r} of type {key[0].name!r} from "
            f"multiplicity {original_multiplicity} to {transformed_multiplicity}. "
            "Preserve contracted labels explicitly in the replacement."
        )

    for key, (label, multiplicity) in transformed_bindings.items():
        if key in original_bindings or multiplicity != 1:
            continue
        raise ValueError(
            f"{stage_name} introduces new free index {label!r} of type "
            f"{key[0].name!r}. Preserve only the source free indices, or close "
            "new labels inside the replacement."
        )


def _transform_term_once(
    term: InteractionTerm,
    *,
    rules: Sequence[FieldTransformation],
    real_symbols: Sequence[object],
    parameters: Sequence[object],
) -> tuple[InteractionTerm, ...]:
    from lagrangian.operator_action import splice_field_replacement

    label_pool = FreshLabelPool(used_labels=_term_used_labels(term))
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
                    label_pool=label_pool,
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
    transformation_names = tuple(
        item[0].display_name
        for item in selected
        if item is not None
    )
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
        for branch in branches:
            _validate_index_structure(
                original=term,
                transformed=branch,
                transformation_names=transformation_names,
                parameters=parameters,
            )
            output.append(branch)
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
                    parameters=lagrangian.parameters,
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
