"""
FeynRules-style metadata objects for field declarations and interaction terms.

The goal of this module is to separate intrinsic model metadata from the
combinatorics engine in ``model_symbolica.py``.

``Field`` stores what a field *is*:
  - spin / statistics / conjugation properties
  - intrinsic index types (Lorentz, spinor, color, ...)
  - optional quantum numbers and bookkeeping metadata

``FieldOccurrence`` and ``ExternalLeg`` store what appears in one concrete
interaction or vertex evaluation:
  - which role the field plays in that term (psi vs psibar, scalar vs scalar_dag)
  - which abstract slot labels are attached in that term (mu, a, i, ...)
  - for legs, which momentum label the external state carries

This keeps the user-facing input much closer to a model-file declaration style
while still normalizing into the legacy list-based engine internally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Mapping, Optional, Sequence

from symbolica import S
from symbolica.community.spenso import Representation, Slot

from spenso_structures import (
    BISPINOR,
    COLOR_ADJ,
    COLOR_ADJ_KIND,
    COLOR_FUND,
    COLOR_FUND_KIND,
    DEFAULT_SLOT_LABEL_PREFIXES,
    LORENTZ,
    LORENTZ_KIND,
    SPINOR_KIND,
)

Statistics = Literal["boson", "fermion"]


def _canonical_spin_key(spin) -> str:
    if isinstance(spin, Fraction):
        if spin.denominator == 1:
            return str(spin.numerator)
        return f"{spin.numerator}/{spin.denominator}"
    if hasattr(spin, "to_canonical_string"):
        return spin.to_canonical_string()
    return str(spin)


def _infer_kind_from_spin(spin) -> str:
    key = _canonical_spin_key(spin)
    if key in {"0", "0.0"}:
        return "scalar"
    if key in {"1/2", "0.5"}:
        return "fermion"
    if key in {"1", "1.0"}:
        return "vector"
    raise ValueError(
        "Could not infer field kind from spin "
        f"{spin!r}. Provide kind explicitly."
    )


def _infer_statistics_from_kind(kind: str) -> Statistics:
    if kind in {"fermion", "ghost"}:
        return "fermion"
    return "boson"


def _infer_role_statistics(role_name: str) -> Optional[Statistics]:
    if role_name in {"psi", "psibar", "ghost", "ghost_dag"}:
        return "fermion"
    if role_name:
        return "boson"
    return None


@dataclass(frozen=True)
class FieldRole:
    name: str
    statistics: Optional[Statistics] = None
    base_kind: Optional[str] = None
    conjugated: bool = False

    def __post_init__(self):
        statistics = self.statistics or _infer_role_statistics(self.name)
        base_kind = self.base_kind
        conjugated = self.conjugated or self.name == "psibar" or self.name.endswith("_dag")
        if base_kind is None:
            if self.name in {"psi", "psibar"}:
                base_kind = "fermion"
            elif self.name.endswith("_dag"):
                base_kind = self.name[:-4]
            else:
                base_kind = self.name

        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "base_kind", base_kind)
        object.__setattr__(self, "conjugated", conjugated)

    @property
    def is_fermion(self) -> bool:
        return self.statistics == "fermion"

    def matches(self, other: "FieldRole | str | None") -> bool:
        other_role = normalize_role(other)
        if other_role is None:
            return True
        return self.name == other_role.name


def normalize_role(role: "FieldRole | str | None") -> Optional[FieldRole]:
    if role is None:
        return None
    if isinstance(role, FieldRole):
        return role
    return FieldRole(str(role))


@dataclass(frozen=True)
class IndexType:
    name: str
    representation: Representation
    kind: Optional[str] = None
    prefix: Optional[str] = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        kind = self.kind or self.name.lower()
        prefix = self.prefix or DEFAULT_SLOT_LABEL_PREFIXES.get(kind, kind.replace(" ", "_"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "prefix", prefix)
        alias_set = {self.name.lower(), kind.lower(), *[alias.lower() for alias in self.aliases]}
        object.__setattr__(self, "aliases", tuple(sorted(alias_set)))

    def matches(self, alias: str) -> bool:
        return str(alias).lower() in self.aliases

    def slot(self, label):
        if isinstance(label, Slot):
            return label
        return self.representation(label)


def normalize_index_signature(indices) -> tuple[IndexType, ...]:
    if indices is None:
        return ()
    return tuple(
        resolve_index_type(index_type)
        for index_type in indices
    )


SPINOR_INDEX = IndexType("Spinor", BISPINOR, kind=SPINOR_KIND, aliases=("spinor",))
LORENTZ_INDEX = IndexType("Lorentz", LORENTZ, kind=LORENTZ_KIND, aliases=("lorentz",))
COLOR_FUND_INDEX = IndexType("ColorFund", COLOR_FUND, kind=COLOR_FUND_KIND, aliases=("color_fund", "colour", "color"))
COLOR_ADJ_INDEX = IndexType("ColorAdj", COLOR_ADJ, kind=COLOR_ADJ_KIND, aliases=("color_adj", "adjoint"))

BUILTIN_INDEX_TYPES = (
    SPINOR_INDEX,
    LORENTZ_INDEX,
    COLOR_FUND_INDEX,
    COLOR_ADJ_INDEX,
)


def _binding_values(labels):
    if labels is None:
        return ()
    if isinstance(labels, tuple):
        return tuple(label for label in labels if label is not None)
    if isinstance(labels, list):
        return tuple(label for label in labels if label is not None)
    return (labels,)


def _candidate_index_types(expected_indices=()):
    ordered = []
    seen = set()
    for index_type in tuple(expected_indices) + BUILTIN_INDEX_TYPES:
        if index_type not in seen:
            ordered.append(index_type)
            seen.add(index_type)
    return ordered


def resolve_index_type(kind_or_type, *, expected_indices=()) -> IndexType:
    if isinstance(kind_or_type, IndexType):
        return kind_or_type

    alias = str(kind_or_type)
    matches = [
        index_type
        for index_type in _candidate_index_types(expected_indices)
        if index_type.matches(alias)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Unknown index kind '{alias}'. Provide an IndexType or use a known alias.")
    raise ValueError(
        f"Ambiguous index kind '{alias}'. Matching index types: "
        + ", ".join(index_type.name for index_type in matches)
    )


@dataclass(frozen=True)
class IndexBinding:
    index_type: IndexType
    labels: tuple[object, ...]

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(label for label in self.labels if label is not None))

    @property
    def kind(self) -> str:
        return self.index_type.kind

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(self.index_type.slot(label) for label in self.labels)


@dataclass(frozen=True)
class ConcreteIndexSlot:
    index_type: IndexType
    label: object = None
    declared_position: int = 0
    kind_position: int = 0

    @property
    def kind(self) -> str:
        return self.index_type.kind

    @property
    def slot(self) -> Optional[Slot]:
        if self.label is None:
            return None
        return self.index_type.slot(self.label)

    @property
    def is_labeled(self) -> bool:
        return self.label is not None


def normalize_concrete_index_slots(
    index_slots,
    *,
    expected_indices=(),
) -> Optional[tuple[ConcreteIndexSlot, ...]]:
    if index_slots is None:
        return None
    if isinstance(index_slots, ConcreteIndexSlot):
        index_slots = (index_slots,)
    elif not isinstance(index_slots, Sequence) or isinstance(index_slots, (str, bytes)):
        raise TypeError("Concrete index slots must be a sequence of ConcreteIndexSlot objects.")

    normalized = []
    for declared_position, index_slot in enumerate(index_slots):
        if not isinstance(index_slot, ConcreteIndexSlot):
            raise TypeError("Concrete index slots must contain only ConcreteIndexSlot objects.")
        normalized.append(
            ConcreteIndexSlot(
                index_type=index_slot.index_type,
                label=index_slot.label,
                declared_position=index_slot.declared_position,
                kind_position=index_slot.kind_position,
            )
        )

    normalized = tuple(normalized)
    if expected_indices:
        expected_indices = tuple(expected_indices)
        if len(normalized) != len(expected_indices):
            raise ValueError(
                "Concrete index slots do not match the declared field signature length: "
                f"expected {len(expected_indices)}, got {len(normalized)}."
            )
        for declared_position, (index_slot, expected_index_type) in enumerate(zip(normalized, expected_indices)):
            if index_slot.index_type != expected_index_type:
                raise ValueError(
                    "Concrete index slot type mismatch at declared position "
                    f"{declared_position}: expected '{expected_index_type.name}', "
                    f"got '{index_slot.index_type.name}'."
                )
    return normalized


def bind_indices(*bindings, expected_indices=(), **labels_by_kind) -> tuple[IndexBinding, ...]:
    merged = {}

    def add(index_type, labels):
        labels = _binding_values(labels)
        if not labels:
            return
        existing = merged.get(index_type, ())
        merged[index_type] = existing + labels

    for binding in bindings:
        if isinstance(binding, IndexBinding):
            add(binding.index_type, binding.labels)
            continue
        if isinstance(binding, tuple) and len(binding) == 2:
            index_type = resolve_index_type(binding[0], expected_indices=expected_indices)
            add(index_type, binding[1])
            continue
        raise TypeError(
            "bind_indices expects IndexBinding instances or "
            "(IndexType-or-alias, labels) tuples."
        )

    for kind, labels in labels_by_kind.items():
        index_type = resolve_index_type(kind, expected_indices=expected_indices)
        add(index_type, labels)

    return tuple(
        IndexBinding(index_type=index_type, labels=labels)
        for index_type, labels in merged.items()
        if labels
    )


def normalize_index_bindings(slot_labels, *, expected_indices=()) -> Optional[tuple[IndexBinding, ...]]:
    if slot_labels is None:
        return None
    if isinstance(slot_labels, dict):
        bindings = bind_indices(expected_indices=expected_indices, **slot_labels)
        return bindings or None
    if isinstance(slot_labels, IndexBinding):
        return (slot_labels,)
    if (
        isinstance(slot_labels, tuple)
        and len(slot_labels) == 2
        and not isinstance(slot_labels[0], IndexBinding)
        and not (
            isinstance(slot_labels[0], tuple)
            and len(slot_labels[0]) == 2
        )
    ):
        bindings = bind_indices(slot_labels, expected_indices=expected_indices)
        return bindings or None
    if isinstance(slot_labels, Sequence) and not isinstance(slot_labels, (str, bytes)):
        bindings = bind_indices(*slot_labels, expected_indices=expected_indices)
        return bindings or None
    raise TypeError(
        "slot_labels must be a dict of alias->labels, an IndexBinding, or "
        "a sequence of IndexBinding / (IndexType-or-alias, labels) tuples."
    )


def ordered_index_slots(
    slot_labels=None,
    *,
    expected_indices=(),
) -> tuple[ConcreteIndexSlot, ...]:
    bindings = normalize_index_bindings(
        slot_labels,
        expected_indices=expected_indices,
    ) or ()

    if expected_indices:
        labels_by_type = {
            binding.index_type: binding.labels
            for binding in bindings
        }
        counts = Counter()
        slots = []
        for declared_position, index_type in enumerate(expected_indices):
            kind_position = counts[index_type]
            labels = labels_by_type.get(index_type, ())
            label = labels[kind_position] if kind_position < len(labels) else None
            slots.append(
                ConcreteIndexSlot(
                    index_type=index_type,
                    label=label,
                    declared_position=declared_position,
                    kind_position=kind_position,
                )
            )
            counts[index_type] += 1
        return tuple(slots)

    slots = []
    declared_position = 0
    for binding in bindings:
        for kind_position, label in enumerate(binding.labels):
            slots.append(
                ConcreteIndexSlot(
                    index_type=binding.index_type,
                    label=label,
                    declared_position=declared_position,
                    kind_position=kind_position,
                )
            )
            declared_position += 1
    return tuple(slots)


def index_bindings_from_slots(index_slots) -> Optional[tuple[IndexBinding, ...]]:
    index_slots = normalize_concrete_index_slots(index_slots)
    if not index_slots:
        return None

    labels_by_type = {}
    order = []
    for index_slot in index_slots:
        if index_slot.label is None:
            continue
        if index_slot.index_type not in labels_by_type:
            labels_by_type[index_slot.index_type] = []
            order.append(index_slot.index_type)
        labels_by_type[index_slot.index_type].append(index_slot.label)

    if not order:
        return None

    return tuple(
        IndexBinding(index_type=index_type, labels=tuple(labels_by_type[index_type]))
        for index_type in order
    )


def index_labels_from_slots(index_slots, index_type_or_alias):
    if index_slots is None:
        return ()
    if not index_slots:
        return ()

    target = resolve_index_type(
        index_type_or_alias,
        expected_indices=[slot.index_type for slot in index_slots],
    )
    return tuple(
        slot.label
        for slot in index_slots
        if slot.index_type == target and slot.label is not None
    )


def primary_index_label_from_slots(index_slots, index_type_or_alias):
    labels = index_labels_from_slots(index_slots, index_type_or_alias)
    if not labels:
        return None
    return labels[0]


def primary_binding_labels(bindings, index_type_or_alias):
    bindings = normalize_index_bindings(bindings)
    if bindings is None:
        return None

    target = resolve_index_type(index_type_or_alias, expected_indices=[binding.index_type for binding in bindings])
    for binding in bindings:
        if binding.index_type == target:
            return binding.labels
    return None


def has_index_type(bindings, index_type_or_alias) -> bool:
    bindings = normalize_index_bindings(bindings)
    if bindings is None:
        return False

    target = resolve_index_type(index_type_or_alias, expected_indices=[binding.index_type for binding in bindings])
    return any(binding.index_type == target and binding.labels for binding in bindings)


def merge_index_bindings(base_bindings, extra_bindings, *, expected_indices=()) -> Optional[tuple[IndexBinding, ...]]:
    merged = bind_indices(
        *(normalize_index_bindings(base_bindings, expected_indices=expected_indices) or ()),
        *(normalize_index_bindings(extra_bindings, expected_indices=expected_indices) or ()),
        expected_indices=expected_indices,
    )
    return merged or None


def default_leg_slot_labels(field_slot_labels):
    generated = []
    for leg_position, bindings in enumerate(field_slot_labels, start=1):
        normalized = normalize_index_bindings(bindings)
        if not normalized:
            generated.append(None)
            continue

        leg_bindings = []
        for binding in normalized:
            labels = []
            for index_position, _ in enumerate(binding.labels, start=1):
                suffix = "" if len(binding.labels) == 1 else f"_{index_position}"
                labels.append(S(f"{binding.index_type.prefix}{leg_position}{suffix}"))
            leg_bindings.append(IndexBinding(binding.index_type, tuple(labels)))
        generated.append(tuple(leg_bindings))

    return generated


@dataclass(frozen=True)
class Field:
    name: str
    spin: object
    self_conjugate: bool
    indices: tuple[IndexType, ...] = ()
    conjugate_indices: Optional[tuple[IndexType, ...]] = None
    kind: Optional[str] = None
    statistics: Optional[Statistics] = None
    symbol: object = None
    conjugate_symbol: object = None
    quantum_numbers: Mapping[str, object] = field(default_factory=dict)
    unphysical: bool = False
    ghost_of: Optional[str] = None

    def __post_init__(self):
        kind = self.kind or _infer_kind_from_spin(self.spin)
        statistics = self.statistics or _infer_statistics_from_kind(kind)
        symbol = self.symbol if self.symbol is not None else S(self.name)
        indices = normalize_index_signature(self.indices)
        conjugate_indices = self.conjugate_indices
        if conjugate_indices is None:
            conjugate_indices = indices
        else:
            conjugate_indices = normalize_index_signature(conjugate_indices)
        if self.self_conjugate and conjugate_indices != indices:
            raise ValueError(
                f"Self-conjugate field '{self.name}' cannot declare a distinct conjugate index signature."
            )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "statistics", statistics)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "conjugate_indices", conjugate_indices)

    def role_for(self, *, conjugated: bool = False) -> FieldRole:
        if self.kind == "fermion":
            return FieldRole(
                "psibar" if conjugated else "psi",
                statistics=self.statistics,
                base_kind=self.kind,
                conjugated=conjugated,
            )
        if conjugated and not self.self_conjugate:
            if self.kind == "scalar":
                return FieldRole(
                    "scalar_dag",
                    statistics=self.statistics,
                    base_kind=self.kind,
                    conjugated=True,
                )
            return FieldRole(
                f"{self.kind}_dag",
                statistics=self.statistics,
                base_kind=self.kind,
                conjugated=True,
            )
        return FieldRole(
            self.kind,
            statistics=self.statistics,
            base_kind=self.kind,
            conjugated=False,
        )

    def symbol_for(self, *, conjugated: bool = False):
        if conjugated and not self.self_conjugate and self.conjugate_symbol is not None:
            return self.conjugate_symbol
        return self.symbol

    def index_types_for(self, *, conjugated: bool = False) -> tuple[IndexType, ...]:
        if conjugated:
            return tuple(self.conjugate_indices)
        return tuple(self.indices)

    def expected_slot_counts(self, *, conjugated: bool = False) -> Counter:
        counts = Counter()
        for index_type in self.index_types_for(conjugated=conjugated):
            counts[index_type] += 1
        return counts

    def validate_role(self, role: FieldRole | str | None) -> Optional[FieldRole]:
        role_obj = normalize_role(role)
        if role_obj is None:
            return None
        if role_obj.base_kind != self.kind:
            raise ValueError(
                f"Role '{role_obj.name}' is incompatible with field '{self.name}' "
                f"of kind '{self.kind}'."
            )
        if role_obj.statistics is not None and role_obj.statistics != self.statistics:
            raise ValueError(
                f"Role '{role_obj.name}' is incompatible with field '{self.name}' "
                f"of statistics '{self.statistics}'."
            )
        if self.self_conjugate and role_obj.conjugated:
            raise ValueError(
                f"Field '{self.name}' is self-conjugate, so role '{role_obj.name}' is invalid."
            )
        return role_obj

    def validate_slot_labels(self, slot_label_map, *, conjugated: bool = False):
        expected_indices = self.index_types_for(conjugated=conjugated)
        slot_bindings = normalize_index_bindings(slot_label_map, expected_indices=expected_indices)
        if slot_bindings is None:
            return

        expected = self.expected_slot_counts(conjugated=conjugated)
        if not expected:
            if slot_bindings:
                raise ValueError(
                    f"Field '{self.name}' does not declare intrinsic indices, "
                    f"but got slot labels {slot_bindings}"
                )
            return

        unknown = [binding.index_type for binding in slot_bindings if binding.index_type not in expected]
        if unknown:
            raise ValueError(
                f"Field '{self.name}' does not declare slot kinds {unknown}. "
                f"Expected only {[index_type.name for index_type in expected]}."
            )

        for binding in slot_bindings:
            if len(binding.labels) != expected[binding.index_type]:
                raise ValueError(
                    f"Field '{self.name}' expects {expected[binding.index_type]} label(s) for "
                    f"index type '{binding.index_type.name}', got {len(binding.labels)}."
                )

    def bound_index_slots(
        self,
        slot_labels=None,
        *,
        conjugated: bool = False,
        concrete_slots=None,
    ) -> tuple[ConcreteIndexSlot, ...]:
        expected_indices = self.index_types_for(conjugated=conjugated)
        normalized_concrete = normalize_concrete_index_slots(
            concrete_slots,
            expected_indices=expected_indices,
        )
        if normalized_concrete is not None:
            if slot_labels is not None:
                normalized_bindings = normalize_index_bindings(
                    slot_labels,
                    expected_indices=expected_indices,
                )
                if normalized_bindings != index_bindings_from_slots(normalized_concrete):
                    raise ValueError(
                        f"Field '{self.name}' received inconsistent slot_labels and concrete index slots."
                    )
            return normalized_concrete

        normalized = normalize_index_bindings(slot_labels, expected_indices=expected_indices)
        self.validate_slot_labels(normalized, conjugated=conjugated)
        return ordered_index_slots(normalized, expected_indices=expected_indices)

    def occurrence(
        self,
        *,
        role: Optional[str] = None,
        conjugated: bool = False,
        slot_labels=None,
        concrete_index_slots=None,
        species=None,
    ) -> "FieldOccurrence":
        return FieldOccurrence(
            field=self,
            role=role or self.role_for(conjugated=conjugated),
            species=self.symbol_for(conjugated=conjugated) if species is None else species,
            slot_labels=slot_labels,
            concrete_index_slots=concrete_index_slots,
        )

    def leg(
        self,
        momentum,
        *,
        role: Optional[str] = None,
        conjugated: bool = False,
        slot_labels=None,
        concrete_index_slots=None,
        species=None,
        spin=None,
    ) -> "ExternalLeg":
        return ExternalLeg(
            field=self,
            role=role or self.role_for(conjugated=conjugated),
            species=self.symbol_for(conjugated=conjugated) if species is None else species,
            momentum=momentum,
            slot_labels=slot_labels,
            concrete_index_slots=concrete_index_slots,
            spin=spin,
        )


@dataclass(frozen=True)
class FieldOccurrence:
    field: Field
    role: FieldRole | str
    species: object
    slot_labels: Optional[object] = None
    concrete_index_slots: Optional[Sequence[ConcreteIndexSlot]] = None
    index_slots: tuple[ConcreteIndexSlot, ...] = field(init=False, default=())

    def __post_init__(self):
        role_obj = self.field.validate_role(self.role)
        normalized = normalize_index_bindings(
            self.slot_labels,
            expected_indices=self.field.index_types_for(conjugated=role_obj.conjugated),
        )
        index_slots = self.field.bound_index_slots(
            normalized,
            conjugated=role_obj.conjugated,
            concrete_slots=self.concrete_index_slots,
        )
        object.__setattr__(self, "role", role_obj)
        object.__setattr__(self, "slot_labels", index_bindings_from_slots(index_slots))
        object.__setattr__(self, "concrete_index_slots", tuple(index_slots))
        object.__setattr__(self, "index_slots", index_slots)

    def primary_index_label(self, index_type_or_alias):
        return primary_index_label_from_slots(self.index_slots, index_type_or_alias)


@dataclass(frozen=True)
class ExternalLeg:
    field: Field
    role: FieldRole | str
    species: object
    momentum: object
    slot_labels: Optional[object] = None
    concrete_index_slots: Optional[Sequence[ConcreteIndexSlot]] = None
    spin: object = None
    index_slots: tuple[ConcreteIndexSlot, ...] = field(init=False, default=())

    def __post_init__(self):
        role_obj = self.field.validate_role(self.role)
        normalized = normalize_index_bindings(
            self.slot_labels,
            expected_indices=self.field.index_types_for(conjugated=role_obj.conjugated),
        )
        index_slots = self.field.bound_index_slots(
            normalized,
            conjugated=role_obj.conjugated,
            concrete_slots=self.concrete_index_slots,
        )
        object.__setattr__(self, "role", role_obj)
        object.__setattr__(self, "slot_labels", index_bindings_from_slots(index_slots))
        object.__setattr__(self, "concrete_index_slots", tuple(index_slots))
        object.__setattr__(self, "index_slots", index_slots)

    def primary_index_label(self, index_type_or_alias):
        return primary_index_label_from_slots(self.index_slots, index_type_or_alias)


@dataclass(frozen=True)
class DerivativeAction:
    target: int
    indices: tuple[object, ...]

    def __post_init__(self):
        object.__setattr__(self, "indices", tuple(self.indices))


@dataclass(frozen=True)
class InteractionTerm:
    coupling: object
    fields: tuple[FieldOccurrence, ...]
    derivatives: tuple[DerivativeAction, ...] = ()
    statistics: Optional[Statistics] = None
    label: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "derivatives", tuple(self.derivatives))
        inferred = "fermion" if any(
            field_occurrence.field.statistics == "fermion"
            for field_occurrence in self.fields
        ) else "boson"
        if self.statistics is None:
            object.__setattr__(self, "statistics", inferred)
        elif self.statistics != inferred:
            raise ValueError(
                "Interaction statistics are inconsistent with the declared fields: "
                f"got statistics='{self.statistics}', but the fields imply "
                f"statistics='{inferred}'."
            )


def default_external_legs_for_interaction(
    term: InteractionTerm,
    *,
    momenta: Optional[Sequence] = None,
) -> tuple[ExternalLeg, ...]:
    if momenta is None:
        momenta = tuple(S(f"p{k + 1}") for k in range(len(term.fields)))
    if len(momenta) != len(term.fields):
        raise ValueError("Default external momenta must match the number of interaction fields")

    return tuple(
        ExternalLeg(
            field=occ.field,
            role=occ.role,
            species=occ.species,
            momentum=momenta[k],
        )
        for k, occ in enumerate(term.fields)
    )
