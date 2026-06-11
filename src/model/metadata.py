"""Declarative metadata layer for model definitions."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Literal, Mapping, Optional, Sequence

_REP_ANSI = re.compile(r"\x1b\[[0-9;]*m")

from symbolica import Expression, S
from symbolica.community.spenso import Representation

from symbolic.spenso_structures import (
    BISPINOR,
    COLOR_ADJ,
    COLOR_FUND,
    LORENTZ,
    WEAK_ADJ,
    WEAK_FUND,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    SPINOR_KIND,
    WEAK_ADJ_KIND,
    WEAK_FUND_KIND,
    gamma5_matrix,
    gamma_matrix,
    gauge_generator,
    lorentz_metric,
    structure_constant,
    weak_gauge_generator,
    weak_structure_constant,
)

Statistics = Literal["boson", "fermion"]
# ---------------------------------------------------------------------------
# FieldRole  (typed replacement for raw strings like "psi", "scalar")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldRole:
    """Typed role that a field plays in an interaction or on an external leg.

    The engine uses duck typing on this object (.is_fermion, .compatible_with)
    so the engine module never needs to import model.py.
    """
    name: str
    statistics: Statistics
    conjugated: bool = False

    @property
    def is_fermion(self) -> bool:
        return self.statistics == "fermion"

    def compatible_with(self, other) -> bool:
        if isinstance(other, FieldRole):
            return self.name == other.name
        return self.name == str(other)

    def __repr__(self):
        return f"FieldRole({self.name!r})"


ROLE_SCALAR = FieldRole("scalar", "boson")
ROLE_SCALAR_DAG = FieldRole("scalar_dag", "boson", conjugated=True)
ROLE_VECTOR = FieldRole("vector", "boson")
ROLE_PSI = FieldRole("psi", "fermion")
ROLE_PSIBAR = FieldRole("psibar", "fermion", conjugated=True)
ROLE_GHOST = FieldRole("ghost", "fermion")
ROLE_GHOST_DAG = FieldRole("ghost_dag", "fermion", conjugated=True)


# ---------------------------------------------------------------------------
# IndexType
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexType:
    """One kind of index that a field can carry (spinor, Lorentz, colour, ...)."""

    name: str
    representation: object
    kind: str
    dimension: Optional[int] = None
    is_flavor: bool = False
    prefix: str = ""

    def __post_init__(self):
        if not self.prefix:
            object.__setattr__(self, "prefix", self.kind[:1])


def representation_family(rep: object) -> str:
    """Stable Spenso representation key for index-family checks."""
    return _REP_ANSI.sub("", str(rep))


def is_spinor_index(index: IndexType) -> bool:
    return representation_family(index.representation).startswith("bis")


def is_lorentz_index(index: IndexType) -> bool:
    return representation_family(index.representation).startswith("mink")


def indices_compatible_for_labels(left: IndexType, right: IndexType) -> bool:
    """Whether two declared indices may share one symbolic label in a monomial."""
    if left == right or left.kind == right.kind:
        return True
    if is_spinor_index(left) and is_spinor_index(right):
        return True
    if is_lorentz_index(left) and is_lorentz_index(right):
        return True
    return False


def spinor_kind_for(indices: Sequence[IndexType]) -> str:
    for index in indices:
        if is_spinor_index(index):
            return index.kind
    return SPINOR_KIND


def lorentz_kind_for(indices: Sequence[IndexType]) -> str:
    for index in indices:
        if is_lorentz_index(index):
            return index.kind
    return LORENTZ_KIND


def lorentz_index_for(indices: Sequence[IndexType]) -> Optional[IndexType]:
    for index in indices:
        if is_lorentz_index(index):
            return index
    return None


def lorentz_slots_for(field) -> tuple[int, ...]:
    return tuple(
        slot for slot, index in enumerate(field.indices) if is_lorentz_index(index)
    )


def spinor_slots_for(field) -> tuple[int, ...]:
    return tuple(
        slot for slot, index in enumerate(field.indices) if is_spinor_index(index)
    )


def unique_lorentz_slot(field, *, purpose: str) -> int:
    slots = lorentz_slots_for(field)
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to expose exactly one Lorentz slot; "
            f"found {len(slots)}."
        )
    return slots[0]


def unique_spinor_slot(field, *, purpose: str) -> int:
    slots = spinor_slots_for(field)
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to expose exactly one spinor slot; "
            f"found {len(slots)}."
        )
    return slots[0]


SPINOR_INDEX = IndexType("Spinor", BISPINOR, SPINOR_KIND, dimension=4, prefix="i")
LORENTZ_INDEX = IndexType("Lorentz", LORENTZ, LORENTZ_KIND, dimension=4, prefix="mu")
COLOR_FUND_INDEX = IndexType(
    "ColorFund", COLOR_FUND, COLOR_FUND_KIND, dimension=3, prefix="c"
)
COLOR_ADJ_INDEX = IndexType(
    "ColorAdj", COLOR_ADJ, COLOR_ADJ_KIND, dimension=8, prefix="a"
)
# SU(2)_L weak isospin: doublet (fundamental) and triplet (adjoint).
WEAK_FUND_INDEX = IndexType(
    "WeakFund", WEAK_FUND, WEAK_FUND_KIND, dimension=2, prefix="w"
)
WEAK_ADJ_INDEX = IndexType(
    "WeakAdj", WEAK_ADJ, WEAK_ADJ_KIND, dimension=3, prefix="aw"
)


def flavor_index(
    name: str = "Flavor",
    dimension: int = 3,
    *,
    prefix: str = "f",
    kind: Optional[str] = None,
) -> IndexType:
    """Return a standard flavor index type for class-member expansion."""

    return IndexType(
        name,
        Representation.cof(dimension),
        kind or name.lower(),
        dimension=dimension,
        is_flavor=True,
        prefix=prefix,
    )


def _default_conjugate_symbol(name: str):
    return S(f"{name}bar")


def _validate_dirac_helper_indices(helper_name: str, indices: tuple["IndexType", ...]):
    if any(index == SPINOR_INDEX for index in indices):
        raise ValueError(
            f"{helper_name} appends SPINOR_INDEX automatically; omit it from `indices`."
        )


def dirac_field(
    name: str,
    *,
    indices: tuple[IndexType, ...] = (),
    symbol=None,
    conjugate_symbol=None,
    mass=None,
    quantum_numbers: Optional[Mapping[str, object]] = None,
    class_members: tuple = (),
    flavor_index: Optional[IndexType] = None,
) -> "Field":
    """Declare a Dirac field with the spinor slot appended automatically.

    When ``class_members`` is given, the field is a flavor class (à la
    FeynRules ``ClassMembers``). Members may be passed either as plain names
    (strings) or as fully-constructed Dirac ``Field`` instances; string members
    are auto-built with the same metadata as the class field minus the flavor
    index slot. Their concrete instances are reachable via ``field.class_members``.
    """

    indices = tuple(indices)
    _validate_dirac_helper_indices("dirac_field(...)", indices)
    if symbol is None:
        symbol = S(name)
    if conjugate_symbol is None:
        conjugate_symbol = _default_conjugate_symbol(name)
    return Field(
        name,
        spin=Fraction(1, 2),
        self_conjugate=False,
        indices=indices + (SPINOR_INDEX,),
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        mass=mass,
        quantum_numbers=dict(quantum_numbers or {}),
        flavor_index=flavor_index,
        class_members=tuple(class_members),
    )


def scalar_field(
    name: str,
    *,
    self_conjugate: bool = True,
    indices: tuple[IndexType, ...] = (),
    symbol=None,
    conjugate_symbol=None,
    mass=None,
    quantum_numbers: Optional[Mapping[str, object]] = None,
    class_members: tuple = (),
    flavor_index: Optional[IndexType] = None,
) -> "Field":
    """Declare a scalar field with lightweight defaults.

    Supports flavor-class declarations through ``class_members`` /
    ``flavor_index`` (see :func:`dirac_field`).
    """

    if symbol is None:
        symbol = S(name)
    if conjugate_symbol is None and not self_conjugate:
        conjugate_symbol = _default_conjugate_symbol(name)
    return Field(
        name,
        spin=0,
        self_conjugate=self_conjugate,
        indices=tuple(indices),
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        mass=mass,
        quantum_numbers=dict(quantum_numbers or {}),
        flavor_index=flavor_index,
        class_members=tuple(class_members),
    )


# ---------------------------------------------------------------------------
# GaugeGroup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaugeRepresentation:
    """One matter representation of a gauge group.

    The representation is identified by the index type carried by the matter
    field, plus a builder that inserts the corresponding generator tensor into
    an interaction coupling.  ``slot`` is optional, but becomes important when
    a field carries the same index type more than once and the active slot must
    be selected explicitly.
    """
    index: IndexType
    generator_builder: Callable[[object, object, object], object]
    name: str = ""
    slot: Optional[int] = None
    slot_policy: Literal["unique", "sum"] = "unique"

    def build_generator(self, adjoint_label, left_label, right_label):
        """Build the concrete representation tensor, e.g. T^a_{ij}."""
        return self.generator_builder(adjoint_label, left_label, right_label)

    def slots_for(self, field: "Field") -> tuple[int, ...]:
        """Resolve which field-index slots this representation acts on.

        Semantics:
        - If ``slot`` is explicitly set, it selects that one slot.
        - Otherwise:
          - if no matching slots exist, returns ().
          - if exactly one matching slot exists, returns (slot,).
          - if multiple matching slots exist:
            - ``slot_policy='unique'`` (default): raise (ambiguity must be explicit)
            - ``slot_policy='sum'``: return all matching slots
        """
        matches = [slot for slot, index in enumerate(field.indices) if index == self.index]
        if self.slot is not None:
            if self.slot < 0 or self.slot >= len(field.indices):
                raise ValueError(
                    f"GaugeRepresentation(slot={self.slot}) is out of range for "
                    f"field {field.name!r}."
                )
            if field.indices[self.slot] != self.index:
                raise ValueError(
                    f"GaugeRepresentation(slot={self.slot}) for index {self.index.name!r} "
                    f"does not match field {field.name!r}."
                )
            return (self.slot,)
        if not matches:
            return ()
        if len(matches) > 1:
            if self.slot_policy == "sum":
                return tuple(matches)
            raise ValueError(
                f"Field {field.name!r} carries repeated index type {self.index.name!r}; "
                "declare GaugeRepresentation(slot=...) for strict selection, or set "
                "GaugeRepresentation(slot_policy='sum') to sum over all matching slots."
            )
        return (matches[0],)

@dataclass(frozen=True)
class GaugeGroup:
    """Gauge symmetry group declaration (mirrors M$GaugeGroups)."""
    name: str
    abelian: bool
    coupling: object
    gauge_boson: Optional[object] = None
    ghost_field: Optional[object] = None
    structure_constant: Optional[object] = None
    representations: tuple[GaugeRepresentation, ...] = ()
    charge: Optional[str] = None

    def _matching_representations(
        self,
        field: "Field",
    ) -> tuple[tuple[GaugeRepresentation, tuple[int, ...]], ...]:
        """Resolve the unique supported representation match for one field.

        The current compiler supports at most one matching representation per
        field within a given gauge group. Repeated slots of one representation
        remain supported through ``GaugeRepresentation(slot_policy='sum')``.
        """
        matches = []
        for rep in self.representations:
            slots = rep.slots_for(field)
            if slots:
                matches.append((rep, tuple(slots)))

        if len(matches) > 1:
            rep_names = ", ".join(rep.name or rep.index.name for rep, _ in matches)
            raise ValueError(
                f"Field {field.name!r} matches multiple representations under gauge group "
                f"{self.name!r} ({rep_names}). This is not currently supported."
            )

        return tuple(matches)

    def matter_representation(self, field: "Field") -> Optional[GaugeRepresentation]:
        """Return the gauge representation carried by a given matter field."""
        matches = self._matching_representations(field)
        if not matches:
            return None
        return matches[0][0]

    def matter_representation_and_slot(
        self,
        field: "Field",
    ) -> Optional[tuple[GaugeRepresentation, int]]:
        """Return the gauge representation and the concrete field slot it uses."""
        matches = self._matching_representations(field)
        if not matches:
            return None

        rep, slots = matches[0]
        if len(slots) != 1:
            raise ValueError(
                f"GaugeRepresentation for index {rep.index.name!r} resolved to {len(slots)} slots; "
                "use matter_representation_and_slots(...) or set slot=... for a unique slot."
            )
        return rep, slots[0]

    def matter_representation_and_slots(
        self,
        field: "Field",
    ) -> Optional[tuple[GaugeRepresentation, tuple[int, ...]]]:
        """Return the gauge representation and all concrete field slots it uses."""
        matches = self._matching_representations(field)
        if matches:
            return matches[0]
        return None


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterAssumptions:
    """Structured parameter metadata exposed to validation and reporting."""

    name: str
    symbol: object
    indices: tuple[IndexType, ...]
    real: bool
    complex: bool
    internal: bool
    external: bool
    has_value: bool
    allow_summation: Optional[bool]
    value: object = None


@dataclass(frozen=True)
class Parameter:
    """Model parameter (coupling constant, mass, Yukawa matrix, ...).

    Parameter behaves like its Symbolica symbol in algebraic expressions.

    ``complex_param`` records whether the parameter should be treated as
    complex-valued by default. ``internal`` distinguishes derived/internal
    parameters from external user inputs. ``value`` is optional and may be
    numeric or symbolic; this first pass only stores it as metadata and does
    not trigger automatic evaluation.
    """
    name: str
    symbol: object = None
    indices: tuple[IndexType, ...] = ()
    complex_param: bool = False
    internal: bool = True
    value: object = None
    components: Mapping[tuple[object, ...], object] = field(default_factory=dict)
    allow_summation: Optional[bool] = None

    def __post_init__(self):
        if self.symbol is None:
            object.__setattr__(self, "symbol", S(self.name))
        if not self.indices and self.components:
            raise ValueError(
                f"Parameter {self.name!r} defines indexed components but carries no indices."
            )
        normalized_components = {}
        for raw_key, component_value in self.components.items():
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            if len(key) != len(self.indices):
                raise ValueError(
                    f"Parameter {self.name!r} expects {len(self.indices)} component index value(s), "
                    f"got {len(key)} for key {raw_key!r}."
                )
            normalized_components[tuple(key)] = component_value
        if normalized_components:
            object.__setattr__(self, "components", normalized_components)

    @property
    def is_real(self) -> bool:
        return not self.complex_param

    @property
    def is_complex(self) -> bool:
        return bool(self.complex_param)

    @property
    def is_internal(self) -> bool:
        return bool(self.internal)

    @property
    def is_external(self) -> bool:
        return not self.internal

    @property
    def has_value(self) -> bool:
        return self.value is not None

    def permits_label_summation(self) -> bool:
        """Whether one flavor label may appear more than twice in one term.

        Single-index flavor parameters default to True, matching the usual
        FeynRules diagonal shorthand ``y(f)`` in ``y(f) * l.bar(f) * l(f)``.
        Set ``allow_summation=False`` to reject that pattern explicitly.
        """
        if self.allow_summation is False:
            return False
        if self.allow_summation is True:
            return True
        return len(self.indices) == 1 and self.indices[0].is_flavor

    def assumptions(self) -> ParameterAssumptions:
        """Return a structured summary of the parameter metadata."""

        return ParameterAssumptions(
            name=self.name,
            symbol=self.symbol,
            indices=self.indices,
            real=self.is_real,
            complex=self.is_complex,
            internal=self.is_internal,
            external=self.is_external,
            has_value=self.has_value,
            allow_summation=self.allow_summation,
            value=self.value,
        )

    def __call__(self, *labels):
        if len(labels) != len(self.indices):
            raise TypeError(
                f"Parameter {self.name!r} takes {len(self.indices)} index label(s), "
                f"got {len(labels)}."
            )
        if not labels:
            return self.symbol
        return self.symbol(*labels)

    def __mul__(self, other):
        return self.symbol * other

    def __rmul__(self, other):
        return other * self.symbol

    def __add__(self, other):
        return self.symbol + other

    def __radd__(self, other):
        return other + self.symbol

    def __sub__(self, other):
        return self.symbol - other

    def __rsub__(self, other):
        return other - self.symbol

    def __truediv__(self, other):
        return self.symbol / other

    def __rtruediv__(self, other):
        return other / self.symbol

    def __pow__(self, other):
        return self.symbol ** other

    def __neg__(self):
        return -self.symbol


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

def _infer_kind(spin) -> str:
    key = str(Fraction(spin))
    if key == "0":
        return "scalar"
    if key == "1/2":
        return "fermion"
    if key == "1":
        return "vector"
    raise ValueError(f"Cannot infer field kind from spin {spin!r}; provide kind explicitly.")


def _infer_statistics(kind: str) -> Statistics:
    if kind in ("fermion", "ghost"):
        return "fermion"
    return "boson"


def _copy_index_label_value(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


def _copy_index_labels(labels: Optional[Mapping]) -> dict:
    if not labels:
        return {}
    return {kind: _copy_index_label_value(value) for kind, value in labels.items()}


def _normalize_index_labels(field: "Field", labels: Optional[Mapping]) -> dict:
    normalized = _copy_index_labels(labels)
    if not normalized:
        return {}

    kind_counts = Counter(index.kind for index in field.indices)
    for kind, value in normalized.items():
        expected = kind_counts.get(kind, 0)
        if expected <= 1 or value is None:
            continue
        if not isinstance(value, tuple):
            raise ValueError(
                f"Field {field.name!r} carries repeated index kind {kind!r}; "
                "provide labels as a tuple/list in slot order."
            )
        if len(value) != expected:
            raise ValueError(
                f"Field {field.name!r} carries {expected} indices of kind {kind!r}; "
                f"got {len(value)} labels."
            )
    return normalized


def _field_reference_text(target) -> str:
    if target is None:
        return "None"
    if isinstance(target, Field):
        return target.name
    return str(target)


def _occurrence_labels_from_call(
    field: "Field",
    positional_labels: tuple[object, ...],
    labels: Optional[Mapping],
) -> dict:
    if len(positional_labels) > len(field.indices):
        raise TypeError(
            f"Field {field.name!r} takes at most {len(field.indices)} index label(s), "
            f"got {len(positional_labels)}."
        )

    slot_labels = {slot: value for slot, value in enumerate(positional_labels)}
    explicit = field.unpack_slot_labels(labels) if labels is not None else {}
    for slot, value in explicit.items():
        if slot in slot_labels and slot_labels[slot] != value:
            raise TypeError(
                f"Field {field.name!r} received conflicting labels for slot {slot + 1}."
            )
        slot_labels[slot] = value
    return field.pack_slot_labels(slot_labels)


@dataclass(frozen=True)
class Field:
    """Particle field declaration (mirrors M$ClassesDescription)."""
    name: str
    spin: object
    self_conjugate: bool
    indices: tuple[IndexType, ...] = ()
    kind: Optional[str] = None
    statistics: Optional[Statistics] = None
    symbol: object = None
    conjugate_symbol: object = None
    mass: object = None
    quantum_numbers: Mapping[str, object] = field(default_factory=dict)
    ghost_of: object = None
    flavor_index: Optional[IndexType] = None
    class_members: tuple = ()

    def __post_init__(self):
        if self.ghost_of is not None:
            if self.kind is None:
                object.__setattr__(self, "kind", "ghost")
            elif self.kind != "ghost":
                raise ValueError(
                    f"Field {self.name!r} declares ghost_of={self.ghost_of!r} "
                    "but kind is not 'ghost'. Use GhostField(...) or kind='ghost'."
                )
        if self.kind is None:
            object.__setattr__(self, "kind", _infer_kind(self.spin))
        if self.statistics is None:
            object.__setattr__(self, "statistics", _infer_statistics(self.kind))
        if self.symbol is None:
            object.__setattr__(self, "symbol", S(self.name))
        if self.flavor_index is not None and not self.flavor_index.is_flavor:
            raise ValueError(
                f"Field {self.name!r} declares flavor_index={self.flavor_index.name!r}, "
                "but that index type is not marked as a flavor index."
            )
        if self.class_members and self.flavor_index is None:
            raise ValueError(
                f"Field {self.name!r} declares class_members but no flavor_index."
            )
        if self.class_members:
            flavor_slots = tuple(
                slot
                for slot, index in enumerate(self.indices)
                if index == self.flavor_index
            )
            if len(flavor_slots) != 1:
                raise ValueError(
                    f"Field {self.name!r} must carry exactly one slot of flavor_index "
                    f"{self.flavor_index.name!r} when class_members are declared."
                )
            if self.flavor_index.dimension is None:
                raise ValueError(
                    f"Field {self.name!r} uses flavor index {self.flavor_index.name!r} "
                    "for class_members, but that index type has no declared dimension."
                )
            if len(self.class_members) != self.flavor_index.dimension:
                raise ValueError(
                    f"Field {self.name!r} declares {len(self.class_members)} class member(s), "
                    f"but flavor index {self.flavor_index.name!r} has dimension "
                    f"{self.flavor_index.dimension}."
                )
            flavor_slot = flavor_slots[0]
            member_indices = self.indices[:flavor_slot] + self.indices[flavor_slot + 1 :]
            built_members: list[Field] = []
            for member in self.class_members:
                if isinstance(member, str):
                    member = self._build_class_member(member, member_indices)
                if not isinstance(member, Field):
                    raise TypeError(
                        f"Field {self.name!r} class_members must be strings or Field instances."
                    )
                if any(index.is_flavor for index in member.indices):
                    raise ValueError(
                        f"Field {self.name!r} class member {member.name!r} still carries a flavor index."
                    )
                if member.indices != member_indices:
                    raise ValueError(
                        f"Field {self.name!r} class member {member.name!r} must carry "
                        f"indices {member_indices!r}, got {member.indices!r}."
                    )
                if (
                    str(Fraction(member.spin)) != str(Fraction(self.spin))
                    or member.kind != self.kind
                    or member.statistics != self.statistics
                    or member.self_conjugate != self.self_conjugate
                ):
                    raise ValueError(
                        f"Field {self.name!r} class member {member.name!r} must match the "
                        "generic field spin/statistics/conjugation metadata."
                    )
                built_members.append(member)
            object.__setattr__(self, "class_members", tuple(built_members))

    def _build_class_member(
        self,
        name: str,
        member_indices: tuple[IndexType, ...],
    ) -> "Field":
        """Build one concrete class member from a name string.

        Members inherit the parent's spin, statistics, kind, self_conjugate,
        mass, and quantum_numbers, and carry the parent's indices minus the
        flavor index slot. The member's own `symbol`/`conjugate_symbol` default
        to ``S(name)``/``S(name + 'bar')`` (the latter only when not
        self-conjugate).
        """
        conjugate_symbol = (
            _default_conjugate_symbol(name) if not self.self_conjugate else None
        )
        return Field(
            name,
            spin=self.spin,
            self_conjugate=self.self_conjugate,
            indices=member_indices,
            kind=self.kind,
            statistics=self.statistics,
            symbol=S(name),
            conjugate_symbol=conjugate_symbol,
            mass=self.mass,
            quantum_numbers=dict(self.quantum_numbers),
        )

    def __hash__(self):
        quantum_numbers = tuple(
            sorted((str(key), str(value)) for key, value in self.quantum_numbers.items())
        )
        return hash((
            self.name,
            str(Fraction(self.spin)),
            self.self_conjugate,
            tuple(
                (
                    index.name,
                    index.kind,
                    index.prefix,
                    index.dimension,
                    index.is_flavor,
                )
                for index in self.indices
            ),
            self.kind,
            self.statistics,
            str(self.symbol),
            str(self.conjugate_symbol),
            str(self.mass),
            quantum_numbers,
            _field_reference_text(self.ghost_of),
            None
            if self.flavor_index is None
            else (
                self.flavor_index.name,
                self.flavor_index.kind,
                self.flavor_index.prefix,
                self.flavor_index.dimension,
                self.flavor_index.is_flavor,
            ),
            tuple(member.name for member in self.class_members),
        ))

    @property
    def is_ghost(self) -> bool:
        return self.kind == "ghost"

    @property
    def is_flavor_generic(self) -> bool:
        return bool(self.class_members)

    def flavor_index_slot(self) -> Optional[int]:
        if self.flavor_index is None:
            return None
        matches = self.index_positions(index=self.flavor_index)
        if len(matches) != 1:
            return None
        return matches[0]

    def class_member_for(self, flavor_value: int) -> "Field":
        if not self.class_members:
            raise ValueError(
                f"Field {self.name!r} has no class_members for flavor expansion."
            )
        if flavor_value < 1 or flavor_value > len(self.class_members):
            raise ValueError(
                f"Field {self.name!r} has no class member for flavor value {flavor_value}."
            )
        return self.class_members[flavor_value - 1]

    def role_for(self, conjugated: bool = False) -> FieldRole:
        """Return the interaction/external-leg role implied by this field slot."""
        if self.kind == "fermion":
            return ROLE_PSIBAR if conjugated else ROLE_PSI
        if self.is_ghost:
            return ROLE_GHOST_DAG if conjugated else ROLE_GHOST
        if self.kind == "vector":
            return ROLE_VECTOR
        if conjugated and not self.self_conjugate:
            return ROLE_SCALAR_DAG
        return ROLE_SCALAR

    def species_for(self, conjugated: bool = False):
        """Return the symbolic species used by the contraction engine."""
        if conjugated and not self.self_conjugate:
            return self.conjugate_symbol or S(self.name + "bar")
        return self.symbol

    def index_positions(
        self,
        *,
        kind: Optional[str] = None,
        index: Optional[IndexType] = None,
    ) -> tuple[int, ...]:
        """Return field-index slots matching one kind or one concrete index type."""
        if (kind is None) == (index is None):
            raise ValueError("Provide exactly one of kind=... or index=...")
        if kind is not None:
            return tuple(slot for slot, item in enumerate(self.indices) if item.kind == kind)
        return tuple(slot for slot, item in enumerate(self.indices) if item == index)

    def index_kind_count(self, kind: str) -> int:
        return sum(1 for index in self.indices if index.kind == kind)

    def pack_slot_labels(self, slot_labels: Mapping[int, object]) -> dict:
        """Pack slot-indexed labels into the engine's kind-keyed label format.

        For repeated kinds, the packed tuple is **ordinal-stable**: its length is
        exactly the number of slots of that kind, using ``None`` placeholders for
        unspecified slots.
        """
        kind_counts = Counter(index.kind for index in self.indices)
        kind_slots: dict[str, list[int]] = {}
        for slot, index in enumerate(self.indices):
            kind_slots.setdefault(index.kind, []).append(slot)

        packed: dict[str, object] = {}
        for kind, slots in kind_slots.items():
            count = kind_counts[kind]
            if count <= 1:
                # Single-slot kinds stay as a single label when present.
                slot = slots[0]
                if slot in slot_labels and slot_labels[slot] is not None:
                    packed[kind] = slot_labels[slot]
                continue

            # Repeated kinds: preserve slot ordinal with None placeholders.
            labels = [None] * count
            for ordinal, slot in enumerate(slots):
                if slot in slot_labels:
                    labels[ordinal] = slot_labels[slot]
            packed[kind] = tuple(labels)

        return packed

    def unpack_slot_labels(self, labels: Optional[Mapping]) -> dict[int, object]:
        """Invert ``pack_slot_labels`` back to slot-indexed labels."""
        normalized = _normalize_index_labels(self, labels)
        if not normalized:
            return {}

        unpacked: dict[int, object] = {}
        for kind, value in normalized.items():
            slots = self.index_positions(kind=kind)
            if len(slots) == 1:
                label = value
                if isinstance(label, tuple):
                    if len(label) != 1:
                        raise ValueError(
                            f"Field {self.name!r} carries one index of kind {kind!r}; "
                            f"got {len(label)} labels."
                        )
                    label = label[0]
                if label is not None:
                    unpacked[slots[0]] = label
                continue

            if not isinstance(value, tuple):
                raise ValueError(
                    f"Field {self.name!r} carries repeated index kind {kind!r}; "
                    "provide labels as a tuple/list in slot order."
                )
            for slot, label in zip(slots, value):
                if label is not None:
                    unpacked[slot] = label

        return unpacked

    def occurrence(self, *, conjugated: bool = False, labels: Optional[dict] = None):
        """Internal helper: create one backend FieldOccurrence."""
        from .interactions import FieldOccurrence

        return FieldOccurrence(
            field=self,
            conjugated=conjugated,
            labels=labels,
        )

    @property
    def bar(self) -> "ConjugateField":
        """Return a conjugated-field marker for use in ``feynman_rule()``."""
        return ConjugateField(self)

    def leg(
        self,
        momentum,
        *,
        conjugated: bool = False,
        species=None,
        spin=None,
        labels: Optional[dict] = None,
    ):
        """Internal helper: create one backend ExternalLeg."""
        from .interactions import ExternalLeg

        return ExternalLeg(
            field=self,
            momentum=momentum,
            conjugated=conjugated,
            species=species,
            spin=spin,
            labels=labels,
        )

    def __mul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__mul__(other)

    def __rmul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__rmul__(other)

    def __add__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__add__(other)

    def __radd__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__radd__(other)

    def __call__(self, *labels, conjugated: bool = False, index_labels: Optional[Mapping] = None):
        """Public shorthand for one labeled field factor in a declaration.

        Positional labels follow ``field.indices`` order exactly. For example,
        if ``field.indices == (SPINOR_INDEX, COLOR_FUND_INDEX)``, then
        ``field(s, c)`` means spinor label ``s`` and color label ``c``.
        Prefer ``index_labels={...}`` when you want the call site to stay
        independent of the field's declared slot order.

        Examples:
        - ``Photon(mu)`` for a vector field with one Lorentz index
        - ``Gluon(mu, a)`` for a field with ``(Lorentz, adjoint)`` slots
        - ``GhostG(a)`` for an adjoint ghost
        """
        return self.occurrence(
            conjugated=conjugated,
            labels=_occurrence_labels_from_call(self, labels, index_labels),
        )


# ---------------------------------------------------------------------------
# ConjugateField  (conjugation marker for the Lagrangian API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConjugateField:
    """Lightweight marker for a conjugated field in ``feynman_rule()``."""
    field: Field

    def __mul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__mul__(other)

    def __rmul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__rmul__(other)

    def __add__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__add__(other)

    def __radd__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__radd__(other)

    def __call__(self, *labels, index_labels: Optional[Mapping] = None):
        """Public shorthand for one conjugated labeled field factor.

        Positional labels follow ``field.indices`` order exactly, just like
        ``Field.__call__``.
        """
        return self.field.occurrence(
            conjugated=True,
            labels=_occurrence_labels_from_call(self.field, labels, index_labels),
        )


def GhostField(
    name: str,
    *,
    ghost_of=None,
    spin=0,
    self_conjugate: bool = False,
    indices: tuple[IndexType, ...] = (),
    symbol=None,
    conjugate_symbol=None,
    mass=None,
    quantum_numbers: Optional[Mapping[str, object]] = None,
) -> Field:
    """Typed ghost-field declaration.

    This keeps the public declaration closer to FeynRules-style model files:
    the field is marked as a ghost, remembers which gauge boson it belongs to,
    and defaults to ghost number ``+1`` unless the caller overrides it.
    """
    numbers = dict(quantum_numbers or {})
    numbers.setdefault("GhostNumber", 1)
    return Field(
        name,
        spin=spin,
        kind="ghost",
        self_conjugate=self_conjugate,
        indices=indices,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        mass=mass,
        quantum_numbers=numbers,
        ghost_of=ghost_of,
    )
