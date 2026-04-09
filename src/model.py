"""
FeynRules-style model declarations for the Symbolica vertex engine.

This module provides thin, declarative objects that mirror the structure of
a FeynRules .fr model file:

  M$GaugeGroups  ->  GaugeGroup
  IndexRange     ->  IndexType
  M$Classes      ->  Field
  M$Parameters   ->  Parameter
  Lagrangian     ->  InteractionTerm  (one monomial at a time)

The key design rule: adding a new index type or field should require only
new metadata declarations, never new conditionals in the engine.

InteractionTerm.to_vertex_kwargs() produces the parallel-list dict consumed
by vertex_factor() in model_symbolica.py, so the engine stays untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Literal, Mapping, Optional, Sequence

from symbolica import S

from spenso_structures import (
    BISPINOR,
    COLOR_ADJ,
    COLOR_FUND,
    LORENTZ,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    SPINOR_KIND,
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
    prefix: str = ""

    def __post_init__(self):
        if not self.prefix:
            object.__setattr__(self, "prefix", self.kind[:1])


SPINOR_INDEX = IndexType("Spinor", BISPINOR, SPINOR_KIND, prefix="i")
LORENTZ_INDEX = IndexType("Lorentz", LORENTZ, LORENTZ_KIND, prefix="mu")
COLOR_FUND_INDEX = IndexType("ColorFund", COLOR_FUND, COLOR_FUND_KIND, prefix="c")
COLOR_ADJ_INDEX = IndexType("ColorAdj", COLOR_ADJ, COLOR_ADJ_KIND, prefix="a")


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

    def slot_for(self, field: "Field") -> Optional[int]:
        """Backward-compatible helper: resolve a unique slot or None.

        This preserves the original strict semantics (ambiguity -> error) unless
        ``slot`` is explicitly set.
        """
        slots = self.slots_for(field)
        if not slots:
            return None
        if len(slots) != 1:
            raise ValueError(
                f"GaugeRepresentation for index {self.index.name!r} resolved to {len(slots)} slots; "
                "use slots_for(...) or set slot=... for a unique slot."
            )
        return slots[0]


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
class Parameter:
    """Model parameter (coupling constant, mass, Yukawa matrix, ...)."""
    name: str
    symbol: object = None
    indices: tuple[IndexType, ...] = ()
    complex_param: bool = False
    internal: bool = True
    value: object = None

    def __post_init__(self):
        if self.symbol is None:
            object.__setattr__(self, "symbol", S(self.name))


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


def _copy_index_labels(labels: Mapping | None) -> dict:
    if not labels:
        return {}
    return {kind: _copy_index_label_value(value) for kind, value in labels.items()}


def _normalize_index_labels(field: "Field", labels: Mapping | None) -> dict:
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

    def __post_init__(self):
        if self.kind is None:
            object.__setattr__(self, "kind", _infer_kind(self.spin))
        if self.statistics is None:
            object.__setattr__(self, "statistics", _infer_statistics(self.kind))
        if self.symbol is None:
            object.__setattr__(self, "symbol", S(self.name))

    def role_for(self, conjugated: bool = False) -> FieldRole:
        """Return the interaction/external-leg role implied by this field slot."""
        if self.kind == "fermion":
            return ROLE_PSIBAR if conjugated else ROLE_PSI
        if self.kind == "ghost":
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

    def occurrence(self, *, conjugated: bool = False, labels: dict | None = None):
        """Create a FieldOccurrence of this field in an interaction term."""
        return FieldOccurrence(
            field=self,
            conjugated=conjugated,
            labels=_normalize_index_labels(self, labels),
        )

    def leg(
        self,
        momentum,
        *,
        conjugated: bool = False,
        species=None,
        spin=None,
        labels: dict | None = None,
    ):
        """Create an ExternalLeg for this field."""
        return ExternalLeg(
            field=self,
            momentum=momentum,
            conjugated=conjugated,
            species=species,
            spin=spin,
            labels=_normalize_index_labels(self, labels),
        )


# ---------------------------------------------------------------------------
# FieldOccurrence  (a field slot inside an interaction term)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldOccurrence:
    """One field factor inside an interaction monomial, with index labels."""
    field: Field
    conjugated: bool = False
    labels: dict = field(default_factory=dict)

    @property
    def species(self):
        return self.field.species_for(self.conjugated)

    @property
    def role(self) -> FieldRole:
        return self.field.role_for(self.conjugated)

    @property
    def spinor_label(self):
        spinor = self.labels.get(SPINOR_KIND)
        if isinstance(spinor, tuple):
            return spinor[0] if spinor else None
        return spinor


# ---------------------------------------------------------------------------
# ExternalLeg  (an external particle in a vertex evaluation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalLeg:
    """External particle leg for vertex evaluation."""
    field: Field
    momentum: object
    conjugated: bool = False
    species: object = None
    spin: object = None
    labels: dict = field(default_factory=dict)

    @property
    def effective_species(self):
        if self.species is not None:
            return self.species
        return self.field.species_for(self.conjugated)

    @property
    def role(self) -> FieldRole:
        return self.field.role_for(self.conjugated)


# ---------------------------------------------------------------------------
# DerivativeAction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DerivativeAction:
    """A derivative acting on one field slot in the interaction."""
    target: int
    lorentz_index: object


# ---------------------------------------------------------------------------
# InteractionTerm
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionTerm:
    """One interaction monomial (mirrors one Lagrangian term).

    This is the bridge between FeynRules-style declarations and the
    contraction engine.  ``to_vertex_kwargs()`` produces the parallel-list
    dict that ``vertex_factor()`` consumes.
    """
    coupling: object
    fields: tuple[FieldOccurrence, ...]
    derivatives: tuple[DerivativeAction, ...] = ()
    label: str = ""

    @property
    def statistics(self) -> Statistics:
        for occ in self.fields:
            if occ.field.statistics == "fermion":
                return "fermion"
        return "boson"

    def to_vertex_kwargs(self, external_legs: Sequence[ExternalLeg]) -> dict:
        """Generate the dict consumed by vertex_factor().

        This is the ONLY bridge between the model layer and the engine.
        """
        n = len(self.fields)
        if len(external_legs) != n:
            raise ValueError(
                f"Need {n} external legs to match {n} field occurrences, "
                f"got {len(external_legs)}."
            )

        alphas = [occ.species for occ in self.fields]
        betas = [leg.effective_species for leg in external_legs]
        ps = [leg.momentum for leg in external_legs]

        field_roles = [occ.role for occ in self.fields]
        leg_roles = [leg.role for leg in external_legs]

        field_index_labels = [_copy_index_labels(occ.labels) for occ in self.fields]
        leg_index_labels = [_copy_index_labels(leg.labels) for leg in external_legs]

        leg_spins = [leg.spin for leg in external_legs]

        derivative_indices = [d.lorentz_index for d in self.derivatives]
        derivative_targets = [d.target for d in self.derivatives]

        return dict(
            coupling=self.coupling,
            alphas=alphas,
            betas=betas,
            ps=ps,
            statistics=self.statistics,
            field_roles=field_roles,
            leg_roles=leg_roles,
            field_index_labels=field_index_labels,
            leg_index_labels=leg_index_labels,
            leg_spins=leg_spins,
            derivative_indices=derivative_indices,
            derivative_targets=derivative_targets,
        )


# ---------------------------------------------------------------------------
# Convention-fixed kinetic terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiracKineticTerm:
    """Model-level declaration for ``psibar i gamma^mu D_mu psi``.

    The current compiler expands only the gauge-interaction part of this term.
    If ``gauge_group`` is omitted, the compiler infers the unique applicable
    gauge group from the model metadata.
    """
    field: object
    gauge_group: object = None
    coefficient: object = 1
    label: str = ""


@dataclass(frozen=True)
class ComplexScalarKineticTerm:
    """Model-level declaration for ``(D_mu phi)^dagger (D^mu phi)``.

    The current compiler expands only the gauge-interaction part of this term.
    If ``gauge_group`` is omitted, the compiler infers the unique applicable
    gauge group from the model metadata.
    """
    field: object
    gauge_group: object = None
    coefficient: object = 1
    label: str = ""


@dataclass(frozen=True)
class GaugeKineticTerm:
    """Model-level declaration for ``-1/4 F_{mu nu} F^{mu nu}``.

    ``gauge_group`` is required because the gauge field and non-abelian
    structure constants are properties of the group declaration, not of a
    separate matter field.
    """
    gauge_group: object
    coefficient: object = 1
    label: str = ""


@dataclass(frozen=True)
class GaugeFixingTerm:
    """Model-level declaration for ``-(1/2 xi) (partial.A)^2``.

    This covers the ordinary unbroken linear covariant gauge-fixing term for one
    declared gauge group. ``xi`` is the usual gauge-fixing parameter.
    """
    gauge_group: object
    xi: object = 1
    coefficient: object = 1
    label: str = ""


@dataclass(frozen=True)
class GhostTerm:
    """Model-level declaration for the ordinary Faddeev-Popov ghost sector.

    The current implementation covers the unbroken non-abelian linear-covariant
    gauge case. The corresponding ghost field is resolved from the parent gauge
    group's ``ghost_field`` metadata.
    """
    gauge_group: object
    coefficient: object = 1
    label: str = ""


CovariantTerm = DiracKineticTerm | ComplexScalarKineticTerm


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Top-level model container (mirrors the full .fr file).

    The model layer stores declarations.  The actual vertex evaluation still
    happens in ``model_symbolica.py`` after these declarations are translated
    into ``InteractionTerm`` objects and then into engine kwargs.
    """
    name: str = ""
    gauge_groups: tuple[GaugeGroup, ...] = ()
    fields: tuple[Field, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    interactions: tuple[InteractionTerm, ...] = ()
    covariant_terms: tuple[CovariantTerm, ...] = ()
    gauge_kinetic_terms: tuple[GaugeKineticTerm, ...] = ()
    gauge_fixing_terms: tuple[GaugeFixingTerm, ...] = ()
    ghost_terms: tuple[GhostTerm, ...] = ()

    def find_field(self, target) -> Optional[Field]:
        """Resolve a field by object identity, declaration name, or symbol."""
        if isinstance(target, Field):
            for field in self.fields:
                if field is target:
                    return field
            return None
        if target is None:
            return None

        target_text = str(target)
        for field in self.fields:
            if field.name == target_text:
                return field
            if str(field.symbol) == target_text:
                return field
            if field.conjugate_symbol is not None and str(field.conjugate_symbol) == target_text:
                return field
        return None

    def find_gauge_group(self, target) -> Optional[GaugeGroup]:
        """Resolve a gauge group by object identity or declaration name."""
        if isinstance(target, GaugeGroup):
            for gauge_group in self.gauge_groups:
                if gauge_group is target:
                    return gauge_group
            return None
        if target is None:
            return None

        target_text = str(target)
        for gauge_group in self.gauge_groups:
            if gauge_group.name == target_text:
                return gauge_group
        return None

    def gauge_boson_field(self, gauge_group: GaugeGroup) -> Field:
        """Resolve the gauge boson field declared for a gauge group."""
        if isinstance(gauge_group.gauge_boson, Field):
            field = self.find_field(gauge_group.gauge_boson)
            if field is None:
                raise ValueError(
                    f"Gauge group {gauge_group.name!r} requires gauge boson "
                    f"{gauge_group.gauge_boson.name!r} to be declared in model.fields."
                )
            return field

        field = self.find_field(gauge_group.gauge_boson)
        if field is None:
            raise ValueError(
                f"Could not resolve gauge boson {gauge_group.gauge_boson!r} "
                f"for gauge group {gauge_group.name!r}."
            )
        return field
