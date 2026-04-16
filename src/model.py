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
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Literal, Mapping, Optional, Sequence
import warnings

from symbolica import Expression, S

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
from decl_lagrangian_lowering import (
    expr_equal as _expr_equal_impl,
    lower_field_strength_monomial as _lower_field_strength_monomial_impl,
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

    @property
    def bar(self) -> "ConjugateField":
        """Return a conjugated-field marker for use in ``Lagrangian.feynman_rule()``."""
        return ConjugateField(self)

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

    def __mul__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__mul__(other)

    def __rmul__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__rmul__(other)

    def __add__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__add__(other)

    def __radd__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self)).__radd__(other)


# ---------------------------------------------------------------------------
# ConjugateField  (conjugation marker for the Lagrangian API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConjugateField:
    """Lightweight marker for a conjugated field in ``Lagrangian.feynman_rule()``."""
    field: Field

    def __mul__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__mul__(other)

    def __rmul__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__rmul__(other)

    def __add__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__add__(other)

    def __radd__(self, other):
        return _DeclaredMonomial.from_factor(_FieldFactor(self.field, conjugated=True)).__radd__(other)


# ---------------------------------------------------------------------------
# Declarative Lagrangian factors  (CovD / Gamma / FieldStrength DSL)
# ---------------------------------------------------------------------------


class _DeclaredFactorMixin:
    def __mul__(self, other):
        return _DeclaredMonomial.from_factor(self).__mul__(other)

    def __rmul__(self, other):
        return _DeclaredMonomial.from_factor(self).__rmul__(other)

    def __add__(self, other):
        return _DeclaredMonomial.from_factor(self).__add__(other)

    def __radd__(self, other):
        return _DeclaredMonomial.from_factor(self).__radd__(other)


@dataclass(frozen=True)
class _FieldFactor(_DeclaredFactorMixin):
    field: Field
    conjugated: bool = False

    def __str__(self):
        if self.conjugated and not self.field.self_conjugate:
            return f"{self.field.name}.bar"
        return self.field.name


@dataclass(frozen=True)
class CovariantDerivativeFactor(_DeclaredFactorMixin):
    field: Field
    lorentz_index: object
    conjugated: bool = False

    @property
    def bar(self) -> "CovariantDerivativeFactor":
        if self.field.self_conjugate:
            return self
        return CovariantDerivativeFactor(
            field=self.field,
            lorentz_index=self.lorentz_index,
            conjugated=not self.conjugated,
        )

    def __str__(self):
        base = f"{self.field.name}.bar" if self.conjugated and not self.field.self_conjugate else self.field.name
        return f"CovD({base}, {self.lorentz_index})"


@dataclass(frozen=True)
class PartialDerivativeFactor(_DeclaredFactorMixin):
    field: Field
    lorentz_indices: tuple[object, ...]
    conjugated: bool = False

    @property
    def bar(self) -> "PartialDerivativeFactor":
        if self.field.self_conjugate:
            return self
        return PartialDerivativeFactor(
            field=self.field,
            lorentz_indices=self.lorentz_indices,
            conjugated=not self.conjugated,
        )

    def __str__(self):
        base = f"{self.field.name}.bar" if self.conjugated and not self.field.self_conjugate else self.field.name
        indices = ", ".join(str(index) for index in self.lorentz_indices)
        return f"PartialD({base}, {indices})"


@dataclass(frozen=True)
class GammaFactor(_DeclaredFactorMixin):
    lorentz_index: object

    def __str__(self):
        return f"Gamma({self.lorentz_index})"


@dataclass(frozen=True)
class FieldStrengthFactor(_DeclaredFactorMixin):
    gauge_group: object
    left_index: object
    right_index: object

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        return f"FieldStrength({group_name}, {self.left_index}, {self.right_index})"


@dataclass(frozen=True)
class GaugeFixingDeclaration:
    gauge_group: object
    xi: object = 1
    coefficient: object = 1
    label: str = ""

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

    def __mul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=self.coefficient * other)
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=other * self.coefficient)
        return NotImplemented

    def __neg__(self):
        return replace(self, coefficient=-self.coefficient)

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        body = f"GaugeFixing({group_name}, xi={self.xi})"
        if self.coefficient == 1:
            return body
        return f"{self.coefficient} * {body}"


@dataclass(frozen=True)
class GhostLagrangianDeclaration:
    gauge_group: object
    coefficient: object = 1
    label: str = ""

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

    def __mul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=self.coefficient * other)
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=other * self.coefficient)
        return NotImplemented

    def __neg__(self):
        return replace(self, coefficient=-self.coefficient)

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        body = f"GhostLagrangian({group_name})"
        if self.coefficient == 1:
            return body
        return f"{self.coefficient} * {body}"


def _is_decl_scalar(value) -> bool:
    return not isinstance(
        value,
        (
            Field,
            ConjugateField,
            _FieldFactor,
            CovariantDerivativeFactor,
            PartialDerivativeFactor,
            GammaFactor,
            FieldStrengthFactor,
            _DeclaredMonomial,
            DeclaredLagrangian,
            InteractionTerm,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
            Lagrangian,
        ),
    )


def _coerce_decl_factor(value):
    if isinstance(value, Field):
        return _FieldFactor(value)
    if isinstance(value, ConjugateField):
        return _FieldFactor(value.field, conjugated=True)
    if isinstance(value, (_FieldFactor, CovariantDerivativeFactor, PartialDerivativeFactor, GammaFactor, FieldStrengthFactor)):
        return value
    return None


@dataclass(frozen=True)
class _DeclaredMonomial:
    coefficient: object = 1
    factors: tuple[object, ...] = ()

    @classmethod
    def from_factor(cls, factor) -> "_DeclaredMonomial":
        return cls(coefficient=1, factors=(factor,))

    def __mul__(self, other):
        if isinstance(other, _DeclaredMonomial):
            return _DeclaredMonomial(
                coefficient=self.coefficient * other.coefficient,
                factors=self.factors + other.factors,
            )
        factor = _coerce_decl_factor(other)
        if factor is not None:
            return _DeclaredMonomial(
                coefficient=self.coefficient,
                factors=self.factors + (factor,),
            )
        if _is_decl_scalar(other):
            return _DeclaredMonomial(
                coefficient=self.coefficient * other,
                factors=self.factors,
            )
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return _DeclaredMonomial(
                coefficient=other * self.coefficient,
                factors=self.factors,
            )
        factor = _coerce_decl_factor(other)
        if factor is not None:
            return _DeclaredMonomial(
                coefficient=self.coefficient,
                factors=(factor,) + self.factors,
            )
        return NotImplemented

    def __neg__(self):
        return _DeclaredMonomial(coefficient=-self.coefficient, factors=self.factors)

    def __add__(self, other):
        return DeclaredLagrangian.from_item(self).__add__(other)

    def __radd__(self, other):
        return DeclaredLagrangian.from_item(self).__radd__(other)

    def __str__(self):
        pieces = [str(factor) for factor in self.factors]
        if self.coefficient != 1 or not pieces:
            pieces = [str(self.coefficient)] + pieces
        return " * ".join(pieces)


def CovD(field, lorentz_index) -> CovariantDerivativeFactor:
    """Declarative covariant derivative factor for ``DeclaredLagrangian``.

    Accepts ``Field``, ``Field.bar``, or ``(Field, bool)`` and can be used in
    expressions such as ``I * Psi.bar * Gamma(mu) * CovD(Psi, mu)``.
    """
    field_obj, conjugated = _parse_field_arg(field)
    return CovariantDerivativeFactor(
        field=field_obj,
        lorentz_index=lorentz_index,
        conjugated=conjugated,
    )


def PartialD(field, lorentz_index) -> PartialDerivativeFactor:
    """Declarative partial derivative factor for local derivative monomials.

    Accepts ``Field``, ``Field.bar``, ``(Field, bool)``, or another
    ``PartialD(...)`` factor to build higher derivatives.
    """
    if isinstance(field, PartialDerivativeFactor):
        return PartialDerivativeFactor(
            field=field.field,
            lorentz_indices=field.lorentz_indices + (lorentz_index,),
            conjugated=field.conjugated,
        )
    field_obj, conjugated = _parse_field_arg(field)
    return PartialDerivativeFactor(
        field=field_obj,
        lorentz_indices=(lorentz_index,),
        conjugated=conjugated,
    )


def Gamma(lorentz_index) -> GammaFactor:
    """Declarative gamma-matrix placeholder for ``DeclaredLagrangian``."""
    return GammaFactor(lorentz_index=lorentz_index)


def FieldStrength(gauge_group, left_index, right_index) -> FieldStrengthFactor:
    """Declarative field-strength placeholder for ``DeclaredLagrangian``."""
    return FieldStrengthFactor(
        gauge_group=gauge_group,
        left_index=left_index,
        right_index=right_index,
    )


def GaugeFixing(gauge_group, *, xi=1, coefficient=1, label="") -> GaugeFixingDeclaration:
    """Declarative ordinary gauge-fixing wrapper for ``DeclaredLagrangian``."""
    return GaugeFixingDeclaration(
        gauge_group=gauge_group,
        xi=xi,
        coefficient=coefficient,
        label=label,
    )


def GhostLagrangian(gauge_group, *, coefficient=1, label="") -> GhostLagrangianDeclaration:
    """Declarative Faddeev-Popov ghost-sector wrapper for ``DeclaredLagrangian``."""
    return GhostLagrangianDeclaration(
        gauge_group=gauge_group,
        coefficient=coefficient,
        label=label,
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

    def __add__(self, other):
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(self, other))
        if isinstance(other, Lagrangian):
            return Lagrangian(terms=(self,) + other.terms)
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=(self,) + decl_terms)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return Lagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other, self))
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=decl_terms + (self,))
        return NotImplemented

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
# Lagrangian  (FeynRules-style vertex extraction)
# ---------------------------------------------------------------------------

def _species_key(species) -> str:
    """Hashable key for a Symbolica species expression."""
    if hasattr(species, "to_canonical_string"):
        return species.to_canonical_string()
    return str(species)


def _field_match_key(field_obj: Field, conjugated: bool) -> tuple:
    """Stable key for matching external fields to interaction slots.

    Match on the declared field metadata rather than only the species symbol,
    so distinct fields that happen to share a symbol do not collide. For
    self-conjugate fields the conjugation flag is irrelevant.
    """
    effective_conjugated = bool(conjugated and not field_obj.self_conjugate)
    return (
        field_obj.name,
        str(Fraction(field_obj.spin)),
        field_obj.kind,
        field_obj.statistics,
        tuple((index.name, index.kind, index.prefix) for index in field_obj.indices),
        _species_key(field_obj.symbol),
        _species_key(field_obj.conjugate_symbol) if field_obj.conjugate_symbol is not None else None,
        effective_conjugated,
    )


def _parse_field_arg(arg) -> tuple[Field, bool]:
    """Normalize a feynman_rule field argument to (Field, conjugated)."""
    if isinstance(arg, tuple) and len(arg) == 2:
        field_obj, conjugated = arg
        if isinstance(field_obj, Field) and isinstance(conjugated, bool):
            return (field_obj, conjugated)
    if isinstance(arg, ConjugateField):
        return (arg.field, True)
    if isinstance(arg, Field):
        return (arg, False)
    raise TypeError(
        "Expected Field, (Field, bool), or Field.bar (ConjugateField), "
        f"got {type(arg).__name__}"
    )


def _auto_leg_labels(field_obj: Field, counter: list[int]) -> dict:
    """Generate sequential index labels i1, i2, ... for all index slots on one leg."""
    kind_counts = Counter(idx.kind for idx in field_obj.indices)
    kind_ordinals: dict[str, int] = {}
    labels: dict[str, object] = {}

    for idx in field_obj.indices:
        label = S(f"i{counter[0]}")
        counter[0] += 1

        ordinal = kind_ordinals.get(idx.kind, 0)
        kind_ordinals[idx.kind] = ordinal + 1
        count = kind_counts[idx.kind]

        if count > 1:
            if idx.kind not in labels:
                labels[idx.kind] = [None] * count
            labels[idx.kind][ordinal] = label
        else:
            labels[idx.kind] = label

    for kind in list(labels):
        if isinstance(labels[kind], list):
            labels[kind] = tuple(labels[kind])

    return labels


def _term_matches_fields(
    term: InteractionTerm,
    parsed_fields: Sequence[tuple[Field, bool]],
) -> bool:
    """Check if an InteractionTerm's field content matches the external fields."""
    if len(term.fields) != len(parsed_fields):
        return False
    term_species = Counter(
        _field_match_key(occ.field, occ.conjugated) for occ in term.fields
    )
    ext_species = Counter(
        _field_match_key(fld, conj) for fld, conj in parsed_fields
    )
    return term_species == ext_species


@dataclass
class Lagrangian:
    """Collection of interaction terms with a single ``feynman_rule()`` entry point.

    Mirrors the FeynRules workflow: declare Lagrangian pieces, compose with
    ``+``, then extract vertex factors by specifying external fields.

    Example::

        L = LGauge + LFermions + LHiggs
        vertex = L.feynman_rule(Phi.bar, Phi, A)
    """
    terms: tuple[InteractionTerm, ...] = ()

    def __add__(self, other):
        if isinstance(other, Lagrangian):
            return Lagrangian(terms=self.terms + other.terms)
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=self.terms + (other,))
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other,) + self.terms)
        return NotImplemented

    def feynman_rule(self, *fields, momenta=None, simplify=True):
        """Compute the Feynman vertex rule for the given external fields.

        Conventions:
        - leg order = argument order
        - momenta default to q1, q2, q3, ...
        - open indices are labeled i1, i2, i3, ... sequentially across legs

        Parameters
        ----------
        *fields : Field, tuple[Field, bool], or ConjugateField
            External fields in leg order.  Use ``field.bar`` or ``(field, True)``
            for conjugated fields (e.g. ``Phi.bar``).
        momenta : list of expressions, optional
            Override the default q1, q2, ... momentum assignment.  Each entry
            can be an algebraic expression (e.g. ``p3 - p6``).
        simplify : bool
            If True (default), apply ``simplify_vertex`` to the result.

        Returns
        -------
        Expression
            The summed, stripped Feynman vertex factor with ``(2 pi)^d Delta``
            momentum conservation.
        """
        from model_symbolica import simplify_vertex, vertex_factor
        from symbolica import Expression

        parsed = [_parse_field_arg(f) for f in fields]
        n = len(parsed)
        if n == 0:
            raise ValueError("At least one external field is required.")

        if momenta is None:
            momenta_list = [S(f"q{k + 1}") for k in range(n)]
        else:
            momenta_list = list(momenta)
        if len(momenta_list) != n:
            raise ValueError(f"Expected {n} momenta, got {len(momenta_list)}.")

        idx_counter = [1]
        legs = []
        for k, (fld, conj) in enumerate(parsed):
            labels = _auto_leg_labels(fld, idx_counter)
            legs.append(ExternalLeg(
                field=fld,
                momentum=momenta_list[k],
                conjugated=conj,
                labels=labels,
            ))

        matching = [t for t in self.terms if _term_matches_fields(t, parsed)]
        if not matching:
            desc = ", ".join(
                f"{fld.name}{'bar' if conj else ''}" for fld, conj in parsed
            )
            raise ValueError(f"No matching interaction terms for: {desc}")

        x = S("x_")
        d = S("d")
        total = Expression.num(0)
        for term in matching:
            total += vertex_factor(
                interaction=term,
                external_legs=legs,
                x=x,
                d=d,
                strip_externals=True,
                include_delta=True,
            )

        if simplify:
            species_map = None
            unique_species = list(dict.fromkeys(
                fld.species_for(conj) for fld, conj in parsed
            ))
            if len(unique_species) > 1:
                species_map = {sp: sp for sp in unique_species}
            total = simplify_vertex(total, species_map=species_map, external_legs=legs)

        return total


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

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))


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

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))


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

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))


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

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))


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

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))


def _match_covariant_monomial(
    term: _DeclaredMonomial,
) -> tuple[DiracKineticTerm | ComplexScalarKineticTerm, tuple[tuple[object, bool], ...]] | None:
    field_factors = [factor for factor in term.factors if isinstance(factor, _FieldFactor)]
    gamma_factors = [factor for factor in term.factors if isinstance(factor, GammaFactor)]
    covd_factors = [factor for factor in term.factors if isinstance(factor, CovariantDerivativeFactor)]
    if len(term.factors) != len(field_factors) + len(gamma_factors) + len(covd_factors):
        return None

    if len(gamma_factors) == 1 and len(covd_factors) == 1:
        gamma_factor = gamma_factors[0]
        covd_factor = covd_factors[0]
        if (
            covd_factor.field.kind == "fermion"
            and not covd_factor.conjugated
            and gamma_factor.lorentz_index == covd_factor.lorentz_index
        ):
            core_slot = next(
                (
                    idx
                    for idx, factor in enumerate(field_factors)
                    if factor.field is covd_factor.field and factor.conjugated
                ),
                None,
            )
            if core_slot is not None:
                normalized = term.coefficient / Expression.I
                if not _expr_equal_impl(Expression.I * normalized, term.coefficient):
                    return None
                core = DiracKineticTerm(field=covd_factor.field, coefficient=normalized)
                spectators = tuple(
                    (factor.field, factor.conjugated)
                    for idx, factor in enumerate(field_factors)
                    if idx != core_slot
                )
                return core, spectators

    if len(gamma_factors) == 0 and len(covd_factors) == 2:
        left, right = covd_factors
        if (
            left.field is right.field
            and left.field.kind == "scalar"
            and not left.field.self_conjugate
            and left.lorentz_index == right.lorentz_index
            and {left.conjugated, right.conjugated} == {False, True}
        ):
            core = ComplexScalarKineticTerm(field=left.field, coefficient=term.coefficient)
            spectators = tuple((factor.field, factor.conjugated) for factor in field_factors)
            return core, spectators

    return None


def _generic_interaction_occurrence_labels(field_factors: Sequence[_FieldFactor]) -> list[dict]:
    labels = [{} for _ in field_factors]
    fermion_slots = [i for i, factor in enumerate(field_factors) if factor.field.kind == "fermion"]
    if len(fermion_slots) <= 2 or len(fermion_slots) % 2 != 0:
        return labels

    pair_slots = [fermion_slots[k : k + 2] for k in range(0, len(fermion_slots), 2)]
    for pair_number, slots in enumerate(pair_slots, start=1):
        left, right = (field_factors[slot] for slot in slots)
        if not left.conjugated or right.conjugated:
            return labels
        spinor_label = S(f"alpha_decl_{pair_number}")
        labels[slots[0]][SPINOR_KIND] = spinor_label
        labels[slots[1]][SPINOR_KIND] = spinor_label

    return labels


def _lower_local_interaction_monomial(term: _DeclaredMonomial):
    base_factors: list[_FieldFactor] = []
    derivative_indices: list[tuple[object, ...]] = []
    gamma_factors: list[GammaFactor] = []

    for factor in term.factors:
        if isinstance(factor, _FieldFactor):
            base_factors.append(factor)
            derivative_indices.append(())
            continue
        if isinstance(factor, PartialDerivativeFactor):
            base_factors.append(_FieldFactor(field=factor.field, conjugated=factor.conjugated))
            derivative_indices.append(tuple(factor.lorentz_indices))
            continue
        if isinstance(factor, GammaFactor):
            gamma_factors.append(factor)
            continue
        return None

    if not base_factors:
        return None

    coupling = term.coefficient
    if gamma_factors:
        if len(gamma_factors) != 1:
            return None
        fermion_slots = [i for i, factor in enumerate(base_factors) if factor.field.kind == "fermion"]
        if len(fermion_slots) != 2:
            return None
        left_slot, right_slot = fermion_slots
        left_factor = base_factors[left_slot]
        right_factor = base_factors[right_slot]
        if not left_factor.conjugated or right_factor.conjugated:
            return None

        from operators import psi_bar_gamma_psi

        occurrence_labels = [{} for _ in base_factors]
        i_bar = S("alpha_decl_bar")
        i_psi = S("alpha_decl")
        occurrence_labels[left_slot][SPINOR_KIND] = i_bar
        occurrence_labels[right_slot][SPINOR_KIND] = i_psi
        coupling *= psi_bar_gamma_psi(i_bar, i_psi, gamma_factors[0].lorentz_index)
    else:
        occurrence_labels = _generic_interaction_occurrence_labels(base_factors)

    return InteractionTerm(
        coupling=coupling,
        fields=tuple(
            factor.field.occurrence(
                conjugated=bool(factor.conjugated and not factor.field.self_conjugate),
                labels=occurrence_labels[idx],
            )
            for idx, factor in enumerate(base_factors)
        ),
        derivatives=tuple(
            DerivativeAction(target=idx, lorentz_index=lorentz_index)
            for idx, indices in enumerate(derivative_indices)
            for lorentz_index in indices
        ),
    )


def _lower_field_strength_monomial(term: _DeclaredMonomial):
    return _lower_field_strength_monomial_impl(
        term,
        field_strength_factor_cls=FieldStrengthFactor,
        gauge_kinetic_term_cls=GaugeKineticTerm,
        expression_module=Expression,
    )


def _validate_declared_monomial(term: _DeclaredMonomial):
    if _match_covariant_monomial(term) is not None:
        return
    if _lower_field_strength_monomial(term) is not None:
        return
    if _lower_local_interaction_monomial(term) is not None:
        return
    raise ValueError(
        "Unsupported declarative Lagrangian term. Supported canonical forms are: "
        "I * Psi.bar * Gamma(mu) * CovD(Psi, mu), "
        "CovD(Phi.bar, mu) * CovD(Phi, mu), "
        "either optionally multiplied by local spectator fields, "
        "-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu), "
        "local monomials built from fields, PartialD(...), and one optional Gamma(...), "
        "pure local field monomials like lam * Phi * Phi * Phi * Phi, "
        "plus explicit InteractionTerm / GaugeFixing(...) / GhostLagrangian(...) "
        "or the legacy GaugeFixingTerm / GhostTerm declarations."
    )
def _declared_source_terms_from_item(item):
    if isinstance(item, DeclaredLagrangian):
        return item.source_terms
    if isinstance(
        item,
        (
            _DeclaredMonomial,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
            InteractionTerm,
        ),
    ):
        return (item,)
    factor = _coerce_decl_factor(item)
    if factor is not None:
        return (_DeclaredMonomial.from_factor(factor),)
    return None


def _source_term_interaction(term) -> InteractionTerm | None:
    if isinstance(term, InteractionTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_local_interaction_monomial(term)
    return None


def _source_term_covariant_core(term) -> DiracKineticTerm | ComplexScalarKineticTerm | None:
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm)):
        return term
    if isinstance(term, _DeclaredMonomial):
        match = _match_covariant_monomial(term)
        if match is not None:
            core, _spectators = match
            return core
    return None


def _source_term_gauge_kinetic(term) -> GaugeKineticTerm | None:
    if isinstance(term, GaugeKineticTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_field_strength_monomial(term)
    return None


def _source_term_gauge_fixing(term) -> GaugeFixingTerm | None:
    if isinstance(term, GaugeFixingTerm):
        return term
    if isinstance(term, GaugeFixingDeclaration):
        return GaugeFixingTerm(
            gauge_group=term.gauge_group,
            xi=term.xi,
            coefficient=term.coefficient,
            label=term.label,
        )
    return None


def _source_term_ghost(term) -> GhostTerm | None:
    if isinstance(term, GhostTerm):
        return term
    if isinstance(term, GhostLagrangianDeclaration):
        return GhostTerm(
            gauge_group=term.gauge_group,
            coefficient=term.coefficient,
            label=term.label,
        )
    return None


def _source_term_needs_compilation(term) -> bool:
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm, GaugeKineticTerm, GaugeFixingTerm, GhostTerm)):
        return True
    if isinstance(term, (GaugeFixingDeclaration, GhostLagrangianDeclaration)):
        return True
    if isinstance(term, _DeclaredMonomial):
        if _match_covariant_monomial(term) is not None:
            return True
        if _lower_field_strength_monomial(term) is not None:
            return True
    return False


def _validate_declared_source_term(term):
    if isinstance(
        term,
        (
            InteractionTerm,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
        ),
    ):
        return
    if isinstance(term, _DeclaredMonomial):
        _validate_declared_monomial(term)
        return
    raise TypeError(f"Unsupported declared Lagrangian term type: {type(term)!r}")


@dataclass(frozen=True)
class DeclaredLagrangian:
    """User-facing declarative Lagrangian built from fields and covariant derivatives.

    This is a source-level declaration container. Terms are preserved in the
    original FeynRules-style form (`CovD(...)`, `FieldStrength(...)`,
    `GaugeFixing(...)`, `GhostLagrangian(...)`, etc.) and compiled one by one
    when `Model.lagrangian()` is built.
    """
    source_terms: tuple[object, ...] = ()

    @classmethod
    def from_item(cls, item) -> "DeclaredLagrangian":
        terms = _declared_source_terms_from_item(item)
        if terms is None:
            raise TypeError(f"Cannot build DeclaredLagrangian from {type(item).__name__}")
        return cls(source_terms=terms)

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=self.source_terms + terms)

    def __radd__(self, other):
        if other == 0:
            return self
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + self.source_terms)

    def __str__(self):
        if not self.source_terms:
            return "0"
        return " + ".join(str(term) for term in self.source_terms)


CovariantTerm = (
    DiracKineticTerm
    | ComplexScalarKineticTerm
)


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
    lagrangian_decl: DeclaredLagrangian | None = None
    covariant_terms: tuple[CovariantTerm, ...] = ()
    gauge_kinetic_terms: tuple[GaugeKineticTerm, ...] = ()
    gauge_fixing_terms: tuple[GaugeFixingTerm, ...] = ()
    ghost_terms: tuple[GhostTerm, ...] = ()

    def __post_init__(self):
        if any(
            getattr(self, attr_name)
            for attr_name in ("covariant_terms", "gauge_kinetic_terms", "gauge_fixing_terms", "ghost_terms")
        ):
            warnings.warn(
                "Model.covariant_terms, gauge_kinetic_terms, gauge_fixing_terms, and ghost_terms "
                "are deprecated. Prefer a unified lagrangian_decl=... declaration.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.lagrangian_decl is None:
            self.lagrangian_decl = DeclaredLagrangian()
        elif not isinstance(self.lagrangian_decl, DeclaredLagrangian):
            self.lagrangian_decl = DeclaredLagrangian.from_item(self.lagrangian_decl)
        for term in self.lagrangian_decl.source_terms:
            _validate_declared_source_term(term)

    def source_lagrangian_terms(self) -> tuple[object, ...]:
        """Return the user-facing declared Lagrangian terms in source form."""
        return (
            self.interactions
            + self.lagrangian_decl.source_terms
            + self.covariant_terms
            + self.gauge_kinetic_terms
            + self.gauge_fixing_terms
            + self.ghost_terms
        )

    def all_interactions(self) -> tuple[InteractionTerm, ...]:
        return self.interactions + tuple(
            interaction
            for term in self.lagrangian_decl.source_terms
            if (interaction := _source_term_interaction(term)) is not None
        )

    def all_covariant_terms(self) -> tuple[CovariantTerm, ...]:
        return self.covariant_terms + tuple(
            covariant
            for term in self.lagrangian_decl.source_terms
            if (covariant := _source_term_covariant_core(term)) is not None
        )

    def all_gauge_kinetic_terms(self) -> tuple[GaugeKineticTerm, ...]:
        return self.gauge_kinetic_terms + tuple(
            gauge_kinetic
            for term in self.lagrangian_decl.source_terms
            if (gauge_kinetic := _source_term_gauge_kinetic(term)) is not None
        )

    def all_gauge_fixing_terms(self) -> tuple[GaugeFixingTerm, ...]:
        return self.gauge_fixing_terms + tuple(
            gauge_fixing
            for term in self.lagrangian_decl.source_terms
            if (gauge_fixing := _source_term_gauge_fixing(term)) is not None
        )

    def all_ghost_terms(self) -> tuple[GhostTerm, ...]:
        return self.ghost_terms + tuple(
            ghost
            for term in self.lagrangian_decl.source_terms
            if (ghost := _source_term_ghost(term)) is not None
        )

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

    def lagrangian(self) -> Lagrangian:
        """Compile all declared terms and return a ``Lagrangian``.

        Local interaction monomials are lowered directly. Terms that contain
        canonical covariant / field-strength / gauge-fixing / ghost structures
        are expanded term by term through the compiler.
        """
        from gauge_compiler import compile_covariant_terms

        return Lagrangian(terms=self.all_interactions() + compile_covariant_terms(self))
