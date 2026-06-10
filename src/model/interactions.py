"""Interaction-term and external-leg structures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Mapping, Optional, Sequence

from symbolica import S

from .metadata import (
    LORENTZ_INDEX,
    ConjugateField,
    Field,
    FieldRole,
    IndexType,
    Statistics,
    _copy_index_labels,
    _normalize_index_labels,
    is_lorentz_index,
    lorentz_index_for,
    lorentz_kind_for,
    spinor_slots_for,
    unique_spinor_slot,
)


def _symbolic_key(value) -> str:
    if hasattr(value, "to_canonical_string"):
        return value.to_canonical_string()
    return str(value)


@dataclass(frozen=True)
class SlotLabels:
    """Slot-ordered index labels for one field occurrence or external leg."""

    field: Field
    values: tuple[Optional[object], ...]

    def __post_init__(self):
        if len(self.values) != len(self.field.indices):
            raise ValueError(
                f"SlotLabels for field {self.field.name!r} need {len(self.field.indices)} "
                f"entries, got {len(self.values)}."
            )

    @classmethod
    def from_legacy(
        cls,
        field: Field,
        labels: Optional[Mapping],
    ) -> "SlotLabels":
        slot_map = field.unpack_slot_labels(labels)
        return cls(
            field=field,
            values=tuple(slot_map.get(slot) for slot in range(len(field.indices))),
        )

    @classmethod
    def from_slot_map(
        cls,
        field: Field,
        slot_labels: Mapping[int, object],
    ) -> "SlotLabels":
        return cls(
            field=field,
            values=tuple(slot_labels.get(slot) for slot in range(len(field.indices))),
        )

    def to_slot_map(self) -> dict[int, object]:
        return {
            slot: label
            for slot, label in enumerate(self.values)
            if label is not None
        }

    def to_legacy(self) -> dict[str, object]:
        return self.field.pack_slot_labels(self.to_slot_map())

    def get(self, slot: int):
        return self.values[slot]

    def replace(self, slot: int, value) -> "SlotLabels":
        updated = list(self.values)
        updated[slot] = value
        return SlotLabels(field=self.field, values=tuple(updated))


@dataclass(frozen=True)
class SlotRef:
    occurrence: int
    slot: int


@dataclass(frozen=True)
class DerivativeRef:
    ordinal: int
    target: int


@dataclass(frozen=True)
class IndexBinding:
    index: IndexType
    label: object
    field_slots: tuple[SlotRef, ...] = ()
    derivatives: tuple[DerivativeRef, ...] = ()

    @property
    def multiplicity(self) -> int:
        return len(self.field_slots) + len(self.derivatives)

    @property
    def is_open(self) -> bool:
        return self.multiplicity == 1


@dataclass(frozen=True)
class DiracBilinear:
    psibar: SlotRef
    psi: SlotRef

    def as_legacy(self) -> tuple[int, int]:
        return self.psibar.occurrence, self.psi.occurrence


def _normalize_legacy_bilinears(
    closed_dirac_bilinears: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for pair in closed_dirac_bilinears:
        if len(pair) != 2:
            raise ValueError(
                "closed_dirac_bilinears entries must be (psibar_slot, psi_slot) pairs."
            )
        normalized.append((int(pair[0]), int(pair[1])))
    return tuple(normalized)


def _structural_bilinear_from_legacy_pair(
    fields: Sequence["FieldOccurrence"],
    pair: tuple[int, int],
) -> DiracBilinear:
    psibar_occurrence, psi_occurrence = pair
    if not (0 <= psibar_occurrence < len(fields) and 0 <= psi_occurrence < len(fields)):
        raise ValueError(
            "closed_dirac_bilinears contains a slot outside the interaction arity."
        )
    return DiracBilinear(
        psibar=SlotRef(
            occurrence=psibar_occurrence,
            slot=unique_spinor_slot(
                fields[psibar_occurrence].field,
                purpose="Dirac bilinear normalization",
            ),
        ),
        psi=SlotRef(
            occurrence=psi_occurrence,
            slot=unique_spinor_slot(
                fields[psi_occurrence].field,
                purpose="Dirac bilinear normalization",
            ),
        ),
    )


def _legacy_bilinears_from_structural(
    dirac_bilinears: Sequence[DiracBilinear],
) -> tuple[tuple[int, int], ...]:
    return tuple(bilinear.as_legacy() for bilinear in dirac_bilinears)


def _validate_structural_dirac_bilinears(
    fields: Sequence["FieldOccurrence"],
    dirac_bilinears: Sequence[DiracBilinear],
) -> tuple[DiracBilinear, ...]:
    validated: list[DiracBilinear] = []

    for bilinear in dirac_bilinears:
        if not isinstance(bilinear, DiracBilinear):
            raise TypeError(
                "dirac_bilinears must contain DiracBilinear entries."
            )

        for ref, endpoint in (
            (bilinear.psibar, "psibar"),
            (bilinear.psi, "psi"),
        ):
            if not isinstance(ref, SlotRef):
                raise TypeError(
                    f"dirac_bilinears {endpoint} endpoint must be a SlotRef."
                )
            if ref.occurrence < 0 or ref.occurrence >= len(fields):
                raise ValueError(
                    f"dirac_bilinears {endpoint} endpoint occurrence {ref.occurrence} "
                    "is outside the interaction arity."
                )

        psibar_occurrence = fields[bilinear.psibar.occurrence]
        psi_occurrence = fields[bilinear.psi.occurrence]

        if psibar_occurrence.field.kind != "fermion":
            raise ValueError(
                "dirac_bilinears psibar endpoint must point to a fermion occurrence."
            )
        if psi_occurrence.field.kind != "fermion":
            raise ValueError(
                "dirac_bilinears psi endpoint must point to a fermion occurrence."
            )
        if not psibar_occurrence.conjugated:
            raise ValueError(
                "dirac_bilinears psibar endpoint must point to a conjugated fermion occurrence."
            )
        if psi_occurrence.conjugated:
            raise ValueError(
                "dirac_bilinears psi endpoint must point to an unconjugated fermion occurrence."
            )

        expected_psibar_slot = unique_spinor_slot(
            psibar_occurrence.field,
            purpose="Dirac bilinear validation",
        )
        expected_psi_slot = unique_spinor_slot(
            psi_occurrence.field,
            purpose="Dirac bilinear validation",
        )
        if bilinear.psibar.slot != expected_psibar_slot:
            raise ValueError(
                f"dirac_bilinears psibar endpoint must target spinor slot "
                f"{expected_psibar_slot}, got {bilinear.psibar.slot}."
            )
        if bilinear.psi.slot != expected_psi_slot:
            raise ValueError(
                f"dirac_bilinears psi endpoint must target spinor slot "
                f"{expected_psi_slot}, got {bilinear.psi.slot}."
            )

        validated.append(bilinear)

    return tuple(validated)


@dataclass(frozen=True)
class FieldOccurrence:
    """One field factor inside an interaction monomial, with index labels."""

    field: Field
    conjugated: bool = False
    labels: dict = field(default_factory=dict)
    slot_labels: SlotLabels = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        normalized = _normalize_index_labels(self.field, self.labels)
        object.__setattr__(self, "labels", normalized)
        object.__setattr__(
            self,
            "slot_labels",
            SlotLabels.from_legacy(self.field, normalized),
        )

    @property
    def species(self):
        return self.field.species_for(self.conjugated)

    @property
    def role(self) -> FieldRole:
        return self.field.role_for(self.conjugated)

    def with_slot_labels(self, slot_labels: SlotLabels) -> "FieldOccurrence":
        if slot_labels.field is not self.field:
            raise ValueError("SlotLabels field does not match FieldOccurrence field.")
        return replace(self, labels=slot_labels.to_legacy())

    def with_slot_label(self, slot: int, value) -> "FieldOccurrence":
        return self.with_slot_labels(self.slot_labels.replace(slot, value))

    def __mul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(
            _FieldFactor(
                self.field,
                conjugated=self.conjugated,
                labels=self.labels,
            )
        ).__mul__(other)

    def __rmul__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(
            _FieldFactor(
                self.field,
                conjugated=self.conjugated,
                labels=self.labels,
            )
        ).__rmul__(other)

    def __add__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(
            _FieldFactor(
                self.field,
                conjugated=self.conjugated,
                labels=self.labels,
            )
        ).__add__(other)

    def __radd__(self, other):
        from .declared import _DeclaredMonomial, _FieldFactor

        return _DeclaredMonomial.from_factor(
            _FieldFactor(
                self.field,
                conjugated=self.conjugated,
                labels=self.labels,
            )
        ).__radd__(other)

    def apply_operator(self, operator, *, max_generated_terms=None):
        """Apply one runtime operator to this single field occurrence.

        This is a convenience wrapper for the common exploratory pattern
        ``field(...).apply_operator(op)``: the occurrence is lifted to a
        one-term compiled Lagrangian with unit coupling, and the existing
        compiled-Lagrangian operator pipeline does the actual work.
        """

        from .lagrangian import CompiledLagrangian

        return CompiledLagrangian(
            terms=(InteractionTerm(coupling=1, fields=(self,)),)
        ).apply_operator(
            operator,
            max_generated_terms=max_generated_terms,
        )


@dataclass(frozen=True)
class ExternalLeg:
    """External particle leg for vertex evaluation."""

    field: Field
    momentum: object
    conjugated: bool = False
    species: object = None
    spin: object = None
    labels: dict = field(default_factory=dict)
    slot_labels: SlotLabels = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        normalized = _normalize_index_labels(self.field, self.labels)
        object.__setattr__(self, "labels", normalized)
        object.__setattr__(
            self,
            "slot_labels",
            SlotLabels.from_legacy(self.field, normalized),
        )

    @property
    def effective_species(self):
        if self.species is not None:
            return self.species
        return self.field.species_for(self.conjugated)

    @property
    def role(self) -> FieldRole:
        return self.field.role_for(self.conjugated)


@dataclass(frozen=True)
class DerivativeAction:
    """A derivative acting on one field slot in the interaction."""

    target: int
    lorentz_index: object


@dataclass(frozen=True)
class InteractionTerm:
    """One interaction monomial after lowering/compilation."""

    coupling: object
    fields: tuple[FieldOccurrence, ...]
    derivatives: tuple[DerivativeAction, ...] = ()
    closed_dirac_bilinears: tuple[tuple[int, int], ...] = ()
    dirac_bilinears: tuple[DiracBilinear, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    label: str = ""
    sector: str = ""
    origin: str = ""
    origin_group: object = None

    def __post_init__(self):
        closed_dirac_bilinears = _normalize_legacy_bilinears(
            self.closed_dirac_bilinears
        )
        raw_dirac_bilinears = tuple(self.dirac_bilinears)
        dirac_bilinears: tuple[DiracBilinear, ...] = ()

        if closed_dirac_bilinears:
            if raw_dirac_bilinears:
                try:
                    validated_structural = _validate_structural_dirac_bilinears(
                        self.fields,
                        raw_dirac_bilinears,
                    )
                except (TypeError, ValueError):
                    validated_structural = ()
                else:
                    legacy_from_structural = _legacy_bilinears_from_structural(
                        validated_structural
                    )
                    if legacy_from_structural == closed_dirac_bilinears:
                        dirac_bilinears = validated_structural
            if not dirac_bilinears:
                dirac_bilinears = _validate_structural_dirac_bilinears(
                    self.fields,
                    tuple(
                        _structural_bilinear_from_legacy_pair(self.fields, pair)
                        for pair in closed_dirac_bilinears
                    ),
                )
        elif raw_dirac_bilinears:
            dirac_bilinears = _validate_structural_dirac_bilinears(
                self.fields,
                raw_dirac_bilinears,
            )
            closed_dirac_bilinears = _legacy_bilinears_from_structural(dirac_bilinears)

        object.__setattr__(self, "closed_dirac_bilinears", closed_dirac_bilinears)
        object.__setattr__(self, "dirac_bilinears", dirac_bilinears)

    @property
    def statistics(self) -> Statistics:
        for occ in self.fields:
            if occ.field.statistics == "fermion":
                return "fermion"
        return "boson"

    @property
    def index_bindings(self) -> tuple[IndexBinding, ...]:
        grouped: dict[tuple[IndexType, str], dict[str, object]] = {}
        order: list[tuple[IndexType, str]] = []

        def ensure_group(index: IndexType, label):
            key = (index, _symbolic_key(label))
            if key not in grouped:
                grouped[key] = {
                    "index": index,
                    "label": label,
                    "field_slots": [],
                    "derivatives": [],
                }
                order.append(key)
            return grouped[key]

        for occurrence_idx, occurrence in enumerate(self.fields):
            for slot, index in enumerate(occurrence.field.indices):
                label = occurrence.slot_labels.get(slot)
                if label is None:
                    continue
                group = ensure_group(index, label)
                group["field_slots"].append(
                    SlotRef(occurrence=occurrence_idx, slot=slot)
                )

        for ordinal, action in enumerate(self.derivatives):
            if not (0 <= action.target < len(self.fields)):
                continue
            target_field = self.fields[action.target].field
            derivative_index = lorentz_index_for(target_field.indices) or LORENTZ_INDEX
            group = ensure_group(derivative_index, action.lorentz_index)
            group["derivatives"].append(
                DerivativeRef(ordinal=ordinal, target=action.target)
            )

        return tuple(
            IndexBinding(
                index=grouped[key]["index"],
                label=grouped[key]["label"],
                field_slots=tuple(grouped[key]["field_slots"]),
                derivatives=tuple(grouped[key]["derivatives"]),
            )
            for key in order
        )

    def __add__(self, other):
        from .lagrangian import CompiledLagrangian

        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(terms=(self, other))
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(terms=(self,) + other.terms)
        return NotImplemented

    def __radd__(self, other):
        from .lagrangian import CompiledLagrangian

        if other == 0:
            return CompiledLagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(terms=(other, self))
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(terms=other.terms + (self,))
        return NotImplemented

    def feynman_rule(
        self,
        *fields,
        momenta=None,
        arity=None,
        select=None,
        simplify=True,
        key_format="names",
        include_delta: bool = False,
        strip_externals: bool = True,
        simplify_gamma: bool = False,
        flavor_expand=False,
    ):
        """Compute one vertex rule or a grouped zero-argument rule mapping."""
        from .lagrangian import CompiledLagrangian

        return CompiledLagrangian(terms=(self,)).feynman_rule(
            *fields,
            momenta=momenta,
            arity=arity,
            select=select,
            simplify=simplify,
            key_format=key_format,
            include_delta=include_delta,
            strip_externals=strip_externals,
            simplify_gamma=simplify_gamma,
            flavor_expand=flavor_expand,
        )

    def to_vertex_kwargs(self, external_legs: Sequence[ExternalLeg]) -> dict:
        """Generate the dict consumed by ``vertex_factor()``."""

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
        field_match_keys = [
            _field_match_key(occ.field, occ.conjugated) for occ in self.fields
        ]
        leg_match_keys = [
            _field_match_key(leg.field, leg.conjugated) for leg in external_legs
        ]

        field_index_labels = [
            _copy_index_labels(occ.slot_labels.to_legacy()) for occ in self.fields
        ]
        field_index_types = [occ.field.indices for occ in self.fields]
        leg_index_labels = [
            _copy_index_labels(leg.slot_labels.to_legacy()) for leg in external_legs
        ]
        leg_index_types = [leg.field.indices for leg in external_legs]

        leg_spins = [leg.spin for leg in external_legs]

        derivative_indices = [d.lorentz_index for d in self.derivatives]
        derivative_targets = [d.target for d in self.derivatives]
        coupling = self.coupling

        if derivative_indices:
            external_lorentz_labels = {
                binding.label
                for binding in self.index_bindings
                if is_lorentz_index(binding.index) and binding.field_slots
            }

            internal_lorentz_map: dict[str, object] = {}
            original_lorentz_map: dict[str, object] = {}
            normalized_derivatives = []
            next_internal = 1

            for lorentz_index in derivative_indices:
                if lorentz_index in external_lorentz_labels:
                    normalized_derivatives.append(lorentz_index)
                    continue

                key = (
                    lorentz_index.to_canonical_string()
                    if hasattr(lorentz_index, "to_canonical_string")
                    else str(lorentz_index)
                )
                mapped = internal_lorentz_map.get(key)
                if mapped is None:
                    mapped = S(f"mu{next_internal}_int")
                    internal_lorentz_map[key] = mapped
                    original_lorentz_map[key] = lorentz_index
                    next_internal += 1
                normalized_derivatives.append(mapped)

            if internal_lorentz_map:
                if hasattr(coupling, "replace"):
                    for key, mapped in internal_lorentz_map.items():
                        coupling = coupling.replace(original_lorentz_map[key], mapped)
                derivative_indices = normalized_derivatives

        return dict(
            coupling=coupling,
            alphas=alphas,
            betas=betas,
            ps=ps,
            statistics=self.statistics,
            field_roles=field_roles,
            leg_roles=leg_roles,
            field_match_keys=field_match_keys,
            leg_match_keys=leg_match_keys,
            field_index_labels=field_index_labels,
            field_index_types=field_index_types,
            leg_index_labels=leg_index_labels,
            leg_index_types=leg_index_types,
            leg_spins=leg_spins,
            derivative_indices=derivative_indices,
            derivative_targets=derivative_targets,
            closed_dirac_bilinears=tuple(
                bilinear.as_legacy() for bilinear in self.dirac_bilinears
            ),
        )


def _species_key(species) -> str:
    """Hashable key for a Symbolica species expression."""

    if hasattr(species, "to_canonical_string"):
        return species.to_canonical_string()
    return str(species)


def _field_match_key(field_obj: Field, conjugated: bool) -> tuple:
    """Stable key for matching external fields to interaction slots."""

    effective_conjugated = bool(conjugated and not field_obj.self_conjugate)
    return (
        field_obj.name,
        str(Fraction(field_obj.spin)),
        field_obj.kind,
        field_obj.statistics,
        tuple((index.name, index.kind, index.prefix) for index in field_obj.indices),
        _species_key(field_obj.symbol),
        _species_key(field_obj.conjugate_symbol)
        if field_obj.conjugate_symbol is not None
        else None,
        effective_conjugated,
    )


def _parse_field_arg(arg) -> tuple[Field, bool]:
    """Normalize a ``feynman_rule`` field argument to ``(Field, conjugated)``."""

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
    """Generate readable default labels for one leg."""

    leg_number = counter[0]
    counter[0] += 1

    kind_counts = Counter(idx.kind for idx in field_obj.indices)
    kind_ordinals: dict[str, int] = {}
    labels: dict[str, object] = {}

    for idx in field_obj.indices:
        ordinal = kind_ordinals.get(idx.kind, 0)
        kind_ordinals[idx.kind] = ordinal + 1
        count = kind_counts[idx.kind]
        base = f"{idx.prefix}{leg_number}"
        label = S(base if ordinal == 0 else f"{base}_{ordinal + 1}")

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
    """Check if an ``InteractionTerm`` matches the requested external fields."""

    if len(term.fields) != len(parsed_fields):
        return False
    term_species = Counter(
        _field_match_key(occ.field, occ.conjugated) for occ in term.fields
    )
    ext_species = Counter(
        _field_match_key(fld, conj) for fld, conj in parsed_fields
    )
    return term_species == ext_species
