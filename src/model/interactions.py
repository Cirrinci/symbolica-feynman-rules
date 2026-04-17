"""Interaction-term and external-leg structures."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Sequence

from symbolica import S

from .declared import PartialDerivativeFactor, _DeclaredMonomial
from .metadata import (
    ConjugateField,
    Field,
    FieldRole,
    SPINOR_KIND,
    Statistics,
    _copy_index_labels,
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
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(self, other))
        if isinstance(other, Lagrangian):
            return Lagrangian(self, other)
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(self, *local_terms)
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=(self,) + decl_terms)
        return NotImplemented

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if other == 0:
            return Lagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other, self))
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(*local_terms, self)
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


def _standalone_lagrangian_context_error() -> str:
    return (
        "Standalone Lagrangian(...) only supports local terms built from "
        "fields, PartialD(...), and one optional Gamma(...). "
        "Use Model(lagrangian_decl=...) for CovD(...), FieldStrength(...), "
        "GaugeFixing(...), and GhostLagrangian(...), since those need "
        "model metadata."
    )# ---------------------------------------------------------------------------
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
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(self, other))
        if isinstance(other, Lagrangian):
            return Lagrangian(self, other)
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(self, *local_terms)
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=(self,) + decl_terms)
        return NotImplemented

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if other == 0:
            return Lagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other, self))
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(*local_terms, self)
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


def _standalone_lagrangian_context_error() -> str:
    return (
        "Standalone Lagrangian(...) only supports local terms built from "
        "fields, PartialD(...), and one optional Gamma(...). "
        "Use Model(lagrangian_decl=...) for CovD(...), FieldStrength(...), "
        "GaugeFixing(...), and GhostLagrangian(...), since those need "
        "model metadata."
    )# ---------------------------------------------------------------------------
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
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(self, other))
        if isinstance(other, Lagrangian):
            return Lagrangian(self, other)
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(self, *local_terms)
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=(self,) + decl_terms)
        return NotImplemented

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if other == 0:
            return Lagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other, self))
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(*local_terms, self)
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


def _standalone_lagrangian_context_error() -> str:
    return (
        "Standalone Lagrangian(...) only supports local terms built from "
        "fields, PartialD(...), and one optional Gamma(...). "
        "Use Model(lagrangian_decl=...) for CovD(...), FieldStrength(...), "
        "GaugeFixing(...), and GhostLagrangian(...), since those need "
        "model metadata."
    )# ---------------------------------------------------------------------------
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
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(self, other))
        if isinstance(other, Lagrangian):
            return Lagrangian(self, other)
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(self, *local_terms)
        decl_terms = _declared_source_terms_from_item(other)
        if decl_terms is not None:
            return DeclaredLagrangian(source_terms=(self,) + decl_terms)
        return NotImplemented

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian, Lagrangian
        from .lowering import (
            _declared_source_terms_from_item,
            _standalone_lagrangian_source_terms_from_item,
        )

        if other == 0:
            return Lagrangian(terms=(self,))
        if isinstance(other, InteractionTerm):
            return Lagrangian(terms=(other, self))
        local_terms = _standalone_lagrangian_source_terms_from_item(other)
        if local_terms is not None:
            return Lagrangian(*local_terms, self)
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


def _standalone_lagrangian_context_error() -> str:
    return (
        "Standalone Lagrangian(...) only supports local terms built from "
        "fields, PartialD(...), and one optional Gamma(...). "
        "Use Model(lagrangian_decl=...) for CovD(...), FieldStrength(...), "
        "GaugeFixing(...), and GhostLagrangian(...), since those need "
        "model metadata."
    )