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

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, Mapping, Optional, Sequence

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
class GaugeGroup:
    """Gauge symmetry group declaration (mirrors M$GaugeGroups)."""
    name: str
    abelian: bool
    coupling: object
    gauge_boson: Optional[str] = None
    structure_constant: Optional[str] = None
    representations: tuple = ()
    charge: Optional[str] = None


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

    def role_for(self, conjugated: bool = False) -> str:
        if self.kind == "fermion":
            return "psibar" if conjugated else "psi"
        return "scalar"

    def species_for(self, conjugated: bool = False):
        if conjugated and not self.self_conjugate:
            return self.conjugate_symbol or S(self.name + "bar")
        return self.symbol

    def occurrence(self, *, conjugated: bool = False, labels: dict | None = None):
        """Create a FieldOccurrence of this field in an interaction term."""
        return FieldOccurrence(
            field=self,
            conjugated=conjugated,
            labels=labels or {},
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
            labels=labels or {},
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
    def role(self) -> str:
        return self.field.role_for(self.conjugated)

    @property
    def spinor_label(self):
        return self.labels.get(SPINOR_KIND)


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
    def role(self) -> str:
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

        field_index_labels = [dict(occ.labels) for occ in self.fields]
        leg_index_labels = [dict(leg.labels) for leg in external_legs]

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
# Model container
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Top-level model container (mirrors the full .fr file)."""
    name: str = ""
    gauge_groups: tuple[GaugeGroup, ...] = ()
    fields: tuple[Field, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    interactions: tuple[InteractionTerm, ...] = ()
