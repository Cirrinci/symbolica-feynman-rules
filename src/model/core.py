"""Top-level model container."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import warnings

from .declared import _DeclaredMonomial
from .interactions import InteractionTerm
from .lagrangian import (
    ComplexScalarKineticTerm,
    CovariantTerm,
    DeclaredLagrangian,
    DiracKineticTerm,
    GaugeFixingTerm,
    GaugeKineticTerm,
    GhostTerm,
    Lagrangian,
)
from .lowering import (
    _normalize_interaction_terms_input,
    _validate_declared_source_term,
    _source_term_interaction,
    _source_term_covariant_core,
    _source_term_gauge_kinetic,
    _source_term_gauge_fixing,
    _source_term_ghost,
    _source_term_needs_compilation,
)
from .metadata import Field, GaugeGroup, Parameter
# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------



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
        from compiler.gauge import compile_covariant_terms

        return Lagrangian(terms=self.all_interactions() + compile_covariant_terms(self))
