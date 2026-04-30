"""Internal covariant-core compilation helpers.

This module keeps the free-bilinear / gauge-only policy logic separate from
the larger ``compiler.gauge`` entrypoint without changing any public APIs.
"""

from __future__ import annotations

from typing import Callable

from symbolica import Expression

from .spectators import (
    _decorate_interactions_with_spectators,
    _materialize_spectator_occurrences,
    _spectator_identity_factor,
)
from lagrangian.operators import psi_bar_gamma_psi
from model import (
    ComplexScalarKineticTerm,
    DerivativeAction,
    DiracKineticTerm,
    Field,
    InteractionTerm,
    Model,
)
from symbolic.spenso_structures import SPINOR_KIND


def _compile_dirac_partial_term(
    fermion: Field,
    *,
    coefficient=1,
    label: str = "",
    unique_slot: Callable,
    symbol: Callable,
) -> InteractionTerm:
    mu = symbol("mu")
    i_bar = symbol(f"i_bar_{fermion.name}_covd")
    i_psi = symbol(f"i_{fermion.name}_covd")
    fermion_spinor_slot = unique_slot(
        fermion,
        SPINOR_KIND,
        purpose="Dirac kinetic partial-term compilation",
    )
    bar_slot_labels = {fermion_spinor_slot: i_bar}
    psi_slot_labels = {fermion_spinor_slot: i_psi}
    core_factor, core_bar_slots, core_psi_slots = _spectator_identity_factor(
        fermion,
        exclude_slots={fermion_spinor_slot},
    )
    bar_slot_labels.update(core_bar_slots)
    psi_slot_labels.update(core_psi_slots)
    bar_labels = fermion.pack_slot_labels(bar_slot_labels)
    psi_labels = fermion.pack_slot_labels(psi_slot_labels)
    return InteractionTerm(
        coupling=Expression.I * coefficient * core_factor * psi_bar_gamma_psi(i_bar, i_psi, mu),
        fields=(
            fermion.occurrence(conjugated=True, labels=bar_labels),
            fermion.occurrence(labels=psi_labels),
        ),
        derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        closed_dirac_bilinears=((0, 1),),
        label=label or f"i {fermion.name}bar gamma^mu d_mu {fermion.name}",
    )


def _compile_complex_scalar_partial_term(
    scalar: Field,
    *,
    coefficient=1,
    label: str = "",
    symbol: Callable,
) -> InteractionTerm:
    mu = symbol("mu")
    core_factor, scalar_bar_slots, scalar_slots = _spectator_identity_factor(
        scalar,
        exclude_slots=(),
    )
    scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slots)
    scalar_labels = scalar.pack_slot_labels(scalar_slots)
    return InteractionTerm(
        coupling=coefficient * core_factor,
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
            scalar.occurrence(labels=scalar_labels),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=1, lorentz_index=mu),
        ),
        label=label or f"(d_mu {scalar.name})^dagger (d^mu {scalar.name})",
    )


def _assemble_full_covariant_operator(
    gauge_terms: tuple[InteractionTerm, ...],
    partial_term: InteractionTerm,
    spectator_factor,
    spectator_occurrences: tuple,
    spectator_bilinears: tuple[tuple[int, int], ...],
) -> tuple[InteractionTerm, ...]:
    """Assemble one declarative ``CovD`` operator from gauge and partial pieces."""
    gauge_decorated = _decorate_interactions_with_spectators(
        gauge_terms,
        spectator_factor=spectator_factor,
        spectator_occurrences=spectator_occurrences,
        spectator_bilinears=spectator_bilinears,
    )
    partial_decorated = _decorate_interactions_with_spectators(
        (partial_term,),
        spectator_factor=spectator_factor,
        spectator_occurrences=spectator_occurrences,
        spectator_bilinears=spectator_bilinears,
    )
    return gauge_decorated + partial_decorated


def _compile_covariant_core(
    model: Model,
    core: DiracKineticTerm | ComplexScalarKineticTerm,
    *,
    include_free_bilinear: bool,
    spectators: tuple[tuple[Field, bool], ...] = (),
    require_declared_field: Callable,
    compile_dirac_kinetic_term: Callable,
    compile_complex_scalar_kinetic_term: Callable,
    unique_slot: Callable,
    symbol: Callable,
) -> tuple[InteractionTerm, ...]:
    """Compile one covariant kinetic core with explicit free-bilinear policy."""
    spectator_factor, spectator_occurrences, spectator_bilinears = (
        _materialize_spectator_occurrences(spectators)
    )

    if isinstance(core, DiracKineticTerm):
        fermion = require_declared_field(
            model,
            core.field,
            purpose="Covariant monomial compilation",
        )
        if fermion.kind != "fermion":
            raise ValueError(
                f"Covariant Dirac monomial requires a fermion field, got kind={fermion.kind!r}."
            )
        gauge_terms = compile_dirac_kinetic_term(model, core)
        if not include_free_bilinear:
            return _decorate_interactions_with_spectators(
                gauge_terms,
                spectator_factor=spectator_factor,
                spectator_occurrences=spectator_occurrences,
                spectator_bilinears=spectator_bilinears,
            )
        partial_term = _compile_dirac_partial_term(
            fermion,
            coefficient=core.coefficient,
            label=core.label or f"i {fermion.name}bar gamma^mu D_mu {fermion.name} partial",
            unique_slot=unique_slot,
            symbol=symbol,
        )
        return _assemble_full_covariant_operator(
            gauge_terms,
            partial_term,
            spectator_factor,
            spectator_occurrences,
            spectator_bilinears,
        )

    if isinstance(core, ComplexScalarKineticTerm):
        scalar = require_declared_field(
            model,
            core.field,
            purpose="Covariant monomial compilation",
        )
        if scalar.kind != "scalar" or scalar.self_conjugate:
            raise ValueError(
                "Covariant complex-scalar monomials require a non-self-conjugate scalar field."
            )
        gauge_terms = compile_complex_scalar_kinetic_term(model, core)
        if not include_free_bilinear:
            return _decorate_interactions_with_spectators(
                gauge_terms,
                spectator_factor=spectator_factor,
                spectator_occurrences=spectator_occurrences,
                spectator_bilinears=spectator_bilinears,
            )
        partial_term = _compile_complex_scalar_partial_term(
            scalar,
            coefficient=core.coefficient,
            label=core.label or f"(D_mu {scalar.name})^dagger (D^mu {scalar.name}) derivative",
            symbol=symbol,
        )
        return _assemble_full_covariant_operator(
            gauge_terms,
            partial_term,
            spectator_factor,
            spectator_occurrences,
            spectator_bilinears,
        )

    raise TypeError(f"Unsupported covariant monomial core type: {type(core)!r}")


def _compile_declared_covariant_core(
    model: Model,
    core: DiracKineticTerm | ComplexScalarKineticTerm,
    *,
    spectators: tuple[tuple[Field, bool], ...] = (),
    require_declared_field: Callable,
    compile_dirac_kinetic_term: Callable,
    compile_complex_scalar_kinetic_term: Callable,
    unique_slot: Callable,
    symbol: Callable,
) -> tuple[InteractionTerm, ...]:
    """Compile one declarative ``CovD`` monomial as the full kinetic operator."""
    return _compile_covariant_core(
        model,
        core,
        include_free_bilinear=True,
        spectators=spectators,
        require_declared_field=require_declared_field,
        compile_dirac_kinetic_term=compile_dirac_kinetic_term,
        compile_complex_scalar_kinetic_term=compile_complex_scalar_kinetic_term,
        unique_slot=unique_slot,
        symbol=symbol,
    )


def _compile_legacy_covariant_core(
    model: Model,
    core: DiracKineticTerm | ComplexScalarKineticTerm,
    *,
    require_declared_field: Callable,
    compile_dirac_kinetic_term: Callable,
    compile_complex_scalar_kinetic_term: Callable,
    unique_slot: Callable,
    symbol: Callable,
) -> tuple[InteractionTerm, ...]:
    """Compile one legacy kinetic declaration as gauge-interaction-only."""
    return _compile_covariant_core(
        model,
        core,
        include_free_bilinear=False,
        require_declared_field=require_declared_field,
        compile_dirac_kinetic_term=compile_dirac_kinetic_term,
        compile_complex_scalar_kinetic_term=compile_complex_scalar_kinetic_term,
        unique_slot=unique_slot,
        symbol=symbol,
    )
