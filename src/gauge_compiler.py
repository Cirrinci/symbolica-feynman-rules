"""
Minimal gauge-interaction compiler for the Symbolica/Spenso prototype.

This module turns model metadata into normalized ``InteractionTerm`` objects
for the gauge interactions that are already supported by the engine:

- fermion-gauge currents for abelian and non-abelian groups
- complex-scalar current/contact terms for abelian and non-abelian groups

The goal is not a full FeynRules compiler yet. It is the smallest model-driven
bridge that replaces hand-written gauge interaction terms in examples.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from symbolica import S, Expression

from model import (
    ComplexScalarKineticTerm,
    DerivativeAction,
    DiracKineticTerm,
    Field,
    GaugeGroup,
    InteractionTerm,
    Model,
)
from operators import psi_bar_gamma_psi, scalar_gauge_contact
from spenso_structures import LORENTZ_KIND, SPINOR_KIND


def _symbol(name: str):
    return S(name)


def _default_spinor_labels(field: Field, gauge_group: GaugeGroup):
    stem = f"{field.name}_{gauge_group.name}"
    return _symbol(f"i_bar_{stem}"), _symbol(f"i_{stem}")


def _default_vector_label(field: Field, gauge_group: GaugeGroup, suffix: str = "mu"):
    del field, gauge_group
    return _symbol(suffix)


def _default_matter_labels(field: Field, rep_prefix: str):
    stem = field.name
    return _symbol(f"{rep_prefix}_bar_{stem}"), _symbol(f"{rep_prefix}_{stem}")


def _default_index_labels(field: Field, index, qualifier: str = "id"):
    stem = f"{field.name}_{index.kind}_{qualifier}"
    return _symbol(f"{index.prefix}_bar_{stem}"), _symbol(f"{index.prefix}_{stem}")


def _first_non_lorentz_index_kind(field: Field) -> Optional[str]:
    for index in field.indices:
        if index.kind != LORENTZ_KIND:
            return index.kind
    return None


def _adjoint_index_kind(gauge_field: Field) -> Optional[str]:
    return _first_non_lorentz_index_kind(gauge_field)


def _field_charge(field: Field, gauge_group: GaugeGroup):
    if gauge_group.charge is None:
        raise ValueError(f"Gauge group {gauge_group.name!r} has no abelian charge label.")
    if gauge_group.charge not in field.quantum_numbers:
        raise ValueError(
            f"Field {field.name!r} has no quantum number {gauge_group.charge!r} "
            f"required by gauge group {gauge_group.name!r}."
        )
    return field.quantum_numbers[gauge_group.charge]


def _field_transforms_under_gauge_group(field: Field, gauge_group: GaugeGroup) -> bool:
    if gauge_group.abelian:
        if gauge_group.charge is None:
            return False
        return field.quantum_numbers.get(gauge_group.charge, 0) != 0
    return gauge_group.matter_representation(field) is not None


def _resolve_covariant_gauge_groups(model: Model, *, field: Field, gauge_group=None) -> tuple[GaugeGroup, ...]:
    if gauge_group is not None:
        if isinstance(gauge_group, (tuple, list)):
            resolved = []
            for item in gauge_group:
                group = model.find_gauge_group(item)
                if group is None:
                    raise ValueError(f"Could not resolve gauge group {item!r}.")
                resolved.append(group)
            return tuple(resolved)

        resolved = model.find_gauge_group(gauge_group)
        if resolved is None:
            raise ValueError(f"Could not resolve gauge group {gauge_group!r}.")
        return (resolved,)

    matches = tuple(group for group in model.gauge_groups if _field_transforms_under_gauge_group(field, group))
    if not matches:
        raise ValueError(f"Field {field.name!r} does not transform under any declared gauge group.")
    return matches


def _spectator_identity_factor(field: Field, *, exclude_index_kinds=()):
    factor = Expression.num(1)
    left_labels = {}
    right_labels = {}

    for index in field.indices:
        if index.kind in exclude_index_kinds or index.kind == LORENTZ_KIND:
            continue
        left_label, right_label = _default_index_labels(field, index)
        factor *= index.representation.g(left_label, right_label).to_expression()
        left_labels[index.kind] = left_label
        right_labels[index.kind] = right_label

    return factor, left_labels, right_labels


def compile_fermion_gauge_current(
    *,
    fermion: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_label=None,
    spinor_labels=None,
    matter_labels=None,
    adjoint_label=None,
    prefactor=1,
    label: str = "",
) -> InteractionTerm:
    """Compile one fermion-gauge current interaction from model metadata."""
    if fermion.kind != "fermion":
        raise ValueError(f"Expected a fermion field, got kind={fermion.kind!r}.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    mu = lorentz_label or _default_vector_label(gauge_field, gauge_group, suffix="mu")
    i_bar, i_psi = spinor_labels or _default_spinor_labels(fermion, gauge_group)

    coupling = prefactor * gauge_group.coupling * psi_bar_gamma_psi(i_bar, i_psi, mu)
    bar_labels = {SPINOR_KIND: i_bar}
    psi_labels = {SPINOR_KIND: i_psi}
    gauge_labels = {LORENTZ_KIND: mu}
    spectator_exclusions = {SPINOR_KIND}

    if gauge_group.abelian:
        coupling *= _field_charge(fermion, gauge_group)
    else:
        rep = gauge_group.matter_representation(fermion)
        if rep is None:
            raise ValueError(
                f"Field {fermion.name!r} carries no representation declared for "
                f"gauge group {gauge_group.name!r}."
            )
        left_label, right_label = matter_labels or _default_matter_labels(fermion, rep.index.prefix)
        adj_kind = _adjoint_index_kind(gauge_field)
        if adj_kind is None:
            raise ValueError(
                f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz "
                "adjoint index."
            )
        adjoint = adjoint_label or _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}")
        coupling *= rep.build_generator(adjoint, left_label, right_label)
        bar_labels[rep.index.kind] = left_label
        psi_labels[rep.index.kind] = right_label
        gauge_labels[adj_kind] = adjoint
        spectator_exclusions.add(rep.index.kind)

    spectator_factor, spectator_left_labels, spectator_right_labels = _spectator_identity_factor(
        fermion,
        exclude_index_kinds=spectator_exclusions,
    )
    coupling *= spectator_factor
    bar_labels.update(spectator_left_labels)
    psi_labels.update(spectator_right_labels)

    return InteractionTerm(
        coupling=coupling,
        fields=(
            fermion.occurrence(conjugated=True, labels=bar_labels),
            fermion.occurrence(labels=psi_labels),
            gauge_field.occurrence(labels=gauge_labels),
        ),
        label=label or f"{gauge_group.name}: {fermion.name} gauge current",
    )


def compile_complex_scalar_gauge_terms(
    *,
    scalar: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_labels=None,
    matter_labels=None,
    adjoint_labels=None,
    internal_label=None,
    current_prefactor=1,
    contact_prefactor=1,
    label_prefix: str = "",
):
    """Compile the complex-scalar gauge current/contact terms."""
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")
    mu, nu = lorentz_labels or (
        _default_vector_label(gauge_field, gauge_group, suffix="mu"),
        _default_vector_label(gauge_field, gauge_group, suffix="nu"),
    )
    scalar_bar_labels = {}
    scalar_labels = {}
    gauge_labels_mu = {LORENTZ_KIND: mu}
    gauge_labels_nu = {LORENTZ_KIND: nu}
    spectator_exclusions = set()

    if gauge_group.abelian:
        charge = _field_charge(scalar, gauge_group)
        current_base = current_prefactor * gauge_group.coupling * charge
        contact_coupling = contact_prefactor * ((gauge_group.coupling * charge) ** 2)
    else:
        rep = gauge_group.matter_representation(scalar)
        if rep is None:
            raise ValueError(
                f"Field {scalar.name!r} carries no representation declared for "
                f"gauge group {gauge_group.name!r}."
            )
        adj_kind = _adjoint_index_kind(gauge_field)
        if adj_kind is None:
            raise ValueError(
                f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz "
                "adjoint index."
            )

        left_label, right_label = matter_labels or _default_matter_labels(scalar, rep.index.prefix)
        adjoint_mu, adjoint_nu = adjoint_labels or (
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_1"),
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_2"),
        )
        middle_label = internal_label or _symbol(f"{rep.index.prefix}_mid_{scalar.name}_{gauge_group.name}")

        generator_mu = rep.build_generator(adjoint_mu, left_label, right_label)
        generator_chain = (
            rep.build_generator(adjoint_mu, left_label, middle_label)
            * rep.build_generator(adjoint_nu, middle_label, right_label)
        )

        current_base = current_prefactor * gauge_group.coupling * generator_mu
        contact_coupling = contact_prefactor * (gauge_group.coupling ** 2) * generator_chain

        scalar_bar_labels[rep.index.kind] = left_label
        scalar_labels[rep.index.kind] = right_label
        gauge_labels_mu[adj_kind] = adjoint_mu
        gauge_labels_nu[adj_kind] = adjoint_nu
        spectator_exclusions.add(rep.index.kind)

    spectator_factor, spectator_left_labels, spectator_right_labels = _spectator_identity_factor(
        scalar,
        exclude_index_kinds=spectator_exclusions,
    )
    current_base *= spectator_factor
    contact_coupling *= spectator_factor
    scalar_bar_labels.update(spectator_left_labels)
    scalar_labels.update(spectator_right_labels)

    current_phi = InteractionTerm(
        coupling=current_base,
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
            scalar.occurrence(labels=scalar_labels),
            gauge_field.occurrence(labels=gauge_labels_mu),
        ),
        derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: scalar current (+)",
    )
    current_phidag = InteractionTerm(
        coupling=-current_base,
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
            scalar.occurrence(labels=scalar_labels),
            gauge_field.occurrence(labels=gauge_labels_mu),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: scalar current (-)",
    )
    contact = InteractionTerm(
        coupling=contact_coupling * scalar_gauge_contact(mu, nu),
        fields=(
            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
            scalar.occurrence(labels=scalar_labels),
            gauge_field.occurrence(labels=gauge_labels_mu),
            gauge_field.occurrence(labels=gauge_labels_nu),
        ),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: scalar contact",
    )
    return (current_phi, current_phidag, contact)


def compile_minimal_gauge_interactions(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile the currently supported gauge interactions from a model."""
    interactions: list[InteractionTerm] = []

    for gauge_group in model.gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)

        for field in model.fields:
            if field == gauge_field:
                continue

            if field.kind == "fermion":
                if gauge_group.abelian:
                    if gauge_group.charge is None or field.quantum_numbers.get(gauge_group.charge, 0) == 0:
                        continue
                else:
                    if gauge_group.matter_representation(field) is None:
                        continue

                interactions.append(
                    compile_fermion_gauge_current(
                        fermion=field,
                        gauge_group=gauge_group,
                        gauge_field=gauge_field,
                    )
                )
                continue

            if field.kind == "scalar" and not field.self_conjugate:
                if gauge_group.abelian:
                    if gauge_group.charge is None or field.quantum_numbers.get(gauge_group.charge, 0) == 0:
                        continue
                else:
                    if gauge_group.matter_representation(field) is None:
                        continue
                interactions.extend(
                    compile_complex_scalar_gauge_terms(
                        scalar=field,
                        gauge_group=gauge_group,
                        gauge_field=gauge_field,
                        current_prefactor=1,
                        contact_prefactor=1,
                    )
                )

    return tuple(interactions)


def with_minimal_gauge_interactions(model: Model) -> Model:
    """Return a copy of a model with compiled gauge interactions appended."""
    compiled = compile_minimal_gauge_interactions(model)
    return replace(model, interactions=model.interactions + compiled)


def compile_dirac_kinetic_term(model: Model, term: DiracKineticTerm) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``psibar i gamma^mu D_mu psi``."""
    fermion = model.find_field(term.field)
    if fermion is None:
        raise ValueError(f"Could not resolve fermion field {term.field!r}.")
    if fermion.kind != "fermion":
        raise ValueError(f"Dirac kinetic term requires a fermion field, got kind={fermion.kind!r}.")

    gauge_groups = _resolve_covariant_gauge_groups(
        model,
        field=fermion,
        gauge_group=term.gauge_group,
    )
    label = term.label or f"i {fermion.name}bar gamma^mu D_mu {fermion.name}"

    interactions = []
    for gauge_group in gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)
        interactions.append(
            compile_fermion_gauge_current(
                fermion=fermion,
                gauge_group=gauge_group,
                gauge_field=gauge_field,
                prefactor=-term.coefficient,
                label=label,
            )
        )
    return tuple(interactions)


def compile_complex_scalar_kinetic_term(
    model: Model,
    term: ComplexScalarKineticTerm,
) -> tuple[InteractionTerm, ...]:
    """Compile the gauge-interaction part of ``(D_mu phi)^dagger (D^mu phi)``."""
    scalar = model.find_field(term.field)
    if scalar is None:
        raise ValueError(f"Could not resolve scalar field {term.field!r}.")
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError(
            "Complex-scalar kinetic terms require a non-self-conjugate scalar field."
        )

    label_prefix = term.label or f"(D_mu {scalar.name})^dagger (D^mu {scalar.name})"
    gauge_groups = _resolve_covariant_gauge_groups(
        model,
        field=scalar,
        gauge_group=term.gauge_group,
    )

    interactions = []
    for gauge_group in gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)
        interactions.extend(
            compile_complex_scalar_gauge_terms(
                scalar=scalar,
                gauge_group=gauge_group,
                gauge_field=gauge_field,
                current_prefactor=Expression.I * term.coefficient,
                contact_prefactor=term.coefficient,
                label_prefix=label_prefix,
            )
        )
    return tuple(interactions)


def compile_covariant_terms(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile all declared covariant-kinetic terms in a model."""
    interactions: list[InteractionTerm] = []

    for term in model.covariant_terms:
        if isinstance(term, DiracKineticTerm):
            interactions.extend(compile_dirac_kinetic_term(model, term))
            continue
        if isinstance(term, ComplexScalarKineticTerm):
            interactions.extend(compile_complex_scalar_kinetic_term(model, term))
            continue
        raise TypeError(f"Unsupported covariant term type: {type(term)!r}")

    return tuple(interactions)


def with_compiled_covariant_terms(model: Model) -> Model:
    """Return a copy of a model with compiled covariant terms appended."""
    compiled = compile_covariant_terms(model)
    return replace(model, interactions=model.interactions + compiled)
