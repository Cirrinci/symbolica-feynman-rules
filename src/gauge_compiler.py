"""
Minimal gauge-interaction compiler for the Symbolica/Spenso prototype.

This module turns model metadata into normalized ``InteractionTerm`` objects
for the gauge interactions that are already supported by the engine:

- fermion-gauge currents for abelian and non-abelian groups
- abelian complex-scalar current/contact terms

The goal is not a full FeynRules compiler yet. It is the smallest model-driven
bridge that replaces hand-written gauge interaction terms in examples.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from symbolica import S

from model import (
    DerivativeAction,
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


def compile_fermion_gauge_current(
    *,
    fermion: Field,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    lorentz_label=None,
    spinor_labels=None,
    matter_labels=None,
    adjoint_label=None,
    label: str = "",
) -> InteractionTerm:
    """Compile one fermion-gauge current interaction from model metadata."""
    if fermion.kind != "fermion":
        raise ValueError(f"Expected a fermion field, got kind={fermion.kind!r}.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    mu = lorentz_label or _default_vector_label(gauge_field, gauge_group, suffix="mu")
    i_bar, i_psi = spinor_labels or _default_spinor_labels(fermion, gauge_group)

    coupling = gauge_group.coupling * psi_bar_gamma_psi(i_bar, i_psi, mu)
    bar_labels = {SPINOR_KIND: i_bar}
    psi_labels = {SPINOR_KIND: i_psi}
    gauge_labels = {LORENTZ_KIND: mu}

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
    label_prefix: str = "",
):
    """Compile the abelian complex-scalar gauge current/contact terms."""
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")
    if not gauge_group.abelian:
        raise NotImplementedError(
            "Non-abelian complex-scalar compilation is not implemented yet."
        )

    charge = _field_charge(scalar, gauge_group)
    mu, nu = lorentz_labels or (
        _default_vector_label(gauge_field, gauge_group, suffix="mu"),
        _default_vector_label(gauge_field, gauge_group, suffix="nu"),
    )
    base = gauge_group.coupling * charge

    current_phi = InteractionTerm(
        coupling=base,
        fields=(
            scalar.occurrence(conjugated=True),
            scalar.occurrence(),
            gauge_field.occurrence(labels={LORENTZ_KIND: mu}),
        ),
        derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: scalar current (+)",
    )
    current_phidag = InteractionTerm(
        coupling=-base,
        fields=(
            scalar.occurrence(conjugated=True),
            scalar.occurrence(),
            gauge_field.occurrence(labels={LORENTZ_KIND: mu}),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: scalar current (-)",
    )
    contact = InteractionTerm(
        coupling=(base ** 2) * scalar_gauge_contact(mu, nu),
        fields=(
            scalar.occurrence(conjugated=True),
            scalar.occurrence(),
            gauge_field.occurrence(labels={LORENTZ_KIND: mu}),
            gauge_field.occurrence(labels={LORENTZ_KIND: nu}),
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

            if field.kind == "scalar" and not field.self_conjugate and gauge_group.abelian:
                if gauge_group.charge is None or field.quantum_numbers.get(gauge_group.charge, 0) == 0:
                    continue
                interactions.extend(
                    compile_complex_scalar_gauge_terms(
                        scalar=field,
                        gauge_group=gauge_group,
                        gauge_field=gauge_field,
                    )
                )

    return tuple(interactions)


def with_minimal_gauge_interactions(model: Model) -> Model:
    """Return a copy of a model with compiled gauge interactions appended."""
    compiled = compile_minimal_gauge_interactions(model)
    return replace(model, interactions=model.interactions + compiled)
