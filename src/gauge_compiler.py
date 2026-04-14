"""
Gauge-structure compilers for the Symbolica/Spenso prototype.

This module exposes two layers on purpose:

- the minimal compiler: structural gauge interactions from metadata
- the covariant/physical compiler: convention-fixed kinetic-term expansion

Frozen conventions for the physical path:

- Fourier transform derivatives act as ``-i p_mu``
- ``vertex_factor(...)`` contributes the universal overall ``+i``
- matter covariant derivatives use ``D_mu = partial_mu + i g A_mu``
- non-abelian field strengths use
  ``F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu``

With these choices, matter currents carry the signs already locked down in the
covariant tests, the Yang-Mills 3-gauge vertex is real, and the 4-gauge vertex
keeps an explicit overall ``i``.
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
    GhostTerm,
    GaugeKineticTerm,
    GaugeFixingTerm,
    GaugeGroup,
    InteractionTerm,
    Model,
)
from operators import psi_bar_gamma_psi, scalar_gauge_contact
from spenso_structures import LORENTZ_KIND, SPINOR_KIND, lorentz_metric


_HALF = Expression.num(1) / Expression.num(2)
_QUARTER = Expression.num(1) / Expression.num(4)


def _symbol(name: str):
    return S(name)


def _is_explicit_zero(value) -> bool:
    """Whether one coefficient is manifestly zero after local normalization."""
    try:
        normalized = value.expand() if hasattr(value, "expand") else value
    except Exception:
        normalized = value
    try:
        return bool(normalized == 0)
    except Exception:
        return False


def _default_spinor_labels(field: Field, gauge_group: GaugeGroup):
    stem = f"{field.name}_{gauge_group.name}"
    return _symbol(f"i_bar_{stem}"), _symbol(f"i_{stem}")


def _slot_suffix(field: Field, slot: Optional[int]) -> str:
    if slot is None:
        return ""
    kind = field.indices[slot].kind
    if field.index_kind_count(kind) <= 1:
        return ""
    return f"_slot{slot + 1}"


def _default_vector_label(field: Field, gauge_group: GaugeGroup, suffix: str = "mu"):
    del field, gauge_group
    return _symbol(suffix)


def _default_matter_labels(field: Field, rep_prefix: str, slot: Optional[int] = None):
    stem = f"{field.name}{_slot_suffix(field, slot)}"
    return _symbol(f"{rep_prefix}_bar_{stem}"), _symbol(f"{rep_prefix}_{stem}")


def _default_index_labels(field: Field, index, qualifier: str = "id", slot: Optional[int] = None):
    stem = f"{field.name}_{index.kind}"
    if slot is not None and field.index_kind_count(index.kind) > 1:
        stem += f"_{slot + 1}"
    stem += f"_{qualifier}"
    return _symbol(f"{index.prefix}_bar_{stem}"), _symbol(f"{index.prefix}_{stem}")


def _first_non_lorentz_index_slot(field: Field) -> Optional[int]:
    for slot, index in enumerate(field.indices):
        if index.kind != LORENTZ_KIND:
            return slot
    return None


def _adjoint_index_kind(gauge_field: Field) -> Optional[str]:
    slot = _first_non_lorentz_index_slot(gauge_field)
    if slot is None:
        return None
    return gauge_field.indices[slot].kind


def _adjoint_index_slot(gauge_field: Field) -> Optional[int]:
    return _first_non_lorentz_index_slot(gauge_field)


def _unique_slot(field: Field, kind: str, *, purpose: str) -> int:
    slots = field.index_positions(kind=kind)
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to expose exactly one {kind!r} slot; "
            f"found {len(slots)}."
        )
    return slots[0]


def _default_gauge_lorentz_labels(field: Field, gauge_group: GaugeGroup, count: int):
    stem = f"{field.name}_{gauge_group.name}"
    return tuple(_symbol(f"mu_{stem}_{slot}") for slot in range(1, count + 1))


def _default_adjoint_labels(gauge_field: Field, gauge_group: GaugeGroup, count: int):
    adj_kind = _adjoint_index_kind(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz adjoint index."
        )
    stem = f"{gauge_field.name}_{gauge_group.name}"
    return tuple(_symbol(f"{adj_kind}_{stem}_{slot}") for slot in range(1, count + 1))


def _default_internal_adjoint_label(gauge_field: Field, gauge_group: GaugeGroup):
    adj_kind = _adjoint_index_kind(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz adjoint index."
        )
    return _symbol(f"{adj_kind}_mid_{gauge_field.name}_{gauge_group.name}")


def _default_ghost_labels(field: Field, index, slot: Optional[int] = None):
    return _default_index_labels(field, index, qualifier="ghost", slot=slot)


def _build_structure_constant(gauge_group: GaugeGroup, left_label, middle_label, right_label):
    if gauge_group.structure_constant is None or not callable(gauge_group.structure_constant):
        raise ValueError(
            f"Gauge group {gauge_group.name!r} needs a callable structure_constant "
            "builder for pure-gauge compilation."
        )
    return gauge_group.structure_constant(left_label, middle_label, right_label)


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


def _require_declared_field(model: Model, target, *, purpose: str) -> Field:
    """Resolve one field strictly from the parent model declarations."""
    if isinstance(target, Field):
        for field in model.fields:
            if field is target:
                return field
        raise ValueError(
            f"{purpose} requires field {target.name!r} to be declared in model.fields."
        )

    resolved = model.find_field(target)
    if resolved is None:
        raise ValueError(f"{purpose} could not resolve field {target!r} in model.fields.")
    return resolved


def _require_declared_gauge_group(model: Model, target, *, purpose: str) -> GaugeGroup:
    """Resolve one gauge group strictly from the parent model declarations."""
    if isinstance(target, GaugeGroup):
        for gauge_group in model.gauge_groups:
            if gauge_group is target:
                return gauge_group
        raise ValueError(
            f"{purpose} requires gauge group {target.name!r} to be declared in model.gauge_groups."
        )

    resolved = model.find_gauge_group(target)
    if resolved is None:
        raise ValueError(
            f"{purpose} could not resolve gauge group {target!r} in model.gauge_groups."
        )
    return resolved


def _require_field_transforms_under_gauge_group(
    field: Field,
    gauge_group: GaugeGroup,
    *,
    purpose: str,
):
    """Reject explicit group selections that do not act on the chosen field."""
    if gauge_group.abelian:
        if gauge_group.charge is None:
            raise ValueError(
                f"{purpose} cannot use abelian gauge group {gauge_group.name!r} "
                "without a declared charge label."
            )
        charge = field.quantum_numbers.get(gauge_group.charge, 0)
        if charge == 0:
            raise ValueError(
                f"{purpose} requires field {field.name!r} to carry non-zero "
                f"charge {gauge_group.charge!r} under gauge group {gauge_group.name!r}."
            )
        return

    if gauge_group.matter_representation(field) is None:
        raise ValueError(
            f"{purpose} requires field {field.name!r} to carry a declared "
            f"representation under gauge group {gauge_group.name!r}."
        )


def _resolve_covariant_gauge_groups(model: Model, *, field: Field, gauge_group=None) -> tuple[GaugeGroup, ...]:
    purpose = f"Covariant compilation for field {field.name!r}"
    if gauge_group is not None:
        if isinstance(gauge_group, (tuple, list)):
            resolved = []
            for item in gauge_group:
                group = _require_declared_gauge_group(model, item, purpose=purpose)
                _require_field_transforms_under_gauge_group(field, group, purpose=purpose)
                resolved.append(group)
            return tuple(resolved)

        resolved = _require_declared_gauge_group(model, gauge_group, purpose=purpose)
        _require_field_transforms_under_gauge_group(field, resolved, purpose=purpose)
        return (resolved,)

    matches = tuple(group for group in model.gauge_groups if _field_transforms_under_gauge_group(field, group))
    if not matches:
        raise ValueError(f"Field {field.name!r} does not transform under any declared gauge group.")
    return matches


def _spectator_identity_factor(field: Field, *, exclude_slots=()):
    factor = Expression.num(1)
    left_slot_labels = {}
    right_slot_labels = {}

    for slot, index in enumerate(field.indices):
        if slot in exclude_slots or index.kind == LORENTZ_KIND:
            continue
        left_label, right_label = _default_index_labels(field, index, slot=slot)
        factor *= index.representation.g(left_label, right_label).to_expression()
        left_slot_labels[slot] = left_label
        right_slot_labels[slot] = right_label

    return factor, left_slot_labels, right_slot_labels


def _nonabelian_rep_and_slots(field: Field, gauge_group: GaugeGroup):
    rep_info = gauge_group.matter_representation_and_slots(field)
    if rep_info is None:
        raise ValueError(
            f"Field {field.name!r} carries no representation declared for "
            f"gauge group {gauge_group.name!r}."
        )
    return rep_info


def _adjoint_slot_info(gauge_field: Field, *, purpose: str):
    adj_kind = _adjoint_index_kind(gauge_field)
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz "
            f"adjoint index for {purpose}."
        )
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot "
            f"for {purpose}."
        )
    return adj_kind, adj_slot


def _mixed_scalar_contact_slot_suffix(active_slots: tuple[int, ...]) -> str:
    if not active_slots:
        return ""
    if len(active_slots) == 1:
        return f" [slot {active_slots[0] + 1}]"
    return f" [slots {', '.join(str(slot + 1) for slot in active_slots)}]"


def _ghost_field_for_group(model: Model, gauge_group: GaugeGroup) -> Field:
    """Resolve and validate the ghost field declared for one gauge group."""
    if gauge_group.ghost_field is None:
        raise ValueError(
            f"Ghost compilation requires gauge group {gauge_group.name!r} to declare ghost_field."
        )
    ghost_field = _require_declared_field(
        model,
        gauge_group.ghost_field,
        purpose="Ghost compilation",
    )
    if ghost_field.kind != "ghost":
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to have kind='ghost'."
        )
    if ghost_field.self_conjugate:
        raise ValueError(
            f"Ghost compilation requires field {ghost_field.name!r} to be non-self-conjugate."
        )
    return ghost_field


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
) -> tuple[InteractionTerm, ...]:
    """Compile one or more fermion-gauge current interactions from model metadata.

    If the relevant gauge representation resolves to multiple index slots and the
    representation has ``slot_policy='sum'``, this returns one interaction term
    per active slot.
    """
    if fermion.kind != "fermion":
        raise ValueError(f"Expected a fermion field, got kind={fermion.kind!r}.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    mu = lorentz_label or _default_vector_label(gauge_field, gauge_group, suffix="mu")
    i_bar, i_psi = spinor_labels or _default_spinor_labels(fermion, gauge_group)
    fermion_spinor_slot = _unique_slot(
        fermion,
        SPINOR_KIND,
        purpose="Fermion gauge-current compilation",
    )
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Gauge-current compilation",
    )

    base_coupling = prefactor * gauge_group.coupling * psi_bar_gamma_psi(i_bar, i_psi, mu)

    if gauge_group.abelian:
        coupling = base_coupling * _field_charge(fermion, gauge_group)
        bar_slot_labels = {fermion_spinor_slot: i_bar}
        psi_slot_labels = {fermion_spinor_slot: i_psi}
        gauge_slot_labels = {gauge_lorentz_slot: mu}

        spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
            fermion,
            exclude_slots={fermion_spinor_slot},
        )
        coupling *= spectator_factor
        bar_slot_labels.update(spectator_left_slots)
        psi_slot_labels.update(spectator_right_slots)

        bar_labels = fermion.pack_slot_labels(bar_slot_labels)
        psi_labels = fermion.pack_slot_labels(psi_slot_labels)
        gauge_labels = gauge_field.pack_slot_labels(gauge_slot_labels)
        return (
            InteractionTerm(
                coupling=coupling,
                fields=(
                    fermion.occurrence(conjugated=True, labels=bar_labels),
                    fermion.occurrence(labels=psi_labels),
                    gauge_field.occurrence(labels=gauge_labels),
                ),
                label=label or f"{gauge_group.name}: {fermion.name} gauge current",
            ),
        )

    rep_info = gauge_group.matter_representation_and_slots(fermion)
    if rep_info is None:
        raise ValueError(
            f"Field {fermion.name!r} carries no representation declared for "
            f"gauge group {gauge_group.name!r}."
        )
    rep, rep_slots = rep_info
    adj_kind = _adjoint_index_kind(gauge_field)
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_kind is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose a non-Lorentz "
            "adjoint index."
        )
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot."
        )

    adjoint = adjoint_label or _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}")
    interactions: list[InteractionTerm] = []
    for rep_slot in rep_slots:
        coupling = base_coupling
        bar_slot_labels = {fermion_spinor_slot: i_bar}
        psi_slot_labels = {fermion_spinor_slot: i_psi}
        gauge_slot_labels = {gauge_lorentz_slot: mu, adj_slot: adjoint}
        spectator_exclusions = {fermion_spinor_slot, rep_slot}

        left_label, right_label = matter_labels or _default_matter_labels(
            fermion,
            rep.index.prefix,
            slot=rep_slot,
        )
        coupling *= rep.build_generator(adjoint, left_label, right_label)
        bar_slot_labels[rep_slot] = left_label
        psi_slot_labels[rep_slot] = right_label

        spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
            fermion,
            exclude_slots=spectator_exclusions,
        )
        coupling *= spectator_factor
        bar_slot_labels.update(spectator_left_slots)
        psi_slot_labels.update(spectator_right_slots)

        bar_labels = fermion.pack_slot_labels(bar_slot_labels)
        psi_labels = fermion.pack_slot_labels(psi_slot_labels)
        gauge_labels = gauge_field.pack_slot_labels(gauge_slot_labels)

        slot_suffix = _slot_suffix(fermion, rep_slot)
        slot_label = f"{label} [{rep.index.name}{slot_suffix}]" if label else ""
        interactions.append(
            InteractionTerm(
                coupling=coupling,
                fields=(
                    fermion.occurrence(conjugated=True, labels=bar_labels),
                    fermion.occurrence(labels=psi_labels),
                    gauge_field.occurrence(labels=gauge_labels),
                ),
                label=slot_label or f"{gauge_group.name}: {fermion.name} gauge current",
            )
        )

    return tuple(interactions)


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
    """Compile the complex-scalar gauge current/contact terms.

    If the relevant gauge representation resolves to multiple index slots and the
    representation has ``slot_policy='sum'``, the output sums over active slots:
    - current terms: one pair (phi / phi^dagger derivative placement) per slot
    - contact terms: one term per ordered slot pair (slot_i, slot_j)
    """
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Complex-scalar gauge-term compilation",
    )
    mu, nu = lorentz_labels or (
        _default_vector_label(gauge_field, gauge_group, suffix="mu"),
        _default_vector_label(gauge_field, gauge_group, suffix="nu"),
    )
    scalar_bar_slot_labels = {}
    scalar_slot_labels = {}
    gauge_slot_labels_mu = {gauge_lorentz_slot: mu}
    gauge_slot_labels_nu = {gauge_lorentz_slot: nu}
    spectator_exclusions = set()

    if gauge_group.abelian:
        charge = _field_charge(scalar, gauge_group)
        current_base = current_prefactor * gauge_group.coupling * charge
        contact_coupling = contact_prefactor * ((gauge_group.coupling * charge) ** 2)
    else:
        rep, rep_slots = _nonabelian_rep_and_slots(scalar, gauge_group)
        adj_kind, adj_slot = _adjoint_slot_info(
            gauge_field,
            purpose="Complex-scalar gauge-term compilation",
        )
        adjoint_mu, adjoint_nu = adjoint_labels or (
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_1"),
            _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_2"),
        )
        gauge_slot_labels_mu[adj_slot] = adjoint_mu
        gauge_slot_labels_nu[adj_slot] = adjoint_nu

        # Current terms: sum over active slots by emitting one term per slot.
        current_terms: list[InteractionTerm] = []
        for rep_slot in rep_slots:
            left_label, right_label = matter_labels or _default_matter_labels(
                scalar,
                rep.index.prefix,
                slot=rep_slot,
            )
            generator_mu = rep.build_generator(adjoint_mu, left_label, right_label)
            current_base = current_prefactor * gauge_group.coupling * generator_mu

            scalar_bar_slot_labels = {rep_slot: left_label}
            scalar_slot_labels = {rep_slot: right_label}
            spectator_exclusions = {rep_slot}
            spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
                scalar,
                exclude_slots=spectator_exclusions,
            )
            current_base *= spectator_factor
            scalar_bar_slot_labels.update(spectator_left_slots)
            scalar_slot_labels.update(spectator_right_slots)

            scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slot_labels)
            scalar_labels = scalar.pack_slot_labels(scalar_slot_labels)
            gauge_labels_mu = gauge_field.pack_slot_labels(gauge_slot_labels_mu)

            slot_suffix = _slot_suffix(scalar, rep_slot)
            prefix = (label_prefix + " " if label_prefix else "")
            current_terms.append(
                InteractionTerm(
                    coupling=current_base,
                    fields=(
                        scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
                        scalar.occurrence(labels=scalar_labels),
                        gauge_field.occurrence(labels=gauge_labels_mu),
                    ),
                    derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
                    label=prefix + f"{gauge_group.name}: scalar current (+){slot_suffix}",
                )
            )
            current_terms.append(
                InteractionTerm(
                    coupling=-current_base,
                    fields=(
                        scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
                        scalar.occurrence(labels=scalar_labels),
                        gauge_field.occurrence(labels=gauge_labels_mu),
                    ),
                    derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
                    label=prefix + f"{gauge_group.name}: scalar current (-){slot_suffix}",
                )
            )

        # Contact terms: sum over ordered slot pairs (i, j).
        contact_terms: list[InteractionTerm] = []
        for slot_i in rep_slots:
            for slot_j in rep_slots:
                scalar_bar_slot_labels = {}
                scalar_slot_labels = {}

                left_i, right_i = matter_labels or _default_matter_labels(
                    scalar,
                    rep.index.prefix,
                    slot=slot_i,
                )
                scalar_bar_slot_labels[slot_i] = left_i
                scalar_slot_labels[slot_i] = right_i

                if slot_j == slot_i:
                    middle = internal_label or _symbol(
                        f"{rep.index.prefix}_mid_{scalar.name}_{gauge_group.name}{_slot_suffix(scalar, slot_i)}"
                    )
                    generator_pair = (
                        rep.build_generator(adjoint_mu, left_i, middle)
                        * rep.build_generator(adjoint_nu, middle, right_i)
                    )
                    exclude_slots = {slot_i}
                else:
                    left_j, right_j = matter_labels or _default_matter_labels(
                        scalar,
                        rep.index.prefix,
                        slot=slot_j,
                    )
                    scalar_bar_slot_labels[slot_j] = left_j
                    scalar_slot_labels[slot_j] = right_j
                    generator_pair = (
                        rep.build_generator(adjoint_mu, left_i, right_i)
                        * rep.build_generator(adjoint_nu, left_j, right_j)
                    )
                    exclude_slots = {slot_i, slot_j}

                contact_coupling = contact_prefactor * (gauge_group.coupling ** 2) * generator_pair

                spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
                    scalar,
                    exclude_slots=exclude_slots,
                )
                contact_coupling *= spectator_factor
                scalar_bar_slot_labels.update(spectator_left_slots)
                scalar_slot_labels.update(spectator_right_slots)

                scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slot_labels)
                scalar_labels = scalar.pack_slot_labels(scalar_slot_labels)
                gauge_labels_mu = gauge_field.pack_slot_labels(gauge_slot_labels_mu)
                gauge_labels_nu = gauge_field.pack_slot_labels(gauge_slot_labels_nu)

                prefix = (label_prefix + " " if label_prefix else "")
                contact_terms.append(
                    InteractionTerm(
                        coupling=contact_coupling * scalar_gauge_contact(mu, nu),
                        fields=(
                            scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
                            scalar.occurrence(labels=scalar_labels),
                            gauge_field.occurrence(labels=gauge_labels_mu),
                            gauge_field.occurrence(labels=gauge_labels_nu),
                        ),
                        label=prefix + f"{gauge_group.name}: scalar contact [slots {slot_i+1},{slot_j+1}]",
                    )
                )

        return tuple(current_terms + contact_terms)

    spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
        scalar,
        exclude_slots=spectator_exclusions,
    )
    current_base *= spectator_factor
    contact_coupling *= spectator_factor
    scalar_bar_slot_labels.update(spectator_left_slots)
    scalar_slot_labels.update(spectator_right_slots)

    scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slot_labels)
    scalar_labels = scalar.pack_slot_labels(scalar_slot_labels)
    gauge_labels_mu = gauge_field.pack_slot_labels(gauge_slot_labels_mu)
    gauge_labels_nu = gauge_field.pack_slot_labels(gauge_slot_labels_nu)

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


def compile_mixed_complex_scalar_contact_terms(
    *,
    scalar: Field,
    left_gauge_group: GaugeGroup,
    left_gauge_field: Field,
    right_gauge_group: GaugeGroup,
    right_gauge_field: Field,
    lorentz_labels=None,
    left_adjoint_label=None,
    right_adjoint_label=None,
    contact_prefactor=1,
    label_prefix: str = "",
):
    """Compile ordered cross-group contact terms from ``(D_mu phi)^dagger (D^mu phi)``.

    This covers the mixed products that arise when a complex scalar transforms
    under more than one gauge group:

    ``(g_r A^r_mu action_r(phi))^dagger (g_s A^{s,mu} action_s(phi))``

    The two group actions are representation-dependent:
    abelian groups contribute a charge factor, while non-abelian groups
    contribute one generator per active representation slot.
    """
    if scalar.kind != "scalar" or scalar.self_conjugate:
        raise ValueError("Complex-scalar gauge terms require a non-self-conjugate scalar field.")
    if left_gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={left_gauge_field.kind!r}.")
    if right_gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={right_gauge_field.kind!r}.")

    left_gauge_lorentz_slot = _unique_slot(
        left_gauge_field,
        LORENTZ_KIND,
        purpose="Mixed scalar contact compilation (left gauge field)",
    )
    right_gauge_lorentz_slot = _unique_slot(
        right_gauge_field,
        LORENTZ_KIND,
        purpose="Mixed scalar contact compilation (right gauge field)",
    )
    mu, nu = lorentz_labels or (
        _default_vector_label(left_gauge_field, left_gauge_group, suffix="mu"),
        _default_vector_label(right_gauge_field, right_gauge_group, suffix="nu"),
    )
    left_gauge_slot_labels = {left_gauge_lorentz_slot: mu}
    right_gauge_slot_labels = {right_gauge_lorentz_slot: nu}

    if left_gauge_group.abelian:
        left_actions = ((None, None),)
        left_charge = _field_charge(scalar, left_gauge_group)
        left_adj = None
    else:
        left_rep, left_rep_slots = _nonabelian_rep_and_slots(scalar, left_gauge_group)
        left_adj_kind, left_adj_slot = _adjoint_slot_info(
            left_gauge_field,
            purpose="Mixed scalar contact compilation (left gauge field)",
        )
        left_adj = left_adjoint_label or _symbol(
            f"{left_adj_kind}_{left_gauge_field.name}_{left_gauge_group.name}_mix"
        )
        left_gauge_slot_labels[left_adj_slot] = left_adj
        left_actions = tuple((left_rep, slot) for slot in left_rep_slots)
        left_charge = None

    if right_gauge_group.abelian:
        right_actions = ((None, None),)
        right_charge = _field_charge(scalar, right_gauge_group)
        right_adj = None
    else:
        right_rep, right_rep_slots = _nonabelian_rep_and_slots(scalar, right_gauge_group)
        right_adj_kind, right_adj_slot = _adjoint_slot_info(
            right_gauge_field,
            purpose="Mixed scalar contact compilation (right gauge field)",
        )
        right_adj = right_adjoint_label or _symbol(
            f"{right_adj_kind}_{right_gauge_field.name}_{right_gauge_group.name}_mix"
        )
        right_gauge_slot_labels[right_adj_slot] = right_adj
        right_actions = tuple((right_rep, slot) for slot in right_rep_slots)
        right_charge = None

    left_gauge_labels = left_gauge_field.pack_slot_labels(left_gauge_slot_labels)
    right_gauge_labels = right_gauge_field.pack_slot_labels(right_gauge_slot_labels)
    prefix = label_prefix + " " if label_prefix else ""

    contact_terms: list[InteractionTerm] = []
    for left_rep_slot in left_actions:
        for right_rep_slot in right_actions:
            scalar_bar_slot_labels = {}
            scalar_slot_labels = {}
            active_slots: list[int] = []
            exclude_slots = set()
            contact_coupling = contact_prefactor * left_gauge_group.coupling * right_gauge_group.coupling

            if left_gauge_group.abelian:
                contact_coupling *= left_charge
                left_rep = None
                left_slot = None
            else:
                left_rep, left_slot = left_rep_slot
                active_slots.append(left_slot)
                exclude_slots.add(left_slot)

            if right_gauge_group.abelian:
                contact_coupling *= right_charge
                right_rep = None
                right_slot = None
            else:
                right_rep, right_slot = right_rep_slot
                active_slots.append(right_slot)
                exclude_slots.add(right_slot)

            if (
                left_rep is not None
                and right_rep is not None
                and left_slot == right_slot
            ):
                left_label, right_label = _default_matter_labels(
                    scalar,
                    left_rep.index.prefix,
                    slot=left_slot,
                )
                middle = _symbol(
                    f"{left_rep.index.prefix}_mid_{scalar.name}_{left_gauge_group.name}_{right_gauge_group.name}"
                    f"{_slot_suffix(scalar, left_slot)}"
                )
                contact_coupling *= (
                    left_rep.build_generator(left_adj, left_label, middle)
                    * right_rep.build_generator(right_adj, middle, right_label)
                )
                scalar_bar_slot_labels[left_slot] = left_label
                scalar_slot_labels[left_slot] = right_label
            else:
                if left_rep is not None:
                    left_label, right_label = _default_matter_labels(
                        scalar,
                        left_rep.index.prefix,
                        slot=left_slot,
                    )
                    contact_coupling *= left_rep.build_generator(left_adj, left_label, right_label)
                    scalar_bar_slot_labels[left_slot] = left_label
                    scalar_slot_labels[left_slot] = right_label

                if right_rep is not None:
                    left_label, right_label = _default_matter_labels(
                        scalar,
                        right_rep.index.prefix,
                        slot=right_slot,
                    )
                    contact_coupling *= right_rep.build_generator(right_adj, left_label, right_label)
                    scalar_bar_slot_labels[right_slot] = left_label
                    scalar_slot_labels[right_slot] = right_label

            spectator_factor, spectator_left_slots, spectator_right_slots = _spectator_identity_factor(
                scalar,
                exclude_slots=exclude_slots,
            )
            contact_coupling *= spectator_factor
            scalar_bar_slot_labels.update(spectator_left_slots)
            scalar_slot_labels.update(spectator_right_slots)

            scalar_bar_labels = scalar.pack_slot_labels(scalar_bar_slot_labels)
            scalar_labels = scalar.pack_slot_labels(scalar_slot_labels)
            unique_active_slots = tuple(dict.fromkeys(active_slots))
            contact_terms.append(
                InteractionTerm(
                    coupling=contact_coupling * scalar_gauge_contact(mu, nu),
                    fields=(
                        scalar.occurrence(conjugated=True, labels=scalar_bar_labels),
                        scalar.occurrence(labels=scalar_labels),
                        left_gauge_field.occurrence(labels=left_gauge_labels),
                        right_gauge_field.occurrence(labels=right_gauge_labels),
                    ),
                    label=(
                        prefix
                        + f"{left_gauge_group.name} x {right_gauge_group.name}: scalar mixed contact"
                        + _mixed_scalar_contact_slot_suffix(unique_active_slots)
                    ),
                )
            )

    return tuple(contact_terms)


def compile_gauge_kinetic_bilinear_terms(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the two-point part of ``-1/4 F_{mu nu} F^{mu nu}``."""
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}")
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}")
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Gauge kinetic compilation",
    )
    identity_factor, left_slots, right_slots = _spectator_identity_factor(
        gauge_field,
        exclude_slots={gauge_lorentz_slot},
    )

    left_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, **left_slots})
    right_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, **right_slots})
    shared_fields = (
        gauge_field.occurrence(labels=left_field_labels),
        gauge_field.occurrence(labels=right_field_labels),
    )
    prefix = label_prefix + " " if label_prefix else ""

    return (
        InteractionTerm(
            coupling=-coefficient * _HALF * identity_factor * lorentz_metric(alpha, beta),
            fields=shared_fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho),
                DerivativeAction(target=1, lorentz_index=rho),
            ),
            label=prefix + f"{gauge_group.name}: gauge kinetic bilinear (metric)",
        ),
        InteractionTerm(
            coupling=(
                coefficient
                * _HALF
                * identity_factor
                * lorentz_metric(rho_left, beta)
                * lorentz_metric(rho_right, alpha)
            ),
            fields=shared_fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=prefix + f"{gauge_group.name}: gauge kinetic bilinear (cross)",
        ),
    )


def compile_yang_mills_cubic_term(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the cubic Yang-Mills term from ``-1/4 F^a_{mu nu} F^{a mu nu}``."""
    if gauge_group.abelian:
        raise ValueError("Abelian gauge groups do not have Yang-Mills cubic self-interactions.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta, gamma = _default_gauge_lorentz_labels(gauge_field, gauge_group, 3)
    adj_left, adj_middle, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 3)
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Yang-Mills cubic compilation",
    )
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot."
        )
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}_cubic")
    coupling = (
        coefficient
        * gauge_group.coupling
        * _build_structure_constant(gauge_group, adj_left, adj_middle, adj_right)
        * lorentz_metric(alpha, gamma)
        * lorentz_metric(rho, beta)
    )

    return InteractionTerm(
        coupling=coupling,
        fields=(
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, adj_slot: adj_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, adj_slot: adj_middle})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: gamma, adj_slot: adj_right})
            ),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: Yang-Mills cubic",
    )


def compile_yang_mills_quartic_term(
    *,
    gauge_group: GaugeGroup,
    gauge_field: Field,
    coefficient=1,
    label_prefix: str = "",
):
    """Compile the quartic Yang-Mills term from ``-1/4 F^a_{mu nu} F^{a mu nu}``."""
    if gauge_group.abelian:
        raise ValueError("Abelian gauge groups do not have Yang-Mills quartic self-interactions.")
    if gauge_field.kind != "vector":
        raise ValueError(f"Expected a vector gauge field, got kind={gauge_field.kind!r}.")

    alpha, beta, gamma, delta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 4)
    adj_left, adj_mid_left, adj_mid_right, adj_right = _default_adjoint_labels(gauge_field, gauge_group, 4)
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Yang-Mills quartic compilation",
    )
    adj_slot = _adjoint_index_slot(gauge_field)
    if adj_slot is None:
        raise ValueError(
            f"Gauge field {gauge_field.name!r} does not expose an adjoint slot."
        )
    internal = _default_internal_adjoint_label(gauge_field, gauge_group)
    coupling = (
        -coefficient
        * _QUARTER
        * (gauge_group.coupling ** 2)
        * _build_structure_constant(gauge_group, adj_left, adj_mid_left, internal)
        * _build_structure_constant(gauge_group, adj_mid_right, adj_right, internal)
        * lorentz_metric(alpha, gamma)
        * lorentz_metric(beta, delta)
    )

    return InteractionTerm(
        coupling=coupling,
        fields=(
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, adj_slot: adj_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, adj_slot: adj_mid_left})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: gamma, adj_slot: adj_mid_right})
            ),
            gauge_field.occurrence(
                labels=gauge_field.pack_slot_labels({gauge_lorentz_slot: delta, adj_slot: adj_right})
            ),
        ),
        label=(label_prefix + " " if label_prefix else "") + f"{gauge_group.name}: Yang-Mills quartic",
    )


def compile_gauge_kinetic_term(model: Model, term: GaugeKineticTerm) -> tuple[InteractionTerm, ...]:
    """Compile ``-1/4 F_{mu nu} F^{mu nu}`` for one declared gauge group.

    For abelian groups this yields only the gauge-boson bilinear.
    For non-abelian groups it also appends the Yang-Mills cubic and quartic
    self-interaction terms.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Gauge-kinetic compilation",
    )

    gauge_field = model.gauge_boson_field(gauge_group)
    label_prefix = term.label or f"-1/4 {gauge_group.name} field strength squared"

    interactions = list(
        compile_gauge_kinetic_bilinear_terms(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    if gauge_group.abelian:
        return tuple(interactions)

    interactions.append(
        compile_yang_mills_cubic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    interactions.append(
        compile_yang_mills_quartic_term(
            gauge_group=gauge_group,
            gauge_field=gauge_field,
            coefficient=term.coefficient,
            label_prefix=label_prefix,
        )
    )
    return tuple(interactions)


def compile_gauge_fixing_term(model: Model, term: GaugeFixingTerm) -> tuple[InteractionTerm, ...]:
    """Compile the ordinary linear-covariant gauge-fixing bilinear.

    The supported form is ``-(coefficient / 2 xi) (partial.A)^2`` for one
    declared gauge group.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Gauge-fixing compilation",
    )
    if _is_explicit_zero(term.xi):
        raise ValueError("Gauge-fixing compilation requires xi to be non-zero.")
    gauge_field = model.gauge_boson_field(gauge_group)
    if gauge_field.kind != "vector":
        raise ValueError(f"Gauge-fixing compilation requires a vector field, got kind={gauge_field.kind!r}.")
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Gauge-fixing compilation",
    )

    alpha, beta = _default_gauge_lorentz_labels(gauge_field, gauge_group, 2)
    rho_left = _symbol(f"rho_left_{gauge_field.name}_{gauge_group.name}_gf")
    rho_right = _symbol(f"rho_right_{gauge_field.name}_{gauge_group.name}_gf")
    identity_factor, left_slots, right_slots = _spectator_identity_factor(
        gauge_field,
        exclude_slots={gauge_lorentz_slot},
    )

    coupling = (
        -term.coefficient
        * _HALF
        / term.xi
        * identity_factor
        * lorentz_metric(alpha, rho_left)
        * lorentz_metric(beta, rho_right)
    )
    left_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: alpha, **left_slots})
    right_field_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: beta, **right_slots})
    label = term.label or f"-(1/2 {term.xi}) ({gauge_group.name} gauge fixing)"

    return (
        InteractionTerm(
            coupling=coupling,
            fields=(
                gauge_field.occurrence(labels=left_field_labels),
                gauge_field.occurrence(labels=right_field_labels),
            ),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=label,
        ),
    )


def compile_ghost_term(model: Model, term: GhostTerm) -> tuple[InteractionTerm, ...]:
    """Compile the ordinary unbroken Faddeev-Popov ghost sector.

    For the current conventions this corresponds to:
    ``-cbar^a partial^mu(D_mu c)^a = (partial cbar)(partial c) - g f (partial cbar) A c``.
    """
    gauge_group = _require_declared_gauge_group(
        model,
        term.gauge_group,
        purpose="Ghost compilation",
    )
    if gauge_group.abelian:
        raise ValueError(
            f"Ghost compilation is only supported for non-abelian gauge groups; got {gauge_group.name!r}."
        )

    gauge_field = model.gauge_boson_field(gauge_group)
    ghost_field = _ghost_field_for_group(model, gauge_group)

    adj_kind, gauge_adj_slot = _adjoint_slot_info(
        gauge_field,
        purpose="Ghost compilation",
    )
    gauge_lorentz_slot = _unique_slot(
        gauge_field,
        LORENTZ_KIND,
        purpose="Ghost compilation",
    )
    ghost_adj_slot = _unique_slot(
        ghost_field,
        adj_kind,
        purpose="Ghost compilation",
    )

    mu = _default_vector_label(gauge_field, gauge_group, suffix="mu")
    nu = _default_vector_label(gauge_field, gauge_group, suffix="nu_ghost")
    rho = _symbol(f"rho_{gauge_field.name}_{gauge_group.name}_ghost")
    a_bar, a_ghost = _default_ghost_labels(
        ghost_field,
        ghost_field.indices[ghost_adj_slot],
        slot=ghost_adj_slot,
    )
    a_gauge = _symbol(f"{adj_kind}_{gauge_field.name}_{gauge_group.name}_ghost")

    ghost_bar_labels = ghost_field.pack_slot_labels({ghost_adj_slot: a_bar})
    ghost_labels = ghost_field.pack_slot_labels({ghost_adj_slot: a_ghost})
    gauge_labels = gauge_field.pack_slot_labels({gauge_lorentz_slot: mu, gauge_adj_slot: a_gauge})
    label_prefix = term.label or f"{gauge_group.name} Faddeev-Popov ghosts"

    ghost_bilinear = InteractionTerm(
        coupling=(
            term.coefficient
            * ghost_field.indices[ghost_adj_slot].representation.g(a_bar, a_ghost).to_expression()
            * lorentz_metric(mu, nu)
        ),
        fields=(
            ghost_field.occurrence(conjugated=True, labels=ghost_bar_labels),
            ghost_field.occurrence(labels=ghost_labels),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=1, lorentz_index=nu),
        ),
        label=label_prefix + " bilinear",
    )

    ghost_gauge = InteractionTerm(
        coupling=(
            -term.coefficient
            * gauge_group.coupling
            * _build_structure_constant(gauge_group, a_bar, a_gauge, a_ghost)
            * lorentz_metric(rho, mu)
        ),
        fields=(
            ghost_field.occurrence(conjugated=True, labels=ghost_bar_labels),
            gauge_field.occurrence(labels=gauge_labels),
            ghost_field.occurrence(labels=ghost_labels),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=label_prefix + " gauge interaction",
    )

    return (ghost_bilinear, ghost_gauge)


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

                interactions.extend(
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
    """Compile the gauge-interaction part of ``psibar i gamma^mu D_mu psi``.

    One model declaration can expand into several interactions when the same
    fermion transforms under multiple declared gauge groups.
    """
    fermion = _require_declared_field(
        model,
        term.field,
        purpose="Dirac kinetic compilation",
    )
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
        interactions.extend(
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
    """Compile the gauge-interaction part of ``(D_mu phi)^dagger (D^mu phi)``.

    The output contains both the current term and the two-gauge contact term
    for each applicable gauge group.
    """
    scalar = _require_declared_field(
        model,
        term.field,
        purpose="Complex-scalar kinetic compilation",
    )
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
    resolved_gauge_fields = []
    for gauge_group in gauge_groups:
        gauge_field = model.gauge_boson_field(gauge_group)
        resolved_gauge_fields.append((gauge_group, gauge_field))
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

    for left_idx, (left_gauge_group, left_gauge_field) in enumerate(resolved_gauge_fields):
        for right_idx, (right_gauge_group, right_gauge_field) in enumerate(resolved_gauge_fields):
            if left_idx == right_idx:
                continue
            interactions.extend(
                compile_mixed_complex_scalar_contact_terms(
                    scalar=scalar,
                    left_gauge_group=left_gauge_group,
                    left_gauge_field=left_gauge_field,
                    right_gauge_group=right_gauge_group,
                    right_gauge_field=right_gauge_field,
                    contact_prefactor=term.coefficient,
                    label_prefix=label_prefix,
                )
            )
    return tuple(interactions)


def compile_covariant_terms(model: Model) -> tuple[InteractionTerm, ...]:
    """Compile all declared physical kinetic terms in a model.

    This is the main entry point for the convention-fixed physical compiler:
    matter covariant terms and pure-gauge kinetic terms are flattened into the
    ordinary ``InteractionTerm`` objects consumed by the core engine.
    """
    interactions: list[InteractionTerm] = []
    buckets = model._declared_piece_buckets()

    all_covariant_terms = model.covariant_terms + buckets["covariant_terms"]
    all_gauge_kinetic_terms = model.gauge_kinetic_terms + buckets["gauge_kinetic_terms"]
    all_gauge_fixing_terms = model.gauge_fixing_terms + buckets["gauge_fixing_terms"]
    all_ghost_terms = model.ghost_terms + buckets["ghost_terms"]

    for term in all_covariant_terms:
        if isinstance(term, DiracKineticTerm):
            interactions.extend(compile_dirac_kinetic_term(model, term))
            continue
        if isinstance(term, ComplexScalarKineticTerm):
            interactions.extend(compile_complex_scalar_kinetic_term(model, term))
            continue
        raise TypeError(f"Unsupported covariant term type: {type(term)!r}")

    for term in all_gauge_kinetic_terms:
        interactions.extend(compile_gauge_kinetic_term(model, term))

    for term in all_gauge_fixing_terms:
        interactions.extend(compile_gauge_fixing_term(model, term))

    for term in all_ghost_terms:
        interactions.extend(compile_ghost_term(model, term))

    return tuple(interactions)


def with_compiled_covariant_terms(model: Model) -> Model:
    """Return a copy of a model with compiled physical kinetic terms appended.

    The returned model has empty declaration slots (covariant_terms,
    gauge_kinetic_terms, gauge_fixing_terms, ghost_terms) so that a
    subsequent call to ``Model.lagrangian()`` does not re-compile and
    double-count the same terms.
    """
    compiled = compile_covariant_terms(model)
    manual_decl_terms = tuple(
        term for term in model.lagrangian_decl.source_terms if isinstance(term, InteractionTerm)
    )
    return replace(
        model,
        interactions=model.interactions + compiled,
        lagrangian_decl=type(model.lagrangian_decl)(source_terms=manual_decl_terms),
        covariant_terms=(),
        gauge_kinetic_terms=(),
        gauge_fixing_terms=(),
        ghost_terms=(),
    )
