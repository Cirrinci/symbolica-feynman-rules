"""
Feynman vertex rule derivation via canonical quantization -- mooooreeee Symbolica (attempt :D)

The implementation follows the canonical-quantization/Wick-contraction logic,
but works directly at the symbolic contraction level rather than constructing
full operator-valued expressions explicitly.

Current scope:
    - bosonic polynomial interaction terms
    - permutation-summed Wick contractions
    - derivative interactions via momentum factors
    - fermionic permutation signs and role-aware contractions

Pipeline:
    1. Select an interaction monomial
    2. Sum over contractions with external legs
    3. Evaluate derivatives as momentum factors per contraction
    4. Replace the plane-wave factor by momentum conservation
    5. Strip external wavefunctions
    6. Multiply by i to obtain the vertex factor
"""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Literal, Optional, Sequence

from symbolica import S, Expression
from symbolica.community.spenso import Representation
from symbolica.community.idenso import simplify_metrics
from model_schema import (
    ConcreteIndexSlot,
    ExternalLeg,
    FieldRole,
    InteractionTerm,
    default_external_legs_for_interaction,
    default_leg_slot_labels,
    external_legs_to_legacy,
    index_bindings_from_slots,
    normalize_concrete_index_slots,
    ordered_index_slots,
    primary_index_label_from_slots,
    interaction_term_to_legacy,
    merge_index_bindings,
    normalize_index_bindings,
    primary_binding_labels,
    normalize_role,
)
from spenso_structures import SPINOR_KIND

# ---------------------------------------------------------------------------
# Module-level Symbolica symbols + Spenso bispinor representation
# ---------------------------------------------------------------------------

phi, psi, adag = S("phi", "psi", "adag")
U = S("U")
UF = S("UF")
UbarF = S("UbarF")
gamma = S("gamma")
delta = S("delta", is_symmetric=True)
bis = Representation.bis(4)   # 4D Dirac bispinor representation
mink = Representation.mink(4)  # 4D Minkowski (Lorentz) representation
Delta = S("Delta")
Dot = S("Dot")
pcomp = S("pcomp")
D = S("D")

I = Expression.I
pi = Expression.PI


'''
!!!NOTE!!!
----
For statistics="fermion", this implementation includes Grassmann permutation
signs and role-aware contractions, with external wavefunctions UF/UbarF.
It does not yet build full gamma/index chains automatically from a Lagrangian.
'''
Statistics = Literal["boson", "fermion"]
FERMION_ROLES = ("psi", "psibar")


@dataclass(frozen=True)
class _FieldSlot:
    position: int
    species: object
    role: Optional[FieldRole]
    slot_labels: Optional[tuple]
    index_slots: tuple[ConcreteIndexSlot, ...] = ()

    @property
    def role_name(self) -> Optional[str]:
        return self.role.name if self.role is not None else None

    @property
    def is_fermion(self) -> bool:
        return self.role is not None and self.role.is_fermion

    @property
    def spinor_label(self):
        return primary_index_label_from_slots(self.index_slots, SPINOR_KIND)

    def index_slot_at(self, declared_position: int) -> Optional[ConcreteIndexSlot]:
        if declared_position < 0 or declared_position >= len(self.index_slots):
            return None
        return self.index_slots[declared_position]

    def compatible_with(self, leg: "_ExternalLegSlot") -> bool:
        if self.role is None or leg.role is None:
            return True
        return self.role.matches(leg.role)


@dataclass(frozen=True)
class _ExternalLegSlot:
    position: int
    species: object
    momentum: object
    role: Optional[FieldRole]
    slot_labels: Optional[tuple]
    spin: object = None
    index_slots: tuple[ConcreteIndexSlot, ...] = ()

    @property
    def role_name(self) -> Optional[str]:
        return self.role.name if self.role is not None else None

    @property
    def spinor_label(self):
        return primary_index_label_from_slots(self.index_slots, SPINOR_KIND)

    def index_slot_at(self, declared_position: int) -> Optional[ConcreteIndexSlot]:
        if declared_position < 0 or declared_position >= len(self.index_slots):
            return None
        return self.index_slots[declared_position]


@dataclass(frozen=True)
class _DerivativeInstruction:
    index: object
    target_slot: int


@dataclass(frozen=True)
class _OpenIndexSlot:
    field_slot_position: int
    index_slot: ConcreteIndexSlot


@dataclass(frozen=True)
class _PreparedSlotSide:
    slot_labels: Optional[tuple]
    index_slots: Optional[tuple[tuple[ConcreteIndexSlot, ...], ...]]
    spinor_labels: Optional[tuple[object, ...]]


@dataclass(frozen=True)
class _NormalizedContractionInput:
    coupling: object
    statistics: Statistics
    field_slots: tuple[_FieldSlot, ...]
    external_legs: tuple[_ExternalLegSlot, ...]
    derivatives: tuple[_DerivativeInstruction, ...]

    @property
    def num_slots(self) -> int:
        return len(self.field_slots)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

#very naive i know
def plane_wave(p, x):
    """exp(-i p.x)"""
    return Expression.EXP(-I * Dot(p, x))


# ---------------------------------------------------------------------------
# Fermion helpers
# ---------------------------------------------------------------------------

def _role_name(value):
    if hasattr(value, "role_name"):
        return value.role_name
    role = normalize_role(value)
    return role.name if role is not None else None


def _spinor_label(value):
    if hasattr(value, "spinor_label"):
        return value.spinor_label
    return value


def _index_slots(value):
    if hasattr(value, "index_slots"):
        return value.index_slots
    return ()


def factor_leg_compatible(field_slot, leg):
    """Compatibility check for matching one field slot with one external leg.

    Species compatibility is intentionally left symbolic via delta(alpha, beta).
    This helper enforces only optional role-level constraints through the
    normalized slot/leg objects.
    """
    return field_slot.compatible_with(leg)


def _fermion_slots_from_roles(field_roles):
    """Return positions of fermionic fields in the interaction ordering."""
    return [i for i, role in enumerate(field_roles) if _role_name(role) in FERMION_ROLES]


def _group_spinor_slots(field_spinor_indices):
    """Group slots by the spinor label carried by the field factor."""
    groups = {}
    for i, si in enumerate(field_spinor_indices):
        spinor_label = _spinor_label(si)
        if spinor_label is None:
            continue
        groups.setdefault(str(spinor_label), []).append(i)
    return groups


def _normalize_slot_label_entry(entry):
    return normalize_index_bindings(entry)


def _coerce_slot_label_structure(slot_labels, *, expected_length, label_name):
    if slot_labels is None:
        return None
    if len(slot_labels) != expected_length:
        raise ValueError(f"{label_name} must have the same length as alphas/ps")
    return [
        _normalize_slot_label_entry(entry)
        for entry in slot_labels
    ]


def _merge_slot_label_structures(
    base_slot_labels,
    spinor_indices,
    roles,
    *,
    expected_length,
    label_name,
):
    merged = _coerce_slot_label_structure(
        base_slot_labels,
        expected_length=expected_length,
        label_name=label_name,
    )
    if merged is None:
        merged = [None] * expected_length

    if spinor_indices is None:
        return merged if any(entry is not None for entry in merged) else None

    if len(spinor_indices) != expected_length:
        raise ValueError(f"{label_name} spinor indices must have the same length as alphas/ps")

    for i, spinor_index in enumerate(spinor_indices):
        if spinor_index is None:
            continue
        if roles is not None and _role_name(roles[i]) not in FERMION_ROLES:
            continue

        existing = primary_binding_labels(merged[i], SPINOR_KIND)
        if existing is not None and existing != (spinor_index,):
            raise ValueError(
                f"Conflicting spinor label specification at slot {i}: "
                f"{existing} vs {(spinor_index,)}"
            )
        if existing == (spinor_index,):
            continue
        merged[i] = merge_index_bindings(
            merged[i],
            ((SPINOR_KIND, (spinor_index,)),),
        )

    return merged if any(entry is not None for entry in merged) else None


def _coerce_index_slot_structure(index_slots, *, expected_length, label_name):
    if index_slots is None:
        return None
    if len(index_slots) != expected_length:
        raise ValueError(f"{label_name} must have the same length as alphas/ps")

    coerced = []
    for entry in index_slots:
        normalized = normalize_concrete_index_slots(entry)
        coerced.append(tuple(normalized) if normalized is not None else ())
    return tuple(coerced)


def _slot_labels_from_index_slot_structure(index_slots):
    if index_slots is None:
        return None
    slot_labels = tuple(index_bindings_from_slots(entry) for entry in index_slots)
    if not any(entry is not None for entry in slot_labels):
        return None
    return slot_labels


def _effective_index_slots(slot_labels, explicit_index_slots):
    if explicit_index_slots is not None and slot_labels is None:
        return tuple(explicit_index_slots)
    if explicit_index_slots is not None:
        expected_indices = tuple(index_slot.index_type for index_slot in explicit_index_slots)
        return ordered_index_slots(slot_labels, expected_indices=expected_indices)
    return ordered_index_slots(slot_labels)


def _primary_index_labels_from_structure(index_slots, index_type_or_alias):
    if index_slots is None:
        return None
    labels = []
    found = False
    for entry in index_slots:
        label = primary_index_label_from_slots(entry, index_type_or_alias)
        labels.append(label)
        if label is not None:
            found = True
    return tuple(labels) if found else None


def _index_slot_structure_has_type(index_slots, index_type_or_alias):
    if index_slots is None:
        return False
    return any(
        primary_index_label_from_slots(entry, index_type_or_alias) is not None
        for entry in index_slots
    )


def _prepare_slot_side(
    *,
    base_slot_labels,
    explicit_index_slots,
    spinor_indices,
    roles,
    expected_length,
    label_name,
) -> _PreparedSlotSide:
    normalized_slot_labels = _coerce_slot_label_structure(
        base_slot_labels,
        expected_length=expected_length,
        label_name=label_name,
    )
    explicit_index_slots = _coerce_index_slot_structure(
        explicit_index_slots,
        expected_length=expected_length,
        label_name=f"{label_name}_index_slots",
    )
    if normalized_slot_labels is None and explicit_index_slots is not None:
        normalized_slot_labels = _slot_labels_from_index_slot_structure(explicit_index_slots)

    merged_slot_labels = _merge_slot_label_structures(
        normalized_slot_labels,
        spinor_indices,
        roles,
        expected_length=expected_length,
        label_name=label_name,
    )

    if merged_slot_labels is None and explicit_index_slots is None:
        return _PreparedSlotSide(
            slot_labels=None,
            index_slots=None,
            spinor_labels=None,
        )

    effective_index_slots = []
    for position in range(expected_length):
        merged_entry = merged_slot_labels[position] if merged_slot_labels is not None else None
        explicit_entry = explicit_index_slots[position] if explicit_index_slots is not None else None
        effective_index_slots.append(_effective_index_slots(merged_entry, explicit_entry))
    effective_index_slots = tuple(effective_index_slots)

    effective_slot_labels = merged_slot_labels
    if effective_slot_labels is None:
        effective_slot_labels = _slot_labels_from_index_slot_structure(effective_index_slots)

    return _PreparedSlotSide(
        slot_labels=effective_slot_labels,
        index_slots=effective_index_slots,
        spinor_labels=_primary_index_labels_from_structure(effective_index_slots, SPINOR_KIND),
    )


def _open_index_slots(field_slots):
    if field_slots is None:
        return []

    counts = Counter()
    for field_slot in field_slots:
        for index_slot in _index_slots(field_slot):
            if index_slot.label is None:
                continue
            counts[(index_slot.index_type, str(index_slot.label))] += 1

    open_labels = []
    for field_slot in field_slots:
        for index_slot in _index_slots(field_slot):
            if index_slot.label is None:
                continue
            if counts[(index_slot.index_type, str(index_slot.label))] == 1:
                open_labels.append(
                    _OpenIndexSlot(
                        field_slot_position=field_slot.position,
                        index_slot=index_slot,
                    )
                )
    return open_labels


def _slot_label_replacements_for_permutation(open_slot_labels, external_legs, perm):
    """Build concrete label replacements for one contraction permutation."""
    replacements = []
    for open_slot in open_slot_labels:
        source_index = open_slot.index_slot
        target_leg = external_legs[perm[open_slot.field_slot_position]]
        target_index = target_leg.index_slot_at(source_index.declared_position)
        if target_index is None:
            raise ValueError(
                "Missing leg index slot at declared position "
                f"{source_index.declared_position} for external leg {target_leg.position}"
            )
        if target_index.index_type != source_index.index_type:
            raise ValueError(
                "Mismatched index types during slot remapping: field slot "
                f"{open_slot.field_slot_position} carries '{source_index.index_type.name}' "
                f"at declared position {source_index.declared_position}, but external leg "
                f"{target_leg.position} carries '{target_index.index_type.name}'."
            )
        if target_index.label is None:
            raise ValueError(
                f"Missing leg slot label for index type '{source_index.index_type.name}' "
                f"at external leg {target_leg.position}"
            )
        replacements.append((source_index.label, target_index.label))
    return replacements


def _apply_label_replacements(expr, replacements):
    """Apply a list of symbolic label replacements to an expression-like object."""
    result = expr
    for source, target in replacements:
        if hasattr(result, "replace"):
            result = result.replace(source, target)
        elif result == source:
            result = target
    return result


def _all_fermion_slots_labeled(field_roles, field_spinor_indices):
    """Whether every fermion slot has an explicit spinor label."""
    if field_roles is None or field_spinor_indices is None:
        return False
    if len(field_roles) != len(field_spinor_indices):
        return False

    for i, role in enumerate(field_roles):
        if _role_name(role) in FERMION_ROLES and _spinor_label(field_spinor_indices[i]) is None:
            return False
    return True


def _fermion_sign_from_slots(perm, fermion_slots):
    """Compute Grassmann sign from permutation restricted to fermion slots."""
    if len(fermion_slots) <= 1:
        return 1

    assigned = [perm[k] for k in fermion_slots]
    inv = 0
    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            if assigned[i] > assigned[j]:
                inv += 1
    return (-1) ** inv


def _default_spin_symbol(leg_position: int):
    return S(f"s{leg_position + 1}")


def _default_leg_spinor_indices(num_legs: int):
    return [S(f"i{k + 1}") for k in range(num_legs)]


def _external_factor_for_contraction(
    *,
    role,
    alpha,
    beta,
    p,
    spin,
    spinor_index,
):
    if role == "psi":
        return delta(alpha, beta) * UF(beta, p, spin, spinor_index)
    if role == "psibar":
        return delta(alpha, beta) * UbarF(beta, p, spin, spinor_index)
    return delta(alpha, beta) * U(beta, p)


def _has_legacy_leg_inputs(
    *,
    betas,
    ps,
    leg_roles,
    leg_spins,
    leg_spinor_indices,
    leg_slot_labels,
    leg_index_slots,
):
    return any(
        value is not None
        for value in (
            betas,
            ps,
            leg_roles,
            leg_spins,
            leg_spinor_indices,
            leg_slot_labels,
            leg_index_slots,
        )
    )


def _normalize_vertex_inputs(
    *,
    interaction: Optional[InteractionTerm],
    external_legs: Optional[Sequence[ExternalLeg]],
    coupling,
    alphas,
    betas,
    ps,
    derivative_indices,
    derivative_targets,
    statistics,
    field_roles,
    leg_roles,
    field_spinor_indices,
    field_slot_labels,
    field_index_slots,
    leg_spins,
    leg_spinor_indices,
    leg_slot_labels,
    leg_index_slots,
):
    if interaction is not None:
        if any(
            value is not None
            for value in (
                coupling,
                alphas,
                field_roles,
                field_spinor_indices,
                field_slot_labels,
                field_index_slots,
            )
        ) or derivative_targets is not None or derivative_indices:
            raise ValueError(
                "When interaction=InteractionTerm(...) is provided, omit the legacy "
                "field-side arguments coupling/alphas/field_roles/field_slot_labels/"
                "field_spinor_indices/field_index_slots/derivative_*."
            )

        normalized = interaction_term_to_legacy(interaction)

        if external_legs is not None:
            if _has_legacy_leg_inputs(
                betas=betas,
                ps=ps,
                leg_roles=leg_roles,
                leg_spins=leg_spins,
                leg_spinor_indices=leg_spinor_indices,
                leg_slot_labels=leg_slot_labels,
                leg_index_slots=leg_index_slots,
            ):
                raise ValueError(
                    "When external_legs=... is provided, omit the legacy leg-side "
                    "arguments betas/ps/leg_roles/leg_slot_labels/leg_index_slots/"
                    "leg_spinor_indices/leg_spins."
                )
            normalized.update(external_legs_to_legacy(external_legs))
            return normalized

        if (
            ps is not None
            and betas is None
            and leg_roles is None
            and leg_spins is None
            and leg_spinor_indices is None
            and leg_slot_labels is None
            and leg_index_slots is None
        ):
            normalized.update(
                external_legs_to_legacy(
                    default_external_legs_for_interaction(interaction, momenta=ps)
                )
            )
            return normalized

        if not _has_legacy_leg_inputs(
            betas=betas,
            ps=ps,
            leg_roles=leg_roles,
            leg_spins=leg_spins,
            leg_spinor_indices=leg_spinor_indices,
            leg_slot_labels=leg_slot_labels,
            leg_index_slots=leg_index_slots,
        ):
            normalized.update(
                external_legs_to_legacy(default_external_legs_for_interaction(interaction))
            )
            return normalized

        if betas is None or ps is None:
            raise ValueError(
                "When combining interaction=... with legacy leg inputs, provide both "
                "betas and ps, or use external_legs=..., or omit leg inputs entirely "
                "to auto-generate them."
            )

        normalized.update(
            dict(
                betas=betas,
                ps=ps,
                leg_roles=leg_roles,
                leg_spins=leg_spins,
                leg_spinor_indices=leg_spinor_indices,
                leg_slot_labels=leg_slot_labels,
                leg_index_slots=leg_index_slots,
            )
        )
        return normalized

    if external_legs is not None:
        if _has_legacy_leg_inputs(
            betas=betas,
            ps=ps,
            leg_roles=leg_roles,
            leg_spins=leg_spins,
                leg_spinor_indices=leg_spinor_indices,
                leg_slot_labels=leg_slot_labels,
                leg_index_slots=leg_index_slots,
            ):
                raise ValueError(
                    "When external_legs=... is provided, omit the legacy leg-side "
                    "arguments betas/ps/leg_roles/leg_slot_labels/leg_index_slots/"
                    "leg_spinor_indices/leg_spins."
                )
        leg_data = external_legs_to_legacy(external_legs)
        betas = leg_data["betas"]
        ps = leg_data["ps"]
        leg_roles = leg_data["leg_roles"]
        leg_spins = leg_data["leg_spins"]
        leg_spinor_indices = leg_data["leg_spinor_indices"]
        leg_slot_labels = leg_data["leg_slot_labels"]
        leg_index_slots = leg_data["leg_index_slots"]

    return dict(
        coupling=coupling,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        field_slot_labels=field_slot_labels,
        field_index_slots=field_index_slots,
        leg_spins=leg_spins,
        leg_spinor_indices=leg_spinor_indices,
        leg_slot_labels=leg_slot_labels,
        leg_index_slots=leg_index_slots,
    )


def _build_contraction_input(
    *,
    coupling,
    statistics,
    alphas,
    betas,
    ps,
    derivative_indices,
    derivative_targets,
    field_roles,
    leg_roles,
    field_slot_labels,
    field_index_slots,
    leg_spins,
    leg_slot_labels,
    leg_index_slots,
) -> _NormalizedContractionInput:
    normalized_field_index_slots = _coerce_index_slot_structure(
        field_index_slots,
        expected_length=len(alphas),
        label_name="field_index_slots",
    )
    normalized_leg_index_slots = _coerce_index_slot_structure(
        leg_index_slots,
        expected_length=len(ps),
        label_name="leg_index_slots",
    )

    field_slots = []
    for position, alpha in enumerate(alphas):
        normalized_slot_labels = field_slot_labels[position] if field_slot_labels is not None else None
        normalized_index_slots = (
            normalized_field_index_slots[position]
            if normalized_field_index_slots is not None
            else ordered_index_slots(normalized_slot_labels)
        )
        field_slots.append(
            _FieldSlot(
                position=position,
                species=alpha,
                role=normalize_role(field_roles[position]) if field_roles is not None else None,
                slot_labels=normalized_slot_labels,
                index_slots=normalized_index_slots,
            )
        )

    external_legs = []
    for position, (beta, momentum) in enumerate(zip(betas, ps)):
        normalized_slot_labels = leg_slot_labels[position] if leg_slot_labels is not None else None
        normalized_index_slots = (
            normalized_leg_index_slots[position]
            if normalized_leg_index_slots is not None
            else ordered_index_slots(normalized_slot_labels)
        )
        external_legs.append(
            _ExternalLegSlot(
                position=position,
                species=beta,
                momentum=momentum,
                role=normalize_role(leg_roles[position]) if leg_roles is not None else None,
                slot_labels=normalized_slot_labels,
                spin=leg_spins[position] if leg_spins is not None else _default_spin_symbol(position),
                index_slots=normalized_index_slots,
            )
        )

    derivatives = tuple(
        _DerivativeInstruction(index=index, target_slot=target)
        for index, target in zip(derivative_indices, derivative_targets)
    )

    return _NormalizedContractionInput(
        coupling=coupling,
        statistics=statistics,
        field_slots=tuple(field_slots),
        external_legs=tuple(external_legs),
        derivatives=derivatives,
    )




# ---------------------------------------------------------------------------
# Wick contractions (permutation sum)
# ---------------------------------------------------------------------------


def _infer_fermion_chains_from_endpoints(field_roles, field_spinor_indices):
    """Infer simple fermion chains with ordered endpoints.

    For now we only support bispinor-metric spinor structure:
      each chain corresponds to one bilinear with exactly two fermion slots:
        (psibar endpoint, psi endpoint)

    The endpoints are inferred by matching equal non-None spinor index symbols
    and requiring one slot to be "psibar" and the other to be "psi".
    """
    if field_roles is None:
        raise ValueError("field_roles must be provided to infer fermion chains")
    if field_spinor_indices is None:
        raise ValueError("field_spinor_indices must be provided to infer fermion chains")
    if len(field_roles) != len(field_spinor_indices):
        raise ValueError("field_roles and field_spinor_indices must have the same length")

    groups = _group_spinor_slots(field_spinor_indices)

    chains = []
    for spinor_label, slots in groups.items():
        if len(slots) == 1:
            # A singleton spinor label can represent an open slot that is
            # carried by an explicit tensor in the coupling, e.g. gamma(mu,i,j).
            continue
        if len(slots) != 2:
            raise ValueError(
                "Each non-None spinor index must appear exactly twice in "
                "a bilinear chain or once as an open slot, got "
                f"{len(slots)} occurrences for index '{spinor_label}'."
            )
        a, b = slots
        role_a = _role_name(field_roles[a])
        role_b = _role_name(field_roles[b])
        if role_a == "psibar" and role_b == "psi":
            chains.append((a, b))
        elif role_a == "psi" and role_b == "psibar":
            chains.append((b, a))
        else:
            raise ValueError(
                "Invalid fermion chain endpoints: expected one 'psibar' and one 'psi' "
                f"for the same spinor index, got roles ({role_a}, {role_b})"
            )
    return chains


def _has_inferred_fermion_chains(field_roles, field_spinor_indices):
    """Whether repeated fermion spinor labels encode bilinear contractions."""
    if field_roles is None or field_spinor_indices is None:
        return False
    return len(_infer_fermion_chains_from_endpoints(field_roles, field_spinor_indices)) > 0


def _validate_supported_fermion_structure(field_roles, field_spinor_indices, coupling=None):
    """Reject underspecified multi-fermion operators.

    The current engine can infer scalar fermion bilinears from repeated dummy
    spinor labels in ``field_spinor_indices``. For interactions with more than
    one fermion bilinear, missing such contraction data leaves the operator
    Lorentz-underspecified, so we reject it instead of silently producing a
    misleading scalar cancellation.
    """
    if field_roles is None:
        return

    fermion_slots = _fermion_slots_from_roles(field_roles)
    if len(fermion_slots) <= 2:
        return

    if field_spinor_indices is None or all(
        _spinor_label(field_spinor_indices[i]) is None
        for i in fermion_slots
    ):
        raise ValueError(
            "Multi-fermion operators require explicit spinor-contraction data. "
            "Provide repeated dummy labels in field_spinor_indices, e.g. "
            "[alpha, alpha, beta, beta] for (psibar psi)(psibar psi)."
        )

    if _has_inferred_fermion_chains(field_roles, field_spinor_indices):
        return

    # Allow fully open-slot encodings where every fermion slot carries a unique
    # spinor label and the coupling tensor is expected to provide the explicit
    # chain structure (e.g. gamma(i1,i2)*gamma(i3,i4)).
    fermion_slots = _fermion_slots_from_roles(field_roles)
    labels = [str(_spinor_label(field_spinor_indices[i])) for i in fermion_slots]
    if all(_spinor_label(field_spinor_indices[i]) is not None for i in fermion_slots) and len(set(labels)) == len(labels):
        if coupling is None:
            raise ValueError(
                "Explicit open-slot multi-fermion encoding requires coupling to "
                "contain those spinor labels, but coupling was not provided."
            )

        coupling_text = coupling.to_canonical_string()
        missing = [
            str(_spinor_label(field_spinor_indices[i]))
            for i in fermion_slots
            if str(_spinor_label(field_spinor_indices[i])) not in coupling_text
        ]
        if missing:
            raise ValueError(
                "Unsupported multi-fermion operator: explicit open-slot labels "
                "must appear in the coupling tensor. Missing labels: "
                + ", ".join(missing)
            )
        return

    raise ValueError(
        "Unsupported multi-fermion operator: provide either inferable bilinear "
        "contractions via repeated dummy labels (e.g. [alpha,alpha,beta,beta]) "
        "or a fully explicit open-slot encoding where each fermion slot has a "
        "distinct spinor label tied to the coupling tensor."
    )


def contract_to_full_expression(
    *,
    interaction: Optional[InteractionTerm] = None,
    external_legs: Optional[Sequence[ExternalLeg]] = None,
    alphas: Optional[Sequence] = None,
    betas: Optional[Sequence] = None,
    ps: Optional[Sequence] = None,
    x,
    derivative_indices=(),
    derivative_targets: Optional[Sequence[int]] = None,
    statistics: Statistics = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    field_slot_labels=None,
    field_index_slots=None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    leg_slot_labels=None,
    leg_index_slots=None,
    coupling=None,
):
    """Sum over all Wick contractions to build the full vacuum matrix element.

    Inputs can be supplied in two equivalent ways:
      1. the legacy parallel-list interface (alphas/betas/ps/roles/labels)
      2. the metadata interface via ``interaction=InteractionTerm(...)`` and
         optional ``external_legs=[...]``. If external legs are omitted, they
         are auto-generated in field order with momenta ``p1, p2, ...``.

    For bosons the sum is a permanent (all signs +1).
    For fermions the sign of each permutation contributes (-1)^parity.

    If derivative information is provided, derivative momentum factors are
    evaluated per permutation, i.e. the derivative acting on field k picks the
    momentum of the external leg that field k contracts with in that
    permutation.

    When spinor leg labels are available, fermion external factors are replaced
    by Spenso bispinor metrics g(bis(4,a), bis(4,b)) connecting legs whose
    fields share a bilinear spinor contraction (inferred from
    field_spinor_indices). Coupling-level open labels are remapped through the
    generic slot-label machinery, so spinor, Lorentz, and gauge labels can all
    follow the contraction permutation. Internally, the normalized
    ``ConcreteIndexSlot`` sequences are the authoritative source of index data;
    grouped slot-label mappings are kept only as compatibility shims. Bosonic
    fields still produce U(beta, p) as usual.
    """

    normalized = _normalize_vertex_inputs(
        interaction=interaction,
        external_legs=external_legs,
        coupling=coupling,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        field_slot_labels=field_slot_labels,
        field_index_slots=field_index_slots,
        leg_spins=leg_spins,
        leg_spinor_indices=leg_spinor_indices,
        leg_slot_labels=leg_slot_labels,
        leg_index_slots=leg_index_slots,
    )
    coupling = normalized["coupling"]
    alphas = normalized["alphas"]
    betas = normalized["betas"]
    ps = normalized["ps"]
    derivative_indices = normalized["derivative_indices"]
    derivative_targets = normalized["derivative_targets"]
    statistics = normalized["statistics"]
    field_roles = normalized["field_roles"]
    leg_roles = normalized["leg_roles"]
    field_spinor_indices = normalized["field_spinor_indices"]
    field_slot_labels = normalized["field_slot_labels"]
    field_index_slots = normalized["field_index_slots"]
    leg_spins = normalized["leg_spins"]
    leg_spinor_indices = normalized["leg_spinor_indices"]
    leg_slot_labels = normalized["leg_slot_labels"]
    leg_index_slots = normalized["leg_index_slots"]

    if alphas is None or betas is None or ps is None:
        raise ValueError(
            "contract_to_full_expression requires either legacy alphas/betas/ps "
            "inputs or interaction=InteractionTerm(...)."
        )

    #first check if the lengths of the sequences are the same
    n = len(alphas)
    if not (len(betas) == len(ps) == n):
        raise ValueError("Nope! alphas, betas, ps must have the same length dear...")

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)

    if len(derivative_targets) != len(derivative_indices):
        raise ValueError("Nope! derivative_targets length must match derivative_indices... this is not gonna work...")

    for tgt in derivative_targets:
        if tgt < 0 or tgt >= n:
            raise ValueError(f"Nope! derivative target index {tgt} out of range for {n} fields... this is not gonna work...")

    if field_roles is not None and len(field_roles) != n:
        raise ValueError("field_roles must have the same length as alphas")
    if leg_roles is not None and len(leg_roles) != n:
        raise ValueError("leg_roles must have the same length as betas")
    if (field_roles is None) != (leg_roles is None):
        raise ValueError("Provide both field_roles and leg_roles, or neither")
    if field_spinor_indices is not None and len(field_spinor_indices) != n:
        raise ValueError("field_spinor_indices must have the same length as alphas")
    if leg_spins is not None and len(leg_spins) != n:
        raise ValueError("leg_spins must have the same length as ps")
    if leg_spinor_indices is not None and len(leg_spinor_indices) != n:
        raise ValueError("leg_spinor_indices must have the same length as ps")

    field_slot_state = _prepare_slot_side(
        base_slot_labels=field_slot_labels,
        explicit_index_slots=field_index_slots,
        spinor_indices=field_spinor_indices,
        roles=field_roles,
        expected_length=n,
        label_name="field_slot_labels",
    )
    leg_slot_state = _prepare_slot_side(
        base_slot_labels=leg_slot_labels,
        explicit_index_slots=leg_index_slots,
        spinor_indices=leg_spinor_indices,
        roles=leg_roles,
        expected_length=n,
        label_name="leg_slot_labels",
    )
    effective_field_slot_labels = field_slot_state.slot_labels
    effective_leg_slot_labels = leg_slot_state.slot_labels
    effective_field_index_slots = field_slot_state.index_slots
    effective_leg_index_slots = leg_slot_state.index_slots
    effective_field_spinor_indices = field_slot_state.spinor_labels
    effective_leg_spinor_indices = leg_slot_state.spinor_labels

    if statistics == "fermion":
        if field_roles is None or leg_roles is None:
            raise ValueError(
                "statistics='fermion' requires both field_roles and leg_roles "
                "to avoid ambiguous Grassmann signs and incompatible contractions"
            )

    use_spinor_deltas = _index_slot_structure_has_type(effective_leg_index_slots, SPINOR_KIND)
    contraction_input = _build_contraction_input(
        coupling=coupling if coupling is not None else Expression.num(1),
        statistics=statistics,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_slot_labels=effective_field_slot_labels,
        field_index_slots=effective_field_index_slots,
        leg_spins=leg_spins,
        leg_slot_labels=effective_leg_slot_labels,
        leg_index_slots=effective_leg_index_slots,
    )

    field_slots = contraction_input.field_slots
    external_legs = contraction_input.external_legs
    derivatives = contraction_input.derivatives

    if statistics == "fermion":
        _validate_supported_fermion_structure(
            field_slots,
            field_slots,
            coupling=coupling,
        )

    fermion_chains = []
    if use_spinor_deltas:
        if field_roles is None:
            raise ValueError("field_roles required when spinor leg labels are given")
        if effective_field_spinor_indices is None:
            raise ValueError("field_spinor_indices or field_slot_labels required when spinor leg labels are given")
        for field_slot in field_slots:
            if field_slot.role_name in ("psi", "psibar") and field_slot.spinor_label is None:
                raise ValueError(
                    "Fermion slots must carry a spinor index when "
                    "spinor leg labels are provided"
                )
        for leg in external_legs:
            if leg.role_name in ("psi", "psibar") and leg.spinor_label is None:
                raise ValueError(
                    "Fermion legs must carry a spinor index when "
                    "spinor leg labels are provided"
                )
        fermion_chains = _infer_fermion_chains_from_endpoints(field_slots, field_slots)

    fermion_slots = (
        _fermion_slots_from_roles(field_slots)
        if statistics == "fermion" else []
    )

    total = Expression.num(0)
    open_slot_labels = []
    if (
        any(index_slot.is_labeled for field_slot in field_slots for index_slot in field_slot.index_slots)
        and any(index_slot.is_labeled for leg in external_legs for index_slot in leg.index_slots)
    ):
        open_slot_labels = _open_index_slots(field_slots)

    for perm in permutations(range(n)):
        term = Expression.num(1)

        # Skip incompatible role assignments early (species stays symbolic via deltas).
        valid = True
        for i, j in enumerate(perm):
            if not factor_leg_compatible(field_slots[i], external_legs[j]):
                valid = False
                break
        if not valid:
            continue

        label_replacements = []
        if open_slot_labels:
            label_replacements = _slot_label_replacements_for_permutation(
                open_slot_labels,
                external_legs,
                perm,
            )

        coupling_term = _apply_label_replacements(
            contraction_input.coupling,
            label_replacements,
        )
        term *= coupling_term

        if fermion_slots:
            term *= Expression.num(_fermion_sign_from_slots(perm, fermion_slots))

        #first we evaluate the derivative momentum factors with the momentum assigned by this permutation
        for derivative in derivatives:
            mapped_mu = _apply_label_replacements(derivative.index, label_replacements)
            term *= (-I) * pcomp(external_legs[perm[derivative.target_slot]].momentum, mapped_mu)

        #now we evaluate the delta and U factors
        p_sum = Expression.num(0)
        for i, j in enumerate(perm):
            field_slot = field_slots[i]
            leg = external_legs[j]
            role = field_slot.role_name

            if use_spinor_deltas and role in ("psi", "psibar"):
                term *= delta(field_slot.species, leg.species)
            else:
                spinor_index = (
                    field_slot.spinor_label
                    if field_slot.spinor_label is not None
                    else S(f"si{i + 1}")
                )
                term *= _external_factor_for_contraction(
                    role=role,
                    alpha=field_slot.species,
                    beta=leg.species,
                    p=leg.momentum,
                    spin=leg.spin,
                    spinor_index=spinor_index,
                )
            p_sum += leg.momentum

        if use_spinor_deltas:
            for psibar_slot, psi_slot in fermion_chains:
                term *= bis.g(
                    external_legs[perm[psibar_slot]].spinor_label,
                    external_legs[perm[psi_slot]].spinor_label,
                ).to_expression()

        term *= plane_wave(p_sum, x)
        total += term

    return total


# ---------------------------------------------------------------------------
# Derivative helpers
# ---------------------------------------------------------------------------


def infer_derivative_targets(field_derivative_map):
    """Build (derivative_indices, derivative_targets) from a per-field spec.

    Parameters
    ----------
    field_derivative_map : list of (field_index, [mu1, mu2, ...]) pairs.
        Example: [(0, [mu]), (2, [mu, nu])] means field 0 has d_mu,
        field 2 has d_mu d_nu.

    Returns
    -------
    (derivative_indices, derivative_targets) : tuple of two lists
    """
    indices = []
    targets = []
    for field_idx, lorentz_indices in field_derivative_map:
        for mu in lorentz_indices:
            indices.append(mu)
            targets.append(field_idx)
    return indices, targets


# ---------------------------------------------------------------------------
# Full vertex factor pipeline
# ---------------------------------------------------------------------------


def vertex_factor(
    *,
    interaction: Optional[InteractionTerm] = None,
    external_legs: Optional[Sequence[ExternalLeg]] = None,
    coupling=None,
    alphas: Optional[Sequence] = None,
    betas: Optional[Sequence] = None,
    ps: Optional[Sequence] = None,
    x,
    derivative_indices=(),
    derivative_targets=None,
    statistics: Statistics = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    field_slot_labels=None,
    field_index_slots=None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    leg_slot_labels=None,
    leg_index_slots=None,
    strip_externals: bool = True,
    include_delta: bool = True,
    d=None,
):
    """Compute the Feynman vertex factor from an interaction term.

    This combines all algorithm steps:
        1. Contract fields with creation operators
        2. Evaluate derivative momentum factors for each contraction
           (permutation-aware assignment)
        3. Integrate over x: under the current local convention, the common
           plane-wave factor is replaced by 1 when ``include_delta=True``
        4. Strip external wavefunctions U(beta, p) -> 1
        5. Multiply by i

    Parameters
    ----------
    interaction : optional InteractionTerm
        Metadata-layer input describing the interaction monomial. When given,
        omit the legacy field-side arguments and optionally provide
        ``external_legs=[...]`` for a custom external ordering. If the legs are
        omitted, default legs are generated in field order.
    external_legs : optional sequence of ExternalLeg
        Metadata-layer external states paired with ``interaction``. This is the
        future-proof alternative to manual betas/ps/leg_roles/... lists.
    coupling : Symbolica expression for the coupling constant/tensor
    alphas : species labels for the fields in the Lagrangian term
    betas : species labels for the external particles
    ps : momentum symbols for each external leg
    x : spacetime position symbol
    derivative_indices : Lorentz indices for derivatives
    derivative_targets : which field each derivative acts on
    statistics : "boson" or "fermion"
    strip_externals : remove external wavefunctions and leftover plane waves.
        For fermions, repeated labels in ``field_spinor_indices`` are treated
        as bilinear contractions and are preserved as open spinor metrics in
        the amputated vertex.
    include_delta : apply the current x-integration convention. In this local
        branch, the common plane-wave factor is replaced by 1 rather than an
        explicit momentum delta.
    d : spacetime dimension symbol (defaults to S('d'))
    field_slot_labels / leg_slot_labels : generic per-slot index-label
        mappings, e.g. spinor, Lorentz, or gauge labels carried by each
        field/external leg. These are used to remap open tensor slots inside
        the coupling according to the contraction permutation.
    field_index_slots / leg_index_slots : direct concrete-index-slot inputs for
        callers that already work with typed ordered slots. These are preserved
        as first-class inputs rather than being reconstructed from labels.
    leg_spinor_indices : legacy convenience argument for spinor labels on
        external legs. If omitted, stripped vertices auto-generate default
        leg labels from the available field-slot labels.
    """
    normalized = _normalize_vertex_inputs(
        interaction=interaction,
        external_legs=external_legs,
        coupling=coupling,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        field_slot_labels=field_slot_labels,
        field_index_slots=field_index_slots,
        leg_spins=leg_spins,
        leg_spinor_indices=leg_spinor_indices,
        leg_slot_labels=leg_slot_labels,
        leg_index_slots=leg_index_slots,
    )
    coupling = normalized["coupling"]
    alphas = normalized["alphas"]
    betas = normalized["betas"]
    ps = normalized["ps"]
    derivative_indices = normalized["derivative_indices"]
    derivative_targets = normalized["derivative_targets"]
    statistics = normalized["statistics"]
    field_roles = normalized["field_roles"]
    leg_roles = normalized["leg_roles"]
    field_spinor_indices = normalized["field_spinor_indices"]
    field_slot_labels = normalized["field_slot_labels"]
    field_index_slots = normalized["field_index_slots"]
    leg_spins = normalized["leg_spins"]
    leg_spinor_indices = normalized["leg_spinor_indices"]
    leg_slot_labels = normalized["leg_slot_labels"]
    leg_index_slots = normalized["leg_index_slots"]

    if coupling is None or alphas is None or betas is None or ps is None:
        raise ValueError(
            "vertex_factor requires either legacy coupling/alphas/betas/ps "
            "inputs or interaction=InteractionTerm(...)."
        )

    field_slot_state = _prepare_slot_side(
        base_slot_labels=field_slot_labels,
        explicit_index_slots=field_index_slots,
        spinor_indices=field_spinor_indices,
        roles=field_roles,
        expected_length=len(ps),
        label_name="field_slot_labels",
    )
    leg_slot_state = _prepare_slot_side(
        base_slot_labels=leg_slot_labels,
        explicit_index_slots=leg_index_slots,
        spinor_indices=leg_spinor_indices,
        roles=leg_roles,
        expected_length=len(ps),
        label_name="leg_slot_labels",
    )
    effective_field_slot_labels = field_slot_state.slot_labels
    effective_field_index_slots = field_slot_state.index_slots
    effective_field_spinor_indices = field_slot_state.spinor_labels
    effective_leg_slot_labels = leg_slot_state.slot_labels
    effective_leg_index_slots = leg_slot_state.index_slots
    effective_leg_spinor_indices = leg_slot_state.spinor_labels
    if (
        strip_externals
        and effective_leg_slot_labels is None
        and effective_field_slot_labels is not None
    ):
        effective_leg_slot_labels = default_leg_slot_labels(effective_field_slot_labels)
        leg_slot_state = _prepare_slot_side(
            base_slot_labels=effective_leg_slot_labels,
            explicit_index_slots=leg_index_slots,
            spinor_indices=leg_spinor_indices,
            roles=leg_roles,
            expected_length=len(ps),
            label_name="leg_slot_labels",
        )
        effective_leg_slot_labels = leg_slot_state.slot_labels
        effective_leg_index_slots = leg_slot_state.index_slots
        effective_leg_spinor_indices = leg_slot_state.spinor_labels

    if (
        strip_externals
        and effective_leg_spinor_indices is None
        and statistics == "fermion"
        and _all_fermion_slots_labeled(field_roles, effective_field_spinor_indices)
    ):
        generated_leg_spinors = _default_leg_spinor_indices(len(ps))
        generated_leg_slot_labels = _merge_slot_label_structures(
            effective_leg_slot_labels,
            generated_leg_spinors,
            leg_roles,
            expected_length=len(ps),
            label_name="leg_slot_labels",
        )
        leg_slot_state = _prepare_slot_side(
            base_slot_labels=generated_leg_slot_labels,
            explicit_index_slots=leg_index_slots,
            spinor_indices=None,
            roles=leg_roles,
            expected_length=len(ps),
            label_name="leg_slot_labels",
        )
        effective_leg_slot_labels = leg_slot_state.slot_labels
        effective_leg_index_slots = leg_slot_state.index_slots
        effective_leg_spinor_indices = leg_slot_state.spinor_labels

    contracted = contract_to_full_expression(
        alphas=alphas,
        betas=betas,
        ps=ps,
        x=x,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=effective_field_spinor_indices,
        field_slot_labels=effective_field_slot_labels,
        field_index_slots=effective_field_index_slots,
        leg_spins=leg_spins,
        leg_spinor_indices=effective_leg_spinor_indices,
        leg_slot_labels=effective_leg_slot_labels,
        leg_index_slots=effective_leg_index_slots,
        coupling=coupling,
    )
    full = contracted

    if include_delta:
        if d is None:
            d = S("d")
        p_sum = Expression.num(0)
        for p in ps:
            p_sum += p
        full = full.replace(plane_wave(p_sum, x), 1)#(2 * pi) ** d * Delta(p_sum))

    if strip_externals:
        beta_, p_ = S("beta_", "p_")
        full = full.replace(U(beta_, p_), 1)
        if effective_leg_spinor_indices is None:
            spin_, si_ = S("spin_", "si_")
            full = full.replace(UF(beta_, p_, spin_, si_), 1)
            full = full.replace(UbarF(beta_, p_, spin_, si_), 1)
        q_, x_ = S("q_", "x_")
        full = full.replace(Expression.EXP(-I * Dot(q_, x_)), 1)

    return I * full


# ---------------------------------------------------------------------------
# Simplification helpers (species deltas + Spenso spinor metrics)
# ---------------------------------------------------------------------------

def simplify_deltas(expr, species_map=None):
    """Simplify species Kronecker deltas via direct substitution.

    Uses a three-step pipeline:
      1. Substitute each beta symbol with its assigned species value.
      2. Collapse delta(x, x) -> 1 for matching arguments.
      3. Kill delta(s_i, s_j) -> 0 only for pairs of *distinct known species*.

    Step 3 is restricted to species that actually appear in the map values,
    so symbolic deltas like delta(i, j) from coupling tensors survive.

    Parameters
    ----------
    expr : Symbolica expression
    species_map : dict mapping beta_symbol -> species_symbol.
        When provided, every occurrence of beta_symbol in the expression
        is replaced by species_symbol (not just inside deltas). Then
        same-species deltas collapse to 1 and cross-species deltas to 0.
        If None, only delta(a, a) -> 1 is applied.
    """
    a_ = S("a_")

    if species_map is not None:
        for beta_sym, species_sym in species_map.items():
            expr = expr.replace(beta_sym, species_sym)
        expr = expr.replace(delta(a_, a_), Expression.num(1))

        known_species = sorted(
            set(species_map.values()),
            key=lambda s: _species_key(s),
        )
        for i in range(len(known_species)):
            for j in range(i + 1, len(known_species)):
                expr = expr.replace(
                    delta(known_species[i], known_species[j]),
                    Expression.num(0),
                )
    else:
        expr = expr.replace(delta(a_, a_), Expression.num(1))

    return expr


def simplify_spinor_indices(expr):
    """Contract repeated bispinor indices using Spenso's metric simplification.

    Spinor deltas produced by the vertex factor are Spenso bispinor metrics
    ``g(bis(4,i),bis(4,j))``.  When two such metrics share a repeated index
    (e.g. from a gamma-matrix chain), ``simplify_metrics`` contracts them
    automatically: ``g(i,j)*g(j,k)`` -> ``g(i,k)``.
    """
    return simplify_metrics(expr)


def simplify_vertex(expr, species_map=None):
    """Simplify a vertex factor expression in one call.

    Chains species-delta simplification and spinor-index contraction:
      1. ``simplify_deltas`` -- resolve species Kronecker deltas
      2. ``simplify_spinor_indices`` -- contract repeated bispinor metrics
    """
    expr = simplify_deltas(expr, species_map=species_map)
    expr = simplify_spinor_indices(expr)
    return expr


def _species_key(x):
    return x.to_canonical_string() if hasattr(x, 'to_canonical_string') else str(x)


def derivative_momentum_sum_expression(
    *,
    ps: Sequence,
    derivative_indices,
    derivative_targets=None,
    field_species: Optional[Sequence] = None,
    leg_species: Optional[Sequence] = None,
):
    """Build a compact momentum-sum expression for general derivative patterns.

    This computes the same momentum part as permutation summation, but groups
    permutations by the derivative-assignment pattern:
      - choose leg assignments for distinct derivative target fields
      - multiply by combinatorial multiplicity of remaining species-compatible
        contractions

    Parameters
    ----------
    ps : sequence of momentum symbols for external legs
    derivative_indices : sequence of Lorentz indices (mu, nu, ...)
    derivative_targets : which field slot each derivative acts on
    field_species : species label for each field slot in the interaction term
    leg_species : species label for each external leg (same length as ps)
    """
    n = len(ps)
    m = len(derivative_indices)

    if derivative_targets is None:
        derivative_targets = [0] * m
    if len(derivative_targets) != m:
        raise ValueError("derivative_targets length must match derivative_indices")

    if field_species is not None and len(field_species) != n:
        raise ValueError("field_species must have same length as ps")
    if leg_species is not None and len(leg_species) != n:
        raise ValueError("leg_species must have same length as ps")

    # Distinct field slots that carry at least one derivative, preserving order.
    unique_targets = []
    for t in derivative_targets:
        if t < 0 or t >= n:
            raise ValueError(f"derivative target index {t} out of range for {n} fields")
        if t not in unique_targets:
            unique_targets.append(t)

    k = len(unique_targets)
    target_slot = {t: s for s, t in enumerate(unique_targets)}
    total = Expression.num(0)

    for assigned_legs in permutations(range(n), k):
        # Species compatibility for targeted fields.
        if field_species is not None and leg_species is not None:
            ok = True
            for t in unique_targets:
                slot = target_slot[t]
                leg = assigned_legs[slot]
                if _species_key(field_species[t]) != _species_key(leg_species[leg]):
                    ok = False
                    break
            if not ok:
                continue

        # Product of momentum factors for this derivative assignment.
        monomial = Expression.num(1)
        for mu, t in zip(derivative_indices, derivative_targets):
            leg = assigned_legs[target_slot[t]]
            monomial *= pcomp(ps[leg], mu)

        # Count remaining species-compatible bijections.
        if field_species is not None and leg_species is not None:
            rem_field = [i for i in range(n) if i not in set(unique_targets)]
            rem_legs = [j for j in range(n) if j not in set(assigned_legs)]

            cf = Counter(_species_key(field_species[i]) for i in rem_field)
            cl = Counter(_species_key(leg_species[j]) for j in rem_legs)
            if cf != cl:
                continue

            multiplicity = 1
            for cnt in cf.values():
                multiplicity *= factorial(cnt)
        else:
            multiplicity = factorial(n - k)

        total += multiplicity * monomial

    return total


def compact_vertex_sum_form(
    *,
    coupling,
    ps: Sequence,
    derivative_indices,
    derivative_targets=None,
    d=None,
    field_species: Optional[Sequence] = None,
    leg_species: Optional[Sequence] = None,
    include_delta: bool = True,
):
    """Compact sum-form vertex for general derivative-target patterns.

    This returns the compact expression:
      i * coupling * (-i)^m * delta_factor * S_momentum
    where S_momentum is built by derivative_momentum_sum_expression(...).
    """
    if d is None:
        d = S("d")

    p_sum = Expression.num(0)
    for p in ps:
        p_sum += p

    m = len(derivative_indices)
    phase = I * ((-I) ** m)
    mom_sum = derivative_momentum_sum_expression(
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        field_species=field_species,
        leg_species=leg_species,
    )

    delta_factor = (2 * pi) ** d * Delta(p_sum) if include_delta else Expression.num(1)
    return phase * coupling * delta_factor * mom_sum


def compact_sum_notation(
    *,
    derivative_indices,
    derivative_targets=None,
    n_legs=None,
):
    """Human-readable sigma notation for generic derivative assignment pattern.

    Example output:
      "(n-k)! * Σ_{a,b distinct} p_{a,mu} p_{b,nu}"
    """
    if n_legs is None:
        if derivative_targets:
            n_legs = max(derivative_targets) + 1
        else:
            raise ValueError("n_legs required when derivative_targets is empty")

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)

    unique_targets = []
    for t in derivative_targets:
        if t not in unique_targets:
            unique_targets.append(t)

    symbols = "abcdefghijklmnopqrstuvwxyz"
    if len(unique_targets) > len(symbols):
        raise ValueError("Too many unique derivative targets for notation helper")

    t_to_var = {t: symbols[i] for i, t in enumerate(unique_targets)}
    vars_used = [t_to_var[t] for t in unique_targets]

    terms = []
    for mu, t in zip(derivative_indices, derivative_targets):
        v = t_to_var[t]
        terms.append(f"p_{{{v},{mu}}}")

    product = " ".join(terms) if terms else "1"
    if len(vars_used) <= 1:
        cond = vars_used[0] if vars_used else ""
    else:
        cond = ", ".join(vars_used) + " distinct"

    pref = f"({n_legs - len(unique_targets)})!"
    if cond:
        return f"{pref} * Σ_{{{cond}}} {product}"
    return f"{pref}"
