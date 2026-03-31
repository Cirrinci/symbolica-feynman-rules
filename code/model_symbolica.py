"""
Metadata-first Feynman vertex engine built on Symbolica and Spenso.

This module owns the contraction logic. User-facing inputs should come from the
metadata layer in ``model_schema.py``: ``Field``, ``FieldOccurrence``,
``ExternalLeg``, and ``InteractionTerm``. The legacy parallel-list interface is
kept separately in ``model_legacy.py`` as a compatibility adapter.

Current scope:
    - bosonic polynomial interaction terms
    - permutation-summed Wick contractions
    - derivative interactions via momentum factors
    - fermionic permutation signs and role-aware contractions
    - open index remapping through typed Spenso-backed slots
"""

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Literal, Optional, Sequence

from symbolica import S, Expression, Replacement
from symbolica.community.spenso import Representation
from symbolica.community.idenso import simplify_metrics
from model_schema import (
    ConcreteIndexSlot,
    ExternalLeg,
    FieldRole,
    InteractionTerm,
    default_external_legs_for_interaction,
    default_leg_slot_labels,
    index_bindings_from_slots,
    normalize_concrete_index_slots,
    ordered_index_slots,
    primary_index_label_from_slots,
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
        groups.setdefault(spinor_label, []).append(i)
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
            counts[(index_slot.index_type, index_slot.label)] += 1

    open_labels = []
    for field_slot in field_slots:
        for index_slot in _index_slots(field_slot):
            if index_slot.label is None:
                continue
            if counts[(index_slot.index_type, index_slot.label)] == 1:
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
    if not replacements:
        return result

    if hasattr(result, "replace_multiple"):
        try:
            return result.replace_multiple(
                [Replacement(source, target) for source, target in replacements]
            )
        except TypeError:
            pass

    for source, target in replacements:
        if hasattr(result, "replace"):
            result = result.replace(source, target)
        elif result == source:
            result = target
    return result


def _expression_symbols(expr) -> set:
    if hasattr(expr, "get_all_symbols"):
        return set(expr.get_all_symbols())
    return set()


def _all_fermion_slots_labeled(field_roles):
    """Whether every fermion slot has an explicit spinor label."""
    if field_roles is None:
        return False
    for field_slot in field_roles:
        if _role_name(field_slot) in FERMION_ROLES and _spinor_label(field_slot) is None:
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


def _derivative_instructions_from_metadata(interaction: InteractionTerm) -> tuple[_DerivativeInstruction, ...]:
    derivatives = []
    for action in interaction.derivatives:
        for index in action.indices:
            derivatives.append(_DerivativeInstruction(index=index, target_slot=action.target))
    return tuple(derivatives)


def _field_slots_from_metadata(interaction: InteractionTerm) -> tuple[_FieldSlot, ...]:
    return tuple(
        _FieldSlot(
            position=position,
            species=occ.species,
            role=occ.role,
            slot_labels=occ.slot_labels,
            index_slots=occ.index_slots,
        )
        for position, occ in enumerate(interaction.fields)
    )


def _external_leg_slots_from_metadata(external_legs: Sequence[ExternalLeg]) -> tuple[_ExternalLegSlot, ...]:
    return tuple(
        _ExternalLegSlot(
            position=position,
            species=leg.species,
            momentum=leg.momentum,
            role=leg.role,
            slot_labels=leg.slot_labels,
            spin=leg.spin if leg.spin is not None else _default_spin_symbol(position),
            index_slots=leg.index_slots,
        )
        for position, leg in enumerate(external_legs)
    )


def _metadata_contraction_input(
    interaction: InteractionTerm,
    external_legs: Optional[Sequence[ExternalLeg]] = None,
) -> _NormalizedContractionInput:
    if external_legs is None:
        external_legs = default_external_legs_for_interaction(interaction)
    if len(interaction.fields) != len(external_legs):
        raise ValueError(
            "Interaction fields and external legs must have the same length "
            f"(got {len(interaction.fields)} fields and {len(external_legs)} legs)."
        )
    return _NormalizedContractionInput(
        coupling=interaction.coupling,
        statistics=interaction.statistics,
        field_slots=_field_slots_from_metadata(interaction),
        external_legs=_external_leg_slots_from_metadata(external_legs),
        derivatives=_derivative_instructions_from_metadata(interaction),
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
    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)
    if len(derivative_targets) != len(derivative_indices):
        raise ValueError("derivative_targets length must match derivative_indices")

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


def _validate_contraction_input(contraction_input: _NormalizedContractionInput):
    n = contraction_input.num_slots
    if len(contraction_input.external_legs) != n:
        raise ValueError(
            "Interaction fields and external legs must have the same length "
            f"(got {n} fields and {len(contraction_input.external_legs)} legs)."
        )

    for derivative in contraction_input.derivatives:
        if derivative.target_slot < 0 or derivative.target_slot >= n:
            raise ValueError(
                f"Derivative target index {derivative.target_slot} out of range for {n} fields."
            )

    field_roles = [field_slot.role for field_slot in contraction_input.field_slots]
    leg_roles = [leg.role for leg in contraction_input.external_legs]
    if (all(role is None for role in field_roles)) != (all(role is None for role in leg_roles)):
        raise ValueError("Provide both field roles and leg roles, or neither.")

    if contraction_input.statistics == "fermion":
        if any(role is None for role in field_roles) or any(role is None for role in leg_roles):
            raise ValueError(
                "statistics='fermion' requires both field roles and leg roles "
                "to avoid ambiguous Grassmann signs and incompatible contractions"
            )


def _clone_external_leg_slot(
    leg: _ExternalLegSlot,
    *,
    slot_labels=None,
    index_slots=None,
) -> _ExternalLegSlot:
    if index_slots is None:
        expected_indices = tuple(index_slot.index_type for index_slot in leg.index_slots)
        index_slots = ordered_index_slots(slot_labels, expected_indices=expected_indices)
    return _ExternalLegSlot(
        position=leg.position,
        species=leg.species,
        momentum=leg.momentum,
        role=leg.role,
        slot_labels=slot_labels,
        spin=leg.spin,
        index_slots=index_slots,
    )


def _clone_contraction_input(
    contraction_input: _NormalizedContractionInput,
    *,
    external_legs=None,
) -> _NormalizedContractionInput:
    return _NormalizedContractionInput(
        coupling=contraction_input.coupling,
        statistics=contraction_input.statistics,
        field_slots=contraction_input.field_slots,
        external_legs=tuple(external_legs) if external_legs is not None else contraction_input.external_legs,
        derivatives=contraction_input.derivatives,
    )




# ---------------------------------------------------------------------------
# Wick contractions (permutation sum)
# ---------------------------------------------------------------------------


def _infer_fermion_chains_from_endpoints(field_slots):
    """Infer simple fermion chains with ordered endpoints.

    For now we only support bispinor-metric spinor structure:
      each chain corresponds to one bilinear with exactly two fermion slots:
        (psibar endpoint, psi endpoint)

    The endpoints are inferred by matching equal non-None spinor index symbols
    and requiring one slot to be "psibar" and the other to be "psi".
    """
    if field_slots is None:
        raise ValueError("field slots must be provided to infer fermion chains")

    groups = _group_spinor_slots(field_slots)

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
        role_a = _role_name(field_slots[a])
        role_b = _role_name(field_slots[b])
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


def _has_inferred_fermion_chains(field_slots):
    """Whether repeated fermion spinor labels encode bilinear contractions."""
    if field_slots is None:
        return False
    return len(_infer_fermion_chains_from_endpoints(field_slots)) > 0


def _validate_supported_fermion_structure(field_slots, coupling=None):
    """Reject underspecified multi-fermion operators.

    The current engine can infer scalar fermion bilinears from repeated dummy
    spinor labels in ``field_spinor_indices``. For interactions with more than
    one fermion bilinear, missing such contraction data leaves the operator
    Lorentz-underspecified, so we reject it instead of silently producing a
    misleading scalar cancellation.
    """
    if field_slots is None:
        return

    fermion_slots = _fermion_slots_from_roles(field_slots)
    if len(fermion_slots) <= 2:
        return

    if all(
        _spinor_label(field_slots[i]) is None
        for i in fermion_slots
    ):
        raise ValueError(
            "Multi-fermion operators require explicit spinor-contraction data. "
            "Provide repeated dummy labels in field_spinor_indices, e.g. "
            "[alpha, alpha, beta, beta] for (psibar psi)(psibar psi)."
        )

    if _has_inferred_fermion_chains(field_slots):
        return

    # Allow fully open-slot encodings where every fermion slot carries a unique
    # spinor label and the coupling tensor is expected to provide the explicit
    # chain structure (e.g. gamma(i1,i2)*gamma(i3,i4)).
    fermion_slots = _fermion_slots_from_roles(field_slots)
    labels = [_spinor_label(field_slots[i]) for i in fermion_slots]
    if all(label is not None for label in labels) and len(set(labels)) == len(labels):
        if coupling is None:
            raise ValueError(
                "Explicit open-slot multi-fermion encoding requires coupling to "
                "contain those spinor labels, but coupling was not provided."
            )

        coupling_symbols = _expression_symbols(coupling)
        missing = [
            label
            for label in labels
            if label not in coupling_symbols
        ]
        if missing:
            raise ValueError(
                "Unsupported multi-fermion operator: explicit open-slot labels "
                "must appear in the coupling tensor. Missing labels: "
                + ", ".join(str(label) for label in missing)
            )
        return

    raise ValueError(
        "Unsupported multi-fermion operator: provide either inferable bilinear "
        "contractions via repeated dummy labels (e.g. [alpha,alpha,beta,beta]) "
        "or a fully explicit open-slot encoding where each fermion slot has a "
        "distinct spinor label tied to the coupling tensor."
    )


def _contract_to_full_expression_core(
    *,
    contraction_input: _NormalizedContractionInput,
    x,
):
    """Internal contraction engine on normalized field/leg objects."""

    _validate_contraction_input(contraction_input)

    field_slots = contraction_input.field_slots
    external_legs = contraction_input.external_legs
    derivatives = contraction_input.derivatives
    statistics = contraction_input.statistics
    n = contraction_input.num_slots

    use_spinor_deltas = any(leg.spinor_label is not None for leg in external_legs)

    if statistics == "fermion":
        _validate_supported_fermion_structure(
            field_slots,
            coupling=contraction_input.coupling,
        )

    fermion_chains = []
    if use_spinor_deltas:
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
        fermion_chains = _infer_fermion_chains_from_endpoints(field_slots)

    fermion_slots = _fermion_slots_from_roles(field_slots) if statistics == "fermion" else []

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


def contract_to_full_expression(
    *,
    interaction: InteractionTerm,
    external_legs: Optional[Sequence[ExternalLeg]] = None,
    x,
):
    """Sum over Wick contractions for a metadata-defined interaction term.

    ``interaction`` is the required source of truth. If ``external_legs`` are
    omitted, they are generated in field order with momenta ``p1, p2, ...``.
    """
    contraction_input = _metadata_contraction_input(interaction, external_legs)
    return _contract_to_full_expression_core(
        contraction_input=contraction_input,
        x=x,
    )


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


def _vertex_factor_core(
    *,
    contraction_input: _NormalizedContractionInput,
    x,
    strip_externals: bool = True,
    include_delta: bool = False,
    d=None,
):
    """Internal vertex pipeline on normalized field/leg objects."""

    effective_input = contraction_input
    external_legs = effective_input.external_legs
    field_slot_labels = tuple(field_slot.slot_labels for field_slot in effective_input.field_slots)
    leg_slot_labels = tuple(leg.slot_labels for leg in external_legs)

    if (
        strip_externals
        and all(labels is None for labels in leg_slot_labels)
        and any(labels is not None for labels in field_slot_labels)
    ):
        generated_leg_slot_labels = default_leg_slot_labels(field_slot_labels)
        external_legs = tuple(
            _clone_external_leg_slot(leg, slot_labels=generated_leg_slot_labels[position])
            for position, leg in enumerate(external_legs)
        )
        effective_input = _clone_contraction_input(effective_input, external_legs=external_legs)

    if (
        strip_externals
        and all(leg.spinor_label is None for leg in external_legs)
        and effective_input.statistics == "fermion"
        and _all_fermion_slots_labeled(effective_input.field_slots)
    ):
        generated_leg_spinors = _default_leg_spinor_indices(len(external_legs))
        generated_external_legs = []
        for position, leg in enumerate(external_legs):
            merged_slot_labels = leg.slot_labels
            if leg.role is not None and leg.role.is_fermion:
                merged_slot_labels = merge_index_bindings(
                    leg.slot_labels,
                    ((SPINOR_KIND, (generated_leg_spinors[position],)),),
                )
            generated_external_legs.append(
                _clone_external_leg_slot(leg, slot_labels=merged_slot_labels)
            )
        external_legs = tuple(generated_external_legs)
        effective_input = _clone_contraction_input(effective_input, external_legs=external_legs)

    contracted = _contract_to_full_expression_core(
        contraction_input=effective_input,
        x=x,
    )
    full = contracted

    if include_delta:
        if d is None:
            d = S("d")
        p_sum = Expression.num(0)
        for leg in effective_input.external_legs:
            p_sum += leg.momentum
        full = full.replace(plane_wave(p_sum, x), (2 * pi) ** d * Delta(p_sum))
    else:
        q_, x_ = S("q_", "x_")
        full = full.replace(Expression.EXP(-I * Dot(q_, x_)), 1)

    if strip_externals:
        beta_, p_ = S("beta_", "p_")
        full = full.replace(U(beta_, p_), 1)
        if all(leg.spinor_label is None for leg in effective_input.external_legs):
            spin_, si_ = S("spin_", "si_")
            full = full.replace(UF(beta_, p_, spin_, si_), 1)
            full = full.replace(UbarF(beta_, p_, spin_, si_), 1)

    return I * full


def vertex_factor(
    *,
    interaction: InteractionTerm,
    external_legs: Optional[Sequence[ExternalLeg]] = None,
    x,
    strip_externals: bool = True,
    include_delta: bool = False,
    d=None,
):
    """Compute the momentum-space vertex from metadata-layer interaction objects.

    By default this returns the reduced vertex with the universal overall
    momentum-conservation factor stripped. Set ``include_delta=True`` to keep
    ``(2*pi)^d Delta(sum p)``.
    """
    contraction_input = _metadata_contraction_input(interaction, external_legs)
    return _vertex_factor_core(
        contraction_input=contraction_input,
        x=x,
        strip_externals=strip_externals,
        include_delta=include_delta,
        d=d,
    )


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

        known_species = list(dict.fromkeys(species_map.values()))
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
                if field_species[t] != leg_species[leg]:
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

            cf = Counter(field_species[i] for i in rem_field)
            cl = Counter(leg_species[j] for j in rem_legs)
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
    include_delta: bool = False,
):
    """Compact sum-form vertex for general derivative-target patterns.

    This returns the compact expression:
      i * coupling * (-i)^m * delta_factor * S_momentum
    where S_momentum is built by derivative_momentum_sum_expression(...).

    By default ``delta_factor = 1`` so the result matches the reduced-vertex
    convention used by ``vertex_factor(...)``.
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
