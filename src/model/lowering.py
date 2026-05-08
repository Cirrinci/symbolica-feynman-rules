"""Internal lowering engine from declarations to interaction terms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence, Union

from symbolica import Expression, S

from lagrangian.lowering import (
    expr_equal as _expr_equal_impl,
    lower_dirac_monomial as _lower_dirac_monomial_impl,
    lower_scalar_covd_monomial as _lower_scalar_covd_monomial_impl,
    lower_field_strength_monomial as _lower_field_strength_monomial_impl,
)

from .declared import (
    _DeclaredMonomial,
    _FieldFactor,
    CovariantDerivativeFactor,
    DifferentiatedCovariantFactor,
    PartialDerivativeFactor,
    GammaFactor,
    Gamma5Factor,
    MetricFactor,
    GeneratorFactor,
    StructureConstantFactor,
    FieldStrengthFactor,
    GaugeFixingDeclaration,
    GhostLagrangianDeclaration,
    _coerce_decl_factor,
    _is_decl_scalar,
)
from .interactions import (
    DerivativeAction,
    InteractionTerm,
    _field_match_key,
)
from .metadata import (
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    SPINOR_KIND,
    Field,
    gamma5_matrix,
    gamma_matrix,
    gauge_generator,
    lorentz_metric,
    structure_constant,
)
from .lagrangian import (
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    GaugeFixingTerm,
    GaugeKineticTerm,
    GhostTerm,
)


@dataclass(frozen=True)
class AnalyzedSourceTerm:
    """Canonical classification of one declared source term.

    This is the model-layer source of truth for declarative term routing. The
    compiler layer should consume these analyzed terms instead of re-deriving
    the classification from the raw declaration objects.
    """

    term: object
    interaction: Optional[InteractionTerm] = None
    covariant_core: Optional[Union[DiracKineticTerm, ComplexScalarKineticTerm]] = None
    covariant_spectators: tuple[tuple[object, bool], ...] = ()
    generic_covariant_monomial: Optional[_DeclaredMonomial] = None
    gauge_kinetic: Optional[GaugeKineticTerm] = None
    gauge_fixing: Optional[GaugeFixingTerm] = None
    ghost: Optional[GhostTerm] = None

    @property
    def needs_compilation(self) -> bool:
        return any(
            (
                self.covariant_core is not None,
                self.generic_covariant_monomial is not None,
                self.gauge_kinetic is not None,
                self.gauge_fixing is not None,
                self.ghost is not None,
            )
        )


def _match_covariant_monomial(
    term: _DeclaredMonomial,
) -> Optional[tuple[Union[DiracKineticTerm, ComplexScalarKineticTerm], tuple[tuple[object, bool], ...]]]:
    """Match one declarative ``CovD`` monomial and preserve any spectator fields.

    The exact Dirac/scalar covariant-core recognition lives in
    ``lagrangian.lowering``. This helper stays model-layer specific only because
    it has one extra job: split out the matched core from any additional local
    spectator field factors that should be carried along by the compiler.
    """
    field_factors = [factor for factor in term.factors if isinstance(factor, _FieldFactor)]
    gamma_factors = [factor for factor in term.factors if isinstance(factor, GammaFactor)]
    covd_factors = [factor for factor in term.factors if isinstance(factor, CovariantDerivativeFactor)]
    if len(term.factors) != len(field_factors) + len(gamma_factors) + len(covd_factors):
        return None

    if len(gamma_factors) == 1 and len(covd_factors) == 1:
        gamma_factor = gamma_factors[0]
        covd_factor = covd_factors[0]
        for core_slot, field_factor in enumerate(field_factors):
            if field_factor.field.kind != "fermion" or covd_factor.field.kind != "fermion":
                continue
            core = _lower_dirac_monomial_impl(
                _DeclaredMonomial(
                    coefficient=term.coefficient,
                    factors=(field_factor, gamma_factor, covd_factor),
                ),
                field_factor_cls=_FieldFactor,
                gamma_factor_cls=GammaFactor,
                covariant_derivative_factor_cls=CovariantDerivativeFactor,
                dirac_kinetic_term_cls=DiracKineticTerm,
                expression_module=Expression,
            )
            if core is None:
                continue
            spectators = tuple(
                (factor.field, factor.conjugated)
                for idx, factor in enumerate(field_factors)
                if idx != core_slot
            )
            return core, spectators

    if len(gamma_factors) == 0 and len(covd_factors) == 2:
        if any(factor.field.kind != "scalar" for factor in covd_factors):
            return None
        core = _lower_scalar_covd_monomial_impl(
            _DeclaredMonomial(
                coefficient=term.coefficient,
                factors=tuple(covd_factors),
            ),
            covariant_derivative_factor_cls=CovariantDerivativeFactor,
            complex_scalar_kinetic_term_cls=ComplexScalarKineticTerm,
        )
        if core is not None:
            spectators = tuple((factor.field, factor.conjugated) for factor in field_factors)
            return core, spectators

    return None


@dataclass(frozen=True)
class _LocalFieldEntry:
    field: Field
    conjugated: bool
    derivative_indices: tuple[object, ...]
    labels: dict


def _local_field_entry_from_factor(factor) -> Optional[_LocalFieldEntry]:
    if isinstance(factor, _FieldFactor):
        return _LocalFieldEntry(
            field=factor.field,
            conjugated=factor.conjugated,
            derivative_indices=(),
            labels=factor.labels,
        )
    if isinstance(factor, PartialDerivativeFactor):
        return _LocalFieldEntry(
            field=factor.field,
            conjugated=factor.conjugated,
            derivative_indices=tuple(factor.lorentz_indices),
            labels=factor.labels,
        )
    return None


def _is_local_chain_factor(factor) -> bool:
    return isinstance(factor, (GammaFactor, Gamma5Factor, GeneratorFactor))


def _is_local_free_tensor_factor(factor) -> bool:
    return isinstance(factor, (MetricFactor, StructureConstantFactor))


def _local_chain_kind(factor) -> str:
    if isinstance(factor, (GammaFactor, Gamma5Factor)):
        return SPINOR_KIND
    if isinstance(factor, GeneratorFactor):
        return factor.index_kind
    raise TypeError(f"Unsupported local chain factor {type(factor).__name__}")


def _local_declared_index_refs(factor) -> tuple[tuple[str, object], ...]:
    if isinstance(factor, PartialDerivativeFactor):
        return tuple((LORENTZ_KIND, index) for index in factor.lorentz_indices)
    if isinstance(factor, GammaFactor):
        return ((LORENTZ_KIND, factor.lorentz_index),)
    if isinstance(factor, MetricFactor):
        return (
            (LORENTZ_KIND, factor.left_index),
            (LORENTZ_KIND, factor.right_index),
        )
    if isinstance(factor, GeneratorFactor):
        return ((COLOR_ADJ_KIND, factor.adjoint_index),)
    if isinstance(factor, StructureConstantFactor):
        return (
            (COLOR_ADJ_KIND, factor.left_index),
            (COLOR_ADJ_KIND, factor.middle_index),
            (COLOR_ADJ_KIND, factor.right_index),
        )
    return ()


def _build_local_free_tensor_expression(factor):
    if isinstance(factor, MetricFactor):
        return lorentz_metric(factor.left_index, factor.right_index)
    if isinstance(factor, StructureConstantFactor):
        return structure_constant(
            factor.left_index,
            factor.middle_index,
            factor.right_index,
        )
    raise TypeError(f"Unsupported free local tensor factor {type(factor).__name__}")


def _fresh_local_label(prefix: str, counters: dict[str, int]):
    counters[prefix] = counters.get(prefix, 0) + 1
    return S(f"{prefix}_decl_{counters[prefix]}")


def _single_slot_position(field_obj: Field, kind: str) -> Optional[int]:
    positions = field_obj.index_positions(kind=kind)
    if len(positions) != 1:
        return None
    return positions[0]


def _resolve_endpoint_slot(
    field_obj: Field,
    slot_label_map: Mapping[int, object],
    kind: str,
) -> Optional[int]:
    positions = field_obj.index_positions(kind=kind)
    if len(positions) == 1:
        return positions[0]

    labeled_positions = [slot for slot in positions if slot in slot_label_map]
    if len(labeled_positions) == 1:
        return labeled_positions[0]
    return None


def _ensure_endpoint_labels(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    left_idx: int,
    right_idx: int,
    kind: str,
    counters: dict[str, int],
    distinct: bool,
) -> Optional[tuple[object, object]]:
    left_slot = _resolve_endpoint_slot(
        field_entries[left_idx].field,
        slot_labels[left_idx],
        kind,
    )
    right_slot = _resolve_endpoint_slot(
        field_entries[right_idx].field,
        slot_labels[right_idx],
        kind,
    )
    if left_slot is None or right_slot is None:
        return None

    left_label = slot_labels[left_idx].get(left_slot)
    right_label = slot_labels[right_idx].get(right_slot)

    if left_label is None:
        left_label = _fresh_local_label(kind, counters)
        slot_labels[left_idx][left_slot] = left_label
    if right_label is None:
        right_label = (
            _fresh_local_label(kind, counters)
            if distinct
            else left_label
        )
        slot_labels[right_idx][right_slot] = right_label

    if distinct and _expr_equal_impl(left_label, right_label):
        return None

    return left_label, right_label


def _build_chain_expression(
    factors: Sequence[object],
    *,
    kind: str,
    left_label,
    right_label,
    counters: dict[str, int],
):
    if len(factors) == 1:
        chain_labels = [left_label, right_label]
    else:
        chain_labels = [left_label] + [
            _fresh_local_label(kind, counters)
            for _ in range(len(factors) - 1)
        ] + [right_label]

    pieces = []
    for factor, start_label, end_label in zip(
        factors,
        chain_labels[:-1],
        chain_labels[1:],
    ):
        if isinstance(factor, GammaFactor):
            pieces.append(gamma_matrix(start_label, end_label, factor.lorentz_index))
            continue
        if isinstance(factor, Gamma5Factor):
            pieces.append(gamma5_matrix(start_label, end_label))
            continue
        if isinstance(factor, GeneratorFactor):
            pieces.append(
                factor.generator_builder(
                    factor.adjoint_index,
                    start_label,
                    end_label,
                )
            )
            continue
        raise TypeError(f"Unsupported chain factor {type(factor).__name__}")

    expr = Expression.num(1)
    for piece in pieces:
        expr *= piece
    return expr


def _assign_default_pair_labels(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    interval_chain_factors: Sequence[tuple[object, ...]],
    counters: dict[str, int],
):
    for interval_idx, (left, right) in enumerate(zip(field_entries, field_entries[1:])):
        if not left.conjugated or right.conjugated:
            continue
        if left.field != right.field:
            continue
        interval_kinds = {_local_chain_kind(factor) for factor in interval_chain_factors[interval_idx]}
        for slot, index in enumerate(left.field.indices):
            if index.kind == LORENTZ_KIND:
                continue
            if slot >= len(right.field.indices):
                continue
            if right.field.indices[slot] != index:
                continue
            if index.kind in interval_kinds:
                continue
            if slot in slot_labels[interval_idx] or slot in slot_labels[interval_idx + 1]:
                continue

            label = _fresh_local_label(index.prefix or index.kind, counters)
            slot_labels[interval_idx][slot] = label
            slot_labels[interval_idx + 1][slot] = label


def _ambiguous_local_attachment_error(
    *,
    factor,
    kind: str,
    labels: Sequence[object],
    available_slots: Sequence[tuple[int, int]],
) -> ValueError:
    rendered_labels = ", ".join(str(label) for label in labels)
    rendered_slots = ", ".join(
        f"{field_idx}:{slot_idx}" for field_idx, slot_idx in available_slots
    ) or "none"
    return ValueError(
        "Ambiguous local tensor attachment in declarative local monomial: "
        f"{factor} cannot uniquely bind {kind} label(s) [{rendered_labels}] "
        f"to available field slots [{rendered_slots}]."
    )


def _bind_declared_factor_indices_to_field_slots(
    *,
    factor,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
):
    declared_refs = _local_declared_index_refs(factor)
    if not declared_refs:
        return

    by_kind: dict[str, list[object]] = {}
    for kind, label in declared_refs:
        by_kind.setdefault(kind, []).append(label)

    for kind, labels in by_kind.items():
        used_slots: set[tuple[int, int]] = set()
        unresolved_labels: list[object] = []

        for label in labels:
            matches: list[tuple[int, int]] = []
            for field_idx, entry in enumerate(field_entries):
                for slot in entry.field.index_positions(kind=kind):
                    key = (field_idx, slot)
                    if key in used_slots:
                        continue
                    slot_label = slot_labels[field_idx].get(slot)
                    if slot_label is not None and _expr_equal_impl(slot_label, label):
                        matches.append(key)

            if len(matches) > 1:
                raise _ambiguous_local_attachment_error(
                    factor=factor,
                    kind=kind,
                    labels=(label,),
                    available_slots=matches,
                )
            if len(matches) == 1:
                used_slots.add(matches[0])
                continue
            unresolved_labels.append(label)

        if not unresolved_labels:
            continue

        available_slots: list[tuple[int, int]] = []
        for field_idx, entry in enumerate(field_entries):
            for slot in entry.field.index_positions(kind=kind):
                key = (field_idx, slot)
                if key in used_slots or slot in slot_labels[field_idx]:
                    continue
                available_slots.append(key)

        if not available_slots:
            continue
        if len(available_slots) != len(unresolved_labels):
            raise _ambiguous_local_attachment_error(
                factor=factor,
                kind=kind,
                labels=tuple(unresolved_labels),
                available_slots=available_slots,
            )

        for (field_idx, slot), label in zip(available_slots, unresolved_labels):
            slot_labels[field_idx][slot] = label


def _bind_declared_indices_to_field_slots(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    declared_factors: Sequence[object],
):
    for factor in declared_factors:
        _bind_declared_factor_indices_to_field_slots(
            factor=factor,
            field_entries=field_entries,
            slot_labels=slot_labels,
        )


def _assign_unique_global_pair_labels(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    counters: dict[str, int],
):
    """Share one remaining unique internal index pair across the whole monomial.

    This is the generic fallback for compact mixed-species bilinears such as
    Yukawas. If, after explicit-label binding, exactly two unlabeled slots of
    one non-Lorentz kind remain in the whole local monomial, and they sit on
    one conjugated and one unconjugated field occurrence, we treat that as the
    unique implied contraction and assign one shared label.
    """

    kinds = []
    seen = set()
    for entry in field_entries:
        for index in entry.field.indices:
            if index.kind == LORENTZ_KIND or index.kind in seen:
                continue
            seen.add(index.kind)
            kinds.append(index.kind)

    for kind in kinds:
        unlabeled_slots: list[tuple[int, int, object]] = []
        for field_idx, entry in enumerate(field_entries):
            for slot in entry.field.index_positions(kind=kind):
                if slot in slot_labels[field_idx]:
                    continue
                unlabeled_slots.append((field_idx, slot, entry.field.indices[slot]))

        if len(unlabeled_slots) != 2:
            continue

        first_idx, first_slot, first_index = unlabeled_slots[0]
        second_idx, second_slot, second_index = unlabeled_slots[1]
        if first_index != second_index:
            continue

        first_conj = bool(field_entries[first_idx].conjugated)
        second_conj = bool(field_entries[second_idx].conjugated)
        if first_conj and not second_conj:
            conj_idx, conj_slot = first_idx, first_slot
            bare_idx, bare_slot = second_idx, second_slot
        elif second_conj and not first_conj:
            conj_idx, conj_slot = second_idx, second_slot
            bare_idx, bare_slot = first_idx, first_slot
        else:
            continue
        if any(
            field_entries[mid].field.kind == "fermion"
            for mid in range(first_idx + 1, second_idx)
        ):
            continue

        label = _fresh_local_label(first_index.prefix or first_index.kind, counters)
        slot_labels[conj_idx][conj_slot] = label
        slot_labels[bare_idx][bare_slot] = label


def _fill_unassigned_local_slot_labels(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    counters: dict[str, int],
):
    for field_idx, entry in enumerate(field_entries):
        for slot, index in enumerate(entry.field.indices):
            if slot in slot_labels[field_idx]:
                continue
            prefix = index.prefix or index.kind
            slot_labels[field_idx][slot] = _fresh_local_label(prefix, counters)


def _local_label_key(label) -> str:
    return label.to_canonical_string() if hasattr(label, "to_canonical_string") else str(label)


def _count_local_field_label_occurrences(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
) -> Counter:
    counts = Counter()
    for field_idx, entry in enumerate(field_entries):
        for slot, label in slot_labels[field_idx].items():
            if label is None:
                continue
            kind = entry.field.indices[slot].kind
            counts[(kind, _local_label_key(label))] += 1
    return counts


def _rewrite_local_lorentz_slot_contraction(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    explicit_slot_labels: Sequence[dict[int, object]],
    field_label_counts: Counter,
    lorentz_index,
    counters: dict[str, int],
) -> Optional[tuple[object, object]]:
    matches: list[object] = []
    lorentz_key = _local_label_key(lorentz_index)
    if field_label_counts[(LORENTZ_KIND, lorentz_key)] != 1:
        return None

    for field_idx, entry in enumerate(field_entries):
        for slot in entry.field.index_positions(kind=LORENTZ_KIND):
            slot_label = explicit_slot_labels[field_idx].get(slot)
            if slot_label is None:
                continue
            if _expr_equal_impl(slot_label, lorentz_index):
                matches.append(slot_label)

    if len(matches) != 1:
        return None
    slot_label = matches[0]
    fresh_internal = _fresh_local_label("mu", counters)
    return slot_label, fresh_internal


def _rewrite_local_lorentz_slot_contractions(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    explicit_slot_labels: Sequence[dict[int, object]],
    counters: dict[str, int],
):
    field_label_counts = _count_local_field_label_occurrences(
        field_entries=field_entries,
        slot_labels=explicit_slot_labels,
    )
    extra_coupling = Expression.num(1)
    rewritten_derivatives: list[tuple[object, ...]] = []

    for idx, entry in enumerate(field_entries):
        rewritten_indices: list[object] = []
        for lorentz_index in entry.derivative_indices:
            rewrite = _rewrite_local_lorentz_slot_contraction(
                field_entries=field_entries,
                explicit_slot_labels=explicit_slot_labels,
                field_label_counts=field_label_counts,
                lorentz_index=lorentz_index,
                counters=counters,
            )
            if rewrite is None:
                rewritten_indices.append(lorentz_index)
                continue
            slot_label, fresh_internal = rewrite
            extra_coupling *= lorentz_metric(slot_label, fresh_internal)
            rewritten_indices.append(fresh_internal)
        rewritten_derivatives.append(tuple(rewritten_indices))

    return extra_coupling, tuple(rewritten_derivatives)


def _unsupported_local_fermion_ordering_error() -> ValueError:
    return ValueError(
        "Unsupported fermion ordering in local monomial. "
        "Local fermion terms must decompose into disjoint ordered closed Dirac "
        "bilinears with one conjugated and one unconjugated fermion endpoint."
    )


def _infer_explicit_local_dirac_bilinears(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    explicit_slot_labels: Sequence[dict[int, object]],
) -> Optional[tuple[tuple[int, int], ...]]:
    """Infer fermion chains from explicit/reused spinor labels.

    This accepts mixed-species local bilinears such as Yukawas when the source
    monomial already makes the spinor contraction explicit through repeated
    spinor labels on one ``fermion.bar`` slot and one ``fermion`` slot.

    The interaction field order is preserved exactly as written.  This helper
    only normalizes the bilinear metadata to ``(psibar_slot, psi_slot)`` so the
    downstream fermion-sign logic can recognize the chain regardless of whether
    the source monomial was written as ψ̄ … ψ or ψ … ψ̄.
    """

    fermion_slots = [
        idx for idx, entry in enumerate(field_entries) if entry.field.kind == "fermion"
    ]
    if not fermion_slots:
        return ()

    by_spinor_label: dict[str, list[int]] = {}
    original_labels: dict[str, object] = {}

    for idx in fermion_slots:
        entry = field_entries[idx]
        spinor_slot = _single_slot_position(entry.field, SPINOR_KIND)
        if spinor_slot is None:
            return None
        label = slot_labels[idx].get(spinor_slot)
        if label is None:
            return None
        key = _local_label_key(label)
        by_spinor_label.setdefault(key, []).append(idx)
        original_labels.setdefault(key, label)

    inferred: list[tuple[int, int]] = []
    covered_slots: list[int] = []
    for key, slots in by_spinor_label.items():
        if len(slots) != 2:
            return None
        first_slot, second_slot = slots
        first_conj = bool(field_entries[first_slot].conjugated)
        second_conj = bool(field_entries[second_slot].conjugated)

        if first_conj and not second_conj:
            psibar_slot, psi_slot = first_slot, second_slot
        elif second_conj and not first_conj:
            psibar_slot, psi_slot = second_slot, first_slot
        else:
            return None
        if any(
            field_entries[mid].field.kind == "fermion"
            for mid in range(first_slot + 1, second_slot)
        ):
            return None
        inferred.append((psibar_slot, psi_slot))
        covered_slots.extend(slots)

    covered_counts = Counter(covered_slots)
    if (
        len(covered_slots) != len(fermion_slots)
        or set(covered_slots) != set(fermion_slots)
        or any(count != 1 for count in covered_counts.values())
    ):
        return None

    return tuple(sorted(inferred))


def _infer_closed_local_dirac_bilinears(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    interval_supports_closed_dirac_bilinear: Sequence[bool],
    slot_labels: Optional[Sequence[dict[int, object]]] = None,
    explicit_slot_labels: Optional[Sequence[dict[int, object]]] = None,
) -> tuple[tuple[int, int], ...]:
    closed_dirac_bilinears: list[tuple[int, int]] = []
    occupied_slots: set[int] = set()
    for interval_idx, (left, right) in enumerate(zip(field_entries, field_entries[1:])):
        if not interval_supports_closed_dirac_bilinear[interval_idx]:
            continue
        if left.field.kind != "fermion" or right.field.kind != "fermion":
            continue
        if left.field.self_conjugate or right.field.self_conjugate:
            continue
        if bool(left.conjugated) == bool(right.conjugated):
            continue
        if interval_idx in occupied_slots or (interval_idx + 1) in occupied_slots:
            continue
        if bool(left.conjugated):
            closed_dirac_bilinears.append((interval_idx, interval_idx + 1))
        else:
            closed_dirac_bilinears.append((interval_idx + 1, interval_idx))
        occupied_slots.update((interval_idx, interval_idx + 1))
    closed_dirac_bilinears = tuple(closed_dirac_bilinears)

    fermion_slots = tuple(
        idx for idx, entry in enumerate(field_entries) if entry.field.kind == "fermion"
    )
    if not fermion_slots:
        return closed_dirac_bilinears

    covered_slots = [slot for pair in closed_dirac_bilinears for slot in pair]
    covered_counts = Counter(covered_slots)
    if (
        len(covered_slots) != len(fermion_slots)
        or set(covered_slots) != set(fermion_slots)
        or any(count != 1 for count in covered_counts.values())
    ):
        if slot_labels is None:
            raise _unsupported_local_fermion_ordering_error()
        if explicit_slot_labels is None:
            explicit_slot_labels = tuple({} for _ in field_entries)
        explicit_bilinears = _infer_explicit_local_dirac_bilinears(
            field_entries=field_entries,
            slot_labels=slot_labels,
            explicit_slot_labels=explicit_slot_labels,
        )
        if explicit_bilinears is None:
            raise _unsupported_local_fermion_ordering_error()
        merged = list(closed_dirac_bilinears)
        for pair in explicit_bilinears:
            if pair not in merged:
                merged.append(pair)
        return tuple(merged)

    return closed_dirac_bilinears


def _interaction_term_matches_canonical_gauge_fixing(term: InteractionTerm) -> bool:
    """Whether one lowered local term has the canonical gauge-fixing structure.

    Accepted shape:
    - exactly two vector-field occurrences of the same field
    - exactly one derivative on each field
    - coupling factorizes into
      spectator identities * g(field_lorentz_1, deriv_1) * g(field_lorentz_2, deriv_2)
      times a scalar prefactor

    This intentionally does not inspect the overall scalar coefficient, so
    helper and manual forms with the same operator structure classify together.
    """

    if len(term.fields) != 2 or len(term.derivatives) != 2:
        return False

    first, second = term.fields
    if first.field is not second.field:
        return False
    if first.conjugated or second.conjugated:
        return False

    field = first.field
    if field.kind != "vector" or not field.self_conjugate:
        return False

    derivatives_by_target = {action.target: action.lorentz_index for action in term.derivatives}
    if set(derivatives_by_target) != {0, 1}:
        return False

    lorentz_slots = field.index_positions(kind=LORENTZ_KIND)
    if len(lorentz_slots) != 1:
        return False
    lorentz_slot = lorentz_slots[0]

    first_slot_labels = field.unpack_slot_labels(first.labels)
    second_slot_labels = field.unpack_slot_labels(second.labels)
    first_lorentz = first_slot_labels.get(lorentz_slot)
    second_lorentz = second_slot_labels.get(lorentz_slot)
    if first_lorentz is None or second_lorentz is None:
        return False

    expected = (
        lorentz_metric(first_lorentz, derivatives_by_target[0])
        * lorentz_metric(second_lorentz, derivatives_by_target[1])
    )
    for slot, index in enumerate(field.indices):
        if slot == lorentz_slot:
            continue
        left_label = first_slot_labels.get(slot)
        right_label = second_slot_labels.get(slot)
        if left_label is None or right_label is None:
            return False
        expected *= index.representation.g(left_label, right_label).to_expression()

    try:
        ratio = (term.coupling / expected).expand()
    except Exception:
        return False

    blocked_labels = {
        _local_label_key(first_lorentz),
        _local_label_key(second_lorentz),
        _local_label_key(derivatives_by_target[0]),
        _local_label_key(derivatives_by_target[1]),
    }
    for slot, index in enumerate(field.indices):
        if slot == lorentz_slot:
            continue
        blocked_labels.add(_local_label_key(first_slot_labels[slot]))
        blocked_labels.add(_local_label_key(second_slot_labels[slot]))

    rendered_ratio = ratio.to_canonical_string() if hasattr(ratio, "to_canonical_string") else str(ratio)
    return not any(label in rendered_ratio for label in blocked_labels)


def _lower_local_interaction_monomial(term: _DeclaredMonomial):
    tokens: list[tuple[str, object]] = []
    field_entries: list[_LocalFieldEntry] = []
    declared_factors: list[object] = []
    free_tensor_factors: list[object] = []

    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is not None:
            field_entries.append(field_entry)
            tokens.append(("field", len(field_entries) - 1))
            if _local_declared_index_refs(factor):
                declared_factors.append(factor)
            continue
        if _is_local_chain_factor(factor):
            tokens.append(("chain", factor))
            if _local_declared_index_refs(factor):
                declared_factors.append(factor)
            continue
        if _is_local_free_tensor_factor(factor):
            tokens.append(("tensor", factor))
            free_tensor_factors.append(factor)
            if _local_declared_index_refs(factor):
                declared_factors.append(factor)
            continue
        return None

    if not field_entries:
        return None

    field_token_positions = [
        idx for idx, (kind, _value) in enumerate(tokens) if kind == "field"
    ]
    if not field_token_positions:
        return None
    if any(kind == "chain" for kind, _value in tokens[: field_token_positions[0]]):
        return None
    if any(kind == "chain" for kind, _value in tokens[field_token_positions[-1] + 1 :]):
        return None

    interval_chain_factors: list[tuple[object, ...]] = [
        () for _ in range(max(len(field_entries) - 1, 0))
    ]
    interval_supports_closed_dirac_bilinear = [
        True for _ in range(max(len(field_entries) - 1, 0))
    ]
    for interval_idx, (left_pos, right_pos) in enumerate(
        zip(field_token_positions, field_token_positions[1:])
    ):
        between = tokens[left_pos + 1 : right_pos]
        if not between:
            continue
        if any(kind == "chain" for kind, _value in between) and any(
            kind != "chain" for kind, _value in between
        ):
            return None
        if all(kind == "chain" for kind, _value in between):
            interval_chain_factors[interval_idx] = tuple(
                value for kind, value in between if kind == "chain"
            )
            continue
        interval_supports_closed_dirac_bilinear[interval_idx] = False

    coupling = term.coefficient
    for factor in free_tensor_factors:
        coupling *= _build_local_free_tensor_expression(factor)

    # Explicit labels seed lowering; auto-generated labels only fill the gaps.
    slot_labels: list[dict[int, object]] = [
        entry.field.unpack_slot_labels(entry.labels)
        for entry in field_entries
    ]
    explicit_slot_labels: list[dict[int, object]] = [dict(labels) for labels in slot_labels]
    counters: dict[str, int] = {}

    for interval_idx, factors in enumerate(interval_chain_factors):
        if not factors:
            continue

        left = field_entries[interval_idx]
        right = field_entries[interval_idx + 1]
        grouped: dict[str, list[object]] = {}
        group_order: list[str] = []
        for factor in factors:
            kind = _local_chain_kind(factor)
            if kind not in grouped:
                grouped[kind] = []
                group_order.append(kind)
            grouped[kind].append(factor)

        for kind in group_order:
            if kind == SPINOR_KIND:
                if left.field.kind != "fermion" or right.field.kind != "fermion":
                    return None
                if bool(left.conjugated) == bool(right.conjugated):
                    return None

            endpoints = _ensure_endpoint_labels(
                field_entries=field_entries,
                slot_labels=slot_labels,
                left_idx=interval_idx,
                right_idx=interval_idx + 1,
                kind=kind,
                counters=counters,
                distinct=True,
            )
            if endpoints is None:
                return None
            left_label, right_label = endpoints
            coupling *= _build_chain_expression(
                grouped[kind],
                kind=kind,
                left_label=left_label,
                right_label=right_label,
                counters=counters,
            )

    _assign_default_pair_labels(
        field_entries=field_entries,
        slot_labels=slot_labels,
        interval_chain_factors=interval_chain_factors,
        counters=counters,
    )
    _bind_declared_indices_to_field_slots(
        field_entries=field_entries,
        slot_labels=slot_labels,
        declared_factors=declared_factors,
    )
    _assign_unique_global_pair_labels(
        field_entries=field_entries,
        slot_labels=slot_labels,
        counters=counters,
    )
    _fill_unassigned_local_slot_labels(
        field_entries=field_entries,
        slot_labels=slot_labels,
        counters=counters,
    )
    divergence_coupling, rewritten_derivatives = _rewrite_local_lorentz_slot_contractions(
        field_entries=field_entries,
        slot_labels=slot_labels,
        explicit_slot_labels=explicit_slot_labels,
        counters=counters,
    )
    coupling *= divergence_coupling

    closed_dirac_bilinears = _infer_closed_local_dirac_bilinears(
        field_entries=field_entries,
        interval_supports_closed_dirac_bilinear=interval_supports_closed_dirac_bilinear,
        slot_labels=slot_labels,
        explicit_slot_labels=explicit_slot_labels,
    )

    interaction = InteractionTerm(
        coupling=coupling,
        fields=tuple(
            entry.field.occurrence(
                conjugated=bool(entry.conjugated and not entry.field.self_conjugate),
                labels=entry.field.pack_slot_labels(slot_labels[idx]),
            )
            for idx, entry in enumerate(field_entries)
        ),
        derivatives=tuple(
            DerivativeAction(target=idx, lorentz_index=lorentz_index)
            for idx, derivative_indices in enumerate(rewritten_derivatives)
            for lorentz_index in derivative_indices
        ),
        closed_dirac_bilinears=closed_dirac_bilinears,
    )
    if _interaction_term_matches_canonical_gauge_fixing(interaction):
        interaction = replace(
            interaction,
            sector="gauge_fixing",
            origin="manual_gauge_fixing",
        )
    return interaction


def _lower_field_strength_monomial(term: _DeclaredMonomial):
    return _lower_field_strength_monomial_impl(
        term,
        field_strength_factor_cls=FieldStrengthFactor,
        gauge_kinetic_term_cls=GaugeKineticTerm,
        expression_module=Expression,
    )


def _analyze_declared_source_term(term) -> Optional[AnalyzedSourceTerm]:
    interaction = _source_term_interaction(term)
    if interaction is not None:
        return AnalyzedSourceTerm(term=term, interaction=interaction)

    covariant_core = _source_term_covariant_core(term)
    if covariant_core is not None:
        spectators: tuple[tuple[object, bool], ...] = ()
        if isinstance(term, _DeclaredMonomial):
            match = _match_covariant_monomial(term)
            if match is not None:
                _core, spectators = match
        return AnalyzedSourceTerm(
            term=term,
            covariant_core=covariant_core,
            covariant_spectators=spectators,
        )

    if isinstance(term, _DeclaredMonomial):
        match = _match_covariant_monomial(term)
        if match is not None:
            core, spectators = match
            return AnalyzedSourceTerm(
                term=term,
                covariant_core=core,
                covariant_spectators=spectators,
            )

    gauge_kinetic = _source_term_gauge_kinetic(term)
    if gauge_kinetic is not None:
        return AnalyzedSourceTerm(term=term, gauge_kinetic=gauge_kinetic)

    gauge_fixing = _source_term_gauge_fixing(term)
    if gauge_fixing is not None:
        return AnalyzedSourceTerm(term=term, gauge_fixing=gauge_fixing)

    ghost = _source_term_ghost(term)
    if ghost is not None:
        return AnalyzedSourceTerm(term=term, ghost=ghost)

    if isinstance(term, _DeclaredMonomial) and _is_generic_covariant_monomial_candidate(term):
        return AnalyzedSourceTerm(term=term, generic_covariant_monomial=term)

    return None


def _unsupported_declared_source_term_error():
    return ValueError(
        "Unsupported declarative Lagrangian term. Supported canonical forms are: "
        "I * Psi.bar * Gamma(mu) * CovD(Psi, mu), "
        "CovD(Phi.bar, mu) * CovD(Phi, mu), "
        "either optionally multiplied by local spectator fields, "
        "more general local monomials with one or more CovD(...) factors that can be "
        "expanded using model gauge metadata, "
        "-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu), "
        "local monomials built from fields, PartialD(...), and one optional Gamma(...), "
        "pure local field monomials like lam * Phi * Phi * Phi * Phi, "
        "plus explicit InteractionTerm / GaugeFixing(...) / GhostLagrangian(...) "
        "or the legacy GaugeFixingTerm / GhostTerm declarations."
    )


def _validate_declared_monomial(term: _DeclaredMonomial):
    if _match_covariant_monomial(term) is not None:
        return
    if _is_generic_covariant_monomial_candidate(term):
        return
    if _lower_field_strength_monomial(term) is not None:
        return
    if _lower_local_interaction_monomial(term) is not None:
        return
    raise ValueError(
        "Unsupported declarative Lagrangian term. Supported canonical forms are: "
        "I * Psi.bar * Gamma(mu) * CovD(Psi, mu), "
        "CovD(Phi.bar, mu) * CovD(Phi, mu), "
        "either optionally multiplied by local spectator fields, "
        "more general local monomials with one or more CovD(...) factors that can be "
        "expanded using model gauge metadata, "
        "-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu), "
        "local monomials built from fields, PartialD(...), and one optional Gamma(...), "
        "pure local field monomials like lam * Phi * Phi * Phi * Phi, "
        "plus explicit InteractionTerm / GaugeFixing(...) / GhostLagrangian(...) "
        "or the legacy GaugeFixingTerm / GhostTerm declarations."
    )


def _declared_source_terms_from_item(item):
    from .lagrangian import DeclaredLagrangian

    if isinstance(item, DeclaredLagrangian):
        return item.source_terms
    if isinstance(
        item,
        (
            _DeclaredMonomial,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
            InteractionTerm,
        ),
    ):
        return (item,)
    factor = _coerce_decl_factor(item)
    if factor is not None:
        return (_DeclaredMonomial.from_factor(factor),)
    return None


def _standalone_lagrangian_context_error() -> str:
    return (
        "Standalone Lagrangian(...) only supports local terms built from "
        "fields, PartialD(...), and one optional Gamma(...). "
        "Use Model(lagrangian_decl=...) for CovD(...), FieldStrength(...), "
        "GaugeFixing(...), and GhostLagrangian(...), since those need "
        "model metadata."
    )


def _compiled_lagrangian_context_error() -> str:
    return (
        "CompiledLagrangian accepts only compiled InteractionTerm values. "
        "Use Model(lagrangian_decl=...).lagrangian() for source declarations."
    )


def _normalize_interaction_terms_input(terms) -> tuple[InteractionTerm, ...]:
    from .lagrangian import CompiledLagrangian

    if isinstance(terms, CompiledLagrangian):
        return terms.terms
    if isinstance(terms, InteractionTerm):
        normalized = (terms,)
    else:
        declared_terms = _declared_source_terms_from_item(terms)
        if declared_terms is not None:
            raise ValueError(_compiled_lagrangian_context_error())
        normalized = tuple(terms)
    if not all(isinstance(term, InteractionTerm) for term in normalized):
        if any(_declared_source_terms_from_item(term) is not None for term in normalized):
            raise ValueError(_compiled_lagrangian_context_error())
        raise TypeError("`terms=` expects InteractionTerm objects or CompiledLagrangian.")
    return normalized


def _standalone_lagrangian_source_terms_from_item(item):
    from .lagrangian import CompiledLagrangian, Lagrangian

    if isinstance(item, Lagrangian):
        return item.source_terms
    if isinstance(item, CompiledLagrangian):
        return item.terms
    if isinstance(item, InteractionTerm):
        return (item,)
    terms = _declared_source_terms_from_item(item)
    if terms is None:
        return None
    if all(_source_term_interaction(term) is not None for term in terms):
        return terms
    return None


def _normalize_lagrangian_source_terms(items) -> tuple[object, ...]:
    normalized: list[object] = []
    for item in items:
        terms = _standalone_lagrangian_source_terms_from_item(item)
        if terms is not None:
            normalized.extend(terms)
            continue
        if _declared_source_terms_from_item(item) is not None:
            raise ValueError(_standalone_lagrangian_context_error())
        raise TypeError(f"Cannot build Lagrangian from {type(item).__name__}")
    return tuple(normalized)


def _lower_standalone_lagrangian_source_term(term) -> InteractionTerm:
    interaction = _source_term_interaction(term)
    if interaction is not None:
        return interaction
    if _source_term_needs_compilation(term):
        raise ValueError(_standalone_lagrangian_context_error())
    raise TypeError(f"Cannot lower {type(term).__name__} into a standalone Lagrangian.")


def _source_term_interaction(term) -> Optional[InteractionTerm]:
    if isinstance(term, InteractionTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_local_interaction_monomial(term)
    return None


def _source_term_covariant_core(term) -> Optional[Union[DiracKineticTerm, ComplexScalarKineticTerm]]:
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm)):
        return term
    if isinstance(term, _DeclaredMonomial):
        match = _match_covariant_monomial(term)
        if match is not None:
            core, _spectators = match
            return core
    return None


def _source_term_gauge_kinetic(term) -> Optional[GaugeKineticTerm]:
    if isinstance(term, GaugeKineticTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_field_strength_monomial(term)
    return None


def _source_term_gauge_fixing(term) -> Optional[GaugeFixingTerm]:
    if isinstance(term, GaugeFixingTerm):
        return term
    if isinstance(term, GaugeFixingDeclaration):
        return GaugeFixingTerm(
            gauge_group=term.gauge_group,
            xi=term.xi,
            coefficient=term.coefficient,
            label=term.label,
        )
    return None


def _source_term_ghost(term) -> Optional[GhostTerm]:
    if isinstance(term, GhostTerm):
        return term
    if isinstance(term, GhostLagrangianDeclaration):
        return GhostTerm(
            gauge_group=term.gauge_group,
            coefficient=term.coefficient,
            label=term.label,
        )
    return None


def _source_term_needs_compilation(term) -> bool:
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm, GaugeKineticTerm, GaugeFixingTerm, GhostTerm)):
        return True
    if isinstance(term, (GaugeFixingDeclaration, GhostLagrangianDeclaration)):
        return True
    if isinstance(term, _DeclaredMonomial):
        if _match_covariant_monomial(term) is not None:
            return True
        if _is_generic_covariant_monomial_candidate(term):
            return True
        if _lower_field_strength_monomial(term) is not None:
            return True
    return False


def _declared_monomial_has_covd(term: _DeclaredMonomial) -> bool:
    return any(
        isinstance(factor, (CovariantDerivativeFactor, DifferentiatedCovariantFactor))
        for factor in term.factors
    )


def _is_generic_covariant_monomial_candidate(term: _DeclaredMonomial) -> bool:
    if not _declared_monomial_has_covd(term):
        return False

    field_factors = [factor for factor in term.factors if isinstance(factor, _FieldFactor)]
    gamma_factors = [factor for factor in term.factors if isinstance(factor, GammaFactor)]
    covd_factors = [
        factor for factor in term.factors if isinstance(factor, CovariantDerivativeFactor)
    ]
    differentiated_covd_factors = [
        factor for factor in term.factors if isinstance(factor, DifferentiatedCovariantFactor)
    ]

    # Keep malformed bare kinetic cores on the old strict-validation path.
    if (
        not differentiated_covd_factors
        and len(covd_factors) == 1
        and len(gamma_factors) == 1
        and len(field_factors) == 1
    ):
        return False
    if (
        not differentiated_covd_factors
        and len(covd_factors) == 2
        and len(gamma_factors) == 0
        and len(field_factors) == 0
    ):
        return False

    return True


def _validate_declared_source_term(term):
    if isinstance(
        term,
        (
            InteractionTerm,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
        ),
    ):
        return
    if isinstance(term, _DeclaredMonomial):
        _validate_declared_monomial(term)
        return
    raise TypeError(f"Unsupported declared Lagrangian term type: {type(term)!r}")
