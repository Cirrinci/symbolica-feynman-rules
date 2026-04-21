"""Internal lowering engine from declarations to interaction terms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

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
    _standalone_lagrangian_context_error,
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
    interaction: InteractionTerm | None = None
    covariant_core: DiracKineticTerm | ComplexScalarKineticTerm | None = None
    covariant_spectators: tuple[tuple[object, bool], ...] = ()
    gauge_kinetic: GaugeKineticTerm | None = None
    gauge_fixing: GaugeFixingTerm | None = None
    ghost: GhostTerm | None = None

    @property
    def needs_compilation(self) -> bool:
        return any(
            (
                self.covariant_core is not None,
                self.gauge_kinetic is not None,
                self.gauge_fixing is not None,
                self.ghost is not None,
            )
        )


def _match_covariant_monomial(
    term: _DeclaredMonomial,
) -> tuple[DiracKineticTerm | ComplexScalarKineticTerm, tuple[tuple[object, bool], ...]] | None:
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


def _local_field_entry_from_factor(factor) -> _LocalFieldEntry | None:
    if isinstance(factor, _FieldFactor):
        return _LocalFieldEntry(
            field=factor.field,
            conjugated=factor.conjugated,
            derivative_indices=(),
        )
    if isinstance(factor, PartialDerivativeFactor):
        return _LocalFieldEntry(
            field=factor.field,
            conjugated=factor.conjugated,
            derivative_indices=tuple(factor.lorentz_indices),
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
        return COLOR_FUND_KIND
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


def _single_slot_position(field_obj: Field, kind: str) -> int | None:
    positions = field_obj.index_positions(kind=kind)
    if len(positions) != 1:
        return None
    return positions[0]


def _ensure_endpoint_labels(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    left_idx: int,
    right_idx: int,
    kind: str,
    counters: dict[str, int],
    distinct: bool,
) -> tuple[object, object] | None:
    left_slot = _single_slot_position(field_entries[left_idx].field, kind)
    right_slot = _single_slot_position(field_entries[right_idx].field, kind)
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
            pieces.append(gauge_generator(factor.adjoint_index, start_label, end_label))
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
    non_lorentz_shared_kinds = (SPINOR_KIND, COLOR_FUND_KIND, COLOR_ADJ_KIND)

    for interval_idx, (left, right) in enumerate(zip(field_entries, field_entries[1:])):
        if not left.conjugated or right.conjugated:
            continue
        interval_kinds = {_local_chain_kind(factor) for factor in interval_chain_factors[interval_idx]}
        for kind in non_lorentz_shared_kinds:
            if kind in interval_kinds:
                continue
            shared = (
                left.field.index_kind_count(kind) == 1
                and right.field.index_kind_count(kind) == 1
            )
            if not shared:
                continue

            left_slot = _single_slot_position(left.field, kind)
            right_slot = _single_slot_position(right.field, kind)
            if left_slot is None or right_slot is None:
                continue
            if left_slot in slot_labels[interval_idx] or right_slot in slot_labels[interval_idx + 1]:
                continue

            label = _fresh_local_label(kind, counters)
            slot_labels[interval_idx][left_slot] = label
            slot_labels[interval_idx + 1][right_slot] = label


def _bind_declared_indices_to_field_slots(
    *,
    field_entries: Sequence[_LocalFieldEntry],
    slot_labels: Sequence[dict[int, object]],
    declared_refs: Sequence[tuple[str, object]],
):
    by_kind: dict[str, list[object]] = {}
    for kind, label in declared_refs:
        by_kind.setdefault(kind, []).append(label)

    for kind, labels in by_kind.items():
        candidates: list[tuple[int, int]] = []
        for field_idx, entry in enumerate(field_entries):
            for slot in entry.field.index_positions(kind=kind):
                if slot in slot_labels[field_idx]:
                    continue
                candidates.append((field_idx, slot))
        for (field_idx, slot), label in zip(candidates, labels):
            slot_labels[field_idx][slot] = label


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


def _lower_local_interaction_monomial(term: _DeclaredMonomial):
    tokens: list[tuple[str, object]] = []
    field_entries: list[_LocalFieldEntry] = []
    declared_refs: list[tuple[str, object]] = []
    free_tensor_factors: list[object] = []

    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is not None:
            field_entries.append(field_entry)
            tokens.append(("field", len(field_entries) - 1))
            declared_refs.extend(_local_declared_index_refs(factor))
            continue
        if _is_local_chain_factor(factor):
            tokens.append(("chain", factor))
            declared_refs.extend(_local_declared_index_refs(factor))
            continue
        if _is_local_free_tensor_factor(factor):
            tokens.append(("tensor", factor))
            free_tensor_factors.append(factor)
            declared_refs.extend(_local_declared_index_refs(factor))
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

    coupling = term.coefficient
    for factor in free_tensor_factors:
        coupling *= _build_local_free_tensor_expression(factor)

    slot_labels: list[dict[int, object]] = [{} for _ in field_entries]
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
                if not left.conjugated or right.conjugated:
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
        declared_refs=declared_refs,
    )
    _fill_unassigned_local_slot_labels(
        field_entries=field_entries,
        slot_labels=slot_labels,
        counters=counters,
    )

    return InteractionTerm(
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
            for idx, entry in enumerate(field_entries)
            for lorentz_index in entry.derivative_indices
        ),
    )


def _lower_field_strength_monomial(term: _DeclaredMonomial):
    return _lower_field_strength_monomial_impl(
        term,
        field_strength_factor_cls=FieldStrengthFactor,
        gauge_kinetic_term_cls=GaugeKineticTerm,
        expression_module=Expression,
    )


def _analyze_declared_source_term(term) -> AnalyzedSourceTerm | None:
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

    return None


def _unsupported_declared_source_term_error():
    return ValueError(
        "Unsupported declarative Lagrangian term. Supported canonical forms are: "
        "I * Psi.bar * Gamma(mu) * CovD(Psi, mu), "
        "CovD(Phi.bar, mu) * CovD(Phi, mu), "
        "either optionally multiplied by local spectator fields, "
        "-1/4 * FieldStrength(G, mu, nu) * FieldStrength(G, mu, nu), "
        "local monomials built from fields, PartialD(...), and one optional Gamma(...), "
        "pure local field monomials like lam * Phi * Phi * Phi * Phi, "
        "plus explicit InteractionTerm / GaugeFixing(...) / GhostLagrangian(...) "
        "or the legacy GaugeFixingTerm / GhostTerm declarations."
    )


def _validate_declared_monomial(term: _DeclaredMonomial):
    if _match_covariant_monomial(term) is not None:
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


def _normalize_interaction_terms_input(terms) -> tuple[InteractionTerm, ...]:
    from .lagrangian import Lagrangian

    if isinstance(terms, Lagrangian):
        return terms.terms
    if isinstance(terms, InteractionTerm):
        normalized = (terms,)
    else:
        normalized = tuple(terms)
    if not all(isinstance(term, InteractionTerm) for term in normalized):
        raise TypeError(
            "`terms=` expects InteractionTerm objects. "
            "For declarative input, use `Lagrangian(...)` or "
            "`Lagrangian(lagrangian_decl=...)`."
        )
    return normalized


def _standalone_lagrangian_source_terms_from_item(item):
    from .lagrangian import Lagrangian

    if isinstance(item, Lagrangian):
        return item.source_terms
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


def _source_term_interaction(term) -> InteractionTerm | None:
    if isinstance(term, InteractionTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_local_interaction_monomial(term)
    return None


def _source_term_covariant_core(term) -> DiracKineticTerm | ComplexScalarKineticTerm | None:
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm)):
        return term
    if isinstance(term, _DeclaredMonomial):
        match = _match_covariant_monomial(term)
        if match is not None:
            core, _spectators = match
            return core
    return None


def _source_term_gauge_kinetic(term) -> GaugeKineticTerm | None:
    if isinstance(term, GaugeKineticTerm):
        return term
    if isinstance(term, _DeclaredMonomial):
        return _lower_field_strength_monomial(term)
    return None


def _source_term_gauge_fixing(term) -> GaugeFixingTerm | None:
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


def _source_term_ghost(term) -> GhostTerm | None:
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
        if _lower_field_strength_monomial(term) is not None:
            return True
    return False


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
