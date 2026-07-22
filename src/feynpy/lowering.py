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
    expand_field_strength_factor as _expand_field_strength_factor_impl,
)

from .declared import (
    _DeclaredMonomial,
    _FieldFactor,
    CovariantDerivativeFactor,
    CovariantDerivativeOperatorFactor,
    DifferentiatedCovariantFactor,
    DifferentiatedOperatorFactor,
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
    COLOR_FUND_INDEX,
    COLOR_ADJ_INDEX,
    IndexType,
    LORENTZ_INDEX,
    Parameter,
    SPINOR_KIND,
    SPINOR_INDEX,
    Field,
    indices_compatible_for_labels,
    is_lorentz_index,
    representation_family,
    lorentz_index_for,
    lorentz_slots_for,
    spinor_kind_for,
    spinor_slots_for,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
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
    field_strength_monomial: Optional[_DeclaredMonomial] = None
    gauge_fixing: Optional[GaugeFixingTerm] = None
    ghost: Optional[GhostTerm] = None

    @property
    def needs_compilation(self) -> bool:
        return any(
            (
                self.covariant_core is not None,
                self.generic_covariant_monomial is not None,
                self.field_strength_monomial is not None,
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
            if spectators and any(
                factor.labels
                for factor in (*covd_factors, *field_factors)
            ):
                # Spectator-decorated scalar cores only preserve the original
                # derivative-slot pairing when the monomial does not pin the
                # spectator contractions with explicit labels. Once labels are
                # explicit, the generic CovD expansion path is the faithful one.
                return None
            return core, spectators

    return None


@dataclass(frozen=True)
class _LocalFieldEntry:
    field: Field
    conjugated: bool
    derivative_indices: tuple[object, ...]
    labels: dict


@dataclass(frozen=True)
class _ParsedLocalMonomial:
    field_entries: tuple[_LocalFieldEntry, ...]
    declared_factors: tuple[object, ...]
    free_tensor_factors: tuple[object, ...]
    interval_chain_factors: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class _CollectedLocalTypedIndexLabels:
    """Typed index references that may attach to local field slots.

    ``refs`` carries the full ``IndexType`` (not just a ``kind`` string) so a
    single coherent pipeline can distinguish, e.g., color-adjoint from
    weak-adjoint labels regardless of whether they originate from a declarative
    factor, a raw Spenso tensor, or an indexed-symbol parameter. ``factor`` is
    either the originating declared factor or a short origin description for
    refs collected from the monomial coefficient.
    """

    factor: object
    refs: tuple[tuple[IndexType, object], ...]


@dataclass(frozen=True)
class _LocalSlotRef:
    field_idx: int
    slot: int


@dataclass(frozen=True)
class _LocalDerivativeRef:
    field_idx: int
    ordinal: int


@dataclass(frozen=True)
class _LocalResolvedBinding:
    kind: str
    label: object
    field_slots: tuple[_LocalSlotRef, ...] = ()
    derivatives: tuple[_LocalDerivativeRef, ...] = ()


@dataclass(frozen=True)
class _LocalChainBinding:
    kind: str
    left: _LocalSlotRef
    right: _LocalSlotRef
    factors: tuple[object, ...]


@dataclass
class _LocalLoweringState:
    """Mutable local-lowering state.

    Phase order matters:
    1. initialize with parsed factors and source-provided ``explicit_slot_labels``
    2. apply chain bindings; this may seed additional slot labels and records
       endpoint topology in ``chain_bindings``
    3. resolve all remaining slot labels structurally, then refresh
       ``resolved_bindings`` from the final ``slot_labels``
    4. derivative rewriting and fermion-pair extraction only read the resolved
       state; they must not mutate slot topology

    Authoritative fields:
    - ``slot_labels``: final field-slot labels after local lowering
    - ``explicit_slot_labels``: labels written explicitly in the source term;
      only used for derivative-slot contraction detection
    - ``chain_bindings``: explicit local chain endpoint topology
    - ``resolved_bindings``: grouped structural view derived from ``slot_labels``
    """

    parsed: _ParsedLocalMonomial
    typed_index_labels: tuple[_CollectedLocalTypedIndexLabels, ...]
    coupling: object
    slot_labels: list[dict[int, object]]
    explicit_slot_labels: list[dict[int, object]]
    counters: dict[str, int]
    chain_bindings: list[_LocalChainBinding]
    resolved_bindings: tuple[_LocalResolvedBinding, ...] = ()

    @property
    def field_entries(self) -> tuple[_LocalFieldEntry, ...]:
        return self.parsed.field_entries


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


def _local_chain_kind(factor, *, spinor_kind: str = SPINOR_KIND) -> str:
    if isinstance(factor, (GammaFactor, Gamma5Factor)):
        return spinor_kind
    if isinstance(factor, GeneratorFactor):
        return factor.index_kind
    raise TypeError(f"Unsupported local chain factor {type(factor).__name__}")


def _factor_has_typed_index_refs(factor) -> bool:
    """Whether a declared factor exposes typed index labels for local binding.

    This is the single gate used to decide which declared factors participate in
    the unified typed-index pipeline; it shares the same extractor
    (``_declared_factor_explicit_label_refs``) as validation and slot binding so
    there is no second, divergent notion of "which indices a factor carries".
    """
    return bool(_declared_factor_explicit_label_refs(factor))


def _parse_local_interaction_factors(
    term: _DeclaredMonomial,
) -> Optional[_ParsedLocalMonomial]:
    tokens: list[tuple[str, object]] = []
    field_entries: list[_LocalFieldEntry] = []
    declared_factors: list[object] = []
    free_tensor_factors: list[object] = []

    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is not None:
            field_entries.append(field_entry)
            tokens.append(("field", len(field_entries) - 1))
            if _factor_has_typed_index_refs(factor):
                declared_factors.append(factor)
            continue
        if _is_local_chain_factor(factor):
            tokens.append(("chain", factor))
            if _factor_has_typed_index_refs(factor):
                declared_factors.append(factor)
            continue
        if _is_local_free_tensor_factor(factor):
            tokens.append(("tensor", factor))
            free_tensor_factors.append(factor)
            if _factor_has_typed_index_refs(factor):
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

    return _ParsedLocalMonomial(
        field_entries=tuple(field_entries),
        declared_factors=tuple(declared_factors),
        free_tensor_factors=tuple(free_tensor_factors),
        interval_chain_factors=tuple(interval_chain_factors),
    )


def _declared_label_name(label) -> str:
    return str(label)


def _atom_symbol_name(head: str) -> str:
    return head.rsplit("::", 1)[-1]


def _expression_variable_names(expr) -> set[str]:
    if not hasattr(expr, "to_atom_tree"):
        return set()
    try:
        root = expr.to_atom_tree()
    except Exception:
        return set()

    names: set[str] = set()

    def visit(node):
        if str(node.atom_type) == "AtomType.Var":
            names.add(_atom_symbol_name(node.head))
        for child in node.tail:
            visit(child)

    visit(root)
    return names


def _is_symbolic_index_label(label) -> bool:
    if isinstance(label, str):
        return True
    if not hasattr(label, "to_atom_tree"):
        return False
    try:
        node = label.to_atom_tree()
    except Exception:
        return False
    return str(node.atom_type) == "AtomType.Var"


def _parameter_head_map(parameters: Sequence[Parameter]) -> dict[str, Parameter]:
    return {
        f"python::{str(parameter.symbol)}": parameter
        for parameter in parameters
    }


_SPENSO_SLOT_HEAD_TO_FAMILY_PREFIX = {
    "spenso::bis": "bis",
    "spenso::mink": "mink",
    "spenso::cof": "cof",
    "spenso::coad": "coad",
}

_CANONICAL_INDEX_TYPES_BY_REP_FAMILY = {
    representation_family(index.representation): index
    for index in (
        SPINOR_INDEX,
        LORENTZ_INDEX,
        COLOR_FUND_INDEX,
        COLOR_ADJ_INDEX,
        WEAK_FUND_INDEX,
        WEAK_ADJ_INDEX,
    )
}


def _spenso_slot_ref(node) -> Optional[tuple[str, str]]:
    if str(node.atom_type) != "AtomType.Fn":
        return None
    family_prefix = _SPENSO_SLOT_HEAD_TO_FAMILY_PREFIX.get(node.head)
    if family_prefix is None or len(node.tail) != 2:
        return None
    dim_node, label_node = node.tail
    if str(label_node.atom_type) != "AtomType.Var":
        return None
    try:
        dimension = int(str(dim_node.head))
    except Exception:
        return None
    return f"{family_prefix}({dimension})", _atom_symbol_name(label_node.head)


def _rep_family_matches_index_type(rep_family: str, index: IndexType) -> bool:
    return representation_family(index.representation) == rep_family


def _coefficient_index_type_candidates(
    term: _DeclaredMonomial,
    *,
    parameters: Sequence[Parameter] = (),
) -> tuple[IndexType, ...]:
    candidates: list[IndexType] = []

    def append_candidate(index: IndexType):
        if any(existing == index for existing in candidates):
            return
        candidates.append(index)

    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is not None:
            for index in field_entry.field.indices:
                append_candidate(index)

    for parameter in parameters:
        for index in parameter.indices:
            append_candidate(index)

    for index in _CANONICAL_INDEX_TYPES_BY_REP_FAMILY.values():
        append_candidate(index)

    return tuple(candidates)


def _resolve_spenso_slot_index_type(
    *,
    rep_family: str,
    label_name: str,
    label_bindings: Mapping[str, tuple[IndexType, str]],
    candidates: Sequence[IndexType],
) -> Optional[IndexType]:
    prior = label_bindings.get(label_name)
    if prior is not None:
        prior_index, _origin = prior
        if _rep_family_matches_index_type(rep_family, prior_index):
            return prior_index

    matches = [
        index
        for index in candidates
        if _rep_family_matches_index_type(rep_family, index)
    ]
    if len(matches) == 1:
        return matches[0]

    return _CANONICAL_INDEX_TYPES_BY_REP_FAMILY.get(rep_family)


def _coefficient_typed_index_refs(
    term: _DeclaredMonomial,
    *,
    label_bindings: Mapping[str, tuple[IndexType, str]],
    parameters: Sequence[Parameter] = (),
) -> tuple[tuple[IndexType, object, str], ...]:
    if not hasattr(term.coefficient, "to_atom_tree"):
        return ()
    try:
        root = term.coefficient.to_atom_tree()
    except Exception:
        return ()

    parameter_heads = _parameter_head_map(parameters)
    candidates = _coefficient_index_type_candidates(term, parameters=parameters)
    refs: list[tuple[IndexType, object, str]] = []

    def visit(node):
        if str(node.atom_type) == "AtomType.Fn":
            parameter = parameter_heads.get(node.head)
            if parameter is not None and len(node.tail) == len(parameter.indices):
                for slot, (index, arg) in enumerate(zip(parameter.indices, node.tail), start=1):
                    if str(arg.atom_type) != "AtomType.Var":
                        continue
                    refs.append(
                        (
                            index,
                            S(_atom_symbol_name(arg.head)),
                            f"{parameter.name} slot {slot}",
                        )
                    )

            slot_ref = _spenso_slot_ref(node)
            if slot_ref is not None:
                rep_family, label_name = slot_ref
                index = _resolve_spenso_slot_index_type(
                    rep_family=rep_family,
                    label_name=label_name,
                    label_bindings=label_bindings,
                    candidates=candidates,
                )
                if index is not None:
                    refs.append(
                        (
                            index,
                            S(label_name),
                            f"coefficient tensor {node.head}",
                        )
                    )

        for child in node.tail:
            visit(child)

    visit(root)
    return tuple(refs)


def _coefficient_spinor_label_edges(expr) -> tuple[tuple[object, object], ...]:
    if not hasattr(expr, "to_atom_tree"):
        return ()
    try:
        root = expr.to_atom_tree()
    except Exception:
        return ()

    edges: list[tuple[object, object]] = []

    def visit(node):
        if str(node.atom_type) == "AtomType.Fn":
            spinor_labels: list[object] = []
            for child in node.tail:
                slot_ref = _spenso_slot_ref(child)
                if slot_ref is None:
                    continue
                rep_family, label_name = slot_ref
                if rep_family.startswith("bis("):
                    spinor_labels.append(S(label_name))
            if len(spinor_labels) == 2:
                edges.append((spinor_labels[0], spinor_labels[1]))
        for child in node.tail:
            visit(child)

    visit(root)
    return tuple(edges)


def _register_declared_label_binding(
    label_bindings: dict[str, tuple[IndexType, str]],
    label,
    index: IndexType,
    *,
    origin: str,
):
    if not _is_symbolic_index_label(label):
        return
    label_name = _declared_label_name(label)
    prior = label_bindings.get(label_name)
    if prior is None:
        label_bindings[label_name] = (index, origin)
        return
    prior_index, prior_origin = prior
    if not indices_compatible_for_labels(prior_index, index):
        raise ValueError(
            f"Index label {label_name!r} is used with incompatible index types "
            f"{prior_index.name!r} and {index.name!r} in one monomial "
            f"({prior_origin}; {origin})."
        )


def _gauge_group_adjoint_index_type(gauge_group) -> Optional[IndexType]:
    """Resolve the adjoint ``IndexType`` carried by a gauge group's boson.

    The adjoint representation is read from the gauge boson field rather than
    assumed to be color, so an SU(2) field strength yields the weak adjoint and
    an SU(3) field strength yields the color adjoint through one code path.
    Returns ``None`` when the group cannot be resolved (e.g. it was supplied by
    name only) so callers can fall back to a sensible default.
    """
    gauge_boson = getattr(gauge_group, "gauge_boson", None)
    if gauge_boson is None or not hasattr(gauge_boson, "indices"):
        return None
    adjoint_indices = [
        index for index in gauge_boson.indices if not is_lorentz_index(index)
    ]
    if len(adjoint_indices) == 1:
        return adjoint_indices[0]
    return None


def _declared_factor_explicit_label_refs(
    factor,
    *,
    lorentz_index: IndexType = LORENTZ_INDEX,
) -> tuple[tuple[IndexType, object, str], ...]:
    refs: list[tuple[IndexType, object, str]] = []
    if isinstance(factor, CovariantDerivativeFactor):
        refs.append((lorentz_index, factor.lorentz_index, f"CovD({factor.field.name})"))
        refs.extend(
            (factor.field.indices[slot], label, f"CovD({factor.field.name})")
            for slot, label in factor.field.unpack_slot_labels(factor.labels).items()
        )
    elif isinstance(factor, DifferentiatedCovariantFactor):
        covariant = factor.covariant_factor
        refs.append((lorentz_index, covariant.lorentz_index, f"CovD({covariant.field.name})"))
        refs.extend(
            (covariant.field.indices[slot], label, f"CovD({covariant.field.name})")
            for slot, label in covariant.field.unpack_slot_labels(covariant.labels).items()
        )
        refs.extend(
            (lorentz_index, lorentz_index_label, f"PartialD(CovD({covariant.field.name}))")
            for lorentz_index_label in factor.lorentz_indices
        )
    elif isinstance(factor, PartialDerivativeFactor):
        refs.extend(
            (lorentz_index, lorentz_index_label, f"PartialD({factor.field.name})")
            for lorentz_index_label in factor.lorentz_indices
        )
    elif isinstance(factor, GammaFactor):
        refs.append((lorentz_index, factor.lorentz_index, "Gamma"))
    elif isinstance(factor, MetricFactor):
        refs.append((lorentz_index, factor.left_index, "Metric"))
        refs.append((lorentz_index, factor.right_index, "Metric"))
    elif isinstance(factor, GeneratorFactor):
        adjoint_index = (
            WEAK_ADJ_INDEX
            if factor.index_kind in (WEAK_FUND_INDEX.kind, WEAK_ADJ_INDEX.kind)
            else COLOR_ADJ_INDEX
        )
        refs.append((adjoint_index, factor.adjoint_index, "T"))
    elif isinstance(factor, StructureConstantFactor):
        refs.append((COLOR_ADJ_INDEX, factor.left_index, "StructureConstant"))
        refs.append((COLOR_ADJ_INDEX, factor.middle_index, "StructureConstant"))
        refs.append((COLOR_ADJ_INDEX, factor.right_index, "StructureConstant"))
    elif isinstance(factor, FieldStrengthFactor):
        refs.append((lorentz_index, factor.left_index, "FieldStrength"))
        refs.append((lorentz_index, factor.right_index, "FieldStrength"))
        if factor.adjoint_index is not None:
            adjoint_index = (
                _gauge_group_adjoint_index_type(factor.gauge_group) or COLOR_ADJ_INDEX
            )
            refs.append((adjoint_index, factor.adjoint_index, "FieldStrength"))
    elif isinstance(factor, CovariantDerivativeOperatorFactor):
        refs.extend(
            _declared_factor_explicit_label_refs(
                factor.operand,
                lorentz_index=lorentz_index,
            )
        )
        refs.append((lorentz_index, factor.lorentz_index, f"DC({factor.operand})"))
    elif isinstance(factor, DifferentiatedOperatorFactor):
        refs.extend(
            _declared_factor_explicit_label_refs(
                factor.operand,
                lorentz_index=lorentz_index,
            )
        )
        refs.extend(
            (lorentz_index, lorentz_index_label, f"PartialD({factor.operand})")
            for lorentz_index_label in factor.lorentz_indices
        )
    return tuple(refs)


def _resolve_lorentz_index_from_term(term: _DeclaredMonomial) -> IndexType:
    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is None:
            continue
        lorentz_index = lorentz_index_for(field_entry.field.indices)
        if lorentz_index is not None:
            return lorentz_index
    return LORENTZ_INDEX


def _declared_only_label_bindings(
    term: _DeclaredMonomial,
    *,
    parameters: Sequence[Parameter] = (),
) -> dict[str, tuple[IndexType, str]]:
    """Typed label bindings from field slots and declared factors only.

    This deliberately excludes raw coefficient tensors so it can be used as the
    *prior context* when resolving the index types of raw coefficient slots
    (which may be ambiguous on their own).
    """
    label_bindings: dict[str, tuple[IndexType, str]] = {}
    lorentz_index = _resolve_lorentz_index_from_term(term)

    for factor in term.factors:
        field_entry = _local_field_entry_from_factor(factor)
        if field_entry is not None:
            slot_labels = field_entry.field.unpack_slot_labels(field_entry.labels)
            for slot, label in slot_labels.items():
                _register_declared_label_binding(
                    label_bindings,
                    label,
                    field_entry.field.indices[slot],
                    origin=f"{field_entry.field.name} slot {slot + 1}",
                )
        for index, label, origin in _declared_factor_explicit_label_refs(
            factor,
            lorentz_index=lorentz_index,
        ):
            _register_declared_label_binding(
                label_bindings,
                label,
                index,
                origin=origin,
            )

    return label_bindings


def _validate_declared_label_bindings(
    term: _DeclaredMonomial,
    *,
    parameters: Sequence[Parameter] = (),
):
    label_bindings = _declared_only_label_bindings(term, parameters=parameters)

    for index, label, origin in _coefficient_typed_index_refs(
        term,
        label_bindings=label_bindings,
        parameters=parameters,
    ):
        _register_declared_label_binding(
            label_bindings,
            label,
            index,
            origin=origin,
        )


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


def _local_slot_key(ref: _LocalSlotRef) -> tuple[int, int]:
    return (ref.field_idx, ref.slot)


def _local_slot_index(
    state: _LocalLoweringState,
    ref: _LocalSlotRef,
) -> IndexType:
    return state.field_entries[ref.field_idx].field.indices[ref.slot]


def _local_slot_label(
    state: _LocalLoweringState,
    ref: _LocalSlotRef,
):
    return state.slot_labels[ref.field_idx].get(ref.slot)


def _local_slot_is_unlabeled(
    state: _LocalLoweringState,
    ref: _LocalSlotRef,
) -> bool:
    return ref.slot not in state.slot_labels[ref.field_idx]


def _assign_local_slot_label(
    state: _LocalLoweringState,
    ref: _LocalSlotRef,
    label,
):
    state.slot_labels[ref.field_idx][ref.slot] = label


def _fresh_local_slot_label(
    state: _LocalLoweringState,
    ref: _LocalSlotRef,
):
    index = _local_slot_index(state, ref)
    return _fresh_local_label(index.prefix or index.kind, state.counters)


def _local_slot_refs_for_kind(
    state: _LocalLoweringState,
    kind: str,
) -> tuple[_LocalSlotRef, ...]:
    refs: list[_LocalSlotRef] = []
    for field_idx, entry in enumerate(state.field_entries):
        refs.extend(
            _LocalSlotRef(field_idx=field_idx, slot=slot)
            for slot in entry.field.index_positions(kind=kind)
        )
    return tuple(refs)


def _local_open_slot_refs_by_kind(
    state: _LocalLoweringState,
    *,
    include_lorentz: bool = False,
) -> dict[str, list[_LocalSlotRef]]:
    grouped: dict[str, list[_LocalSlotRef]] = {}
    for field_idx, entry in enumerate(state.field_entries):
        for slot, index in enumerate(entry.field.indices):
            if not include_lorentz and is_lorentz_index(index):
                continue
            if slot in state.slot_labels[field_idx]:
                continue
            grouped.setdefault(index.kind, []).append(
                _LocalSlotRef(field_idx=field_idx, slot=slot)
            )
    return grouped


def _local_derivative_index_type(entry: _LocalFieldEntry) -> IndexType:
    return lorentz_index_for(entry.field.indices) or LORENTZ_INDEX


def _build_local_resolved_bindings(
    state: _LocalLoweringState,
    *,
    slot_labels: Sequence[dict[int, object]],
) -> tuple[_LocalResolvedBinding, ...]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    order: list[tuple[str, str]] = []

    def ensure_group(kind: str, label):
        key = (kind, _local_label_key(label))
        if key not in grouped:
            grouped[key] = {
                "kind": kind,
                "label": label,
                "field_slots": [],
                "derivatives": [],
            }
            order.append(key)
        return grouped[key]

    for field_idx, entry in enumerate(state.field_entries):
        for slot, label in slot_labels[field_idx].items():
            if label is None:
                continue
            kind = entry.field.indices[slot].kind
            group = ensure_group(kind, label)
            group["field_slots"].append(_LocalSlotRef(field_idx=field_idx, slot=slot))

        for ordinal, derivative_label in enumerate(entry.derivative_indices):
            group = ensure_group(
                _local_derivative_index_type(entry).kind,
                derivative_label,
            )
            group["derivatives"].append(
                _LocalDerivativeRef(field_idx=field_idx, ordinal=ordinal)
            )

    return tuple(
        _LocalResolvedBinding(
            kind=grouped[key]["kind"],
            label=grouped[key]["label"],
            field_slots=tuple(grouped[key]["field_slots"]),
            derivatives=tuple(grouped[key]["derivatives"]),
        )
        for key in order
    )


def _refresh_state_resolved_bindings(state: _LocalLoweringState):
    state.resolved_bindings = _build_local_resolved_bindings(
        state,
        slot_labels=state.slot_labels,
    )


def _ambiguous_local_chain_attachment_error(
    *,
    field: Field,
    kind: str,
    positions: Sequence[int],
) -> ValueError:
    rendered_positions = ", ".join(str(slot + 1) for slot in positions)
    return ValueError(
        "Ambiguous local chain attachment in declarative local monomial: "
        f"{field.name} has repeated {kind} slots [{rendered_positions}] "
        "and needs explicit labels to choose the chain endpoint."
    )


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
    if len(positions) > 1:
        raise _ambiguous_local_chain_attachment_error(
            field=field_obj,
            kind=kind,
            positions=positions,
        )
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
) -> Optional[tuple[int, int, object, object]]:
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

    return left_slot, right_slot, left_label, right_label


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


def _free_coefficient_index_refs(
    coeff_refs: Sequence[tuple[IndexType, object, str]],
) -> tuple[tuple[IndexType, object], ...]:
    """Coefficient index labels that are free (appear exactly once).

    A label that appears more than once inside the coefficient is internally
    contracted (Einstein summation, e.g. the shared spinor index of a raw
    ``gamma(s1,s2,mu) gamma(s2,s3,nu)`` chain) and must not be attached to a
    field slot. Only genuinely free labels are candidates for slot binding.
    """
    counts: Counter = Counter(_local_label_key(label) for _index, label, _origin in coeff_refs)
    free: list[tuple[IndexType, object]] = []
    seen: set[str] = set()
    for index, label, _origin in coeff_refs:
        key = _local_label_key(label)
        if counts[key] != 1 or key in seen:
            continue
        seen.add(key)
        free.append((index, label))
    return tuple(free)


def _collect_local_typed_index_labels(
    term: _DeclaredMonomial,
    parsed: _ParsedLocalMonomial,
    *,
    parameters: Sequence[Parameter] = (),
) -> tuple[_CollectedLocalTypedIndexLabels, ...]:
    """Collect every typed index reference that may bind to a field slot.

    This is the single, source-agnostic entry point: declarative factors, raw
    Spenso coefficient tensors, and indexed-symbol parameters all contribute
    refs carrying their resolved ``IndexType``, so slot attachment follows one
    coherent path regardless of how the index was written.
    """
    lorentz_index = _resolve_lorentz_index_from_term(term)
    collected: list[_CollectedLocalTypedIndexLabels] = []

    for factor in parsed.declared_factors:
        refs = tuple(
            (index, label)
            for index, label, _origin in _declared_factor_explicit_label_refs(
                factor,
                lorentz_index=lorentz_index,
            )
            if _is_symbolic_index_label(label)
        )
        if refs:
            collected.append(
                _CollectedLocalTypedIndexLabels(factor=factor, refs=refs)
            )

    declared_bindings = _declared_only_label_bindings(term, parameters=parameters)
    coeff_refs = _coefficient_typed_index_refs(
        term,
        label_bindings=declared_bindings,
        parameters=parameters,
    )
    free_refs = _free_coefficient_index_refs(coeff_refs)
    if free_refs:
        collected.append(
            _CollectedLocalTypedIndexLabels(
                factor="coefficient tensors",
                refs=free_refs,
            )
        )

    return tuple(collected)


def _build_local_tensor_coupling(
    *,
    coefficient,
    free_tensor_factors: Sequence[object],
):
    coupling = coefficient
    for factor in free_tensor_factors:
        coupling *= _build_local_free_tensor_expression(factor)
    return coupling


def _seed_local_field_slot_labels(
    field_entries: Sequence[_LocalFieldEntry],
) -> tuple[list[dict[int, object]], list[dict[int, object]]]:
    slot_labels = [
        entry.field.unpack_slot_labels(entry.labels)
        for entry in field_entries
    ]
    return slot_labels, [dict(labels) for labels in slot_labels]


def _initialize_local_lowering_state(
    *,
    coefficient,
    parsed: _ParsedLocalMonomial,
    typed_index_labels: Sequence[_CollectedLocalTypedIndexLabels],
):
    coupling = _build_local_tensor_coupling(
        coefficient=coefficient,
        free_tensor_factors=parsed.free_tensor_factors,
    )
    slot_labels, explicit_slot_labels = _seed_local_field_slot_labels(
        parsed.field_entries
    )
    return _LocalLoweringState(
        parsed=parsed,
        typed_index_labels=tuple(typed_index_labels),
        coupling=coupling,
        slot_labels=slot_labels,
        explicit_slot_labels=explicit_slot_labels,
        counters={},
        chain_bindings=[],
    )


def _bind_state_typed_index_labels_to_slots(
    state: _LocalLoweringState,
):
    for collected in state.typed_index_labels:
        # Group by ``kind`` (derived from the resolved IndexType). Distinct
        # adjoint families such as color-adjoint and weak-adjoint carry distinct
        # kinds, so grouping by kind keeps them apart while remaining lenient for
        # compatible-but-distinct Lorentz/spinor index types.
        by_kind: dict[str, list[object]] = {}
        for index, label in collected.refs:
            by_kind.setdefault(index.kind, []).append(label)

        for kind, labels in by_kind.items():
            candidates = _local_slot_refs_for_kind(state, kind)
            used_slots: set[tuple[int, int]] = set()
            unresolved_labels: list[object] = []

            for label in labels:
                matches = [
                    ref
                    for ref in candidates
                    if _local_slot_key(ref) not in used_slots
                    and (slot_label := _local_slot_label(state, ref)) is not None
                    and _expr_equal_impl(slot_label, label)
                ]
                if matches:
                    # The label is already present on one or more slots, i.e. it
                    # is already placed/contracted (e.g. a shared flavor label
                    # joining two field slots). Mark those slots used; there is
                    # nothing to assign.
                    for ref in matches:
                        used_slots.add(_local_slot_key(ref))
                    continue
                unresolved_labels.append(label)

            if not unresolved_labels:
                continue

            available = [
                ref
                for ref in candidates
                if _local_slot_key(ref) not in used_slots
                and _local_slot_is_unlabeled(state, ref)
            ]
            if not available:
                continue
            if len(available) != len(unresolved_labels):
                raise _ambiguous_local_attachment_error(
                    factor=collected.factor,
                    kind=kind,
                    labels=tuple(unresolved_labels),
                    available_slots=tuple(_local_slot_key(ref) for ref in available),
                )

            for ref, label in zip(available, unresolved_labels):
                _assign_local_slot_label(state, ref, label)


def _adjacent_local_pair_candidates(
    state: _LocalLoweringState,
) -> tuple[tuple[_LocalSlotRef, _LocalSlotRef], ...]:
    candidates: list[tuple[_LocalSlotRef, _LocalSlotRef]] = []
    for interval_idx, (left, right) in enumerate(
        zip(state.field_entries, state.field_entries[1:])
    ):
        if not left.conjugated or right.conjugated:
            continue
        if left.field != right.field:
            continue
        interval_kinds = {
            _local_chain_kind(
                factor,
                spinor_kind=spinor_kind_for(left.field.indices),
            )
            for factor in state.parsed.interval_chain_factors[interval_idx]
        }
        for slot, index in enumerate(left.field.indices):
            if is_lorentz_index(index):
                continue
            if slot >= len(right.field.indices):
                continue
            if right.field.indices[slot] != index:
                continue
            if index.kind in interval_kinds:
                continue
            candidates.append(
                (
                    _LocalSlotRef(field_idx=interval_idx, slot=slot),
                    _LocalSlotRef(field_idx=interval_idx + 1, slot=slot),
                )
            )
    return tuple(candidates)


def _can_share_unique_open_pair(
    state: _LocalLoweringState,
    left: _LocalSlotRef,
    right: _LocalSlotRef,
) -> bool:
    if _local_slot_index(state, left) != _local_slot_index(state, right):
        return False
    left_conj = bool(state.field_entries[left.field_idx].conjugated)
    right_conj = bool(state.field_entries[right.field_idx].conjugated)
    if left_conj == right_conj:
        return False
    start = min(left.field_idx, right.field_idx)
    end = max(left.field_idx, right.field_idx)
    return not any(
        state.field_entries[mid].field.kind == "fermion"
        for mid in range(start + 1, end)
    )


def _resolve_structural_local_slot_labels(
    state: _LocalLoweringState,
):
    # Explicit typed indices (declarative factors *and* raw coefficient tensors
    # / indexed-symbol parameters) bind first, so an index written on a tensor
    # always claims its matching field slot before any structural defaulting.
    _bind_state_typed_index_labels_to_slots(state)

    for left, right in _adjacent_local_pair_candidates(state):
        if not _local_slot_is_unlabeled(state, left):
            continue
        if not _local_slot_is_unlabeled(state, right):
            continue
        label = _fresh_local_slot_label(state, left)
        _assign_local_slot_label(state, left, label)
        _assign_local_slot_label(state, right, label)

    for kind, refs in _local_open_slot_refs_by_kind(state).items():
        if len(refs) != 2:
            continue
        first, second = refs
        if not _can_share_unique_open_pair(state, first, second):
            continue
        label = _fresh_local_slot_label(state, first)
        _assign_local_slot_label(state, first, label)
        _assign_local_slot_label(state, second, label)

    for field_idx, entry in enumerate(state.field_entries):
        for slot, index in enumerate(entry.field.indices):
            ref = _LocalSlotRef(field_idx=field_idx, slot=slot)
            if not _local_slot_is_unlabeled(state, ref):
                continue
            _assign_local_slot_label(
                state,
                ref,
                _fresh_local_label(index.prefix or index.kind, state.counters),
            )


def _apply_local_chain_factor_bindings(
    state: _LocalLoweringState,
) -> bool:
    for interval_idx, factors in enumerate(state.parsed.interval_chain_factors):
        if not factors:
            continue

        left = state.field_entries[interval_idx]
        right = state.field_entries[interval_idx + 1]
        grouped: dict[str, list[object]] = {}
        group_order: list[str] = []
        spinor_kind = spinor_kind_for(left.field.indices)
        for factor in factors:
            kind = _local_chain_kind(factor, spinor_kind=spinor_kind)
            if kind not in grouped:
                grouped[kind] = []
                group_order.append(kind)
            grouped[kind].append(factor)

        for kind in group_order:
            if kind == spinor_kind:
                if left.field.kind != "fermion" or right.field.kind != "fermion":
                    return False
                if bool(left.conjugated) == bool(right.conjugated):
                    return False

            endpoints = _ensure_endpoint_labels(
                field_entries=state.field_entries,
                slot_labels=state.slot_labels,
                left_idx=interval_idx,
                right_idx=interval_idx + 1,
                kind=kind,
                counters=state.counters,
                distinct=True,
            )
            if endpoints is None:
                return False
            left_slot, right_slot, left_label, right_label = endpoints
            state.chain_bindings.append(
                _LocalChainBinding(
                    kind=kind,
                    left=_LocalSlotRef(field_idx=interval_idx, slot=left_slot),
                    right=_LocalSlotRef(field_idx=interval_idx + 1, slot=right_slot),
                    factors=tuple(grouped[kind]),
                )
            )
            state.coupling *= _build_chain_expression(
                grouped[kind],
                kind=kind,
                left_label=left_label,
                right_label=right_label,
                counters=state.counters,
            )

    return True


def _identify_local_field_slot_labels(
    *,
    coefficient,
    parsed: _ParsedLocalMonomial,
    typed_index_labels: Sequence[_CollectedLocalTypedIndexLabels],
) -> Optional[_LocalLoweringState]:
    state = _initialize_local_lowering_state(
        coefficient=coefficient,
        parsed=parsed,
        typed_index_labels=typed_index_labels,
    )
    if not _apply_local_chain_factor_bindings(state):
        return None
    _resolve_structural_local_slot_labels(state)
    _refresh_state_resolved_bindings(state)
    return state


def _local_label_key(label) -> str:
    return label.to_canonical_string() if hasattr(label, "to_canonical_string") else str(label)


def _rewrite_local_lorentz_slot_contractions(
    state: _LocalLoweringState,
):
    explicit_bindings = _build_local_resolved_bindings(
        state,
        slot_labels=state.explicit_slot_labels,
    )
    lorentz_field_slot_by_label: dict[str, Optional[_LocalSlotRef]] = {}
    for binding in explicit_bindings:
        lorentz_slots = [
            ref
            for ref in binding.field_slots
            if is_lorentz_index(_local_slot_index(state, ref))
        ]
        if not lorentz_slots:
            continue
        key = _local_label_key(binding.label)
        current = lorentz_field_slot_by_label.get(key)
        if current is None and key not in lorentz_field_slot_by_label:
            if len(lorentz_slots) == 1:
                lorentz_field_slot_by_label[key] = lorentz_slots[0]
            else:
                lorentz_field_slot_by_label[key] = None
            continue
        lorentz_field_slot_by_label[key] = None

    extra_coupling = Expression.num(1)
    rewritten_derivatives: list[tuple[object, ...]] = []

    for idx, entry in enumerate(state.field_entries):
        rewritten_indices: list[object] = []
        for lorentz_index in entry.derivative_indices:
            slot_ref = lorentz_field_slot_by_label.get(_local_label_key(lorentz_index))
            if slot_ref is None:
                rewritten_indices.append(lorentz_index)
                continue
            slot_label = state.explicit_slot_labels[slot_ref.field_idx][slot_ref.slot]
            fresh_internal = _fresh_local_label("mu", state.counters)
            extra_coupling *= lorentz_metric(slot_label, fresh_internal)
            rewritten_indices.append(fresh_internal)
        rewritten_derivatives.append(tuple(rewritten_indices))

    return extra_coupling, tuple(rewritten_derivatives)


def _identify_local_derivative_labels(
    state: _LocalLoweringState,
):
    return _rewrite_local_lorentz_slot_contractions(state)


def _unsupported_local_fermion_ordering_error() -> ValueError:
    return ValueError(
        "Unsupported fermion ordering in local monomial. "
        "Local fermion terms must decompose into disjoint ordered closed Dirac "
        "bilinears with one conjugated and one unconjugated fermion endpoint."
    )


def _inconsistent_local_fermion_pairing_error() -> ValueError:
    return ValueError(
        "Inconsistent local fermion pairing inferred from chain and resolved "
        "spinor bindings."
    )


def _spinor_pair_from_slot_refs(
    state: _LocalLoweringState,
    left: _LocalSlotRef,
    right: _LocalSlotRef,
) -> Optional[tuple[int, int]]:
    left_entry = state.field_entries[left.field_idx]
    right_entry = state.field_entries[right.field_idx]
    if left_entry.field.kind != "fermion" or right_entry.field.kind != "fermion":
        return None

    left_spinor_slots = spinor_slots_for(left_entry.field)
    right_spinor_slots = spinor_slots_for(right_entry.field)
    if len(left_spinor_slots) != 1 or len(right_spinor_slots) != 1:
        return None
    if left.slot != left_spinor_slots[0] or right.slot != right_spinor_slots[0]:
        return None

    left_conj = bool(left_entry.conjugated)
    right_conj = bool(right_entry.conjugated)
    if left_conj and not right_conj:
        return (left.field_idx, right.field_idx)
    if right_conj and not left_conj:
        return (right.field_idx, left.field_idx)
    return None


def _register_local_dirac_pair(
    pairs_by_slot: dict[int, tuple[int, int]],
    pair_order: list[tuple[int, int]],
    pair: tuple[int, int],
):
    for slot in pair:
        prior = pairs_by_slot.get(slot)
        if prior is not None and prior != pair:
            raise _inconsistent_local_fermion_pairing_error()

    if pair in pair_order:
        for slot in pair:
            pairs_by_slot[slot] = pair
        return

    pair_order.append(pair)
    for slot in pair:
        pairs_by_slot[slot] = pair


def _register_local_pairs_from_coefficient_spinor_graph(
    state: _LocalLoweringState,
    *,
    pairs_by_slot: dict[int, tuple[int, int]],
    pair_order: list[tuple[int, int]],
):
    edges = _coefficient_spinor_label_edges(state.coupling)
    if not edges:
        return

    adjacency: dict[str, set[str]] = {}

    def add_node(label) -> str:
        key = _local_label_key(label)
        adjacency.setdefault(key, set())
        return key

    for left_label, right_label in edges:
        left_key = add_node(left_label)
        right_key = add_node(right_label)
        adjacency[left_key].add(right_key)
        adjacency[right_key].add(left_key)

    unpaired_spinor_refs = [
        _LocalSlotRef(field_idx=field_idx, slot=spinor_slots_for(entry.field)[0])
        for field_idx, entry in enumerate(state.field_entries)
        if entry.field.kind == "fermion"
        and field_idx not in pairs_by_slot
        and len(spinor_slots_for(entry.field)) == 1
    ]
    if not unpaired_spinor_refs:
        return

    refs_by_component: dict[str, list[_LocalSlotRef]] = {}
    visited: set[str] = set()
    component_root_by_label: dict[str, str] = {}

    def register_component(start: str) -> str:
        stack = [start]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency.get(current, ()))
        root = min(component)
        for label_key in component:
            component_root_by_label[label_key] = root
        return root

    for ref in unpaired_spinor_refs:
        label = _local_slot_label(state, ref)
        if label is None:
            continue
        key = _local_label_key(label)
        if key not in adjacency:
            continue
        root = component_root_by_label.get(key)
        if root is None:
            root = register_component(key)
        refs_by_component.setdefault(root, []).append(ref)

    for refs in refs_by_component.values():
        if len(refs) != 2:
            raise _unsupported_local_fermion_ordering_error()
        pair = _spinor_pair_from_slot_refs(state, refs[0], refs[1])
        if pair is None:
            raise _unsupported_local_fermion_ordering_error()
        _register_local_dirac_pair(pairs_by_slot, pair_order, pair)


def _ordered_local_dirac_bilinears(
    state: _LocalLoweringState,
    pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    fermion_slots = tuple(
        idx for idx, entry in enumerate(state.field_entries) if entry.field.kind == "fermion"
    )
    if not fermion_slots:
        return ()

    covered_slots = [slot for pair in pairs for slot in pair]
    covered_counts = Counter(covered_slots)
    if (
        len(covered_slots) != len(fermion_slots)
        or set(covered_slots) != set(fermion_slots)
        or any(count != 1 for count in covered_counts.values())
    ):
        raise _unsupported_local_fermion_ordering_error()

    ordered_intervals = sorted(
        (min(psibar, psi), max(psibar, psi), (psibar, psi))
        for psibar, psi in pairs
    )
    for (_, prev_end, _), (next_start, _, _) in zip(
        ordered_intervals,
        ordered_intervals[1:],
    ):
        if next_start <= prev_end:
            raise _unsupported_local_fermion_ordering_error()

    return tuple(pair for _, _, pair in ordered_intervals)


def _identify_local_contraction_pairs(
    state: _LocalLoweringState,
) -> tuple[tuple[int, int], ...]:
    pair_order: list[tuple[int, int]] = []
    pairs_by_slot: dict[int, tuple[int, int]] = {}

    for binding in state.chain_bindings:
        pair = _spinor_pair_from_slot_refs(state, binding.left, binding.right)
        if pair is not None:
            _register_local_dirac_pair(pairs_by_slot, pair_order, pair)

    for binding in state.resolved_bindings:
        spinor_refs = [
            ref
            for ref in binding.field_slots
            if ref.slot in spinor_slots_for(state.field_entries[ref.field_idx].field)
        ]
        if not spinor_refs:
            continue
        if len(spinor_refs) == 1 and len(binding.field_slots) == 1:
            continue
        if len(spinor_refs) != 2 or len(spinor_refs) != len(binding.field_slots):
            raise _unsupported_local_fermion_ordering_error()
        pair = _spinor_pair_from_slot_refs(
            state,
            spinor_refs[0],
            spinor_refs[1],
        )
        if pair is None:
            raise _unsupported_local_fermion_ordering_error()
        _register_local_dirac_pair(pairs_by_slot, pair_order, pair)

    _register_local_pairs_from_coefficient_spinor_graph(
        state,
        pairs_by_slot=pairs_by_slot,
        pair_order=pair_order,
    )

    return _ordered_local_dirac_bilinears(state, pair_order)


def _build_local_interaction_term(
    state: _LocalLoweringState,
    *,
    rewritten_derivatives: Sequence[tuple[object, ...]],
    closed_dirac_bilinears: tuple[tuple[int, int], ...],
) -> InteractionTerm:
    return InteractionTerm(
        coupling=state.coupling,
        fields=tuple(
            entry.field.occurrence(
                conjugated=bool(entry.conjugated and not entry.field.self_conjugate),
                labels=entry.field.pack_slot_labels(state.slot_labels[idx]),
            )
            for idx, entry in enumerate(state.field_entries)
        ),
        derivatives=tuple(
            DerivativeAction(target=idx, lorentz_index=lorentz_index)
            for idx, derivative_indices in enumerate(rewritten_derivatives)
            for lorentz_index in derivative_indices
        ),
        closed_dirac_bilinears=closed_dirac_bilinears,
    )


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

    lorentz_slots = lorentz_slots_for(field)
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
        if not (
            _is_symbolic_index_label(left_label)
            and _is_symbolic_index_label(right_label)
        ):
            return False
        expected *= index.representation.g(left_label, right_label).to_expression()

    try:
        ratio = (term.coupling / expected).expand()
    except Exception:
        return False

    blocked_labels = {
        _declared_label_name(first_lorentz),
        _declared_label_name(second_lorentz),
        _declared_label_name(derivatives_by_target[0]),
        _declared_label_name(derivatives_by_target[1]),
    }
    for slot, index in enumerate(field.indices):
        if slot == lorentz_slot:
            continue
        blocked_labels.add(_declared_label_name(first_slot_labels[slot]))
        blocked_labels.add(_declared_label_name(second_slot_labels[slot]))

    return _expression_variable_names(ratio).isdisjoint(blocked_labels)


def _lower_local_interaction_monomial(
    term: _DeclaredMonomial,
    *,
    parameters: Sequence[Parameter] = (),
):
    _validate_declared_label_bindings(term, parameters=parameters)
    parsed = _parse_local_interaction_factors(term)
    if parsed is None:
        return None

    # Phase order is fixed here:
    # parse -> seed/apply chain endpoints -> resolve remaining slot labels ->
    # refresh resolved bindings -> rewrite derivatives / infer fermion pairs ->
    # build the final InteractionTerm.
    typed_index_labels = _collect_local_typed_index_labels(
        term,
        parsed,
        parameters=parameters,
    )
    field_slot_state = _identify_local_field_slot_labels(
        coefficient=term.coefficient,
        parsed=parsed,
        typed_index_labels=typed_index_labels,
    )
    if field_slot_state is None:
        return None

    divergence_coupling, rewritten_derivatives = _identify_local_derivative_labels(
        field_slot_state
    )
    field_slot_state.coupling *= divergence_coupling

    closed_dirac_bilinears = _identify_local_contraction_pairs(field_slot_state)

    interaction = _build_local_interaction_term(
        field_slot_state,
        rewritten_derivatives=rewritten_derivatives,
        closed_dirac_bilinears=closed_dirac_bilinears,
    )
    if _interaction_term_matches_canonical_gauge_fixing(interaction):
        interaction = replace(
            interaction,
            sector="gauge_fixing",
            origin="manual_gauge_fixing",
        )
    return interaction


def _monomial_has_field_strength(term: _DeclaredMonomial) -> bool:
    return any(isinstance(factor, FieldStrengthFactor) for factor in term.factors)


def _declared_factor_has_covd(factor) -> bool:
    if isinstance(
        factor,
        (
            CovariantDerivativeFactor,
            DifferentiatedCovariantFactor,
            CovariantDerivativeOperatorFactor,
        ),
    ):
        return True
    if isinstance(factor, DifferentiatedOperatorFactor):
        return _declared_factor_has_covd(factor.operand)
    return False


def _declared_factor_requires_recursive_operator_expansion(factor) -> bool:
    return isinstance(
        factor,
        (
            CovariantDerivativeOperatorFactor,
            DifferentiatedOperatorFactor,
        ),
    )


def _field_strength_adjoint_label_counts(term: _DeclaredMonomial) -> Counter:
    """Count user-declared adjoint labels across one field-strength monomial.

    Adjoint labels come from ``FieldStrength`` adjoint indices and any explicit
    tensor factors that carry the matching adjoint slots, including raw Spenso
    tensors that live in ``term.coefficient``. Internally generated expansion
    labels are not counted here; they are always contracted.
    """
    counts: Counter = Counter()
    for factor in term.factors:
        if isinstance(factor, FieldStrengthFactor) and factor.adjoint_index is not None:
            counts[_local_label_key(factor.adjoint_index)] += 1
        elif isinstance(factor, StructureConstantFactor):
            for idx in (factor.left_index, factor.middle_index, factor.right_index):
                counts[_local_label_key(idx)] += 1
        elif isinstance(factor, GeneratorFactor):
            counts[_local_label_key(factor.adjoint_index)] += 1

    if hasattr(term.coefficient, "to_atom_tree"):
        for index, label, _origin in _coefficient_typed_index_refs(
            term,
            label_bindings={},
        ):
            if representation_family(index.representation).startswith("coad("):
                counts[_local_label_key(label)] += 1
    return counts


def _validate_field_strength_scalar(term: _DeclaredMonomial) -> None:
    counts = _field_strength_adjoint_label_counts(term)
    open_labels = sorted(label for label, count in counts.items() if count == 1)
    if open_labels:
        raise ValueError(
            "FieldStrength monomial has open (uncontracted) adjoint index/indices "
            f"{open_labels}; the Lagrangian term must be a gauge singlet. Contract "
            "them against another field strength or a tensor carrying the matching "
            "adjoint label(s)."
        )


def _gauge_field_lorentz_slot(gauge_field: Field, *, purpose: str) -> int:
    slots = [slot for slot, index in enumerate(gauge_field.indices) if is_lorentz_index(index)]
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires gauge field {gauge_field.name!r} to expose exactly one "
            f"Lorentz slot; found {len(slots)}."
        )
    return slots[0]


def _gauge_field_adjoint_slot(gauge_field: Field, *, purpose: str) -> int:
    slots = [
        slot for slot, index in enumerate(gauge_field.indices) if not is_lorentz_index(index)
    ]
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} requires gauge field {gauge_field.name!r} to expose exactly one "
            f"adjoint slot; found {len(slots)}."
        )
    return slots[0]


def _expand_one_field_strength(fs: FieldStrengthFactor, model, fresh_adjoint_label):
    purpose = "FieldStrength compilation"
    gauge_group = model.find_gauge_group(fs.gauge_group)
    if gauge_group is None:
        raise ValueError(
            f"{purpose} could not resolve gauge group {fs.gauge_group!r} in model.gauge_groups."
        )
    if _expr_equal_impl(fs.left_index, fs.right_index):
        raise ValueError(
            "FieldStrength indices must be distinct within one factor; "
            f"got {fs.left_index} twice in {fs}."
        )
    gauge_field = model.gauge_boson_field(gauge_group)
    lorentz_slot = _gauge_field_lorentz_slot(gauge_field, purpose=purpose)

    if gauge_group.abelian:
        if fs.adjoint_index is not None:
            raise ValueError(
                f"{purpose}: abelian gauge group {gauge_group.name!r} field strength does "
                "not take an adjoint index; write FieldStrength(group, mu, nu)."
            )
        adjoint_slot = None
        coupling = None
        structure_constant_builder = None
    else:
        if fs.adjoint_index is None:
            raise ValueError(
                f"{purpose}: non-abelian gauge group {gauge_group.name!r} field strength "
                "requires an explicit adjoint index, e.g. FieldStrength(group, mu, nu, a)."
            )
        if gauge_group.structure_constant is None or not callable(gauge_group.structure_constant):
            raise ValueError(
                f"{purpose}: non-abelian gauge group {gauge_group.name!r} needs a callable "
                "structure_constant builder for field-strength expansion."
            )
        adjoint_slot = _gauge_field_adjoint_slot(gauge_field, purpose=purpose)
        coupling = gauge_group.coupling
        structure_constant_builder = gauge_group.structure_constant

    return _expand_field_strength_factor_impl(
        gauge_field=gauge_field,
        abelian=gauge_group.abelian,
        lorentz_slot=lorentz_slot,
        adjoint_slot=adjoint_slot,
        left_index=fs.left_index,
        right_index=fs.right_index,
        adjoint_index=fs.adjoint_index,
        coupling=coupling,
        structure_constant_builder=structure_constant_builder,
        fresh_adjoint_label=fresh_adjoint_label,
        field_factor_cls=_FieldFactor,
        partial_derivative_factor_cls=PartialDerivativeFactor,
        declared_monomial_cls=_DeclaredMonomial,
        expression_module=Expression,
    )


def _expand_field_strengths_in_monomial(term: _DeclaredMonomial, model):
    """Expand every ``FieldStrength`` factor in one declared monomial.

    Returns the additive tuple of plain local monomials (the Cartesian product
    over each field strength's expansion pieces), or ``None`` when the monomial
    has no field-strength factors. The resulting monomials are ordinary local
    products of fields / derivatives / structure constants that
    ``_lower_local_interaction_monomial`` compiles unchanged, so the number of
    emitted interactions is determined entirely by the expansion.
    """
    if not _monomial_has_field_strength(term):
        return None
    _validate_field_strength_scalar(term)

    fs_factors = [factor for factor in term.factors if isinstance(factor, FieldStrengthFactor)]
    other_factors = tuple(
        factor for factor in term.factors if not isinstance(factor, FieldStrengthFactor)
    )

    counters = {"adj": 0}

    def fresh_adjoint_label():
        counters["adj"] += 1
        return S(f"fs_adj_decl_{counters['adj']}")

    monomials = [_DeclaredMonomial(coefficient=term.coefficient, factors=other_factors)]
    for fs in fs_factors:
        pieces = _expand_one_field_strength(fs, model, fresh_adjoint_label)
        monomials = [
            _DeclaredMonomial(
                coefficient=base.coefficient * piece.coefficient,
                factors=base.factors + piece.factors,
            )
            for base in monomials
            for piece in pieces
        ]
    return tuple(monomials)


def _canonical_field_strength_kinetic_info(term):
    """Recognize a canonical ``c * F(G,...) F(G,...)`` bilinear for diagnostics.

    Returns ``(gauge_group_target, normalized_coefficient)`` where the
    normalized coefficient equals ``-4 * c`` (so a canonical ``-1/4 F^2`` maps
    to ``1``), or ``None`` for non-canonical / higher field-strength monomials.
    This is a read-only validation helper and plays no role in compilation.
    """
    if not isinstance(term, _DeclaredMonomial):
        return None
    if any(not isinstance(factor, FieldStrengthFactor) for factor in term.factors):
        return None
    fs_factors = [factor for factor in term.factors if isinstance(factor, FieldStrengthFactor)]
    if len(fs_factors) != 2:
        return None
    left, right = fs_factors
    if left.gauge_group != right.gauge_group:
        return None
    if not _expr_equal_impl(left.left_index, right.left_index):
        return None
    if not _expr_equal_impl(left.right_index, right.right_index):
        return None
    if (left.adjoint_index is None) != (right.adjoint_index is None):
        return None
    if left.adjoint_index is not None and not _expr_equal_impl(
        left.adjoint_index, right.adjoint_index
    ):
        return None
    return left.gauge_group, -Expression.num(4) * term.coefficient


def _analyze_declared_source_term(
    term,
    *,
    parameters: Sequence[Parameter] = (),
) -> Optional[AnalyzedSourceTerm]:
    if isinstance(term, _DeclaredMonomial):
        _validate_declared_label_bindings(term, parameters=parameters)
    interaction = _source_term_interaction(term, parameters=parameters)
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

    if isinstance(term, _DeclaredMonomial) and _is_generic_covariant_monomial_candidate(term):
        return AnalyzedSourceTerm(term=term, generic_covariant_monomial=term)

    gauge_fixing = _source_term_gauge_fixing(term)
    if gauge_fixing is not None:
        return AnalyzedSourceTerm(term=term, gauge_fixing=gauge_fixing)

    ghost = _source_term_ghost(term)
    if ghost is not None:
        return AnalyzedSourceTerm(term=term, ghost=ghost)

    if isinstance(term, _DeclaredMonomial) and _monomial_has_field_strength(term):
        return AnalyzedSourceTerm(term=term, field_strength_monomial=term)

    return None


def _unsupported_declared_source_term_error():
    return ValueError(
        "Unsupported declarative Lagrangian term. Supported canonical forms are: "
        "I * Psi.bar * Gamma(mu) * CovD(Psi, mu), "
        "CovD(Phi.bar, mu) * CovD(Phi, mu), "
        "either optionally multiplied by local spectator fields, "
        "more general local monomials with one or more CovD(...) factors that can be "
        "expanded using model gauge metadata, "
        "nested operator monomials such as DC(DC(field, nu), mu), "
        "PartialD(DC(...), mu), PartialD(FieldStrength(...), mu), and "
        "DC(FieldStrength(...), mu), "
        "arbitrary products of FieldStrength(G, mu, nu[, a]) factors (e.g. "
        "-1/4 * FieldStrength(G, mu, nu, a) * FieldStrength(G, mu, nu, a), or higher "
        "F^n operators contracted with StructureConstant(...)), "
        "local monomials built from fields, PartialD(...), and one optional Gamma(...), "
        "pure local field monomials like lam * Phi * Phi * Phi * Phi, "
        "plus GaugeFixing(...) / GhostLagrangian(...) or the legacy "
        "GaugeFixingTerm / GhostTerm declarations."
    )


def _validate_declared_monomial(
    term: _DeclaredMonomial,
    *,
    parameters: Sequence[Parameter] = (),
):
    _validate_declared_label_bindings(term, parameters=parameters)
    if _match_covariant_monomial(term) is not None:
        return
    if _is_generic_covariant_monomial_candidate(term):
        return
    if _monomial_has_field_strength(term):
        _validate_field_strength_scalar(term)
        return
    if _lower_local_interaction_monomial(term, parameters=parameters) is not None:
        return
    raise ValueError(_unsupported_declared_source_term_error().args[0])


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
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
        ),
    ):
        return (item,)
    factor = _coerce_decl_factor(item)
    if factor is not None:
        return (_DeclaredMonomial.from_factor(factor),)
    return None


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


def _source_term_interaction(
    term,
    *,
    parameters: Sequence[Parameter] = (),
) -> Optional[InteractionTerm]:
    if isinstance(term, _DeclaredMonomial):
        return _lower_local_interaction_monomial(term, parameters=parameters)
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


def _declared_monomial_has_covd(term: _DeclaredMonomial) -> bool:
    return any(_declared_factor_has_covd(factor) for factor in term.factors)


def _declared_monomial_requires_recursive_operator_expansion(
    term: _DeclaredMonomial,
) -> bool:
    return any(
        _declared_factor_requires_recursive_operator_expansion(factor)
        for factor in term.factors
    )


def _is_generic_covariant_monomial_candidate(term: _DeclaredMonomial) -> bool:
    has_covd = _declared_monomial_has_covd(term)
    has_recursive_operator = _declared_monomial_requires_recursive_operator_expansion(term)
    if not has_covd and not has_recursive_operator:
        return False
    if has_recursive_operator:
        return True

    field_factors = [factor for factor in term.factors if isinstance(factor, _FieldFactor)]
    gamma_factors = [factor for factor in term.factors if isinstance(factor, GammaFactor)]
    covd_factors = [
        factor for factor in term.factors if isinstance(factor, CovariantDerivativeFactor)
    ]
    field_strength_factors = [
        factor for factor in term.factors if isinstance(factor, FieldStrengthFactor)
    ]
    differentiated_covd_factors = [
        factor for factor in term.factors if isinstance(factor, DifferentiatedCovariantFactor)
    ]

    if field_strength_factors:
        return True

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


def _validate_declared_source_term(term, *, parameters: Sequence[Parameter] = ()):
    if isinstance(
        term,
        (
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
        ),
    ):
        return
    if isinstance(term, _DeclaredMonomial):
        _validate_declared_monomial(term, parameters=parameters)
        return
    raise TypeError(f"Unsupported declared Lagrangian term type: {type(term)!r}")
