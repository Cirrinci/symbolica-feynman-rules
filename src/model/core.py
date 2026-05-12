"""Top-level model container."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional
import warnings

from .declared import (
    _DeclaredMonomial,
    _FieldFactor,
    CovariantDerivativeFactor,
    DifferentiatedCovariantFactor,
    FieldStrengthFactor,
    GaugeFixingDeclaration,
    GhostLagrangianDeclaration,
    PartialDerivativeFactor,
)
from .interactions import InteractionTerm
from .lagrangian import (
    CompiledLagrangian,
    ComplexScalarKineticTerm,
    CovariantTerm,
    DeclaredLagrangian,
    DiracKineticTerm,
    GaugeFixingTerm,
    GaugeKineticTerm,
    GhostTerm,
)
from .lowering import (
    _normalize_interaction_terms_input,
    _analyze_declared_source_term,
    _unsupported_declared_source_term_error,
)
from .metadata import Field, GaugeGroup, Parameter, ParameterAssumptions


ValidationSeverity = Literal["error", "warning"]


def _append_unique_identity(items: list[object], value):
    if value is None:
        return
    if any(existing is value for existing in items):
        return
    items.append(value)


def _collect_declared_term_metadata(term, *, fields: list[Field], gauge_groups: list[GaugeGroup]):
    if isinstance(term, InteractionTerm):
        for occurrence in term.fields:
            _append_unique_identity(fields, occurrence.field)
        return

    if isinstance(term, _DeclaredMonomial):
        for factor in term.factors:
            if isinstance(factor, (_FieldFactor, CovariantDerivativeFactor, PartialDerivativeFactor)):
                _append_unique_identity(fields, factor.field)
                continue
            if isinstance(factor, DifferentiatedCovariantFactor):
                _append_unique_identity(fields, factor.covariant_factor.field)
                continue
            if isinstance(factor, FieldStrengthFactor) and isinstance(factor.gauge_group, GaugeGroup):
                _append_unique_identity(gauge_groups, factor.gauge_group)
        return

    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm)):
        if isinstance(term.field, Field):
            _append_unique_identity(fields, term.field)
        if isinstance(term.gauge_group, GaugeGroup):
            _append_unique_identity(gauge_groups, term.gauge_group)
        return

    if isinstance(
        term,
        (
            GaugeKineticTerm,
            GaugeFixingTerm,
            GhostTerm,
            GaugeFixingDeclaration,
            GhostLagrangianDeclaration,
        ),
    ):
        if isinstance(term.gauge_group, GaugeGroup):
            _append_unique_identity(gauge_groups, term.gauge_group)


def _infer_model_metadata(
    *,
    explicit_fields: tuple[Field, ...],
    explicit_gauge_groups: tuple[GaugeGroup, ...],
    source_terms: tuple[object, ...],
) -> tuple[tuple[Field, ...], tuple[GaugeGroup, ...]]:
    fields: list[Field] = list(explicit_fields)
    gauge_groups: list[GaugeGroup] = list(explicit_gauge_groups)

    for term in source_terms:
        _collect_declared_term_metadata(term, fields=fields, gauge_groups=gauge_groups)

    for gauge_group in tuple(gauge_groups):
        if isinstance(gauge_group.gauge_boson, Field):
            _append_unique_identity(fields, gauge_group.gauge_boson)
        if isinstance(gauge_group.ghost_field, Field):
            _append_unique_identity(fields, gauge_group.ghost_field)

    for field in tuple(fields):
        for member in getattr(field, "class_members", ()):
            _append_unique_identity(fields, member)

    return tuple(fields), tuple(gauge_groups)


@dataclass(frozen=True)
class ValidationIssue:
    """One model-validation diagnostic."""

    code: str
    message: str
    severity: ValidationSeverity = "error"


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation result returned by ``Model.validate()``."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors
# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Top-level model container (mirrors the full .fr file).

    Use ``Model`` when declarations depend on model metadata such as declared
    fields, gauge groups, charges, representations, or ghost assignments.
    The recommended source entry point is ``lagrangian_decl=...`` with
    declarative factors such as ``CovD(...)``, ``FieldStrength(...)``,
    ``GaugeFixing(...)``, and ``GhostLagrangian(...)``. Call
    ``model.lagrangian()`` to compile those source declarations into a
    ``CompiledLagrangian`` before vertex extraction.

    For metadata-free local operators that are already expanded, prefer
    ``Lagrangian(...)`` directly instead of wrapping them in a ``Model``.

    The actual vertex evaluation still happens in ``symbolic/vertex_engine.py``
    after declarations are translated into ``InteractionTerm`` objects and then
    into engine kwargs.
    """
    name: str = ""
    gauge_groups: tuple[GaugeGroup, ...] = ()
    fields: tuple[Field, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    interactions: tuple[InteractionTerm, ...] = ()
    lagrangian_decl: Optional[DeclaredLagrangian] = None
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

        inferred_fields, inferred_gauge_groups = _infer_model_metadata(
            explicit_fields=self.fields,
            explicit_gauge_groups=self.gauge_groups,
            source_terms=self.source_lagrangian_terms(),
        )
        self.fields = inferred_fields
        self.gauge_groups = inferred_gauge_groups

        for term in self.lagrangian_decl.source_terms:
            if _analyze_declared_source_term(term) is None:
                raise _unsupported_declared_source_term_error()

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
            analyzed.interaction
            for analyzed in self.analyzed_source_terms()
            if analyzed.interaction is not None
        )

    def all_covariant_terms(self) -> tuple[CovariantTerm, ...]:
        return self.covariant_terms + tuple(
            analyzed.covariant_core
            for analyzed in self.analyzed_source_terms()
            if analyzed.covariant_core is not None
        )

    def all_gauge_kinetic_terms(self) -> tuple[GaugeKineticTerm, ...]:
        return self.gauge_kinetic_terms + tuple(
            analyzed.gauge_kinetic
            for analyzed in self.analyzed_source_terms()
            if analyzed.gauge_kinetic is not None
        )

    def all_gauge_fixing_terms(self) -> tuple[GaugeFixingTerm, ...]:
        return self.gauge_fixing_terms + tuple(
            analyzed.gauge_fixing
            for analyzed in self.analyzed_source_terms()
            if analyzed.gauge_fixing is not None
        )

    def all_ghost_terms(self) -> tuple[GhostTerm, ...]:
        return self.ghost_terms + tuple(
            analyzed.ghost
            for analyzed in self.analyzed_source_terms()
            if analyzed.ghost is not None
        )

    def analyzed_source_terms(self) -> tuple[object, ...]:
        """Return the normalized interpretation of each declarative source term."""
        analyzed_terms = []
        for term in self.lagrangian_decl.source_terms:
            analyzed = _analyze_declared_source_term(term)
            if analyzed is None:
                raise _unsupported_declared_source_term_error()
            analyzed_terms.append(analyzed)
        return tuple(analyzed_terms)

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

    def find_parameter(self, target) -> Optional[Parameter]:
        """Resolve a parameter by object identity, declaration name, or symbol."""
        if isinstance(target, Parameter):
            for parameter in self.parameters:
                if parameter is target:
                    return parameter
            return None
        if target is None:
            return None

        if hasattr(target, "to_canonical_string"):
            target_text = target.to_canonical_string()
        else:
            target_text = str(target)

        for parameter in self.parameters:
            if parameter.name == target_text:
                return parameter
            symbol = parameter.symbol
            if hasattr(symbol, "to_canonical_string"):
                symbol_text = symbol.to_canonical_string()
            else:
                symbol_text = str(symbol)
            if symbol_text == target_text:
                return parameter
        return None

    def parameter_assumptions(self, target) -> Optional[ParameterAssumptions]:
        """Return structured assumptions for one declared parameter."""

        parameter = self.find_parameter(target)
        if parameter is None:
            return None
        return parameter.assumptions()

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

    def validate(self) -> ValidationReport:
        """Return structured model diagnostics without changing compilation behavior.

        The initial validation pass focuses on gauge-sector consistency checks
        that can be established directly from model metadata and source
        declarations before any vertex extraction is attempted.
        """
        issues: list[ValidationIssue] = []

        def add_issue(code: str, message: str, *, severity: ValidationSeverity = "error"):
            issue = ValidationIssue(code=code, message=message, severity=severity)
            if issue not in issues:
                issues.append(issue)

        def coefficient_status(value):
            if value == 1:
                return True
            if isinstance(value, bool):
                return True if value else False
            if isinstance(value, (int, float)):
                return value == 1
            if hasattr(value, "numerator") and hasattr(value, "denominator"):
                return value == 1
            assumptions = self.parameter_assumptions(value)
            if assumptions is not None:
                return None
            if hasattr(value, "to_canonical_string"):
                text = value.to_canonical_string()
                if text == "1":
                    return True
                if any(ch.isalpha() for ch in text):
                    return None
                return False
            return None

        def normalize_group_target(target):
            if target is None:
                return ("__auto__",)
            if isinstance(target, (tuple, list)):
                items = tuple(target)
            else:
                items = (target,)

            normalized = []
            for item in items:
                gauge_group = self.find_gauge_group(item)
                if gauge_group is not None:
                    normalized.append(gauge_group.name)
                elif isinstance(item, GaugeGroup):
                    normalized.append(item.name)
                else:
                    normalized.append(repr(item))
            return tuple(normalized)

        def field_name(target) -> str:
            field = self.find_field(target)
            if field is not None:
                return field.name
            if isinstance(target, Field):
                return target.name
            return repr(target)

        kinetic_duplicates: dict[tuple[object, ...], int] = {}

        for term in self.all_covariant_terms():
            if isinstance(term, DiracKineticTerm):
                kind_label = "Dirac"
                duplicate_key = ("dirac", field_name(term.field), normalize_group_target(term.gauge_group))
            else:
                kind_label = "complex-scalar"
                duplicate_key = ("scalar", field_name(term.field), normalize_group_target(term.gauge_group))

            kinetic_duplicates[duplicate_key] = kinetic_duplicates.get(duplicate_key, 0) + 1

            status = coefficient_status(term.coefficient)
            if status is False:
                add_issue(
                    "kinetic_normalization",
                    f"{kind_label} kinetic term for field {field_name(term.field)!r} "
                    f"has non-canonical coefficient {term.coefficient!r}; expected 1.",
                )

        for term in self.all_gauge_kinetic_terms():
            duplicate_key = ("vector", normalize_group_target(term.gauge_group))
            kinetic_duplicates[duplicate_key] = kinetic_duplicates.get(duplicate_key, 0) + 1

            status = coefficient_status(term.coefficient)
            if status is False:
                group = self.find_gauge_group(term.gauge_group)
                group_name = group.name if group is not None else repr(term.gauge_group)
                add_issue(
                    "kinetic_normalization",
                    f"Gauge kinetic term for gauge group {group_name!r} has "
                    f"non-canonical coefficient {term.coefficient!r}; expected 1.",
                )

        for key, count in kinetic_duplicates.items():
            if count <= 1:
                continue

            if key[0] == "dirac":
                _, field_label, groups = key
                add_issue(
                    "duplicate_kinetic_term",
                    f"Duplicate Dirac kinetic declarations found for field {field_label!r} "
                    f"with gauge-group selection {groups}.",
                )
            elif key[0] == "scalar":
                _, field_label, groups = key
                add_issue(
                    "duplicate_kinetic_term",
                    f"Duplicate complex-scalar kinetic declarations found for field {field_label!r} "
                    f"with gauge-group selection {groups}.",
                )
            else:
                _, groups = key
                add_issue(
                    "duplicate_kinetic_term",
                    f"Duplicate gauge kinetic declarations found for gauge-group selection {groups}.",
                )

        for term in self.all_gauge_fixing_terms():
            gauge_group = self.find_gauge_group(term.gauge_group)
            if gauge_group is None:
                add_issue(
                    "undeclared_gauge_group",
                    "Gauge-fixing validation could not resolve gauge group "
                    f"{term.gauge_group!r} in model.gauge_groups.",
                )

        for term in self.all_ghost_terms():
            gauge_group = self.find_gauge_group(term.gauge_group)
            if gauge_group is None:
                add_issue(
                    "undeclared_gauge_group",
                    "Ghost validation could not resolve gauge group "
                    f"{term.gauge_group!r} in model.gauge_groups.",
                )
                continue

            if gauge_group.abelian:
                add_issue(
                    "abelian_ghost_sector",
                    "Ghost validation only supports non-abelian gauge groups; "
                    f"got {gauge_group.name!r}.",
                )
                continue

            if gauge_group.structure_constant is None or not callable(gauge_group.structure_constant):
                add_issue(
                    "missing_structure_constant",
                    "Ghost validation requires non-abelian gauge group "
                    f"{gauge_group.name!r} to declare a callable structure_constant.",
                )

            if gauge_group.ghost_field is None:
                add_issue(
                    "missing_ghost_field",
                    "Ghost validation requires gauge group "
                    f"{gauge_group.name!r} to declare ghost_field.",
                )
                continue

            ghost_field = self.find_field(gauge_group.ghost_field)
            if ghost_field is None:
                add_issue(
                    "missing_ghost_field",
                    "Ghost validation could not resolve ghost_field "
                    f"{gauge_group.ghost_field!r} for gauge group {gauge_group.name!r} "
                    "in model.fields.",
                )

        def normalize_explicit_groups(group_target):
            if group_target is None:
                return None
            if isinstance(group_target, (tuple, list)):
                return tuple(group_target)
            return (group_target,)

        def field_index_names(field: Field) -> str:
            if not field.indices:
                return "(none)"
            return ", ".join(index.name for index in field.indices)

        def representation_index_names(gauge_group: GaugeGroup) -> str:
            if not gauge_group.representations:
                return "(none)"
            return ", ".join(rep.index.name for rep in gauge_group.representations)

        for term in self.all_covariant_terms():
            field = self.find_field(term.field)
            if field is None:
                continue

            explicit_groups = normalize_explicit_groups(term.gauge_group)
            if explicit_groups is None:
                candidate_groups = tuple(group for group in self.gauge_groups if not group.abelian)
            else:
                candidate_groups = []
                for group_target in explicit_groups:
                    gauge_group = self.find_gauge_group(group_target)
                    if gauge_group is None:
                        add_issue(
                            "undeclared_gauge_group",
                            "Covariant validation could not resolve gauge group "
                            f"{group_target!r} in model.gauge_groups.",
                        )
                        continue
                    candidate_groups.append(gauge_group)
                candidate_groups = tuple(candidate_groups)

            for gauge_group in candidate_groups:
                if gauge_group.abelian:
                    continue

                try:
                    rep_info = gauge_group.matter_representation_and_slots(field)
                except ValueError as exc:
                    if explicit_groups is None and not any(
                        rep.index == field_index
                        for rep in gauge_group.representations
                        for field_index in field.indices
                    ):
                        continue
                    add_issue(
                        "gauge_representation_resolution",
                        "Covariant validation could not resolve the representation metadata "
                        f"for field {field.name!r} under gauge group {gauge_group.name!r}: {exc}",
                    )
                    continue

                if rep_info is None and explicit_groups is not None:
                    add_issue(
                        "missing_gauge_representation",
                        "Covariant validation requires field "
                        f"{field.name!r} to carry a declared representation under "
                        f"gauge group {gauge_group.name!r}. Field indices: "
                        f"{field_index_names(field)}. Declared representation indices: "
                        f"{representation_index_names(gauge_group)}.",
                    )

        return ValidationReport(issues=tuple(issues))

    def lagrangian(self) -> CompiledLagrangian:
        """Compile all declared terms and return a ``CompiledLagrangian``."""
        from compiler.gauge import compile_covariant_terms

        return CompiledLagrangian(
            terms=self.all_interactions() + compile_covariant_terms(self),
            parameters=self.parameters,
        )
