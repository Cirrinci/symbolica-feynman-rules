"""Read-only model and compiled-Lagrangian validation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .interactions import InteractionTerm, _field_match_key
from .lagrangian import (
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    GaugeFixingTerm,
    GhostTerm,
)
from .lowering import (
    _canonical_field_strength_kinetic_info,
    _source_term_covariant_core,
    _source_term_gauge_fixing,
    _source_term_ghost,
)
from .metadata import Field, GaugeGroup

if TYPE_CHECKING:
    from .core import Model
    from .lagrangian import CompiledLagrangian

ValidationSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """One validation diagnostic."""

    code: str
    message: str
    severity: ValidationSeverity = "error"


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation result."""

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


def _base_field_match_key(field_obj):
    return _field_match_key(field_obj, False)


def _is_canonical_mass_like_pair(first_occ, second_occ) -> bool:
    first_field = first_occ.field
    second_field = second_occ.field
    if first_field.kind != second_field.kind:
        return False

    if first_field.kind == "scalar":
        if first_field.self_conjugate or second_field.self_conjugate:
            return False
        return bool(first_occ.conjugated and not second_occ.conjugated)

    if first_field.kind == "fermion":
        return bool(first_occ.conjugated and not second_occ.conjugated)

    return False


def validate_model(model: Model) -> ValidationReport:
    """Return structured model diagnostics without changing compilation behavior."""
    issues: list[ValidationIssue] = []
    source_terms = model.lagrangian_decl.source_terms

    def add_issue(code: str, message: str, *, severity: ValidationSeverity = "error"):
        issue = ValidationIssue(code=code, message=message, severity=severity)
        if issue not in issues:
            issues.append(issue)

    def parameter_assumptions(value):
        parameter = model.find_parameter(value)
        if parameter is None:
            return None
        return parameter.assumptions()

    def coefficient_status(value):
        if value == 1:
            return True
        if isinstance(value, bool):
            return True if value else False
        if isinstance(value, (int, float)):
            return value == 1
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return value == 1
        if parameter_assumptions(value) is not None:
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
        items = tuple(target) if isinstance(target, (tuple, list)) else (target,)

        normalized = []
        for item in items:
            gauge_group = model.find_gauge_group(item)
            if gauge_group is not None:
                normalized.append(gauge_group.name)
            elif isinstance(item, GaugeGroup):
                normalized.append(item.name)
            else:
                normalized.append(repr(item))
        return tuple(normalized)

    def field_name(target) -> str:
        field = model.find_field(target)
        if field is not None:
            return field.name
        if isinstance(target, Field):
            return target.name
        return repr(target)

    kinetic_duplicates: dict[tuple[object, ...], int] = {}

    for term in source_terms:
        covariant = _source_term_covariant_core(term)
        if covariant is None:
            continue

        if isinstance(covariant, DiracKineticTerm):
            kind_label = "Dirac"
            duplicate_key = ("dirac", field_name(covariant.field), normalize_group_target(covariant.gauge_group))
        else:
            kind_label = "complex-scalar"
            duplicate_key = ("scalar", field_name(covariant.field), normalize_group_target(covariant.gauge_group))

        kinetic_duplicates[duplicate_key] = kinetic_duplicates.get(duplicate_key, 0) + 1

        status = coefficient_status(covariant.coefficient)
        if status is False:
            add_issue(
                "kinetic_normalization",
                f"{kind_label} kinetic term for field {field_name(covariant.field)!r} "
                f"has non-canonical coefficient {covariant.coefficient!r}; expected 1.",
            )

    for term in source_terms:
        kinetic_info = _canonical_field_strength_kinetic_info(term)
        if kinetic_info is None:
            continue
        group_target, normalized_coefficient = kinetic_info

        duplicate_key = ("vector", normalize_group_target(group_target))
        kinetic_duplicates[duplicate_key] = kinetic_duplicates.get(duplicate_key, 0) + 1

        status = coefficient_status(normalized_coefficient)
        if status is False:
            group = model.find_gauge_group(group_target)
            group_name = group.name if group is not None else repr(group_target)
            add_issue(
                "kinetic_normalization",
                f"Gauge kinetic term for gauge group {group_name!r} has "
                f"non-canonical coefficient; expected -1/4 * F(G, mu, nu, a)^2.",
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

    for term in source_terms:
        gauge_fixing = _source_term_gauge_fixing(term)
        if gauge_fixing is None:
            continue
        gauge_group = model.find_gauge_group(gauge_fixing.gauge_group)
        if gauge_group is None:
            add_issue(
                "undeclared_gauge_group",
                "Gauge-fixing validation could not resolve gauge group "
                f"{gauge_fixing.gauge_group!r} in model.gauge_groups.",
            )

    for term in source_terms:
        ghost = _source_term_ghost(term)
        if ghost is None:
            continue

        gauge_group = model.find_gauge_group(ghost.gauge_group)
        if gauge_group is None:
            add_issue(
                "undeclared_gauge_group",
                "Ghost validation could not resolve gauge group "
                f"{ghost.gauge_group!r} in model.gauge_groups.",
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

        ghost_field = model.find_field(gauge_group.ghost_field)
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

    for term in source_terms:
        covariant = _source_term_covariant_core(term)
        if covariant is None:
            continue

        field = model.find_field(covariant.field)
        if field is None:
            continue

        explicit_groups = normalize_explicit_groups(covariant.gauge_group)
        if explicit_groups is None:
            candidate_groups = tuple(group for group in model.gauge_groups if not group.abelian)
        else:
            candidate_groups = []
            for group_target in explicit_groups:
                gauge_group = model.find_gauge_group(group_target)
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


def validate_compiled_lagrangian(compiled: CompiledLagrangian) -> ValidationReport:
    """Return structured diagnostics inferred from compiled interaction terms."""
    issues: list[ValidationIssue] = []

    def add_issue(code: str, message: str, *, severity: ValidationSeverity = "error"):
        issue = ValidationIssue(code=code, message=message, severity=severity)
        if issue not in issues:
            issues.append(issue)

    for term in compiled.terms:
        if not isinstance(term, InteractionTerm):
            continue
        if len(term.fields) != 2:
            continue
        if term.derivatives:
            continue

        first, second = term.fields
        first_field = first.field
        second_field = second.field
        if first_field.kind not in ("scalar", "fermion"):
            continue
        if second_field.kind not in ("scalar", "fermion"):
            continue
        if not _is_canonical_mass_like_pair(first, second):
            continue
        if _base_field_match_key(first_field) == _base_field_match_key(second_field):
            continue

        kind_label = "fermion" if first_field.kind == "fermion" else "scalar"
        add_issue(
            "mass_structure_mixing",
            f"Off-diagonal {kind_label} mass-like bilinear detected between "
            f"fields {first_field.name!r} and {second_field.name!r}; "
            "compiled term has 0 derivatives and only 2 matter fields.",
            severity="warning",
        )

    return ValidationReport(issues=tuple(issues))
