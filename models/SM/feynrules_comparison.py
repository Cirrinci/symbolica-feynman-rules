"""FeynRules comparison helpers for the packaged Standard Model."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from feynrules.comparison import (
    FeynRulesVertex,
    VertexComparisonReport,
    compare_feynrules_bosonic_vertices,
    compare_feynrules_gauge_vertices,
    compare_feynrules_matter_vertices,
    compare_feynrules_yukawa_vertices,
)


def standard_model_feynrules_field_map(fields) -> dict[str, object]:
    """Return the SM.fr external field-name map for physical SM fields."""

    result = {
        "A": fields.A,
        "Z": fields.Z,
        "W": fields.W,
        "Wbar": fields.W.bar,
        "G": fields.G,
        "H": fields.H,
        "G0": fields.G0,
        "GP": fields.GP,
        "GPbar": fields.GP.bar,
        "ghA": fields.ghA,
        "ghAbar": fields.ghA.bar,
        "ghZ": fields.ghZ,
        "ghZbar": fields.ghZ.bar,
        "ghWp": fields.ghWp,
        "ghWpbar": fields.ghWp.bar,
        "ghWm": fields.ghWm,
        "ghWmbar": fields.ghWm.bar,
        "ghG": fields.ghG,
        "ghGbar": fields.ghG.bar,
    }
    for field_class in (fields.vl, fields.l, fields.uq, fields.dq):
        for member in field_class.class_members:
            result[member.name] = member
            result[f"{member.name}bar"] = member.bar
    return result


def standard_model_feynrules_name_aliases(fields) -> dict[str, str]:
    """Return FeynPy-to-SM.fr external field-name aliases."""

    aliases = {
        "W.bar": "Wbar",
        "GP.bar": "GPbar",
        "ghA.bar": "ghAbar",
        "ghZ.bar": "ghZbar",
        "ghWp.bar": "ghWpbar",
        "ghWm.bar": "ghWmbar",
        "ghG.bar": "ghGbar",
    }
    aliases.update(
        {
            f"{member.name}.bar": f"{member.name}bar"
            for field_class in (fields.vl, fields.l, fields.uq, fields.dq)
            for member in field_class.class_members
        }
    )
    return aliases


def _field_spin(field) -> Fraction:
    return Fraction(field.field.spin if hasattr(field, "field") else field.spin)


def _field_kind(field) -> str:
    return field.field.kind if hasattr(field, "field") else field.kind


def compare_feynrules_standard_model_vertices(
    lagrangian,
    references: Sequence[FeynRulesVertex],
    *,
    field_map: Mapping[str, object],
    parameter_substitutions: Mapping[str, object] | None = None,
    diagonal_yukawa_names: Mapping[str, str] | None = None,
    feynpy_substitutions: Mapping[object, object] | None = None,
    feynpy_name_aliases: Mapping[str, str] | None = None,
) -> VertexComparisonReport:
    """Compare the complete flavor-expanded SM interaction vertex set.

    The official ``SM.fr`` output is partitioned by field content and routed
    through the tensor adapter appropriate to each Standard Model sector.
    """

    sectors: dict[str, list[FeynRulesVertex]] = {
        "gauge": [],
        "matter": [],
        "yukawa": [],
        "higgs": [],
        "ghost": [],
    }
    for reference in references:
        mapped_fields = tuple(field_map[name] for name in reference.fields)
        ghost_count = sum(
            _field_kind(field) == "ghost"
            for field in mapped_fields
        )
        fermion_count = sum(
            _field_spin(field) == Fraction(1, 2)
            for field in mapped_fields
        )
        scalar_count = sum(
            _field_spin(field) == 0
            and _field_kind(field) != "ghost"
            for field in mapped_fields
        )
        if ghost_count >= 2:
            sectors["ghost"].append(reference)
        elif fermion_count == 2:
            sectors[
                "yukawa" if scalar_count else "matter"
            ].append(reference)
        elif scalar_count:
            sectors["higgs"].append(reference)
        else:
            sectors["gauge"].append(reference)

    if diagonal_yukawa_names is None:
        diagonal_yukawa_names = {"yl": "ye"}

    reports = (
        compare_feynrules_gauge_vertices(
            lagrangian,
            sectors["gauge"],
            field_map=field_map,
            parameter_substitutions=parameter_substitutions,
            feynpy_name_aliases=feynpy_name_aliases,
        ),
        compare_feynrules_matter_vertices(
            lagrangian,
            sectors["matter"],
            field_map=field_map,
            parameter_substitutions=parameter_substitutions,
            feynpy_substitutions=feynpy_substitutions,
            feynpy_name_aliases=feynpy_name_aliases,
        ),
        compare_feynrules_yukawa_vertices(
            lagrangian,
            sectors["yukawa"],
            field_map=field_map,
            diagonal_yukawa_names=diagonal_yukawa_names,
            feynpy_substitutions=feynpy_substitutions,
            feynpy_name_aliases=feynpy_name_aliases,
        ),
        compare_feynrules_bosonic_vertices(
            lagrangian,
            sectors["higgs"],
            field_map=field_map,
            parameter_substitutions=parameter_substitutions,
            feynpy_substitutions=feynpy_substitutions,
            feynpy_name_aliases=feynpy_name_aliases,
            minimum_scalar_fields=1,
            scalar_relations=("cw**2 + sw**2 - 1",),
        ),
        compare_feynrules_bosonic_vertices(
            lagrangian,
            sectors["ghost"],
            field_map=field_map,
            parameter_substitutions=parameter_substitutions,
            feynpy_substitutions=feynpy_substitutions,
            feynpy_name_aliases=feynpy_name_aliases,
            minimum_ghost_fields=2,
            use_momentum_conservation=True,
            scalar_relations=("cw**2 + sw**2 - 1",),
        ),
    )
    row_by_key = {
        row.reference.key: row
        for report in reports
        for row in report.rows
    }
    return VertexComparisonReport(
        rows=tuple(row_by_key[reference.key] for reference in references),
        feynrules_only=tuple(
            sorted({
                signature
                for report in reports
                for signature in report.feynrules_only
            })
        ),
        feynpy_only=tuple(
            sorted({
                signature
                for report in reports
                for signature in report.feynpy_only
            })
        ),
    )


__all__ = (
    "compare_feynrules_standard_model_vertices",
    "standard_model_feynrules_field_map",
    "standard_model_feynrules_name_aliases",
)
