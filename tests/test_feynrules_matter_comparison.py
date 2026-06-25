from __future__ import annotations

from pathlib import Path

from symbolica import Expression, S

from feynpy.comparison import (
    compare_feynrules_matter_vertices,
    load_feynrules_json,
    parse_feynrules_matter_rule,
)
from theories import build_standard_model


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    REPO_ROOT
    / "sandbox"
    / "wolframnotebook"
    / "matter_vertices_FeynRules.json"
)


def _matter_field_map(fields):
    result = {
        "A": fields.A,
        "Z": fields.Z,
        "W": fields.W,
        "Wbar": fields.W.bar,
        "G": fields.G,
    }
    for field_class in (fields.vl, fields.l, fields.uq, fields.dq):
        for member in field_class.class_members:
            result[member.name] = member
            result[f"{member.name}bar"] = member.bar
    return result


def test_feynrules_matter_parser_rejects_unsupported_tensor_syntax():
    try:
        parse_feynrules_matter_rule("Unsupported[Index[Spin, Ext[1]]]")
    except ValueError as error:
        assert "Unsupported FeynRules matter syntax" in str(error)
    else:
        raise AssertionError("Unsupported FeynRules syntax was silently accepted")


def test_standard_model_matter_vertices_match_feynrules_json_exactly():
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    references = load_feynrules_json(REFERENCE_PATH)
    fields = sm.fields
    parameters = sm.parameters
    half = Expression.num(1) / Expression.num(2)
    gauge_norm = (
        parameters.g1.symbol**2 + parameters.g2.symbol**2
    ) ** half
    cabi = S("cabi")

    report = compare_feynrules_matter_vertices(
        sm.lagrangian,
        references,
        field_map=_matter_field_map(fields),
        parameter_substitutions={
            "ee": parameters.g1.symbol * parameters.g2.symbol / gauge_norm,
            "cw": parameters.g2.symbol / gauge_norm,
            "sw": parameters.g1.symbol / gauge_norm,
            "gs": parameters.g3.symbol,
        },
        feynpy_substitutions={
            S("cos")(cabi): Expression.num(1),
            S("sin")(cabi): Expression.num(0),
        },
        feynpy_name_aliases={
            "W.bar": "Wbar",
            **{
                f"{member.name}.bar": f"{member.name}bar"
                for field_class in (fields.vl, fields.l, fields.uq, fields.dq)
                for member in field_class.class_members
            },
        },
    )

    assert report.feynrules_only == ()
    assert report.feynpy_only == ()
    assert report.all_match, [
        (
            row.reference.key,
            row.status,
            row.detail,
            (
                row.difference.to_canonical_string()
                if row.difference is not None
                else ""
            ),
        )
        for row in report.rows
        if not row.matches
    ]
