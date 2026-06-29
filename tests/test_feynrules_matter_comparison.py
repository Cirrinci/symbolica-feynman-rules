from __future__ import annotations

from pathlib import Path

from feynrules.comparison import (
    compare_feynrules_matter_vertices,
    load_feynrules_json,
    parse_feynrules_matter_rule,
)
from theories import build_standard_model
from theories.standard_model_feynrules import (
    standard_model_feynrules_field_map,
    standard_model_feynrules_name_aliases,
)


REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "feynrules"
    / "sm"
    / "matter_vertices_FeynRules.json"
)


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
    report = compare_feynrules_matter_vertices(
        sm.lagrangian,
        references,
        field_map=standard_model_feynrules_field_map(sm.fields),
        feynpy_name_aliases=standard_model_feynrules_name_aliases(sm.fields),
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
    assert report.matched == 51
