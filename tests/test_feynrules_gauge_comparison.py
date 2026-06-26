from __future__ import annotations

from pathlib import Path

from feynpy.comparison import (
    compare_feynrules_gauge_vertices,
    load_feynrules_json,
    parse_feynrules_gauge_rule,
)
from theories import build_standard_model


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    REPO_ROOT
    / "sandbox"
    / "wolframnotebook"
    / "gauge_vertices_FeynRules.json"
)


def test_feynrules_gauge_parser_rejects_unsupported_tensor_syntax():
    try:
        parse_feynrules_gauge_rule("Unsupported[Index[Lorentz, Ext[1]]]")
    except ValueError as error:
        assert "Unsupported FeynRules gauge syntax" in str(error)
    else:
        raise AssertionError("Unsupported FeynRules syntax was silently accepted")


def test_standard_model_gauge_vertices_match_feynrules_json_exactly():
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    references = load_feynrules_json(REFERENCE_PATH)
    fields = sm.fields
    report = compare_feynrules_gauge_vertices(
        sm.lagrangian,
        references,
        field_map={
            "A": fields.A,
            "W": fields.W,
            "Wbar": fields.W.bar,
            "Z": fields.Z,
            "G": fields.G,
        },
        feynpy_name_aliases={"W.bar": "Wbar"},
    )

    assert report.feynrules_only == ()
    assert report.feynpy_only == ()
    assert report.all_match, [
        (row.reference.key, row.status, row.detail)
        for row in report.rows
        if not row.matches
    ]
