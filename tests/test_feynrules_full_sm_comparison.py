from pathlib import Path

from feynrules.comparison import load_feynrules_json
from theories import build_standard_model
from theories.standard_model_feynrules import (
    compare_feynrules_standard_model_vertices,
    standard_model_feynrules_field_map,
    standard_model_feynrules_name_aliases,
)


REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "feynrules"
    / "sm"
    / "sm_vertices_FeynRules.json"
)


def test_complete_standard_model_interaction_output_matches_sm_fr():
    sm = build_standard_model(
        include_ghosts=True,
        include_gauge_fixing=False,
    )
    report = compare_feynrules_standard_model_vertices(
        sm.lagrangian,
        load_feynrules_json(REFERENCE_PATH),
        field_map=standard_model_feynrules_field_map(sm.fields),
        feynpy_name_aliases=standard_model_feynrules_name_aliases(sm.fields),
    )

    assert report.feynrules_only == ()
    assert report.feynpy_only == ()
    assert report.matched == 163
    assert report.all_match, [
        (row.reference.key, row.status, row.detail)
        for row in report.rows
        if not row.matches
    ]
