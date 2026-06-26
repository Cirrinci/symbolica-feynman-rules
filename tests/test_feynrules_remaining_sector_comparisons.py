from pathlib import Path

from feynrules.comparison import (
    compare_feynrules_bosonic_vertices,
    compare_feynrules_yukawa_vertices,
    load_feynrules_json,
)
from theories import build_standard_model
from theories.standard_model_feynrules import (
    standard_model_feynrules_field_map,
    standard_model_feynrules_name_aliases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "sandbox" / "wolframnotebook"


def _assert_report_matches(report):
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


def test_standard_model_yukawa_vertices_match_feynrules():
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    report = compare_feynrules_yukawa_vertices(
        sm.lagrangian,
        load_feynrules_json(
            REFERENCE_DIR / "yukawa_vertices_FeynRules.json"
        ),
        field_map=standard_model_feynrules_field_map(sm.fields),
        diagonal_yukawa_names={"yl": "ye"},
        feynpy_name_aliases=standard_model_feynrules_name_aliases(sm.fields),
    )
    _assert_report_matches(report)
    assert report.matched == 42


def test_standard_model_higgs_vertices_match_feynrules():
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    report = compare_feynrules_bosonic_vertices(
        sm.lagrangian,
        load_feynrules_json(
            REFERENCE_DIR / "higgs_vertices_FeynRules.json"
        ),
        field_map=standard_model_feynrules_field_map(sm.fields),
        feynpy_name_aliases=standard_model_feynrules_name_aliases(sm.fields),
        minimum_scalar_fields=1,
        scalar_relations=("cw**2 + sw**2 - 1",),
    )
    _assert_report_matches(report)
    assert report.matched == 38


def test_standard_model_ghost_vertices_match_feynrules():
    sm = build_standard_model(
        include_ghosts=True,
        include_gauge_fixing=True,
    )
    report = compare_feynrules_bosonic_vertices(
        sm.lagrangian,
        load_feynrules_json(
            REFERENCE_DIR / "ghost_vertices_FeynRules.json"
        ),
        field_map=standard_model_feynrules_field_map(sm.fields),
        feynpy_name_aliases=standard_model_feynrules_name_aliases(sm.fields),
        minimum_ghost_fields=2,
        use_momentum_conservation=True,
        scalar_relations=("cw**2 + sw**2 - 1",),
    )
    _assert_report_matches(report)
    assert report.matched == 24
