from pathlib import Path

from feynpy.comparison import (
    compare_feynrules_bosonic_vertices,
    compare_feynrules_yukawa_vertices,
    load_feynrules_json,
)
from theories import build_standard_model


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "sandbox" / "wolframnotebook"


def _physical_field_map(fields):
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


def _name_aliases(fields):
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
        field_map=_physical_field_map(sm.fields),
        diagonal_yukawa_names={"yl": "ye"},
        feynpy_name_aliases=_name_aliases(sm.fields),
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
        field_map=_physical_field_map(sm.fields),
        feynpy_name_aliases=_name_aliases(sm.fields),
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
        field_map=_physical_field_map(sm.fields),
        feynpy_name_aliases=_name_aliases(sm.fields),
        minimum_ghost_fields=2,
        use_momentum_conservation=True,
        scalar_relations=("cw**2 + sw**2 - 1",),
    )
    _assert_report_matches(report)
    assert report.matched == 24
