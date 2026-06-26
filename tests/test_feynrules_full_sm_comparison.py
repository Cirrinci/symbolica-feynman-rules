from pathlib import Path

from feynpy.comparison import (
    compare_feynrules_standard_model_vertices,
    load_feynrules_json,
)
from theories import build_standard_model


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    REPO_ROOT
    / "sandbox"
    / "wolframnotebook"
    / "sm_vertices_FeynRules.json"
)


def _field_map(fields):
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


def test_complete_standard_model_interaction_output_matches_sm_fr():
    sm = build_standard_model(
        include_ghosts=True,
        include_gauge_fixing=False,
    )
    report = compare_feynrules_standard_model_vertices(
        sm.lagrangian,
        load_feynrules_json(REFERENCE_PATH),
        field_map=_field_map(sm.fields),
        diagonal_yukawa_names={"yl": "ye"},
        feynpy_name_aliases=_name_aliases(sm.fields),
    )

    assert report.feynrules_only == ()
    assert report.feynpy_only == ()
    assert report.matched == 163
    assert report.all_match, [
        (row.reference.key, row.status, row.detail)
        for row in report.rows
        if not row.matches
    ]
