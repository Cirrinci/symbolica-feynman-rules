from dataclasses import replace
from pathlib import Path
import re

import pytest

from feynrules.comparison import (
    compare_feynrules_bosonic_vertices,
    compare_feynrules_gauge_vertices,
    compare_feynrules_matter_vertices,
    load_feynrules_json,
)
from models.SM import build_standard_model
from models.SM.feynrules_comparison import (
    standard_model_feynrules_field_map,
    standard_model_feynrules_name_aliases,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "reference" / "feynrules"
)


@pytest.fixture(scope="module")
def comparison_context():
    sm = build_standard_model(
        include_ghosts=True,
        include_gauge_fixing=False,
    )
    return (
        sm,
        standard_model_feynrules_field_map(sm.fields),
        standard_model_feynrules_name_aliases(sm.fields),
    )


def _reference(sector: str, key: str):
    return next(
        reference
        for reference in load_feynrules_json(
            FIXTURE_DIR / f"{sector}_vertices_FeynRules.json"
        )
        if reference.key == key
    )


def _assert_single_mismatch(report):
    assert report.feynrules_only == ()
    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.status == "MISMATCH", row.detail
    assert row.difference is not None
    assert row.difference.to_canonical_string() != "0"


def test_matter_comparison_detects_chirality_flip(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("matter", "dbar|u|Wbar")
    mutated = replace(
        reference,
        rule=reference.rule.replace("ProjM", "ProjP", 1),
    )

    report = compare_feynrules_matter_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
    )

    _assert_single_mismatch(report)


def test_matter_comparison_detects_wrong_ckm_conjugation(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("matter", "b|tbar|W")
    ckm_component = "CKM[Index[Generation, 3], Index[Generation, 3]]"
    mutated = replace(
        reference,
        rule=reference.rule.replace(
            ckm_component,
            f"Conjugate[{ckm_component}]",
            1,
        ),
    )

    report = compare_feynrules_matter_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
    )

    _assert_single_mismatch(report)


def test_gauge_comparison_detects_removed_imaginary_unit(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("gauge", "A|A|W|Wbar")
    mutated = replace(
        reference,
        rule=re.sub(r"\bI\b", "1", reference.rule),
    )

    report = compare_feynrules_gauge_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
    )

    _assert_single_mismatch(report)


def test_matter_comparison_detects_reversed_color_generator(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("matter", "G|u|ubar")
    generator = (
        "T[Index[Gluon, Ext[3]], Index[Colour, Ext[1]], "
        "Index[Colour, Ext[2]]]"
    )
    reversed_generator = (
        "T[Index[Gluon, Ext[3]], Index[Colour, Ext[2]], "
        "Index[Colour, Ext[1]]]"
    )
    mutated = replace(
        reference,
        rule=reference.rule.replace(generator, reversed_generator, 1),
    )

    report = compare_feynrules_matter_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
    )

    _assert_single_mismatch(report)


def test_ghost_comparison_detects_wrong_derivative_momentum(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("ghost", "G|ghG|ghGbar")
    mutated = replace(
        reference,
        rule=reference.rule.replace("FV[2,", "FV[1,", 1),
    )

    report = compare_feynrules_bosonic_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
        minimum_ghost_fields=2,
        use_momentum_conservation=True,
        scalar_relations=("cw**2 + sw**2 - 1",),
    )

    _assert_single_mismatch(report)


def test_gauge_comparison_detects_complete_sign_flip(comparison_context):
    sm, field_map, aliases = comparison_context
    reference = _reference("gauge", "A|W|Wbar")
    mutated = replace(reference, rule=f"-({reference.rule})")

    report = compare_feynrules_gauge_vertices(
        sm.lagrangian,
        (mutated,),
        field_map=field_map,
        feynpy_name_aliases=aliases,
    )

    _assert_single_mismatch(report)
