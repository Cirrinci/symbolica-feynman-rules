import json
from pathlib import Path

from feynpy import Model
from models.SMEFT2 import OMITTED_SECTORS, build_smeft_green_bpreserving


MODEL_DIR = Path(__file__).resolve().parents[1]


def test_smeft2_supported_subset_builds_and_compiles():
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    signatures = {signature.names for signature in lagrangian.vertex_signatures()}

    assert len(lagrangian.terms) == 2541
    assert ("QL.bar", "QL", "B") in signatures
    assert ("Phi.bar", "QL.bar", "UR", "G") in signatures
    assert ("LL.bar", "LR", "DR.bar", "QL") in signatures
    assert ("LL.bar", "LL", "Phi", "Phi") in signatures
    assert ("QL.bar", "LR", "DR.bar", "LL") in signatures
    assert ("QL.bar", "UR", "Phi.bar", "B", "B") in signatures
    assert ("QL.bar", "QL", "G") in signatures
    assert ("QL.bar", "QL", "G", "G") in signatures


def test_smeft2_ltot_is_eft_only_and_lfull_keeps_sm_core():
    bundle = build_smeft_green_bpreserving()
    full_model = Model(
        name="SMEFT_Green_Bpreserving_full",
        gauge_groups=tuple(bundle.gauge_groups.values()),
        fields=tuple(bundle.fields.values()),
        parameters=tuple(bundle.parameters.values()),
        lagrangian_decl=bundle.lagrangians["Lfull"],
    )

    assert bundle.model.lagrangian_decl is bundle.lagrangians["Ltot"]
    assert len(bundle.model.lagrangian().terms) == 2541
    assert len(full_model.lagrangian().terms) == 2595


def test_smeft2_has_no_omitted_sectors():
    assert "LWeinberg" not in OMITTED_SECTORS
    assert "LH4D2[alphaRHDpp]" not in OMITTED_SECTORS
    assert "LEvF2XH" not in OMITTED_SECTORS
    assert "LEv4q" not in OMITTED_SECTORS
    assert "LEvF2HD2" not in OMITTED_SECTORS
    assert "LEvCCRRLL" not in OMITTED_SECTORS
    assert OMITTED_SECTORS == ()


def test_smeft2_omitted_sectors_are_named_empty_lagrangians():
    bundle = build_smeft_green_bpreserving()
    for sector in OMITTED_SECTORS:
        model = Model(
            name=sector,
            gauge_groups=tuple(bundle.gauge_groups.values()),
            fields=tuple(bundle.fields.values()),
            parameters=tuple(bundle.parameters.values()),
            lagrangian_decl=bundle.lagrangians[sector],
        )
        assert len(model.lagrangian().terms) == 0


def test_smeft2_comparison_report_uses_eft_only_basis():
    report = json.loads(
        (MODEL_DIR / "vertex_comparison_report.json").read_text(encoding="utf-8")
    )

    assert report["summary"]["comparison_basis"]["reference_ltot"] == "EFT-only FeynRules Ltot"
    assert report["summary"]["comparison_basis"]["local_ltot"] == "EFT-only FeynPy Ltot"
    assert report["summary"]["reference_vertex_count"] == 184
    assert report["summary"]["feynpy_signature_count_3_to_6"] == 192
    assert report["summary"]["shared_head_matches"] == 168
    assert report["summary"]["shared_head_count_matches"] == 83
    assert report["summary"]["shared_head_count_mismatches"] == 99
    assert report["summary"]["shared_head_count_benign_expansions"] == 9
    assert report["summary"]["shared_head_count_unexplained_mismatches"] == 90
    assert report["summary"]["benign_head_count_delta_heads"] == 15
    assert report["summary"]["unexplained_head_count_delta_heads"] == 331
    assert all(
        "head_count_status" in row
        and "reference_head_counts" in row
        and "feynpy_head_counts" in row
        and "head_count_delta" in row
        and "benign_head_count_delta_reasons" in row
        and "unexplained_head_count_delta" in row
        for row in report["reference_vertices"]
    )

    rows_by_key = {row["key"]: row for row in report["reference_vertices"]}
    assert rows_by_key["B|Phi|qL|uRbar"]["benign_head_count_delta_reasons"] == {
        "alphaEuB": "DUAL_FS_ANTISYMMETRY"
    }
    assert rows_by_key["G|qL|qLbar"]["benign_head_count_delta_reasons"] == {
        "alphaRqD": "DUMMY_LORENTZ_MERGE",
        "g3": "DUMMY_LORENTZ_MERGE",
    }
    assert rows_by_key["Wi|qL|qLbar"]["benign_head_count_delta_reasons"] == {
        "alphaRqD": "DUMMY_LORENTZ_MERGE",
        "g2": "DUMMY_LORENTZ_MERGE",
    }
    assert rows_by_key["B|qL|qLbar"]["head_count_status"] == "COUNT_BENIGN_EXPANSION"
