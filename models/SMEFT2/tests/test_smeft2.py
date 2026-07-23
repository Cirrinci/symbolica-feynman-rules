import json
from pathlib import Path

from feynrules.comparison import compare_canonical_coefficient_maps
from feynpy import Model
from models.SMEFT2 import OMITTED_SECTORS, build_smeft_green_bpreserving
from symbolic.tensor_canonicalization import canonical_external_index_set
from symbolica import S


MODEL_DIR = Path(__file__).resolve().parents[1]


def _reference_vertex_by_key(key: str) -> dict:
    vertices = json.loads(
        (MODEL_DIR / "reference" / "Ltot_SMEFT_FeynRules.json").read_text(
            encoding="utf-8"
        )
    )
    return next(
        vertex
        for vertex in vertices
        if "|".join(sorted(vertex["fields"])) == key
    )


def _feynpy_vertex_by_key(key: str) -> dict:
    vertices = json.loads(
        (MODEL_DIR / "feynpy_vertices.json").read_text(encoding="utf-8")
    )
    return next(vertex for vertex in vertices if vertex["key"] == key)


def test_smeft2_supported_subset_builds_and_compiles():
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    signatures = {signature.names for signature in lagrangian.vertex_signatures()}

    assert len(lagrangian.terms) == 2545
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
    assert len(bundle.model.lagrangian().terms) == 2545
    assert len(full_model.lagrangian().terms) == 2599


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
    # Literal exact-field-multiset signature coverage (independent of the
    # charge-conjugation overlay).
    assert report["summary"]["shared_signatures"] == 182
    assert report["summary"]["reference_only_signatures"] == 2
    assert report["summary"]["feynpy_only_signatures"] == 8
    assert report["summary"]["feynpy_only_charge_conjugation_partners"] == 8
    assert report["summary"]["feynpy_only_unexplained_signatures"] == 0
    assert report["summary"]["feynpy_only_zero_signatures"] == 2
    # Operator-content matching (coefficient-head set), incl. charge conjugation.
    assert report["summary"]["shared_head_matches"] == 176
    assert report["summary"]["charge_conjugation_packaging_matches"] == 8
    assert report["summary"]["operator_content_matches_including_cc"] == 184
    assert report["summary"]["shared_head_count_matches"] == 90
    assert report["summary"]["shared_head_count_mismatches"] == 92
    assert report["summary"]["shared_head_count_benign_expansions"] == 9
    assert report["summary"]["shared_head_count_unexplained_mismatches"] == 83
    assert report["summary"]["exact_symbolic_supported_vertices"] == 32
    assert report["summary"]["exact_symbolic_equal_vertices"] == 32
    assert report["summary"]["exact_symbolic_unequal_vertices"] == 0
    assert report["summary"]["exact_symbolic_missing_local_vertices"] == 0
    assert report["summary"]["exact_symbolic_error_vertices"] == 0
    assert report["summary"]["canonical_map_supported_vertices"] == 32
    assert report["summary"]["canonical_map_equal_vertices"] == 32
    assert report["summary"]["canonical_map_unequal_vertices"] == 0
    assert report["summary"]["canonical_map_error_vertices"] == 0
    assert report["summary"]["canonical_map_supported_coefficient_sectors"] == 93
    assert report["summary"]["canonical_map_equal_coefficient_sectors"] == 93
    assert report["summary"]["canonical_map_unequal_coefficient_sectors"] == 0
    assert report["summary"]["benign_head_count_delta_heads"] == 15
    assert report["summary"]["unexplained_head_count_delta_heads"] == 295
    assert all(
        "head_count_status" in row
        and "reference_head_counts" in row
        and "feynpy_head_counts" in row
        and "head_count_delta" in row
        and "benign_head_count_delta_reasons" in row
        and "unexplained_head_count_delta" in row
        and "canonical_map_status" in row
        and "canonical_map_coefficients" in row
        and "canonical_map_error" in row
        and "exact_symbolic_family" in row
        and "exact_symbolic_status" in row
        and "exact_symbolic_detail" in row
        for row in report["reference_vertices"]
    )

    rows_by_key = {row["key"]: row for row in report["reference_vertices"]}

    # Charge-conjugation packaging overlay: FeynRules keeps the Weinberg and
    # four-fermion "Ec" operators in one bilinear bar assignment while FeynPy
    # emits the charge-conjugate packaging under a bar-flipped signature. These
    # are the same operator and are credited as operator-content matches modulo
    # charge conjugation. The overlay is an annotation only: it does NOT change
    # the literal signature-coverage metrics, so the Weinberg reference row stays
    # reference-only (no exact local signature) and keeps its own empty
    # feynpy_heads; the matched head is recorded in charge_conjugation_matched_heads.
    weinberg = rows_by_key["Phi|Phi|lL|lL"]
    assert weinberg["status"] == "MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING"
    assert weinberg["charge_conjugation_partner"] == "Phi|Phi|lL|lLbar"
    assert weinberg["charge_conjugation_matched_heads"] == ["alphaWeinberg"]
    assert weinberg["feynpy_heads"] == []  # literal per-row head set untouched
    assert weinberg["head_count_status"] == "NO_LOCAL_SIGNATURE"
    ec = rows_by_key["dRbar|eR|lLbar|qL"]
    assert ec["status"] == "SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH"
    assert ec["charge_conjugation_partner"] == "dRbar|eR|lL|qLbar"

    # The FeynPy-only signatures are exactly the eight charge-conjugate partners
    # of reference operators; each is cross-linked back and none is unexplained.
    feynpy_only = {row["key"]: row for row in report["feynpy_only_signatures"]}
    assert len(feynpy_only) == 8
    assert all(
        row["status"] == "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
        and "charge_conjugation_partner" in row
        for row in feynpy_only.values()
    )
    assert "Phi|Phi|lL|lLbar" in feynpy_only
    assert "dRbar|eR|lL|qLbar" in feynpy_only

    # The two B-boson O_Hud vertices are algebraically zero (their canonical
    # coefficient-head set is empty): the U(1) piece of the O_Hud covariant
    # derivative cancels under canonical tensor identities, so FeynRules
    # correctly omits them. They are recorded as zero-signature artifacts, not
    # residual unmatched operator content.
    zero_keys = {row["key"] for row in report["feynpy_only_zero_signatures"]}
    assert zero_keys == {"B|Phi|Phi|dR|uRbar", "B|Phibar|Phibar|dRbar|uR"}
    assert all(
        row["status"] == "FEYNPY_ONLY_ALGEBRAICALLY_ZERO" and row["feynpy_heads"] == []
        for row in report["feynpy_only_zero_signatures"]
    )

    # The triplet pp Higgs-derivative heads no longer survive as unmatched B
    # vertices once both Higgs covariant derivatives keep their explicit weak
    # labels in the model source.
    assert rows_by_key["B|Phi|Phibar|lL|lLbar"]["status"] == "SHARED_HEADS_MATCH"
    assert rows_by_key["B|Phi|Phibar|qL|qLbar"]["status"] == "SHARED_HEADS_MATCH"
    assert rows_by_key["B|Phi|Phibar|lL|lLbar"]["feynpy_extra_heads"] == []
    assert rows_by_key["B|Phi|Phibar|qL|qLbar"]["feynpy_extra_heads"] == []
    assert rows_by_key["Phi|Phibar|Wi|lL|lLbar"]["feynpy_head_counts"]["alphaRHl3pp"] == 2
    assert rows_by_key["Phi|Phibar|Wi|qL|qLbar"]["feynpy_head_counts"]["alphaRHq3pp"] == 2

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
    assert rows_by_key["B|B|B|B|Phi|Phibar"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["B|B|Phi|Phibar"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["B|B|Phi|Phibar"]["canonical_map_status"] == "CANONICAL_MAP_MATCH"
    assert (
        rows_by_key["B|B|Phi|Phibar"]["canonical_map_coefficients"]["alphaRBDH"][
            "matches"
        ]
        is True
    )
    assert rows_by_key["G|G|G"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["G|G|G|G|G"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["Phi|Phibar|Wi"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["Phi|Phibar|Wi"]["canonical_map_status"] == "CANONICAL_MAP_MATCH"
    assert rows_by_key["Phi|Phibar|Wi"]["canonical_map_coefficients"]["alphaRWDH"][
        "matches"
    ] is True
    assert rows_by_key["Phi|Phibar|Wi|Wi"]["canonical_map_coefficients"]["alphaOHW"][
        "matches"
    ] is True
    assert rows_by_key["Phi|Phibar|Wi|Wi"]["canonical_map_coefficients"]["alphaOHWt"][
        "matches"
    ] is True
    assert rows_by_key["Phi|Phibar|Wi|Wi"]["canonical_map_coefficients"]["alphaRWDH"][
        "matches"
    ] is True
    assert rows_by_key["Phi|Phi|Phibar|Phibar"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["Phi|Phi|Phibar|Phibar|Wi"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["Phi|Phi|Phibar|Phibar|Wi|Wi"]["canonical_map_coefficients"][
        "alphaRHDp"
    ]["matches"] is True
    assert rows_by_key["B|qL|qLbar"]["exact_symbolic_status"] == "EXACT_UNSUPPORTED"
    assert rows_by_key["G|G|G"]["canonical_map_status"] == "CANONICAL_MAP_MATCH"
    assert rows_by_key["G|G|G|G|G"]["canonical_map_status"] == "CANONICAL_MAP_MATCH"
    assert rows_by_key["G|G|G"]["canonical_map_coefficients"]["alphaR2G"][
        "matches"
    ] is True


def test_smeft2_five_gluon_canonical_map_matches_feynrules_reference():
    reference = _reference_vertex_by_key("G|G|G|G|G")
    local = _feynpy_vertex_by_key("G|G|G|G|G")
    external_indices = canonical_external_index_set(
        lorentz=tuple(S(f"mu{slot}") for slot in range(1, 6)),
        color_adjoint=tuple(S(f"a{slot}") for slot in range(1, 6)),
    )

    comparisons = compare_canonical_coefficient_maps(
        local["rule"],
        reference["rule"],
        coefficients=("alphaO3G", "alphaO3Gt", "alphaR2G"),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )

    assert all(comparison.matches for comparison in comparisons.values())
    assert {
        coefficient: (
            comparison.feynpy_raw_terms,
            comparison.feynrules_raw_terms,
            comparison.feynpy_canonical_terms,
            comparison.feynrules_canonical_terms,
        )
        for coefficient, comparison in comparisons.items()
    } == {
        "alphaO3G": (720, 240, 120, 120),
        "alphaO3Gt": (720, 420, 180, 180),
        "alphaR2G": (720, 360, 360, 360),
    }


def test_smeft2_bbphiphibar_canonical_map_matches_feynrules_reference_order():
    reference = _reference_vertex_by_key("B|B|Phi|Phibar")
    bundle = build_smeft_green_bpreserving()
    local_rule = bundle.model.lagrangian().feynman_rule(
        bundle.fields["B"],
        bundle.fields["B"],
        bundle.fields["Phi"],
        bundle.fields["Phi"].bar,
        simplify=True,
    )
    external_indices = canonical_external_index_set(
        lorentz=(S("mu1"), S("mu2")),
        weak_fund=(S("w3"), S("w4")),
    )

    comparisons = compare_canonical_coefficient_maps(
        local_rule,
        reference["rule"],
        coefficients=("alphaKH", "alphaOHB", "alphaOHBt", "alphaRBDH", "alphaRDH"),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )

    assert all(comparison.matches for comparison in comparisons.values())
    assert {
        coefficient: (
            comparison.feynpy_raw_terms,
            comparison.feynrules_raw_terms,
            comparison.feynpy_canonical_terms,
            comparison.feynrules_canonical_terms,
        )
        for coefficient, comparison in comparisons.items()
    } == {
        "alphaKH": (1, 1, 1, 1),
        "alphaOHB": (2, 2, 2, 2),
        "alphaOHBt": (8, 2, 1, 1),
        "alphaRBDH": (4, 4, 4, 4),
        "alphaRDH": (9, 9, 9, 9),
    }


def test_smeft2_phiphibarwi_canonical_map_matches_feynrules_reference_order():
    reference = _reference_vertex_by_key("Phi|Phibar|Wi")
    bundle = build_smeft_green_bpreserving()
    local_rule = bundle.model.lagrangian().feynman_rule(
        bundle.fields["Phi"],
        bundle.fields["Phi"].bar,
        bundle.fields["Wi"],
        simplify=True,
    )
    external_indices = canonical_external_index_set(
        lorentz=(S("mu3"),),
        weak_fund=(S("w1"), S("w2")),
        weak_adjoint=(S("aw3"),),
    )

    comparisons = compare_canonical_coefficient_maps(
        local_rule,
        reference["rule"],
        coefficients=("alphaKH", "alphaRDH", "alphaRWDH"),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )

    assert all(comparison.matches for comparison in comparisons.values())
    assert {
        coefficient: (
            comparison.feynpy_raw_terms,
            comparison.feynrules_raw_terms,
            comparison.feynpy_canonical_terms,
            comparison.feynrules_canonical_terms,
        )
        for coefficient, comparison in comparisons.items()
    } == {
        "alphaKH": (2, 2, 2, 2),
        "alphaRDH": (4, 4, 4, 4),
        "alphaRWDH": (4, 4, 4, 4),
    }
