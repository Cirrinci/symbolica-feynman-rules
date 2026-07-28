import json
from collections import defaultdict
from pathlib import Path

import models.SMEFT2.comparison as smeft2_comparison
from feynrules.comparison import compare_canonical_coefficient_maps
from feynpy import Model
from models.SMEFT2 import OMITTED_SECTORS, build_smeft_green_bpreserving
from symbolic.tensor_canonicalization import canonical_external_index_set
from symbolica import Expression, S


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


def _weinberg_vertex_by_key(key: str) -> dict:
    vertices = json.loads(
        (MODEL_DIR / "weinberg_vertices.json").read_text(encoding="utf-8")
    )
    return next(vertex for vertex in vertices if vertex["key"] == key)


def _comparison_report() -> dict:
    return json.loads(
        (MODEL_DIR / "vertex_comparison_report.json").read_text(encoding="utf-8")
    )


def _weinberg_comparison_report() -> dict:
    return json.loads(
        (MODEL_DIR / "weinberg_comparison_report.json").read_text(encoding="utf-8")
    )


def _compact_rule_text(rule: str) -> str:
    return rule.replace("python::{}::", "").replace("python::{real}::", "")


def _report_row_by_key(key: str) -> dict:
    report = _comparison_report()
    return next(row for row in report["reference_vertices"] if row["key"] == key)


def _pinned_cc_rows(report: dict) -> list[dict]:
    return [
        row
        for row in report["reference_vertices"]
        if row["exact_symbolic_status"] == "MATCH_MODULO_CC_PACKAGING"
    ]


def _assert_reference_row_exact_match(key: str):
    reference = _reference_vertex_by_key(key)
    report_row = _report_row_by_key(key)
    bundle = build_smeft_green_bpreserving()
    field_map = smeft2_comparison._comparison_field_map(bundle)
    fields = tuple(field_map[name] for name in reference["fields"])
    external_indices = smeft2_comparison._external_index_set_from_fields(fields)
    local_rule = bundle.model.lagrangian().feynman_rule(*fields, simplify=True)
    reference_rule = smeft2_comparison.parse_smeft2_matter_rule(reference["rule"])
    comparisons = smeft2_comparison._compare_smeft2_canonical_coefficient_maps(
        local_rule,
        reference_rule,
        coefficients=tuple(
            head for head in report_row["reference_heads"] if head.startswith("alpha")
        ),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )

    assert comparisons
    assert all(comparison.matches for comparison in comparisons.values())


def _comparison_context():
    bundle = build_smeft_green_bpreserving()
    references_by_key = defaultdict(list)
    for reference in smeft2_comparison.load_feynrules_json(
        smeft2_comparison.REFERENCE
    ):
        key = smeft2_comparison._name_key(reference.fields)
        references_by_key[key].append(reference)
    return (
        bundle.model.lagrangian(),
        smeft2_comparison._comparison_field_map(bundle),
        set(bundle.parameters) | smeft2_comparison.GENERIC_PARAMETER_NAMES,
        references_by_key,
    )


def _reference_with_head(references_by_key, key: str, head: str, parameter_names):
    candidates = []
    for reference in references_by_key[key]:
        reference_heads = smeft2_comparison._reference_heads(
            reference,
            parameter_names,
        )
        if head in reference_heads:
            candidates.append(reference)
    assert len(candidates) == 1
    return candidates[0]


def _check_summary(**overrides):
    summary = {
        "operator_content_matches_including_cc": 184,
        "reference_vertex_count": 184,
        "shared_head_matches": 176,
        "charge_conjugation_packaging_matches": 8,
        "exact_symbolic_equal_vertices": 184,
        "exact_symbolic_direct_match_vertices": 184,
        "exact_symbolic_supported_vertices": 184,
        "exact_symbolic_unequal_vertices": 0,
        "exact_symbolic_missing_local_vertices": 0,
        "exact_symbolic_error_vertices": 0,
        "cc_packaging_pinned_match_vertices": 0,
        "cc_packaging_unresolved_vertices": 0,
        "canonical_map_equal_vertices": 32,
        "canonical_map_supported_vertices": 32,
        "canonical_map_equal_coefficient_sectors": 93,
        "canonical_map_supported_coefficient_sectors": 93,
        "canonical_map_unequal_vertices": 0,
        "canonical_map_error_vertices": 0,
        "shared_head_count_matches": 100,
        "shared_signatures": 182,
        "reference_only_signatures": 2,
        "feynpy_only_signatures": 8,
        "feynpy_only_unexplained_signatures": 0,
        "shared_head_count_mismatches": 82,
    }
    summary.update(overrides)
    return summary


def _patch_passing_weinberg_comparison(monkeypatch):
    monkeypatch.setattr(
        smeft2_comparison,
        "compare_reconstructed_weinberg",
        lambda _reference=smeft2_comparison.REFERENCE: (
            {
                "summary": {
                    "reference_vertices": 2,
                    "direct_matches": 2,
                    "wrong_sign_matches": 0,
                    "coefficient_checks": 4,
                    "coefficient_matches": 4,
                }
            },
            [],
        ),
    )


def test_smeft2_check_requires_direct_exact_by_default(monkeypatch):
    def fake_compare(_reference):
        return {"summary": _check_summary()}, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 0


def test_smeft2_check_rejects_pinned_cc_without_flag(monkeypatch):
    def fake_compare(_reference):
        return {
            "summary": _check_summary(
                exact_symbolic_equal_vertices=182,
                exact_symbolic_direct_match_vertices=182,
                cc_packaging_pinned_match_vertices=2,
            )
        }, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 1
    assert smeft2_comparison.main(["--check", "--allow-cc-packaging"]) == 0


def test_smeft2_check_rejects_unresolved_cc_even_with_allow_flag(monkeypatch):
    def fake_compare(_reference):
        return {
            "summary": _check_summary(
                exact_symbolic_equal_vertices=176,
                exact_symbolic_direct_match_vertices=176,
                cc_packaging_pinned_match_vertices=2,
                cc_packaging_unresolved_vertices=6,
            )
        }, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 1
    assert smeft2_comparison.main(["--check", "--allow-cc-packaging"]) == 1


def test_smeft2_check_rejects_incomplete_exact_status_accounting(monkeypatch):
    def fake_compare(_reference):
        return {
            "summary": _check_summary(
                exact_symbolic_equal_vertices=176,
                exact_symbolic_direct_match_vertices=176,
            )
        }, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 1


def test_smeft2_check_still_rejects_unexplained_or_strict_count_gaps(monkeypatch):
    def fake_compare(_reference):
        return {
            "summary": _check_summary(feynpy_only_unexplained_signatures=1)
        }, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 1

    def fake_compare_with_raw_count_gap(_reference):
        return {"summary": _check_summary()}, ()

    monkeypatch.setattr(
        smeft2_comparison,
        "compare",
        fake_compare_with_raw_count_gap,
    )
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check", "--strict-counts"]) == 1


def test_smeft2_check_rejects_strict_exact_mismatches(monkeypatch):
    def fake_compare(_reference):
        return {
            "summary": _check_summary(
                exact_symbolic_equal_vertices=175,
                exact_symbolic_direct_match_vertices=175,
                exact_symbolic_unequal_vertices=1,
            )
        }, ()

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    _patch_passing_weinberg_comparison(monkeypatch)
    assert smeft2_comparison.main(["--check"]) == 1


def test_smeft2_check_rejects_weinberg_sidecar_mismatch(monkeypatch):
    def fake_compare(_reference):
        return {"summary": _check_summary()}, ()

    def fake_weinberg_compare(_reference):
        return {
            "summary": {
                "reference_vertices": 2,
                "direct_matches": 1,
                "wrong_sign_matches": 0,
                "coefficient_checks": 4,
                "coefficient_matches": 4,
            }
        }, []

    monkeypatch.setattr(smeft2_comparison, "compare", fake_compare)
    monkeypatch.setattr(
        smeft2_comparison,
        "compare_reconstructed_weinberg",
        fake_weinberg_compare,
    )

    assert smeft2_comparison.main(["--check"]) == 1


def test_smeft2_indexed_coefficient_filter_is_not_vacuous():
    comparisons = smeft2_comparison._compare_smeft2_canonical_coefficient_maps(
        Expression.parse("alphaKq(f1,f2)*x"),
        Expression.num(0),
        coefficients=("alphaKq",),
        external_indices=canonical_external_index_set(),
    )

    comparison = comparisons["alphaKq"]
    assert comparison.feynpy_raw_terms == 1
    assert comparison.feynrules_raw_terms == 0
    assert not comparison.matches


def test_smeft2_pinned_cc_rows_are_only_proven_weinberg_or_ec_classes():
    report = _comparison_report()
    rows = _pinned_cc_rows(report)

    assert rows
    assert len(rows) == report["summary"]["cc_packaging_pinned_match_vertices"]
    assert all(
        row["reference_heads"] == ["alphaWeinberg"]
        or any(head.startswith("alphaEc") for head in row["reference_heads"])
        for row in rows
    )


def test_smeft2_weinberg_packaging_is_proven_by_antisymmetric_canonical_maps():
    report = _comparison_report()
    pinned_rows = _pinned_cc_rows(report)
    weinberg_rows = [
        row
        for row in pinned_rows
        if row["reference_heads"] == ["alphaWeinberg"]
    ]
    ec_rows = [
        row
        for row in pinned_rows
        if any(head.startswith("alphaEc") for head in row["reference_heads"])
    ]
    assert len(weinberg_rows) + len(ec_rows) == len(pinned_rows)
    assert weinberg_rows

    lagrangian, field_map, parameter_names, references_by_key = _comparison_context()

    for row in weinberg_rows:
        reference = _reference_with_head(
            references_by_key,
            row["key"],
            "alphaWeinberg",
            parameter_names,
        )
        packaged_orders = smeft2_comparison._weinberg_packaged_field_orders(
            reference.fields
        )
        assert packaged_orders is not None
        assert row["charge_conjugation_partner"] in {
            smeft2_comparison._name_key(order) for order in packaged_orders
        }

        fields = tuple(field_map[name] for name in reference.fields)
        external_indices = smeft2_comparison._external_index_set_from_fields(fields)
        assert external_indices is not None
        first_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[0]),
            simplify=True,
        )
        second_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[1]),
            simplify=True,
        )
        reference_rule = smeft2_comparison.parse_smeft2_matter_rule(
            reference.rule,
            projector_as_dirac_c=True,
        )

        antisymmetric = smeft2_comparison._compare_smeft2_canonical_coefficient_maps(
            (first_rule - second_rule).cancel().expand(),
            reference_rule,
            coefficients=("alphaWeinberg",),
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )["alphaWeinberg"]
        symmetric = smeft2_comparison._compare_smeft2_canonical_coefficient_maps(
            (first_rule + second_rule).cancel().expand(),
            reference_rule,
            coefficients=("alphaWeinberg",),
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )["alphaWeinberg"]

        assert antisymmetric.matches
        assert antisymmetric.feynpy_only == {}
        assert antisymmetric.feynrules_only == {}
        assert antisymmetric.coefficient_mismatches == {}
        assert not symmetric.matches


def test_smeft2_reconstructed_weinberg_sidecar_export_shape():
    vertex_keys = {
        vertex["key"]
        for vertex in json.loads(
            (MODEL_DIR / "weinberg_vertices.json").read_text(encoding="utf-8")
        )
    }
    assert vertex_keys == {
        "Phi|Phi|lL|lL",
        "Phibar|Phibar|lLbar|lLbar",
    }

    weinberg = _weinberg_vertex_by_key("Phi|Phi|lL|lL")
    assert weinberg["fields"] == ["lL", "lL", "Phi", "Phi"]
    assert weinberg["source_orders"] == {
        "first": ["lLbar", "lL", "Phi", "Phi"],
        "second": ["lL", "lLbar", "Phi", "Phi"],
        "combination": "first - second",
    }
    assert weinberg["spinor_representation"] == "PL"
    assert weinberg["flavor_structures"] == [
        "alphaWeinberg(f1,f2)",
        "alphaWeinberg(f2,f1)",
    ]
    weinberg_rule = _compact_rule_text(weinberg["rule"])
    assert "alphaWeinberg(f1,f2)" in weinberg_rule
    assert "alphaWeinberg(f2,f1)" in weinberg_rule
    assert "PL(" in weinberg_rule

    weinberg_hc = _weinberg_vertex_by_key("Phibar|Phibar|lLbar|lLbar")
    assert weinberg_hc["fields"] == ["lLbar", "lLbar", "Phibar", "Phibar"]
    assert weinberg_hc["source_orders"] == {
        "first": ["lLbar", "lL", "Phibar", "Phibar"],
        "second": ["lL", "lLbar", "Phibar", "Phibar"],
        "combination": "first - second",
    }
    assert weinberg_hc["spinor_representation"] == "PR"
    assert weinberg_hc["flavor_structures"] == [
        "conj(alphaWeinberg(f1,f2))",
        "conj(alphaWeinberg(f2,f1))",
    ]
    weinberg_hc_rule = _compact_rule_text(weinberg_hc["rule"])
    assert "conj(alphaWeinberg(f1,f2))" in weinberg_hc_rule
    assert "conj(alphaWeinberg(f2,f1))" in weinberg_hc_rule
    assert "PR(" in weinberg_hc_rule

    report = _weinberg_comparison_report()
    assert report["summary"]["reference_vertices"] == 2
    assert report["summary"]["direct_matches"] == 2
    assert report["summary"]["wrong_sign_matches"] == 0
    assert report["summary"]["coefficient_checks"] == 4
    assert report["summary"]["coefficient_matches"] == 4


def test_smeft2_reconstructed_weinberg_matches_feynrules_by_flavor_and_sign():
    report, vertices = smeft2_comparison.compare_reconstructed_weinberg()
    assert {vertex["key"] for vertex in vertices} == {
        "Phi|Phi|lL|lL",
        "Phibar|Phibar|lLbar|lLbar",
    }
    assert report["summary"]["reference_vertices"] == 2
    assert report["summary"]["direct_matches"] == 2
    assert report["summary"]["wrong_sign_matches"] == 0
    assert report["summary"]["coefficient_checks"] == 4
    assert report["summary"]["coefficient_matches"] == 4

    expected_coefficients = {
        "Phi|Phi|lL|lL": {
            "alphaWeinberg(f1,f2)",
            "alphaWeinberg(f2,f1)",
        },
        "Phibar|Phibar|lLbar|lLbar": {
            "conj(alphaWeinberg(f1,f2))",
            "conj(alphaWeinberg(f2,f1))",
        },
    }
    for row in report["vertices"]:
        assert row["matches"]
        assert not row["wrong_sign_matches"]
        checks = row["coefficient_checks"]
        assert {check["coefficient"] for check in checks} == expected_coefficients[
            row["key"]
        ]
        assert all(check["matches"] for check in checks)
        assert all(check["feynpy_coefficient"] != "0" for check in checks)
        assert all(check["feynrules_coefficient"] != "0" for check in checks)

    lagrangian, field_map, parameter_names, references_by_key = _comparison_context()
    for key in expected_coefficients:
        reference = _reference_with_head(
            references_by_key,
            key,
            "alphaWeinberg",
            parameter_names,
        )
        external_indices = smeft2_comparison._weinberg_external_indices(
            reference,
            field_map,
        )
        feynrules_rule = smeft2_comparison._parse_weinberg_fermion_flow_rule(
            reference.rule
        )
        first_minus_second = smeft2_comparison._reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=-1,
        )
        first_plus_second = smeft2_comparison._reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=1,
        )

        assert smeft2_comparison._weinberg_canonical_zero(
            (first_minus_second - feynrules_rule).cancel().expand(),
            external_indices=external_indices,
        )
        assert not smeft2_comparison._weinberg_canonical_zero(
            (first_plus_second - feynrules_rule).cancel().expand(),
            external_indices=external_indices,
        )


def test_smeft2_ec_partner_packaging_rules_are_proven_by_canonical_maps():
    rules = smeft2_comparison._EC_PARTNER_PACKAGING_RULES
    report = _comparison_report()
    ec_rows = [
        row
        for row in _pinned_cc_rows(report)
        if any(head.startswith("alphaEc") for head in row["reference_heads"])
    ]
    expected_rule_keys = {
        (row["key"], head)
        for row in ec_rows
        for head in row["reference_heads"]
        if head.startswith("alphaEc")
    }
    assert expected_rule_keys
    assert set(rules) == expected_rule_keys

    lagrangian, field_map, parameter_names, references_by_key = _comparison_context()
    local_vertices = smeft2_comparison._local_vertices(parameter_names)

    for reference_key, coefficient in sorted(rules):
        reference = _reference_with_head(
            references_by_key,
            reference_key,
            coefficient,
            parameter_names,
        )
        fields = tuple(field_map[name] for name in reference.fields)
        external_indices = smeft2_comparison._external_index_set_from_fields(fields)
        assert external_indices is not None

        feynrules_report = smeft2_comparison._canonical_report_for_coefficient_head(
            smeft2_comparison.parse_smeft2_matter_rule(reference.rule),
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
        result = smeft2_comparison._ec_partner_packaging_comparison(
            reference=reference,
            coefficient=coefficient,
            feynrules_report=feynrules_report,
            local_vertices=local_vertices,
            lagrangian=lagrangian,
            field_map=field_map,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )

        assert result is not None
        comparison, detail = result
        assert comparison.matches
        assert comparison.feynpy_only == {}
        assert comparison.feynrules_only == {}
        assert comparison.coefficient_mismatches == {}
        assert rules[(reference_key, coefficient)].partner_key in detail


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
    assert report["summary"]["shared_head_count_matches"] == 100
    assert report["summary"]["shared_head_count_mismatches"] == 82
    assert report["summary"]["shared_head_count_benign_expansions"] == 82
    assert report["summary"]["shared_head_count_unexplained_mismatches"] == 0
    assert report["summary"]["exact_symbolic_supported_vertices"] == 184
    assert report["summary"]["exact_symbolic_direct_match_vertices"] == 176
    assert report["summary"]["exact_symbolic_equal_vertices"] == 176
    assert report["summary"]["cc_packaging_pinned_match_vertices"] == 8
    assert report["summary"]["cc_packaging_unresolved_vertices"] == 0
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
    assert report["summary"]["benign_head_count_delta_heads"] == 285
    assert report["summary"]["unexplained_head_count_delta_heads"] == 0
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
    assert rows_by_key["B|qL|qLbar"]["exact_symbolic_status"] == "EXACT_MATCH"
    assert rows_by_key["G|G|G"]["benign_head_count_delta_reasons"]["alphaO3G"] == (
        "EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE"
    )
    assert (
        rows_by_key["dRbar|eR|lLbar|qL"]["exact_symbolic_status"]
        == "MATCH_MODULO_CC_PACKAGING"
    )
    assert rows_by_key["dRbar|eR|lLbar|qL"]["benign_head_count_delta_reasons"][
        "alphaEcqedl"
    ] == "PINNED_CC_CANONICAL_EQUIVALENCE"
    assert "pinned charge-conjugation partner" in rows_by_key["dRbar|eR|lLbar|qL"][
        "exact_symbolic_detail"
    ]
    assert (
        rows_by_key["Phi|Phi|lL|lL"]["exact_symbolic_status"]
        == "MATCH_MODULO_CC_PACKAGING"
    )
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
