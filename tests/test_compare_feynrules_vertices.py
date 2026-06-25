from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_module(
    "compare_feynrules_vertices",
    "scripts/compare_feynrules_vertices.py",
)
export = _load_module(
    "export_current_sm_source_vertices",
    "scripts/export_current_sm_source_vertices.py",
)


def test_compare_normalizes_current_builder_names_and_namespaces():
    assert compare.normalize_name("QL.bar") == "qLbar"
    assert compare.normalize_name("LL") == "lL"
    assert compare.normalize_name("lR.bar") == "eRbar"

    normalized = compare.normalize_rule(
        "python::{}::g3*spenso::{real}::f("
        "spenso::{spenso::upper}::coad(8,python::{}::a1),"
        "spenso::{spenso::upper}::coad(8,python::{}::a2),"
        "spenso::{spenso::upper}::coad(8,python::{}::a3))"
    )
    assert "python::" not in normalized
    assert "spenso::" not in normalized
    assert "f(coad(8,a1),coad(8,a2),coad(8,a3))" in normalized


def test_current_source_export_matches_reference_signatures_and_selected_factors(tmp_path):
    current_path = tmp_path / "current_sm_source.txt"
    current_path.write_text(
        export.build_source_vertex_export_text(),
        encoding="utf-8",
    )

    fr_vertices = compare.parse_vertices(
        REPO_ROOT / "scripts" / "aligned_outputs" / "normalized_feynrules.txt"
    )
    current_vertices = compare.parse_vertices(current_path)

    fr_grouped = compare.group_by_signature(fr_vertices)
    current_grouped = compare.group_by_signature(current_vertices)
    _both, fr_only, current_only = compare.compare_signatures(
        fr_grouped,
        current_grouped,
    )

    assert fr_only == []
    assert current_only == []

    for signature, expected_factors, hc_base in compare.selected_vertex_specs():
        rule = compare.first_rule_for(current_grouped, signature)
        assert rule is not None, signature
        checks = compare.check_expected_factors(rule, expected_factors)
        assert all(check.ok for check in checks), (signature, checks)
        if hc_base is not None:
            status, _detail = compare.check_hc_yukawa(rule, hc_base)
            assert status != "FAIL", (signature, status)
