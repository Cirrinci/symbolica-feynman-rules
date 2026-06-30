import hashlib
import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "reference" / "feynrules"
)
SM_FR_SHA256 = "44690e769ecc4ed649033d2f9d58c5672203d8e820c56c90a378464204c99edc"
SM_VERTEX_EXPORT_SHA256 = (
    "01dc1b98feb6a112b65e8c1cb42aa9e005fb6ae362767850d7b5888e8689913f"
)


def test_feynrules_standard_model_fixture_provenance_is_stable():
    model_metadata = json.loads(
        (FIXTURE_DIR / "sm_model_FeynRules.json").read_text(encoding="utf-8")
    )
    vertex_export_hash = hashlib.sha256(
        (FIXTURE_DIR / "sm_vertices_FeynRules.json").read_bytes()
    ).hexdigest()

    assert model_metadata["source_sha256"] == SM_FR_SHA256
    assert vertex_export_hash == SM_VERTEX_EXPORT_SHA256


def test_sector_fixtures_are_an_exact_partition_of_full_export():
    full_vertices = json.loads(
        (FIXTURE_DIR / "sm_vertices_FeynRules.json").read_text(encoding="utf-8")
    )
    def comparison_payload(vertex):
        return {
            key: vertex[key]
            for key in ("key", "fields", "legs", "rule")
        }

    full_by_key = {
        vertex["key"]: comparison_payload(vertex)
        for vertex in full_vertices
    }

    sector_by_key = {}
    for sector in ("gauge", "matter", "yukawa", "higgs", "ghost"):
        vertices = json.loads(
            (FIXTURE_DIR / f"{sector}_vertices_FeynRules.json").read_text(
                encoding="utf-8"
            )
        )
        for vertex in vertices:
            assert vertex["key"] not in sector_by_key
            sector_by_key[vertex["key"]] = comparison_payload(vertex)

    assert sector_by_key == full_by_key
