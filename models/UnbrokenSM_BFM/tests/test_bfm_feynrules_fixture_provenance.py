import hashlib
import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "reference" / "feynrules"
)
MODEL_SHA256 = "f0952159e47b807d7b5daaa89f70d00666413f68fca7d0d4fe10cbe891c372d9"
NOTEBOOK_SHA256 = "c66c28303da3406a9e2089e31c64e20dd2f3b0dd86616105c5bdf41167522f4b"
VERTEX_EXPORT_SHA256 = "fd34adaed860c50278b6dba965f8a2bc532b1f8f32336e85bfcb8d6430bc8033"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feynrules_bfm_reference_material_is_present_and_stable():
    assert _sha256(FIXTURE_DIR / "UnbrokenSM_BFM.fr") == MODEL_SHA256
    assert _sha256(FIXTURE_DIR / "UnbrokenSM_BFM_export.nb") == NOTEBOOK_SHA256
    assert _sha256(FIXTURE_DIR / "LSM_full_FeynRules.json") == VERTEX_EXPORT_SHA256


def test_feynrules_bfm_vertex_export_has_complete_unique_signatures():
    vertices = json.loads(
        (FIXTURE_DIR / "LSM_full_FeynRules.json").read_text(encoding="utf-8")
    )

    assert len(vertices) == 67
    assert sum(len(vertex["fields"]) == 3 for vertex in vertices) == 42
    assert sum(len(vertex["fields"]) == 4 for vertex in vertices) == 25

    keys = [vertex["key"] for vertex in vertices]
    assert len(keys) == len(set(keys))
    for vertex in vertices:
        assert sorted(vertex["key"].split("|")) == sorted(vertex["fields"])
        assert len(vertex["legs"]) == len(vertex["fields"])
        assert vertex["rule"]
