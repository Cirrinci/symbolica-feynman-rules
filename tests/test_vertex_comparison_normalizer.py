from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZER_PATH = REPO_ROOT / "scripts" / "normalize_vertex_comparison.py"

spec = importlib.util.spec_from_file_location("normalize_vertex_comparison", NORMALIZER_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


VERTEX_16_BLOCK = """================================================================================
Vertex 16
Signature: {Phi, Phi, Phibar, Phibar}
Status: BOTH
FR Leg Map: 1:Phi, 2:Phi, 3:Phibar, 4:Phibar
Verdict: DIFFERENT
FeynRules Normalized:
-2*I*delta[SU2D[1],SU2D[2]]*delta[SU2D[3],SU2D[4]]*lam-2*I*delta[SU2D[1],SU2D[4]]*delta[SU2D[3],SU2D[2]]*lam
Python Normalized:
-2I*delta[SU2D[1],SU2D[2]]*delta[SU2D[3],SU2D[4]]*lam-2I*delta[SU2D[3],SU2D[2]]*delta[SU2D[4],SU2D[1]]*lam
"""


def test_vertex_16_normalizes_to_non_different_verdict(tmp_path):
    output_path = tmp_path / "vertex16_normalized.txt"
    blocks = module.parse_blocks(
        (REPO_ROOT / "scripts" / "aligned_outputs" / "aligned_vertex_comparison.txt").read_text(
            encoding="utf-8"
        )
    )
    vertex_16 = next(block for block in blocks if block.vertex_id == 16)

    row = module.process_block(vertex_16)

    assert row["verdict"] in {"EXACT", "MATCH", "EXACT_UP_TO_CANONICALIZATION"}
    assert row["verdict"] != "DIFFERENT"

    output_path.write_text(module.render_report([row]), encoding="utf-8")
    assert output_path.read_text(encoding="utf-8").strip()


def test_vertex_20_generation_labels_normalize_to_exact_match():
    blocks = module.parse_blocks(
        (REPO_ROOT / "scripts" / "aligned_outputs" / "aligned_vertex_comparison.txt").read_text(
            encoding="utf-8"
        )
    )
    vertex_20 = next(block for block in blocks if block.vertex_id == 20)

    row = module.process_block(vertex_20)

    assert row["verdict"] == "EXACT"


def test_vertex_22_epsilon_normalizes_to_exact_match():
    blocks = module.parse_blocks(
        (REPO_ROOT / "scripts" / "aligned_outputs" / "aligned_vertex_comparison.txt").read_text(
            encoding="utf-8"
        )
    )
    vertex_22 = next(block for block in blocks if block.vertex_id == 22)

    row = module.process_block(vertex_22)

    assert row["verdict"] == "EXACT"


def test_vertex_27_triple_gauge_normalizes():
    blocks = module.parse_blocks(
        (REPO_ROOT / "scripts" / "aligned_outputs" / "aligned_vertex_comparison.txt").read_text(
            encoding="utf-8"
        )
    )
    vertex_27 = next(block for block in blocks if block.vertex_id == 27)

    row = module.process_block(vertex_27)

    # Should not be DIFFERENT after canonicalization of Lorentz dummies/metrics
    assert row["verdict"] != "DIFFERENT"
