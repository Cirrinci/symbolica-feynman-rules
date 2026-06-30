"""Build the broken Standard Model from gauge-basis declarations."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.SM import build_standard_model  
from symbolic.tensor_canonicalization import canonize_full  


if __name__ == "__main__":
    sm = build_standard_model()
    L = sm.lagrangian
    fields = sm.fields

    print("Gauge-basis terms:", len(sm.source_model.lagrangian().terms))
    print("Broken-phase terms:", len(L.terms))
    print()
    print("Gamma(W-, W+, A):")
    print(L.feynman_rule(fields.W.bar, fields.W, fields.A))
    print()
    print("Gamma(H, W-, W+):")
    print(L.feynman_rule(fields.H, fields.W.bar, fields.W))
    print()
    print("Gamma(lbar, l, A):")
    print(
        canonize_full(
            L.feynman_rule(fields.l.bar, fields.l, fields.A),
            infer_indices=True,
            field_heads=tuple(sm.model.fields),
            run_color=False,
        )
    )
    print()
    print("Gamma(ubar, d, W+), including CKM:")
    print(L.feynman_rule(fields.uq.bar, fields.dq, fields.W))
    print()
    print("Gamma(ghost W+ bar, ghost W+):")
    print(L.feynman_rule(fields.ghWp.bar, fields.ghWp))
