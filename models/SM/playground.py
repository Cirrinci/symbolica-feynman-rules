"""Minimal Standard Model playground.

Run with:
    .venv/bin/python models/SM/playground.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feynpy import *  # noqa: F401,F403
from models.SM import *  # noqa: F401,F403


if __name__ == "__main__":
    L = sm_model(L_gauge, name="SM gauge sector")
    print("Gauge-sector vertices:")
    print(L.feynman_rule())
    print()

    L2 = sm_model(
        -Yd(f2, f3)
        * CKM(f1, f2)
        * QL.bar(spinor, weak_left, f1, colour)
        * dR(spinor, f3, colour)
        * Phi(weak_left),
    )
    print("Custom Yukawa vertices:")
    print(L2.feynman_rule())
