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


L_gauge_model = sm_model(L_gauge, name="SM gauge sector")
L_fermions_model = sm_model(L_fermions, name="SM fermion sector")
L_higgs_model = sm_model(L_higgs, name="SM Higgs sector")
L_yukawa_model = sm_model(L_yukawa, name="SM Yukawa sector")
L_gauge_fixing_model = sm_model(L_gauge_fixing, name="SM gauge-fixing sector")
L_ghost_model = sm_model(L_ghost, name="SM ghost sector")
L_tot_model = sm_model(L_tot, name="SM total lagrangian")

SECTOR_MODELS = {
    "Gauge Sector": L_gauge_model,
    "Fermion Sector": L_fermions_model,
    "Higgs Sector": L_higgs_model,
    "Yukawa Sector": L_yukawa_model,
    "Gauge-Fixing Sector": L_gauge_fixing_model,
    "Ghost Sector": L_ghost_model,
    "Total Lagrangian": L_tot_model,
}

custom_yukawa_model = sm_model(
    -Yd(f2, f3)
    * CKM(f1, f2)
    * QL.bar(spinor, weak_left, f1, colour)
    * dR(spinor, f3, colour)
    * Phi(weak_left),
    name="Custom Yukawa term",
)


if __name__ == "__main__":
    for title, model in SECTOR_MODELS.items():
        print(f"===== {title} =====")
        show_model(model)
        print()

    print("===== Custom Yukawa Example =====")
    show_model(custom_yukawa_model)
