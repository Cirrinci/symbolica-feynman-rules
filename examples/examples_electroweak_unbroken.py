"""
Minimal unbroken electroweak example.

Prints three representative vertices:
- fermion doublet current: Lbar L W
- Higgs-doublet hypercharge current: Hdag H B
- mixed electroweak contact: Hdag H W B
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from symbolica import S

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import (  # noqa: E402
    CovD,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    Gamma,
    LORENTZ_INDEX,
    Model,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
)
from symbolic.spenso_structures import weak_gauge_generator, weak_structure_constant  # noqa: E402
from symbolic.vertex_engine import I  # noqa: E402

mu = S("mu")
g1 = S("g1")
g2 = S("g2")
yL = S("yL")
yH = S("yH")

WField = Field(
    "W",
    spin=1,
    self_conjugate=True,
    symbol=S("W0"),
    indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
)

BField = Field(
    "B",
    spin=1,
    self_conjugate=True,
    symbol=S("B0"),
    indices=(LORENTZ_INDEX,),
)

LDoublet = Field(
    "L",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("L0"),
    conjugate_symbol=S("Lbar0"),
    indices=(SPINOR_INDEX, WEAK_FUND_INDEX),
    quantum_numbers={"Y": yL},
)

HDoublet = Field(
    "H",
    spin=0,
    self_conjugate=False,
    symbol=S("H0"),
    conjugate_symbol=S("Hdag0"),
    indices=(WEAK_FUND_INDEX,),
    quantum_numbers={"Y": yH},
)

WEAK_DOUBLET_REP = GaugeRepresentation(
    index=WEAK_FUND_INDEX,
    generator_builder=weak_gauge_generator,
    name="doublet",
)

SU2L = GaugeGroup(
    name="SU2L",
    abelian=False,
    coupling=g2,
    gauge_boson="W",
    structure_constant=weak_structure_constant,
    representations=(WEAK_DOUBLET_REP,),
)

U1Y = GaugeGroup(
    name="U1Y",
    abelian=True,
    coupling=g1,
    gauge_boson="B",
    charge="Y",
)

fermion_model = Model(
    name="EW-fermion-demo",
    gauge_groups=(SU2L, U1Y),
    fields=(LDoublet, WField, BField),
    lagrangian_decl=I * LDoublet.bar * Gamma(mu) * CovD(LDoublet, mu),
)

higgs_model = Model(
    name="EW-higgs-demo",
    gauge_groups=(SU2L, U1Y),
    fields=(HDoublet, WField, BField),
    lagrangian_decl=CovD(HDoublet.bar, mu) * CovD(HDoublet, mu),
)


if __name__ == "__main__":
    print("Unbroken electroweak fermion current Γ(Lbar, L, W):")
    print(fermion_model.lagrangian().feynman_rule(LDoublet.bar, LDoublet, WField))
    print()

    print("Unbroken electroweak Higgs current Γ(Hdag, H, B):")
    print(higgs_model.lagrangian().feynman_rule(HDoublet.bar, HDoublet, BField))
    print()

    print("Unbroken electroweak mixed contact Γ(Hdag, H, W, B):")
    print(higgs_model.lagrangian().feynman_rule(HDoublet.bar, HDoublet, WField, BField))
