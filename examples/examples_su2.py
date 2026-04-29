"""
Minimal SU(2)_L (unbroken) example.

Prints a few representative vertices:
- fermion doublet current: Lbar L W
- scalar doublet current: Hdag H W
- Yang-Mills cubic self-interaction: W W W
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

from symbolica import Expression, S

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import (  # noqa: E402
    CovD,
    Field,
    FieldStrength,
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

mu, nu = S("mu", "nu")
g2 = S("g2")

WField = Field(
    "W",
    spin=1,
    self_conjugate=True,
    symbol=S("W0"),
    indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
)

LDoublet = Field(
    "L",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("L0"),
    conjugate_symbol=S("Lbar0"),
    indices=(SPINOR_INDEX, WEAK_FUND_INDEX),
)

HDoublet = Field(
    "H",
    spin=0,
    self_conjugate=False,
    symbol=S("H0"),
    conjugate_symbol=S("Hdag0"),
    indices=(WEAK_FUND_INDEX,),
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

fermion_model = Model(
    name="SU2L-fermion-demo",
    gauge_groups=(SU2L,),
    fields=(LDoublet, WField),
    lagrangian_decl=I * LDoublet.bar * Gamma(mu) * CovD(LDoublet, mu),
)

scalar_model = Model(
    name="SU2L-scalar-demo",
    gauge_groups=(SU2L,),
    fields=(HDoublet, WField),
    lagrangian_decl=CovD(HDoublet.bar, mu) * CovD(HDoublet, mu),
)

ym_model = Model(
    name="SU2L-ym-demo",
    gauge_groups=(SU2L,),
    fields=(WField,),
    lagrangian_decl=-(Expression.num(1) / Expression.num(4))
    * FieldStrength(SU2L, mu, nu) * FieldStrength(SU2L, mu, nu),
)


if __name__ == "__main__":
    print("SU(2)_L fermion current vertex Γ(Lbar, L, W):")
    print(fermion_model.lagrangian().feynman_rule(LDoublet.bar, LDoublet, WField))
    print()

    print("SU(2)_L scalar current vertex Γ(Hdag, H, W):")
    print(scalar_model.lagrangian().feynman_rule(HDoublet.bar, HDoublet, WField))
    print()

    print("SU(2)_L Yang-Mills cubic vertex Γ(W, W, W):")
    print(ym_model.lagrangian().feynman_rule(WField, WField, WField))
