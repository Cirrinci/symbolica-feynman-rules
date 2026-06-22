"""Minimal FeynRules-style flavor-class example."""

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

from feynpy import (  # noqa: E402
    COLOR_FUND_INDEX,
    GaugeGroup,
    GaugeRepresentation,
    Model,
    Parameter,
    dirac_field,
    flavor_index,
    scalar_field,
)
from symbolic.spenso_structures import gauge_generator, structure_constant  # noqa: E402


def build_example():
    # Indices
    Generation = flavor_index("Generation", 3, prefix="f")
    Colour = COLOR_FUND_INDEX

    # Gauge representations / gauge groups
    colour_fund = GaugeRepresentation(
        index=Colour,
        generator_builder=gauge_generator,
        name="fund",
    )
    SU3C = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gs"),
        gauge_boson="G",
        structure_constant=structure_constant,
        representations=(colour_fund,),
    )

    # Fields / particle classes: FeynRules-style ClassMembers declarations.
    # Members inherit the class metadata and drop the flavor-index slot.
    l = dirac_field(
        "l",
        class_members=("e", "mu", "ta"),
        indices=(Generation,),
        flavor_index=Generation,
        quantum_numbers={"Q": -1, "LeptonNumber": 1},
    )
    uq = dirac_field(
        "uq",
        class_members=("u", "c", "t"),
        indices=(Generation, Colour),
        flavor_index=Generation,
        quantum_numbers={"Q": Fraction(2, 3)},
    )
    dq = dirac_field(
        "dq",
        class_members=("d", "s", "b"),
        indices=(Generation, Colour),
        flavor_index=Generation,
        quantum_numbers={"Q": Fraction(-1, 3)},
    )
    Phi = scalar_field("Phi")

    # Parameters
    gQ = Parameter("gQ")
    yu = Parameter(
        "yu",
        indices=(Generation, Generation),
        components={
            (1, 2): 0,
            (1, 3): 0,
            (2, 1): 0,
            (2, 3): 0,
            (3, 1): 0,
            (3, 2): 0,
        },
    )

    # Lagrangian
    f, h, c = S("f", "h", "c")
    model = Model(
        name="Flavor-class demo",
        gauge_groups=(SU3C,),
        fields=(l, uq, dq, Phi),
        parameters=(gQ, yu),
        lagrangian_decl=(
            gQ * uq.bar(f, c) * uq(f, c) * Phi
            + yu(f, h) * uq.bar(f, c) * uq(h, c) * Phi
        ),
    )
    return model, Generation, l, uq, dq, Phi


def main():
    model, Generation, l, uq, dq, Phi = build_example()
    _e, _mu, _ta = l.class_members
    u, c, t = uq.class_members
    _d, _s, _b = dq.class_members

    print("compact signatures:")
    for signature in model.vertex_signatures(flavor_expand=False):
        print(" ", signature.names)

    print("expanded signatures (Generation):")
    for signature in model.vertex_signatures(flavor_expand=Generation):
        print(" ", signature.names)

    print("example diagonal Yukawa rule Γ(u.bar, u, Phi):")
    print(
        model.feynman_rule(
            u.bar,
            u,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=Generation,
        )
    )

    print("example diagonal Yukawa rule Γ(t.bar, t, Phi):")
    print(
        model.feynman_rule(
            t.bar,
            t,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=Generation,
        )
    )

    print("off-diagonal Yukawa entries are removed by components={(i,j): 0}:")
    try:
        print(
            model.feynman_rule(
                u.bar,
                c,
                Phi,
                simplify=True,
                include_delta=True,
                flavor_expand=Generation,
            )
        )
    except ValueError as exc:
        print(exc)


if __name__ == "__main__":
    main()
