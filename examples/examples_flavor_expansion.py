"""Minimal flavor-class expansion example."""

from fractions import Fraction

from symbolica import S
from symbolica.community.spenso import Representation

from model import Field, IndexRole, IndexType, Lagrangian, Parameter, SPINOR_INDEX


def _dirac_field(name: str, *, indices=()):
    return Field(
        name,
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S(name),
        conjugate_symbol=S(f"{name}bar"),
        indices=indices,
    )


def build_example():
    generation = IndexType(
        "Generation",
        Representation.cof(3),
        "generation",
        dimension=3,
        role=IndexRole.FLAVOR,
        prefix="f",
    )
    e = _dirac_field("e", indices=(SPINOR_INDEX,))
    mu = _dirac_field("mu", indices=(SPINOR_INDEX,))
    tau = _dirac_field("tau", indices=(SPINOR_INDEX,))
    psi = Field(
        "Psi",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psi"),
        conjugate_symbol=S("psibar"),
        indices=(generation, SPINOR_INDEX),
        flavor_index=generation,
        class_members=(e, mu, tau),
    )
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    yukawa = Parameter("Y", indices=(generation, generation))
    lam = S("lam")
    f, h = S("f", "h")

    compact = Lagrangian(lam * psi.bar(f) * psi(f) * phi)
    mixed = Lagrangian(lam * psi.bar(f) * yukawa(f, h) * psi(h) * phi)
    return compact, mixed, (e, mu, tau), phi


def main():
    compact, mixed, (e, mu, tau), phi = build_example()

    print("compact signatures:")
    for signature in compact.feynman_rules(
        flavor_expand=False,
        key_format="names",
        simplify=True,
        include_delta=True,
    ):
        print(" ", signature)

    print("expanded signatures:")
    for signature in compact.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    ):
        print(" ", signature)

    print("example mixed vertex (e.bar, mu, Phi):")
    print(
        mixed.feynman_rule(
            e.bar,
            mu,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


if __name__ == "__main__":
    main()
