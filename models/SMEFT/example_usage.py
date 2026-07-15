"""Worked examples of how to use the SMEFT Green-basis model.

Run it directly::

    python -m models.SMEFT.example_usage

Everything below only uses the public API exported from :mod:`models.SMEFT`:

* :func:`build_sm_core`      - the shared unbroken-SM foundation,
* :func:`get_operator`       - one operator by its Appendix-D name,
* :func:`operators_in`       - filter operators by sector / type / table / status,
* :func:`build_smeft`        - compile a whole sector (or the whole basis) at once,
* the :class:`Operator` methods ``structure`` / ``term`` / ``lagrangian`` /
  ``feynman_rule`` / ``canonical_dimensions``.
"""

from __future__ import annotations

from models.SMEFT import (
    build_sm_core,
    build_smeft,
    get_operator,
    operators_in,
)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def example_1_foundation():
    """The shared foundation: unbroken-SM fields, parameters, gauge groups."""
    section("1. The foundation (build_sm_core)")
    core = build_sm_core()

    print("fields     :", ", ".join(f.name for f in core.operator_fields))
    print("gauge groups:", ", ".join(g.name for g in core.group_tuple))
    print("parameters :", ", ".join(p.symbol.to_canonical_string()
                                     for p in core.all_parameters))
    # The renormalizable SM (Eq. D.1) is available for cross-checks.
    print("renormalizable SM terms:", len(core.renormalizable.terms))
    return core


def example_2_browse_registry():
    """Discover which operators exist, filtered by sector / type / table."""
    section("2. Browsing the registry (operators_in)")

    for sector in ("bosonic", "two_fermion", "four_fermion"):
        for otype in ("physical", "redundant", "evanescent"):
            ops = operators_in(sector=sector, otype=otype)
            if ops:
                names = ", ".join(o.name for o in ops[:8])
                more = "" if len(ops) <= 8 else f", ... (+{len(ops) - 8} more)"
                print(f"{sector:>13} / {otype:<10}: {len(ops):>3}  [{names}{more}]")

    blocked = operators_in(status="blocked")
    print("\nblocked (charge-conjugation, Tables 8-9):",
          ", ".join(o.name for o in blocked))


def example_3_inspect_one_operator(core):
    """Inspect a single operator without compiling anything."""
    section("3. Inspecting one operator (get_operator / structure)")

    op = get_operator("O3G")
    print("name        :", op.name)
    print("label       :", op.label)
    print("sector/type :", op.sector, "/", op.otype, "(Table", op.table, ")")
    print("Wilson coeff:", op.wilson_coefficient(core).symbol.to_canonical_string())
    print("mass dim.   :", op.canonical_dimensions(core), "(must be {6})")

    # The bare declared structure (Wilson coefficient set to 1).
    structure = op.structure(core)
    print("declared monomials:", len(structure.source_terms))


def example_4_feynman_rules(core):
    """Compile individual operators and read off Feynman rules."""
    section("4. Feynman rules from single operators (feynman_rule)")

    f = core.fields

    # Purely gluonic triple-field-strength operator -> 3-gluon vertex.
    o3g = get_operator("O3G")
    rule = o3g.feynman_rule(core, f.G, f.G, f.G)
    print("O3G  ->  GGG vertex generated:", rule is not None)

    # Chromomagnetic dipole O_uG = (qbar sigma^{mu nu} T^A u) Htilde G^A_{mu nu}
    # -> a quark-quark-Higgs-gluon contact vertex.
    oug = get_operator("OuG")
    rule = oug.feynman_rule(core, f.q.bar, f.u, f.H.bar, f.G)
    print("OuG  ->  q-qbar-H-G vertex generated:", rule is not None)

    # The two-fermion current operator O_Hq^(1).
    ohq1 = get_operator("OHq1")
    lag = ohq1.lagrangian(core)
    sigs = lag.vertex_signatures()
    print("OHq1 ->  vertex signatures:", len(sigs))


def example_5_build_a_sector():
    """Compile a whole sector at once with build_smeft."""
    section("5. Assembling a sector (build_smeft)")

    smeft = build_smeft(sectors=["bosonic"])
    print("selected operators:", len(smeft.operators))
    print("compiled vertex signatures:", len(smeft.lagrangian.vertex_signatures()))

    # Restrict further: only the physical two-fermion dipoles/currents.
    phys = build_smeft(sectors=["two_fermion"], types=["physical"])
    print("two-fermion physical operators:", len(phys.operators))


def example_6_blocked_operator(core):
    """Blocked operators still expose their declared structure."""
    section("6. A blocked operator (declared but not compilable)")

    op = get_operator("Ecuu")
    print("name/status :", op.name, "/", op.status)
    print("note        :", (op.note or "(see checklist.md)")[:70])
    structure = op.structure(core)  # inspectable
    print("declared monomials:", len(structure.source_terms))
    print("-> compilation is intentionally not attempted (charge-conjugation).")


def main():
    core = example_1_foundation()
    example_2_browse_registry()
    example_3_inspect_one_operator(core)
    example_4_feynman_rules(core)
    example_5_build_a_sector()
    example_6_blocked_operator(core)
    print("\nDone.")


if __name__ == "__main__":
    main()
