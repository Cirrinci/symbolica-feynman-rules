"""
Compact electroweak SSB example.

Demonstrates:
- the standard Higgs doublet before breaking
- the explicit vev expansion around ``v``
- the charged and neutral electroweak mixing relations
- ``W`` and ``Z`` mass terms from the broken Higgs sector
- absence of a Higgs-induced photon mass term
- one simple diagonal Yukawa mass plus ``h fbar f`` coupling
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
    DiagonalYukawaAssignment,
    ElectroweakGaugeFixing,
    Field,
    SPINOR_INDEX,
    build_broken_electroweak_sector,
    standard_model_higgs_doublet,
)

g1 = S("g1")
g2 = S("g2")
v = S("v")
ye = S("y_e")
lamH = S("lamH")
xiW = S("xiW")
xiZ = S("xiZ")
xiA = S("xiA")

electron = Field(
    "e",
    spin=Fraction(1, 2),
    self_conjugate=False,
    symbol=S("e0"),
    conjugate_symbol=S("ebar0"),
    indices=(SPINOR_INDEX,),
)

higgs_doublet = standard_model_higgs_doublet()
broken = build_broken_electroweak_sector(
    g1=g1,
    g2=g2,
    vev=v,
    include_gauge_sector=True,
    higgs_quartic=lamH,
    gauge_fixing=ElectroweakGaugeFixing(xi_w=xiW, xi_z=xiZ, xi_a=xiA),
    higgs_doublet=higgs_doublet,
    yukawas=(DiagonalYukawaAssignment(electron, ye, label="electron Yukawa"),),
)
L = broken.model.lagrangian()


def _has_higgs_sector_photon_mass_term() -> bool:
    photon = broken.fields.photon
    for term in L.terms:
        if term.derivatives:
            continue
        term_fields = tuple((occ.field, occ.conjugated) for occ in term.fields)
        if term_fields == ((photon, False), (photon, False)):
            return True
    return False


if __name__ == "__main__":
    print("Standard Higgs doublet before breaking:")
    print(higgs_doublet)
    print("Hypercharge Y =", higgs_doublet.quantum_numbers["Y"])
    print()

    print("Higgs expansion around the vev:")
    for relation in broken.higgs_expansion:
        print(" ", relation)
    print()

    print("Charged electroweak mixing:")
    for relation in broken.charged_mixing:
        print(" ", relation)
    print()

    print("Neutral electroweak mixing:")
    for relation in broken.neutral_mixing:
        print(" ", relation)
    print()

    print("Tree-level masses from the broken Higgs sector:")
    print(" MW =", broken.masses.mw)
    print(" MZ =", broken.masses.mz)
    print(" Mh =", broken.masses.mh)
    print(" photon mass =", broken.masses.photon)
    print(" me =", broken.masses.fermions[0][1])
    print()

    print("Physical gauge couplings:")
    print(" e =", broken.gauge_couplings.electric_charge)
    print(" g_WWA =", broken.gauge_couplings.g_ww_a)
    print(" g_WWZ =", broken.gauge_couplings.g_ww_z)
    print()

    print("Higgs self-couplings from the potential:")
    print(" lambda_hhh vertex coefficient =", broken.higgs_potential.hhh_vertex_coefficient)
    print(" lambda_hhhh vertex coefficient =", broken.higgs_potential.hhhh_vertex_coefficient)
    print()

    print("Broken-phase W two-point vertex Γ(W-, W+):")
    print(L.feynman_rule(broken.fields.charged_w.bar, broken.fields.charged_w))
    print()

    print("Broken-phase Z two-point vertex Γ(Z, Z):")
    print(L.feynman_rule(broken.fields.z_boson, broken.fields.z_boson))
    print()

    print("Higgs-sector photon mass term present?")
    print(_has_higgs_sector_photon_mass_term())
    print()

    print("Broken-phase h W W vertex Γ(h, W-, W+):")
    print(L.feynman_rule(broken.fields.higgs, broken.fields.charged_w.bar, broken.fields.charged_w))
    print()

    print("Physical-basis gauge vertex Γ(W-, W+, A):")
    print(L.feynman_rule(broken.fields.charged_w.bar, broken.fields.charged_w, broken.fields.photon))
    print()

    print("Physical-basis gauge vertex Γ(W-, W+, A, Z):")
    print(L.feynman_rule(
        broken.fields.charged_w.bar,
        broken.fields.charged_w,
        broken.fields.photon,
        broken.fields.z_boson,
    ))
    print()

    print("Gauge-fixed Goldstone/Z bilinear Γ(G0, Z):")
    print(L.feynman_rule(broken.fields.goldstone_neutral, broken.fields.z_boson))
    print()

    print("Gauge-fixed charged ghost bilinear Γ(cWbar, cW):")
    print(L.feynman_rule(broken.fields.ghost_charged.bar, broken.fields.ghost_charged))
    print()

    print("Broken-phase h e e vertex Γ(ebar, e, h):")
    print(L.feynman_rule(electron.bar, electron, broken.fields.higgs))
