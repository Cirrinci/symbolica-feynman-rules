"""Reusable low-level builders for tests."""

from fractions import Fraction

from symbolica import Expression, S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    CovD,
    Field,
    FieldStrength,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    GhostField,
    LORENTZ_INDEX,
    SPINOR_INDEX,
)
from symbolic.spenso_structures import gauge_generator, structure_constant
from symbolic.vertex_engine import I


def canon(expr):
    return expr.expand().to_canonical_string()


def make_photon(*, name="A", symbol=None):
    if symbol is None:
        symbol = S(name)
    return Field(name, spin=1, self_conjugate=True, symbol=symbol, indices=(LORENTZ_INDEX,))


def make_gluon(*, name="G", symbol=None):
    if symbol is None:
        symbol = S(name)
    return Field(
        name,
        spin=1,
        self_conjugate=True,
        symbol=symbol,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )


def make_ghost(*, name="ghG", ghost_of=None, symbol=None, conjugate_symbol=None):
    if symbol is None:
        symbol = S(name)
    if conjugate_symbol is None:
        conjugate_symbol = S(f"{name}bar")
    return GhostField(
        name,
        ghost_of=ghost_of,
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=(COLOR_ADJ_INDEX,),
    )


def make_dirac_fermion(
    name: str,
    *,
    symbol=None,
    conjugate_symbol=None,
    color=False,
    charge=None,
):
    if symbol is None:
        symbol = S(name.lower())
    if conjugate_symbol is None:
        conjugate_symbol = S(f"{name.lower()}bar")
    indices = (SPINOR_INDEX, COLOR_FUND_INDEX) if color else (SPINOR_INDEX,)
    quantum_numbers = {"Q": charge} if charge is not None else {}
    return Field(
        name,
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=indices,
        quantum_numbers=quantum_numbers,
    )


def make_complex_scalar(
    name: str,
    *,
    symbol=None,
    conjugate_symbol=None,
    color=False,
    charge=None,
):
    if symbol is None:
        symbol = S(name.lower())
    if conjugate_symbol is None:
        conjugate_symbol = S(f"{name.lower()}bar")
    indices = (COLOR_FUND_INDEX,) if color else ()
    quantum_numbers = {"Q": charge} if charge is not None else {}
    return Field(
        name,
        spin=0,
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=indices,
        quantum_numbers=quantum_numbers,
    )


def make_u1(coupling, gauge_boson_sym, *, name="U1", charge="Q"):
    return GaugeGroup(
        name=name,
        abelian=True,
        coupling=coupling,
        gauge_boson=gauge_boson_sym,
        charge=charge,
    )


def make_su3(coupling, gauge_boson_sym, *, ghost_sym=None, name="SU3"):
    return GaugeGroup(
        name=name,
        abelian=False,
        coupling=coupling,
        gauge_boson=gauge_boson_sym,
        ghost_field=ghost_sym,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )


def dirac_covd_decl(field, *, mu=None):
    if mu is None:
        mu = S("mu_decl")
    return I * field.bar * Gamma(mu) * CovD(field, mu)


def scalar_covd_decl(field, *, mu=None):
    if mu is None:
        mu = S("mu_decl")
    return CovD(field.bar, mu) * CovD(field, mu)


def gauge_kinetic_decl(group, *, mu=None, nu=None, adjoint=None):
    if mu is None or nu is None:
        mu, nu = S("mu_decl", "nu_decl")
    prefactor = -(Expression.num(1) / Expression.num(4))
    if getattr(group, "abelian", False):
        return prefactor * FieldStrength(group, mu, nu) * FieldStrength(group, mu, nu)
    if adjoint is None:
        adjoint = S("a_decl")
    return (
        prefactor
        * FieldStrength(group, mu, nu, adjoint)
        * FieldStrength(group, mu, nu, adjoint)
    )
