"""Bosonic operators of the SMEFT Green basis (Table 1).

Physical operators: the ``X^3`` triple field strengths (and their CP-odd duals),
the ``X^2 H^2`` classes, ``H^6`` and ``H^4 D^2``.  Redundant bosonic operators
(``R2G/R2W/R2B``, ``RDH``, ``R'HD``, ``R''HD``, ``RWDH``, ``RBDH``) are added in
the redundant stage.

All operators follow Appendix D exactly: field strengths from Eqs. (D.4)-(D.6),
dual tensors from Eq. (D.12), covariant derivatives from Eq. (D.3).  No EOM/IBP
reduction is applied.
"""

from __future__ import annotations

from symbolica import Expression, S

from feynpy import DC, PartialD
from symbolic.spenso_structures import structure_constant, weak_structure_constant

from .registry import operator
from .tensors import (
    Poly,
    b_field_strength,
    covariant_derivative_adjoint,
    covariant_derivative_doublet,
    covariant_derivative_higgs,
    dual_field_strength,
    eps2,
    field_strength,
    fresh,
    g_field_strength,
    higgs_lr_derivative,
    higgs_lr_derivative_isospin,
    partial,
    pauli,
    prod,
    summed,
    w_field_strength,
)


ONE = Expression.num(1)
HALF = ONE / Expression.num(2)


# ---------------------------------------------------------------------------
# Helpers local to the bosonic sector
# ---------------------------------------------------------------------------


def _higgs_norm(core, r=None):
    """``(H^dag H)`` with an optionally supplied doublet label."""
    H = core.fields.H
    if r is None:
        r = fresh("w")
    return prod(H.bar(r), H(r))


def _fs(core, group, mu, nu, adj=None):
    return field_strength(core, group, mu, nu, adj)


def _dual(core, group, mu, nu, adj=None):
    return dual_field_strength(core, group, mu, nu, adj)


# ---------------------------------------------------------------------------
# X^3
# ---------------------------------------------------------------------------


@operator("O3G", "O_{3G}", "bosonic", "physical", 1)
def _o3g(core, C, flav):
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    A, Bc, Cc = fresh("ac"), fresh("ac"), fresh("ac")
    return prod(
        C * structure_constant(A, Bc, Cc),
        _fs(core, "SU3C", mu, nu, A),
        _fs(core, "SU3C", nu, rho, Bc),
        _fs(core, "SU3C", rho, mu, Cc),
    )


@operator("O3Gtilde", "O_{3\\tilde G}", "bosonic", "physical", 1)
def _o3gt(core, C, flav):
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    A, Bc, Cc = fresh("ac"), fresh("ac"), fresh("ac")
    return prod(
        C * structure_constant(A, Bc, Cc),
        _dual(core, "SU3C", mu, nu, A),
        _fs(core, "SU3C", nu, rho, Bc),
        _fs(core, "SU3C", rho, mu, Cc),
    )


@operator("O3W", "O_{3W}", "bosonic", "physical", 1)
def _o3w(core, C, flav):
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    I_, J, K = fresh("aw"), fresh("aw"), fresh("aw")
    return prod(
        C * weak_structure_constant(I_, J, K),
        _fs(core, "SU2L", mu, nu, I_),
        _fs(core, "SU2L", nu, rho, J),
        _fs(core, "SU2L", rho, mu, K),
    )


@operator("O3Wtilde", "O_{3\\tilde W}", "bosonic", "physical", 1)
def _o3wt(core, C, flav):
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    I_, J, K = fresh("aw"), fresh("aw"), fresh("aw")
    return prod(
        C * weak_structure_constant(I_, J, K),
        _dual(core, "SU2L", mu, nu, I_),
        _fs(core, "SU2L", nu, rho, J),
        _fs(core, "SU2L", rho, mu, K),
    )


# ---------------------------------------------------------------------------
# X^2 H^2
# ---------------------------------------------------------------------------


def _x2h2(core, C, group, dual=False, adjoint=True):
    mu, nu = fresh("mu"), fresh("mu")
    adj = fresh("ac" if group == "SU3C" else "aw") if adjoint else None
    first = _dual(core, group, mu, nu, adj) if dual else _fs(core, group, mu, nu, adj)
    second = _fs(core, group, mu, nu, adj)
    return prod(C, first, second) * _higgs_norm(core)


@operator("OHG", "O_{HG}", "bosonic", "physical", 1)
def _ohg(core, C, flav):
    return _x2h2(core, C, "SU3C")


@operator("OHGtilde", "O_{H\\tilde G}", "bosonic", "physical", 1)
def _ohgt(core, C, flav):
    return _x2h2(core, C, "SU3C", dual=True)


@operator("OHW", "O_{HW}", "bosonic", "physical", 1)
def _ohw(core, C, flav):
    return _x2h2(core, C, "SU2L")


@operator("OHWtilde", "O_{H\\tilde W}", "bosonic", "physical", 1)
def _ohwt(core, C, flav):
    return _x2h2(core, C, "SU2L", dual=True)


@operator("OHB", "O_{HB}", "bosonic", "physical", 1)
def _ohb(core, C, flav):
    return _x2h2(core, C, "U1Y", adjoint=False)


@operator("OHBtilde", "O_{H\\tilde B}", "bosonic", "physical", 1)
def _ohbt(core, C, flav):
    return _x2h2(core, C, "U1Y", dual=True, adjoint=False)


def _hwb(core, C, dual=False):
    """``W^I_{mu nu} B^{mu nu} (H^dag sigma^I H)`` (optionally with dual W)."""
    mu, nu = fresh("mu"), fresh("mu")
    I_ = fresh("aw")
    r, s = fresh("w"), fresh("w")
    W = _dual(core, "SU2L", mu, nu, I_) if dual else _fs(core, "SU2L", mu, nu, I_)
    H = core.fields.H
    higgs_triplet = prod(H.bar(r), pauli(I_, r, s), H(s))
    return prod(C, W, _fs(core, "U1Y", mu, nu)) * higgs_triplet


@operator("OHWB", "O_{HWB}", "bosonic", "physical", 1)
def _ohwb(core, C, flav):
    return _hwb(core, C)


@operator("OHWBtilde", "O_{H\\tilde WB}", "bosonic", "physical", 1)
def _ohwbt(core, C, flav):
    return _hwb(core, C, dual=True)


# ---------------------------------------------------------------------------
# H^6
# ---------------------------------------------------------------------------


@operator("OH", "O_H", "bosonic", "physical", 1)
def _oh(core, C, flav):
    return prod(C) * _higgs_norm(core) * _higgs_norm(core) * _higgs_norm(core)


# ---------------------------------------------------------------------------
# H^4 D^2
# ---------------------------------------------------------------------------


@operator("OHbox", "O_{H\\Box}", "bosonic", "physical", 1)
def _ohbox(core, C, flav):
    """``(H^dag H) Box(H^dag H)`` with ``Box = d_mu d^mu`` (H^dag H is a singlet)."""
    H = core.fields.H
    mu = fresh("mu")
    r = fresh("w")
    # Box(H^dag H) via the Leibniz rule for the ordinary derivative.
    box = summed(
        prod(PartialD(PartialD(H.bar(r), mu), mu), H(r)),
        prod(Expression.num(2), PartialD(H.bar(r), mu), PartialD(H(r), mu)),
        prod(H.bar(r), PartialD(PartialD(H(r), mu), mu)),
    )
    return prod(C) * _higgs_norm(core) * box


@operator("OHD", "O_{HD}", "bosonic", "physical", 1)
def _ohd(core, C, flav):
    """``(H^dag D_mu H)^dag (H^dag D^mu H)``.

    Written with explicit doublet labels via
    :func:`covariant_derivative_higgs` to avoid any ambiguity in the weak
    contraction of the four Higgs doublets.
    """
    H = core.fields.H
    mu = fresh("mu")
    r, s = fresh("w"), fresh("w")
    # (H^dag D_mu H) = H^dag_r (D_mu H)_r
    current = prod(H.bar(r)) * covariant_derivative_higgs(core, mu, r)
    # (H^dag D_mu H)^dag = (D_mu H)^dag_s H_s
    current_dag = covariant_derivative_higgs(core, mu, s, conjugated=True) * prod(H(s))
    return prod(C) * current_dag * current


# ---------------------------------------------------------------------------
# Redundant bosonic operators (Table 1)
# ---------------------------------------------------------------------------


def _higgs_fn(core):
    H = core.fields.H
    return lambda r: prod(H(r))


def _higgs_bar_fn(core):
    H = core.fields.H
    return lambda r: prod(H.bar(r))


def _box_higgs(core, r, *, conjugated=False):
    """``(D_mu D^mu H)_r`` (or its conjugate) as a Poly."""
    Y = core.fields.H.quantum_numbers["Y"]
    mu = fresh("mu")
    base_fn = _higgs_bar_fn(core) if conjugated else _higgs_fn(core)
    d1 = covariant_derivative_doublet(core, base_fn, mu, hypercharge=Y, conjugated=conjugated)
    d2 = covariant_derivative_doublet(core, d1, mu, hypercharge=Y, conjugated=conjugated)
    return d2(r)


@operator("RDH", "R_{DH}", "bosonic", "redundant", 1)
def _rdh(core, C, flav):
    """``(D_mu D^mu H)^dag (D_nu D^nu H)`` (iterated covariant derivative)."""
    r = fresh("w")
    box_h = _box_higgs(core, r)
    box_h_dag = _box_higgs(core, r, conjugated=True)
    return prod(C) * box_h_dag * box_h


@operator("RpHD", "R'_{HD}", "bosonic", "redundant", 1)
def _rphd(core, C, flav):
    """``(H^dag H)(D_mu H)^dag (D^mu H)``."""
    H = core.fields.H
    mu = fresh("mu")
    r, a = fresh("w"), fresh("w")
    dh = covariant_derivative_higgs(core, mu, r)
    dh_dag = covariant_derivative_higgs(core, mu, r, conjugated=True)
    return prod(C, H.bar(a), H(a)) * dh_dag * dh


@operator("RppHD", "R''_{HD}", "bosonic", "redundant", 1)
def _rpphd(core, C, flav):
    """``(H^dag H) d_mu(H^dag i<->D^mu H)`` (the outer derivative is d_mu)."""
    H = core.fields.H
    mu = fresh("mu")
    a = fresh("w")
    current = higgs_lr_derivative(core, mu)
    return prod(C, H.bar(a), H(a)) * partial(current, mu)


@operator("R2G", "R_{2G}", "bosonic", "redundant", 1)
def _r2g(core, C, flav):
    """``-1/2 (D_mu G^{A mu nu})(D_rho G^{A rho nu})``."""
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    A = fresh("ac")
    g_fn = lambda a: g_field_strength(core, mu, nu, a)
    dg = covariant_derivative_adjoint(core, g_fn, mu, group="SU3C")(A)
    g_fn2 = lambda a: g_field_strength(core, rho, nu, a)
    dg2 = covariant_derivative_adjoint(core, g_fn2, rho, group="SU3C")(A)
    return prod(-C / 2) * dg * dg2


@operator("R2W", "R_{2W}", "bosonic", "redundant", 1)
def _r2w(core, C, flav):
    """``-1/2 (D_mu W^{I mu nu})(D_rho W^{I rho nu})``."""
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    I_ = fresh("aw")
    w_fn = lambda a: w_field_strength(core, mu, nu, a)
    dw = covariant_derivative_adjoint(core, w_fn, mu, group="SU2L")(I_)
    w_fn2 = lambda a: w_field_strength(core, rho, nu, a)
    dw2 = covariant_derivative_adjoint(core, w_fn2, rho, group="SU2L")(I_)
    return prod(-C / 2) * dw * dw2


@operator("R2B", "R_{2B}", "bosonic", "redundant", 1)
def _r2b(core, C, flav):
    """``-1/2 (d_mu B^{mu nu})(d_rho B^{rho nu})``."""
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    db = partial(b_field_strength(core, mu, nu), mu)
    db2 = partial(b_field_strength(core, rho, nu), rho)
    return prod(-C / 2) * db * db2


@operator("RWDH", "R_{WDH}", "bosonic", "redundant", 1)
def _rwdh(core, C, flav):
    """``D_nu W^{I mu nu} (H^dag i<->D^I_mu H)``."""
    mu, nu = fresh("mu"), fresh("mu")
    I_ = fresh("aw")
    w_fn = lambda a: w_field_strength(core, mu, nu, a)
    dw = covariant_derivative_adjoint(core, w_fn, nu, group="SU2L")(I_)
    current = higgs_lr_derivative_isospin(core, mu, I_)
    return prod(C) * dw * current


@operator("RBDH", "R_{BDH}", "bosonic", "redundant", 1)
def _rbdh(core, C, flav):
    """``d_nu B^{mu nu} (H^dag i<->D_mu H)``."""
    mu, nu = fresh("mu"), fresh("mu")
    db = partial(b_field_strength(core, mu, nu), nu)
    current = higgs_lr_derivative(core, mu)
    return prod(C) * db * current
