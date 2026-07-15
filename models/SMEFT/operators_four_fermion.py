"""Four-fermion operators of the SMEFT Green basis (Table 3).

Baryon- and lepton-number conserving four-fermion physical operators
(four-quark, four-lepton and semileptonic) are registered here.  Colour-octet
operators share one adjoint index ``A`` between the two currents through
``T^A`` insertions; isospin-triplet operators share one adjoint index ``I``
through ``sigma^I``; the ``epsilon_{rs}`` structures contract two barred
doublets.  Chirality is explicit via one projector per bilinear.

The scalar/tensor ``LR`` operators (``O_{quqd}``, ``O_{lequ}``, ``O_{ledq}``)
are non-Hermitian and enter the Lagrangian together with their Hermitian
conjugate; the builders implement the Table 3 entry itself.
"""

from __future__ import annotations

from symbolica import Expression, S

from symbolic.spenso_structures import gauge_generator

from .registry import operator
from .sm_core import occ
from .tensors import (
    Poly,
    eps2,
    fresh,
    gamma,
    pauli,
    prod,
    projector,
    sigma,
)


# ---------------------------------------------------------------------------
# Bilinear building blocks
# ---------------------------------------------------------------------------


def _has_colour(field):
    return field.index_kind_count("color_fund") > 0


def _has_weak(field):
    return field.index_kind_count("weak_fund") > 0


def _vector_bilinear(core, mu, bar_field, field, fi, fj, *, colour_adjoint=None,
                     weak_adjoint=None):
    """``(psibar_i gamma^mu [T^A] [sigma^I] psi_j)`` for one same-type bilinear."""
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    factors = [gamma(s0, s1, mu), projector(field, s1, s2)]
    bar_labels = {"sp": s0, "f": fi}
    field_labels = {"sp": s2, "f": fj}
    if _has_weak(field):
        if weak_adjoint is not None:
            wb, wf = fresh("w"), fresh("w")
            factors.append(pauli(weak_adjoint, wb, wf))
            bar_labels["w"], field_labels["w"] = wb, wf
        else:
            w = fresh("w")
            bar_labels["w"], field_labels["w"] = w, w
    if colour_adjoint is not None:
        c, cp = fresh("c"), fresh("c")
        factors.append(gauge_generator(colour_adjoint, c, cp))
        bar_labels["c"], field_labels["c"] = c, cp
    elif _has_colour(field):
        c = fresh("c")
        bar_labels["c"], field_labels["c"] = c, c
    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    return prod(*factors)


def _vv(core, C, flav, barA, fA, barB, fB, *, octet=False, triplet=False):
    """Product of two vector currents (Table 3 vector operators)."""
    i, j, k, l = flav
    mu = fresh("mu")
    colour_adjoint = fresh("ac") if octet else None
    weak_adjoint = fresh("aw") if triplet else None
    b1 = _vector_bilinear(core, mu, barA, fA, i, j, colour_adjoint=colour_adjoint,
                          weak_adjoint=weak_adjoint)
    b2 = _vector_bilinear(core, mu, barB, fB, k, l, colour_adjoint=colour_adjoint,
                          weak_adjoint=weak_adjoint)
    return prod(C) * b1 * b2


# ---------------------------------------------------------------------------
# Four-quark
# ---------------------------------------------------------------------------


@operator("Oqq1", "O_{qq}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqq1(core, C, flav):
    q = core.fields.q
    return _vv(core, C, flav, q, q, q, q)


@operator("Oqq3", "O_{qq}^{(3)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqq3(core, C, flav):
    q = core.fields.q
    return _vv(core, C, flav, q, q, q, q, triplet=True)


@operator("Ouu", "O_{uu}", "four_fermion", "physical", 3, n_flavour=4)
def _ouu(core, C, flav):
    u = core.fields.u
    return _vv(core, C, flav, u, u, u, u)


@operator("Odd", "O_{dd}", "four_fermion", "physical", 3, n_flavour=4)
def _odd(core, C, flav):
    d = core.fields.d
    return _vv(core, C, flav, d, d, d, d)


@operator("Oud1", "O_{ud}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _oud1(core, C, flav):
    u, d = core.fields.u, core.fields.d
    return _vv(core, C, flav, u, u, d, d)


@operator("Oud8", "O_{ud}^{(8)}", "four_fermion", "physical", 3, n_flavour=4)
def _oud8(core, C, flav):
    u, d = core.fields.u, core.fields.d
    return _vv(core, C, flav, u, u, d, d, octet=True)


@operator("Oqu1", "O_{qu}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqu1(core, C, flav):
    q, u = core.fields.q, core.fields.u
    return _vv(core, C, flav, q, q, u, u)


@operator("Oqu8", "O_{qu}^{(8)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqu8(core, C, flav):
    q, u = core.fields.q, core.fields.u
    return _vv(core, C, flav, q, q, u, u, octet=True)


@operator("Oqd1", "O_{qd}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqd1(core, C, flav):
    q, d = core.fields.q, core.fields.d
    return _vv(core, C, flav, q, q, d, d)


@operator("Oqd8", "O_{qd}^{(8)}", "four_fermion", "physical", 3, n_flavour=4)
def _oqd8(core, C, flav):
    q, d = core.fields.q, core.fields.d
    return _vv(core, C, flav, q, q, d, d, octet=True)


# ---------------------------------------------------------------------------
# Four-lepton
# ---------------------------------------------------------------------------


@operator("Oll", "O_{ll}", "four_fermion", "physical", 3, n_flavour=4)
def _oll(core, C, flav):
    l = core.fields.l
    return _vv(core, C, flav, l, l, l, l)


@operator("Oee", "O_{ee}", "four_fermion", "physical", 3, n_flavour=4)
def _oee(core, C, flav):
    e = core.fields.e
    return _vv(core, C, flav, e, e, e, e)


@operator("Ole", "O_{le}", "four_fermion", "physical", 3, n_flavour=4)
def _ole(core, C, flav):
    l, e = core.fields.l, core.fields.e
    return _vv(core, C, flav, l, l, e, e)


# ---------------------------------------------------------------------------
# Semileptonic (vector)
# ---------------------------------------------------------------------------


@operator("Olq1", "O_{lq}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _olq1(core, C, flav):
    l, q = core.fields.l, core.fields.q
    return _vv(core, C, flav, l, l, q, q)


@operator("Olq3", "O_{lq}^{(3)}", "four_fermion", "physical", 3, n_flavour=4)
def _olq3(core, C, flav):
    l, q = core.fields.l, core.fields.q
    return _vv(core, C, flav, l, l, q, q, triplet=True)


@operator("Oeu", "O_{eu}", "four_fermion", "physical", 3, n_flavour=4)
def _oeu(core, C, flav):
    e, u = core.fields.e, core.fields.u
    return _vv(core, C, flav, e, e, u, u)


@operator("Oed", "O_{ed}", "four_fermion", "physical", 3, n_flavour=4)
def _oed(core, C, flav):
    e, d = core.fields.e, core.fields.d
    return _vv(core, C, flav, e, e, d, d)


@operator("Oqe", "O_{qe}", "four_fermion", "physical", 3, n_flavour=4)
def _oqe(core, C, flav):
    q, e = core.fields.q, core.fields.e
    return _vv(core, C, flav, q, q, e, e)


@operator("Olu", "O_{lu}", "four_fermion", "physical", 3, n_flavour=4)
def _olu(core, C, flav):
    l, u = core.fields.l, core.fields.u
    return _vv(core, C, flav, l, l, u, u)


@operator("Old", "O_{ld}", "four_fermion", "physical", 3, n_flavour=4)
def _old(core, C, flav):
    l, d = core.fields.l, core.fields.d
    return _vv(core, C, flav, l, l, d, d)


# ---------------------------------------------------------------------------
# Semileptonic / four-quark scalar & tensor (LR) operators (+ h.c.)
# ---------------------------------------------------------------------------


def _scalar_bilinear(core, bar_field, field, fi, fj, *, wbar=None, wfield=None,
                     colour_label=None, colour_adjoint=None):
    """``(psibar_i [T^A] psi_j)`` scalar bilinear with optional open doublet index."""
    s0, s2 = fresh("sp"), fresh("sp")
    factors = [projector(field, s0, s2)]
    bar_labels = {"sp": s0, "f": fi}
    field_labels = {"sp": s2, "f": fj}
    if _has_weak(bar_field):
        bar_labels["w"] = wbar
    if _has_weak(field):
        field_labels["w"] = wfield
    if colour_adjoint is not None:
        c, cp = fresh("c"), fresh("c")
        factors.append(gauge_generator(colour_adjoint, c, cp))
        bar_labels["c"], field_labels["c"] = c, cp
    elif _has_colour(field) and _has_colour(bar_field):
        c = colour_label or fresh("c")
        bar_labels["c"], field_labels["c"] = c, c
    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    return prod(*factors)


def _tensor_bilinear(core, mu, nu, bar_field, field, fi, fj, *, wbar=None, wfield=None):
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    factors = [sigma(s0, s1, mu, nu), projector(field, s1, s2)]
    bar_labels = {"sp": s0, "f": fi}
    field_labels = {"sp": s2, "f": fj}
    if _has_weak(bar_field):
        bar_labels["w"] = wbar
    if _has_weak(field):
        field_labels["w"] = wfield
    if _has_colour(field) and _has_colour(bar_field):
        c = fresh("c")
        bar_labels["c"], field_labels["c"] = c, c
    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    return prod(*factors)


@operator("Oquqd1", "O_{quqd}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _oquqd1(core, C, flav):
    """``(qbar^r u) eps_{rs} (qbar^s d)``."""
    i, j, k, l = flav
    q, u, d = core.fields.q, core.fields.u, core.fields.d
    r, s = fresh("w"), fresh("w")
    b1 = _scalar_bilinear(core, q, u, i, j, wbar=r)
    b2 = _scalar_bilinear(core, q, d, k, l, wbar=s)
    return prod(C, eps2(r, s)) * b1 * b2


@operator("Oquqd8", "O_{quqd}^{(8)}", "four_fermion", "physical", 3, n_flavour=4)
def _oquqd8(core, C, flav):
    """``(qbar^r T^A u) eps_{rs} (qbar^s T^A d)``."""
    i, j, k, l = flav
    q, u, d = core.fields.q, core.fields.u, core.fields.d
    r, s = fresh("w"), fresh("w")
    A = fresh("ac")
    b1 = _scalar_bilinear(core, q, u, i, j, wbar=r, colour_adjoint=A)
    b2 = _scalar_bilinear(core, q, d, k, l, wbar=s, colour_adjoint=A)
    return prod(C, eps2(r, s)) * b1 * b2


@operator("Oledq", "O_{ledq}", "four_fermion", "physical", 3, n_flavour=4)
def _oledq(core, C, flav):
    """``(lbar_r e)(dbar q_r)`` (doublet indices of lbar and q contracted)."""
    i, j, k, l = flav
    lep, e, d, q = core.fields.l, core.fields.e, core.fields.d, core.fields.q
    r = fresh("w")
    b1 = _scalar_bilinear(core, lep, e, i, j, wbar=r)
    b2 = _scalar_bilinear(core, d, q, k, l, wfield=r)
    return prod(C) * b1 * b2


@operator("Olequ1", "O_{lequ}^{(1)}", "four_fermion", "physical", 3, n_flavour=4)
def _olequ1(core, C, flav):
    """``(lbar^r e) eps_{rs} (qbar^s u)``."""
    i, j, k, l = flav
    lep, e, q, u = core.fields.l, core.fields.e, core.fields.q, core.fields.u
    r, s = fresh("w"), fresh("w")
    b1 = _scalar_bilinear(core, lep, e, i, j, wbar=r)
    b2 = _scalar_bilinear(core, q, u, k, l, wbar=s)
    return prod(C, eps2(r, s)) * b1 * b2


@operator("Olequ3", "O_{lequ}^{(3)}", "four_fermion", "physical", 3, n_flavour=4)
def _olequ3(core, C, flav):
    """``(lbar^r sigma^{mu nu} e) eps_{rs} (qbar^s sigma^{mu nu} u)``."""
    i, j, k, l = flav
    lep, e, q, u = core.fields.l, core.fields.e, core.fields.q, core.fields.u
    r, s = fresh("w"), fresh("w")
    mu, nu = fresh("mu"), fresh("mu")
    b1 = _tensor_bilinear(core, mu, nu, lep, e, i, j, wbar=r)
    b2 = _tensor_bilinear(core, mu, nu, q, u, k, l, wbar=s)
    return prod(C, eps2(r, s)) * b1 * b2
