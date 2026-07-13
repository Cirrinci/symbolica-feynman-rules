"""Evanescent operators of the SMEFT Green basis (Tables 4-9).

Two- and four-fermion evanescent operators are defined with ordered gamma-matrix
chains ``gamma^{mu1...mun} = gamma^{mu1} ... gamma^{mun}`` (no antisymmetrization
and no four-dimensional reduction).  Ordered chains are preserved because the
whole basis is built with explicit projectors and compiled with
``simplify_gamma=False``.  Charge-conjugation structures (Tables 8-9) are
attempted at the end; the genuine engine blocker is documented in
``checklist.md``.
"""

from __future__ import annotations

from symbolica import Expression

from symbolic.spenso_structures import gauge_generator
from symbolic.vertex_engine import I

from .registry import operator
from .sm_core import HALF, occ
from .tensors import (
    Poly,
    b_field_strength,
    charge_conjugation,
    covariant_derivative_adjoint,
    covariant_derivative_doublet,
    covariant_derivative_fermion,
    eps2,
    fermion_fn,
    fresh,
    g_field_strength,
    gamma,
    gamma_chain,
    htilde,
    levi,
    partial,
    pauli,
    prod,
    projector,
    sigma,
    w_field_strength,
)


def _has_colour(field):
    return field.index_kind_count("color_fund") > 0


def _has_weak(field):
    return field.index_kind_count("weak_fund") > 0


# ---------------------------------------------------------------------------
# Explicit (dual) field strengths -- kept fully explicit so a monomial that
# also carries bare gauge fields (from an expanded covariant derivative) is not
# rejected by the engine's field-strength gauge-singlet check.
# ---------------------------------------------------------------------------


def _explicit_fs(core, group, mu, nu, adj):
    if group == "SU3C":
        return g_field_strength(core, mu, nu, adj)
    if group == "SU2L":
        return w_field_strength(core, mu, nu, adj)
    return b_field_strength(core, mu, nu)


def _explicit_dual_fs(core, group, mu, nu, adj):
    a, b = fresh("mu"), fresh("mu")
    return prod(HALF * levi(mu, nu, a, b)) * _explicit_fs(core, group, a, b, adj)


def _dnu_fs(core, group, mu, nu, adj, dual=False):
    """``D^nu X_{mu nu}`` / ``D^nu Xtilde_{mu nu}`` as an explicit ``Poly``."""
    build = _explicit_dual_fs if dual else _explicit_fs
    if group == "U1Y":
        return partial(build(core, group, mu, nu, adj), nu)
    return covariant_derivative_adjoint(
        core, lambda a: build(core, group, mu, nu, a), nu, group=group
    )(adj)


def _drho_fs(core, group, mu, nu, rho, adj, dual=False):
    """``D_rho X_{mu nu}`` / ``D_rho Xtilde_{mu nu}`` (open ``mu, nu, rho``)."""
    build = _explicit_dual_fs if dual else _explicit_fs
    if group == "U1Y":
        return partial(build(core, group, mu, nu, adj), rho)
    return covariant_derivative_adjoint(
        core, lambda a: build(core, group, mu, nu, a), rho, group=group
    )(adj)


# ---------------------------------------------------------------------------
# psi^2 X H evanescent (Table 4): the dipole operators with the *dual* field
# strength.  Structurally identical to the physical dipoles of Table 2 with
# X_{mu nu} -> Xtilde_{mu nu}.
# ---------------------------------------------------------------------------


def _higgs_doublet(core, w, use_tilde):
    if use_tilde:
        return htilde(core, w)
    return prod(core.fields.H(w))


def _evan_dipole(core, C, i, j, bar_field, field, group, *, octet=False,
                 triplet=False, use_tilde=False):
    """``(psibar [T^A] sigma^{mu nu} psi) [sigma^I] H(tilde) Xtilde_{mu nu}``."""
    mu, nu = fresh("mu"), fresh("mu")
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    w = fresh("w")
    factors = [sigma(s0, s1, mu, nu), projector(field, s1, s2)]
    bar_labels = {"sp": s0, "f": i, "w": w}
    field_labels = {"sp": s2, "f": j}
    adj = None
    if octet:
        adj = fresh("ac")
        c, cp = fresh("c"), fresh("c")
        factors.append(gauge_generator(adj, c, cp))
        bar_labels["c"], field_labels["c"] = c, cp
    elif _has_colour(field):
        c = fresh("c")
        bar_labels["c"], field_labels["c"] = c, c
    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    current = prod(*factors)

    if triplet:
        adj = fresh("aw")
        ws = fresh("w")
        higgs = prod(pauli(adj, w, ws)) * _higgs_doublet(core, ws, use_tilde)
        strength = _explicit_dual_fs(core, "SU2L", mu, nu, adj)
    else:
        higgs = _higgs_doublet(core, w, use_tilde)
        strength = _explicit_dual_fs(core, group, mu, nu, adj)
    return prod(C) * current * higgs * strength


@operator("EuG", "E_{uG}", "two_fermion", "evanescent", 4, n_flavour=2)
def _euG(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.u, "SU3C",
                        octet=True, use_tilde=True)


@operator("EuW", "E_{uW}", "two_fermion", "evanescent", 4, n_flavour=2)
def _euW(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.u, "SU2L",
                        triplet=True, use_tilde=True)


@operator("EuB", "E_{uB}", "two_fermion", "evanescent", 4, n_flavour=2)
def _euB(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.u, "U1Y",
                        use_tilde=True)


@operator("EdG", "E_{dG}", "two_fermion", "evanescent", 4, n_flavour=2)
def _edG(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.d, "SU3C",
                        octet=True)


@operator("EdW", "E_{dW}", "two_fermion", "evanescent", 4, n_flavour=2)
def _edW(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.d, "SU2L",
                        triplet=True)


@operator("EdB", "E_{dB}", "two_fermion", "evanescent", 4, n_flavour=2)
def _edB(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.q, core.fields.d, "U1Y")


@operator("EeW", "E_{eW}", "two_fermion", "evanescent", 4, n_flavour=2)
def _eeW(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.l, core.fields.e, "SU2L",
                        triplet=True)


@operator("EeB", "E_{eB}", "two_fermion", "evanescent", 4, n_flavour=2)
def _eeB(core, C, flav):
    i, j = flav
    return _evan_dipole(core, C, i, j, core.fields.l, core.fields.e, "U1Y")


# ---------------------------------------------------------------------------
# psi^2 X D evanescent (Table 4)
# ---------------------------------------------------------------------------


def _current_plumbing(field, group):
    """Return ``(gen_factors, adj, wbar, wfld, cbar, cfld)`` for one bilinear."""
    weak, colour = _has_weak(field), _has_colour(field)
    gen = []
    adj = None
    wb = wf = cb = cf = None
    if group == "SU3C":
        adj = fresh("ac")
        cb, cf = fresh("c"), fresh("c")
        gen.append(gauge_generator(adj, cb, cf))
        if weak:
            wb = wf = fresh("w")
    elif group == "SU2L":
        adj = fresh("aw")
        wb, wf = fresh("w"), fresh("w")
        gen.append(pauli(adj, wb, wf))
        if colour:
            cb = cf = fresh("c")
    else:
        if weak:
            wb = wf = fresh("w")
        if colour:
            cb = cf = fresh("c")
    return gen, adj, wb, wf, cb, cf


def _evan_E(core, C, i, j, field, group):
    """``(psibar Gamma (sigma^{mu nu} gamma^rho + gamma^rho sigma^{mu nu}) psi)
    D_rho Xtilde_{mu nu}`` (EGf / EWf / EBf)."""
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    gen, adj, wb, wf, cb, cf = _current_plumbing(field, group)
    s0, sm, s2, s3 = fresh("sp"), fresh("sp"), fresh("sp"), fresh("sp")
    bar_labels = {"sp": s0, "f": i}
    fld_labels = {"sp": s3, "f": j}
    if wb is not None:
        bar_labels["w"], fld_labels["w"] = wb, wf
    if cb is not None:
        bar_labels["c"], fld_labels["c"] = cb, cf
    bar = occ(field, conjugated=True, **bar_labels)
    fld = occ(field, **fld_labels)

    def chain(factors):
        return prod(*factors, *gen, projector(field, s2, s3), bar) * prod(fld)

    bilinear = chain([sigma(s0, sm, mu, nu), gamma(sm, s2, rho)]) + chain(
        [gamma(s0, sm, rho), sigma(sm, s2, mu, nu)]
    )
    return prod(C) * bilinear * _drho_fs(core, group, mu, nu, rho, adj, dual=True)


def _evan_Eprime(core, C, i, j, field, group, *, dual):
    """``i psibar (Gamma sigma^{mu nu} /D - <-/D sigma^{mu nu} Gamma) psi X_{mu nu}``."""
    mu, nu, rho = fresh("mu"), fresh("mu"), fresh("mu")
    Y = field.quantum_numbers["Y"]
    weak, colour = _has_weak(field), _has_colour(field)
    gen, adj, wb, wf, cb, cf = _current_plumbing(field, group)
    s0, sm, s2, s3 = fresh("sp"), fresh("sp"), fresh("sp"), fresh("sp")

    barfn = fermion_fn(core, field, s0, i, conjugated=True)
    fldfn = fermion_fn(core, field, s3, j)
    d_fld = covariant_derivative_fermion(
        core, fldfn, rho, hypercharge=Y, weak=weak, colour=colour
    )
    d_bar = covariant_derivative_fermion(
        core, barfn, rho, hypercharge=Y, weak=weak, colour=colour, conjugated=True
    )
    # term1: i qbar Gamma sigma^{mu nu} gamma^rho (D_rho psi)
    chain1 = prod(sigma(s0, sm, mu, nu), gamma(sm, s2, rho), *gen,
                  projector(field, s2, s3))
    term1 = prod(I) * barfn(wb, cb) * chain1 * d_fld(wf, cf)
    # term2: -i (D_rho qbar) gamma^rho sigma^{mu nu} Gamma psi
    chain2 = prod(gamma(s0, sm, rho), sigma(sm, s2, mu, nu), *gen,
                  projector(field, s2, s3))
    term2 = prod(-I) * d_bar(wb, cb) * chain2 * fldfn(wf, cf)

    strength = (
        _explicit_dual_fs(core, group, mu, nu, adj)
        if dual
        else _explicit_fs(core, group, mu, nu, adj)
    )
    return prod(C) * (term1 + term2) * strength


def _register_psi2XD_evan(field_name, tag, groups):
    field_attr = field_name
    for grp, gletter in groups:
        # E (sigma gamma + gamma sigma) x D Xtilde
        @operator(f"E{gletter}{tag}", f"E_{{{gletter}{tag}}}", "two_fermion",
                  "evanescent", 4, n_flavour=2)
        def _e(core, C, flav, _g=grp, _f=field_attr):
            i, j = flav
            return _evan_E(core, C, i, j, getattr(core.fields, _f), _g)

        # E' (sigma /D - <-/D sigma) x X
        @operator(f"Ep{gletter}{tag}", f"E'_{{{gletter}{tag}}}", "two_fermion",
                  "evanescent", 4, n_flavour=2)
        def _ep(core, C, flav, _g=grp, _f=field_attr):
            i, j = flav
            return _evan_Eprime(core, C, i, j, getattr(core.fields, _f), _g, dual=False)

        # E'~ (sigma /D - <-/D sigma) x Xtilde
        @operator(f"Ep{gletter}t{tag}", f"E'_{{{gletter}tilde {tag}}}",
                  "two_fermion", "evanescent", 4, n_flavour=2)
        def _ept(core, C, flav, _g=grp, _f=field_attr):
            i, j = flav
            return _evan_Eprime(core, C, i, j, getattr(core.fields, _f), _g, dual=True)


# quark doublet: gluon, W, B ; up/down singlets: gluon, B ; lepton doublet: W, B ;
# charged-lepton singlet: B.
_register_psi2XD_evan("q", "q", [("SU3C", "G"), ("SU2L", "W"), ("U1Y", "B")])
_register_psi2XD_evan("u", "u", [("SU3C", "G"), ("U1Y", "B")])
_register_psi2XD_evan("d", "d", [("SU3C", "G"), ("U1Y", "B")])
_register_psi2XD_evan("l", "l", [("SU2L", "W"), ("U1Y", "B")])
_register_psi2XD_evan("e", "e", [("U1Y", "B")])


# ---------------------------------------------------------------------------
# psi^2 H D^2 evanescent (Table 4): EuH, EdH, EeH
# ---------------------------------------------------------------------------


def _evan_psi2HD2(core, C, i, j, bar_field, field, *, use_tilde):
    """``(psibar sigma^{mu nu} D_rho psi) D_sigma H(tilde) eps^{mu nu rho sigma}``."""
    mu, nu, rho, sg = fresh("mu"), fresh("mu"), fresh("mu"), fresh("mu")
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    w = fresh("w")
    c = fresh("c") if _has_colour(field) else None
    bar_labels = {"sp": s0, "f": i, "w": w}
    if c is not None:
        bar_labels["c"] = c
    bar = occ(bar_field, conjugated=True, **bar_labels)
    fldfn = fermion_fn(core, field, s2, j)
    d_fld = covariant_derivative_fermion(
        core, fldfn, rho, hypercharge=field.quantum_numbers["Y"], weak=False,
        colour=_has_colour(field),
    )
    bilinear = prod(sigma(s0, s1, mu, nu), projector(field, s1, s2), bar) * d_fld(None, c)

    if use_tilde:
        hfn, Yh = (lambda r: htilde(core, r)), -HALF
    else:
        hfn, Yh = (lambda r: Poly.of(core.fields.H(r))), HALF
    d_h = covariant_derivative_doublet(core, hfn, sg, hypercharge=Yh)(w)
    return prod(C, levi(mu, nu, rho, sg)) * bilinear * d_h


@operator("EuH", "E_{uH}", "two_fermion", "evanescent", 4, n_flavour=2)
def _euH(core, C, flav):
    i, j = flav
    return _evan_psi2HD2(core, C, i, j, core.fields.q, core.fields.u, use_tilde=True)


@operator("EdH", "E_{dH}", "two_fermion", "evanescent", 4, n_flavour=2)
def _edH(core, C, flav):
    i, j = flav
    return _evan_psi2HD2(core, C, i, j, core.fields.q, core.fields.d, use_tilde=False)


@operator("EeH", "E_{eH}", "two_fermion", "evanescent", 4, n_flavour=2)
def _eeH(core, C, flav):
    i, j = flav
    return _evan_psi2HD2(core, C, i, j, core.fields.l, core.fields.e, use_tilde=False)


# ===========================================================================
# Four-fermion evanescent operators (Tables 5-7, no charge conjugation)
# ===========================================================================
#
# Each operator is a product of two Dirac bilinears whose Lorentz structure is
# an *ordered* gamma chain ``gamma^{mu1...mun} = gamma^{mu1} ... gamma^{mun}``
# (n = 0, 1, 2, 3) with no antisymmetrisation.  The two bilinears share the
# open Lorentz indices ``mu1...mun``.  Colour-octet operators share one adjoint
# index through ``T^A`` insertions; isospin-triplet operators share one adjoint
# index through ``sigma^I``.  Chirality is fixed by one projector per bilinear
# next to its unbarred fermion.  ``simplify_gamma=False`` keeps the chains
# intact, which is the entire point of the evanescent basis.


def _lorentz(n):
    return [fresh("mu") for _ in range(n)]


def _spin_chain(s0, s2, lor, proj_field):
    if lor:
        sm = fresh("sp")
        return [gamma_chain(s0, sm, *lor), projector(proj_field, sm, s2)]
    return [projector(proj_field, s0, s2)]


def _one_bilinear(core, barfield, field, fi, fj, lor, cadj, wadj, *,
                  wbar=None, wfld=None):
    """``(barfield_i gamma^{lor} [T^A] [sigma^I] field_j)`` with explicit links."""
    s0, s2 = fresh("sp"), fresh("sp")
    factors = _spin_chain(s0, s2, lor, field)
    bl = {"sp": s0, "f": fi}
    fl = {"sp": s2, "f": fj}

    bw, fw = _has_weak(barfield), _has_weak(field)
    if wadj is not None:  # isospin triplet: sigma^I between the two doublets
        wb = wbar if wbar is not None else fresh("w")
        wf = wfld if wfld is not None else fresh("w")
        factors.append(pauli(wadj, wb, wf))
        bl["w"], fl["w"] = wb, wf
    elif bw and fw:  # both doublets: internal weak singlet
        w = wbar if wbar is not None else (wfld if wfld is not None else fresh("w"))
        bl["w"], fl["w"] = w, w
    else:  # at most one doublet: leave its index open for cross-linking
        if bw:
            bl["w"] = wbar if wbar is not None else fresh("w")
        if fw:
            fl["w"] = wfld if wfld is not None else fresh("w")

    bc, fc = _has_colour(barfield), _has_colour(field)
    if cadj is not None:  # colour octet: T^A between the two triplets
        c1, c2 = fresh("c"), fresh("c")
        factors.append(gauge_generator(cadj, c1, c2))
        if bc:
            bl["c"] = c1
        if fc:
            fl["c"] = c2
    elif bc and fc:
        c = fresh("c")
        bl["c"], fl["c"] = c, c
    else:
        if bc:
            bl["c"] = fresh("c")
        if fc:
            fl["c"] = fresh("c")

    factors.append(occ(barfield, conjugated=True, **bl))
    factors.append(occ(field, **fl))
    return prod(*factors)


def _ff_direct(core, C, flav, A, B, n, color="singlet", weak="singlet"):
    """``(Abar_i Gamma A_j)(Bbar_k Gamma B_l)`` sharing the Lorentz chain."""
    i, j, k, l = flav
    lor = _lorentz(n)
    cadj = fresh("ac") if color == "octet" else None
    wadj = fresh("aw") if weak == "triplet" else None
    b1 = _one_bilinear(core, A, A, i, j, lor, cadj, wadj)
    b2 = _one_bilinear(core, B, B, k, l, lor, cadj, wadj)
    return prod(C) * b1 * b2


def _ff_crossed(core, C, flav, A, B, n, color="singlet", weak="singlet"):
    """``(Abar_i Gamma B_j)(Bbar_k Gamma A_l)`` (the ``LR RL`` topology)."""
    i, j, k, l = flav
    lor = _lorentz(n)
    cadj = fresh("ac") if color == "octet" else None
    wadj = fresh("aw") if weak == "triplet" else None
    aw, bw = _has_weak(A), _has_weak(B)
    link = fresh("w") if (wadj is None and aw != bw) else None
    # bil1 = (Abar, B): open weak on whichever of Abar/B is the lone doublet
    kw1 = {}
    if link is not None:
        if aw and not bw:
            kw1["wbar"] = link
        else:
            kw1["wfld"] = link
    b1 = _one_bilinear(core, A, B, i, j, lor, cadj, wadj, **kw1)
    # bil2 = (Bbar, A): the matching open weak index links back
    kw2 = {}
    if link is not None:
        if aw and not bw:
            kw2["wfld"] = link
        else:
            kw2["wbar"] = link
    b2 = _one_bilinear(core, B, A, k, l, lor, cadj, wadj, **kw2)
    return prod(C) * b1 * b2


def _ff_eps(core, C, flav, barA, f1, barC, f2, n, color="singlet"):
    """``(barA_r Gamma f1) eps_{rs} (barC_s Gamma f2)`` (the ``LR LR`` topology)."""
    i, j, k, l = flav
    lor = _lorentz(n)
    cadj = fresh("ac") if color == "octet" else None
    r, s = fresh("w"), fresh("w")
    b1 = _one_bilinear(core, barA, f1, i, j, lor, cadj, None, wbar=r)
    b2 = _one_bilinear(core, barC, f2, k, l, lor, cadj, None, wbar=s)
    return prod(C, eps2(r, s)) * b1 * b2


def _ff_ledq(core, C, flav, n):
    """``(lbar_r Gamma e)(dbar Gamma q_r)`` (semileptonic ``ledq`` topology)."""
    i, j, k, l = flav
    lep, e, d, q = core.fields.l, core.fields.e, core.fields.d, core.fields.q
    lor = _lorentz(n)
    r = fresh("w")
    b1 = _one_bilinear(core, lep, e, i, j, lor, None, None, wbar=r)
    b2 = _one_bilinear(core, d, q, k, l, lor, None, None, wfld=r)
    return prod(C) * b1 * b2


# -- registration helpers ---------------------------------------------------


def _reg_direct(name, label, table, A, B, n, color="singlet", weak="singlet"):
    @operator(name, label, "four_fermion", "evanescent", table, n_flavour=4)
    def _b(core, C, flav, _A=A, _B=B, _n=n, _c=color, _w=weak):
        return _ff_direct(core, C, flav, getattr(core.fields, _A),
                          getattr(core.fields, _B), _n, _c, _w)


def _reg_crossed(name, label, table, A, B, n, color="singlet", weak="singlet"):
    @operator(name, label, "four_fermion", "evanescent", table, n_flavour=4)
    def _b(core, C, flav, _A=A, _B=B, _n=n, _c=color, _w=weak):
        return _ff_crossed(core, C, flav, getattr(core.fields, _A),
                           getattr(core.fields, _B), _n, _c, _w)


def _reg_eps(name, label, table, barA, f1, barC, f2, n, color="singlet"):
    @operator(name, label, "four_fermion", "evanescent", table, n_flavour=4)
    def _b(core, C, flav, _bA=barA, _f1=f1, _bC=barC, _f2=f2, _n=n, _c=color):
        return _ff_eps(core, C, flav, getattr(core.fields, _bA),
                       getattr(core.fields, _f1), getattr(core.fields, _bC),
                       getattr(core.fields, _f2), _n, _c)


# ---------------------------------------------------------------------------
# Table 5: four-quark
# ---------------------------------------------------------------------------

# LRbar RLbar  (scalar / tensor, crossed)
_reg_crossed("Equ", "E_{qu}", 5, "q", "u", 0)
_reg_crossed("E8qu", "E^{(8)}_{qu}", 5, "q", "u", 0, "octet")
_reg_crossed("Eqd", "E_{qd}", 5, "q", "d", 0)
_reg_crossed("E8qd", "E^{(8)}_{qd}", 5, "q", "d", 0, "octet")
_reg_crossed("E2qu", "E^{[2]}_{qu}", 5, "q", "u", 2)
_reg_crossed("E2_8qu", "E^{[2](8)}_{qu}", 5, "q", "u", 2, "octet")
_reg_crossed("E2qd", "E^{[2]}_{qd}", 5, "q", "d", 2)
_reg_crossed("E2_8qd", "E^{[2](8)}_{qd}", 5, "q", "d", 2, "octet")

# RRbar RRbar
_reg_direct("E8uu", "E^{(8)}_{uu}", 5, "u", "u", 1, "octet")
_reg_direct("E3uu", "E^{[3]}_{uu}", 5, "u", "u", 3)
_reg_direct("E3_8uu", "E^{[3](8)}_{uu}", 5, "u", "u", 3, "octet")
_reg_direct("E8dd", "E^{(8)}_{dd}", 5, "d", "d", 1, "octet")
_reg_direct("E3dd", "E^{[3]}_{dd}", 5, "d", "d", 3)
_reg_direct("E3_8dd", "E^{[3](8)}_{dd}", 5, "d", "d", 3, "octet")
_reg_crossed("Eud", "E_{ud}", 5, "u", "d", 1)
_reg_crossed("E8ud", "E^{(8)}_{ud}", 5, "u", "d", 1, "octet")
_reg_crossed("E3ud", "E^{[3]}_{ud}", 5, "u", "d", 3)
_reg_crossed("E3_8ud", "E^{[3](8)}_{ud}", 5, "u", "d", 3, "octet")
_reg_direct("Ep3ud", "E'^{[3]}_{ud}", 5, "u", "d", 3)
_reg_direct("Ep3_8ud", "E'^{[3](8)}_{ud}", 5, "u", "d", 3, "octet")

# LLbar RRbar
_reg_direct("E3qu", "E^{[3]}_{qu}", 5, "q", "u", 3)
_reg_direct("E3_8qu", "E^{[3](8)}_{qu}", 5, "q", "u", 3, "octet")
_reg_direct("E3qd", "E^{[3]}_{qd}", 5, "q", "d", 3)
_reg_direct("E3_8qd", "E^{[3](8)}_{qd}", 5, "q", "d", 3, "octet")

# LLbar LLbar
_reg_direct("E8qq", "E^{(8)}_{qq}", 5, "q", "q", 1, "octet")
_reg_direct("E38qq", "E^{(3,8)}_{qq}", 5, "q", "q", 1, "octet", "triplet")
_reg_direct("E3_1qq", "E^{[3](1)}_{qq}", 5, "q", "q", 3)
_reg_direct("E3_3qq", "E^{[3](3)}_{qq}", 5, "q", "q", 3, "singlet", "triplet")
_reg_direct("E3_8qq", "E^{[3](8)}_{qq}", 5, "q", "q", 3, "octet")
_reg_direct("E3_38qq", "E^{[3](3,8)}_{qq}", 5, "q", "q", 3, "octet", "triplet")

# LRbar LRbar (eps)
_reg_eps("E2quqd", "E^{[2]}_{quqd}", 5, "q", "u", "q", "d", 2)
_reg_eps("E2_8quqd", "E^{[2](8)}_{quqd}", 5, "q", "u", "q", "d", 2, "octet")


# ---------------------------------------------------------------------------
# Table 6: semileptonic
# ---------------------------------------------------------------------------

# LRbar RLbar / RRbar RRbar (crossed)
_reg_crossed("Elu", "E_{lu}", 6, "l", "u", 0)
_reg_crossed("Eld", "E_{ld}", 6, "l", "d", 0)
_reg_crossed("Eqe", "E_{qe}", 6, "q", "e", 0)
_reg_crossed("Eeu", "E_{eu}", 6, "e", "u", 1)
_reg_crossed("Eed", "E_{ed}", 6, "e", "d", 1)
_reg_crossed("E3eu", "E^{[3]}_{eu}", 6, "e", "u", 3)
_reg_crossed("E3ed", "E^{[3]}_{ed}", 6, "e", "d", 3)
_reg_crossed("E2lu", "E^{[2]}_{lu}", 6, "l", "u", 2)
_reg_crossed("E2ld", "E^{[2]}_{ld}", 6, "l", "d", 2)
_reg_crossed("E2qe", "E^{[2]}_{qe}", 6, "q", "e", 2)

# LLbar RRbar (direct)
_reg_direct("E3lu", "E^{[3]}_{lu}", 6, "l", "u", 3)
_reg_direct("E3ld", "E^{[3]}_{ld}", 6, "l", "d", 3)
_reg_direct("E3qe", "E^{[3]}_{qe}", 6, "q", "e", 3)
_reg_direct("Ep3eu", "E'^{[3]}_{eu}", 6, "e", "u", 3)
_reg_direct("Ep3ed", "E'^{[3]}_{ed}", 6, "e", "d", 3)

# LLbar LLbar (crossed l-q)
_reg_crossed("Elq", "E_{lq}", 6, "l", "q", 1)
_reg_crossed("E3lq_1", "E^{(3)}_{lq}", 6, "l", "q", 1, "singlet", "triplet")
_reg_crossed("E3lq", "E^{[3]}_{lq}", 6, "l", "q", 3)
_reg_crossed("E3_3lq", "E^{[3](3)}_{lq}", 6, "l", "q", 3, "singlet", "triplet")
_reg_direct("Ep3lq", "E'^{[3]}_{lq}", 6, "l", "q", 3)
_reg_direct("Ep3_3lq", "E'^{[3](3)}_{lq}", 6, "l", "q", 3, "singlet", "triplet")


# ledq tensor
@operator("E2ledq", "E^{[2]}_{ledq}", "four_fermion", "evanescent", 6, n_flavour=4)
def _e2ledq(core, C, flav):
    return _ff_ledq(core, C, flav, 2)


# LRbar LRbar (eps) semileptonic
_reg_eps("E2lequ", "E^{[2]}_{lequ}", 6, "l", "e", "q", "u", 2)
_reg_eps("Eluqe", "E_{luqe}", 6, "l", "u", "q", "e", 0)
_reg_eps("E2luqe", "E^{[2]}_{luqe}", 6, "l", "u", "q", "e", 2)


# ---------------------------------------------------------------------------
# Table 7: leptonic
# ---------------------------------------------------------------------------

_reg_direct("E3ee", "E^{[3]}_{ee}", 7, "e", "e", 3)
_reg_direct("E1_3ll", "E^{(3)}_{ll}", 7, "l", "l", 1, "singlet", "triplet")
_reg_direct("E3le", "E^{[3]}_{le}", 7, "l", "e", 3)
_reg_direct("E3ll", "E^{[3]}_{ll}", 7, "l", "l", 3)
_reg_direct("E3_3ll", "E^{[3](3)}_{ll}", 7, "l", "l", 3, "singlet", "triplet")


# ===========================================================================
# Charge-conjugation four-fermion evanescent operators (Tables 8-9)
# ===========================================================================
#
# These operators contain charge-conjugated spinors, e.g. the bilinear
# ``(psi^c-bar Gamma chi) = psi^T C Gamma chi`` (two *unconjugated* endpoints
# joined by the charge-conjugation matrix ``C``) and its partner
# ``(psibar Gamma chi^c) = psibar Gamma C chibar^T`` (two *conjugated*
# endpoints).  The declared structure is built exactly as written, so each
# operator can be inspected with ``op.structure(core)``.
#
# BLOCKER: the engine's local fermion-flow lowering requires every closed Dirac
# chain to have exactly one conjugated and one unconjugated endpoint
# (``_unsupported_local_fermion_ordering_error`` in ``feynpy/lowering.py``).
# A ``C``-joined pair has two endpoints of the *same* conjugation, so these
# operators cannot yet be compiled to Feynman rules.  They are therefore
# registered with ``status="blocked"``; ``op.structure(core)`` works, but
# ``op.lagrangian(core)`` raises.  Supporting them needs a genuine (non-trivial)
# engine extension for charge-conjugation fermion flow; see ``checklist.md``.


def _cc_unbarred_bilinear(core, psiA, psiB, fi, fj, lor, *, wA=None, wB=None,
                          cA=None, cB=None):
    """``(psiA^c-bar Gamma psiB) = psiA^T C gamma^{lor} P psiB`` (both unbarred)."""
    s0, sc, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    factors = [charge_conjugation(s0, sc)]
    if lor:
        sm = fresh("sp")
        factors += [gamma_chain(sc, sm, *lor), projector(psiB, sm, s2)]
    else:
        factors.append(projector(psiB, sc, s2))
    la = {"sp": s0, "f": fi}
    lb = {"sp": s2, "f": fj}
    if _has_weak(psiA):
        la["w"] = wA if wA is not None else fresh("w")
    if _has_weak(psiB):
        lb["w"] = wB if wB is not None else fresh("w")
    if _has_colour(psiA):
        la["c"] = cA if cA is not None else fresh("c")
    if _has_colour(psiB):
        lb["c"] = cB if cB is not None else fresh("c")
    factors.append(occ(psiA, **la))
    factors.append(occ(psiB, **lb))
    return prod(*factors)


def _cc_barred_bilinear(core, psiA, psiB, fi, fj, lor, *, wA=None, wB=None,
                        cA=None, cB=None):
    """``(psiAbar Gamma psiB^c) = psiAbar gamma^{lor} P C psiBbar^T`` (both barred)."""
    s0, sc, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    if lor:
        sm = fresh("sp")
        chain = [gamma_chain(s0, sm, *lor), projector(psiB, sm, sc)]
    else:
        chain = [projector(psiB, s0, sc)]
    factors = chain + [charge_conjugation(sc, s2)]
    la = {"sp": s0, "f": fi}
    lb = {"sp": s2, "f": fj}
    if _has_weak(psiA):
        la["w"] = wA if wA is not None else fresh("w")
    if _has_weak(psiB):
        lb["w"] = wB if wB is not None else fresh("w")
    if _has_colour(psiA):
        la["c"] = cA if cA is not None else fresh("c")
    if _has_colour(psiB):
        lb["c"] = cB if cB is not None else fresh("c")
    factors.append(occ(psiA, conjugated=True, **la))
    factors.append(occ(psiB, conjugated=True, **lb))
    return prod(*factors)


def _cc_operator(core, C, flav, field, n, *, colour=True):
    """``(psi^c-bar Gamma psi)(psibar Gamma psi^c)`` with the ``c``-indices linked."""
    i, j, k, l = flav
    lor = _lorentz(n)
    ca = fresh("c") if colour and _has_colour(field) else None
    cb = fresh("c") if colour and _has_colour(field) else None
    wa = fresh("w") if _has_weak(field) else None
    wb = fresh("w") if _has_weak(field) else None
    # bil1 = (psi^c-bar_i Gamma psi_j): c-field carries (wa, ca), psi carries (wb, cb)
    b1 = _cc_unbarred_bilinear(core, field, field, i, j, lor, wA=wa, wB=wb,
                               cA=ca, cB=cb)
    # bil2 = (psibar_k Gamma psi^c_l): psibar carries (wb, cb), c-field carries (wa, ca)
    b2 = _cc_barred_bilinear(core, field, field, k, l, lor, wA=wb, wB=wa,
                             cA=cb, cB=ca)
    return prod(C) * b1 * b2


_CC_NOTE = (
    "Charge-conjugation C-chain: two same-conjugation Dirac endpoints joined by "
    "the charge-conjugation matrix. Declared structure builds and is inspectable "
    "via op.structure(core); compilation is blocked by the engine's local "
    "fermion-flow lowering (one conjugated + one unconjugated endpoint required)."
)


def _reg_cc(name, label, table, field, n):
    @operator(name, label, "four_fermion", "evanescent", table, n_flavour=4,
              status="blocked", note=_CC_NOTE)
    def _b(core, C, flav, _f=field, _n=n):
        return _cc_operator(core, C, flav, getattr(core.fields, _f), _n)


# Representative charge-conjugation operators (Table 8: quarks; Table 9:
# leptons) spanning the scalar / vector / tensor structures.  The full Table
# 8-9 enumeration and the shared blocker are recorded in checklist.md.
_reg_cc("Ecuu", "E^{c}_{uu}", 8, "u", 0)
_reg_cc("Ecdd", "E^{c}_{dd}", 8, "d", 0)
_reg_cc("Ecqq", "E^{c}_{qq}", 8, "q", 0)
_reg_cc("Ec2uu", "E^{c[2]}_{uu}", 8, "u", 2)
_reg_cc("Ec2dd", "E^{c[2]}_{dd}", 8, "d", 2)
_reg_cc("Ecee", "E^{c}_{ee}", 9, "e", 0)
_reg_cc("Ecll", "E^{c}_{ll}", 9, "l", 0)
_reg_cc("Ec2ee", "E^{c[2]}_{ee}", 9, "e", 2)
