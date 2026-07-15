"""Two-fermion operators of the SMEFT Green basis (Table 2 and Table 4).

Physical operators (currents ``psi2DH2``, dipoles ``psi2XH``, Yukawa-like
``psi2H3``) are registered here; redundant (Table 2) and evanescent (Table 4)
two-fermion operators are added in their respective stages.

Chirality is explicit: ``q, l`` are left-handed, ``u, d, e`` right-handed, so a
single chiral projector next to the non-barred fermion fixes each bilinear.
Non-Hermitian operators (all dipoles, the ``psi2H3`` Yukawa-like operators and
``OHud``) appear in the Lagrangian together with their Hermitian conjugate; the
builders below implement the operator exactly as written in Table 2 (its
Wilson coefficient is complex), matching the table entries one to one.
"""

from __future__ import annotations

from symbolica import Expression, S

from symbolic.spenso_structures import gauge_generator

from .registry import operator
from .sm_core import HALF, occ
from .tensors import (
    Poly,
    b_field_strength,
    covariant_derivative_adjoint,
    covariant_derivative_doublet,
    covariant_derivative_fermion,
    covariant_derivative_higgs,
    dual_field_strength,
    eps2,
    fermion_fn,
    field_strength,
    fresh,
    g_field_strength,
    gamma,
    levi,
    higgs_lr_derivative,
    higgs_lr_derivative_isospin,
    htilde,
    partial,
    pauli,
    prod,
    projector,
    sigma,
    w_field_strength,
)
from symbolic.vertex_engine import I


# ---------------------------------------------------------------------------
# psi^2 D H^2 : fermionic currents times a Higgs current
# ---------------------------------------------------------------------------


def _vector_current(core, bar_field, field, i, j, mu, *, colour=True, weak_singlet=True,
                    triplet_index=None):
    """Build ``(psibar_i gamma^mu [sigma^I] psi_j)`` with correct chirality."""
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    factors = [gamma(s0, s1, mu), projector(field, s1, s2)]
    bar_labels = {"sp": s0, "f": i}
    field_labels = {"sp": s2, "f": j}
    if colour:
        c = fresh("c")
        bar_labels["c"] = c
        field_labels["c"] = c
    if triplet_index is not None:
        wb, wf = fresh("w"), fresh("w")
        factors.append(pauli(triplet_index, wb, wf))
        bar_labels["w"] = wb
        field_labels["w"] = wf
    elif weak_singlet and bar_field.index_kind_count("weak_fund"):
        w = fresh("w")
        bar_labels["w"] = w
        field_labels["w"] = w
    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    return prod(*factors)


def _psi2DH2_singlet(core, C, i, j, bar_field, field, *, colour):
    mu = fresh("mu")
    current = _vector_current(core, bar_field, field, i, j, mu, colour=colour)
    return prod(C) * current * higgs_lr_derivative(core, mu)


def _psi2DH2_triplet(core, C, i, j, bar_field, field, *, colour):
    mu = fresh("mu")
    I_ = fresh("aw")
    current = _vector_current(core, bar_field, field, i, j, mu, colour=colour,
                              triplet_index=I_)
    return prod(C) * current * higgs_lr_derivative_isospin(core, mu, I_)


@operator("OHq1", "O_{Hq}^{(1)}", "two_fermion", "physical", 2, n_flavour=2)
def _ohq1(core, C, flav):
    i, j = flav
    return _psi2DH2_singlet(core, C, i, j, core.fields.q, core.fields.q, colour=True)


@operator("OHq3", "O_{Hq}^{(3)}", "two_fermion", "physical", 2, n_flavour=2)
def _ohq3(core, C, flav):
    i, j = flav
    return _psi2DH2_triplet(core, C, i, j, core.fields.q, core.fields.q, colour=True)


@operator("OHu", "O_{Hu}", "two_fermion", "physical", 2, n_flavour=2)
def _ohu(core, C, flav):
    i, j = flav
    return _psi2DH2_singlet(core, C, i, j, core.fields.u, core.fields.u, colour=True)


@operator("OHd", "O_{Hd}", "two_fermion", "physical", 2, n_flavour=2)
def _ohd(core, C, flav):
    i, j = flav
    return _psi2DH2_singlet(core, C, i, j, core.fields.d, core.fields.d, colour=True)


@operator("OHl1", "O_{Hl}^{(1)}", "two_fermion", "physical", 2, n_flavour=2)
def _ohl1(core, C, flav):
    i, j = flav
    return _psi2DH2_singlet(core, C, i, j, core.fields.l, core.fields.l, colour=False)


@operator("OHl3", "O_{Hl}^{(3)}", "two_fermion", "physical", 2, n_flavour=2)
def _ohl3(core, C, flav):
    i, j = flav
    return _psi2DH2_triplet(core, C, i, j, core.fields.l, core.fields.l, colour=False)


@operator("OHe", "O_{He}", "two_fermion", "physical", 2, n_flavour=2)
def _ohe(core, C, flav):
    i, j = flav
    return _psi2DH2_singlet(core, C, i, j, core.fields.e, core.fields.e, colour=False)


@operator("OHud", "O_{Hud}", "two_fermion", "physical", 2, n_flavour=2)
def _ohud(core, C, flav):
    """``(ubar gamma^mu d)(Htilde^dag i D_mu H)`` (+ h.c.)."""
    i, j = flav
    u, d = core.fields.u, core.fields.d
    mu = fresh("mu")
    r = fresh("w")
    current = _vector_current(core, u, d, i, j, mu, colour=True)
    higgs = prod(I) * htilde(core, r, conjugated=True) * covariant_derivative_higgs(core, mu, r)
    return prod(C) * current * higgs


# ---------------------------------------------------------------------------
# psi^2 X H : dipole operators (+ h.c.)
# ---------------------------------------------------------------------------


def _higgs_doublet(core, w, use_tilde):
    """``Htilde_w`` (up-type dipole) or ``H_w`` (down-type / lepton dipole)."""
    if use_tilde:
        return htilde(core, w)
    return prod(core.fields.H(w))


def _dipole(core, C, i, j, bar_field, field, group, *, octet=False, triplet=False,
            use_tilde=False):
    """``(psibar_i [T^A] sigma^{mu nu} psi_j) [sigma^I] H(tilde) X_{mu nu}`` (+h.c.)."""
    mu, nu = fresh("mu"), fresh("mu")
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    w = fresh("w")
    factors = [sigma(s0, s1, mu, nu), projector(field, s1, s2)]
    bar_labels = {"sp": s0, "f": i, "w": w}
    field_labels = {"sp": s2, "f": j}

    # colour: octet inserts T^A (adjoint shared with the gluon field strength);
    # otherwise a fundamental colour index is contracted directly.
    colour_adjoint = None
    if octet:
        colour_adjoint = fresh("ac")
        c, cp = fresh("c"), fresh("c")
        factors.append(gauge_generator(colour_adjoint, c, cp))
        bar_labels["c"] = c
        field_labels["c"] = cp
    elif bar_field.index_kind_count("color_fund"):
        c = fresh("c")
        bar_labels["c"] = c
        field_labels["c"] = c

    factors.append(occ(bar_field, conjugated=True, **bar_labels))
    factors.append(occ(field, **field_labels))
    current = prod(*factors)

    # weak / Higgs / field strength
    if triplet:
        I_ = fresh("aw")
        ws = fresh("w")
        higgs = prod(pauli(I_, w, ws)) * _higgs_doublet(core, ws, use_tilde)
        strength = field_strength(core, group, mu, nu, I_)
    else:
        higgs = _higgs_doublet(core, w, use_tilde)
        if group == "SU3C":
            strength = field_strength(core, group, mu, nu, colour_adjoint)
        else:
            strength = field_strength(core, group, mu, nu)

    return prod(C) * current * higgs * strength


@operator("OuG", "O_{uG}", "two_fermion", "physical", 2, n_flavour=2)
def _oug(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.u, "SU3C",
                   octet=True, use_tilde=True)


@operator("OuW", "O_{uW}", "two_fermion", "physical", 2, n_flavour=2)
def _ouw(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.u, "SU2L",
                   triplet=True, use_tilde=True)


@operator("OuB", "O_{uB}", "two_fermion", "physical", 2, n_flavour=2)
def _oub(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.u, "U1Y", use_tilde=True)


@operator("OdG", "O_{dG}", "two_fermion", "physical", 2, n_flavour=2)
def _odg(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.d, "SU3C", octet=True)


@operator("OdW", "O_{dW}", "two_fermion", "physical", 2, n_flavour=2)
def _odw(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.d, "SU2L", triplet=True)


@operator("OdB", "O_{dB}", "two_fermion", "physical", 2, n_flavour=2)
def _odb(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.q, core.fields.d, "U1Y")


@operator("OeW", "O_{eW}", "two_fermion", "physical", 2, n_flavour=2)
def _oew(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.l, core.fields.e, "SU2L", triplet=True)


@operator("OeB", "O_{eB}", "two_fermion", "physical", 2, n_flavour=2)
def _oeb(core, C, flav):
    i, j = flav
    return _dipole(core, C, i, j, core.fields.l, core.fields.e, "U1Y")


# ---------------------------------------------------------------------------
# psi^2 H^3 : Yukawa-like operators (+ h.c.)
# ---------------------------------------------------------------------------


def _scalar_bilinear(core, i, j, bar_field, field, w):
    """Scalar bilinear ``(psibar_i psi_j)`` with the doublet index ``w`` open."""
    s0, s2 = fresh("sp"), fresh("sp")
    bar_labels = {"sp": s0, "f": i, "w": w}
    field_labels = {"sp": s2, "f": j}
    if bar_field.index_kind_count("color_fund"):
        c = fresh("c")
        bar_labels["c"] = c
        field_labels["c"] = c
    return prod(
        projector(field, s0, s2),
        occ(bar_field, conjugated=True, **bar_labels),
        occ(field, **field_labels),
    )


def _psi2H3(core, C, i, j, bar_field, field, *, use_tilde):
    H = core.fields.H
    a, w = fresh("w"), fresh("w")
    higgs_norm = prod(H.bar(a), H(a))
    bilinear = _scalar_bilinear(core, i, j, bar_field, field, w)
    return prod(C) * higgs_norm * bilinear * _higgs_doublet(core, w, use_tilde)


@operator("OuH", "O_{uH}", "two_fermion", "physical", 2, n_flavour=2)
def _ouh(core, C, flav):
    i, j = flav
    return _psi2H3(core, C, i, j, core.fields.q, core.fields.u, use_tilde=True)


@operator("OdH", "O_{dH}", "two_fermion", "physical", 2, n_flavour=2)
def _odh(core, C, flav):
    i, j = flav
    return _psi2H3(core, C, i, j, core.fields.q, core.fields.d, use_tilde=False)


@operator("OeH", "O_{eH}", "two_fermion", "physical", 2, n_flavour=2)
def _oeh(core, C, flav):
    i, j = flav
    return _psi2H3(core, C, i, j, core.fields.l, core.fields.e, use_tilde=False)


# ===========================================================================
# Redundant two-fermion operators (Table 2)
# ===========================================================================
#
# All of these are built in the same explicit style, adding the general
# fermion covariant derivative (``covariant_derivative_fermion``), the
# left-right covariant derivative ``i<->D`` on a fermion current, and the
# covariant derivative of a gauge field strength (``D^nu X_{mu nu}``) and of the
# isospin-triplet Higgs current.  Nothing here reduces the operators (no EOM /
# IBP / Fierz); every covariant derivative is expanded by the Leibniz rule so
# each redundant operator produces the full contact + gauge-emission vertices.


def _has_weak(field):
    return bool(field.index_kind_count("weak_fund"))


def _has_colour(field):
    return bool(field.index_kind_count("color_fund"))


def _current_indices(field, group):
    """Return ``(mid_factors, adj, wb, wf, cb, cf)`` for a ``psibar Gamma psi``.

    Threads the weak/colour indices between the barred and unbarred fermion and
    inserts the group generator (``T^A`` octet, ``sigma^I`` triplet) so its
    adjoint index ``adj`` can be shared with an external field strength.
    """

    mid = []
    adj = None
    wb = wf = cb = cf = None
    weak, colour = _has_weak(field), _has_colour(field)
    if group == "SU3C":
        adj = fresh("ac")
        cb, cf = fresh("c"), fresh("c")
        mid.append(gauge_generator(adj, cb, cf))
        if weak:
            wb = wf = fresh("w")
    elif group == "SU2L":
        adj = fresh("aw")
        wb, wf = fresh("w"), fresh("w")
        mid.append(pauli(adj, wb, wf))
        if colour:
            cb = cf = fresh("c")
    else:  # U1Y / singlet contraction
        if weak:
            wb = wf = fresh("w")
        if colour:
            cb = cf = fresh("c")
    return mid, adj, wb, wf, cb, cf


def _plain_current(core, field, i, j, mu, group):
    """``(psibar_i Gamma^mu psi_j)`` with open Lorentz ``mu`` (and adjoint)."""
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    mid, adj, wb, wf, cb, cf = _current_indices(field, group)
    barfn = fermion_fn(core, field, s0, i, conjugated=True)
    fldfn = fermion_fn(core, field, s2, j)
    midp = prod(gamma(s0, s1, mu), *mid, projector(field, s1, s2))
    return barfn(wb, cb) * midp * fldfn(wf, cf), adj


def _lr_current(core, field, i, j, mu, nu, group):
    """``i(psibar_i Gamma^mu D^nu psi_j - (D^nu psibar_i) Gamma^mu psi_j)``.

    The antisymmetric (left-right) covariant derivative ``i<->D^nu`` acting on
    the current; setting ``nu == mu`` realises the slashed ``i<->/D``.  The
    generator's adjoint index is shared by both terms so it contracts with a
    single external field strength.
    """
    s0, s1, s2 = fresh("sp"), fresh("sp"), fresh("sp")
    mid, adj, wb, wf, cb, cf = _current_indices(field, group)
    midp = prod(gamma(s0, s1, mu), *mid, projector(field, s1, s2))
    Y = field.quantum_numbers["Y"]
    weak, colour = _has_weak(field), _has_colour(field)
    barfn = fermion_fn(core, field, s0, i, conjugated=True)
    fldfn = fermion_fn(core, field, s2, j)
    d_fld = covariant_derivative_fermion(
        core, fldfn, nu, hypercharge=Y, weak=weak, colour=colour
    )
    d_bar = covariant_derivative_fermion(
        core, barfn, nu, hypercharge=Y, weak=weak, colour=colour, conjugated=True
    )
    term1 = barfn(wb, cb) * midp * d_fld(wf, cf)
    term2 = d_bar(wb, cb) * midp * fldfn(wf, cf)
    return prod(I) * term1 - prod(I) * term2, adj


def _field_strength_dnu(core, group, mu, nu, adj):
    """``D^nu X_{mu nu}`` (covariant for SU(2)/SU(3), ``partial`` for U(1))."""
    if group == "SU3C":
        fs = lambda a: g_field_strength(core, mu, nu, a)
        return covariant_derivative_adjoint(core, fs, nu, group="SU3C")(adj)
    if group == "SU2L":
        fs = lambda a: w_field_strength(core, mu, nu, a)
        return covariant_derivative_adjoint(core, fs, nu, group="SU2L")(adj)
    return partial(b_field_strength(core, mu, nu), nu)


def _psi2XD_R(core, C, i, j, field, group):
    """``(psibar Gamma^mu psi) D^nu X_{mu nu}`` (RGf / RWf / RBf)."""
    mu, nu = fresh("mu"), fresh("mu")
    current, adj = _plain_current(core, field, i, j, mu, group)
    return prod(C) * current * _field_strength_dnu(core, group, mu, nu, adj)


def _explicit_field_strength(core, group, mu, nu, adj):
    """Field strength as an explicit ``Poly`` (no engine ``FieldStrengthFactor``).

    Needed whenever the operator's current already contains bare gauge fields
    (from an expanded ``i<->D``): the engine's field-strength gauge-singlet
    check counts only field-strength / generator / structure-constant adjoint
    labels, not those carried by bare gauge fields, so the fully explicit form
    keeps every individual monomial acceptable while the expansion stays exact.
    """
    if group == "SU3C":
        return g_field_strength(core, mu, nu, adj)
    if group == "SU2L":
        return w_field_strength(core, mu, nu, adj)
    return b_field_strength(core, mu, nu)


def _explicit_dual_field_strength(core, group, mu, nu, adj):
    """``Xtilde_{mu nu} = 1/2 eps_{mu nu a b} X^{a b}`` from the explicit ``X``."""
    a, b = fresh("mu"), fresh("mu")
    return prod(HALF * levi(mu, nu, a, b)) * _explicit_field_strength(
        core, group, a, b, adj
    )


def _psi2XD_Rprime(core, C, i, j, field, group, *, dual=False):
    """``1/2 (psibar Gamma^mu i<->D^nu psi) X_{mu nu}`` (R'Xf, or dual R'Xtilde f)."""
    mu, nu = fresh("mu"), fresh("mu")
    current, adj = _lr_current(core, field, i, j, mu, nu, group)
    if dual:
        strength = _explicit_dual_field_strength(core, group, mu, nu, adj)
    else:
        strength = _explicit_field_strength(core, group, mu, nu, adj)
    return prod(C, HALF) * current * strength


# -- psi^2 D^3 : RqD, RuD, RdD, RlD, ReD ------------------------------------


def _cov_fermion(core, field, sp, f, *lorentz, conjugated=False):
    """``D_{l1} ... D_{ln} psi`` as a callable ``(w, c) -> Poly``.

    The rightmost Lorentz index in ``lorentz`` is the innermost derivative.
    """
    fn = fermion_fn(core, field, sp, f, conjugated=conjugated)
    Y = field.quantum_numbers["Y"]
    weak, colour = _has_weak(field), _has_colour(field)
    for mu in reversed(lorentz):
        fn = covariant_derivative_fermion(
            core, fn, mu, hypercharge=Y, weak=weak, colour=colour, conjugated=conjugated
        )
    return fn


def _psi2D3(core, C, i, j, field):
    """``i/2 psibar {D_mu D^mu, /D} psi`` with the anticommutator expanded."""
    s0, s2 = fresh("sp"), fresh("sp")
    mu, nu = fresh("mu"), fresh("mu")
    w = fresh("w") if _has_weak(field) else None
    c = fresh("c") if _has_colour(field) else None
    bar_labels = {"sp": s0, "f": i}
    if w is not None:
        bar_labels["w"] = w
    if c is not None:
        bar_labels["c"] = c
    bar = occ(field, conjugated=True, **bar_labels)
    # {D^2, /D} psi = D_mu D^mu (gamma^nu D_nu psi) + gamma^nu D_nu (D_mu D^mu psi)
    term_a = _cov_fermion(core, field, s2, j, mu, mu, nu)  # D_mu D^mu D_nu psi
    term_b = _cov_fermion(core, field, s2, j, nu, mu, mu)  # D_nu D_mu D^mu psi
    body = term_a(w, c) + term_b(w, c)
    return prod(C * I / 2, bar, gamma(s0, s2, nu)) * body


@operator("RqD", "R_{qD}", "two_fermion", "redundant", 2, n_flavour=2)
def _rqd(core, C, flav):
    i, j = flav
    return _psi2D3(core, C, i, j, core.fields.q)


@operator("RuD", "R_{uD}", "two_fermion", "redundant", 2, n_flavour=2)
def _rud(core, C, flav):
    i, j = flav
    return _psi2D3(core, C, i, j, core.fields.u)


@operator("RdD", "R_{dD}", "two_fermion", "redundant", 2, n_flavour=2)
def _rdd(core, C, flav):
    i, j = flav
    return _psi2D3(core, C, i, j, core.fields.d)


@operator("RlD", "R_{lD}", "two_fermion", "redundant", 2, n_flavour=2)
def _rld(core, C, flav):
    i, j = flav
    return _psi2D3(core, C, i, j, core.fields.l)


@operator("ReD", "R_{eD}", "two_fermion", "redundant", 2, n_flavour=2)
def _red(core, C, flav):
    i, j = flav
    return _psi2D3(core, C, i, j, core.fields.e)


# -- psi^2 X D : R / R' / R'tilde -------------------------------------------
# Quark doublet q: gluon (octet), W (triplet), B (singlet).

@operator("RGq", "R_{Gq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rgq(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.q, "SU3C")


@operator("RpGq", "R'_{Gq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "SU3C")


@operator("RpGtq", "R'_{Gtilde q}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgtq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "SU3C", dual=True)


@operator("RWq", "R_{Wq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rwq(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.q, "SU2L")


@operator("RpWq", "R'_{Wq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpwq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "SU2L")


@operator("RpWtq", "R'_{Wtilde q}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpwtq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "SU2L", dual=True)


@operator("RBq", "R_{Bq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rbq(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.q, "U1Y")


@operator("RpBq", "R'_{Bq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "U1Y")


@operator("RpBtq", "R'_{Btilde q}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbtq(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.q, "U1Y", dual=True)


# up-type singlet u: gluon (octet), B (singlet).
@operator("RGu", "R_{Gu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rgu(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.u, "SU3C")


@operator("RpGu", "R'_{Gu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgu(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.u, "SU3C")


@operator("RpGtu", "R'_{Gtilde u}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgtu(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.u, "SU3C", dual=True)


@operator("RBu", "R_{Bu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rbu(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.u, "U1Y")


@operator("RpBu", "R'_{Bu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbu(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.u, "U1Y")


@operator("RpBtu", "R'_{Btilde u}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbtu(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.u, "U1Y", dual=True)


# down-type singlet d: gluon (octet), B (singlet).
@operator("RGd", "R_{Gd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rgd(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.d, "SU3C")


@operator("RpGd", "R'_{Gd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgd(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.d, "SU3C")


@operator("RpGtd", "R'_{Gtilde d}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpgtd(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.d, "SU3C", dual=True)


@operator("RBd", "R_{Bd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rbd(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.d, "U1Y")


@operator("RpBd", "R'_{Bd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbd(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.d, "U1Y")


@operator("RpBtd", "R'_{Btilde d}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbtd(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.d, "U1Y", dual=True)


# lepton doublet l: W (triplet), B (singlet).
@operator("RWl", "R_{Wl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rwl(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.l, "SU2L")


@operator("RpWl", "R'_{Wl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpwl(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.l, "SU2L")


@operator("RpWtl", "R'_{Wtilde l}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpwtl(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.l, "SU2L", dual=True)


@operator("RBl", "R_{Bl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rbl(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.l, "U1Y")


@operator("RpBl", "R'_{Bl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbl(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.l, "U1Y")


@operator("RpBtl", "R'_{Btilde l}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbtl(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.l, "U1Y", dual=True)


# charged-lepton singlet e: B (singlet).
@operator("RBe", "R_{Be}", "two_fermion", "redundant", 2, n_flavour=2)
def _rbe(core, C, flav):
    i, j = flav
    return _psi2XD_R(core, C, i, j, core.fields.e, "U1Y")


@operator("RpBe", "R'_{Be}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbe(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.e, "U1Y")


@operator("RpBte", "R'_{Btilde e}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpbte(core, C, flav):
    i, j = flav
    return _psi2XD_Rprime(core, C, i, j, core.fields.e, "U1Y", dual=True)


# -- psi^2 H D^2 + h.c. : RfHD1..4 ------------------------------------------


def _higgs_deriv_doublet(core, use_tilde):
    """Return ``(hfn, Y)`` where ``hfn(r)`` is ``H_r`` (or ``Htilde_r``)."""
    if use_tilde:
        return (lambda r: htilde(core, r)), -HALF
    return (lambda r: Poly.of(core.fields.H(r))), HALF


def _psi2HD2(core, C, i, j, bar_field, field, variant, *, use_tilde):
    """One ``psi^2 H D^2`` operator (variant in 1..4)."""
    s0, s2 = fresh("sp"), fresh("sp")
    w = fresh("w")
    c = fresh("c") if _has_colour(field) else None
    bar_labels = {"sp": s0, "f": i, "w": w}
    if c is not None:
        bar_labels["c"] = c
    bar = occ(bar_field, conjugated=True, **bar_labels)
    fldfn = fermion_fn(core, field, s2, j)
    Yf = field.quantum_numbers["Y"]
    colour = _has_colour(field)
    hfn, _Yh = _higgs_deriv_doublet(core, use_tilde)
    d_h = covariant_derivative_doublet  # alias

    if variant == 1:
        # (psibar psi) D_mu D^mu H
        mu = fresh("mu")
        bilinear = prod(projector(field, s0, s2), bar) * fldfn(None, c)
        step = d_h(core, hfn, mu, hypercharge=_Yh)
        higgs = d_h(core, step, mu, hypercharge=_Yh)(w)
        return prod(C) * bilinear * higgs
    if variant == 2:
        # (psibar i sigma^{mu nu} D_mu psi) D_nu H
        mu, nu = fresh("mu"), fresh("mu")
        s1 = fresh("sp")
        d_fld = covariant_derivative_fermion(
            core, fldfn, mu, hypercharge=Yf, weak=False, colour=colour
        )
        bilinear = (
            prod(I, sigma(s0, s1, mu, nu), projector(field, s1, s2), bar)
            * d_fld(None, c)
        )
        higgs = d_h(core, hfn, nu, hypercharge=_Yh)(w)
        return prod(C) * bilinear * higgs
    if variant == 3:
        # (psibar D_mu D^mu psi) H
        mu = fresh("mu")
        d1 = covariant_derivative_fermion(
            core, fldfn, mu, hypercharge=Yf, weak=False, colour=colour
        )
        d2 = covariant_derivative_fermion(
            core, d1, mu, hypercharge=Yf, weak=False, colour=colour
        )
        bilinear = prod(projector(field, s0, s2), bar) * d2(None, c)
        return prod(C) * bilinear * hfn(w)
    if variant == 4:
        # (psibar D_mu psi) D^mu H
        mu = fresh("mu")
        d1 = covariant_derivative_fermion(
            core, fldfn, mu, hypercharge=Yf, weak=False, colour=colour
        )
        bilinear = prod(projector(field, s0, s2), bar) * d1(None, c)
        higgs = d_h(core, hfn, mu, hypercharge=_Yh)(w)
        return prod(C) * bilinear * higgs
    raise ValueError(variant)


def _register_psi2HD2(prefix, label, bar_name, field_name, use_tilde):
    for v in (1, 2, 3, 4):
        name = f"{prefix}{v}"

        @operator(name, f"{label}{v}", "two_fermion", "redundant", 2,
                  n_flavour=2)
        def _builder(core, C, flav, _v=v, _bar=bar_name, _fld=field_name,
                     _tilde=use_tilde):
            i, j = flav
            return _psi2HD2(
                core, C, i, j, getattr(core.fields, _bar),
                getattr(core.fields, _fld), _v, use_tilde=_tilde,
            )


_register_psi2HD2("RuHD", "R_{uHD}", "q", "u", True)
_register_psi2HD2("RdHD", "R_{dHD}", "q", "d", False)
_register_psi2HD2("ReHD", "R_{eHD}", "l", "e", False)


# -- psi^2 D H^2 (redundant) : R', R'' singlet and triplet ------------------


def _higgs_singlet_norm(core):
    a = fresh("w")
    return prod(core.fields.H.bar(a), core.fields.H(a))


def _higgs_triplet(core, adj):
    r, s = fresh("w"), fresh("w")
    return prod(core.fields.H.bar(r), pauli(adj, r, s), core.fields.H(s))


def _Rp_singlet(core, C, i, j, field):
    """``(psibar i<->/D psi)(H^dag H)``."""
    mu = fresh("mu")
    current, _ = _lr_current(core, field, i, j, mu, mu, "U1Y")
    return prod(C) * current * _higgs_singlet_norm(core)


def _Rpp_singlet(core, C, i, j, field):
    """``(psibar gamma^mu psi) d_mu(H^dag H)``."""
    mu = fresh("mu")
    current, _ = _plain_current(core, field, i, j, mu, "U1Y")
    return prod(C) * current * partial(_higgs_singlet_norm(core), mu)


def _Rp_triplet(core, C, i, j, field):
    """``(psibar i<->/D^I psi)(H^dag sigma^I H)``."""
    mu = fresh("mu")
    current, adj = _lr_current(core, field, i, j, mu, mu, "SU2L")
    return prod(C) * current * _higgs_triplet(core, adj)


def _Rpp_triplet(core, C, i, j, field):
    """``(psibar sigma^I gamma^mu psi) D_mu(H^dag sigma^I H)``."""
    mu = fresh("mu")
    current, adj = _plain_current(core, field, i, j, mu, "SU2L")
    o_fn = lambda a: _higgs_triplet(core, a)
    d_o = covariant_derivative_adjoint(core, o_fn, mu, group="SU2L")(adj)
    return prod(C) * current * d_o


@operator("Rp1Hq", "R'^{(1)}_{Hq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rp1hq(core, C, flav):
    i, j = flav
    return _Rp_singlet(core, C, i, j, core.fields.q)


@operator("Rpp1Hq", "R''^{(1)}_{Hq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpp1hq(core, C, flav):
    i, j = flav
    return _Rpp_singlet(core, C, i, j, core.fields.q)


@operator("Rp3Hq", "R'^{(3)}_{Hq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rp3hq(core, C, flav):
    i, j = flav
    return _Rp_triplet(core, C, i, j, core.fields.q)


@operator("Rpp3Hq", "R''^{(3)}_{Hq}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpp3hq(core, C, flav):
    i, j = flav
    return _Rpp_triplet(core, C, i, j, core.fields.q)


@operator("RpHu", "R'_{Hu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rphu(core, C, flav):
    i, j = flav
    return _Rp_singlet(core, C, i, j, core.fields.u)


@operator("RppHu", "R''_{Hu}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpphu(core, C, flav):
    i, j = flav
    return _Rpp_singlet(core, C, i, j, core.fields.u)


@operator("RpHd", "R'_{Hd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rphd(core, C, flav):
    i, j = flav
    return _Rp_singlet(core, C, i, j, core.fields.d)


@operator("RppHd", "R''_{Hd}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpphd(core, C, flav):
    i, j = flav
    return _Rpp_singlet(core, C, i, j, core.fields.d)


@operator("Rp1Hl", "R'^{(1)}_{Hl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rp1hl(core, C, flav):
    i, j = flav
    return _Rp_singlet(core, C, i, j, core.fields.l)


@operator("Rpp1Hl", "R''^{(1)}_{Hl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpp1hl(core, C, flav):
    i, j = flav
    return _Rpp_singlet(core, C, i, j, core.fields.l)


@operator("Rp3Hl", "R'^{(3)}_{Hl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rp3hl(core, C, flav):
    i, j = flav
    return _Rp_triplet(core, C, i, j, core.fields.l)


@operator("Rpp3Hl", "R''^{(3)}_{Hl}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpp3hl(core, C, flav):
    i, j = flav
    return _Rpp_triplet(core, C, i, j, core.fields.l)


@operator("RpHe", "R'_{He}", "two_fermion", "redundant", 2, n_flavour=2)
def _rphe(core, C, flav):
    i, j = flav
    return _Rp_singlet(core, C, i, j, core.fields.e)


@operator("RppHe", "R''_{He}", "two_fermion", "redundant", 2, n_flavour=2)
def _rpphe(core, C, flav):
    i, j = flav
    return _Rpp_singlet(core, C, i, j, core.fields.e)
