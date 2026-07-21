"""Low-level tensor helpers for writing SMEFT Green-basis operators.

Everything here stays in the *fully explicit* style: raw Spenso tensors with
hand-threaded index labels, combined with the declarative ``DC``/``FS`` factors
of the FeynPy engine.  No chiral field transformation is used, so ordered
gamma-matrix chains are preserved exactly (mandatory for the evanescent
operators of Tables 4-9).

Two ingredients make the operators readable:

* a tiny :class:`Poly` algebra over declarative monomials, so that structures
  which are naturally *sums* (chiral projectors, the antisymmetric derivative
  ``i<->D``, products of expanded covariant derivatives) can be multiplied and
  added while preserving the left-to-right field ordering that fixes the
  fermion pairing;
* physics helpers for ``sigma^{mu nu}``, dual field strengths ``Xtilde``, the
  Higgs currents ``H^dag i<->D H`` (isospin singlet and triplet), the conjugate
  doublet ``Htilde`` and charge-conjugated spinors.

Index labels are threaded explicitly.  Call :func:`fresh` for internal
(summed) labels.
"""

from __future__ import annotations

from itertools import count
from typing import Iterable

from symbolica import Expression, S

from feynpy import DC, FS, PartialD
from feynpy.declared import (
    CovariantDerivativeFactor,
    DifferentiatedCovariantFactor,
    DifferentiatedOperatorFactor,
    PartialDerivativeFactor,
    _DeclaredMonomial,
    _FieldFactor,
    _coerce_decl_factor,
)
from feynpy.lagrangian import DeclaredLagrangian
from symbolic.spenso_structures import (
    chiral_projector_left,
    chiral_projector_right,
    dirac_charge_conjugation,
    gamma5_matrix,
    gamma_matrix,
    gauge_generator,
    lorentz_levi_civita,
    sigma_tensor,
    spinor_metric,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I


_FRESH = count()


def fresh(prefix: str = "x") -> Expression:
    """Return a fresh unique index label (a Symbolica symbol)."""
    return S(f"{prefix}_smeft_{next(_FRESH)}")


# ---------------------------------------------------------------------------
# Poly: distribute products/sums of declarative monomials
# ---------------------------------------------------------------------------


def as_monomials(item) -> tuple[_DeclaredMonomial, ...]:
    """Coerce a factor / field / monomial / lagrangian / scalar into monomials."""

    if isinstance(item, Poly):
        return item.terms
    if isinstance(item, _DeclaredMonomial):
        return (item,)
    if isinstance(item, DeclaredLagrangian):
        return item.source_terms
    factor = _coerce_decl_factor(item)
    if factor is not None:
        return (_DeclaredMonomial.from_factor(factor),)
    # Numeric or symbolic scalar coefficient.
    return (_DeclaredMonomial(coefficient=item, factors=()),)


class Poly:
    """A sum of declarative monomials with order-preserving multiplication."""

    __slots__ = ("terms",)

    def __init__(self, terms: Iterable[_DeclaredMonomial] = ()):
        self.terms = tuple(terms)

    @classmethod
    def of(cls, item) -> "Poly":
        return cls(as_monomials(item))

    def __mul__(self, other) -> "Poly":
        other_terms = as_monomials(other)
        return Poly(a * b for a in self.terms for b in other_terms)

    def __rmul__(self, other) -> "Poly":
        other_terms = as_monomials(other)
        return Poly(b * a for b in other_terms for a in self.terms)

    def __add__(self, other) -> "Poly":
        return Poly(self.terms + as_monomials(other))

    def __radd__(self, other) -> "Poly":
        if other == 0:
            return self
        return Poly(as_monomials(other) + self.terms)

    def __sub__(self, other) -> "Poly":
        return Poly(self.terms + tuple(-m for m in as_monomials(other)))

    def __neg__(self) -> "Poly":
        return Poly(-m for m in self.terms)

    def declared(self) -> DeclaredLagrangian:
        return DeclaredLagrangian(source_terms=self.terms)


def prod(*parts) -> Poly:
    """Ordered product of the given parts, distributing every internal sum."""

    result = Poly((_DeclaredMonomial(coefficient=1, factors=()),))
    for part in parts:
        result = result * part
    return result


def summed(*parts) -> Poly:
    total = Poly(())
    for part in parts:
        total = total + Poly.of(part)
    return total


# ---------------------------------------------------------------------------
# Chiral projectors and spinor structures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ordinary derivative (Leibniz) and generic covariant derivatives
# ---------------------------------------------------------------------------


def _differentiate_factor(factor, mu):
    """Return d_mu of one field-like declarative factor, or None."""

    if isinstance(factor, _FieldFactor):
        return PartialDerivativeFactor(
            field=factor.field,
            lorentz_indices=(mu,),
            conjugated=factor.conjugated,
            labels=dict(factor.labels),
        )
    if isinstance(factor, CovariantDerivativeFactor):
        return DifferentiatedOperatorFactor(
            operand=factor,
            lorentz_indices=(mu,),
        )
    if isinstance(factor, DifferentiatedCovariantFactor):
        return DifferentiatedOperatorFactor(
            operand=factor.covariant_factor,
            lorentz_indices=factor.lorentz_indices + (mu,),
        )
    if isinstance(factor, DifferentiatedOperatorFactor):
        return DifferentiatedOperatorFactor(
            operand=factor.operand,
            lorentz_indices=factor.lorentz_indices + (mu,),
        )
    if isinstance(factor, PartialDerivativeFactor):
        return PartialDerivativeFactor(
            field=factor.field,
            lorentz_indices=factor.lorentz_indices + (mu,),
            conjugated=factor.conjugated,
            labels=dict(factor.labels),
        )
    return None


def partial(item, mu) -> Poly:
    """Ordinary derivative ``d_mu`` acting by the Leibniz rule on field factors.

    Non-field factors (gamma matrices, generators, metrics, numeric/symbolic
    coefficients) are spacetime-independent and are left untouched.  Products of
    fields are differentiated term by term.
    """

    poly = item if isinstance(item, Poly) else Poly.of(item)
    out: list[_DeclaredMonomial] = []
    for monomial in poly.terms:
        factors = monomial.factors
        for position, factor in enumerate(factors):
            derivative = _differentiate_factor(factor, mu)
            if derivative is None:
                continue
            new_factors = factors[:position] + (derivative,) + factors[position + 1 :]
            out.append(
                _DeclaredMonomial(coefficient=monomial.coefficient, factors=new_factors)
            )
    return Poly(out)


def covariant_derivative_doublet(core, x_fn, mu, *, hypercharge, conjugated=False):
    """Covariant derivative of a weak-doublet-valued quantity ``x_fn(r)``.

    ``x_fn`` maps an open weak-doublet label to a :class:`Poly`.  Returns a new
    function of the doublet label implementing ``(D_mu x)_r`` following Eq.
    (D.3) with the connection acting on the doublet index; iterating it (calling
    the result again) produces ``D_mu D_nu x`` etc.
    """

    p = core.parameters
    B, W = core.fields.B, core.fields.W
    Y = hypercharge

    def result(r):
        s, adj = fresh("w"), fresh("aw")
        base = partial(x_fn(r), mu)
        if not conjugated:
            return (
                base
                - prod(I * p.g1 * Y, B(mu)) * x_fn(r)
                - prod(I * p.g2, weak_gauge_generator(adj, r, s), W(mu, adj)) * x_fn(s)
            )
        return (
            base
            + prod(I * p.g1 * Y, B(mu)) * x_fn(r)
            + prod(I * p.g2, weak_gauge_generator(adj, s, r), W(mu, adj)) * x_fn(s)
        )

    return result


def fermion_fn(core, field, sp, f, *, conjugated=False):
    """Return a callable ``(w, c) -> Poly`` for a bare fermion occurrence.

    ``w`` / ``c`` are the (optional) weak-doublet and colour-fundamental labels;
    pass ``None`` when the field does not carry that index.
    """

    from .sm_core import occ

    def fn(w=None, c=None):
        kwargs = {"sp": sp, "f": f}
        if w is not None:
            kwargs["w"] = w
        if c is not None:
            kwargs["c"] = c
        return Poly.of(occ(field, conjugated=conjugated, **kwargs))

    return fn


def covariant_derivative_fermion(core, x_fn, mu, *, hypercharge, weak, colour,
                                 conjugated=False):
    """Covariant derivative of a fermion-valued quantity ``x_fn(w, c)``.

    Implements Eq. (D.3) acting on the colour-fundamental and weak-doublet
    indices with hypercharge ``Y``.  Iterating the result (calling it again with
    the same Lorentz index) yields ``D_mu D^mu psi`` etc.  For a conjugated
    fermion the connection sign flips and the generators act from the right.
    """

    p = core.parameters
    B, W, G = core.fields.B, core.fields.W, core.fields.G
    sign = 1 if conjugated else -1

    def result(w=None, c=None):
        out = partial(x_fn(w, c), mu) + prod(sign * I * p.g1 * hypercharge, B(mu)) * x_fn(w, c)
        if weak:
            ws, adj = fresh("w"), fresh("aw")
            gen = (
                weak_gauge_generator(adj, ws, w)
                if conjugated
                else weak_gauge_generator(adj, w, ws)
            )
            out = out + prod(sign * I * p.g2, gen, W(mu, adj)) * x_fn(ws, c)
        if colour:
            cs, adjc = fresh("c"), fresh("ac")
            genc = (
                gauge_generator(adjc, cs, c)
                if conjugated
                else gauge_generator(adjc, c, cs)
            )
            out = out + prod(sign * I * p.g3, genc, G(mu, adjc)) * x_fn(w, cs)
        return out

    return result


def covariant_derivative_adjoint(core, x_fn, rho, *, group):
    """Covariant derivative of an adjoint-valued quantity ``x_fn(A)``.

    Implements ``(D_rho X)^A = d_rho X^A + g f^{ABC} V^B_rho X^C`` (Eqs.
    D.7-D.9): SU(3) uses ``f^{ABC}`` and the gluon; SU(2) uses ``eps^{IJK}`` and
    the ``W`` boson; the abelian case reduces to the ordinary derivative.
    """

    if group == "U1Y":
        return lambda A: partial(x_fn(A), rho)

    p = core.parameters
    if group == "SU3C":
        boson, coupling, struct = core.fields.G, p.g3, structure_constant
        fresh_adj = "ac"
    elif group == "SU2L":
        boson, coupling, struct = core.fields.W, p.g2, weak_structure_constant
        fresh_adj = "aw"
    else:
        raise ValueError(f"Unknown non-abelian group {group!r}.")

    def result(A):
        Bc, Cc = fresh(fresh_adj), fresh(fresh_adj)
        return partial(x_fn(A), rho) + prod(
            coupling * struct(A, Bc, Cc), boson(rho, Bc)
        ) * x_fn(Cc)

    return result


# ---------------------------------------------------------------------------
# Explicit field strengths as Poly (functions of the adjoint index)
# ---------------------------------------------------------------------------


def b_field_strength(core, mu, nu) -> Poly:
    """``B_{mu nu} = d_mu B_nu - d_nu B_mu`` (Eq. D.6)."""
    B = core.fields.B
    return Poly.of(PartialD(B(nu), mu)) - Poly.of(PartialD(B(mu), nu))


def w_field_strength(core, mu, nu, I_) -> Poly:
    """``W^I_{mu nu} = d_mu W^I_nu - d_nu W^I_mu + g2 eps^{IJK} W^J_mu W^K_nu``."""
    W = core.fields.W
    p = core.parameters
    J, K = fresh("aw"), fresh("aw")
    return (
        Poly.of(PartialD(W(nu, I_), mu))
        - Poly.of(PartialD(W(mu, I_), nu))
        + prod(p.g2 * weak_structure_constant(I_, J, K), W(mu, J), W(nu, K))
    )


def g_field_strength(core, mu, nu, A) -> Poly:
    """``G^A_{mu nu} = d_mu G^A_nu - d_nu G^A_mu + g3 f^{ABC} G^B_mu G^C_nu``."""
    G = core.fields.G
    p = core.parameters
    Bc, Cc = fresh("ac"), fresh("ac")
    return (
        Poly.of(PartialD(G(nu, A), mu))
        - Poly.of(PartialD(G(mu, A), nu))
        + prod(p.g3 * structure_constant(A, Bc, Cc), G(mu, Bc), G(nu, Cc))
    )


def PL(left, right):
    """Left chiral projector ``(1 - gamma5)/2`` with explicit spinor slots."""
    return chiral_projector_left(left, right)


def PR(left, right):
    """Right chiral projector ``(1 + gamma5)/2`` with explicit spinor slots."""
    return chiral_projector_right(left, right)


def gamma(left, right, mu):
    return gamma_matrix(left, right, mu)


def gamma5(left, right):
    return gamma5_matrix(left, right)


def gamma_chain(left, right, *lorentz):
    """Ordered gamma chain ``gamma^{mu1} ... gamma^{mun}`` (no reduction).

    With no Lorentz index it returns the spinor metric (identity).
    """

    if not lorentz:
        return spinor_metric(left, right)
    labels = [left] + [fresh("sp") for _ in range(len(lorentz) - 1)] + [right]
    expr = Expression.num(1)
    for mu, a, b in zip(lorentz, labels[:-1], labels[1:]):
        expr = expr * gamma_matrix(a, b, mu)
    return expr


def sigma(left, right, mu, nu):
    """The tensor bilinear ``sigma^{mu nu}`` with explicit spinor slots.

    Uses the Spenso ``sigma`` tensor, defined as ``(i/2)[gamma^mu, gamma^nu]``
    (see :func:`sigma_from_gamma` for the explicit commutator form used to
    validate the convention).
    """

    return sigma_tensor(left, right, mu, nu)


def sigma_from_gamma(left, right, mu, nu):
    """Explicit ``(i/2)(gamma^mu gamma^nu - gamma^nu gamma^mu)``."""

    m = fresh("sp")
    return (I / 2) * (
        gamma_matrix(left, m, mu) * gamma_matrix(m, right, nu)
        - gamma_matrix(left, m, nu) * gamma_matrix(m, right, mu)
    )


def charge_conjugation(left, right):
    """Antisymmetric Dirac charge-conjugation matrix ``C_{ij}``."""
    return dirac_charge_conjugation(left, right)


# ---------------------------------------------------------------------------
# Chirality bookkeeping for the unbroken-phase fermions
# ---------------------------------------------------------------------------

_LEFT_FIELD_NAMES = frozenset({"q", "l"})
_RIGHT_FIELD_NAMES = frozenset({"u", "d", "e"})


def chirality(field) -> str:
    """Return ``'L'`` for the doublets ``q, l`` and ``'R'`` for ``u, d, e``."""
    if field.name in _LEFT_FIELD_NAMES:
        return "L"
    if field.name in _RIGHT_FIELD_NAMES:
        return "R"
    raise ValueError(f"Field {field.name!r} has no assigned chirality.")


def projector(field, left, right):
    """Chiral projector matching ``field``'s chirality on the given spinor slots.

    Attaching one projector next to the *non-barred* fermion of a bilinear fully
    fixes the chirality of that bilinear (``qbar gamma^mu P_L q`` etc.).
    """
    return PL(left, right) if chirality(field) == "L" else PR(left, right)


# ---------------------------------------------------------------------------
# Weak-isospin structures
# ---------------------------------------------------------------------------


def pauli(adjoint, left, right):
    """Pauli matrix ``sigma^I_{rs} = 2 t^I_{rs}`` on the weak doublet."""
    return 2 * weak_gauge_generator(adjoint, left, right)


def eps2(i, j):
    """Antisymmetric SU(2) doublet invariant ``epsilon_{ij}`` (``eps_{12}=+1``)."""
    return weak_eps2(i, j)


# ---------------------------------------------------------------------------
# Levi-Civita / dual field strengths
# ---------------------------------------------------------------------------


def levi(mu, nu, rho, sigma_index):
    return lorentz_levi_civita(mu, nu, rho, sigma_index)


def field_strength(core, group_name: str, mu, nu, adjoint=None):
    """Return the engine ``FS`` factor for one gauge group by attribute name."""
    group = getattr(core.gauge_groups, group_name)
    if adjoint is None:
        return FS(group, mu, nu)
    return FS(group, mu, nu, adjoint)


def dual_field_strength(core, group_name: str, mu, nu, adjoint=None) -> Poly:
    """Dual field strength ``Xtilde_{mu nu} = 1/2 eps_{mu nu a b} X^{a b}``."""

    a, b = fresh("mu"), fresh("mu")
    fs = field_strength(core, group_name, a, b, adjoint)
    return prod(Expression.num(1) / 2 * levi(mu, nu, a, b), fs)


# ---------------------------------------------------------------------------
# Higgs currents H^dag i<->D_mu H
# ---------------------------------------------------------------------------


def higgs_lr_derivative(core, mu) -> Poly:
    """Isospin-singlet current ``H^dag i<->D_mu H = i[H^dag D_mu H - (D_mu H)^dag H]``."""

    H = core.fields.H
    return prod(I, H.bar) * DC(H, mu) - prod(I, DC(H.bar, mu)) * H


def higgs_lr_derivative_isospin(core, mu, adjoint) -> Poly:
    """Isospin-triplet current ``H^dag i<->D^I_mu H`` with an open adjoint index.

    Equals ``i[H^dag sigma^I (D_mu H) - (D_mu H)^dag sigma^I H]``.  The weak
    doublet indices of ``H^dag`` and ``DC(H)`` are threaded through the Pauli
    matrix ``sigma^I``; the compiler contracts the covariant-derivative doublet
    index with the free generator index structurally.
    """

    H = core.fields.H
    r, t = fresh("w"), fresh("w")
    left = prod(I, H.bar(r), pauli(adjoint, r, t), DC(H, mu))
    right = prod(I, DC(H.bar, mu), pauli(adjoint, r, t), H(t))
    # ``DC(H, mu)`` threads its doublet index to ``t``; ``DC(H.bar, mu)`` threads
    # to ``r`` in the second term (the other generator slot).
    return left - right


def covariant_derivative_higgs(core, mu, weak, *, conjugated: bool = False) -> Poly:
    """Explicit ``(D_mu H)_r`` (or ``(D_mu H^dag)_r``) with an open doublet index.

    Written out term by term so the doublet index ``weak`` stays under the
    caller's control (needed when several Higgs doublets appear in one operator,
    e.g. ``OHD``, or when iterating the derivative).  Follows Eq. (D.3):
    ``D_mu H = (d_mu - i g1 Y B_mu - i g2 sigma^I/2 W^I_mu) H`` with ``Y = 1/2``.
    """

    p = core.parameters
    H, B, W = core.fields.H, core.fields.B, core.fields.W
    Y = core.fields.H.quantum_numbers["Y"]
    s, adj = fresh("w"), fresh("aw")
    if not conjugated:
        return (
            Poly.of(PartialD(H(weak), mu))
            - prod(I * p.g1 * Y, B(mu), H(weak))
            - prod(I * p.g2, weak_gauge_generator(adj, weak, s), W(mu, adj), H(s))
        )
    return (
        Poly.of(PartialD(H.bar(weak), mu))
        + prod(I * p.g1 * Y, B(mu), H.bar(weak))
        + prod(I * p.g2, weak_gauge_generator(adj, s, weak), W(mu, adj), H.bar(s))
    )


# ---------------------------------------------------------------------------
# Conjugate Higgs doublet Htilde_r = eps_{rs} (H^*)_s
# ---------------------------------------------------------------------------


def htilde(core, r, *, conjugated: bool = False) -> Poly:
    """Return ``Htilde_r = eps_{rs} Hbar_s`` (or its conjugate for ``bar``)."""

    H = core.fields.H
    s = fresh("w")
    if not conjugated:
        return prod(eps2(r, s), H.bar(s))
    # (Htilde^dag)_r = eps_{rs} H_s  (eps real, eps^dag = eps^T = -eps; but the
    # standard identity gives (Htilde)^dag_r = eps_{rs} H_s with eps_{12}=+1).
    return prod(eps2(r, s), H(s))


__all__ = (
    "Poly",
    "as_monomials",
    "prod",
    "summed",
    "fresh",
    "partial",
    "covariant_derivative_doublet",
    "fermion_fn",
    "covariant_derivative_fermion",
    "covariant_derivative_adjoint",
    "b_field_strength",
    "w_field_strength",
    "g_field_strength",
    "PL",
    "PR",
    "gamma",
    "gamma5",
    "gamma_chain",
    "sigma",
    "sigma_from_gamma",
    "charge_conjugation",
    "chirality",
    "projector",
    "pauli",
    "eps2",
    "levi",
    "field_strength",
    "dual_field_strength",
    "higgs_lr_derivative",
    "higgs_lr_derivative_isospin",
    "covariant_derivative_higgs",
    "htilde",
)
