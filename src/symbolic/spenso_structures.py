"""
Thin wrappers for Spenso HEP tensor objects used by the Symbolica prototype.

These helpers ensure that gamma matrices, metrics, and gauge tensors are
represented as native Spenso tensors with typed index slots.

In addition to the original gamma/generator/structure-constant helpers, this
module exposes a typed catalogue of *invariant tensors* (Kronecker deltas,
Levi-Civita symbols on Minkowski / SU(2) doublet / SU(3) fundamental, the
charge-conjugation matrix, and the totally symmetric SU(N) constant
``d^{abc}``).  Each invariant is built from a Spenso ``TensorName`` so its
index slots are always typed; symmetry information is attached at the
canonicalisation layer (see ``symbolic.tensor_canonicalization``) rather than
on the live Spenso head, which keeps the Symbolica namespace stable across the
swap-and-simplify cycle.

The module also provides a single-call ``simplify_invariants`` pipeline that
threads Symbolica's idenso passes (``simplify_metrics``, ``simplify_gamma``,
``simplify_color``) into the existing vertex post-processing flow.
"""

from itertools import count, permutations

from symbolica import Expression, S
from symbolica.community.idenso import (
    simplify_color,
    simplify_gamma,
    simplify_metrics,
)
from symbolica.community.spenso import (
    LibraryTensor,
    Representation,
    Slot,
    TensorLibrary,
    TensorName,
    TensorNetwork,
    TensorStructure,
)


BISPINOR = Representation.bis(4)
LORENTZ = Representation.mink(4)
COLOR_FUND = Representation.cof(3)
COLOR_ADJ = Representation.coad(8)
# SU(2) weak-isospin representations (doublet, adjoint triplet).
WEAK_FUND = Representation.cof(2)
WEAK_ADJ = Representation.coad(3)

_HEP_LIBRARY = TensorLibrary.hep_lib()
_ONE = Expression.num(1)
_TWO = Expression.num(2)
_GAMMA_LOWERED_COUNTER = count()
_PROJECTOR_LEFT = S("PL", is_real=True)
_PROJECTOR_RIGHT = S("PR", is_real=True)

SPINOR_KIND = "spinor"
LORENTZ_KIND = "lorentz"
COLOR_FUND_KIND = "color_fund"
COLOR_ADJ_KIND = "color_adj"
WEAK_FUND_KIND = "weak_fund"
WEAK_ADJ_KIND = "weak_adj"


def _slot(rep, index):
    if isinstance(index, Slot):
        return index
    return rep(index)


def bispinor_index(index):
    return _slot(BISPINOR, index)


def lorentz_index(index):
    return _slot(LORENTZ, index)


def color_fund_index(index):
    return _slot(COLOR_FUND, index)


def color_adj_index(index):
    return _slot(COLOR_ADJ, index)


def weak_fund_index(index):
    return _slot(WEAK_FUND, index)


def weak_adj_index(index):
    return _slot(WEAK_ADJ, index)


def _fresh_index_name(prefix):
    return f"{prefix}_{next(_GAMMA_LOWERED_COUNTER)}"


def gamma_matrix(left_spinor, right_spinor, lorentz):
    """Return a Spenso-backed gamma^mu tensor as a Symbolica expression."""
    return TensorName.gamma()(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
        lorentz_index(lorentz),
    ).to_expression()


def gamma_lowered_matrix(
    left_spinor,
    right_spinor,
    lowered_lorentz,
    summed_lorentz=None,
):
    if summed_lorentz is None:
        summed_lorentz = _fresh_index_name("rho_gamma_tmp")
    return (
        lorentz_metric(summed_lorentz, lowered_lorentz)
        * gamma_matrix(left_spinor, right_spinor, summed_lorentz)
    )


def gamma5_matrix(left_spinor, right_spinor):
    """Return gamma5 with explicit bispinor slots."""
    return TensorName.gamma5()(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


def sigma_tensor(left_spinor, right_spinor, *lorentz_indices):
    """Return sigma^{mu nu...} with explicit spinor and Lorentz slots."""
    slots = [
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
        *[lorentz_index(mu) for mu in lorentz_indices],
    ]
    return TensorName.sigma()(*slots).to_expression()


def lorentz_metric(mu, nu):
    return LORENTZ.g(mu, nu).to_expression()


def spinor_metric(left_spinor, right_spinor):
    return BISPINOR.g(left_spinor, right_spinor).to_expression()


def chiral_projector_left(left_spinor, right_spinor):
    return (_ONE / _TWO) * (
        spinor_metric(left_spinor, right_spinor)
        - gamma5_matrix(left_spinor, right_spinor)
    )


def chiral_projector_right(left_spinor, right_spinor):
    return (_ONE / _TWO) * (
        spinor_metric(left_spinor, right_spinor)
        + gamma5_matrix(left_spinor, right_spinor)
    )


def projector_left(left_spinor, right_spinor):
    """Return a compact left-chiral projector head ``PL(i, j)``."""

    return _PROJECTOR_LEFT(left_spinor, right_spinor)


def projector_right(left_spinor, right_spinor):
    """Return a compact right-chiral projector head ``PR(i, j)``."""

    return _PROJECTOR_RIGHT(left_spinor, right_spinor)


def gauge_generator(adj_index, fund_left, fund_right):
    """Return the fundamental-representation generator tensor t^a_{ij}."""
    return TensorName.t()(
        color_adj_index(adj_index),
        color_fund_index(fund_left),
        color_fund_index(fund_right),
    ).to_expression()


def structure_constant(a, b, c):
    """Return the adjoint structure constant tensor f^{abc}."""
    return TensorName.f()(
        color_adj_index(a),
        color_adj_index(b),
        color_adj_index(c),
    ).to_expression()


def weak_gauge_generator(adj_index, fund_left, fund_right):
    """Return the SU(2) fundamental-representation generator t^a_{ij} (doublet)."""
    return TensorName.t()(
        weak_adj_index(adj_index),
        weak_fund_index(fund_left),
        weak_fund_index(fund_right),
    ).to_expression()


def weak_structure_constant(a, b, c):
    """Return the SU(2) adjoint structure constant tensor f^{abc} = epsilon^{abc}."""
    return TensorName.f()(
        weak_adj_index(a),
        weak_adj_index(b),
        weak_adj_index(c),
    ).to_expression()


def gamma_anticommutator(left_spinor, right_spinor, mu, nu):
    i = bispinor_index(left_spinor)
    k = bispinor_index(right_spinor)
    j = bispinor_index("j_gamma_tmp")

    return (
        gamma_matrix(i, j, mu) * gamma_matrix(j, k, nu)
        + gamma_matrix(i, j, nu) * gamma_matrix(j, k, mu)
    )


def gamma_commutator(left_spinor, right_spinor, mu, nu):
    i = bispinor_index(left_spinor)
    k = bispinor_index(right_spinor)
    j = bispinor_index("j_gamma_tmp")

    return (
        gamma_matrix(i, j, mu) * gamma_matrix(j, k, nu)
        - gamma_matrix(i, j, nu) * gamma_matrix(j, k, mu)
    )


def simplify_gamma_chain(expr):
    """Apply the standard gamma then metric simplification pass."""
    expr = simplify_gamma(expr)
    expr = simplify_metrics(expr)
    return expr


def simplify_invariants(expr, *, run_gamma: bool = True, run_color: bool = True):
    """One-call idenso simplification pass over a tensor expression.

    Composes Symbolica's idenso primitives in a stable order:

    1. ``simplify_metrics`` to contract obvious ``g(mu, nu) g(nu, rho)``-type
       chains and absorb metric-against-tensor contractions.
    2. (optional) ``simplify_color`` to apply SU(N) color identities
       (Fierz-style collapses on ``T^a`` and ``f^{abc}``).
    3. (optional) ``simplify_gamma`` for Dirac chains.
    4. ``simplify_metrics`` again because the previous passes can expose new
       metric pairs.

    The pass is conservative: each idenso step is a no-op on subexpressions it
    does not understand, so it is safe to call on any vertex factor.
    """

    expr = simplify_metrics(expr)
    if run_color:
        expr = simplify_color(expr)
    if run_gamma:
        expr = simplify_gamma(expr)
    expr = simplify_metrics(expr)
    return expr


# ---------------------------------------------------------------------------
# Invariant tensors with typed Spenso slots
# ---------------------------------------------------------------------------
#
# All heads below use plain ``TensorName`` (no ``is_symmetric`` /
# ``is_antisymmetric`` attribute). Symmetry is registered at the
# ``tensor_canonicalization`` layer through the swap-to-``canon::`` pattern,
# which is the same approach already used for the SU(N) structure constant
# ``f^{abc}`` (which we know works thanks to the existing regression test).
#
# Naming convention:
#   - ``weak_eps2``       -- antisymmetric SU(2) doublet ε_{ij} on cof(2)
#   - ``lor_levi_civita`` -- totally antisymmetric ε_{μνρσ} on mink(4)
#   - ``color_eps3``      -- totally antisymmetric ε_{ijk} on cof(3)
#   - ``color_d``         -- totally symmetric SU(N) d^{abc} on coad
#   - ``dirac_C``         -- antisymmetric charge-conjugation matrix on bis(4)


WEAK_EPS2 = TensorName("weak_eps2")
LOR_LEVI_CIVITA = TensorName("lor_levi_civita")
COLOR_EPS3 = TensorName("color_eps3")
COLOR_D = TensorName("color_d")
DIRAC_C = TensorName("dirac_C")


def weak_eps2(i, j):
    """SU(2) doublet antisymmetric invariant ε_{ij} on ``cof(2)``.

    Accepts either a Spenso ``Slot`` or a Symbolica/string label that will be
    auto-promoted to a ``cof(2)`` slot.  Antisymmetry is enforced by the
    canonicalisation layer (see ``SPENSO_TENSOR_HEAD_SPECS``).
    """
    return WEAK_EPS2(weak_fund_index(i), weak_fund_index(j)).to_expression()


def lorentz_levi_civita(mu, nu, rho, sigma):
    """Totally antisymmetric Levi-Civita symbol on Minkowski ``mink(4)``."""
    slots = (
        lorentz_index(mu),
        lorentz_index(nu),
        lorentz_index(rho),
        lorentz_index(sigma),
    )
    return LOR_LEVI_CIVITA(*slots).to_expression()


def color_levi_civita(c1, c2, c3):
    """Totally antisymmetric SU(3) fundamental Levi-Civita ε_{ijk}."""
    slots = (
        color_fund_index(c1),
        color_fund_index(c2),
        color_fund_index(c3),
    )
    return COLOR_EPS3(*slots).to_expression()


def color_symmetric_constant(a, b, c):
    """Totally symmetric SU(N) adjoint invariant ``d^{abc}`` on ``coad(8)``."""
    slots = (
        color_adj_index(a),
        color_adj_index(b),
        color_adj_index(c),
    )
    return COLOR_D(*slots).to_expression()


def dirac_charge_conjugation(i, j):
    """Antisymmetric Dirac charge-conjugation matrix ``C_{ij}`` on ``bis(4)``."""
    return DIRAC_C(bispinor_index(i), bispinor_index(j)).to_expression()


# ---------------------------------------------------------------------------
# Numeric Spenso library covering the new invariants
# ---------------------------------------------------------------------------
#
# The default ``TensorLibrary.hep_lib()`` ships components for the standard
# HEP tensors (gamma, sigma, generators, metrics).  The custom invariants
# above are not part of that library, so we register their explicit
# components here.  This makes ``TensorNetwork(...).execute(library=...)``
# capable of evaluating expressions that mix HEP tensors with these
# project-defined invariants.
#
# Convention used for ε:  ε_{12} = +1, ε_{123} = +1.


def _levi_civita_components(rank: int) -> list[float]:
    """Flat row-major components of the totally antisymmetric ``rank``-tensor."""
    components = [0.0] * (rank ** rank)
    for perm in permutations(range(rank)):
        sign = 1.0
        for i in range(rank):
            for j in range(i + 1, rank):
                if perm[i] > perm[j]:
                    sign = -sign
        flat = 0
        for axis in perm:
            flat = flat * rank + axis
        components[flat] = sign
    return components


def _register_dense(library, name_obj, representations, components):
    structure = TensorStructure(*representations, name=name_obj)
    library.register(LibraryTensor.dense(structure, components))


def _build_extended_hep_library():
    library = TensorLibrary.hep_lib()
    _register_dense(
        library,
        WEAK_EPS2,
        (WEAK_FUND, WEAK_FUND),
        _levi_civita_components(2),
    )
    _register_dense(
        library,
        COLOR_EPS3,
        (COLOR_FUND, COLOR_FUND, COLOR_FUND),
        _levi_civita_components(3),
    )
    return library


_EXTENDED_HEP_LIBRARY = _build_extended_hep_library()


def extended_hep_library():
    """Return the project-extended HEP tensor library.

    The returned library is the standard ``TensorLibrary.hep_lib()`` with
    additional numeric components for ``weak_eps2`` and ``color_eps3`` so
    these custom invariants can participate in ``TensorNetwork`` execution.
    """
    return _EXTENDED_HEP_LIBRARY


def hep_tensor_scalar(expr):
    """Evaluate a tensor network down to a scalar using the HEP tensor library."""
    network = TensorNetwork(expr, library=_EXTENDED_HEP_LIBRARY)
    network.execute(library=_EXTENDED_HEP_LIBRARY)
    return network.result_scalar()


def hep_tensor_result(expr):
    network = TensorNetwork(expr, library=_EXTENDED_HEP_LIBRARY)
    network.execute(library=_EXTENDED_HEP_LIBRARY)
    return network.result_tensor(library=_EXTENDED_HEP_LIBRARY)
