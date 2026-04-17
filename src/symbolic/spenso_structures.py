"""
Thin wrappers for Spenso HEP tensor objects used by the Symbolica prototype.

These helpers ensure that gamma matrices, metrics, and gauge tensors are
represented as native Spenso tensors with typed index slots.
"""

from itertools import count

from symbolica import Expression
from symbolica.community.idenso import simplify_gamma, simplify_metrics
from symbolica.community.spenso import (
    Representation,
    Slot,
    TensorLibrary,
    TensorName,
    TensorNetwork,
)


BISPINOR = Representation.bis(4)
LORENTZ = Representation.mink(4)
COLOR_FUND = Representation.cof(3)
COLOR_ADJ = Representation.coad(8)

_HEP_LIBRARY = TensorLibrary.hep_lib()
_ONE = Expression.num(1)
_TWO = Expression.num(2)
_GAMMA_LOWERED_COUNTER = count()

SPINOR_KIND = "spinor"
LORENTZ_KIND = "lorentz"
COLOR_FUND_KIND = "color_fund"
COLOR_ADJ_KIND = "color_adj"


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


def hep_tensor_scalar(expr):
    """Evaluate a tensor network down to a scalar using the HEP tensor library."""
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_scalar()


def hep_tensor_result(expr):
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_tensor(library=_HEP_LIBRARY)
