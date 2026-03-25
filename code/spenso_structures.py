"""
Thin wrappers for Spenso HEP tensor objects used by the Symbolica prototype.

These helpers keep the rest of the code readable while ensuring that gamma
matrices, gauge generators, and metrics are represented as native Spenso
tensors with typed index slots.
"""

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


def gamma_matrix(left_spinor, right_spinor, lorentz):
    return TensorName.gamma()(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
        lorentz_index(lorentz),
    ).to_expression()


def gamma_lowered_matrix(left_spinor, right_spinor, lowered_lorentz, summed_lorentz="rho_gamma_tmp"):
    return (
        lorentz_metric(summed_lorentz, lowered_lorentz)
        * gamma_matrix(left_spinor, right_spinor, summed_lorentz)
    )


def gamma5_matrix(left_spinor, right_spinor):
    return TensorName.gamma5()(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


def sigma_tensor(left_spinor, right_spinor, *lorentz_indices):
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
    return TensorName.t()(
        color_adj_index(adj_index),
        color_fund_index(fund_left),
        color_fund_index(fund_right),
    ).to_expression()


def structure_constant(a, b, c):
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
    expr = simplify_gamma(expr)
    expr = simplify_metrics(expr)
    return expr


def hep_tensor_scalar(expr):
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_scalar()


def hep_tensor_result(expr):
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_tensor(library=_HEP_LIBRARY)
