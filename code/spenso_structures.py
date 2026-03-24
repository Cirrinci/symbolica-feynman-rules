"""
Thin wrappers for Spenso HEP tensor objects used by the Symbolica prototype.

These helpers keep the rest of the code readable while ensuring that gamma
matrices, gauge generators, and metrics are represented as native Spenso
tensors with typed index slots.
"""

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


def gamma5_matrix(left_spinor, right_spinor):
    return TensorName.gamma5()(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


def lorentz_metric(mu, nu):
    return LORENTZ.g(lorentz_index(mu), lorentz_index(nu)).to_expression()


def spinor_metric(left_spinor, right_spinor):
    return BISPINOR.g(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


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


def hep_tensor_scalar(expr):
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_scalar()


def hep_tensor_result(expr):
    network = TensorNetwork(expr, library=_HEP_LIBRARY)
    network.execute(library=_HEP_LIBRARY)
    return network.result_tensor(library=_HEP_LIBRARY)
