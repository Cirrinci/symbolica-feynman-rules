"""
Thin wrappers for Spenso HEP tensor objects used by the Symbolica prototype.

These helpers ensures that gamma
matrices, and metrics are represented as native Spenso
tensors with typed index slots.
"""

from itertools import count
from typing import Mapping, Sequence

from symbolica import Expression, S
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

DEFAULT_SLOT_LABEL_PREFIXES = {
    SPINOR_KIND: "i",
    LORENTZ_KIND: "mu",
    COLOR_FUND_KIND: "c",
    COLOR_ADJ_KIND: "a",
}


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


def _label_tuple(labels):
    if labels is None:
        return ()
    if isinstance(labels, tuple):
        return tuple(label for label in labels if label is not None)
    if isinstance(labels, list):
        return tuple(label for label in labels if label is not None)
    return (labels,)


def slot_labels(**labels_by_kind):
    """Normalize per-slot index labels into a consistent mapping."""
    normalized = {}
    for kind, labels in labels_by_kind.items():
        values = _label_tuple(labels)
        if values:
            normalized[kind] = values
    return normalized


def default_leg_slot_labels(
    field_slot_labels: Sequence[Mapping[str, Sequence]],
    *,
    prefix_overrides=None,
):
    """Generate default external-leg labels from a field-slot layout."""
    prefix_map = dict(DEFAULT_SLOT_LABEL_PREFIXES)
    if prefix_overrides is not None:
        prefix_map.update(prefix_overrides)

    generated = []
    for leg_position, entry in enumerate(field_slot_labels, start=1):
        if not entry:
            generated.append(None)
            continue

        leg_entry = {}
        for kind, labels in entry.items():
            prefix = prefix_map.get(kind, kind.replace(" ", "_"))
            values = []
            for index_position, _ in enumerate(labels, start=1):
                suffix = "" if len(labels) == 1 else f"_{index_position}"
                values.append(S(f"{prefix}{leg_position}{suffix}"))
            leg_entry[kind] = tuple(values)
        generated.append(leg_entry)

    return generated


def gamma_matrix(left_spinor, right_spinor, lorentz):
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
