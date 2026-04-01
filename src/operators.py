"""
Reusable operator builders for the Symbolica/Spenso prototype.

Keep this module small and composable: it should expose the tensor/operator
vocabulary used repeatedly across examples and model definitions, but not the
full model compiler.
"""

from model_symbolica import bis
from spenso_structures import (
    gamma5_matrix,
    gamma_lowered_matrix,
    gamma_matrix,
    gauge_generator,
    lorentz_metric,
)


def psi_bar_psi(left_spinor, right_spinor):
    """Spinor metric for a fermion bilinear."""
    return bis.g(left_spinor, right_spinor).to_expression()


def psi_bar_gamma_psi(left_spinor, right_spinor, lorentz):
    """Vector current gamma-chain."""
    return gamma_matrix(left_spinor, right_spinor, lorentz)


def psi_bar_gamma5_psi(left_spinor, right_spinor):
    """Chiral gamma5 bilinear."""
    return gamma5_matrix(left_spinor, right_spinor)


def psi_bar_gamma_lowered_psi(left_spinor, right_spinor, lorentz, summed_lorentz=None):
    """Lowered-index gamma-chain with metric contraction."""
    return gamma_lowered_matrix(left_spinor, right_spinor, lorentz, summed_lorentz)


def current_current(a_bar, a_psi, b_bar, b_psi, lorentz):
    """Current-current four-fermion structure."""
    return psi_bar_gamma_psi(a_bar, a_psi, lorentz) * psi_bar_gamma_lowered_psi(
        b_bar, b_psi, lorentz
    )


def quark_gluon_current(i_bar_q, i_psi_q, lorentz, a_g, c_bar_q, c_psi_q):
    """Quark-gluon current structure."""
    return psi_bar_gamma_psi(i_bar_q, i_psi_q, lorentz) * gauge_generator(
        a_g, c_bar_q, c_psi_q
    )


def scalar_gauge_current_term(coupling, derivative_index, target):
    """Small helper for the scalar-gauge current derivative bookkeeping."""
    return dict(
        coupling=coupling,
        derivative_indices=[derivative_index],
        derivative_targets=[target],
    )


def scalar_gauge_contact(mu, nu):
    """Scalar QED contact tensor."""
    return lorentz_metric(mu, nu)
