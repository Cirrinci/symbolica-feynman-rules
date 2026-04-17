"""
Reusable operator builders for the Symbolica/Spenso prototype.

Keep this module small and composable: it should expose the tensor/operator
vocabulary used repeatedly across examples and model definitions, but not the
full model compiler.
"""

from symbolica import Expression

from symbolic.model_symbolica import bis, pcomp
from symbolic.spenso_structures import (
    COLOR_ADJ,
    gamma5_matrix,
    gamma_lowered_matrix,
    gamma_matrix,
    gauge_generator,
    lorentz_metric,
    structure_constant,
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
    """Two-gauge-boson contact tensor for complex scalar kinetic terms."""
    return lorentz_metric(mu, nu)


def gauge_kinetic_bilinear(mu, nu, p_left, p_right, contracted_lorentz):
    """Compact two-gauge-field kinetic tensor after metric contraction."""
    return (
        lorentz_metric(mu, nu) * pcomp(p_left, contracted_lorentz) * pcomp(p_right, contracted_lorentz)
        - pcomp(p_left, nu) * pcomp(p_right, mu)
    )


def gauge_fixing_bilinear(mu, nu, p_left, p_right):
    """Compact two-gauge-field gauge-fixing tensor after metric contraction."""
    return pcomp(p_left, nu) * pcomp(p_right, mu)


def gauge_kinetic_bilinear_raw(
    mu,
    nu,
    p_left,
    p_right,
    contracted_lorentz,
    left_derivative_lorentz,
    right_derivative_lorentz,
):
    """Unsimplified two-gauge-field kinetic tensor matching compiler output."""
    return (
        lorentz_metric(mu, nu) * pcomp(p_left, contracted_lorentz) * pcomp(p_right, contracted_lorentz)
        - (Expression.num(1) / Expression.num(2))
        * lorentz_metric(mu, left_derivative_lorentz)
        * lorentz_metric(nu, right_derivative_lorentz)
        * pcomp(p_left, right_derivative_lorentz)
        * pcomp(p_right, left_derivative_lorentz)
        - (Expression.num(1) / Expression.num(2))
        * lorentz_metric(mu, right_derivative_lorentz)
        * lorentz_metric(nu, left_derivative_lorentz)
        * pcomp(p_left, left_derivative_lorentz)
        * pcomp(p_right, right_derivative_lorentz)
    )


def gauge_fixing_bilinear_raw(
    mu,
    nu,
    p_left,
    p_right,
    left_derivative_lorentz,
    right_derivative_lorentz,
):
    """Unsimplified two-gauge-field gauge-fixing tensor matching compiler output."""
    return (
        (Expression.num(1) / Expression.num(2))
        * lorentz_metric(mu, left_derivative_lorentz)
        * lorentz_metric(nu, right_derivative_lorentz)
        * pcomp(p_left, left_derivative_lorentz)
        * pcomp(p_right, right_derivative_lorentz)
        + (Expression.num(1) / Expression.num(2))
        * lorentz_metric(mu, right_derivative_lorentz)
        * lorentz_metric(nu, left_derivative_lorentz)
        * pcomp(p_left, right_derivative_lorentz)
        * pcomp(p_right, left_derivative_lorentz)
    )


def yang_mills_three_vertex_raw(adj_left, adj_mid, adj_right, mu, nu, rho, p_left, p_mid, p_right):
    """Unsimplified three-gauge tensor matching the current compiler output."""
    return (
        lorentz_metric(mu, nu)
        * (
            structure_constant(adj_left, adj_right, adj_mid) * pcomp(p_left, rho)
            + structure_constant(adj_mid, adj_right, adj_left) * pcomp(p_mid, rho)
        )
        + lorentz_metric(mu, rho)
        * (
            structure_constant(adj_left, adj_mid, adj_right) * pcomp(p_left, nu)
            + structure_constant(adj_right, adj_mid, adj_left) * pcomp(p_right, nu)
        )
        + lorentz_metric(nu, rho)
        * (
            structure_constant(adj_mid, adj_left, adj_right) * pcomp(p_mid, mu)
            + structure_constant(adj_right, adj_left, adj_mid) * pcomp(p_right, mu)
        )
    )


def yang_mills_three_vertex_metric_raw(
    adj_left,
    adj_mid,
    adj_right,
    mu,
    nu,
    rho,
    p_left,
    p_mid,
    p_right,
    derivative_lorentz,
):
    """Compiler-raw three-gauge tensor before metric contraction simplification."""
    return (
        lorentz_metric(mu, nu)
        * lorentz_metric(rho, derivative_lorentz)
        * (
            structure_constant(adj_left, adj_right, adj_mid) * pcomp(p_left, derivative_lorentz)
            + structure_constant(adj_mid, adj_right, adj_left) * pcomp(p_mid, derivative_lorentz)
        )
        + lorentz_metric(mu, rho)
        * lorentz_metric(nu, derivative_lorentz)
        * (
            structure_constant(adj_left, adj_mid, adj_right) * pcomp(p_left, derivative_lorentz)
            + structure_constant(adj_right, adj_mid, adj_left) * pcomp(p_right, derivative_lorentz)
        )
        + lorentz_metric(nu, rho)
        * lorentz_metric(mu, derivative_lorentz)
        * (
            structure_constant(adj_mid, adj_left, adj_right) * pcomp(p_mid, derivative_lorentz)
            + structure_constant(adj_right, adj_left, adj_mid) * pcomp(p_right, derivative_lorentz)
        )
    )


def _four_gauge_channel_raw(adj_left, adj_right, adj_other_left, adj_other_right, internal):
    """One color-channel block used in the raw Yang-Mills quartic vertex."""
    return (
        structure_constant(adj_other_right, adj_left, internal)
        * structure_constant(adj_other_left, adj_right, internal)
        + structure_constant(adj_other_right, adj_right, internal)
        * structure_constant(adj_other_left, adj_left, internal)
        + structure_constant(adj_left, adj_other_right, internal)
        * structure_constant(adj_right, adj_other_left, internal)
        + structure_constant(adj_left, adj_other_left, internal)
        * structure_constant(adj_right, adj_other_right, internal)
    )


def yang_mills_four_vertex_raw(adj1, adj2, adj3, adj4, mu, nu, rho, sigma, internal):
    """Unsimplified four-gauge tensor matching the current compiler output.

    This keeps the separate metric channels explicit so the raw compiled
    expression can be compared directly against compiler output before any
    physics-specific compactification step.
    """
    return (
        lorentz_metric(mu, nu) * lorentz_metric(rho, sigma)
        * _four_gauge_channel_raw(adj1, adj2, adj3, adj4, internal)
        + lorentz_metric(mu, rho) * lorentz_metric(nu, sigma)
        * _four_gauge_channel_raw(adj1, adj3, adj2, adj4, internal)
        + lorentz_metric(mu, sigma) * lorentz_metric(nu, rho)
        * _four_gauge_channel_raw(adj1, adj4, adj2, adj3, internal)
    )


def ghost_kinetic_raw(
    adjoint_bar,
    adjoint_ghost,
    p_bar,
    p_ghost,
    bar_derivative_lorentz,
    ghost_derivative_lorentz,
):
    """Unsimplified ghost bilinear tensor matching compiler output."""
    return (
        COLOR_ADJ.g(adjoint_bar, adjoint_ghost).to_expression()
        * lorentz_metric(bar_derivative_lorentz, ghost_derivative_lorentz)
        * pcomp(p_bar, bar_derivative_lorentz)
        * pcomp(p_ghost, ghost_derivative_lorentz)
    )


def ghost_kinetic(adjoint_bar, adjoint_ghost, p_bar, p_ghost, contracted_lorentz):
    """Compact ghost bilinear tensor after Lorentz-metric contraction."""
    return (
        COLOR_ADJ.g(adjoint_bar, adjoint_ghost).to_expression()
        * pcomp(p_bar, contracted_lorentz)
        * pcomp(p_ghost, contracted_lorentz)
    )


def ghost_gauge_raw(
    adjoint_bar,
    adjoint_gauge,
    adjoint_ghost,
    gauge_lorentz,
    derivative_lorentz,
    p_bar,
):
    """Unsimplified ghost-gauge tensor matching compiler output."""
    return (
        structure_constant(adjoint_bar, adjoint_gauge, adjoint_ghost)
        * lorentz_metric(derivative_lorentz, gauge_lorentz)
        * pcomp(p_bar, derivative_lorentz)
    )


def ghost_gauge(adjoint_bar, adjoint_gauge, adjoint_ghost, gauge_lorentz, p_bar):
    """Compact ghost-gauge tensor after Lorentz-metric contraction."""
    return structure_constant(adjoint_bar, adjoint_gauge, adjoint_ghost) * pcomp(p_bar, gauge_lorentz)
