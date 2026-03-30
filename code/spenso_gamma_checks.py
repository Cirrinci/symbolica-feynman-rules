"""
Focused gamma-matrix checks for the Symbolica/Spenso prototype.

Currently covered here:
- Clifford anticommutator
- gamma5, sigma, and chiral projectors
- gamma^mu gamma^nu vs gamma^nu gamma^mu ordering
- longer gamma chains and projected currents
- stripped current-current operator

Note:
This script follows the current local convention in model_symbolica.py where
the plane-wave factor is replaced by 1 rather than by (2*pi)^d Delta(sum p).
"""

from symbolica import Expression

from model_symbolica import I, S, simplify_deltas, vertex_factor
from spenso_structures import (
    chiral_projector_left,
    chiral_projector_right,
    gamma_anticommutator,
    gamma5_matrix,
    gamma_lowered_matrix,
    gamma_matrix,
    hep_tensor_scalar,
    lorentz_metric,
    sigma_tensor,
    simplify_gamma_chain,
    spinor_metric,
)

x = S("x")
d = S("d")

psi0 = S("psi0")
psibar0 = S("psibar0")

b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")

g_bilin = S("gBilin")

mu = S("mu")
nu = S("nu")
rho = S("rho")
i_bar = S("i_bar")
i_psi = S("i_psi")
alpha = S("alpha")
beta = S("beta")
s1 = S("s1")
s2 = S("s2")
s3 = S("s3")
s4 = S("s4")

rho_mu = "rho_mu"
rho_nu = "rho_nu"


def _check(got, expected, label):
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


def _display_expr(expr):
    return Expression.parse(expr.to_canonical_string())


def _gamma_simplified(expr):
    return simplify_gamma_chain(_display_expr(expr)).expand()


def _show(title, expr):
    print(title)
    print(_gamma_simplified(expr))
    print()


def _show_raw_and_simplified(title, expr):
    print(title)
    print("raw:")
    print(_display_expr(expr).expand())
    print("simplified:")
    print(_gamma_simplified(expr))
    print()


def _vertex(coupling, *, strip_externals=True, alphas, betas, ps, field_roles, leg_roles, field_spinor_indices):
    return simplify_deltas(
        vertex_factor(
            coupling=coupling,
            alphas=alphas,
            betas=betas,
            ps=ps,
            statistics="fermion",
            field_roles=field_roles,
            leg_roles=leg_roles,
            field_spinor_indices=field_spinor_indices,
            strip_externals=strip_externals,
            x=x,
            d=d,
        ),
        species_map={beta_: species for beta_, species in zip(betas, alphas)},
    )


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("  Demo: spenso-gamma")
    print(f"{'='*80}\n")

    print("== gamma identities ==")
    print(
        "Tr(gamma^mu gamma_mu) =",
        hep_tensor_scalar(
            gamma_matrix("si", "sj", "mu") * gamma_matrix("sj", "si", "mu")
        ),
    )
    print()
    print("{gamma^mu, gamma^nu} =")
    anticom = simplify_gamma_chain(gamma_anticommutator("si", "sk", "mu", "nu")).expand()
    print(anticom)
    print()

    print("== bilinears ==")
    bilinear_up = _vertex(
        g_bilin * gamma_matrix(i_bar, i_psi, mu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar gamma^mu psi", bilinear_up)

    bilinear_g5 = _vertex(
        g_bilin * gamma5_matrix(i_bar, i_psi),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar gamma5 psi", bilinear_g5)

    bilinear_sigma = _vertex(
        g_bilin * sigma_tensor(i_bar, i_psi, mu, nu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar sigma^{mu nu} psi", bilinear_sigma)

    bilinear_pl = _vertex(
        g_bilin * chiral_projector_left(i_bar, i_psi),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar P_L psi", bilinear_pl)

    bilinear_pr = _vertex(
        g_bilin * chiral_projector_right(i_bar, i_psi),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar P_R psi", bilinear_pr)

    bilinear_down = _vertex(
        g_bilin * gamma_lowered_matrix(i_bar, i_psi, mu, rho_mu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    print("gBilin * psibar gamma_mu psi")
    print(
        "equivalent after metric simplification:",
        _gamma_simplified(bilinear_up).to_canonical_string()
        == _gamma_simplified(bilinear_down).to_canonical_string(),
    )
    print()

    bilinear_mu_nu = _vertex(
        g_bilin * gamma_matrix(i_bar, alpha, mu) * gamma_matrix(alpha, i_psi, nu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar gamma^mu gamma^nu psi", bilinear_mu_nu)

    bilinear_mu_nu_rho = _vertex(
        g_bilin
        * gamma_matrix(i_bar, alpha, mu)
        * gamma_matrix(alpha, beta, nu)
        * gamma_matrix(beta, i_psi, rho),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show_raw_and_simplified("gBilin * psibar gamma^mu gamma^nu gamma^rho psi", bilinear_mu_nu_rho)

    bilinear_mu_nu_g5 = _vertex(
        g_bilin
        * gamma_matrix(i_bar, alpha, mu)
        * gamma_matrix(alpha, beta, nu)
        * gamma5_matrix(beta, i_psi),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar gamma^mu gamma^nu gamma5 psi", bilinear_mu_nu_g5)

    bilinear_nu_mu = _vertex(
        g_bilin * gamma_matrix(i_bar, alpha, nu) * gamma_matrix(alpha, i_psi, mu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    _show("gBilin * psibar gamma^nu gamma^mu psi", bilinear_nu_mu)
    _show("sum of both orderings", bilinear_mu_nu + bilinear_nu_mu)

    bilinear_down_down = _vertex(
        g_bilin
        * gamma_lowered_matrix(i_bar, alpha, mu, rho_mu)
        * gamma_lowered_matrix(alpha, i_psi, nu, rho_nu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    print("gBilin * psibar gamma_mu gamma_nu psi")
    print(
        "equivalent after metric simplification:",
        _gamma_simplified(bilinear_mu_nu).to_canonical_string()
        == _gamma_simplified(bilinear_down_down).to_canonical_string(),
    )
    print()

    print("== current-current operator ==")
    current_current = _vertex(
        g_bilin
        * gamma_matrix(s1, s2, mu)
        * gamma_lowered_matrix(s3, s4, mu, rho_mu),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
    )
    _show("gBilin * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]", current_current)

    axial_current_current = _vertex(
        g_bilin
        * gamma_matrix(s1, alpha, mu)
        * gamma5_matrix(alpha, s2)
        * gamma_lowered_matrix(s3, beta, mu, rho_mu)
        * gamma5_matrix(beta, s4),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
    )
    _show(
        "gBilin * (psibar gamma^mu gamma5 psi)(psibar gamma_mu gamma5 psi)  [stripped]",
        axial_current_current,
    )

    left_projected_current_current = _vertex(
        g_bilin
        * gamma_matrix(s1, alpha, mu)
        * chiral_projector_left(alpha, s2)
        * gamma_lowered_matrix(s3, beta, mu, rho_mu)
        * chiral_projector_left(beta, s4),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
    )
    _show(
        "gBilin * (psibar gamma^mu P_L psi)(psibar gamma_mu P_L psi)  [stripped]",
        left_projected_current_current,
    )

    print("== tests ==")
    i1_sym, i2_sym, i3_sym, i4_sym = S("i1", "i2", "i3", "i4")

    _check(
        anticom,
        2 * lorentz_metric("mu", "nu") * spinor_metric("si", "sk"),
        "Clifford anticommutator",
    )
    _check(
        bilinear_up,
        I * g_bilin * gamma_matrix(i1_sym, i2_sym, mu),
        "psibar gamma^mu psi",
    )
    _check(
        bilinear_g5,
        I * g_bilin * gamma5_matrix(i1_sym, i2_sym),
        "psibar gamma5 psi",
    )
    _check(
        bilinear_sigma,
        I * g_bilin * sigma_tensor(i1_sym, i2_sym, mu, nu),
        "psibar sigma^{mu nu} psi",
    )
    _check(
        _gamma_simplified((bilinear_pl + bilinear_pr).expand()),
        I * g_bilin * spinor_metric(i1_sym, i2_sym),
        "P_L + P_R = 1",
    )
    _check(
        _gamma_simplified((bilinear_pr - bilinear_pl).expand()),
        I * g_bilin * gamma5_matrix(i1_sym, i2_sym),
        "P_R - P_L = gamma5",
    )
    _check(
        bilinear_down,
        I
        * g_bilin
        * gamma_lowered_matrix(i1_sym, i2_sym, mu, rho_mu),
        "psibar gamma_mu psi",
    )
    _check(
        bilinear_mu_nu,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, mu)
        * gamma_matrix(alpha, i2_sym, nu),
        "psibar gamma^mu gamma^nu psi",
    )
    _check(
        bilinear_nu_mu,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, nu)
        * gamma_matrix(alpha, i2_sym, mu),
        "psibar gamma^nu gamma^mu psi",
    )
    _check(
        bilinear_down_down,
        I
        * g_bilin
        * gamma_lowered_matrix(i1_sym, alpha, mu, rho_mu)
        * gamma_lowered_matrix(alpha, i2_sym, nu, rho_nu),
        "psibar gamma_mu gamma_nu psi",
    )
    _check(
        _gamma_simplified((bilinear_mu_nu + bilinear_nu_mu).expand()),
        2
        * I
        * g_bilin
        * lorentz_metric(mu, nu)
        * spinor_metric(i1_sym, i2_sym),
        "gamma^mu gamma^nu + gamma^nu gamma^mu",
    )
    _check(
        _gamma_simplified(current_current),
        2
        * I
        * g_bilin
        * (
            gamma_matrix(i1_sym, i2_sym, mu) * gamma_matrix(i3_sym, i4_sym, mu)
            - gamma_matrix(i1_sym, i4_sym, mu) * gamma_matrix(i3_sym, i2_sym, mu)
        ),
        "stripped current-current",
    )
    _check(
        _display_expr(bilinear_mu_nu_rho).expand(),
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, mu)
        * gamma_matrix(alpha, beta, nu)
        * gamma_matrix(beta, i2_sym, rho),
        "raw psibar gamma^mu gamma^nu gamma^rho psi",
    )
    _check(
        _gamma_simplified(bilinear_mu_nu_rho),
        I
        * g_bilin
        * (
            gamma_matrix(alpha, i2_sym, rho)
            * gamma_matrix(beta, alpha, nu)
            * gamma_matrix(i1_sym, beta, mu)
            - 2 * lorentz_metric(mu, nu) * gamma_matrix(i1_sym, i2_sym, rho)
            + 2 * lorentz_metric(mu, rho) * gamma_matrix(i1_sym, i2_sym, nu)
        ),
        "simplified psibar gamma^mu gamma^nu gamma^rho psi",
    )
    _check(
        bilinear_mu_nu_rho,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, mu)
        * gamma_matrix(alpha, beta, nu)
        * gamma_matrix(beta, i2_sym, rho),
        "psibar gamma^mu gamma^nu gamma^rho psi",
    )
    _check(
        bilinear_mu_nu_g5,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, mu)
        * gamma_matrix(alpha, beta, nu)
        * gamma5_matrix(beta, i2_sym),
        "psibar gamma^mu gamma^nu gamma5 psi",
    )

    print("\nAll selected tests passed.")
