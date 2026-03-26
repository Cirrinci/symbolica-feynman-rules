"""
Focused gamma-matrix checks for the Symbolica/Spenso prototype.

This file is intentionally simple: each example is written out directly,
printed directly, and then checked directly.

Currently covered here:
- Clifford anticommutator
- raised vs lowered gamma bilinears
- gamma^mu gamma^nu vs gamma^nu gamma^mu ordering
- stripped current-current operator
"""

from symbolica import Expression

from model_symbolica import Delta, I, S, simplify_deltas, vertex_factor
from spenso_structures import (
    gamma_anticommutator,
    gamma_lowered_matrix,
    gamma_matrix,
    hep_tensor_scalar,
    lorentz_metric,
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


def _show(title, expr):
    print(title)
    print(simplify_gamma_chain(expr).expand())
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
        simplify_gamma_chain(bilinear_up).expand().to_canonical_string()
        == simplify_gamma_chain(bilinear_down).expand().to_canonical_string(),
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
        simplify_gamma_chain(bilinear_mu_nu).expand().to_canonical_string()
        == simplify_gamma_chain(bilinear_down_down).expand().to_canonical_string(),
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

    print("== tests ==")
    i1_sym, i2_sym, i3_sym, i4_sym = S("i1", "i2", "i3", "i4")
    total_p2 = p1 + p2
    total_p4 = p1 + p2 + p3 + p4

    _check(
        anticom,
        2 * lorentz_metric("mu", "nu") * spinor_metric("si", "sk"),
        "Clifford anticommutator",
    )
    _check(
        bilinear_up,
        I * g_bilin * gamma_matrix(i1_sym, i2_sym, mu) * (2 * Expression.PI) ** d * Delta(total_p2),
        "psibar gamma^mu psi",
    )
    _check(
        bilinear_down,
        I
        * g_bilin
        * gamma_lowered_matrix(i1_sym, i2_sym, mu, rho_mu)
        * (2 * Expression.PI) ** d
        * Delta(total_p2),
        "psibar gamma_mu psi",
    )
    _check(
        bilinear_mu_nu,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, mu)
        * gamma_matrix(alpha, i2_sym, nu)
        * (2 * Expression.PI) ** d
        * Delta(total_p2),
        "psibar gamma^mu gamma^nu psi",
    )
    _check(
        bilinear_nu_mu,
        I
        * g_bilin
        * gamma_matrix(i1_sym, alpha, nu)
        * gamma_matrix(alpha, i2_sym, mu)
        * (2 * Expression.PI) ** d
        * Delta(total_p2),
        "psibar gamma^nu gamma^mu psi",
    )
    _check(
        bilinear_down_down,
        I
        * g_bilin
        * gamma_lowered_matrix(i1_sym, alpha, mu, rho_mu)
        * gamma_lowered_matrix(alpha, i2_sym, nu, rho_nu)
        * (2 * Expression.PI) ** d
        * Delta(total_p2),
        "psibar gamma_mu gamma_nu psi",
    )
    _check(
        simplify_gamma_chain((bilinear_mu_nu + bilinear_nu_mu).expand()),
        2
        * I
        * g_bilin
        * lorentz_metric(mu, nu)
        * spinor_metric(i1_sym, i2_sym)
        * (2 * Expression.PI) ** d
        * Delta(total_p2),
        "gamma^mu gamma^nu + gamma^nu gamma^mu",
    )
    _check(
        simplify_gamma_chain(current_current),
        2
        * I
        * g_bilin
        * (2 * Expression.PI) ** d
        * (
            gamma_matrix(i1_sym, i2_sym, mu) * gamma_matrix(i3_sym, i4_sym, mu)
            - gamma_matrix(i1_sym, i4_sym, mu) * gamma_matrix(i3_sym, i2_sym, mu)
        )
        * Delta(total_p4),
        "stripped current-current",
    )

    print("\nAll selected tests passed.")
