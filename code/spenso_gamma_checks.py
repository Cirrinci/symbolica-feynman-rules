"""
Focused gamma-matrix checks for the Symbolica/Spenso prototype.

This script keeps only the examples that are currently useful:
- basic gamma identities
- simple bilinears with raised/lowered Lorentz indices
- current-current four-fermion operators in stripped and unstripped form
"""

import argparse

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
s1 = S("s1")
s2 = S("s2")
s3 = S("s3")
s4 = S("s4")


def _check(got, expected, label):
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


def _compute_vertex(*, coupling, alphas, betas, ps, field_roles, leg_roles, field_spinor_indices, strip_externals=True):
    vertex = vertex_factor(
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
    )
    return simplify_deltas(
        vertex,
        species_map={beta: species for beta, species in zip(betas, alphas)},
    )


def _demo_gamma_identities():
    print("== gamma identities ==")
    print("Tr(gamma^mu gamma_mu) =", hep_tensor_scalar(gamma_matrix("si", "sj", "mu") * gamma_matrix("sj", "si", "mu")))
    print()
    print("{gamma^mu, gamma^nu} =")
    print(simplify_gamma_chain(gamma_anticommutator("si", "sk", "mu", "nu")))
    print()


def _demo_bilinears():
    gamma_up = _compute_vertex(
        coupling=g_bilin * gamma_matrix(i_bar, i_psi, mu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    gamma_down = _compute_vertex(
        coupling=g_bilin * gamma_lowered_matrix(i_bar, i_psi, mu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    gamma2_up = _compute_vertex(
        coupling=g_bilin * gamma_matrix(i_bar, alpha, mu) * gamma_matrix(alpha, i_psi, nu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )
    gamma2_down = _compute_vertex(
        coupling=g_bilin * gamma_lowered_matrix(i_bar, alpha, mu) * gamma_lowered_matrix(alpha, i_psi, nu),
        alphas=[psibar0, psi0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=["psibar", "psi"],
        leg_roles=["psibar", "psi"],
        field_spinor_indices=[i_bar, i_psi],
    )

    print("== bilinears ==")
    print("gBilin psibar gamma^mu psi")
    print(gamma_up)
    print()
    print("gBilin psibar gamma_mu psi")
    print(gamma_down)
    print()
    print("gBilin psibar gamma^mu gamma^nu psi")
    print(gamma2_up)
    print()
    print("gBilin psibar gamma_mu gamma_nu psi")
    print(gamma2_down)
    print()

    return gamma_up, gamma_down, gamma2_up, gamma2_down


def _demo_current_current():
    stripped_single = _compute_vertex(
        coupling=g_bilin * gamma_matrix(s1, s2, mu) * gamma_lowered_matrix(s3, s4, mu),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
        strip_externals=True,
    )
    unstripped_single = _compute_vertex(
        coupling=g_bilin * gamma_matrix(s1, s2, mu) * gamma_lowered_matrix(s3, s4, mu),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
        strip_externals=False,
    )
    stripped_double = _compute_vertex(
        coupling=(
            g_bilin
            * gamma_matrix(s1, alpha, mu)
            * gamma_matrix(alpha, s2, nu)
            * gamma_lowered_matrix(s3, S("beta"), mu)
            * gamma_lowered_matrix(S("beta"), s4, nu)
        ),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
        strip_externals=True,
    )
    unstripped_double = _compute_vertex(
        coupling=(
            g_bilin
            * gamma_matrix(s1, alpha, mu)
            * gamma_matrix(alpha, s2, nu)
            * gamma_lowered_matrix(s3, S("beta"), mu)
            * gamma_lowered_matrix(S("beta"), s4, nu)
        ),
        alphas=[psibar0, psi0, psibar0, psi0],
        betas=[b1, b2, b3, b4],
        ps=[p1, p2, p3, p4],
        field_roles=["psibar", "psi", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
        field_spinor_indices=[s1, s2, s3, s4],
        strip_externals=False,
    )

    print("== current-current operators ==")
    print("gBilin (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]")
    print(stripped_single)
    print()
    print("gBilin (psibar gamma^mu psi)(psibar gamma_mu psi)  [unstripped]")
    print(unstripped_single)
    print()
    print("gBilin (psibar gamma^mu gamma^nu psi)(psibar gamma_mu gamma_nu psi)  [stripped]")
    print(stripped_double)
    print()
    print("gBilin (psibar gamma^mu gamma^nu psi)(psibar gamma_mu gamma_nu psi)  [unstripped]")
    print(unstripped_double)
    print()

    return stripped_single, unstripped_single, stripped_double, unstripped_double


def _run_tests():
    gamma_up, gamma_down, gamma2_up, gamma2_down = _demo_bilinears()
    stripped_single, unstripped_single, stripped_double, unstripped_double = _demo_current_current()

    _check(
        simplify_gamma_chain(gamma_anticommutator("si", "sk", "mu", "nu")),
        2 * lorentz_metric("mu", "nu") * spinor_metric("si", "sk"),
        "Clifford anticommutator",
    )

    _check(
        gamma_up,
        I * g_bilin * gamma_matrix(i_bar, i_psi, mu) * (2 * Expression.PI) ** d * Delta(p1 + p2),
        "psibar gamma^mu psi",
    )
    _check(
        gamma_down,
        I * g_bilin * gamma_lowered_matrix(i_bar, i_psi, mu) * (2 * Expression.PI) ** d * Delta(p1 + p2),
        "psibar gamma_mu psi",
    )
    _check(
        gamma2_up,
        I * g_bilin * gamma_matrix(i_bar, alpha, mu) * gamma_matrix(alpha, i_psi, nu) * (2 * Expression.PI) ** d * Delta(p1 + p2),
        "psibar gamma^mu gamma^nu psi",
    )
    _check(
        gamma2_down,
        I * g_bilin * gamma_lowered_matrix(i_bar, alpha, mu) * gamma_lowered_matrix(alpha, i_psi, nu) * (2 * Expression.PI) ** d * Delta(p1 + p2),
        "psibar gamma_mu gamma_nu psi",
    )

    assert stripped_single.to_canonical_string() == Expression.num(0).to_canonical_string(), "stripped current-current single gamma should vanish"
    print("  stripped current-current single gamma: PASS")
    assert stripped_double.to_canonical_string() == Expression.num(0).to_canonical_string(), "stripped current-current double gamma should vanish"
    print("  stripped current-current double gamma: PASS")

    assert "UF(" in unstripped_single.to_canonical_string() and "UbarF(" in unstripped_single.to_canonical_string(), "unstripped current-current single gamma should retain external spinors"
    print("  unstripped current-current single gamma: PASS")
    assert "UF(" in unstripped_double.to_canonical_string() and "UbarF(" in unstripped_double.to_canonical_string(), "unstripped current-current double gamma should retain external spinors"
    print("  unstripped current-current double gamma: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run focused Spenso gamma checks.")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("  Demo: spenso-gamma")
    print(f"{'='*80}\n")

    _demo_gamma_identities()
    _demo_bilinears()
    _demo_current_current()

    if not args.skip_tests:
        print(f"\n{'='*80}")
        print("  Tests: spenso-gamma")
        print(f"{'='*80}\n")
        _run_tests()
        print("\nAll selected tests passed.")
