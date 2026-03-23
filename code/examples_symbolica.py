"""
Scalar vertex examples using (model_symbolica.py).

"""

import argparse

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
    UF,
    UbarF,
    gamma,
    delta,
    Delta,
    pcomp,
    vertex_factor,
    simplify_deltas,
    infer_derivative_targets,
    compact_vertex_sum_form,
    compact_sum_notation,
)

# ---------------------------------------------------------------------------
# Common symbols
# ---------------------------------------------------------------------------

x = S("x")
d = S("d")

p1, p2, p3, p4, p5, p6 = S("p1", "p2", "p3", "p4", "p5", "p6")
b1, b2, b3, b4, b5, b6 = S("b1", "b2", "b3", "b4", "b5", "b6")

phi0 = S("phi0")
chi0 = S("chi0")

mu, nu = S("mu", "nu")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def show_vertex(
    title,
    *,
    coupling,
    alphas,
    betas,
    ps,
    derivative_indices=(),
    derivative_targets=None,
    statistics="boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices=None,
    leg_spins=None,
    strip_externals=True,
    species_map=None,
    compact_override=None,
    show_sum_notation=False,
):
    print("=" * 80)
    print(f"  {title}")
    print(f"  alphas = {alphas}")
    print(f"  betas  = {betas}")
    print(f"  ps     = {ps}")

    V = vertex_factor(
        coupling=coupling,
        alphas=alphas,
        betas=betas,
        ps=ps,
        x=x,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        leg_spins=leg_spins,
        strip_externals=strip_externals,
        include_delta=True,
        d=d,
    )
    compact = compact_override
    if compact is None:
        compact = simplify_deltas(V, species_map=species_map)

    print("\n  Raw vertex:")
    print(" ", V)
    print("\n  Compact form:")
    print(" ", compact)
    if show_sum_notation and derivative_indices:
        print("  Sum notation:")
        print(
            " ",
            compact_sum_notation(
                derivative_indices=derivative_indices,
                derivative_targets=derivative_targets,
                n_legs=len(ps),
            ),
        )
    print()
    return V, compact


# ---------------------------------------------------------------------------
# 1) Single-species phi^4:  L_int = lam4 * phi^4
# ---------------------------------------------------------------------------

lam4 = S("lam4")

L_phi4 = dict(
    coupling=lam4,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# ---------------------------------------------------------------------------
# 2) Single-species phi^6:  L_int = lam6 * phi^6
# ---------------------------------------------------------------------------

lam6 = S("lam6")

L_phi6 = dict(
    coupling=lam6,
    alphas=[phi0] * 6,
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# ---------------------------------------------------------------------------
# 3) Two-species phi^2 chi^2:  L_int = g * phi^2 * chi^2
# ---------------------------------------------------------------------------

g_sym = S("g")

L_phi2chi2 = dict(
    coupling=g_sym,
    alphas=[phi0, phi0, chi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# ---------------------------------------------------------------------------
# 4) Derivative interaction:  L_int = gD * (d_mu phi)(d_nu phi) phi phi
# ---------------------------------------------------------------------------

gD = S("gD")

deriv_indices, deriv_targets = infer_derivative_targets([(0, [mu]), (1, [nu])])

L_deriv = dict(
    coupling=gD,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices,
    derivative_targets=deriv_targets,
)

# ---------------------------------------------------------------------------
# 5) Derivative interaction (same index):
#    L_int = gD2 * (d_mu phi)(d_mu phi) phi phi
# ---------------------------------------------------------------------------

gD2 = S("gD2")

deriv_indices2, deriv_targets2 = infer_derivative_targets([(0, [mu]), (1, [mu])])

L_deriv2 = dict(
    coupling=gD2,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2,
    derivative_targets=deriv_targets2,
)

# ---------------------------------------------------------------------------
# 6) Multi-species:  L_int = gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2
# ---------------------------------------------------------------------------

idx_i, idx_j, idx_k = S("i", "j", "k")
gijk = S("gijk")

L_multi = dict(
    coupling=gijk(idx_i, idx_j, idx_k),
    alphas=[idx_i, idx_i, idx_j, idx_j, idx_k, idx_k],
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# ---------------------------------------------------------------------------
# 7) Fermion Yukawa-like toy term: L_int = yF * psibar * psi * phi
# ---------------------------------------------------------------------------

yF = S("yF")
psibar0, psi0 = S("psibar0", "psi0")
i_psi_bar, i_psi = S("i_psi_bar", "i_psi")
s1, s2, s3, s4 = S("s1", "s2", "s3", "s4")
A0 = S("A0")

L_yukawa = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

# Vector current toy term: gV * psibar gamma^mu psi A_mu
gV = S("gV")
L_vec_current = dict(
    coupling=gV * gamma(mu, i_psi_bar, i_psi),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

# Four-fermion toy ordering: psi psibar psi psibar
g4F = S("g4F")
L_4fermion = dict(
    coupling=g4F,
    alphas=[psi0, psibar0, psi0, psibar0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psi", "psibar", "psi", "psibar"],
    leg_roles=["psi", "psibar", "psi", "psibar"],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

#something very repetitive here, but it's just to test the vertex_factor function
COMPACT_DERIV = compact_vertex_sum_form(
    coupling=gD,
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices,
    derivative_targets=deriv_targets,
    d=d,
    field_species=[phi0, phi0, phi0, phi0],
    leg_species=[phi0, phi0, phi0, phi0],
)

COMPACT_DERIV2 = compact_vertex_sum_form(
    coupling=gD2,
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2,
    derivative_targets=deriv_targets2,
    d=d,
    field_species=[phi0, phi0, phi0, phi0],
    leg_species=[phi0, phi0, phi0, phi0],
)


def _run_scalar_tests():
    # phi^4
    V_phi4 = vertex_factor(**L_phi4, x=x, d=d)
    V_phi4_s = simplify_deltas(
        V_phi4, species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    )
    expected_phi4 = 24 * I * lam4 * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert (
        V_phi4_s.expand().to_canonical_string()
        == expected_phi4.expand().to_canonical_string()
    ), f"phi^4 failed: {V_phi4_s}"
    print("phi^4 vertex: PASS  (24 * i * lam4)")

    # phi^2 chi^2
    V_phi2chi2 = vertex_factor(**L_phi2chi2, x=x, d=d)
    V_phi2chi2_s = simplify_deltas(
        V_phi2chi2, species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0}
    )
    expected_phi2chi2 = 4 * I * g_sym * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert (
        V_phi2chi2_s.expand().to_canonical_string()
        == expected_phi2chi2.expand().to_canonical_string()
    ), f"phi^2 chi^2 failed: {V_phi2chi2_s}"
    print("phi^2 chi^2 vertex: PASS  (4 * i * g)")

    # Derivative vertex
    V_deriv = vertex_factor(**L_deriv, x=x, d=d)
    V_deriv_s = simplify_deltas(
        V_deriv, species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    )
    expected_deriv = COMPACT_DERIV
    assert (
        V_deriv_s.expand().to_canonical_string()
        == expected_deriv.expand().to_canonical_string()
    ), f"Derivative vertex failed: {V_deriv_s}"
    print("Derivative vertex: PASS  (permutation-aware momentum assignment)")

    # Derivative vertex (same index)
    V_deriv2 = vertex_factor(**L_deriv2, x=x, d=d)
    V_deriv2_s = simplify_deltas(
        V_deriv2, species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    )
    expected_deriv2 = COMPACT_DERIV2
    assert (
        V_deriv2_s.expand().to_canonical_string()
        == expected_deriv2.expand().to_canonical_string()
    ), f"Derivative vertex (same index) failed: {V_deriv2_s}"
    print("Derivative vertex (same mu): PASS  (contracted Lorentz indices)")

    # Multi-species: each distinct assignment pattern has factor 8
    V_multi = vertex_factor(**L_multi, x=x, d=d)
    V_multi_str = str(V_multi)
    assert "gijk" in V_multi_str, "Multi-species vertex must contain gijk"
    assert "delta" in V_multi_str, "Multi-species vertex must contain delta"
    print("Multi-species vertex: PASS  (contains gijk, delta)")

    print("\nScalar+derivative tests passed.")


def _run_fermion_tests():
    V_yuk = vertex_factor(**L_yukawa, x=x, d=d)
    V_yuk_s = simplify_deltas(
        V_yuk, species_map={b1: psibar0, b2: psi0, b3: phi0}
    )
    expected_yuk = I * yF * (2 * pi) ** d * Delta(p1 + p2 + p3)
    assert (
        V_yuk_s.expand().to_canonical_string()
        == expected_yuk.expand().to_canonical_string()
    ), f"Yukawa-like fermion vertex failed: {V_yuk_s}"
    print("Fermion Yukawa-like vertex: PASS  (role-aware contraction)")

    # Unstripped form must carry external u_s/ubar_s wavefunctions.
    V_yuk_full = vertex_factor(**L_yukawa, x=x, d=d, strip_externals=False)
    V_yuk_full_s = simplify_deltas(
        V_yuk_full, species_map={b1: psibar0, b2: psi0, b3: phi0}
    )
    V_yuk_full_str = V_yuk_full_s.to_canonical_string()
    assert "UbarF" in V_yuk_full_str and "UF" in V_yuk_full_str, (
        f"Expected UF/UbarF in unstripped Yukawa vertex, got: {V_yuk_full_s}"
    )
    print("Fermion Yukawa wavefunctions: PASS  (contains UF and UbarF)")

    V_vec = vertex_factor(**L_vec_current, x=x, d=d)
    V_vec_s = simplify_deltas(
        V_vec, species_map={b1: psibar0, b2: psi0, b3: A0}
    )
    expected_vec = I * gV * gamma(mu, i_psi_bar, i_psi) * (2 * pi) ** d * Delta(
        p1 + p2 + p3
    )
    assert (
        V_vec_s.expand().to_canonical_string()
        == expected_vec.expand().to_canonical_string()
    ), f"Vector-current fermion vertex failed: {V_vec_s}"
    print("Fermion vector current: PASS  (gamma structure preserved)")

    # With this ordering and identical species, fermionic permutation signs cancel.
    V_4f = vertex_factor(**L_4fermion, x=x, d=d)
    V_4f_s = simplify_deltas(
        V_4f, species_map={b1: psi0, b2: psibar0, b3: psi0, b4: psibar0}
    )
    expected_4f = Expression.num(0)
    assert (
        V_4f_s.expand().to_canonical_string()
        == expected_4f.to_canonical_string()
    ), f"Four-fermion psi-psibar-psi-psibar test failed: {V_4f_s}"
    print("Four-fermion psi psibar psi psibar: PASS  (sign cancellation)")

    print("\nFermion tests passed.")


def _run_suite_tests(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_tests()
    if suite in ("fermion", "all"):
        _run_fermion_tests()
    print("\nAll selected tests passed.")


def _run_scalar_demo():
    print("\n=== phi^4 ===")
    show_vertex(
        "lam4 * phi^4",
        **L_phi4,
        species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0},
    )

    print("\n=== phi^6 ===")
    show_vertex(
        "lam6 * phi^6",
        **L_phi6,
        species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0, b5: phi0, b6: phi0},
    )

    print("\n=== phi^2 chi^2 ===")
    show_vertex(
        "g * phi^2 * chi^2",
        **L_phi2chi2,
        species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0},
    )

    print("\n=== derivative: gD * (d_mu phi)(d_nu phi) phi phi ===")
    show_vertex(
        "gD * (d_mu phi)(d_nu phi) phi phi",
        **L_deriv,
        species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0},
        compact_override=COMPACT_DERIV,
        show_sum_notation=True,
    )

    print("\n=== derivative (same index): gD2 * (d_mu phi)(d_mu phi) phi phi ===")
    show_vertex(
        "gD2 * (d_mu phi)(d_mu phi) phi phi",
        **L_deriv2,
        species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0},
        compact_override=COMPACT_DERIV2,
        show_sum_notation=True,
    )

    print("\n=== multi-species: gijk * phi_i^2 phi_j^2 phi_k^2 ===")
    show_vertex("gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2", **L_multi)


def _run_fermion_demo():
    print("\n=== fermion: yF * psibar * psi * phi ===")
    show_vertex(
        "yF * psibar * psi * phi",
        **L_yukawa,
        species_map={b1: psibar0, b2: psi0, b3: phi0},
    )
    print("\n=== fermion unstripped: yF * psibar * psi * phi ===")
    show_vertex(
        "yF * psibar * psi * phi  [unstripped]",
        **L_yukawa,
        strip_externals=False,
        species_map={b1: psibar0, b2: psi0, b3: phi0},
    )

    print("\n=== fermion: gV * psibar gamma^mu psi A_mu ===")
    show_vertex(
        "gV * psibar gamma^mu psi A_mu",
        **L_vec_current,
        species_map={b1: psibar0, b2: psi0, b3: A0},
    )

    print("\n=== fermion: g4F * psi * psibar * psi * psibar ===")
    show_vertex(
        "g4F * psi * psibar * psi * psibar",
        **L_4fermion,
        species_map={b1: psi0, b2: psibar0, b3: psi0, b4: psibar0},
    )


def _run_suite_demo(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Symbolica examples by suite."
    )
    parser.add_argument(
        "--suite",
        choices=("scalar", "fermion", "all"),
        default="scalar",
        help="Which example suite to run.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Run demo output only.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    print(f"\n=== Running suite: {args.suite} ===")
    _run_suite_demo(args.suite)
    if not args.skip_tests:
        print("\n=== Running tests ===")
        _run_suite_tests(args.suite)
