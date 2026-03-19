"""
Scalar vertex examples using (model_symbolica.py).

"""

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
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
        strip_externals=True,
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


#now the tests

def _run_tests():
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

    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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

    print("\n=== Running tests ===")
    _run_tests()
