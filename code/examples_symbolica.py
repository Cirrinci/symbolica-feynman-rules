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
    bis,
    Delta,
    pcomp,
    vertex_factor,
    simplify_deltas,
    simplify_spinor_indices,
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
    leg_spinor_indices=None,
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
        leg_spinor_indices=leg_spinor_indices,
        strip_externals=strip_externals,
        include_delta=True,
        d=d,
    )
    compact = compact_override
    if compact is None:
        compact = simplify_deltas(V, species_map=species_map)

    print("\n  Simplified (after simplify_deltas):")
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
# 9) Four-fermion bilinear: L_int = -(g/2) (psibar psi)(psibar psi)
#    Spinor indices: alpha shared in first bilinear, beta in second.
#    Stripped → 0 (Fierz cancellation without spinor structure).
#    Unstripped → non-zero (direct − exchange channel in spinor space).
# ---------------------------------------------------------------------------

g_psi4 = S("g_psi4")
alpha_s, beta_s = S("alpha_s", "beta_s")

L_psibar_psi_sq = dict(
    coupling=-g_psi4 / Expression.num(2),
    alphas=[psibar0, psi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "psibar", "psi"],
    leg_roles=["psibar", "psi", "psibar", "psi"],
    field_spinor_indices=[alpha_s, alpha_s, beta_s, beta_s],
    leg_spins=[s1, s2, s3, s4],
)

# Same interaction with leg_spinor_indices → spinor delta output
i1, i2, i3, i4 = S("i1", "i2", "i3", "i4")

L_psibar_psi_sq_spinor = dict(
    coupling=-g_psi4 / Expression.num(2),
    alphas=[psibar0, psi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "psibar", "psi"],
    leg_roles=["psibar", "psi", "psibar", "psi"],
    field_spinor_indices=[alpha_s, alpha_s, beta_s, beta_s],
    leg_spins=[s1, s2, s3, s4],
    leg_spinor_indices=[i1, i2, i3, i4],
)

# ---------------------------------------------------------------------------
# 10) Mixed derivative fermion+scalar interactions (4+ external legs)
# ---------------------------------------------------------------------------
#
# Reuse existing couplings (yF, g_sym) as requested.
#
# A) yF * (d_mu psibar) * psi * phi * chi
L_mix_dpsibar = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=[mu],
    derivative_targets=[0],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None, None],
    leg_spins=[s1, s2, s3, s4],
)

# B) yF * psibar * (d_nu psi) * phi * chi
L_mix_dpsi = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=[nu],
    derivative_targets=[1],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None, None],
    leg_spins=[s1, s2, s3, s4],
)

# C) yF * psibar * psi * (d_mu phi) * (d_nu chi)
L_mix_dphi_dchi = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=[mu, nu],
    derivative_targets=[2, 3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None, None],
    leg_spins=[s1, s2, s3, s4],
)

# D) g * (d_mu psibar) * (d_nu psi) * phi * phi * chi
L_mix_5pt = dict(
    coupling=g_sym,
    alphas=[psibar0, psi0, phi0, phi0, chi0],
    betas=[b1, b2, b3, b4, b5],
    ps=[p1, p2, p3, p4, p5],
    derivative_indices=[mu, nu],
    derivative_targets=[0, 1],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None, None, None],
    leg_spins=[s1, s2, s3, s4, s1],
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

    # Multi-species: gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2
    # For a fixed external assignment with two legs of each species, the
    # combinatorial factor is 2! * 2! * 2! = 8.
    V_multi = vertex_factor(**L_multi, x=x, d=d)
    sm_multi_base = {
        b1: idx_i, b2: idx_i,
        b3: idx_j, b4: idx_j,
        b5: idx_k, b6: idx_k,
    }
    V_multi_base = simplify_deltas(V_multi, species_map=sm_multi_base)
    expected_multi_base = (
        8 * I * gijk(idx_i, idx_j, idx_k)
        * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5 + p6)
    )
    assert (
        V_multi_base.expand().to_canonical_string()
        == expected_multi_base.expand().to_canonical_string()
    ), f"Multi-species base assignment failed: {V_multi_base}"

    # Same species multiplicities, different leg ordering -> same compact result.
    sm_multi_perm = {
        b1: idx_j, b2: idx_k,
        b3: idx_i, b4: idx_j,
        b5: idx_k, b6: idx_i,
    }
    V_multi_perm = simplify_deltas(V_multi, species_map=sm_multi_perm)
    expected_multi_perm = (
        8 * I * gijk(idx_i, idx_j, idx_k)
        * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5 + p6)
    )
    assert (
        V_multi_perm.expand().to_canonical_string()
        == expected_multi_perm.expand().to_canonical_string()
    ), f"Multi-species permuted assignment failed: {V_multi_perm}"

    # Wrong multiplicity (3,2,1) is incompatible with phi_i^2 phi_j^2 phi_k^2 -> 0.
    sm_multi_bad = {
        b1: idx_i, b2: idx_i, b3: idx_i,
        b4: idx_j, b5: idx_j, b6: idx_k,
    }
    V_multi_bad = simplify_deltas(V_multi, species_map=sm_multi_bad)
    assert (
        V_multi_bad.expand().to_canonical_string()
        == Expression.num(0).to_canonical_string()
    ), f"Multi-species incompatible multiplicity should vanish: {V_multi_bad}"
    print("Multi-species vertex: PASS  (8 factor + permutation invariance + incompatible=0)")

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

    # (psibar psi)^2: stripped must be 0, unstripped must be non-zero.
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    V_pp2 = vertex_factor(**L_psibar_psi_sq, x=x, d=d, strip_externals=True)
    V_pp2_s = simplify_deltas(V_pp2, species_map=sm4)
    assert (
        V_pp2_s.expand().to_canonical_string()
        == Expression.num(0).to_canonical_string()
    ), f"(psibar psi)^2 stripped should be 0, got: {V_pp2_s}"
    print("(psibar psi)^2 stripped: PASS  (0 — spinor structure erased)")

    V_pp2_full = vertex_factor(**L_psibar_psi_sq, x=x, d=d, strip_externals=False)
    V_pp2_full_s = simplify_deltas(V_pp2_full, species_map=sm4)
    V_pp2_str = V_pp2_full_s.to_canonical_string()
    assert V_pp2_str != Expression.num(0).to_canonical_string(), (
        f"(psibar psi)^2 unstripped should be non-zero"
    )
    assert "UbarF" in V_pp2_str and "UF" in V_pp2_str, (
        f"Expected UF/UbarF in unstripped (psibar psi)^2, got: {V_pp2_full_s}"
    )
    print("(psibar psi)^2 unstripped: PASS  (non-zero Fierz structure)")

    # Spinor-delta form: should give -ig * (2pi)^d * Delta * [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]
    # where g(...) is the Spenso bispinor metric bis.g(...)
    V_sp = vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d)
    V_sp = simplify_deltas(V_sp, species_map=sm4)
    expected_sp = (
        -I * g_psi4 * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
        * (bis.g(i1, i2).to_expression() * bis.g(i3, i4).to_expression()
           - bis.g(i1, i4).to_expression() * bis.g(i3, i2).to_expression())
    )
    assert (
        V_sp.expand().to_canonical_string()
        == expected_sp.expand().to_canonical_string()
    ), f"(psibar psi)^2 spinor-delta form failed:\n  got:      {V_sp}\n  expected: {expected_sp}"
    print("(psibar psi)^2 spinor deltas: PASS  (-ig)[g₁₂g₃₄ - g₁₄g₃₂]")

    print("\nFermion tests passed.")


def _run_fermion_derivative_mixed_tests():
    """Extra derivative tests mixing fermions and scalars with >=4 legs."""
    sm4 = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}

    # A) derivative on psibar slot -> picks p1, one derivative => no overall i.
    V_a = simplify_deltas(vertex_factor(**L_mix_dpsibar, x=x, d=d), species_map=sm4)
    expected_a = yF * pcomp(p1, mu) * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert (
        V_a.expand().to_canonical_string()
        == expected_a.expand().to_canonical_string()
    ), f"Mixed derivative test A failed: {V_a}"
    print("Mixed deriv A (d_mu psibar) : PASS")

    # B) derivative on psi slot -> picks p2.
    V_b = simplify_deltas(vertex_factor(**L_mix_dpsi, x=x, d=d), species_map=sm4)
    expected_b = yF * pcomp(p2, nu) * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    assert (
        V_b.expand().to_canonical_string()
        == expected_b.expand().to_canonical_string()
    ), f"Mixed derivative test B failed: {V_b}"
    print("Mixed deriv B (d_nu psi)    : PASS")

    # C) derivatives on phi and chi slots -> picks p3 and p4, phase i*(-i)^2 = -i.
    V_c = simplify_deltas(vertex_factor(**L_mix_dphi_dchi, x=x, d=d), species_map=sm4)
    expected_c = (
        -I * yF * pcomp(p3, mu) * pcomp(p4, nu)
        * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    )
    assert (
        V_c.expand().to_canonical_string()
        == expected_c.expand().to_canonical_string()
    ), f"Mixed derivative test C failed: {V_c}"
    print("Mixed deriv C (d_mu phi)(d_nu chi): PASS")

    # D) 5-point mixed case with two identical phis -> combinatorial factor 2.
    sm5 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0}
    V_d = simplify_deltas(vertex_factor(**L_mix_5pt, x=x, d=d), species_map=sm5)
    expected_d = (
        -2 * I * g_sym * pcomp(p1, mu) * pcomp(p2, nu)
        * (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5)
    )
    assert (
        V_d.expand().to_canonical_string()
        == expected_d.expand().to_canonical_string()
    ), f"Mixed derivative test D failed: {V_d}"
    print("Mixed deriv D 5pt (with phi-phi combinatorics): PASS")

    print("\nMixed fermion+scalar derivative tests passed.")


def _relation_to_original(candidate, original):
    cand = candidate.expand().to_canonical_string()
    orig = original.expand().to_canonical_string()
    if cand == orig:
        return "== V_orig"
    if cand == (-original).expand().to_canonical_string():
        return "== -V_orig"
    return "different (not ±V_orig)"


def _run_swap_diagnostics():
    """Optional diagnostic prints for operator-order sensitivity."""
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    V_orig = simplify_deltas(vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d), species_map=sm4)

    # Swap inside BOTH bilinears: (psibar psi)(psibar psi) -> (psi psibar)(psi psibar)
    L_swap_both = dict(L_psibar_psi_sq_spinor)
    L_swap_both.update(
        alphas=[psi0, psibar0, psi0, psibar0],
        field_roles=["psi", "psibar", "psi", "psibar"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_both = simplify_deltas(vertex_factor(**L_swap_both, x=x, d=d), species_map=sm4)
    print("(psibar psi)^2 swap-both bilinears:")
    print("  relation:", _relation_to_original(V_swap_both, V_orig))

    # Swap inside ONE bilinear
    L_swap_one = dict(L_psibar_psi_sq_spinor)
    L_swap_one.update(
        alphas=[psi0, psibar0, psibar0, psi0],
        field_roles=["psi", "psibar", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_one = simplify_deltas(vertex_factor(**L_swap_one, x=x, d=d), species_map=sm4)
    print("(psibar psi)^2 swap-one bilinear:")
    print("  relation:", _relation_to_original(V_swap_one, V_orig))


def _run_suite_tests(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_tests()
    if suite in ("fermion", "all"):
        _run_fermion_tests()
        _run_fermion_derivative_mixed_tests()
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
    show_vertex(
        "gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2",
        **L_multi,
        species_map={
            b1: idx_i, b2: idx_i,
            b3: idx_j, b4: idx_j,
            b5: idx_k, b6: idx_k,
        },
    )


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

    print("\n=== fermion: -(g/2)(psibar psi)(psibar psi)  [stripped → 0] ===")
    show_vertex(
        "-(g/2)(psibar psi)(psibar psi)  [stripped]",
        **L_psibar_psi_sq,
        species_map={b1: psibar0, b2: psi0, b3: psibar0, b4: psi0},
    )

    print("\n=== fermion: -(g/2)(psibar psi)(psibar psi)  [unstripped → spinor structure] ===")
    show_vertex(
        "-(g/2)(psibar psi)(psibar psi)  [unstripped]",
        **L_psibar_psi_sq,
        strip_externals=False,
        species_map={b1: psibar0, b2: psi0, b3: psibar0, b4: psi0},
    )

    print("\n=== fermion: -(g/2)(psibar psi)(psibar psi)  [spinor delta form] ===")
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    V_spinor = vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d)
    V_spinor = simplify_deltas(V_spinor, species_map=sm4)
    print("=" * 80)
    print("  -(g/2)(psibar psi)(psibar psi)  →  V = (-ig)[g(i1,i2)*g(i3,i4) - ...]")
    print(f"\n  Result:\n  {V_spinor}")
    print()
    _run_swap_diagnostics()

    print("\n=== mixed derivatives (fermion+scalar): yF * (d_mu psibar) * psi * phi * chi ===")
    show_vertex(
        "yF * (d_mu psibar) * psi * phi * chi",
        **L_mix_dpsibar,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0},
    )

    print("\n=== mixed derivatives (fermion+scalar): yF * psibar * (d_nu psi) * phi * chi ===")
    show_vertex(
        "yF * psibar * (d_nu psi) * phi * chi",
        **L_mix_dpsi,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0},
    )

    print("\n=== mixed derivatives (fermion+scalar): yF * psibar * psi * (d_mu phi) * (d_nu chi) ===")
    show_vertex(
        "yF * psibar * psi * (d_mu phi) * (d_nu chi)",
        **L_mix_dphi_dchi,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: chi0},
    )

    print("\n=== mixed derivatives 5pt: g * (d_mu psibar) * (d_nu psi) * phi * phi * chi ===")
    show_vertex(
        "g * (d_mu psibar) * (d_nu psi) * phi * phi * chi",
        **L_mix_5pt,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0},
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
        default="all",
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

