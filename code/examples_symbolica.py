"""
Vertex-factor examples and tests using model_symbolica.py.
"""

import argparse

from model_symbolica import (
    S,
    Expression,
    I,
    pi,
    UF,
    UbarF,
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
from spenso_structures import (
    gamma_lowered_matrix,
    gamma_matrix,
    gamma5_matrix,
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

yF = S("yF")
psibar0, psi0 = S("psibar0", "psi0")
i_psi_bar, i_psi = S("i_psi_bar", "i_psi")
s1, s2, s3, s4 = S("s1", "s2", "s3", "s4")
A0 = S("A0")
gV = S("gV")
g4F = S("g4F")
g_psi4 = S("g_psi4")
gJJ = S("gJJ")
alpha_s, beta_s = S("alpha_s", "beta_s")
i1, i2, i3, i4 = S("i1", "i2", "i3", "i4")
idx_i, idx_j, idx_k = S("i", "j", "k")

lam4 = S("lam4")
lam6 = S("lam6")
g_sym = S("g")
gD = S("gD")
gD2 = S("gD2")
gijk = S("gijk")
g1 = S("g1")
g2 = S("g2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(got, expected, label):
    """Assert two expressions are equal after expand + canonicalize, then print PASS."""
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


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

    if compact_override is not None:
        compact = compact_override
        result_label = "Compact override"
    else:
        compact = simplify_deltas(V, species_map=species_map)
        result_label = "Vertex"

    print("=" * 80)
    print(f"  {title}")
    print(f"  alphas = {alphas}")
    print(f"  betas  = {betas}")
    print(f"  ps     = {ps}")
    print()
    print(f"  {result_label}:")
    print(f"  {compact}")
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
# Interaction definitions
# ---------------------------------------------------------------------------

# 1) phi^4
L_phi4 = dict(
    coupling=lam4,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# 2) phi^6
L_phi6 = dict(
    coupling=lam6,
    alphas=[phi0] * 6,
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# 3) phi^2 chi^2
L_phi2chi2 = dict(
    coupling=g_sym,
    alphas=[phi0, phi0, chi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
)

# 4) gD * (d_mu phi)(d_nu phi) phi phi
deriv_indices, deriv_targets = infer_derivative_targets([(0, [mu]), (1, [nu])])
L_deriv = dict(
    coupling=gD,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices,
    derivative_targets=deriv_targets,
)

# 5) gD2 * (d_mu phi)(d_mu phi) phi phi
deriv_indices2, deriv_targets2 = infer_derivative_targets([(0, [mu]), (1, [mu])])
L_deriv2 = dict(
    coupling=gD2,
    alphas=[phi0, phi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2,
    derivative_targets=deriv_targets2,
)

# 6) gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2
L_multi = dict(
    coupling=gijk(idx_i, idx_j, idx_k),
    alphas=[idx_i, idx_i, idx_j, idx_j, idx_k, idx_k],
    betas=[b1, b2, b3, b4, b5, b6],
    ps=[p1, p2, p3, p4, p5, p6],
)

# 7) Yukawa: yF * psibar * psi * phi
L_yukawa = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None],
    leg_spins=[s1, s2, s3],
)

# 8) Vector current: gV * psibar gamma^mu psi A_mu
L_vec_current = dict(
    coupling=gV * gamma_matrix(i_psi_bar, i_psi, mu),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

L_axial_current = dict(
    coupling=gV * gamma_matrix(i_psi_bar, alpha_s, mu) * gamma5_matrix(alpha_s, i_psi),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

# 9) Underspecified multi-fermion product kept as a validation diagnostic
L_4fermion = dict(
    coupling=g4F,
    alphas=[psi0, psibar0, psi0, psibar0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psi", "psibar", "psi", "psibar"],
    leg_roles=["psi", "psibar", "psi", "psibar"],
)

# 10) -(g/2)(psibar psi)^2  (amputated / matrix-element / explicit-spinor modes)
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

L_psibar_psi_sq_spinor = dict(
    **L_psibar_psi_sq,
    leg_spinor_indices=[i1, i2, i3, i4],
)

# 10b) Current-current four-fermion operator:
#     gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)
L_current_current = dict(
    coupling=gJJ * gamma_matrix(alpha_s, alpha_s, mu) * gamma_lowered_matrix(beta_s, beta_s, mu),
    alphas=[psibar0, psi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "psibar", "psi"],
    leg_roles=["psibar", "psi", "psibar", "psi"],
    field_spinor_indices=[alpha_s, alpha_s, beta_s, beta_s],
    leg_spins=[s1, s2, s3, s4],
)

# 11-14) Mixed derivative fermion+scalar interactions
_MIX_BASE = dict(
    alphas=[psibar0, psi0, phi0, chi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None, None],
    leg_spins=[s1, s2, s3, s4],
)

L_mix_dpsibar = dict(**_MIX_BASE, coupling=yF, derivative_indices=[mu], derivative_targets=[0])
L_mix_dpsi = dict(**_MIX_BASE, coupling=yF, derivative_indices=[nu], derivative_targets=[1])
L_mix_dphi_dchi = dict(**_MIX_BASE, coupling=yF, derivative_indices=[mu, nu], derivative_targets=[2, 3])
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
    field_spinor_indices=[alpha_s, alpha_s, None, None, None],
    leg_spins=[s1, s2, s3, s4, s1],
)

# 15-16) Double-derivative scalars with fermion bilinear
# L1 = g1 * psibar * psi * (d^2 phi) * chi   [d^2 modeled as d_mu d_mu on the phi slot]
deriv_indices_l1, deriv_targets_l1 = infer_derivative_targets([(2, [mu, mu])])
L_double_deriv_phi_chi = dict(
    **_MIX_BASE,
    coupling=g1,
    derivative_indices=deriv_indices_l1,
    derivative_targets=deriv_targets_l1,
)

# L2 = g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)
deriv_indices_l2, deriv_targets_l2 = infer_derivative_targets([(2, [mu, nu]), (3, [mu, nu])])
L_double_deriv_phi_phi = dict(
    coupling=g2,
    alphas=[psibar0, psi0, phi0, phi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices_l2,
    derivative_targets=deriv_targets_l2,
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar", "scalar"],
    leg_roles=["psibar", "psi", "scalar", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None, None],
    leg_spins=[s1, s2, s3, s4],
)

# Pre-built compact forms for derivative tests
COMPACT_DERIV = compact_vertex_sum_form(
    coupling=gD, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices, derivative_targets=deriv_targets, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
)
COMPACT_DERIV2 = compact_vertex_sum_form(
    coupling=gD2, ps=[p1, p2, p3, p4],
    derivative_indices=deriv_indices2, derivative_targets=deriv_targets2, d=d,
    field_species=[phi0] * 4, leg_species=[phi0] * 4,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_scalar_tests():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)

    V = simplify_deltas(vertex_factor(**L_phi4, x=x, d=d), species_map=sm_phi)
    _check(V, 24 * I * lam4 * D4, "phi^4")

    V = simplify_deltas(vertex_factor(**L_phi2chi2, x=x, d=d),
                        species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0})
    _check(V, 4 * I * g_sym * D4, "phi^2 chi^2")

    V = simplify_deltas(vertex_factor(**L_deriv, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV, "Derivative (mu,nu)")

    V = simplify_deltas(vertex_factor(**L_deriv2, x=x, d=d), species_map=sm_phi)
    _check(V, COMPACT_DERIV2, "Derivative (mu,mu)")

    # Multi-species: gijk * phi_i^2 phi_j^2 phi_k^2
    D6 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5 + p6)
    V_multi = vertex_factor(**L_multi, x=x, d=d)
    expected_multi = 8 * I * gijk(idx_i, idx_j, idx_k) * D6

    sm_base = {b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k}
    _check(simplify_deltas(V_multi, species_map=sm_base), expected_multi, "Multi-species (base)")

    sm_perm = {b1: idx_j, b2: idx_k, b3: idx_i, b4: idx_j, b5: idx_k, b6: idx_i}
    _check(simplify_deltas(V_multi, species_map=sm_perm), expected_multi, "Multi-species (perm)")

    sm_bad = {b1: idx_i, b2: idx_i, b3: idx_i, b4: idx_j, b5: idx_j, b6: idx_k}
    _check(simplify_deltas(V_multi, species_map=sm_bad), Expression.num(0), "Multi-species (bad mult -> 0)")

    print("\n  Scalar+derivative tests passed.\n")


def _run_fermion_tests():
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    D3 = (2 * pi) ** d * Delta(p1 + p2 + p3)
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    G12 = bis.g(i1, i2).to_expression()

    V = simplify_deltas(vertex_factor(**L_yukawa, x=x, d=d), species_map=sm3)
    _check(V, I * yF * G12 * D3, "Yukawa (amputated)")

    V_full = simplify_deltas(
        vertex_factor(**L_yukawa, x=x, d=d, strip_externals=False), species_map=sm3
    )
    s = V_full.to_canonical_string()
    assert "UbarF" in s and "UF" in s, f"Yukawa unstripped missing UF/UbarF: {V_full}"
    print("  Yukawa (unstripped, has UF/UbarF): PASS")

    V = simplify_deltas(vertex_factor(**L_vec_current, x=x, d=d),
                        species_map={b1: psibar0, b2: psi0, b3: A0})
    _check(V, I * gV * gamma_matrix(i_psi_bar, i_psi, mu) * D3, "Vector current")

    V = simplify_deltas(vertex_factor(**L_axial_current, x=x, d=d),
                        species_map={b1: psibar0, b2: psi0, b3: A0})
    _check(
        V,
        I * gV * gamma_matrix(i_psi_bar, alpha_s, mu) * gamma5_matrix(alpha_s, i_psi) * D3,
        "Axial current",
    )

    try:
        vertex_factor(**L_4fermion, x=x, d=d)
    except ValueError as exc:
        assert "Multi-fermion operators require explicit spinor-contraction data" in str(exc)
        print("  underspecified multi-fermion operator rejected: PASS")
    else:
        raise AssertionError("Bare psi*psibar*psi*psibar should be rejected as underspecified")

    # (psibar psi)^2 amputated -> open spinor structure
    V = simplify_deltas(vertex_factor(**L_psibar_psi_sq, x=x, d=d), species_map=sm4)
    expected_sp = (
        -I * g_psi4 * D4
        * (bis.g(i1, i2).to_expression() * bis.g(i3, i4).to_expression()
           - bis.g(i1, i4).to_expression() * bis.g(i3, i2).to_expression())
    )
    _check(V, expected_sp, "(psibar psi)^2 amputated")

    # (psibar psi)^2 unstripped -> non-zero with UF/UbarF
    V_full = simplify_deltas(
        vertex_factor(**L_psibar_psi_sq, x=x, d=d, strip_externals=False), species_map=sm4
    )
    s = V_full.to_canonical_string()
    assert s != Expression.num(0).to_canonical_string(), "(psibar psi)^2 unstripped should be non-zero"
    assert "UbarF" in s and "UF" in s, f"Missing UF/UbarF: {V_full}"
    print("  (psibar psi)^2 matrix element (non-zero direct/exchange): PASS")

    # Spinor-delta form: -ig * D4 * [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]
    V_sp = simplify_deltas(vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d), species_map=sm4)
    _check(V_sp, expected_sp, "(psibar psi)^2 spinor deltas")

    # Current-current stripped -> 0 for the present stripped setup
    V = simplify_deltas(vertex_factor(**L_current_current, x=x, d=d), species_map=sm4)
    _check(V, Expression.num(0), "Current-current stripped -> 0")

    # Current-current unstripped -> non-zero; contraction topologies are still distinguishable
    V_full = simplify_deltas(
        vertex_factor(**L_current_current, x=x, d=d, strip_externals=False), species_map=sm4
    )
    s = V_full.to_canonical_string()
    assert s != Expression.num(0).to_canonical_string(), "Current-current unstripped should be non-zero"
    assert "UbarF" in s and "UF" in s, f"Current-current unstripped missing UF/UbarF: {V_full}"
    print("  Current-current unstripped (non-zero): PASS")

    invalid_spinor_legs = dict(L_psibar_psi_sq_spinor)
    invalid_spinor_legs["leg_spinor_indices"] = [i1, None, i3, i4]
    try:
        vertex_factor(
            **invalid_spinor_legs,
            x=x,
            d=d,
        )
        raise AssertionError("Expected ValueError for missing fermion leg spinor index")
    except ValueError as exc:
        assert "Fermion legs must carry a spinor index" in str(exc), exc
    print("  Missing fermion leg spinor index -> ValueError: PASS")

    print("\n  Fermion tests passed.\n")


def _run_fermion_derivative_mixed_tests():
    sm4 = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}
    D4 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4)
    D5 = (2 * pi) ** d * Delta(p1 + p2 + p3 + p4 + p5)
    G12 = bis.g(i1, i2).to_expression()

    V = simplify_deltas(vertex_factor(**L_mix_dpsibar, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p1, mu) * G12 * D4, "d_mu psibar")

    V = simplify_deltas(vertex_factor(**L_mix_dpsi, x=x, d=d), species_map=sm4)
    _check(V, yF * pcomp(p2, nu) * G12 * D4, "d_nu psi")

    V = simplify_deltas(vertex_factor(**L_mix_dphi_dchi, x=x, d=d), species_map=sm4)
    _check(V, -I * yF * pcomp(p3, mu) * pcomp(p4, nu) * G12 * D4, "(d_mu phi)(d_nu chi)")

    sm5 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0}
    V = simplify_deltas(vertex_factor(**L_mix_5pt, x=x, d=d), species_map=sm5)
    _check(V, -2 * I * g_sym * pcomp(p1, mu) * pcomp(p2, nu) * G12 * D5, "5pt mixed")

    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_chi, x=x, d=d), species_map=sm4)
    _check(V, -I * g1 * G12 * D4 * pcomp(p3, mu) * pcomp(p3, mu), "g1 * psibar psi (d^2 phi) chi")

    sm_phi2 = {b1: psibar0, b2: psi0, b3: phi0, b4: phi0}
    V = simplify_deltas(vertex_factor(**L_double_deriv_phi_phi, x=x, d=d), species_map=sm_phi2)
    _check(
        V,
        2
        * I
        * g2
        * G12
        * D4
        * pcomp(p3, mu)
        * pcomp(p3, nu)
        * pcomp(p4, mu)
        * pcomp(p4, nu),
        "g2 * psibar psi (d_mu d_nu phi)^2",
    )

    print("\n  Mixed fermion+scalar derivative tests passed.\n")


def _relation_to_original(candidate, original):
    cand = candidate.expand().to_canonical_string()
    if cand == original.expand().to_canonical_string():
        return "== V_orig"
    if cand == (-original).expand().to_canonical_string():
        return "== -V_orig"
    return "different (not +/-V_orig)"


def _run_swap_diagnostics():
    """Diagnostic: operator-order sensitivity for (psibar psi)^2."""
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    V_orig = simplify_deltas(vertex_factor(**L_psibar_psi_sq_spinor, x=x, d=d), species_map=sm4)

    L_swap_both = dict(L_psibar_psi_sq_spinor)
    L_swap_both.update(
        alphas=[psi0, psibar0, psi0, psibar0],
        field_roles=["psi", "psibar", "psi", "psibar"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_both = simplify_deltas(vertex_factor(**L_swap_both, x=x, d=d), species_map=sm4)
    print("  swap-both bilinears: ", _relation_to_original(V_swap_both, V_orig))

    L_swap_one = dict(L_psibar_psi_sq_spinor)
    L_swap_one.update(
        alphas=[psi0, psibar0, psibar0, psi0],
        field_roles=["psi", "psibar", "psibar", "psi"],
        leg_roles=["psibar", "psi", "psibar", "psi"],
    )
    V_swap_one = simplify_deltas(vertex_factor(**L_swap_one, x=x, d=d), species_map=sm4)
    print("  swap-one bilinear:  ", _relation_to_original(V_swap_one, V_orig))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run_suite_tests(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_tests()
    if suite in ("fermion", "all"):
        _run_fermion_tests()
        _run_fermion_derivative_mixed_tests()
    print("All selected tests passed.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_scalar_demo():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}

    print("\n=== scalar: phi^4 ===")
    show_vertex("lam4 * phi^4", **L_phi4, species_map=sm_phi)

    print("\n=== scalar: phi^6 ===")
    show_vertex("lam6 * phi^6", **L_phi6,
                species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0, b5: phi0, b6: phi0})

    print("\n=== scalar: phi^2 chi^2 ===")
    show_vertex("g * phi^2 * chi^2", **L_phi2chi2,
                species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0})

    print("\n=== scalar: derivative (mu,nu) * phi^4 ===")
    show_vertex("gD * (d_mu phi)(d_nu phi) phi phi", **L_deriv,
                species_map=sm_phi, compact_override=COMPACT_DERIV, show_sum_notation=True)

    print("\n=== scalar: derivative (mu,mu) * phi^4 ===")
    show_vertex("gD2 * (d_mu phi)(d_mu phi) phi phi", **L_deriv2,
                species_map=sm_phi, compact_override=COMPACT_DERIV2, show_sum_notation=True)

    print("\n=== scalar: multi-species phi_i^2 phi_j^2 phi_k^2 ===")
    show_vertex("gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2", **L_multi,
                species_map={b1: idx_i, b2: idx_i, b3: idx_j, b4: idx_j, b5: idx_k, b6: idx_k})


def _run_fermion_demo():
    sm3 = {b1: psibar0, b2: psi0, b3: phi0}
    sm4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}

    print("\n=== fermion: Yukawa [amputated] ===")
    show_vertex("yF * psibar * psi * phi", **L_yukawa, species_map=sm3)

    print("\n=== fermion: Yukawa [matrix element] ===")
    show_vertex("yF * psibar * psi * phi  [matrix element]", **L_yukawa,
                strip_externals=False, species_map=sm3)

    print("\n=== fermion: vector current ===")
    show_vertex("gV * psibar gamma^mu psi A_mu", **L_vec_current,
                species_map={b1: psibar0, b2: psi0, b3: A0})

    print("\n=== fermion: axial current ===")
    show_vertex("gV * psibar gamma^mu gamma5 psi A_mu", **L_axial_current,
                species_map={b1: psibar0, b2: psi0, b3: A0})

    print("\n=== fermion: underspecified product diagnostic ===")
    print("=" * 80)
    print("  g4F * psi * psibar * psi * psibar  [no spinor contractions]")
    print("  rejected: multi-fermion operators need explicit spinor contractions")

    print("\n=== fermion: -(g/2)(psibar psi)^2  [amputated] ===")
    show_vertex("-(g/2)(psibar psi)^2  [amputated]", **L_psibar_psi_sq, species_map=sm4)

    print("\n=== fermion: -(g/2)(psibar psi)^2  [matrix element] ===")
    show_vertex("-(g/2)(psibar psi)^2  [matrix element]", **L_psibar_psi_sq,
                strip_externals=False, species_map=sm4)

    print("\n=== fermion: -(g/2)(psibar psi)^2  [explicit spinor labels] ===")
    show_vertex("-(g/2)(psibar psi)^2  [spinor delta]", **L_psibar_psi_sq_spinor, species_map=sm4)
    print("=" * 80)
    print(
        "  -(g/2)(psibar psi)(psibar psi)  →  "
        "V = (-ig)[g(i1,i2)*g(i3,i4) - g(i1,i4)*g(i3,i2)] * (2*pi)^d * Delta(Σp)"
    )
    print()

    print("\n=== fermion: current-current operator ===")
    show_vertex("gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]", **L_current_current, species_map=sm4)
    show_vertex(
        "gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)  [unstripped]",
        **L_current_current,
        strip_externals=False,
        species_map=sm4,
    )
    print("  Interpretation: stripped mode cancels contraction topologies; unstripped mode keeps them visible.")
    print()

    print("\n=== fermion: operator-order diagnostics (psibar psi)^2 ===")
    _run_swap_diagnostics()
    print()

    sm_mix = {b1: psibar0, b2: psi0, b3: phi0, b4: chi0}
    print("\n=== fermion+scalar: mixed derivatives ===")
    show_vertex("yF * (d_mu psibar) * psi * phi * chi", **L_mix_dpsibar, species_map=sm_mix)
    show_vertex("yF * psibar * (d_nu psi) * phi * chi", **L_mix_dpsi, species_map=sm_mix)
    show_vertex("yF * psibar * psi * (d_mu phi) * (d_nu chi)", **L_mix_dphi_dchi, species_map=sm_mix)
    show_vertex("g * (d_mu psibar)(d_nu psi) phi phi chi", **L_mix_5pt,
                species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0, b5: chi0})
    show_vertex("g1 * psibar * psi * (d^2 phi) * chi", **L_double_deriv_phi_chi, species_map=sm_mix)
    show_vertex(
        "g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)",
        **L_double_deriv_phi_phi,
        species_map={b1: psibar0, b2: psi0, b3: phi0, b4: phi0},
    )


def _run_suite_demo(suite: str):
    if suite in ("scalar", "all"):
        _run_scalar_demo()
    if suite in ("fermion", "all"):
        _run_fermion_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Symbolica vertex examples.")
    parser.add_argument("--suite", choices=("scalar", "fermion", "all"), default="all")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  Demo: {args.suite}")
    print(f"{'='*80}\n")
    _run_suite_demo(args.suite)

    if not args.skip_tests:
        print(f"\n{'='*80}")
        print(f"  Tests: {args.suite}")
        print(f"{'='*80}\n")
        _run_suite_tests(args.suite)
