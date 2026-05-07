"""Regression tests for the typed invariant tensors and the idenso pipeline.

These tests verify that:

- the project-defined invariant tensors created via Spenso ``TensorName``
  are recognised by ``canonize_spenso_tensors`` with the right symmetry,
- ``canonize_full`` does not regress the existing structure-constant
  antisymmetry behaviour,
- the typed ``weak_eps2`` survives end-to-end through the Lagrangian-API
  pipeline.
"""

from __future__ import annotations

from fractions import Fraction

from symbolica import Expression, S

from model import Field, Model
from symbolic.spenso_structures import (
    color_levi_civita,
    color_symmetric_constant,
    dirac_charge_conjugation,
    lorentz_levi_civita,
    simplify_invariants,
    weak_eps2,
)
from symbolic.tensor_canonicalization import (
    canonize_full,
    canonize_spenso_tensors,
)


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_weak_eps2_antisymmetry_is_recognised():
    i, j = S("i"), S("j")
    expr = weak_eps2(i, j) + weak_eps2(j, i)
    canon, _, _ = canonize_spenso_tensors(expr, weak_fund_indices=(i, j))
    assert _canon(canon) == _canon(Expression.num(0))


def test_lorentz_levi_civita_antisymmetry():
    mu, nu, rho, sig = S("mu"), S("nu"), S("rho"), S("sigma")
    expr = (
        lorentz_levi_civita(mu, nu, rho, sig)
        + lorentz_levi_civita(nu, mu, rho, sig)
    )
    canon, _, _ = canonize_spenso_tensors(expr, lorentz_indices=(mu, nu, rho, sig))
    assert _canon(canon) == _canon(Expression.num(0))


def test_color_eps3_antisymmetry():
    i, j, k = S("i"), S("j"), S("k")
    expr = color_levi_civita(i, j, k) + color_levi_civita(j, i, k)
    canon, _, _ = canonize_spenso_tensors(expr, color_fund_indices=(i, j, k))
    assert _canon(canon) == _canon(Expression.num(0))


def test_color_symmetric_constant_d_abc_symmetry():
    a, b, c = S("a"), S("b"), S("c")
    expr = color_symmetric_constant(a, b, c) - color_symmetric_constant(b, a, c)
    canon, _, _ = canonize_spenso_tensors(expr, adjoint_indices=(a, b, c))
    assert _canon(canon) == _canon(Expression.num(0))


def test_dirac_charge_conjugation_antisymmetry():
    i, j = S("i"), S("j")
    expr = dirac_charge_conjugation(i, j) + dirac_charge_conjugation(j, i)
    canon, _, _ = canonize_spenso_tensors(expr, spinor_indices=(i, j))
    assert _canon(canon) == _canon(Expression.num(0))


def test_simplify_invariants_is_no_op_on_simple_eps2():
    """simplify_invariants must not change a single eps_{ij} factor."""
    i, j = S("i"), S("j")
    expr = weak_eps2(i, j)
    assert _canon(simplify_invariants(expr)) == _canon(expr)


def test_typed_eps2_in_yukawa_model_runs_and_carries_eps2_head():
    """End-to-end: typed weak_eps2 in a Lagrangian Model produces a vertex
    that still contains the typed eps2 head (no silent collapse) and that
    runs cleanly through the simplifier."""
    from model import (
        SPINOR_INDEX,
        COLOR_FUND_INDEX,
        WEAK_FUND_INDEX,
    )

    qL = Field(
        "qL",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("qL0"),
        conjugate_symbol=S("qLbar0"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX, WEAK_FUND_INDEX),
    )
    uR = Field(
        "uR",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("uR0"),
        conjugate_symbol=S("uRbar0"),
        indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
    )
    H = Field(
        "H",
        spin=0,
        self_conjugate=False,
        symbol=S("H0"),
        conjugate_symbol=S("Hbar0"),
        indices=(WEAK_FUND_INDEX,),
    )

    yu = S("yu")
    i, j = S("i_qu"), S("j_qu")

    typed_model = Model(
        fields=(qL, uR, H),
        lagrangian_decl=yu
        * weak_eps2(i, j)
        * qL.bar(index_labels={WEAK_FUND_INDEX.kind: i})
        * H.bar(j)
        * uR,
    )
    typed_vertex = typed_model.lagrangian().feynman_rule(
        qL.bar, H.bar, uR, simplify=True, include_delta=False,
    )

    text = typed_vertex.to_canonical_string()
    assert "weak_eps2" in text


def test_canonize_full_simplifies_eps2_pair_in_yukawa_form():
    """eps2(i,j) - eps2(j,i) must collapse to 2*eps2(i,j) under canonize_full."""
    i, j = S("i"), S("j")
    expr = weak_eps2(i, j) - weak_eps2(j, i)

    canon = canonize_full(expr, weak_fund_indices=(i, j))
    expected = 2 * weak_eps2(i, j)
    assert _canon(canon) == _canon(expected)
