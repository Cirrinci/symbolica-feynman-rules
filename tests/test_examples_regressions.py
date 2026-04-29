"""Focused regressions extracted from the example scripts."""

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from symbolica import Expression  # noqa: E402

from examples.examples import (  # noqa: E402
    A0,
    L_4fermion,
    L_yukawa,
    SPINOR_KIND,
    b1,
    b2,
    b3,
    d,
    lamC,
    p1,
    p2,
    phiC0,
    phiCdag0,
    phi0,
    psi0,
    psibar0,
    x,
)
from model import (  # noqa: E402
    Field,
    Lagrangian,
    ROLE_PSI,
    ROLE_PSIBAR,
    ROLE_SCALAR,
    ROLE_SCALAR_DAG,
    ROLE_VECTOR,
)
from symbolic.vertex_engine import Delta, I, S, pi, simplify_deltas, vertex_factor  # noqa: E402


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_direct_api_yukawa_unstripped_keeps_external_spinors():
    expr = simplify_deltas(
        vertex_factor(**L_yukawa, x=x, d=d, strip_externals=False),
        species_map={b1: psibar0, b2: psi0, b3: phi0},
    )
    text = expr.to_canonical_string()
    assert "UbarF" in text
    assert "UF" in text


def test_direct_api_rejects_underspecified_multi_fermion_operator():
    with pytest.raises(ValueError, match="Multi-fermion"):
        vertex_factor(**L_4fermion, x=x, d=d)


def test_direct_api_rejects_partial_fermion_leg_spinor_labels():
    with pytest.raises(ValueError, match="all fermion external legs"):
        vertex_factor(
            **L_yukawa,
            x=x,
            d=d,
            leg_index_labels=[
                {SPINOR_KIND: L_yukawa["field_spinor_indices"][0]},
                {},
                {},
            ],
        )


def test_role_based_complex_scalar_filtering_regression():
    expected = I * lamC * (2 * pi) ** d * Delta(p1 + p2)

    expr = vertex_factor(
        coupling=lamC,
        alphas=[phiCdag0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        x=x,
        d=d,
    )
    raw = expr.expand().to_canonical_string()
    assert "delta" in raw

    simplified_no_map = simplify_deltas(expr)
    simplified_with_map = simplify_deltas(expr, species_map={b1: phiCdag0, b2: phiC0})

    assert _canon(simplified_with_map) == _canon(expected)
    assert _canon(simplified_no_map) != _canon(Expression.num(0))


def test_role_based_reversed_leg_query_still_works():
    expected = I * lamC * (2 * pi) ** d * Delta(p1 + p2)

    expr = vertex_factor(
        coupling=lamC,
        alphas=[phiCdag0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_SCALAR_DAG, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR, ROLE_SCALAR_DAG],
        x=x,
        d=d,
    )
    simplified = simplify_deltas(expr, species_map={b1: phiC0, b2: phiCdag0})
    assert _canon(simplified) == _canon(expected)


def test_role_based_vector_scalar_nonmatching_is_filtered():
    expected = I * lamC * (2 * pi) ** d * Delta(p1 + p2)

    expr = vertex_factor(
        coupling=lamC,
        alphas=[A0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_VECTOR, ROLE_SCALAR],
        leg_roles=[ROLE_SCALAR, ROLE_VECTOR],
        x=x,
        d=d,
    )
    simplified = simplify_deltas(expr, species_map={b1: phiC0, b2: A0})
    assert _canon(simplified) == _canon(expected)

    no_match = vertex_factor(
        coupling=lamC,
        alphas=[A0, phiC0],
        betas=[b1, b2],
        ps=[p1, p2],
        field_roles=[ROLE_VECTOR, ROLE_SCALAR],
        leg_roles=[ROLE_VECTOR, ROLE_VECTOR],
        x=x,
        d=d,
    )
    assert _canon(simplify_deltas(no_match, species_map={b1: A0, b2: A0})) == _canon(
        Expression.num(0)
    )


def test_field_role_object_semantics():
    assert ROLE_PSI.is_fermion
    assert ROLE_PSIBAR.is_fermion
    assert not ROLE_SCALAR.is_fermion
    assert not ROLE_VECTOR.is_fermion
    assert ROLE_PSI.compatible_with(ROLE_PSI)
    assert not ROLE_PSI.compatible_with(ROLE_PSIBAR)
    assert not ROLE_SCALAR.compatible_with(ROLE_SCALAR_DAG)
    assert not ROLE_SCALAR.compatible_with(ROLE_VECTOR)
    assert ROLE_PSI.compatible_with("psi")
    assert not ROLE_PSI.compatible_with("psibar")


def test_lagrangian_tuple_field_syntax_matches_complex_scalar_bilinear():
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=phiC0,
        conjugate_symbol=phiCdag0,
    )
    expected = I * lamC * (2 * pi) ** d * Delta(S("q1") + S("q2"))
    lagrangian = Lagrangian(lamC * phi.bar * phi)

    assert _canon(lagrangian.feynman_rule(phi.bar, phi)) == _canon(expected)
    assert _canon(lagrangian.feynman_rule(phi, phi.bar)) == _canon(expected)
    assert _canon(lagrangian.feynman_rule((phi, True), phi)) == _canon(expected)


def test_lagrangian_same_symbol_distinct_fields_do_not_match():
    shared_symbol = S("Y_shared")
    phi_alias = Field("PhiAlias", spin=0, self_conjugate=True, symbol=shared_symbol)
    chi_alias = Field("ChiAlias", spin=0, self_conjugate=True, symbol=shared_symbol)
    lagrangian = Lagrangian(lamC * phi_alias)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(chi_alias)


def test_lagrangian_same_symbol_distinct_roles_do_not_match():
    shared_symbol = S("X_shared")
    scalar = Field("ScalarShared", spin=0, self_conjugate=True, symbol=shared_symbol)
    vector = Field("VectorShared", spin=1, self_conjugate=True, symbol=shared_symbol)
    lagrangian = Lagrangian(lamC * scalar)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(vector)
