"""Focused regressions extracted from the example scripts."""

import pytest

from symbolica import S

from model import (
    Field,
    Lagrangian,
    Model,
    ROLE_PSI,
    ROLE_PSIBAR,
    ROLE_SCALAR,
    ROLE_SCALAR_DAG,
    ROLE_VECTOR,
)
from symbolic.vertex_engine import Delta, I, pi, vertex_factor


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_direct_vertex_factor_kwargs_are_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'coupling'"):
        vertex_factor(
            coupling=S("y"),
            alphas=[S("psibar0"), S("psi0"), S("phi0")],
            betas=[S("b1"), S("b2"), S("b3")],
            ps=[S("p1"), S("p2"), S("p3")],
            x=S("x"),
        )


def test_lagrangian_source_declarations_are_rejected():
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phiC0"),
        conjugate_symbol=S("phiCdag0"),
    )

    with pytest.raises(TypeError, match="Use Model"):
        Lagrangian(S("lamC") * phi.bar * phi)


def test_model_tuple_field_syntax_matches_complex_scalar_bilinear():
    phi = Field(
        "PhiC",
        spin=0,
        self_conjugate=False,
        symbol=S("phiC0"),
        conjugate_symbol=S("phiCdag0"),
    )
    lamC = S("lamC")
    expected = I * lamC * (2 * pi) ** S("d") * Delta(S("q1") + S("q2"))
    model = Model(lamC * phi.bar * phi)

    assert _canon(model.feynman_rule(phi.bar, phi, include_delta=True)) == _canon(expected)
    assert _canon(model.feynman_rule(phi, phi.bar, include_delta=True)) == _canon(expected)
    assert _canon(model.feynman_rule((phi, True), phi, include_delta=True)) == _canon(expected)


def test_model_same_symbol_distinct_fields_do_not_match():
    shared_symbol = S("Y_shared")
    phi_alias = Field("PhiAlias", spin=0, self_conjugate=True, symbol=shared_symbol)
    chi_alias = Field("ChiAlias", spin=0, self_conjugate=True, symbol=shared_symbol)
    model = Model(S("lamC") * phi_alias)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        model.feynman_rule(chi_alias)


def test_model_same_symbol_distinct_roles_do_not_match():
    shared_symbol = S("X_shared")
    scalar = Field("ScalarShared", spin=0, self_conjugate=True, symbol=shared_symbol)
    vector = Field("VectorShared", spin=1, self_conjugate=True, symbol=shared_symbol)
    model = Model(S("lamC") * scalar)

    with pytest.raises(ValueError, match="No matching interaction terms"):
        model.feynman_rule(vector)


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
