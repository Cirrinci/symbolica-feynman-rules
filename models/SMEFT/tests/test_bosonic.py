"""Tests for the bosonic operators of the SMEFT Green basis (Table 1)."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core
from models.SMEFT.registry import get_operator, operators_in
from feynpy.declared import CovariantDerivativeFactor, DifferentiatedOperatorFactor


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_all_bosonic_compile(core):
    ops = operators_in(sector="bosonic")
    assert ops
    for op in ops:
        if op.status == "blocked":
            continue
        lag = op.lagrangian(core)
        assert lag.terms, op.name


def test_bosonic_canonical_dimension(core):
    for op in operators_in(sector="bosonic"):
        if op.status == "blocked":
            continue
        assert op.canonical_dimensions(core) == {6}, op.name


@pytest.mark.parametrize(
    "name, fields",
    [
        ("O3G", ("G", "G", "G")),
        ("O3W", ("W", "W", "W")),
        ("OHG", ("H.bar", "H", "G", "G")),
        ("OHB", ("H.bar", "H", "B", "B")),
        ("OHWB", ("H.bar", "H", "W", "B")),
        ("OH", ("H.bar", "H", "H.bar", "H", "H.bar", "H")),
    ],
)
def test_bosonic_expected_vertices(core, name, fields):
    lag = get_operator(name).lagrangian(core)
    names = {sig.names for sig in lag.vertex_signatures()}
    assert tuple(fields) in {tuple(sorted(n)) for n in names} or any(
        set(fields) <= set(n) for n in names
    ), (name, names)


@pytest.mark.parametrize("name", ["O3Gtilde", "O3Wtilde", "OHGtilde", "OHWtilde",
                                   "OHBtilde", "OHWBtilde"])
def test_dual_operators_carry_levi_civita(core, name):
    op = get_operator(name)
    lag = op.lagrangian(core)
    # any vertex of the dual operator must contain the Levi-Civita tensor.
    sig = lag.vertex_signatures()[0]
    rule = lag.feynman_rule(*sig.fields)
    text = rule.expand().to_canonical_string().lower()
    assert "levi" in text or "\u03f5" in text or "eps" in text


def test_non_dual_has_no_levi_civita(core):
    op = get_operator("OHG")
    lag = op.lagrangian(core)
    rule = lag.feynman_rule(core.fields.H.bar, core.fields.H, core.fields.G, core.fields.G)
    assert "levi" not in rule.expand().to_canonical_string().lower()


def test_rpphd_outer_derivative_hits_covariant_derivative(core):
    op = get_operator("RppHD")
    terms = op.structure(core).source_terms
    assert len(terms) == 4
    assert any(
        any(
            isinstance(factor, DifferentiatedOperatorFactor)
            and isinstance(factor.operand, CovariantDerivativeFactor)
            for factor in term.factors
        )
        for term in terms
    )
