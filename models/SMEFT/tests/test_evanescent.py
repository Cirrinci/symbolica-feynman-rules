"""Tests for the evanescent operators of the SMEFT Green basis (Tables 4-9)."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core
from models.SMEFT.registry import get_operator, operators_in


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_two_fermion_evanescent_compile(core):
    ops = operators_in(sector="two_fermion", otype="evanescent")
    # 8 psi2XH (dual dipoles) + 30 psi2XD + 3 psi2HD2
    assert len(ops) == 41
    for op in ops:
        lag = op.lagrangian(core)
        assert lag.terms, op.name
        assert op.canonical_dimensions(core) == {6}, op.name


def test_four_fermion_evanescent_compile(core):
    ops = operators_in(sector="four_fermion", otype="evanescent", status="implemented")
    assert len(ops) == 62
    for op in ops:
        lag = op.lagrangian(core)
        assert lag.terms, op.name
        assert op.canonical_dimensions(core) == {6}, op.name


def test_ordered_triple_gamma_chain_preserved(core):
    """The whole point of the evanescent basis: ordered gamma^{mu nu rho} chains
    must survive to the Feynman rule (no 4D reduction)."""
    op = get_operator("E3ll")
    l = core.fields.l
    text = op.feynman_rule(core, l.bar, l, l.bar, l).expand().to_canonical_string()
    # a genuine triple-gamma chain leaves many ordered gamma factors intact
    assert text.count("gamma(") >= 12


def test_evanescent_dipole_uses_dual_field_strength(core):
    op = get_operator("EuB")
    f = core.fields
    text = op.feynman_rule(core, f.q.bar, f.u, f.H.bar, f.B) \
        .expand().to_canonical_string()
    assert "levi" in text or "epsilon" in text.lower()


def test_Eprime_has_sigma_and_derivative(core):
    # E'_{Bq} = i qbar (sigma^{mu nu} /D - <-/D sigma^{mu nu}) q B_{mu nu}
    op = get_operator("EpBq")
    lag = op.lagrangian(core)
    sigs = {tuple(s.names) for s in lag.vertex_signatures()}
    assert ("q.bar", "q", "B") in sigs


def test_charge_conjugation_operators_declare_but_are_blocked(core):
    cc = operators_in(status="blocked")
    assert len(cc) == 8
    for op in cc:
        # declared structure is inspectable
        assert op.structure(core).source_terms
        # but compilation hits the documented fermion-flow limitation
        with pytest.raises(ValueError):
            op.lagrangian(core)


def test_non_hermitian_evanescent_include_hc(core):
    f = core.fields
    assert get_operator("EuG").feynman_rule(core, f.u.bar, f.q, f.H, f.G)
    assert get_operator("EuH").feynman_rule(core, f.u.bar, f.q, f.H)
