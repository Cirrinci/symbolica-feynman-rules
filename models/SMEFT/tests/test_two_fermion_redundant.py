"""Tests for the redundant two-fermion operators of the Green basis (Table 2)."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core
from models.SMEFT.registry import get_operator, operators_in


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_all_two_fermion_redundant_compile(core):
    ops = operators_in(sector="two_fermion", otype="redundant")
    # psi2D3 (5) + psi2XD (30) + psi2HD2 (12) + psi2DH2-redundant (14)
    assert len(ops) == 61
    for op in ops:
        lag = op.lagrangian(core)
        assert lag.terms, op.name
        assert op.canonical_dimensions(core) == {6}, op.name


def test_psi2D3_is_three_derivatives(core):
    # i/2 qbar {D^2, /D} q : a two-point qbar-q term plus gauge emissions.
    op = get_operator("RqD")
    sigs = op.lagrangian(core).vertex_signatures()
    names = {tuple(s.names) for s in sigs}
    assert ("q.bar", "q") in names
    # at least one single gauge-boson emission vertex
    assert any(len(n) == 3 and n[:1] == ("q.bar",) for n in names)


def test_psi2XD_R_has_field_strength_derivative(core):
    # RGq = (qbar T^A gamma^mu q) D^nu G^A_{mu nu}: contains a two-gluon vertex
    op = get_operator("RGq")
    sigs = {tuple(s.names) for s in op.lagrangian(core).vertex_signatures()}
    assert ("q.bar", "q", "G") in sigs
    # the non-abelian D^nu G^A_{mu nu} also yields a three-gluon emission
    assert ("q.bar", "q", "G", "G", "G") in sigs


def test_psi2XD_Rprime_dual_uses_levi_civita(core):
    op = get_operator("RpBtq")
    text = op.feynman_rule(core, core.fields.q.bar, core.fields.q, core.fields.B) \
        .expand().to_canonical_string()
    assert "levi" in text or "\u03b5" in text or "epsilon" in text.lower()


def test_psi2HD2_scalar_bilinear_has_higgs(core):
    op = get_operator("RdHD1")
    sigs = {tuple(s.names) for s in op.lagrangian(core).vertex_signatures()}
    # (qbar d) D^2 H : the leading vertex is qbar d H
    assert ("q.bar", "d", "H") in sigs


def test_psi2DH2_redundant_singlet_current(core):
    # R'(1)_Hq = (qbar i<->/D q)(H^dag H): four-point qbar q H H vertex present
    op = get_operator("Rp1Hq")
    sigs = {tuple(s.names) for s in op.lagrangian(core).vertex_signatures()}
    assert ("q.bar", "q", "H.bar", "H") in sigs


def test_non_hermitian_redundant_psi2hd2_include_hc(core):
    f = core.fields
    assert get_operator("RuHD1").feynman_rule(core, f.u.bar, f.q, f.H)
