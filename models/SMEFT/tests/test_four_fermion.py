"""Tests for the four-fermion operators of the SMEFT Green basis (Table 3)."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core
from models.SMEFT.registry import get_operator, operators_in


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_all_four_fermion_physical_compile(core):
    ops = operators_in(sector="four_fermion", otype="physical")
    assert len(ops) == 25
    for op in ops:
        lag = op.lagrangian(core)
        assert lag.terms, op.name
        assert op.canonical_dimensions(core) == {6}, op.name


def test_four_fermion_have_four_fermions(core):
    for op in operators_in(sector="four_fermion", otype="physical"):
        sig = op.lagrangian(core).vertex_signatures()[0]
        # exactly four fermion legs
        fermions = [n for n in sig.names if n.rstrip(".bar") in ("q", "l", "u", "d", "e")]
        assert len(fermions) == 4, (op.name, sig.names)


def test_octet_operators_carry_colour_generator(core):
    op = get_operator("Oqu8")
    f = core.fields
    rule = op.feynman_rule(core, f.q.bar, f.q, f.u.bar, f.u)
    text = rule.expand().to_canonical_string()
    assert "coad" in text  # colour adjoint index from T^A


def test_tensor_operator_has_sigma(core):
    op = get_operator("Olequ3")
    f = core.fields
    rule = op.feynman_rule(core, f.l.bar, f.e, f.q.bar, f.u)
    text = rule.expand().to_canonical_string()
    assert "sigma" in text or "\u03c3" in text


def test_non_hermitian_four_fermion_include_hc(core):
    f = core.fields
    assert get_operator("Oquqd1").feynman_rule(core, f.u.bar, f.q, f.d.bar, f.q)
    assert get_operator("Olequ3").feynman_rule(core, f.e.bar, f.l, f.u.bar, f.q)
