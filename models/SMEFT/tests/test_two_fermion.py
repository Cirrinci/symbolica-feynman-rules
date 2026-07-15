"""Tests for the two-fermion operators of the SMEFT Green basis (Table 2)."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core
from models.SMEFT.registry import get_operator, operators_in


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_all_two_fermion_physical_compile(core):
    ops = operators_in(sector="two_fermion", otype="physical")
    assert len(ops) == 19
    for op in ops:
        lag = op.lagrangian(core)
        assert lag.terms, op.name
        assert op.canonical_dimensions(core) == {6}, op.name


@pytest.mark.parametrize(
    "name, contains",
    [
        ("OHq1", ("q.bar", "q", "H.bar", "H")),
        ("OHu", ("u.bar", "u", "H.bar", "H")),
        ("OHe", ("e.bar", "e", "H.bar", "H")),
        ("OuG", ("q.bar", "u", "G")),
        ("OuW", ("q.bar", "u", "W")),
        ("OeB", ("l.bar", "e", "B")),
        ("OuH", ("q.bar", "u")),
        ("OeH", ("l.bar", "e")),
    ],
)
def test_two_fermion_vertex_content(core, name, contains):
    lag = get_operator(name).lagrangian(core)
    names = [set(sig.names) for sig in lag.vertex_signatures()]
    assert any(set(contains) <= n for n in names), (name, names)


def test_currents_are_chiral(core):
    """Fermionic currents carry a chiral projector (gamma5 in the vertex)."""
    for name in ("OHq1", "OHu", "OHl1", "OHe"):
        op = get_operator(name)
        f = core.fields
        field = {"OHq1": f.q, "OHu": f.u, "OHl1": f.l, "OHe": f.e}[name]
        rule = op.feynman_rule(core, field.bar, field, f.H.bar, f.H)
        assert "gamma5" in rule.expand().to_canonical_string(), name


def test_dipole_has_sigma(core):
    op = get_operator("OuB")
    f = core.fields
    rule = op.feynman_rule(core, f.q.bar, f.u, f.H.bar, f.B)
    text = rule.expand().to_canonical_string()
    assert "sigma" in text or "\u03c3" in text


def test_non_hermitian_two_fermion_operators_include_hc(core):
    f = core.fields
    assert get_operator("OuG").feynman_rule(core, f.u.bar, f.q, f.H, f.G)
    assert get_operator("OuH").feynman_rule(core, f.u.bar, f.q, f.H, f.H.bar, f.H)
