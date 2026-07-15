"""Tests for the unbroken Standard Model foundation of the SMEFT Green basis."""

from __future__ import annotations

import pytest

from models.SMEFT.sm_core import build_sm_core, occ


@pytest.fixture(scope="module")
def core():
    return build_sm_core()


def test_core_imports_and_builds(core):
    assert core.name == "SMEFT"
    assert core.renormalizable.terms


def test_field_hypercharges(core):
    f = core.fields
    from fractions import Fraction

    def y(field):
        return Fraction(str(field.quantum_numbers["Y"]))

    assert y(f.q) == Fraction(1, 6)
    assert y(f.l) == Fraction(-1, 2)
    assert y(f.u) == Fraction(2, 3)
    assert y(f.d) == Fraction(-1, 3)
    assert y(f.e) == Fraction(-1)
    assert y(f.H) == Fraction(1, 2)


def test_renormalizable_has_gauge_fermion_vertices(core):
    L = core.renormalizable
    f = core.fields
    assert L.vertex_signatures(signature=(f.q.bar, f.q, f.B))
    assert L.vertex_signatures(signature=(f.q.bar, f.q, f.G))
    assert L.vertex_signatures(signature=(f.l.bar, f.l, f.W))


def test_renormalizable_gauge_fermion_vertex_is_left_chiral(core):
    """The q-qbar-B vertex must project onto the left-handed component."""
    L = core.renormalizable
    f = core.fields
    rule = L.feynman_rule(f.q.bar, f.q, f.B)
    text = rule.expand().to_canonical_string()
    # A chiral coupling contains gamma5 (P_L = (1 - gamma5)/2); a vector-like
    # coupling would not.
    assert "gamma5" in text


def test_occ_requires_all_labels(core):
    from symbolica import S

    f = core.fields
    # q carries (spinor, weak, generation, colour): all four must be provided.
    with pytest.raises(TypeError):
        occ(f.q, sp=S("s"))
    occurrence = occ(f.q, sp=S("s"), w=S("w"), f=S("g"), c=S("c"))
    assert occurrence is not None
