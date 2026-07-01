"""Public naming compatibility for FeynRules-style declaration helpers."""

from symbolica import S

from feynpy import (
    CovD,
    CovariantDerivativeFactor,
    DC,
    FS,
    FieldStrength,
    FieldStrengthFactor,
)
from tests.support.builders import make_complex_scalar, make_u1


def test_feynrules_names_and_descriptive_names_are_the_same_callables():
    assert DC is CovD
    assert FS is FieldStrength


def test_feynrules_names_construct_the_existing_factor_types():
    mu, nu = S("mu", "nu")
    phi = make_complex_scalar("Phi")
    u1 = make_u1(S("g1"), S("B"), name="U1Y")

    derivative = DC(phi, mu)
    strength = FS(u1, mu, nu)

    assert isinstance(derivative, CovariantDerivativeFactor)
    assert isinstance(strength, FieldStrengthFactor)
    assert str(derivative) == "DC(Phi, mu)"
    assert str(strength) == "FS(U1Y, mu, nu)"

