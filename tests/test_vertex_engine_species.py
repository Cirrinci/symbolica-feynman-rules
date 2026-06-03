from symbolica import S

from symbolic.vertex_engine import factor_leg_compatible


def test_factor_leg_compatible_requires_matching_species():
    phi, chi = S("phi", "chi")

    assert factor_leg_compatible(0, 0, [phi], [phi])
    assert not factor_leg_compatible(0, 0, [phi], [chi])


def test_factor_leg_compatible_checks_species_before_role_match():
    phi, chi = S("phi", "chi")

    assert not factor_leg_compatible(
        0,
        0,
        [phi],
        [chi],
        field_roles=["scalar"],
        leg_roles=["scalar"],
    )


def test_factor_leg_compatible_accepts_aliased_leg_species_with_match_keys():
    phi, alias = S("phi", "b1")
    key = ("Phi", "scalar")

    assert factor_leg_compatible(
        0,
        0,
        [phi],
        [alias],
        field_roles=["scalar"],
        leg_roles=["scalar"],
        field_match_keys=[key],
        leg_match_keys=[key],
    )
