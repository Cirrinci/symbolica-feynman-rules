from symbolica import Expression

from theories import build_standard_model


def test_charged_goldstone_flavor_expansion_does_not_multiply_disjoint_sums():
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    fields = sm.fields
    bottom = fields.dq.class_members[2]
    top = fields.uq.class_members[2]

    rule = sm.lagrangian.feynman_rule(
        bottom.bar,
        top,
        fields.GP.bar,
        simplify=True,
        include_delta=False,
        flavor_expand=True,
    )
    text = rule.to_canonical_string()

    assert "3*python" not in text
    assert "yd3" in text
    assert "yu3" in text
    assert text.count("yd3") == 1
    assert text.count("yu3") == 1
    assert rule != Expression.num(0)
