from collections import Counter

import pytest
import feynpy.flavor as flavor_module

from symbolica import S

from feynpy import (
    COLOR_FUND_INDEX,
    Field,
    Model,
    Parameter,
    SPINOR_INDEX,
    dirac_field,
    flavor_index,
    scalar_field,
)
from feynpy.interactions import _field_match_key


def _canon(expr):
    return expr.expand().to_canonical_string()


def _compiled(expr, *, parameters=()):
    return Model(parameters=parameters, lagrangian_decl=expr).lagrangian()


def _off_diagonal_zero_components(index):
    return {
        (row, col): 0
        for row in range(1, index.dimension + 1)
        for col in range(1, index.dimension + 1)
        if row != col
    }


def _charged_lepton_class(generation):
    l = dirac_field(
        "l",
        class_members=("e", "mu", "ta"),
        indices=(generation,),
        flavor_index=generation,
    )
    return l, l.class_members


def _up_quark_class(generation):
    uq = dirac_field(
        "uq",
        class_members=("u", "c", "t"),
        indices=(generation, COLOR_FUND_INDEX),
        flavor_index=generation,
    )
    return uq, uq.class_members


def _down_quark_class(generation):
    dq = dirac_field(
        "dq",
        class_members=("d", "s", "b"),
        indices=(generation, COLOR_FUND_INDEX),
        flavor_index=generation,
    )
    return dq, dq.class_members


def test_charged_lepton_field_class_metadata_exposes_feynrules_like_structure():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")

    assert l.indices == (Generation, SPINOR_INDEX)
    assert l.flavor_index is Generation
    assert l.flavor_index_slot() == 0
    assert tuple(member.name for member in l.class_members) == ("e", "mu", "ta")
    assert e.indices == (SPINOR_INDEX,)
    assert mu.indices == (SPINOR_INDEX,)
    assert ta.indices == (SPINOR_INDEX,)
    assert l.class_member_for(1) is e
    assert l.class_member_for(2) is mu
    assert l.class_member_for(3) is ta

    model = Model(
        fields=(l, Phi),
        lagrangian_decl=S("g") * l.bar(S("f")) * l(S("f")) * Phi,
    )
    assert model.find_field(e) is e
    assert model.find_field(mu) is mu
    assert model.find_field(ta) is ta


def test_up_quark_field_class_metadata_preserves_colour_and_spinor():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (u, c, t) = _up_quark_class(Generation)

    assert uq.indices == (Generation, COLOR_FUND_INDEX, SPINOR_INDEX)
    assert uq.flavor_index is Generation
    assert uq.flavor_index_slot() == 0
    assert u.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert c.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert t.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert Generation not in u.indices
    assert Generation not in c.indices
    assert Generation not in t.indices


def test_down_quark_field_class_metadata_preserves_colour_and_spinor():
    Generation = flavor_index("Generation", 3, prefix="f")
    dq, (d, s, b) = _down_quark_class(Generation)

    assert dq.indices == (Generation, COLOR_FUND_INDEX, SPINOR_INDEX)
    assert dq.flavor_index is Generation
    assert dq.flavor_index_slot() == 0
    assert d.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert s.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert b.indices == (COLOR_FUND_INDEX, SPINOR_INDEX)
    assert Generation not in d.indices
    assert Generation not in s.indices
    assert Generation not in b.indices


def test_diagonal_flavor_expansion_keeps_compact_rule_and_produces_only_diagonal_members():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f = S("f")
    lagrangian = _compiled(S("g") * l.bar(f) * l(f) * Phi)

    compact = lagrangian.feynman_rule(
        flavor_expand=False,
        key_format="names",
        simplify=True,
        include_delta=True,
    )
    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(compact) == {("l.bar", "l", "Phi")}
    assert set(expanded) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("ta.bar", "ta", "Phi"),
    }

    flavor_identity = _canon(
        Generation.representation.g(S("f1"), S("f2")).to_expression()
    )
    assert flavor_identity in _canon(compact[("l.bar", "l", "Phi")])
    assert "g(1,2)" not in _canon(expanded[("e.bar", "e", "Phi")])

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            mu,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_non_diagonal_flavor_tensor_expands_to_all_member_pairs():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f, h = S("f", "h")
    yl = Parameter("yl", indices=(Generation, Generation))
    model = Model(
        fields=(l, Phi),
        parameters=(yl,),
        lagrangian_decl=S("g") * l.bar(f) * yl(f, h) * l(h) * Phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert len(expanded) == 9
    assert "yl(1,2)" in _canon(
        lagrangian.feynman_rule(
            e.bar,
            mu,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "yl(2,1)" in _canon(
        lagrangian.feynman_rule(
            mu.bar,
            e,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "yl(3,3)" in _canon(
        lagrangian.feynman_rule(
            ta.bar,
            ta,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_generic_unitary_parameter_products_contract_to_flavor_metric():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, _ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f, h, g = S("f", "h", "g")
    U = Parameter(
        "U",
        indices=(Generation, Generation),
        unitary_partner="UDag",
    )
    UDag = Parameter(
        "UDag",
        indices=(Generation, Generation),
        unitary_partner="U",
    )
    lagrangian = _compiled(
        S("lam") * l.bar(f) * U(f, h) * UDag(h, g) * l(g) * Phi,
        parameters=(U, UDag),
    )

    compact = lagrangian.feynman_rule(
        l.bar,
        l,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=False,
    )
    compact_text = _canon(compact)
    assert "U(" not in compact_text
    assert "UDag(" not in compact_text
    assert _canon(Generation.representation.g(S("f1"), S("f2")).to_expression()) in compact_text

    diagonal = lagrangian.feynman_rule(
        e.bar,
        e,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    diagonal_text = _canon(diagonal)
    assert "U(" not in diagonal_text
    assert "UDag(" not in diagonal_text

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            mu,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_generic_unitary_contraction_handles_transposed_orientation():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, _ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f, h, g = S("f", "h", "g")
    U = Parameter(
        "U",
        indices=(Generation, Generation),
        unitary_partner="UDag",
    )
    UDag = Parameter(
        "UDag",
        indices=(Generation, Generation),
        unitary_partner="U",
    )
    lagrangian = _compiled(
        S("lam") * l.bar(f) * U(h, f) * UDag(g, h) * l(g) * Phi,
        parameters=(U, UDag),
    )

    compact = lagrangian.feynman_rule(
        l.bar,
        l,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=False,
    )
    compact_text = _canon(compact)
    assert "U(" not in compact_text
    assert "UDag(" not in compact_text
    assert _canon(Generation.representation.g(S("f1"), S("f2")).to_expression()) in compact_text

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            mu,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_flavor_expansion_matches_manual_normalization_for_mixed_and_diagonal_terms():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    lam = S("lam")
    f, h = S("f", "h")
    yl = Parameter("yl", indices=(Generation, Generation))

    mixed = _compiled(lam * l.bar(f) * yl(f, h) * l(h) * Phi, parameters=(yl,))
    manual = _compiled(lam * e.bar * yl(1, 2) * mu * Phi, parameters=(yl,))
    compact = _compiled(lam * l.bar(f) * l(f) * Phi)
    manual_diag = _compiled(lam * e.bar * e * Phi)

    mixed_rule = mixed.feynman_rule(
        e.bar,
        mu,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    manual_rule = manual.feynman_rule(
        e.bar,
        mu,
        Phi,
        simplify=True,
        include_delta=True,
    )
    diag_rule = compact.feynman_rule(
        e.bar,
        e,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    manual_diag_rule = manual_diag.feynman_rule(
        e.bar,
        e,
        Phi,
        simplify=True,
        include_delta=True,
    )

    assert _canon(mixed_rule - manual_rule) == "0"
    assert _canon(diag_rule - manual_diag_rule) == "0"

    expanded_terms = mixed._expanded_terms(flavor_expand=True)
    assert len(expanded_terms) == 9

    target = Counter((
        _field_match_key(e, True),
        _field_match_key(mu, False),
        _field_match_key(Phi, False),
    ))
    contributors = [
        term
        for term in expanded_terms
        if Counter(_field_match_key(occ.field, occ.conjugated) for occ in term.fields) == target
    ]
    assert len(contributors) == 1
    assert _canon(contributors[0].coupling) == _canon(lam * yl(1, 2))


def test_diagonal_one_index_flavor_parameter_allows_summation_by_default():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f = S("f")
    y = Parameter("y", indices=(Generation,))
    model = Model(
        fields=(l, Phi),
        parameters=(y,),
        lagrangian_decl=y(f) * l.bar(f) * l(f) * Phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("ta.bar", "ta", "Phi"),
    }
    assert "y(2)" in _canon(
        lagrangian.feynman_rule(
            mu.bar,
            mu,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_diagonal_one_index_flavor_parameter_can_opt_out_of_summation():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, mu, ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f = S("f")
    y = Parameter("y", indices=(Generation,), allow_summation=False)
    model = Model(
        fields=(l, Phi),
        parameters=(y,),
        lagrangian_decl=y(f) * l.bar(f) * l(f) * Phi,
    )

    with pytest.raises(ValueError, match="allow_summation=True"):
        model.lagrangian().vertex_signatures(flavor_expand=True)


def test_zero_flavor_tensor_components_are_dropped_after_expansion():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (u, c, t) = _up_quark_class(Generation)
    Phi = scalar_field("Phi")
    f, h, colour = S("f", "h", "c")
    yu = Parameter(
        "yu",
        indices=(Generation, Generation),
        components=_off_diagonal_zero_components(Generation),
    )
    model = Model(
        fields=(uq, Phi),
        parameters=(yu,),
        lagrangian_decl=S("g") * uq.bar(f, colour) * yu(f, h) * uq(h, colour) * Phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("u.bar", "u", "Phi"),
        ("c.bar", "c", "Phi"),
        ("t.bar", "t", "Phi"),
    }
    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            u.bar,
            c,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_flavor_expansion_preserves_colour_indices_in_terms_and_rules():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (u, c, t) = _up_quark_class(Generation)
    Phi = scalar_field("Phi")
    f, colour = S("f", "c")
    gQ = S("gQ")
    lagrangian = _compiled(gQ * uq.bar(f, colour) * uq(f, colour) * Phi)

    signatures = lagrangian.vertex_signatures(flavor_expand=True)
    expanded_terms = lagrangian._expanded_terms(flavor_expand=True)
    expanded_rule = lagrangian.feynman_rule(
        u.bar,
        u,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    manual_rule = _compiled(gQ * u.bar(colour) * u(colour) * Phi).feynman_rule(
        u.bar,
        u,
        Phi,
        simplify=True,
        include_delta=True,
    )

    assert {signature.names for signature in signatures} == {
        ("u.bar", "u", "Phi"),
        ("c.bar", "c", "Phi"),
        ("t.bar", "t", "Phi"),
    }
    assert len(expanded_terms) == 3
    for term in expanded_terms:
        assert term.fields[0].labels["color_fund"] == colour
        assert term.fields[1].labels["color_fund"] == colour
    assert _canon(expanded_rule - manual_rule) == "0"


def test_selected_flavor_expand_can_target_one_index_type():
    Generation = flavor_index("Generation", 3, prefix="f")
    SU2D = flavor_index("SU2D", 2, prefix="d")
    uq, (_u, _c, _t) = _up_quark_class(Generation)
    chi = dirac_field(
        "chi",
        class_members=("chi1", "chi2"),
        indices=(SU2D,),
        flavor_index=SU2D,
    )
    _chi1, _chi2 = chi.class_members
    Phi = scalar_field("Phi")
    f, d, colour = S("f", "d", "c")
    lagrangian = _compiled(S("g") * uq.bar(f, colour) * chi(d) * Phi)

    generation_only = lagrangian.vertex_signatures(flavor_expand=Generation)
    su2d_only = lagrangian.vertex_signatures(flavor_expand=SU2D)
    selected_both = lagrangian.vertex_signatures(flavor_expand=(Generation, SU2D))

    assert {signature.names for signature in generation_only} == {
        ("u.bar", "chi", "Phi"),
        ("c.bar", "chi", "Phi"),
        ("t.bar", "chi", "Phi"),
    }
    assert {signature.names for signature in su2d_only} == {
        ("uq.bar", "chi1", "Phi"),
        ("uq.bar", "chi2", "Phi"),
    }
    assert len(selected_both) == 6


def test_flavor_expansion_cache_reuses_expanded_terms_across_rule_queries(monkeypatch):
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, _mu, _ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    f = S("f")
    lagrangian = _compiled(S("g") * l.bar(f) * l(f) * Phi)

    calls = {"count": 0}
    original = flavor_module.expand_flavor_terms

    def counted_expand_flavor_terms(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(flavor_module, "expand_flavor_terms", counted_expand_flavor_terms)

    expanded_first = lagrangian._expanded_terms(flavor_expand=True)
    expanded_second = lagrangian._expanded_terms(flavor_expand=True)
    signatures = lagrangian.vertex_signatures(flavor_expand=True)
    rules = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )
    vertex = lagrangian.feynman_rule(
        e.bar,
        e,
        Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )

    assert expanded_first is expanded_second
    assert calls["count"] == 1
    assert {signature.names for signature in signatures} == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("ta.bar", "ta", "Phi"),
    }
    assert set(rules) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("ta.bar", "ta", "Phi"),
    }
    assert _canon(vertex - rules[("e.bar", "e", "Phi")]) == "0"


def test_flavor_expansion_cache_keys_by_normalized_selection(monkeypatch):
    Generation = flavor_index("Generation", 3, prefix="f")
    SU2D = flavor_index("SU2D", 2, prefix="d")
    uq, (_u, _c, _t) = _up_quark_class(Generation)
    chi = dirac_field(
        "chi",
        class_members=("chi1", "chi2"),
        indices=(SU2D,),
        flavor_index=SU2D,
    )
    Phi = scalar_field("Phi")
    f, d, colour = S("f", "d", "c")
    lagrangian = _compiled(S("g") * uq.bar(f, colour) * chi(d) * Phi)

    calls = {"count": 0}
    original = flavor_module.expand_flavor_terms

    def counted_expand_flavor_terms(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(flavor_module, "expand_flavor_terms", counted_expand_flavor_terms)

    generation_first = lagrangian._expanded_terms(flavor_expand=Generation)
    generation_second = lagrangian._expanded_terms(flavor_expand=(Generation,))
    su2_first = lagrangian._expanded_terms(flavor_expand=SU2D)
    su2_second = lagrangian._expanded_terms(flavor_expand=[SU2D])
    both_first = lagrangian._expanded_terms(flavor_expand=(Generation, SU2D))
    both_second = lagrangian._expanded_terms(flavor_expand=[Generation, SU2D])
    both_reversed = lagrangian._expanded_terms(flavor_expand=(SU2D, Generation))

    assert generation_first is generation_second
    assert su2_first is su2_second
    assert both_first is both_second
    assert both_first is both_reversed
    assert calls["count"] == 3
    assert len(lagrangian._expanded_terms_cache) == 3

    assert {signature.names for signature in lagrangian.vertex_signatures(flavor_expand=Generation)} == {
        ("u.bar", "chi", "Phi"),
        ("c.bar", "chi", "Phi"),
        ("t.bar", "chi", "Phi"),
    }
    assert {signature.names for signature in lagrangian.vertex_signatures(flavor_expand=SU2D)} == {
        ("uq.bar", "chi1", "Phi"),
        ("uq.bar", "chi2", "Phi"),
    }
    assert len(lagrangian.vertex_signatures(flavor_expand=(Generation, SU2D))) == 6


def test_smeft_like_three_generation_labels_expand_correctly():
    Generation = flavor_index("Generation", 3, prefix="f")
    left = dirac_field(
        "lL",
        class_members=("eL", "muL", "taL"),
        indices=(Generation,),
        flavor_index=Generation,
    )
    e_left, _mu_left, _ta_left = left.class_members
    right = dirac_field(
        "lR",
        class_members=("eR", "muR", "taR"),
        indices=(Generation,),
        flavor_index=Generation,
    )
    _e_right, mu_right, _ta_right = right.class_members
    chi = scalar_field(
        "chi",
        class_members=("chi1", "chi2", "chi3"),
        indices=(Generation,),
        flavor_index=Generation,
    )
    _chi1, _chi2, chi3 = chi.class_members
    Phi = scalar_field("Phi")
    f, h, k = S("f", "h", "k")
    coefficient = Parameter("C", indices=(Generation, Generation, Generation))
    model = Model(
        fields=(left, right, chi, Phi),
        parameters=(coefficient,),
        lagrangian_decl=coefficient(f, h, k) * left.bar(f) * right(h) * chi(k) * Phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert len(expanded) == 27
    assert ("eL.bar", "muR", "chi3", "Phi") in expanded
    assert "C(1,2,3)" in _canon(
        lagrangian.feynman_rule(
            e_left.bar,
            mu_right,
            chi3,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_flavor_expansion_cache_does_not_leak_across_rebuild_or_term_mutation():
    Generation = flavor_index("Generation", 3, prefix="f")
    l, (e, _mu, _ta) = _charged_lepton_class(Generation)
    Phi = scalar_field("Phi")
    Xi = scalar_field("Xi")
    f = S("f")
    lagrangian = _compiled(S("g") * l.bar(f) * l(f) * Phi)

    expanded_before = lagrangian._expanded_terms(flavor_expand=True)
    assert len(expanded_before) == 3
    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            e,
            Xi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )

    rebuilt = _compiled(S("g") * l.bar(f) * l(f) * Phi + S("h") * l.bar(f) * l(f) * Xi)
    rebuilt_expanded = rebuilt._expanded_terms(flavor_expand=True)
    assert len(rebuilt_expanded) == 6
    assert any(term.fields[-1].field is Xi for term in rebuilt_expanded)

    extra_terms = _compiled(S("h") * l.bar(f) * l(f) * Xi).terms
    lagrangian.terms = lagrangian.terms + extra_terms
    expanded_after_mutation = lagrangian._expanded_terms(flavor_expand=True)
    manual_xi = _compiled(S("h") * e.bar * e * Xi).feynman_rule(
        e.bar,
        e,
        Xi,
        simplify=True,
        include_delta=True,
    )

    assert expanded_after_mutation is not expanded_before
    assert len(expanded_after_mutation) == 6
    assert any(term.fields[-1].field is Xi for term in expanded_after_mutation)
    assert _canon(
        lagrangian.feynman_rule(
            e.bar,
            e,
            Xi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
        - manual_xi
    ) == "0"


def test_two_flavor_classes_can_share_one_generation_label():
    Generation = flavor_index("Generation", 3, prefix="f")
    left = dirac_field(
        "lL",
        class_members=("eL", "muL", "taL"),
        indices=(Generation,),
        flavor_index=Generation,
    )
    e_left, mu_left, ta_left = left.class_members
    right = dirac_field(
        "lR",
        class_members=("eR", "muR", "taR"),
        indices=(Generation,),
        flavor_index=Generation,
    )
    e_right, mu_right, ta_right = right.class_members
    Phi = scalar_field("Phi")
    f = S("f")
    y = Parameter("yLR", indices=(Generation,), allow_summation=True)
    model = Model(
        fields=(left, right, Phi),
        parameters=(y,),
        lagrangian_decl=y(f) * left.bar(f) * right(f) * Phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("eL.bar", "eR", "Phi"),
        ("muL.bar", "muR", "Phi"),
        ("taL.bar", "taR", "Phi"),
    }
    assert "yLR(3)" in _canon(
        lagrangian.feynman_rule(
            ta_left.bar,
            ta_right,
            Phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_independent_flavor_labels_expand_across_two_field_classes():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (u, c, t) = _up_quark_class(Generation)
    dq, (d, s, b) = _down_quark_class(Generation)
    W = scalar_field("W")
    f, h, colour = S("f", "h", "c")
    V = Parameter("V", indices=(Generation, Generation))
    model = Model(
        fields=(uq, dq, W),
        parameters=(V,),
        lagrangian_decl=uq.bar(f, colour) * V(f, h) * dq(h, colour) * W,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rule(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert len(expanded) == 9
    assert "V(1,2)" in _canon(
        lagrangian.feynman_rule(
            u.bar,
            s,
            W,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "V(3,3)" in _canon(
        lagrangian.feynman_rule(
            t.bar,
            b,
            W,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_invalid_flavor_class_declarations_and_missing_members_raise_clear_errors():
    Generation = flavor_index("Generation", 3, prefix="f")

    with pytest.raises(ValueError, match="declares 2 class member"):
        dirac_field(
            "BadPsi",
            class_members=("e", "mu"),
            indices=(Generation,),
            flavor_index=Generation,
        )

    member_with_flavor = dirac_field("bad_member", indices=(Generation,))
    with pytest.raises(ValueError, match="still carries a flavor index"):
        Field(
            "BadClass",
            spin=member_with_flavor.spin,
            self_conjugate=member_with_flavor.self_conjugate,
            symbol=S("badclass"),
            conjugate_symbol=S("badclassbar"),
            indices=(Generation, SPINOR_INDEX),
            flavor_index=Generation,
            class_members=(member_with_flavor, member_with_flavor, member_with_flavor),
        )

    template = dirac_field(
        "PsiNoMembers",
        indices=(Generation,),
        symbol=S("psinm"),
        conjugate_symbol=S("psinmbar"),
    )
    psi = Field(
        template.name,
        spin=template.spin,
        self_conjugate=template.self_conjugate,
        indices=template.indices,
        kind=template.kind,
        statistics=template.statistics,
        symbol=template.symbol,
        conjugate_symbol=template.conjugate_symbol,
        flavor_index=Generation,
    )
    Phi = scalar_field("PhiMissing")
    f = S("f")
    lagrangian = _compiled(S("g") * psi.bar(f) * psi(f) * Phi)

    with pytest.raises(ValueError, match="no class members are defined"):
        lagrangian.vertex_signatures(flavor_expand=True)


def test_rejects_same_label_used_as_generation_and_colour():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (_u, _c, _t) = _up_quark_class(Generation)
    Phi = scalar_field("Phi")
    f, col = S("f", "col")

    with pytest.raises(ValueError, match="incompatible index types"):
        Model(
            fields=(uq, Phi),
            lagrangian_decl=uq.bar(f, col) * uq(col, f) * Phi,
        ).lagrangian()

def test_rejects_parameter_label_used_in_incompatible_index_space():
    Generation = flavor_index("Generation", 3, prefix="f")
    uq, (_u, _c, _t) = _up_quark_class(Generation)
    Phi = scalar_field("Phi")
    yu = Parameter("yu", indices=(Generation, Generation))
    f, col = S("f", "col")

    with pytest.raises(ValueError, match="incompatible index types"):
        Model(
            fields=(uq, Phi),
            parameters=(yu,),
            lagrangian_decl=yu(f, col) * uq.bar(f, col) * uq(f, col) * Phi,
        ).lagrangian()
