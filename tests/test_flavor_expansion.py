from fractions import Fraction
from collections import Counter

import pytest

from symbolica import S
from symbolica.community.spenso import Representation

from model import (
    COLOR_FUND_INDEX,
    Field,
    IndexRole,
    IndexType,
    Lagrangian,
    Model,
    Parameter,
    SPINOR_INDEX,
)
from model.interactions import _field_match_key


def _canon(expr):
    return expr.expand().to_canonical_string()


def _generation_index(dimension: int, *, name: str = "Generation", prefix: str = "f"):
    return IndexType(
        name,
        Representation.cof(dimension),
        kind=name.lower(),
        dimension=dimension,
        role=IndexRole.FLAVOR,
        prefix=prefix,
    )


def _dirac_field(name: str, *, symbol=None, conjugate_symbol=None, indices=()):
    if symbol is None:
        symbol = S(name)
    if conjugate_symbol is None:
        conjugate_symbol = S(f"{name}bar")
    return Field(
        name,
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=indices,
    )


def _scalar_field(name: str, *, symbol=None):
    if symbol is None:
        symbol = S(name)
    return Field(name, spin=0, self_conjugate=True, symbol=symbol)


def _flavor_family(
    generic_name: str,
    member_names: tuple[str, ...],
    generation,
    *,
    extra_indices=(),
):
    members = tuple(
        _dirac_field(
            member_name,
            symbol=S(member_name),
            conjugate_symbol=S(f"{member_name}bar"),
            indices=extra_indices + (SPINOR_INDEX,),
        )
        for member_name in member_names
    )
    generic = _dirac_field(
        generic_name,
        symbol=S(generic_name.lower()),
        conjugate_symbol=S(f"{generic_name.lower()}bar"),
        indices=(generation,) + extra_indices + (SPINOR_INDEX,),
    )
    generic = Field(
        generic.name,
        spin=generic.spin,
        self_conjugate=generic.self_conjugate,
        symbol=generic.symbol,
        conjugate_symbol=generic.conjugate_symbol,
        indices=generic.indices,
        flavor_index=generation,
        class_members=members,
    )
    return generic, members


def test_field_class_metadata_tracks_generic_member_mapping():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")

    assert psi.flavor_index is generation
    assert psi.flavor_index_slot() == 0
    assert psi.indices == (generation, SPINOR_INDEX)
    assert e.indices == (SPINOR_INDEX,)
    assert mu.indices == (SPINOR_INDEX,)
    assert tau.indices == (SPINOR_INDEX,)
    assert psi.class_member_for(1) is e
    assert psi.class_member_for(2) is mu
    assert psi.class_member_for(3) is tau

    model = Model(fields=(psi, phi), lagrangian_decl=S("g") * psi.bar(S("f")) * psi(S("f")) * phi)
    assert model.find_field(e) is e
    assert model.find_field(mu) is mu
    assert model.find_field(tau) is tau


def test_diagonal_flavor_expansion_keeps_compact_rule_and_produces_only_diagonal_members():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")
    f = S("f")
    lagrangian = Lagrangian(S("g") * psi.bar(f) * psi(f) * phi)

    compact = lagrangian.feynman_rules(
        flavor_expand=False,
        key_format="names",
        simplify=True,
        include_delta=True,
    )
    expanded = lagrangian.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(compact) == {("Psi.bar", "Psi", "Phi")}
    assert set(expanded) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("tau.bar", "tau", "Phi"),
    }

    flavor_identity = _canon(
        generation.representation.g(S("f1"), S("f2")).to_expression()
    )
    assert flavor_identity in _canon(compact[("Psi.bar", "Psi", "Phi")])
    assert "Y(" not in _canon(expanded[("e.bar", "e", "Phi")])

    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            mu,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_non_diagonal_flavor_tensor_expands_to_all_member_pairs():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")
    f, g = S("f", "g")
    yukawa = Parameter("Y", indices=(generation, generation))
    model = Model(
        fields=(psi, phi),
        parameters=(yukawa,),
        lagrangian_decl=S("g") * psi.bar(f) * yukawa(f, g) * psi(g) * phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert len(expanded) == 9
    assert "Y(1,2)" in _canon(
        lagrangian.feynman_rule(
            e.bar,
            mu,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "Y(2,1)" in _canon(
        lagrangian.feynman_rule(
            mu.bar,
            e,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "Y(3,3)" in _canon(
        lagrangian.feynman_rule(
            tau.bar,
            tau,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_flavor_expansion_matches_manual_normalization_for_mixed_and_diagonal_terms():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")
    lam = S("lam")
    f, h = S("f", "h")
    yukawa = Parameter("Y", indices=(generation, generation))

    mixed = Lagrangian(lam * psi.bar(f) * yukawa(f, h) * psi(h) * phi)
    manual = Lagrangian(lam * e.bar * yukawa(1, 2) * mu * phi)
    compact = Lagrangian(lam * psi.bar(f) * psi(f) * phi)
    manual_diag = Lagrangian(lam * e.bar * e * phi)

    mixed_rule = mixed.feynman_rule(
        e.bar,
        mu,
        phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    manual_rule = manual.feynman_rule(
        e.bar,
        mu,
        phi,
        simplify=True,
        include_delta=True,
    )
    diag_rule = compact.feynman_rule(
        e.bar,
        e,
        phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    manual_diag_rule = manual_diag.feynman_rule(
        e.bar,
        e,
        phi,
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
        _field_match_key(phi, False),
    ))
    contributors = [
        term
        for term in expanded_terms
        if Counter(_field_match_key(occ.field, occ.conjugated) for occ in term.fields) == target
    ]
    assert len(contributors) == 1
    assert _canon(contributors[0].coupling) == _canon(lam * yukawa(1, 2))


def test_diagonal_one_index_parameter_requires_allow_summation_metadata():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")
    f = S("f")
    y = Parameter("y", indices=(generation,))
    model = Model(
        fields=(psi, phi),
        parameters=(y,),
        lagrangian_decl=y(f) * psi.bar(f) * psi(f) * phi,
    )

    with pytest.raises(ValueError, match="allow_summation=True"):
        model.lagrangian().vertex_signatures(flavor_expand=True)

    y_diag = Parameter("yDiag", indices=(generation,), allow_summation=True)
    diagonal_model = Model(
        fields=(psi, phi),
        parameters=(y_diag,),
        lagrangian_decl=y_diag(f) * psi.bar(f) * psi(f) * phi,
    )
    diagonal_lagrangian = diagonal_model.lagrangian()

    expanded = diagonal_lagrangian.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("tau.bar", "tau", "Phi"),
    }
    assert "yDiag(2)" in _canon(
        diagonal_lagrangian.feynman_rule(
            mu.bar,
            mu,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_zero_flavor_tensor_components_are_dropped_after_expansion():
    generation = _generation_index(3)
    psi, (e, mu, tau) = _flavor_family("Psi", ("e", "mu", "tau"), generation)
    phi = _scalar_field("Phi")
    f, g = S("f", "g")
    diagonal_matrix = Parameter(
        "Ydiag",
        indices=(generation, generation),
        components={
            (1, 2): 0,
            (1, 3): 0,
            (2, 1): 0,
            (2, 3): 0,
            (3, 1): 0,
            (3, 2): 0,
        },
    )
    model = Model(
        fields=(psi, phi),
        parameters=(diagonal_matrix,),
        lagrangian_decl=S("g") * psi.bar(f) * diagonal_matrix(f, g) * psi(g) * phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("e.bar", "e", "Phi"),
        ("mu.bar", "mu", "Phi"),
        ("tau.bar", "tau", "Phi"),
    }
    with pytest.raises(ValueError, match="No matching interaction terms"):
        lagrangian.feynman_rule(
            e.bar,
            mu,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )


def test_gauge_indices_remain_symbolic_when_flavor_members_expand():
    generation = _generation_index(2)
    q, (d, u) = _flavor_family(
        "q",
        ("d", "u"),
        generation,
        extra_indices=(COLOR_FUND_INDEX,),
    )
    phi = _scalar_field("Phi")
    f, c = S("f", "c")
    lagrangian = Lagrangian(S("g") * q.bar(f, c) * q(f, c) * phi)

    signatures = lagrangian.vertex_signatures(flavor_expand=True)
    expanded_terms = lagrangian._expanded_terms(flavor_expand=True)

    assert {signature.names for signature in signatures} == {
        ("d.bar", "d", "Phi"),
        ("u.bar", "u", "Phi"),
    }
    assert len(expanded_terms) == 2
    for term in expanded_terms:
        assert term.fields[0].labels["color_fund"] == c
        assert term.fields[1].labels["color_fund"] == c


def test_two_flavor_classes_can_share_one_generation_label():
    generation = _generation_index(3)
    left, (e_left, mu_left, tau_left) = _flavor_family(
        "L",
        ("eL", "muL", "tauL"),
        generation,
    )
    right, (e_right, mu_right, tau_right) = _flavor_family(
        "R",
        ("eR", "muR", "tauR"),
        generation,
    )
    phi = _scalar_field("Phi")
    f = S("f")
    y = Parameter("yLR", indices=(generation,), allow_summation=True)
    model = Model(
        fields=(left, right, phi),
        parameters=(y,),
        lagrangian_decl=y(f) * left.bar(f) * right(f) * phi,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rules(
        flavor_expand=True,
        key_format="names",
        simplify=True,
        include_delta=True,
    )

    assert set(expanded) == {
        ("eL.bar", "eR", "Phi"),
        ("muL.bar", "muR", "Phi"),
        ("tauL.bar", "tauR", "Phi"),
    }
    assert "yLR(3)" in _canon(
        lagrangian.feynman_rule(
            tau_left.bar,
            tau_right,
            phi,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_independent_flavor_labels_expand_across_two_field_classes():
    generation = _generation_index(3)
    up, (u, c, t) = _flavor_family("U", ("u", "c", "t"), generation)
    down, (d, s, b) = _flavor_family("D", ("d", "s", "b"), generation)
    w = _scalar_field("W")
    f, g = S("f", "g")
    mixing = Parameter("V", indices=(generation, generation))
    model = Model(
        fields=(up, down, w),
        parameters=(mixing,),
        lagrangian_decl=up.bar(f) * mixing(f, g) * down(g) * w,
    )
    lagrangian = model.lagrangian()

    expanded = lagrangian.feynman_rules(
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
            w,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )
    assert "V(3,3)" in _canon(
        lagrangian.feynman_rule(
            t.bar,
            b,
            w,
            simplify=True,
            include_delta=True,
            flavor_expand=True,
        )
    )


def test_invalid_flavor_class_declarations_and_missing_members_raise_clear_errors():
    generation = _generation_index(3)

    with pytest.raises(ValueError, match="declares 2 class member"):
        _flavor_family("BadPsi", ("e", "mu"), generation)

    member_with_flavor = Field(
        "bad_member",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("bad_member"),
        conjugate_symbol=S("bad_memberbar"),
        indices=(generation, SPINOR_INDEX),
    )
    with pytest.raises(ValueError, match="still carries a flavor index"):
        Field(
            "BadClass",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("badclass"),
            conjugate_symbol=S("badclassbar"),
            indices=(generation, SPINOR_INDEX),
            flavor_index=generation,
            class_members=(member_with_flavor, member_with_flavor, member_with_flavor),
        )

    psi = Field(
        "PsiNoMembers",
        spin=Fraction(1, 2),
        self_conjugate=False,
        symbol=S("psinm"),
        conjugate_symbol=S("psinmbar"),
        indices=(generation, SPINOR_INDEX),
        flavor_index=generation,
    )
    phi = _scalar_field("PhiMissing")
    f = S("f")
    lagrangian = Lagrangian(S("g") * psi.bar(f) * psi(f) * phi)

    with pytest.raises(ValueError, match="no class members are defined"):
        lagrangian.vertex_signatures(flavor_expand=True)
