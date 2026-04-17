import sys
from pathlib import Path

import pytest


# Allow importing from repo `src/` without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S, Expression  # noqa: E402

from gauge_compiler import compile_covariant_terms, expand_cov_der  # noqa: E402
from model import (  # noqa: E402
    COLOR_FUND_INDEX,
    COLOR_ADJ_INDEX,
    LORENTZ_INDEX,
    CovD,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    Model,
    COLOR_FUND_KIND,
    COLOR_ADJ_KIND,
    LORENTZ_KIND,
)
from model_symbolica import Delta, I, pi, pcomp, simplify_deltas, vertex_factor  # noqa: E402
from operators import scalar_gauge_contact  # noqa: E402
from spenso_structures import gauge_generator, structure_constant  # noqa: E402
from tensor_canonicalization import canonize_spenso_tensors  # noqa: E402


def _make_bislot_scalar():
    phi = S("phi")
    phidag = S("phidag")
    return Field(
        "PhiBiTest",
        spin=0,
        self_conjugate=False,
        symbol=phi,
        conjugate_symbol=phidag,
        indices=(COLOR_FUND_INDEX, COLOR_FUND_INDEX),
    )


def _make_gluon():
    G = S("G")
    return Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=G,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )


def _make_bislot_sum_model():
    gS = S("gS")
    scalar = _make_bislot_scalar()
    gluon = _make_gluon()
    rep_sum = GaugeRepresentation(
        index=COLOR_FUND_INDEX,
        generator_builder=gauge_generator,
        name="fund_sum",
        slot_policy="sum",
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(rep_sum,),
    )
    model = Model(
        name="bislot-sum",
        gauge_groups=(su3,),
        fields=(scalar, gluon),
        lagrangian_decl=CovD(scalar.bar, S("mu_decl")) * CovD(scalar, S("mu_decl")),
    )
    return model, scalar, gluon, su3


def _model_vertex(*, interaction, external_legs, species_map):
    x = S("x")
    d = S("d")
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=x,
        d=d,
    )
    return simplify_deltas(expr, species_map=species_map)


def _symmetrized_generator_contact(adj_left, adj_right, color_left, color_right, color_middle):
    return (
        gauge_generator(adj_left, color_left, color_middle)
        * gauge_generator(adj_right, color_middle, color_right)
        + gauge_generator(adj_right, color_left, color_middle)
        * gauge_generator(adj_left, color_middle, color_right)
    )


def test_ambiguity_is_error_by_default_for_repeated_slots():
    gS = S("gS")
    scalar = _make_bislot_scalar()
    gluon = _make_gluon()

    rep_unique_default = GaugeRepresentation(
        index=COLOR_FUND_INDEX,
        generator_builder=gauge_generator,
        name="fund",
        # slot_policy defaults to "unique"
    )
    su3 = GaugeGroup(
        name="SU3",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(rep_unique_default,),
    )
    model = Model(
        name="bislot-ambiguous",
        gauge_groups=(su3,),
        fields=(scalar, gluon),
        lagrangian_decl=CovD(scalar.bar, S("mu_decl")) * CovD(scalar, S("mu_decl")),
    )

    with pytest.raises(ValueError, match=r"repeated index type|slot_policy='sum'|slot=\\.+"):
        compile_covariant_terms(model)


def test_slot_policy_sum_expands_currents_and_contacts_over_slots():
    model, _, _, _ = _make_bislot_sum_model()
    compiled = compile_covariant_terms(model)

    # Complex-scalar kinetic term, bislot with slot_policy='sum':
    # - currents: 2 slots * 2 (phi vs phidag derivative placement) = 4 terms
    # - contact: ordered slot pairs (0,0), (0,1), (1,0), (1,1) = 4 terms
    kinetic_terms = [t for t in compiled if "scalar" in t.label]
    assert len(kinetic_terms) == 8

    contact_terms = [t for t in kinetic_terms if "contact" in t.label]
    assert len(contact_terms) == 4

    current_terms = [t for t in kinetic_terms if "current" in t.label]
    assert len(current_terms) == 4

    # Sanity: each contact term must contain two gauge fields in its InteractionTerm fields.
    for term in contact_terms:
        assert len(term.fields) == 4


def test_expand_cov_der_exposes_repeated_slot_metadata():
    model, scalar, _, su3 = _make_bislot_sum_model()

    expanded = expand_cov_der(model, CovD(scalar, S("mu_decl")))

    assert len(expanded.gauge_current_pieces) == 2
    assert expanded.contact_ready_data == expanded.gauge_current_pieces

    active_slots = tuple(piece.active_slot for piece in expanded.gauge_current_pieces)
    assert active_slots == (0, 1)

    for piece in expanded.gauge_current_pieces:
        assert piece.metadata.gauge_group is su3
        assert piece.metadata.representation is not None
        assert piece.metadata.representation.name == "fund_sum"
        assert piece.metadata.representation_slots == (0, 1)
        assert piece.metadata.repeated_index is True
        assert piece.metadata.conjugated is False


def test_slot_policy_sum_current_matches_expected_bislot_vertex():
    d = S("d")
    p1, p2, p3 = S("p1", "p2", "p3")
    b1, b2, b3 = S("b1", "b2", "b3")
    c1, c2, c3, c4 = S("c1", "c2", "c3", "c4")
    mu3 = S("mu3")
    a3 = S("a3")
    gS = S("gS")

    model, scalar, gluon, _ = _make_bislot_sum_model()
    compiled = compile_covariant_terms(model)
    current_terms = [term for term in compiled if "current" in term.label]
    assert len(current_terms) == 4

    current_index = current_terms[0].derivatives[0].lorentz_index
    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: (c1, c3)}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: (c2, c4)}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
    )
    species_map = {b1: scalar.conjugate_symbol, b2: scalar.symbol, b3: gluon.symbol}
    got = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=legs,
                species_map=species_map,
            )
            for term in current_terms
        ),
        Expression.num(0),
    )

    spectator_slot_2 = COLOR_FUND_INDEX.representation.g(c3, c4).to_expression()
    spectator_slot_1 = COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
    expected = (
        I
        * gS
        * (
            gauge_generator(a3, c1, c2) * spectator_slot_2
            + gauge_generator(a3, c3, c4) * spectator_slot_1
        )
        * (pcomp(p2, current_index) - pcomp(p1, current_index))
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3)
    )
    assert got.expand().to_canonical_string() == expected.expand().to_canonical_string()


def test_slot_policy_sum_contact_matches_expected_bislot_tensor_structure():
    d = S("d")
    p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
    b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")
    c1, c2, c3, c4 = S("c1", "c2", "c3", "c4")
    mu3, mu4 = S("mu3", "mu4")
    a3, a4 = S("a3", "a4")
    gS = S("gS")
    k1, k2 = S("k1", "k2")

    model, scalar, gluon, _ = _make_bislot_sum_model()
    compiled = compile_covariant_terms(model)
    contact_terms = [term for term in compiled if "contact" in term.label]
    assert len(contact_terms) == 4

    legs = (
        scalar.leg(p1, conjugated=True, species=b1, labels={COLOR_FUND_KIND: (c1, c3)}),
        scalar.leg(p2, species=b2, labels={COLOR_FUND_KIND: (c2, c4)}),
        gluon.leg(p3, species=b3, labels={LORENTZ_KIND: mu3, COLOR_ADJ_KIND: a3}),
        gluon.leg(p4, species=b4, labels={LORENTZ_KIND: mu4, COLOR_ADJ_KIND: a4}),
    )
    species_map = {b1: scalar.conjugate_symbol, b2: scalar.symbol, b3: gluon.symbol, b4: gluon.symbol}
    got = sum(
        (
            _model_vertex(
                interaction=term,
                external_legs=legs,
                species_map=species_map,
            )
            for term in contact_terms
        ),
        Expression.num(0),
    )

    spectator_slot_2 = COLOR_FUND_INDEX.representation.g(c3, c4).to_expression()
    spectator_slot_1 = COLOR_FUND_INDEX.representation.g(c1, c2).to_expression()
    expected = (
        I
        * (gS ** 2)
        * scalar_gauge_contact(mu3, mu4)
        * (
            _symmetrized_generator_contact(a3, a4, c1, c2, k1) * spectator_slot_2
            + _symmetrized_generator_contact(a3, a4, c3, c4, k2) * spectator_slot_1
            + 2 * gauge_generator(a3, c1, c2) * gauge_generator(a4, c3, c4)
            + 2 * gauge_generator(a3, c3, c4) * gauge_generator(a4, c1, c2)
        )
        * (2 * pi) ** d
        * Delta(p1 + p2 + p3 + p4)
    )

    got_canon, _, _ = canonize_spenso_tensors(
        got.expand(),
        lorentz_indices=(mu3, mu4),
        adjoint_indices=(a3, a4),
        color_fund_indices=(
            c1,
            c2,
            c3,
            c4,
            S("c_mid_PhiBiTest_SU3_slot1"),
            S("c_mid_PhiBiTest_SU3_slot2"),
        ),
    )
    expected_canon, _, _ = canonize_spenso_tensors(
        expected.expand(),
        lorentz_indices=(mu3, mu4),
        adjoint_indices=(a3, a4),
        color_fund_indices=(c1, c2, c3, c4, k1, k2),
    )
    assert got_canon.to_canonical_string() == expected_canon.to_canonical_string()
