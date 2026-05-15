from __future__ import annotations

import re
from collections import Counter

from model import build_unbroken_standard_model
from model.interactions import _field_match_key


def _canon(expr):
    return expr.expand().to_canonical_string()


def _signature_counter(*field_args):
    return Counter(_field_match_key(field, conjugated) for field, conjugated in field_args)


def _matching_term(sm, *field_args):
    target = _signature_counter(*field_args)
    for term in sm.model.lagrangian().terms:
        counter = Counter(_field_match_key(occ.field, occ.conjugated) for occ in term.fields)
        if counter == target:
            return term
    raise AssertionError(f"Missing compiled term for signature {field_args!r}.")


def _assert_metric_count(text: str, representation: str, expected: int):
    assert text.count(representation) == expected


def test_unbroken_standard_model_yukawa_parameters_are_complex_generation_matrices():
    sm = build_unbroken_standard_model()
    Generation = sm.indices.generation

    for name in ("Yu", "YuDag", "Yd", "YdDag", "Ye", "YeDag"):
        parameter = getattr(sm.parameters, name)
        assert parameter.indices == (Generation, Generation)
        assert parameter.complex_param is True


def test_all_standard_model_fermions_carry_the_generation_index():
    sm = build_unbroken_standard_model()
    Generation = sm.indices.generation

    for field in (sm.fields.qL, sm.fields.uR, sm.fields.dR, sm.fields.lL, sm.fields.eR):
        assert Generation in field.indices
        assert field.flavor_index is Generation


def test_generation_flavor_classes_are_declared_for_flavor_expansion():
    sm = build_unbroken_standard_model()

    assert tuple(member.name for member in sm.fields.qL.class_members) == ("qL1", "qL2", "qL3")
    assert tuple(member.name for member in sm.fields.uR.class_members) == ("u", "c", "t")
    assert tuple(member.name for member in sm.fields.dR.class_members) == ("d", "s", "b")
    assert tuple(member.name for member in sm.fields.lL.class_members) == ("lL1", "lL2", "lL3")
    assert tuple(member.name for member in sm.fields.eR.class_members) == ("e", "mu", "ta")


def test_yukawa_forward_and_hc_terms_use_the_expected_parameter_families():
    sm = build_unbroken_standard_model()

    down = _canon(_matching_term(sm, (sm.fields.qL, True), (sm.fields.dR, False), (sm.fields.Phi, False)).coupling)
    lepton = _canon(_matching_term(sm, (sm.fields.lL, True), (sm.fields.eR, False), (sm.fields.Phi, False)).coupling)
    up = _canon(_matching_term(sm, (sm.fields.qL, True), (sm.fields.Phi, True), (sm.fields.uR, False)).coupling)
    down_hc = _canon(_matching_term(sm, (sm.fields.dR, True), (sm.fields.Phi, True), (sm.fields.qL, False)).coupling)
    lepton_hc = _canon(_matching_term(sm, (sm.fields.eR, True), (sm.fields.Phi, True), (sm.fields.lL, False)).coupling)
    up_hc = _canon(_matching_term(sm, (sm.fields.uR, True), (sm.fields.Phi, False), (sm.fields.qL, False)).coupling)

    assert "Yd(" in down and "YdDag(" not in down
    assert "Ye(" in lepton and "YeDag(" not in lepton
    assert "Yu(" in up and "YuDag(" not in up

    assert "YdDag(" in down_hc and "Yd(" not in down_hc
    assert "YeDag(" in lepton_hc and "Ye(" not in lepton_hc
    assert "YuDag(" in up_hc and "Yu(" not in up_hc


def test_hc_yukawa_terms_use_reversed_flavor_indices():
    sm = build_unbroken_standard_model()

    down_hc = _canon(_matching_term(sm, (sm.fields.dR, True), (sm.fields.Phi, True), (sm.fields.qL, False)).coupling)
    lepton_hc = _canon(_matching_term(sm, (sm.fields.eR, True), (sm.fields.Phi, True), (sm.fields.lL, False)).coupling)
    up_hc = _canon(_matching_term(sm, (sm.fields.uR, True), (sm.fields.Phi, False), (sm.fields.qL, False)).coupling)

    assert re.search(r"YdDag\([^,]*ff2,[^)]*ff1\)", down_hc)
    assert re.search(r"YeDag\([^,]*ff2,[^)]*ff1\)", lepton_hc)
    assert re.search(r"YuDag\([^,]*ff2,[^)]*ff1\)", up_hc)


def test_up_yukawa_terms_keep_typed_weak_eps2_with_the_correct_slot_orientation():
    sm = build_unbroken_standard_model()

    up = _matching_term(sm, (sm.fields.qL, True), (sm.fields.Phi, True), (sm.fields.uR, False))
    up_hc = _matching_term(sm, (sm.fields.uR, True), (sm.fields.Phi, False), (sm.fields.qL, False))

    assert "weak_eps2" in _canon(up.coupling)
    assert "weak_eps2" in _canon(up_hc.coupling)

    assert str(up.fields[0].labels["weak_fund"]) == "ii"
    assert str(up.fields[1].labels["weak_fund"]) == "jj"
    assert str(up_hc.fields[2].labels["weak_fund"]) == "ii"
    assert str(up_hc.fields[1].labels["weak_fund"]) == "jj"


def test_yukawa_source_terms_have_the_requested_explicit_index_contractions():
    sm = build_unbroken_standard_model()

    down = _matching_term(sm, (sm.fields.qL, True), (sm.fields.dR, False), (sm.fields.Phi, False))
    lepton = _matching_term(sm, (sm.fields.lL, True), (sm.fields.eR, False), (sm.fields.Phi, False))
    up = _matching_term(sm, (sm.fields.qL, True), (sm.fields.Phi, True), (sm.fields.uR, False))
    up_hc = _matching_term(sm, (sm.fields.uR, True), (sm.fields.Phi, False), (sm.fields.qL, False))

    assert down.fields[0].labels["spinor"] == down.fields[1].labels["spinor"]
    assert down.fields[0].labels["color_fund"] == down.fields[1].labels["color_fund"]
    assert down.fields[0].labels["weak_fund"] == down.fields[2].labels["weak_fund"]

    assert lepton.fields[0].labels["spinor"] == lepton.fields[1].labels["spinor"]
    assert lepton.fields[0].labels["weak_fund"] == lepton.fields[2].labels["weak_fund"]

    assert up.fields[0].labels["spinor"] == up.fields[2].labels["spinor"]
    assert up.fields[0].labels["color_fund"] == up.fields[2].labels["color_fund"]
    assert up.fields[0].labels["weak_fund"] != up.fields[1].labels["weak_fund"]

    assert up_hc.fields[0].labels["spinor"] == up_hc.fields[2].labels["spinor"]
    assert up_hc.fields[0].labels["color_fund"] == up_hc.fields[2].labels["color_fund"]
    assert str(up_hc.fields[2].labels["weak_fund"]) == "ii"
    assert str(up_hc.fields[1].labels["weak_fund"]) == "jj"


def test_unbroken_standard_model_builds_and_validates():
    sm = build_unbroken_standard_model()

    assert sm.model.name == "SM-unbroken-non-BFM"
    assert len(sm.lagrangians.LGauge.source_terms) == 3
    assert len(sm.lagrangians.LFermions.source_terms) == 5
    assert len(sm.lagrangians.LHiggs.source_terms) == 3
    assert len(sm.lagrangians.LYukawa.source_terms) == 6
    assert len(sm.lagrangians.LSM.source_terms) == 17

    report = sm.model.validate()

    assert report.ok
    assert report.issues == ()


def test_flavor_expand_true_now_works_for_unbroken_standard_model():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()
    Generation = sm.indices.generation
    qL1, qL2, qL3 = sm.fields.qL.class_members
    d, s, b = sm.fields.dR.class_members
    u, c, t = sm.fields.uR.class_members
    lL1, lL2, lL3 = sm.fields.lL.class_members
    e, mu, ta = sm.fields.eR.class_members

    expanded_signatures = {signature.names for signature in lagrangian.vertex_signatures(flavor_expand=Generation)}

    assert ("qL1.bar", "d", "Phi") in expanded_signatures
    assert ("qL2.bar", "s", "Phi") in expanded_signatures
    assert ("qL3.bar", "b", "Phi") in expanded_signatures
    assert ("lL1.bar", "e", "Phi") in expanded_signatures
    assert ("lL2.bar", "mu", "Phi") in expanded_signatures
    assert ("lL3.bar", "ta", "Phi") in expanded_signatures
    assert ("qL1.bar", "Phi.bar", "u") in expanded_signatures
    assert ("qL2.bar", "Phi.bar", "c") in expanded_signatures
    assert ("qL3.bar", "Phi.bar", "t") in expanded_signatures

    down_rule = lagrangian.feynman_rule(
        qL1.bar,
        d,
        sm.fields.Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    lepton_rule = lagrangian.feynman_rule(
        lL2.bar,
        mu,
        sm.fields.Phi,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )
    up_rule = lagrangian.feynman_rule(
        qL3.bar,
        sm.fields.Phi.bar,
        t,
        simplify=True,
        include_delta=True,
        flavor_expand=True,
    )

    assert "Yd(1,1)" in _canon(down_rule)
    assert "Ye(2,2)" in _canon(lepton_rule)
    assert "Yu(3,3)" in _canon(up_rule)


def test_unbroken_standard_model_selected_vertex_signatures_are_present():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    pure_gauge_signatures = {signature.names for signature in lagrangian.vertex_signatures(sector="pure_gauge")}
    assert ("G", "G", "G") in pure_gauge_signatures
    assert ("G", "G", "G", "G") in pure_gauge_signatures
    assert ("Wi", "Wi", "Wi") in pure_gauge_signatures
    assert ("Wi", "Wi", "Wi", "Wi") in pure_gauge_signatures

    qcd_current = lagrangian.vertex_signatures(
        contains_fields=(sm.fields.qL.bar, sm.fields.qL, sm.fields.G),
    )
    assert len(qcd_current) == 1
    assert set(qcd_current[0].names) == {"qL.bar", "qL", "G"}

    mixed_higgs_contact = lagrangian.vertex_signatures(
        contains_fields=(sm.fields.Phi.bar, sm.fields.Phi, sm.fields.Wi, sm.fields.B),
    )
    assert len(mixed_higgs_contact) == 1
    assert set(mixed_higgs_contact[0].names) == {"Phi.bar", "Phi", "Wi", "B"}

    up_yukawa = lagrangian.vertex_signatures(
        contains_fields=(sm.fields.qL.bar, sm.fields.Phi.bar, sm.fields.uR),
    )
    assert len(up_yukawa) == 1
    assert set(up_yukawa[0].names) == {"qL.bar", "Phi.bar", "uR"}


def test_down_vertex_contains_spin_colour_weak_and_yd():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    text = _canon(
        lagrangian.feynman_rule(
            sm.fields.qL.bar,
            sm.fields.dR,
            sm.fields.Phi,
            simplify=True,
            include_delta=True,
        )
    )

    assert "Yd(" in text
    _assert_metric_count(text, "bis(4", 2)
    _assert_metric_count(text, "cof(2", 2)
    _assert_metric_count(text, "cof(3", 2)


def test_lepton_vertex_contains_spin_weak_and_ye():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    text = _canon(
        lagrangian.feynman_rule(
            sm.fields.lL.bar,
            sm.fields.eR,
            sm.fields.Phi,
            simplify=True,
            include_delta=True,
        )
    )

    assert "Ye(" in text
    _assert_metric_count(text, "bis(4", 2)
    _assert_metric_count(text, "cof(2", 2)
    assert "cof(3" not in text


def test_up_vertex_contains_spin_colour_typed_weak_epsilon_and_yu():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    text = _canon(
        lagrangian.feynman_rule(
            sm.fields.qL.bar,
            sm.fields.Phi.bar,
            sm.fields.uR,
            simplify=True,
            include_delta=True,
        )
    )

    assert "Yu(" in text
    assert "weak_eps2" in text
    _assert_metric_count(text, "bis(4", 2)
    _assert_metric_count(text, "cof(3", 2)


def test_hc_yukawa_vertices_use_conjugate_parameters_with_reversed_vertex_flavor_order():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    down_text = _canon(
        lagrangian.feynman_rule(
            sm.fields.dR.bar,
            sm.fields.Phi.bar,
            sm.fields.qL,
            simplify=True,
            include_delta=True,
        )
    )
    lepton_text = _canon(
        lagrangian.feynman_rule(
            sm.fields.eR.bar,
            sm.fields.Phi.bar,
            sm.fields.lL,
            simplify=True,
            include_delta=True,
        )
    )
    up_text = _canon(
        lagrangian.feynman_rule(
            sm.fields.uR.bar,
            sm.fields.Phi,
            sm.fields.qL,
            simplify=True,
            include_delta=True,
        )
    )

    assert "YdDag(" in down_text
    assert re.search(r"YdDag\([^,]*fl1,[^)]*fl3\)", down_text)
    assert "YeDag(" in lepton_text
    assert re.search(r"YeDag\([^,]*fl1,[^)]*fl3\)", lepton_text)
    assert "YuDag(" in up_text
    assert re.search(r"YuDag\([^,]*fl1,[^)]*fl3\)", up_text)
    assert "weak_eps2" in up_text
