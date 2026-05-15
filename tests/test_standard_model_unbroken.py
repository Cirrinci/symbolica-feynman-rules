from __future__ import annotations

from model import build_unbroken_standard_model


def _canon(expr):
    return expr.expand().to_canonical_string()


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


def test_unbroken_standard_model_representative_vertices_compile():
    sm = build_unbroken_standard_model()
    lagrangian = sm.model.lagrangian()

    qcd_rule = lagrangian.feynman_rule(
        sm.fields.qL.bar,
        sm.fields.qL,
        sm.fields.G,
        simplify=True,
        include_delta=True,
    )
    mixed_higgs_rule = lagrangian.feynman_rule(
        sm.fields.Phi.bar,
        sm.fields.Phi,
        sm.fields.Wi,
        sm.fields.B,
        simplify=True,
        include_delta=True,
    )
    up_yukawa_rule = lagrangian.feynman_rule(
        sm.fields.qL.bar,
        sm.fields.Phi.bar,
        sm.fields.uR,
        simplify=True,
        include_delta=True,
    )

    qcd_text = _canon(qcd_rule)
    mixed_higgs_text = _canon(mixed_higgs_rule)
    up_yukawa_text = _canon(up_yukawa_rule)

    assert qcd_text != "0"
    assert "g3" in qcd_text

    assert mixed_higgs_text != "0"
    assert "g1" in mixed_higgs_text
    assert "g2" in mixed_higgs_text

    assert up_yukawa_text != "0"
    assert "Yu(" in up_yukawa_text
    assert "weak_eps2" in up_yukawa_text
