from symbolic.vertex_engine import Delta, Expression, I, S, pi

from examples.examples import (
    d,
    gS,
    GaugeField,
    GhostGluonField,
    GluonField,
    MODEL_QCD_GAUGE_COVARIANT,
    MODEL_QCD_GHOST_COVARIANT,
    MODEL_QED_GAUGE_COVARIANT,
)
from compiler.covariant import compile_covariant_terms
from lagrangian.operators import (
    gauge_kinetic_bilinear,
    ghost_gauge,
    ghost_kinetic,
    yang_mills_four_vertex_raw,
    yang_mills_three_vertex_raw,
)
from symbolic.tensor_canonicalization import canonize_spenso_tensors, contract_spenso_lorentz_metrics


q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_contract_spenso_lorentz_metrics_simplifies_qed_gauge_bilinear():
    vertex = MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(GaugeField, GaugeField)
    contracted = contract_spenso_lorentz_metrics(vertex)

    compiled = compile_covariant_terms(MODEL_QED_GAUGE_COVARIANT)
    rho = compiled[0].derivatives[0].lorentz_index
    expected = I * gauge_kinetic_bilinear(S("i1"), S("i2"), q1, q2, rho) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_bilinear():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GhostGluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)

    compiled = compile_covariant_terms(MODEL_QCD_GHOST_COVARIANT)
    rho = compiled[0].derivatives[0].lorentz_index
    expected = -I * ghost_kinetic(S("i1"), S("i2"), q1, q2, rho) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_gluon_vertex():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GluonField, GhostGluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    expected = -gS * ghost_gauge(S("i1"), S("i3"), S("i4"), S("i2"), q1) * D3

    assert _canon(contracted) == _canon(expected)


def test_contract_then_canonize_matches_compact_yang_mills_cubic():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    canon_got, _, _ = canonize_spenso_tensors(
        contracted,
        lorentz_indices=(S("i1"), S("i3"), S("i5")),
        adjoint_indices=(S("i2"), S("i4"), S("i6")),
    )

    expected = gS * yang_mills_three_vertex_raw(
        S("i2"), S("i4"), S("i6"),
        S("i1"), S("i3"), S("i5"),
        q1, q2, q3,
    ) * D3
    canon_expected, _, _ = canonize_spenso_tensors(
        expected,
        lorentz_indices=(S("i1"), S("i3"), S("i5")),
        adjoint_indices=(S("i2"), S("i4"), S("i6")),
    )

    assert _canon(canon_got) == _canon(canon_expected)


def test_canonized_yang_mills_quartic_matches_grouped_color_channels():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField, GluonField,
    )
    canon_got, _, _ = canonize_spenso_tensors(
        vertex,
        lorentz_indices=(S("i1"), S("i3"), S("i5"), S("i7")),
        adjoint_indices=(S("i2"), S("i4"), S("i6"), S("i8"), S("color_adj_mid_G_SU3C")),
    )

    compact = (
        -I
        * Expression.num(1)
        / Expression.num(2)
        * (gS ** 2)
        * yang_mills_four_vertex_raw(
            S("i2"), S("i4"), S("i6"), S("i8"),
            S("i1"), S("i3"), S("i5"), S("i7"),
            S("color_adj_mid_G_SU3C"),
        )
        * D4
    )
    canon_compact, _, _ = canonize_spenso_tensors(
        compact,
        lorentz_indices=(S("i1"), S("i3"), S("i5"), S("i7")),
        adjoint_indices=(S("i2"), S("i4"), S("i6"), S("i8"), S("color_adj_mid_G_SU3C")),
    )

    assert _canon(canon_got) == _canon(canon_compact)
