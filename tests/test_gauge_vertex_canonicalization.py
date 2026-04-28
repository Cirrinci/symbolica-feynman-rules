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
from compiler.gauge import compile_covariant_terms
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

    expected = I * gauge_kinetic_bilinear(S("mu1"), S("mu2"), q1, q2, S("mu_int_1")) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_bilinear():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GhostGluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)

    expected = -I * ghost_kinetic(S("a1"), S("a2"), q1, q2, S("mu_int_1")) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_gluon_vertex():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GluonField, GhostGluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    expected = -gS * ghost_gauge(S("a1"), S("a2"), S("a3"), S("mu2"), q1) * D3

    assert _canon(contracted) == _canon(expected)


def test_contract_then_canonize_matches_compact_yang_mills_cubic():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    canon_got, _, _ = canonize_spenso_tensors(
        contracted,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3")),
        adjoint_indices=(S("a1"), S("a2"), S("a3")),
    )

    expected = gS * yang_mills_three_vertex_raw(
        S("a1"), S("a2"), S("a3"),
        S("mu1"), S("mu2"), S("mu3"),
        q1, q2, q3,
    ) * D3
    canon_expected, _, _ = canonize_spenso_tensors(
        expected,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3")),
        adjoint_indices=(S("a1"), S("a2"), S("a3")),
    )

    assert _canon(canon_got) == _canon(canon_expected)


def test_canonized_yang_mills_quartic_matches_grouped_color_channels():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField, GluonField,
    )
    canon_got, _, _ = canonize_spenso_tensors(
        vertex,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3"), S("mu4")),
        adjoint_indices=(S("a1"), S("a2"), S("a3"), S("a4"), S("color_adj_mid_G_SU3C")),
    )

    compact = (
        -I
        * Expression.num(1)
        / Expression.num(2)
        * (gS ** 2)
        * yang_mills_four_vertex_raw(
            S("a1"), S("a2"), S("a3"), S("a4"),
            S("mu1"), S("mu2"), S("mu3"), S("mu4"),
            S("color_adj_mid_G_SU3C"),
        )
        * D4
    )
    canon_compact, _, _ = canonize_spenso_tensors(
        compact,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3"), S("mu4")),
        adjoint_indices=(S("a1"), S("a2"), S("a3"), S("a4"), S("color_adj_mid_G_SU3C")),
    )

    assert _canon(canon_got) == _canon(canon_compact)
