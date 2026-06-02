from symbolic.vertex_engine import Delta, Expression, I, S, pi

from model import GhostLagrangian, Model
from tests.support.builders import (
    canon as _canon,
    gauge_kinetic_decl,
    make_ghost,
    make_gluon,
    make_photon,
    make_su3,
    make_u1,
)
from lagrangian.operators import (
    gauge_kinetic_bilinear,
    ghost_gauge,
    ghost_kinetic,
    yang_mills_four_vertex_raw,
    yang_mills_three_vertex_raw,
)
from symbolic.spenso_structures import lorentz_metric, structure_constant
from symbolic.tensor_canonicalization import (
    canonize_spenso_tensors,
    canonize_structure_constant_products,
    contract_spenso_lorentz_metrics,
)


q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
d, gS = S("d", "gS")
eQED = S("eQED")
mu, nu = S("mu", "nu")

GaugeField = make_photon(name="A", symbol=S("A0"))
GluonField = make_gluon(name="G", symbol=S("G0"))
GhostGluonField = make_ghost(name="ghG", symbol=S("ghG0"), conjugate_symbol=S("ghGbar0"))

QED_GROUP = make_u1(eQED, GaugeField.symbol, name="U1QED")
QCD_GROUP = make_su3(gS, GluonField.symbol, ghost_sym=GhostGluonField.symbol, name="SU3C")

MODEL_QED_GAUGE_COVARIANT = Model(
    gauge_groups=(QED_GROUP,),
    fields=(GaugeField,),
    lagrangian_decl=gauge_kinetic_decl(QED_GROUP, mu=mu, nu=nu),
)
MODEL_QCD_GAUGE_COVARIANT = Model(
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    lagrangian_decl=gauge_kinetic_decl(QCD_GROUP, mu=mu, nu=nu),
)
MODEL_QCD_GHOST_COVARIANT = Model(
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField, GhostGluonField),
    lagrangian_decl=GhostLagrangian(QCD_GROUP),
)

D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)


def test_contract_spenso_lorentz_metrics_simplifies_qed_gauge_bilinear():
    vertex = MODEL_QED_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GaugeField, GaugeField, include_delta=True,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)

    expected = I * gauge_kinetic_bilinear(S("mu1"), S("mu2"), q1, q2, S("mu1_int")) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_bilinear():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GhostGluonField, include_delta=True,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)

    expected = -I * ghost_kinetic(S("a1"), S("a2"), q1, q2, S("mu1_int")) * D2

    assert _canon(contracted) == _canon(expected)


def test_contract_spenso_lorentz_metrics_simplifies_ghost_gluon_vertex():
    vertex = MODEL_QCD_GHOST_COVARIANT.lagrangian().feynman_rule(
        GhostGluonField.bar, GluonField, GhostGluonField, include_delta=True,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    expected = gS * ghost_gauge(S("a1"), S("a2"), S("a3"), S("mu2"), q1) * D3

    assert _canon(contracted) == _canon(expected)


def test_contract_then_canonize_matches_compact_yang_mills_cubic():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField, include_delta=True,
    )
    contracted = contract_spenso_lorentz_metrics(vertex)
    canon_got, _, _ = canonize_spenso_tensors(
        contracted,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3")),
        adjoint_indices=(S("a1"), S("a2"), S("a3")),
    )

    expected = -gS * yang_mills_three_vertex_raw(
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
        GluonField, GluonField, GluonField, GluonField, include_delta=True,
    )
    canon_got, _, _ = canonize_spenso_tensors(
        vertex,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3"), S("mu4")),
        adjoint_indices=(S("a1"), S("a2"), S("a3"), S("a4"), S("a_decl")),
    )

    compact = (
        -I
        * Expression.num(1)
        / Expression.num(2)
        * (gS ** 2)
        * yang_mills_four_vertex_raw(
            S("a1"), S("a2"), S("a3"), S("a4"),
            S("mu1"), S("mu2"), S("mu3"), S("mu4"),
            S("a_decl"),
        )
        * D4
    )
    canon_compact, _, _ = canonize_spenso_tensors(
        compact,
        lorentz_indices=(S("mu1"), S("mu2"), S("mu3"), S("mu4")),
        adjoint_indices=(S("a1"), S("a2"), S("a3"), S("a4"), S("a_decl")),
    )

    assert _canon(canon_got) == _canon(canon_compact)


def test_quartic_gluon_vertex_matches_feynrules_after_structure_constant_canonicalization():
    vertex = MODEL_QCD_GAUGE_COVARIANT.lagrangian().feynman_rule(
        GluonField, GluonField, GluonField, GluonField, include_delta=True,
    )

    a1, a2, a3, a4, b = S("a1", "a2", "a3", "a4", "b")
    mu1, mu2, mu3, mu4 = S("mu1", "mu2", "mu3", "mu4")
    feynrules = (
        I
        * (gS ** 2)
        * structure_constant(a1, a3, b)
        * structure_constant(a2, a4, b)
        * lorentz_metric(mu1, mu4)
        * lorentz_metric(mu2, mu3)
        + I
        * (gS ** 2)
        * structure_constant(a1, a2, b)
        * structure_constant(a3, a4, b)
        * lorentz_metric(mu1, mu4)
        * lorentz_metric(mu2, mu3)
        + I
        * (gS ** 2)
        * structure_constant(a1, a4, b)
        * structure_constant(a2, a3, b)
        * lorentz_metric(mu1, mu3)
        * lorentz_metric(mu2, mu4)
        - I
        * (gS ** 2)
        * structure_constant(a1, a2, b)
        * structure_constant(a3, a4, b)
        * lorentz_metric(mu1, mu3)
        * lorentz_metric(mu2, mu4)
        - I
        * (gS ** 2)
        * structure_constant(a1, a4, b)
        * structure_constant(a2, a3, b)
        * lorentz_metric(mu1, mu2)
        * lorentz_metric(mu3, mu4)
        - I
        * (gS ** 2)
        * structure_constant(a1, a3, b)
        * structure_constant(a2, a4, b)
        * lorentz_metric(mu1, mu2)
        * lorentz_metric(mu3, mu4)
    ) * D4

    canon_vertex = canonize_structure_constant_products(
        vertex,
        lorentz_indices=(mu1, mu2, mu3, mu4),
        adjoint_indices=(a1, a2, a3, a4, b, S("a_decl")),
    )
    canon_feynrules = canonize_structure_constant_products(
        feynrules,
        lorentz_indices=(mu1, mu2, mu3, mu4),
        adjoint_indices=(a1, a2, a3, a4, b, S("a_decl")),
    )

    assert _canon(canon_vertex) == _canon(canon_feynrules)
