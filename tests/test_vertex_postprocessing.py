from symbolica.community.spenso import Representation  # noqa: E402
from symbolica import S  # noqa: E402

from model import COLOR_ADJ_INDEX, Field, Gamma, IndexType, Model  # noqa: E402
from model.interactions import ExternalLeg, _auto_leg_labels  # noqa: E402
from lagrangian.operators import yang_mills_three_vertex_raw  # noqa: E402
from tests.support.builders import (  # noqa: E402
    canon as _canon,
    gauge_kinetic_decl,
    make_dirac_fermion as _make_dirac_fermion,
    make_gluon,
    make_su3,
)
from symbolic.tensor_canonicalization import (  # noqa: E402
    canonize_spenso_tensors,
    contract_spenso_lorentz_metrics,
)
from symbolic.vertex_engine import (  # noqa: E402
    Delta,
    Dot,
    I,
    U,
    UF,
    UbarF,
    contract_to_full_expression,
    pi,
    plane_wave,
    simplify_vertex,
    vertex_factor,
)
from symbolic.vertex_postprocessing import (  # noqa: E402
    apply_vertex_output_policy,
    canonicalize_vector_vertex,
    simplify_deltas,
    simplify_spinor_indices,
)
from symbolic.spenso_structures import gamma_anticommutator, simplify_gamma_chain  # noqa: E402


gS = S("gS")
mu, nu = S("mu", "nu")
GluonField = make_gluon(name="G", symbol=S("G0"))
QCD_GROUP = make_su3(gS, GluonField.symbol, name="SU3C")
MODEL_QCD_GAUGE_COVARIANT = Model(
    gauge_groups=(QCD_GROUP,),
    fields=(GluonField,),
    lagrangian_decl=gauge_kinetic_decl(QCD_GROUP, mu=mu, nu=nu),
)


def test_vertex_factor_output_policy_matches_manual_postprocessing_for_quartic_scalar():
    lam = S("lam_post")
    phi = Field("phi_post", spin=0, self_conjugate=True, symbol=S("phi_post"))
    lagrangian = Model(lam * phi * phi * phi * phi).lagrangian()
    term = lagrangian.terms[0]

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    legs = tuple(
        ExternalLeg(field=phi, momentum=q)
        for q in (q1, q2, q3, q4)
    )

    got = vertex_factor(
        interaction=term,
        external_legs=legs,
        x=S("x_"),
        d=S("d"),
        strip_externals=True,
        include_delta=True,
    )

    kwargs = term.to_vertex_kwargs(legs)
    contracted = contract_to_full_expression(x=S("x_"), **kwargs)
    manual = I * apply_vertex_output_policy(
        contracted,
        ps=kwargs["ps"],
        x=S("x_"),
        include_delta=True,
        strip_externals=True,
        leg_index_labels=kwargs["leg_index_labels"],
        d=S("d"),
        plane_wave=plane_wave,
        delta_symbol=Delta,
        pi_symbol=pi,
        u_symbol=U,
        uf_symbol=UF,
        ubarf_symbol=UbarF,
        dot_symbol=Dot,
        i_symbol=I,
    )

    assert _canon(got) == _canon(manual)


def test_simplify_vertex_matches_explicit_chain_for_four_gluon_vertex():
    lagrangian = MODEL_QCD_GAUGE_COVARIANT.lagrangian()
    raw = lagrangian.feynman_rule(
        GluonField,
        GluonField,
        GluonField,
        GluonField,
        simplify=False,
    )

    q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
    counter = [1]
    legs = tuple(
        ExternalLeg(
            field=GluonField,
            momentum=q,
            labels=_auto_leg_labels(GluonField, counter),
        )
        for q in (q1, q2, q3, q4)
    )

    got = simplify_vertex(raw, external_legs=legs)
    manual = canonicalize_vector_vertex(
        contract_spenso_lorentz_metrics(
            simplify_spinor_indices(
                simplify_deltas(raw)
            )
        ),
        external_legs=legs,
    )

    assert _canon(got) == _canon(manual)


def test_canonicalize_vector_vertex_accepts_custom_lorentz_kind():
    lorentz = IndexType("Lorentz", Representation.mink(4), "lorentzz", prefix="mu")
    custom_gluon = Field(
        "Gcustom",
        spin=1,
        self_conjugate=True,
        symbol=S("Gcustom"),
        indices=(lorentz, COLOR_ADJ_INDEX),
    )
    mu1, mu2, mu3 = S("mu1", "mu2", "mu3")
    a1, a2, a3 = S("a1", "a2", "a3")
    q1, q2, q3 = S("q1", "q2", "q3")
    expr = yang_mills_three_vertex_raw(a2, a1, a3, mu2, mu1, mu3, q2, q1, q3)
    legs = (
        ExternalLeg(
            field=custom_gluon,
            momentum=q1,
            labels={lorentz.kind: mu1, COLOR_ADJ_INDEX.kind: a1},
        ),
        ExternalLeg(
            field=custom_gluon,
            momentum=q2,
            labels={lorentz.kind: mu2, COLOR_ADJ_INDEX.kind: a2},
        ),
        ExternalLeg(
            field=custom_gluon,
            momentum=q3,
            labels={lorentz.kind: mu3, COLOR_ADJ_INDEX.kind: a3},
        ),
    )

    got = canonicalize_vector_vertex(expr, external_legs=legs)
    manual, _, _ = canonize_spenso_tensors(
        expr,
        lorentz_indices=(mu1, mu2, mu3),
        adjoint_indices=(a1, a2, a3),
    )

    assert _canon(got) == _canon(manual)


def test_simplify_vertex_default_behavior_is_unchanged():
    expr = gamma_anticommutator(S("i_left"), S("i_right"), S("mu"), S("nu"))

    got = simplify_vertex(expr)
    explicit_default = simplify_vertex(expr, simplify_gamma=False)
    manual = contract_spenso_lorentz_metrics(
        simplify_spinor_indices(
            simplify_deltas(expr)
        )
    )

    assert _canon(got) == _canon(explicit_default)
    assert _canon(got) == _canon(manual)


def test_simplify_vertex_with_simplify_gamma_applies_gamma_chain():
    expr = gamma_anticommutator(S("i_left"), S("i_right"), S("mu"), S("nu"))

    got = simplify_vertex(expr, simplify_gamma=True)
    default = simplify_vertex(expr, simplify_gamma=False)
    manual = contract_spenso_lorentz_metrics(
        simplify_spinor_indices(
            simplify_gamma_chain(
                simplify_deltas(expr)
            )
        )
    )

    assert _canon(got) == _canon(manual)
    assert _canon(got) != _canon(default)


def test_lagrangian_feynman_rule_default_simplify_is_unchanged():
    psi = _make_dirac_fermion("Psi")
    chi = _make_dirac_fermion("Chi")
    gV = S("gV_post")
    mu = S("mu_post")
    L = Model(gV * psi.bar * Gamma(mu) * psi * chi.bar * Gamma(mu) * chi).lagrangian()
    term = L.terms[0]

    q1, q2, q3, q4 = S("q1"), S("q2"), S("q3"), S("q4")
    counter = [1]
    legs = tuple(
        ExternalLeg(
            field=field,
            momentum=q,
            conjugated=conjugated,
            labels=_auto_leg_labels(field, counter),
        )
        for (field, conjugated), q in zip(
            ((psi, True), (psi, False), (chi, True), (chi, False)),
            (q1, q2, q3, q4),
        )
    )

    got = L.feynman_rule(psi.bar, psi, chi.bar, chi, simplify=True, include_delta=True)
    species_map = {
        species: species
        for species in (
            psi.species_for(True),
            psi.species_for(False),
            chi.species_for(True),
            chi.species_for(False),
        )
    }
    manual = simplify_vertex(
        vertex_factor(
            interaction=term,
            external_legs=legs,
            x=S("x_"),
            d=S("d"),
            strip_externals=True,
            include_delta=True,
        ),
        species_map=species_map,
        external_legs=legs,
        simplify_gamma=False,
    )

    assert _canon(got) == _canon(manual)
