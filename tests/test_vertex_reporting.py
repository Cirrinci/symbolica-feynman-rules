from symbolica import Expression, S

from model import (
    CovD,
    DerivativeAction,
    PartialD,
    Field,
    FieldStrength,
    Gamma,
    GaugeFixing,
    GhostLagrangian,
    InteractionTerm,
    Lagrangian,
    Model,
)
from model.lagrangian import KNOWN_VERTEX_SECTORS
from symbolic.vertex_engine import I
from tests.support.builders import (
    dirac_covd_decl,
    gauge_kinetic_decl,
    make_complex_scalar,
    make_dirac_fermion,
    make_ghost,
    make_gluon,
    make_photon,
    make_su3,
    make_u1,
)


def test_vertex_report_enumerates_scalar_signatures_deterministically():
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
        InteractionTerm(
            coupling=S("lam"),
            fields=tuple(phi.occurrence() for _ in range(4)),
        ),
    ))

    report = lagrangian.vertex_report()

    assert report.total_terms == 3
    assert report.total_signatures == 3
    assert report.matched_signatures == 3
    assert report.matched_terms == 3
    assert [signature.names for signature in report.signatures] == [
        ("Phi", "Phi"),
        ("Phi", "Chi", "Chi"),
        ("Phi", "Phi", "Phi", "Phi"),
    ]
    assert [signature.arity for signature in report.signatures] == [2, 3, 4]
    assert [signature.term_count for signature in report.signatures] == [1, 1, 1]


def test_vertex_signatures_filter_by_arity():
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
    ))

    signatures = lagrangian.vertex_signatures(arity=3)

    assert len(signatures) == 1
    assert signatures[0].names == ("Phi", "Chi", "Chi")
    assert signatures[0].arity == 3


def test_vertex_signatures_filter_by_exact_qed_signature():
    eQED, qPsi, mu = S("eQED", "qPsi", "mu")
    fermion = make_dirac_fermion(
        "PsiQED",
        symbol=S("psiQED"),
        conjugate_symbol=S("psibarQED"),
        charge=qPsi,
    )
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(eQED, photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(fermion, photon),
        lagrangian_decl=dirac_covd_decl(fermion, mu=mu),
    )

    compiled = model.lagrangian()
    report = compiled.vertex_report(signature=(photon, fermion.bar, fermion))

    assert report.total_terms == 2
    assert report.total_signatures == 2
    assert report.matched_signatures == 1
    assert report.matched_terms == 1

    signature = report.signatures[0]
    assert signature.names == ("PsiQED.bar", "PsiQED", "A")
    assert signature.arity == 3
    assert signature.term_count == 1


def test_vertex_signatures_contains_fields_is_multiplicity_aware_for_qcd():
    gS, mu, nu = S("gS", "mu", "nu")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(gS, gluon.symbol, name="SU3C")
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=gauge_kinetic_decl(su3, mu=mu, nu=nu),
    )

    signatures = model.lagrangian().vertex_signatures(
        contains_fields=(gluon, gluon, gluon),
    )

    assert [signature.arity for signature in signatures] == [3, 4]
    assert [signature.names for signature in signatures] == [
        ("G", "G", "G"),
        ("G", "G", "G", "G"),
    ]


def _qcd_with_quark_and_ghost_compiled():
    gS = S("gS")
    mu, nu = S("mu_d"), S("nu_d")
    gluon = make_gluon(name="G", symbol=S("G0"))
    ghost = make_ghost(name="ghG", symbol=S("ghG"), conjugate_symbol=S("ghGbar"))
    quark = make_dirac_fermion(
        "q",
        symbol=S("q0"),
        conjugate_symbol=S("qbar0"),
        color=True,
    )
    su3 = make_su3(gS, gluon.symbol, ghost_sym=ghost.symbol, name="SU3C")
    decl = (
        -(Expression.num(1) / Expression.num(4))
        * FieldStrength(su3, mu, nu)
        * FieldStrength(su3, mu, nu)
        + GaugeFixing(su3, xi=S("xiQCD"))
        + GhostLagrangian(su3)
        + dirac_covd_decl(quark, mu=mu)
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(quark, gluon, ghost),
        lagrangian_decl=decl,
    )
    return model.lagrangian(), {
        "gluon": gluon,
        "ghost": ghost,
        "quark": quark,
    }


def test_vertex_signatures_invalid_sector_is_rejected():
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
    ))

    try:
        lagrangian.vertex_signatures(sector="bogus")
    except ValueError as exc:
        assert "expected one of" in str(exc)
        for known in KNOWN_VERTEX_SECTORS:
            assert known in str(exc)
    else:
        raise AssertionError("expected ValueError for bogus sector tag")


def test_vertex_signatures_qcd_sector_split():
    compiled, _ = _qcd_with_quark_and_ghost_compiled()

    pure_gauge_signatures = compiled.vertex_signatures(sector="pure_gauge")
    pure_gauge_names = sorted(sig.names for sig in pure_gauge_signatures)
    assert pure_gauge_names == [
        ("G", "G"),
        ("G", "G", "G"),
        ("G", "G", "G", "G"),
    ]

    gauge_fixing_signatures = compiled.vertex_signatures(sector="gauge_fixing")
    assert [sig.names for sig in gauge_fixing_signatures] == [("G", "G")]
    assert gauge_fixing_signatures[0].term_count == 1
    assert gauge_fixing_signatures[0].sectors == ("gauge_fixing",)

    ghost_signatures = compiled.vertex_signatures(sector="ghost")
    ghost_names = sorted(sig.names for sig in ghost_signatures)
    assert ghost_names == [("ghG.bar", "G", "ghG"), ("ghG.bar", "ghG")]

    matter_signatures = compiled.vertex_signatures(sector="matter")
    matter_names = sorted(sig.names for sig in matter_signatures)
    assert matter_names == [("q.bar", "q"), ("q.bar", "q", "G")]

    unknown_signatures = compiled.vertex_signatures(sector="unknown")
    assert unknown_signatures == ()


def test_vertex_report_helper_qed_gauge_fixing_sector_includes_two_photon_rule():
    xi = S("xi_qed_report")
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(S("eQED_report"), photon.symbol, name="U1QED")
    model = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=GaugeFixing(u1, xi=xi),
    )

    report = model.lagrangian().vertex_report(sector="gauge_fixing")

    assert [sig.names for sig in report.signatures] == [("A", "A")]
    assert report.matched_terms == 1


def test_vertex_report_manual_qed_gauge_fixing_sector_matches_helper():
    xi = S("xi_qed_report_manual")
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(S("eQED_report_manual"), photon.symbol, name="U1QED")
    mu = S("mu_manual_qed")
    nu = S("nu_manual_qed")

    manual = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=(
            -(Expression.num(1) / (Expression.num(2) * xi))
            * PartialD(photon(mu), mu)
            * PartialD(photon(nu), nu)
        ),
    )
    helper = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=GaugeFixing(u1, xi=xi),
    )

    assert [sig.names for sig in manual.lagrangian().vertex_report(sector="gauge_fixing").signatures] == [
        ("A", "A")
    ]
    assert (
        manual.lagrangian().vertex_report(sector="gauge_fixing").matched_terms
        == helper.lagrangian().vertex_report(sector="gauge_fixing").matched_terms
        == 1
    )


def test_vertex_report_manual_su3_gauge_fixing_sector_matches_helper():
    xi = S("xi_qcd_report_manual")
    gluon = make_gluon(name="G", symbol=S("G0"))
    su3 = make_su3(S("gS_report_manual"), gluon.symbol, name="SU3C")
    mu = S("mu_manual_qcd")
    nu = S("nu_manual_qcd")
    a = S("a_manual_qcd")

    manual = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=(
            -(Expression.num(1) / (Expression.num(2) * xi))
            * PartialD(gluon(mu, a), mu)
            * PartialD(gluon(nu, a), nu)
        ),
    )
    helper = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=GaugeFixing(su3, xi=xi),
    )

    assert [sig.names for sig in manual.lagrangian().vertex_report(sector="gauge_fixing").signatures] == [
        ("G", "G")
    ]
    assert (
        manual.lagrangian().vertex_report(sector="gauge_fixing").matched_terms
        == helper.lagrangian().vertex_report(sector="gauge_fixing").matched_terms
        == 1
    )


def test_vertex_report_non_gauge_fixing_vector_bilinear_is_not_misclassified():
    photon = make_photon(name="A", symbol=S("A0"))
    rho = S("rho_not_gf")
    mu = S("mu_not_gf")
    nu = S("nu_not_gf")
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=Expression.num(1),
            fields=(photon(mu), photon(nu)),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho),
                DerivativeAction(target=1, lorentz_index=rho),
            ),
        ),
    ))

    assert lagrangian.vertex_signatures(sector="gauge_fixing") == ()


def test_vertex_report_manual_gauge_fixing_is_dummy_label_invariant():
    xi = S("xi_qed_dummy")
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(S("eQED_dummy"), photon.symbol, name="U1QED")

    first = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=(
            -(Expression.num(1) / (Expression.num(2) * xi))
            * PartialD(photon(S("alpha_label")), S("alpha_label"))
            * PartialD(photon(S("beta_label")), S("beta_label"))
        ),
    )
    second = Model(
        gauge_groups=(u1,),
        fields=(photon,),
        lagrangian_decl=(
            -(Expression.num(1) / (Expression.num(2) * xi))
            * PartialD(photon(S("mu_anything")), S("mu_anything"))
            * PartialD(photon(S("nu_anything")), S("nu_anything"))
        ),
    )

    assert [sig.names for sig in first.lagrangian().vertex_report(sector="gauge_fixing").signatures] == [
        ("A", "A")
    ]
    assert [sig.names for sig in second.lagrangian().vertex_report(sector="gauge_fixing").signatures] == [
        ("A", "A")
    ]


def test_vertex_signatures_every_signature_is_covered_by_some_sector():
    compiled, _ = _qcd_with_quark_and_ghost_compiled()

    covered = set()
    for sector in KNOWN_VERTEX_SECTORS:
        for sig in compiled.vertex_signatures(sector=sector):
            covered.add(sig.names)

    all_names = {sig.names for sig in compiled.vertex_signatures()}
    assert covered == all_names


def test_vertex_signatures_qed_sector_split():
    eQED, qPsi, mu, nu = S("eQED", "qPsi", "mu_d", "nu_d")
    fermion = make_dirac_fermion(
        "PsiQED",
        symbol=S("psiQED"),
        conjugate_symbol=S("psibarQED"),
        charge=qPsi,
    )
    photon = make_photon(name="A", symbol=S("A0"))
    u1 = make_u1(eQED, photon.symbol, name="U1QED")
    decl = (
        -(Expression.num(1) / Expression.num(4))
        * FieldStrength(u1, mu, nu)
        * FieldStrength(u1, mu, nu)
        + dirac_covd_decl(fermion, mu=mu)
    )
    model = Model(
        gauge_groups=(u1,),
        fields=(fermion, photon),
        lagrangian_decl=decl,
    )

    compiled = model.lagrangian()
    pure_gauge_names = sorted(
        sig.names for sig in compiled.vertex_signatures(sector="pure_gauge")
    )
    assert pure_gauge_names == [("A", "A")]

    matter_names = sorted(
        sig.names for sig in compiled.vertex_signatures(sector="matter")
    )
    assert matter_names == [("PsiQED.bar", "PsiQED"), ("PsiQED.bar", "PsiQED", "A")]

    assert compiled.vertex_signatures(sector="ghost") == ()
    assert compiled.vertex_signatures(sector="gauge_fixing") == ()


def test_vertex_signatures_local_scalar_sector_is_matter():
    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
        InteractionTerm(
            coupling=S("lam"),
            fields=tuple(phi.occurrence() for _ in range(4)),
        ),
    ))

    matter_names = sorted(
        sig.names for sig in lagrangian.vertex_signatures(sector="matter")
    )
    assert matter_names == [
        ("Phi", "Chi", "Chi"),
        ("Phi", "Phi"),
        ("Phi", "Phi", "Phi", "Phi"),
    ]
    assert lagrangian.vertex_signatures(sector="pure_gauge") == ()
    assert lagrangian.vertex_signatures(sector="ghost") == ()
    assert lagrangian.vertex_signatures(sector="gauge_fixing") == ()


def test_vertex_report_sector_filter_preserves_total_counts():
    compiled, _ = _qcd_with_quark_and_ghost_compiled()
    full_report = compiled.vertex_report()
    pure_gauge_report = compiled.vertex_report(sector="pure_gauge")
    gauge_fixing_report = compiled.vertex_report(sector="gauge_fixing")

    assert pure_gauge_report.total_terms == full_report.total_terms
    assert pure_gauge_report.total_signatures == full_report.total_signatures
    assert pure_gauge_report.matched_signatures < full_report.matched_signatures
    assert pure_gauge_report.matched_terms == 4
    assert gauge_fixing_report.matched_terms == 1
    for signature in pure_gauge_report.signatures:
        assert signature.sectors == ("pure_gauge",)


def test_vertex_signatures_mixed_sector_bilinear_has_local_counts_per_filter():
    compiled, _ = _qcd_with_quark_and_ghost_compiled()

    all_signatures = {sig.names: sig for sig in compiled.vertex_signatures()}
    pure_gauge_signatures = {sig.names: sig for sig in compiled.vertex_signatures(sector="pure_gauge")}
    gauge_fixing_signatures = {sig.names: sig for sig in compiled.vertex_signatures(sector="gauge_fixing")}

    assert all_signatures[("G", "G")].term_count == 3
    assert all_signatures[("G", "G")].sectors == ("gauge_fixing", "pure_gauge")

    assert pure_gauge_signatures[("G", "G")].term_count == 2
    assert pure_gauge_signatures[("G", "G")].sectors == ("pure_gauge",)

    assert gauge_fixing_signatures[("G", "G")].term_count == 1
    assert gauge_fixing_signatures[("G", "G")].sectors == ("gauge_fixing",)


def test_vertex_signature_records_sector_tags_for_each_entry():
    compiled, _ = _qcd_with_quark_and_ghost_compiled()
    for signature in compiled.vertex_signatures():
        for sector in signature.sectors:
            assert sector in KNOWN_VERTEX_SECTORS
