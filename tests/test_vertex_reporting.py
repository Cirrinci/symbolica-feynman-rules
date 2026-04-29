from symbolica import S

from model import (
    Field,
    InteractionTerm,
    Lagrangian,
    Model,
)
from tests.support.builders import (
    dirac_covd_decl,
    gauge_kinetic_decl,
    make_dirac_fermion,
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
