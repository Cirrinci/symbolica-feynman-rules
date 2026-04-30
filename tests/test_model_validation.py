import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from model import (  # noqa: E402
    COLOR_FUND_INDEX,
    ComplexScalarKineticTerm,
    DiracKineticTerm,
    Field,
    GaugeFixing,
    GaugeGroup,
    GaugeKineticTerm,
    GaugeRepresentation,
    GhostLagrangian,
    Model,
    SPINOR_INDEX,
)
from symbolic.spenso_structures import gauge_generator, structure_constant  # noqa: E402
from tests.support.builders import make_complex_scalar, make_dirac_fermion, make_ghost, make_gluon, make_photon  # noqa: E402


def test_model_validate_reports_missing_ghost_field():
    gluon = make_gluon(symbol=S("G"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=GhostLagrangian(su3),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["missing_ghost_field"]
    assert "declare ghost_field" in report.issues[0].message


def test_model_validate_reports_abelian_ghost_sector():
    photon = make_photon(symbol=S("A"))
    ghost = make_ghost(name="ghA", symbol=S("ghA"), conjugate_symbol=S("ghAbar"))
    u1 = GaugeGroup(
        name="U1QED",
        abelian=True,
        coupling=S("eQED"),
        gauge_boson=photon.symbol,
        ghost_field=ghost.symbol,
        charge="Q",
    )
    model = Model(
        gauge_groups=(u1,),
        fields=(photon, ghost),
        lagrangian_decl=GhostLagrangian(u1),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["abelian_ghost_sector"]
    assert "non-abelian gauge groups" in report.issues[0].message


def test_model_validate_reports_undeclared_gauge_group_references():
    photon = make_photon(symbol=S("A"))
    declared = GaugeGroup(
        name="U1QED",
        abelian=True,
        coupling=S("eQED"),
        gauge_boson=photon.symbol,
        charge="Q",
    )
    missing = GaugeGroup(
        name="MissingSU3",
        abelian=False,
        coupling=S("gMissing"),
        gauge_boson=S("Gmissing"),
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(declared,),
        fields=(photon,),
        lagrangian_decl=GaugeFixing(missing, xi=S("xiMissing")) + GhostLagrangian(missing),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == [
        "undeclared_gauge_group",
        "undeclared_gauge_group",
    ]
    assert "Gauge-fixing validation" in report.issues[0].message
    assert "Ghost validation" in report.issues[1].message


def test_model_validate_reports_missing_structure_constant_for_nonabelian_ghosts():
    gluon = make_gluon(symbol=S("G"))
    ghost = make_ghost(symbol=S("ghG"), conjugate_symbol=S("ghGbar"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        ghost_field=ghost.symbol,
        structure_constant=None,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GhostLagrangian(su3),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["missing_structure_constant"]
    assert "callable structure_constant" in report.issues[0].message


def test_model_validate_accepts_valid_nonabelian_ghost_and_gauge_fixing_setup():
    gluon = make_gluon(symbol=S("G"))
    ghost = make_ghost(symbol=S("ghG"), conjugate_symbol=S("ghGbar"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        ghost_field=ghost.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=GaugeFixing(su3, xi=S("xiQCD")) + GhostLagrangian(su3),
    )

    report = model.validate()

    assert report.ok
    assert report.issues == ()


def test_model_validate_accepts_canonical_scalar_kinetic_term():
    scalar = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    model = Model(fields=(scalar,), lagrangian_decl=ComplexScalarKineticTerm(field=scalar))

    report = model.validate()

    assert report.ok
    assert report.issues == ()


def test_model_validate_reports_noncanonical_scalar_kinetic_normalization():
    scalar = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    model = Model(
        fields=(scalar,),
        lagrangian_decl=ComplexScalarKineticTerm(field=scalar, coefficient=2),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["kinetic_normalization"]
    assert "non-canonical coefficient 2" in report.issues[0].message


def test_model_validate_reports_duplicate_scalar_kinetic_term():
    scalar = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    model = Model(
        fields=(scalar,),
        lagrangian_decl=(
            ComplexScalarKineticTerm(field=scalar)
            + ComplexScalarKineticTerm(field=scalar)
        ),
    )

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["duplicate_kinetic_term"]
    assert "Duplicate complex-scalar kinetic declarations" in report.issues[0].message


def test_model_validate_accepts_canonical_dirac_kinetic_term():
    fermion = make_dirac_fermion("Psi", symbol=S("psi"), conjugate_symbol=S("psibar"))
    model = Model(fields=(fermion,), lagrangian_decl=DiracKineticTerm(field=fermion))

    report = model.validate()

    assert report.ok
    assert report.issues == ()


def test_model_validate_accepts_canonical_vector_kinetic_term():
    gluon = make_gluon(symbol=S("G"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon,),
        lagrangian_decl=GaugeKineticTerm(gauge_group=su3),
    )

    report = model.validate()

    assert report.ok
    assert report.issues == ()


def test_model_validate_accepts_valid_nonabelian_representation_usage():
    quark = make_dirac_fermion("q", symbol=S("q"), conjugate_symbol=S("qbar"), color=True)
    gluon = make_gluon(symbol=S("G"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )
    model = Model(gauge_groups=(su3,), fields=(quark, gluon))
    model.covariant_terms = (DiracKineticTerm(field=quark, gauge_group=su3),)

    report = model.validate()

    assert report.ok
    assert report.issues == ()


def test_model_validate_reports_invalid_representation_slot():
    quark = make_dirac_fermion("q", symbol=S("q"), conjugate_symbol=S("qbar"), color=True)
    gluon = make_gluon(symbol=S("G"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
                slot=3,
            ),
        ),
    )
    model = Model(gauge_groups=(su3,), fields=(quark, gluon))
    model.covariant_terms = (DiracKineticTerm(field=quark, gauge_group=su3),)

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["gauge_representation_resolution"]
    assert "out of range" in report.issues[0].message


def test_model_validate_reports_missing_nonabelian_representation_for_explicit_term():
    scalar = Field(
        "PhiSinglet",
        spin=0,
        self_conjugate=False,
        symbol=S("phi_singlet"),
        conjugate_symbol=S("phidag_singlet"),
    )
    gluon = make_gluon(symbol=S("G"))
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=S("gS"),
        gauge_boson=gluon.symbol,
        structure_constant=structure_constant,
        representations=(
            GaugeRepresentation(
                index=COLOR_FUND_INDEX,
                generator_builder=gauge_generator,
                name="fund",
            ),
        ),
    )
    model = Model(gauge_groups=(su3,), fields=(scalar, gluon))
    model.covariant_terms = (ComplexScalarKineticTerm(field=scalar, gauge_group=su3),)

    report = model.validate()

    assert not report.ok
    assert [issue.code for issue in report.issues] == ["missing_gauge_representation"]
    assert "PhiSinglet" in report.issues[0].message
    assert "ColorFund" in report.issues[0].message


# ---------------------------------------------------------------------------
# 6.3 mass-structure compiled-term diagnostics
# ---------------------------------------------------------------------------


def test_compiled_validate_diagonal_complex_scalar_mass_term_passes_silently():
    from model import InteractionTerm, Lagrangian

    scalar = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("m"),
            fields=(scalar.occurrence(conjugated=True), scalar.occurrence()),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert report.issues == ()


def test_compiled_validate_diagonal_real_scalar_mass_term_passes_silently():
    from model import InteractionTerm, Lagrangian

    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(coupling=S("m"), fields=(phi.occurrence(), phi.occurrence())),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert report.issues == ()


def test_compiled_validate_diagonal_dirac_mass_term_passes_silently():
    from model import InteractionTerm, Lagrangian

    psi = make_dirac_fermion("Psi", symbol=S("psi"), conjugate_symbol=S("psibar"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("mPsi"),
            fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert report.issues == ()


def test_compiled_validate_reports_scalar_mass_mixing():
    from model import InteractionTerm, Lagrangian

    phi1 = make_complex_scalar("Phi1", symbol=S("phi1"), conjugate_symbol=S("phi1dag"))
    phi2 = make_complex_scalar("Phi2", symbol=S("phi2"), conjugate_symbol=S("phi2dag"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("m12"),
            fields=(phi1.occurrence(conjugated=True), phi2.occurrence()),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok  # warning-only, not an error
    assert [issue.code for issue in report.issues] == ["mass_structure_mixing"]
    issue = report.issues[0]
    assert issue.severity == "warning"
    assert "Phi1" in issue.message
    assert "Phi2" in issue.message
    assert "scalar" in issue.message


def test_compiled_validate_reports_fermion_mass_mixing():
    from model import InteractionTerm, Lagrangian

    psi = make_dirac_fermion("Psi", symbol=S("psi"), conjugate_symbol=S("psibar"))
    chi = make_dirac_fermion("Chi", symbol=S("chi"), conjugate_symbol=S("chibar"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("mMix"),
            fields=(psi.occurrence(conjugated=True), chi.occurrence()),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert [issue.code for issue in report.issues] == ["mass_structure_mixing"]
    issue = report.issues[0]
    assert issue.severity == "warning"
    assert "Psi" in issue.message
    assert "Chi" in issue.message
    assert "fermion" in issue.message


def test_compiled_validate_skips_kinetic_bilinears_with_derivatives():
    from symbolica import Expression
    from model import CovD, FieldStrength, GaugeFixing, GhostLagrangian
    from tests.support.builders import dirac_covd_decl, make_su3

    gluon = make_gluon(name="G", symbol=S("G0"))
    ghost = make_ghost(name="ghG", symbol=S("ghG"), conjugate_symbol=S("ghGbar"))
    quark = make_dirac_fermion("q", symbol=S("q0"), conjugate_symbol=S("qbar0"), color=True)
    su3 = make_su3(S("gS"), gluon.symbol, ghost_sym=ghost.symbol, name="SU3C")
    mu, nu = S("mu_d"), S("nu_d")
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

    compiled = model.lagrangian()
    report = compiled.validate()

    assert report.ok
    assert all(issue.code != "mass_structure_mixing" for issue in report.issues)


def test_compiled_validate_ignores_higher_arity_interactions():
    from model import InteractionTerm, Lagrangian

    phi = Field("Phi", spin=0, self_conjugate=True, symbol=S("phi"))
    chi = Field("Chi", spin=0, self_conjugate=True, symbol=S("chi"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("g"),
            fields=(phi.occurrence(), chi.occurrence(), chi.occurrence()),
        ),
        InteractionTerm(
            coupling=S("lam"),
            fields=tuple(phi.occurrence() for _ in range(4)),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert report.issues == ()


def test_compiled_validate_skips_mixed_statistics_bilinears():
    from model import InteractionTerm, Lagrangian

    phi = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    psi = make_dirac_fermion("Psi", symbol=S("psi"), conjugate_symbol=S("psibar"))
    lagrangian = Lagrangian(terms=(
        InteractionTerm(
            coupling=S("c"),
            fields=(phi.occurrence(), psi.occurrence()),
        ),
    ))

    report = lagrangian.validate()

    assert report.ok
    assert all(issue.code != "mass_structure_mixing" for issue in report.issues)
