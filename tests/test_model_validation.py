import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from model import (  # noqa: E402
    COLOR_FUND_INDEX,
    DiracKineticTerm,
    ComplexScalarKineticTerm,
    Field,
    GaugeFixing,
    GaugeGroup,
    GaugeRepresentation,
    GhostLagrangian,
    Model,
    SPINOR_INDEX,
)
from symbolic.spenso_structures import gauge_generator, structure_constant  # noqa: E402
from tests.support.builders import make_dirac_fermion, make_ghost, make_gluon, make_photon  # noqa: E402


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
