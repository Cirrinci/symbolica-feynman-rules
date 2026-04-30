import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from model import ComplexScalarKineticTerm, Model, Parameter  # noqa: E402
from tests.support.builders import make_complex_scalar  # noqa: E402


def test_model_find_parameter_by_name_symbol_and_identity():
    g_strong = Parameter(name="gStrong", symbol=S("gS"))
    lambda_h = Parameter(name="lambdaH", symbol=S("lamH"))
    model = Model(parameters=(g_strong, lambda_h))

    assert model.find_parameter(g_strong) is g_strong
    assert model.find_parameter("gStrong") is g_strong
    assert model.find_parameter(S("gS")) is g_strong
    assert model.find_parameter("lambdaH") is lambda_h
    assert model.find_parameter(S("lamH")) is lambda_h
    assert model.find_parameter("missing") is None


def test_parameter_assumptions_expose_real_complex_external_internal_and_value():
    yukawa = Parameter(
        name="Yu",
        symbol=S("Yu"),
        complex_param=True,
        internal=False,
        value=S("Yu_input"),
    )
    model = Model(parameters=(yukawa,))

    assumptions = model.parameter_assumptions(S("Yu"))

    assert assumptions is not None
    assert assumptions.name == "Yu"
    assert str(assumptions.symbol) == "Yu"
    assert assumptions.real is False
    assert assumptions.complex is True
    assert assumptions.internal is False
    assert assumptions.external is True
    assert assumptions.has_value is True
    assert str(assumptions.value) == "Yu_input"


def test_parameter_properties_preserve_basic_metadata():
    mu_param = Parameter(name="mu", complex_param=False, internal=True)

    assert mu_param.is_real is True
    assert mu_param.is_complex is False
    assert mu_param.is_internal is True
    assert mu_param.is_external is False
    assert mu_param.has_value is False
    assert mu_param.assumptions().real is True


def test_validation_accepts_declared_parameter_metadata_without_behavior_change():
    scalar = make_complex_scalar("Phi", symbol=S("phi"), conjugate_symbol=S("phidag"))
    z_phi = Parameter(
        name="ZPhi",
        symbol=S("ZPhi"),
        complex_param=False,
        internal=False,
    )
    model = Model(
        fields=(scalar,),
        parameters=(z_phi,),
        lagrangian_decl=ComplexScalarKineticTerm(field=scalar, coefficient=z_phi.symbol),
    )

    report = model.validate()
    assumptions = model.parameter_assumptions(z_phi.symbol)

    assert report.ok
    assert report.issues == ()
    assert assumptions is not None
    assert assumptions.real is True
    assert assumptions.external is True
