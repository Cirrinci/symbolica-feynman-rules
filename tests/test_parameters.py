from symbolica import S  # noqa: E402

from model import Model, Parameter  # noqa: E402
from tests.support.builders import make_complex_scalar  # noqa: E402


def _canon(expr):
    return expr.expand().to_canonical_string()


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

    assumptions = yukawa.assumptions()

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


def test_parameter_object_can_be_used_directly_as_declared_coupling():
    phi = make_complex_scalar("Phi4", symbol=S("phi4"), conjugate_symbol=S("phi4dag"))
    g4 = Parameter(name="g4", symbol=S("g4"))
    model = Model(
        fields=(phi,),
        parameters=(g4,),
        lagrangian_decl=g4 * phi * phi.bar * phi * phi.bar,
    )

    expr = model.lagrangian().feynman_rule(phi.bar, phi, phi.bar, phi, simplify=True)
    expr_from_symbol = Model(
        fields=(phi,),
        parameters=(g4,),
        lagrangian_decl=g4.symbol * phi * phi.bar * phi * phi.bar,
    ).lagrangian().feynman_rule(phi.bar, phi, phi.bar, phi, simplify=True)

    assert "Parameter(" not in str(expr)
    assert "g4" in str(expr)
    assert _canon(expr) == _canon(expr.expand())
    assert _canon(expr) == _canon(expr_from_symbol)


def test_parameter_object_matches_symbol_path_in_feynman_rule():
    phi = make_complex_scalar("PhiSym", symbol=S("phi_sym"), conjugate_symbol=S("phi_symdag"))
    g4 = Parameter(name="g4sym", symbol=S("g4sym"))

    model_parameter = Model(
        fields=(phi,),
        parameters=(g4,),
        lagrangian_decl=g4 * phi.bar * phi * phi.bar * phi,
    )
    model_symbol = Model(
        fields=(phi,),
        parameters=(g4,),
        lagrangian_decl=g4.symbol * phi.bar * phi * phi.bar * phi,
    )

    got_parameter = model_parameter.lagrangian().feynman_rule(phi.bar, phi, phi.bar, phi, simplify=True)
    got_symbol = model_symbol.lagrangian().feynman_rule(phi.bar, phi, phi.bar, phi, simplify=True)

    assert _canon(got_parameter) == _canon(got_symbol)
