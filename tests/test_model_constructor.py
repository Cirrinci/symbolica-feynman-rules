import feynpy as feynpy_pkg
import theories as theories_pkg
from symbolica import S

from feynpy import GaugeFixing, Model, PartialD
from theories import build_standard_model
from tests.support.builders import make_complex_scalar


def test_model_accepts_single_positional_declared_term():
    phi = make_complex_scalar("PhiDF", symbol=S("phi_df"), conjugate_symbol=S("phi_dfbar"))
    decl = S("c") * PartialD(phi, S("mu"))

    shorthand = Model(decl)
    explicit = Model(lagrangian_decl=decl)

    assert shorthand.name == ""
    assert shorthand.lagrangian_decl == explicit.lagrangian_decl
    assert shorthand.find_field(phi) is phi


def test_model_single_string_positional_argument_still_sets_name():
    model = Model("fan_model")

    assert model.name == "fan_model"
    assert model.lagrangian_decl.source_terms == ()


def test_engine_and_concrete_model_packages_are_separate():
    assert not hasattr(feynpy_pkg, "build_standard_model")
    assert hasattr(theories_pkg, "build_standard_model")


def test_model_accepts_positional_declared_term_with_name_and_metadata_keywords():
    sm = build_standard_model(include_ghosts=False)
    xiB = S("xiB")
    xiW = S("xiW")
    xiG = S("xiG")

    sm_decl_with_gf = (
        sm.lagrangians.LSM
        + GaugeFixing(sm.gauge_groups.U1Y, xi=xiB)
        + GaugeFixing(sm.gauge_groups.SU2L, xi=xiW)
        + GaugeFixing(sm.gauge_groups.SU3C, xi=xiG)
    )

    model = Model(
        sm_decl_with_gf,
        name="SM-unbroken-with-gf",
        gauge_groups=(sm.gauge_groups.U1Y, sm.gauge_groups.SU2L, sm.gauge_groups.SU3C),
        parameters=sm.model.parameters,
    )

    assert model.name == "SM-unbroken-with-gf"
    assert len(model.lagrangian_decl.source_terms) == len(sm_decl_with_gf.source_terms)
    assert model.parameters == sm.model.parameters
