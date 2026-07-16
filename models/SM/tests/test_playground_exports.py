from __future__ import annotations

from symbolica import S

import models.SM as sm_pkg


def test_star_import_exposes_playground_names():
    namespace: dict[str, object] = {}
    exec("from models.SM import *", namespace)

    assert namespace["QL"] is sm_pkg.QL
    assert namespace["Yd"] is sm_pkg.Yd
    assert namespace["L_gauge"] == sm_pkg.LGauge
    assert namespace["L_tot"] == sm_pkg.LSM
    assert namespace["SM_GAUGE_GROUPS"] == (sm_pkg.U1Y, sm_pkg.SU2L, sm_pkg.SU3C)
    assert namespace["SM_PARAMETERS"] == sm_pkg.default_standard_model().model.parameters


def test_sm_model_builds_custom_term_with_standard_model_metadata():
    spinor = S("sp")
    weak_left = S("ii")
    colour = S("cc")
    f1, f2, f3 = S("ff1", "ff2", "ff3")

    model = sm_pkg.sm_model(
        -sm_pkg.Yd(f2, f3)
        * sm_pkg.CKM(f1, f2)
        * sm_pkg.QL.bar(spinor, weak_left, f1, colour)
        * sm_pkg.dR(spinor, f3, colour)
        * sm_pkg.Phi(weak_left),
        name="custom-yukawa",
    )

    assert model.name == "custom-yukawa"
    assert model.gauge_groups == sm_pkg.SM_GAUGE_GROUPS
    assert model.parameters == sm_pkg.SM_PARAMETERS
    assert model.find_field(sm_pkg.QL) is sm_pkg.QL
    assert model.find_field(sm_pkg.dR) is sm_pkg.dR
    assert model.find_field(sm_pkg.Phi) is sm_pkg.Phi
