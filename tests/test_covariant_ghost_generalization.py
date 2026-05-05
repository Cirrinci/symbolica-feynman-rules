import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))


from symbolica import S  # noqa: E402

from compiler.gauge import expand_cov_der  # noqa: E402
from model import COLOR_ADJ_INDEX, CovD, Field, GaugeGroup, GhostField, LORENTZ_INDEX, Model, PartialD  # noqa: E402
from symbolic.spenso_structures import structure_constant  # noqa: E402
from symbolic.vertex_engine import pcomp  # noqa: E402
from tests.support.builders import canon  # noqa: E402


def _make_qcd_ghost_covd_model():
    gS = S("gS")
    mu = S("mu_decl")
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        symbol=S("G"),
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "ghG",
        ghost_of=gluon,
        self_conjugate=False,
        symbol=S("ghG"),
        conjugate_symbol=S("ghGbar"),
        indices=(COLOR_ADJ_INDEX,),
        quantum_numbers={"GhostNumber": 1},
    )
    su3 = GaugeGroup(
        name="SU3C",
        abelian=False,
        coupling=gS,
        gauge_boson=gluon.symbol,
        ghost_field=ghost.symbol,
        structure_constant=structure_constant,
    )
    model = Model(
        gauge_groups=(su3,),
        fields=(gluon, ghost),
        lagrangian_decl=-(ghost.bar * PartialD(CovD(ghost, mu), mu)),
    )
    return model, ghost, gluon, su3


def test_expand_cov_der_infers_adjoint_representation_for_ghost_field():
    model, ghost, gluon, su3 = _make_qcd_ghost_covd_model()

    expanded = expand_cov_der(model, CovD(ghost, S("mu_decl")))

    assert expanded.field is ghost
    assert expanded.conjugated is False
    assert len(expanded.gauge_current_pieces) == 1

    piece = expanded.gauge_current_pieces[0]
    assert piece.metadata.gauge_group is su3
    assert piece.metadata.gauge_field is gluon
    assert piece.metadata.representation is not None
    assert piece.metadata.representation.name == "adjoint"
    assert piece.metadata.representation.index == COLOR_ADJ_INDEX
    assert piece.metadata.representation_slots == (0,)
    assert piece.active_slot == 0


def test_feynrules_style_ghost_covd_vertex_keeps_product_rule_momenta():
    """The direct `-cbar * PartialD(CovD(c, mu), mu)` form keeps `p_ghost + p_gluon`.

    This intentionally differs from the integrated-by-parts `GhostLagrangian(...)`
    helper, whose compact convention places the momentum on the antighost leg.
    In the direct product-rule form, both momentum terms are rendered on the
    external gluon Lorentz label carried by leg 3.
    """
    model, ghost, _gluon, _su3 = _make_qcd_ghost_covd_model()

    got = model.lagrangian().feynman_rule(
        ghost.bar,
        _gluon,
        ghost,
        include_delta=False,
    )
    expected = (
        S("gS")
        * structure_constant(S("a2"), S("a1"), S("a3"))
        * (pcomp(S("q2"), S("mu2")) + pcomp(S("q3"), S("mu2")))
    )

    assert canon(got) == canon(expected)


def test_feynrules_style_ghost_covd_vertex_ghbar_gh_gluon_uses_gluon_leg_lorentz_index():
    """For leg order `(ghG.bar, ghG, G)`, both momentum terms carry the gluon index."""
    model, ghost, gluon, _su3 = _make_qcd_ghost_covd_model()

    got = model.lagrangian().feynman_rule(
        ghost.bar,
        ghost,
        gluon,
        include_delta=False,
        simplify=True,
    )
    expected = (
        S("gS")
        * structure_constant(S("a3"), S("a1"), S("a2"))
        * (pcomp(S("q2"), S("mu3")) + pcomp(S("q3"), S("mu3")))
    )

    assert canon(got) == canon(expected)
