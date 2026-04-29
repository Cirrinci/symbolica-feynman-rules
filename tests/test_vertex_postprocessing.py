import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from symbolica import S  # noqa: E402

from examples.examples import MODEL_QCD_GAUGE_COVARIANT, GluonField  # noqa: E402
from model import Field, Lagrangian  # noqa: E402
from model.interactions import ExternalLeg, _auto_leg_labels  # noqa: E402
from symbolic.tensor_canonicalization import contract_spenso_lorentz_metrics  # noqa: E402
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


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_vertex_factor_output_policy_matches_manual_postprocessing_for_quartic_scalar():
    lam = S("lam_post")
    phi = Field("phi_post", spin=0, self_conjugate=True, symbol=S("phi_post"))
    lagrangian = Lagrangian(lam * phi * phi * phi * phi)
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
