from __future__ import annotations

import pytest

from symbolica import Expression, S

from model import CompiledLagrangian, Model, dirac_field, scalar_field
from model.interactions import InteractionTerm
from model.metadata import SPINOR_KIND


def test_explicit_disjoint_local_dirac_pairing_overrides_adjacent_heuristic():
    psi = dirac_field("psi", indices=())
    chi = dirac_field("chi", indices=())
    phi = scalar_field("phi", self_conjugate=True)
    alpha, beta = S("alpha", "beta")

    lagrangian = Model(
        S("g") * psi.bar(alpha) * phi * psi(alpha) * chi.bar(beta) * chi(beta)
    ).lagrangian()

    assert len(lagrangian.terms) == 1
    assert lagrangian.terms[0].closed_dirac_bilinears == ((0, 2), (3, 4))
    assert lagrangian.feynman_rule(psi.bar, phi, psi, chi.bar, chi)


def test_crossed_explicit_local_dirac_pairings_are_rejected():
    psi = dirac_field("psi", indices=())
    chi = dirac_field("chi", indices=())
    alpha, beta = S("alpha", "beta")

    with pytest.raises(ValueError, match="Unsupported fermion ordering"):
        Model(
            S("g") * psi.bar(alpha) * chi.bar(beta) * psi(alpha) * chi(beta)
        ).lagrangian()


def test_vertex_engine_rejects_overlapping_manual_closed_dirac_bilinears():
    psi = dirac_field("psi", indices=())
    chi = dirac_field("chi", indices=())
    alpha, beta = S("alpha", "beta")

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True, labels={SPINOR_KIND: alpha}),
            chi.occurrence(conjugated=True, labels={SPINOR_KIND: beta}),
            psi.occurrence(labels={SPINOR_KIND: alpha}),
            chi.occurrence(labels={SPINOR_KIND: beta}),
        ),
        closed_dirac_bilinears=((0, 2), (1, 3)),
    )

    with pytest.raises(ValueError, match="disjoint source-order Dirac pairings"):
        CompiledLagrangian(terms=(term,)).feynman_rule(psi.bar, chi.bar, psi, chi)
