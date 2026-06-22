from __future__ import annotations

from collections import Counter

import pytest
from symbolica import Expression, S

from models import build_standard_model
from model.interactions import InteractionTerm, _field_match_key
from model.lagrangian import CompiledLagrangian
from symbolic.spenso_structures import (
    chiral_projector_left,
    chiral_projector_right,
    gamma_matrix,
    lorentz_metric,
    spinor_metric,
)
from symbolic.tensor_canonicalization import canonize_full
from symbolic.vertex_engine import Delta, I, pcomp, pi


ZERO = Expression.num(0)
ONE = Expression.num(1)
TWO = Expression.num(2)
HALF = ONE / TWO
INV_SQRT2 = HALF**HALF

q1, q2, q3, q4 = S("q1", "q2", "q3", "q4")
d = S("d")
D2 = (2 * pi) ** d * Delta(q1 + q2)
D3 = (2 * pi) ** d * Delta(q1 + q2 + q3)
D4 = (2 * pi) ** d * Delta(q1 + q2 + q3 + q4)


@pytest.fixture(scope="module")
def sm():
    return build_standard_model()


@pytest.fixture(scope="module")
def sm_rxi():
    return build_standard_model(
        xiA=S("xiA"),
        xiZ=S("xiZ"),
        xiW=S("xiW"),
        xiG=S("xiG"),
    )


def _canon(expr):
    return expr.expand().to_canonical_string()


def _field_counter(*field_args):
    return Counter(
        _field_match_key(
            field.field if hasattr(field, "field") else field,
            hasattr(field, "field"),
        )
        for field in field_args
    )


def _sector(lagrangian, *field_args, differentiated: bool):
    target = _field_counter(*field_args)
    terms = tuple(
        term
        for term in lagrangian.terms
        if bool(term.derivatives) is differentiated
        and Counter(
            _field_match_key(occurrence.field, occurrence.conjugated)
            for occurrence in term.fields
        )
        == target
    )
    return CompiledLagrangian(
        terms=terms,
        parameters=lagrangian.parameters,
    )


def _rule(sector, *field_args, flavor_expand=False):
    if not sector.terms:
        return ZERO
    return sector.feynman_rule(
        *field_args,
        simplify=True,
        include_delta=True,
        flavor_expand=flavor_expand,
    )


def _assert_rational_equal(got, expected, denominator):
    symbols = sorted(
        denominator.get_all_symbols(),
        key=lambda symbol: symbol.to_canonical_string(),
    )
    for values in ((3, 4), (5, 2)):
        difference = got - expected
        for symbol, value in zip(symbols, values):
            difference = difference.replace(symbol, Expression.num(value))
        if hasattr(difference, "cancel"):
            difference = difference.cancel()
        assert _canon(difference) == _canon(ZERO)


def _assert_symbolic_equal(got, expected):
    difference = got - expected
    symbols = sorted(
        difference.get_all_symbols(),
        key=lambda symbol: symbol.to_canonical_string(),
    )
    samples = (
        (2, 3, 5, 7, 11, 13, 17, 19),
        (23, 29, 31, 37, 41, 43, 47, 53),
    )
    for sample in samples:
        candidate = difference
        for symbol, value in zip(symbols, sample):
            candidate = candidate.replace(symbol, Expression.num(value))
        if hasattr(candidate, "cancel"):
            candidate = candidate.cancel()
        assert _canon(candidate) == _canon(ZERO)


def _internal_momentum_square(expr):
    internal_labels = sorted(
        (
            symbol
            for symbol in expr.get_all_symbols()
            if symbol.to_canonical_string().endswith("_int")
        ),
        key=lambda symbol: symbol.to_canonical_string(),
    )
    assert internal_labels
    mu = internal_labels[0]
    return pcomp(q1, mu) * pcomp(q2, mu)


def _collapse_internal_labels(expr):
    internal_labels = sorted(
        (
            symbol
            for symbol in expr.get_all_symbols()
            if symbol.to_canonical_string().endswith("_int")
        ),
        key=lambda symbol: symbol.to_canonical_string(),
    )
    if not internal_labels:
        return expr
    collapsed = expr
    base = internal_labels[0]
    for label in internal_labels[1:]:
        collapsed = collapsed.replace(label, base)
    return collapsed


def test_standard_model_builds_from_source_basis_and_validates(sm):
    assert sm.source_model.validate().ok
    assert sm.model.validate().ok
    assert sm.lagrangian.terms

    source_fields = {
        sm.fields.LL,
        sm.fields.lR,
        sm.fields.QL,
        sm.fields.uR,
        sm.fields.dR,
        sm.fields.Phi,
        sm.fields.B,
        sm.fields.Wi,
        sm.fields.ghB,
        sm.fields.ghWi,
    }
    assert all(
        occurrence.field not in source_fields
        for term in sm.lagrangian.terms
        for occurrence in term.fields
    )
    assert sm.fields.A.mass == ZERO
    assert sm.fields.W.mass == sm.parameters.MW.symbol
    assert sm.fields.Z.mass == sm.parameters.MZ.symbol
    assert sm.fields.H.mass == sm.parameters.MH.symbol
    assert sm.fields.G0.goldstone_of is sm.fields.Z
    assert sm.fields.GP.goldstone_of is sm.fields.W
    assert sm.fields.ghZ.ghost_of is sm.fields.Z
    assert sm.fields.ghWp.ghost_of is sm.fields.W
    assert sm.fields.GP.quantum_numbers["Q"] == ONE
    assert sm.fields.ghWm.quantum_numbers["Q"] == -ONE


def test_barred_source_fermion_transformations_include_conjugated_projectors(sm):
    source = CompiledLagrangian(
        terms=(
            InteractionTerm(
                coupling=1,
                fields=(sm.fields.LL.bar(S("sp1"), 1, S("ff1")),),
            ),
        )
    )
    rule = next(
        transformation
        for transformation in sm.transformations
        if transformation.source is sm.fields.LL
        and transformation.components == {1: 1}
    )

    result = source.transform_fields(rule, repeat=False)

    assert len(result.terms) == 1
    occurrence = result.terms[0].fields[0]
    assert occurrence.field is sm.fields.vl
    assert occurrence.conjugated is True
    assert occurrence.slot_labels.get(1) == S("ff1")
    assert _canon(result.terms[0].coupling) == _canon(
        chiral_projector_right(
            occurrence.slot_labels.get(0),
            S("sp1"),
        )
    )


def test_canonical_scalar_and_vector_kinetic_terms(sm):
    L = sm.lagrangian
    fields = sm.fields
    scalar_kinetic = _rule(
        _sector(L, fields.H, fields.H, differentiated=True),
        fields.H,
        fields.H,
    )
    expected_scalar = (
        -I
        * pcomp(q1, S("mu1_int"))
        * pcomp(q2, S("mu1_int"))
        * D2
    )
    assert _canon(scalar_kinetic) == _canon(expected_scalar)

    photon_kinetic = _rule(
        _sector(L, fields.A, fields.A, differentiated=True),
        fields.A,
        fields.A,
    )
    expected_vector = I * (
        lorentz_metric(S("mu1"), S("mu2"))
        * pcomp(q1, S("mu1_int"))
        * pcomp(q2, S("mu1_int"))
        - pcomp(q1, S("mu2")) * pcomp(q2, S("mu1"))
        + pcomp(q1, S("mu1")) * pcomp(q2, S("mu2"))
    ) * D2
    denominator = sm.parameters.g1.symbol**2 + sm.parameters.g2.symbol**2
    _assert_rational_equal(photon_kinetic, expected_vector, denominator)


def test_rxi_longitudinal_vector_terms_and_gauge_goldstone_cancellation(sm_rxi):
    L = sm_rxi.lagrangian
    fields = sm_rxi.fields

    photon_kinetic = _rule(
        _sector(L, fields.A, fields.A, differentiated=True),
        fields.A,
        fields.A,
    )
    expected_photon = I * (
        lorentz_metric(S("mu1"), S("mu2"))
        * pcomp(q1, S("mu1_int"))
        * pcomp(q2, S("mu1_int"))
        - pcomp(q1, S("mu2")) * pcomp(q2, S("mu1"))
        + sm_rxi.parameters.xiA.symbol**-1
        * pcomp(q1, S("mu1"))
        * pcomp(q2, S("mu2"))
    ) * D2
    _assert_symbolic_equal(photon_kinetic, expected_photon)

    w_kinetic = _rule(
        _sector(L, fields.W.bar, fields.W, differentiated=True),
        fields.W.bar,
        fields.W,
    )
    expected_w = I * (
        lorentz_metric(S("mu1"), S("mu2"))
        * pcomp(q1, S("mu1_int"))
        * pcomp(q2, S("mu1_int"))
        - pcomp(q1, S("mu2")) * pcomp(q2, S("mu1"))
        + sm_rxi.parameters.xiW.symbol**-1
        * pcomp(q1, S("mu1"))
        * pcomp(q2, S("mu2"))
    ) * D2
    _assert_symbolic_equal(w_kinetic, expected_w)

    zg0 = _rule(
        _sector(L, fields.Z, fields.G0, differentiated=True),
        fields.Z,
        fields.G0,
    )
    wgp = _rule(
        _sector(L, fields.W.bar, fields.GP, differentiated=True),
        fields.W.bar,
        fields.GP,
    )
    assert _canon(zg0) == _canon(ZERO)
    assert _canon(wgp) == _canon(ZERO)


def test_neutral_kinetic_and_mass_matrices_are_diagonal(sm):
    L = sm.lagrangian
    fields = sm.fields
    az_kinetic = _rule(
        _sector(L, fields.A, fields.Z, differentiated=True),
        fields.A,
        fields.Z,
    )
    az_mass = _rule(
        _sector(L, fields.A, fields.Z, differentiated=False),
        fields.A,
        fields.Z,
    )
    photon_mass = _rule(
        _sector(L, fields.A, fields.A, differentiated=False),
        fields.A,
        fields.A,
    )

    assert _canon(az_kinetic) == _canon(ZERO)
    assert _canon(az_mass) == _canon(ZERO)
    assert _canon(photon_mass) == _canon(ZERO)


def test_rxi_goldstone_and_ghost_masses_follow_xi_parameters(sm_rxi):
    L = sm_rxi.lagrangian
    fields = sm_rxi.fields
    g1 = sm_rxi.parameters.g1.symbol
    g2 = sm_rxi.parameters.g2.symbol
    vev = sm_rxi.parameters.vev.symbol

    g0_mass = _rule(
        _sector(L, fields.G0, fields.G0, differentiated=False),
        fields.G0,
        fields.G0,
    )
    expected_g0 = -I * sm_rxi.parameters.xiZ.symbol * (g1**2 + g2**2) * vev**2 / 4 * D2
    _assert_symbolic_equal(g0_mass, expected_g0)

    gp_mass = _rule(
        _sector(L, fields.GP.bar, fields.GP, differentiated=False),
        fields.GP.bar,
        fields.GP,
    )
    expected_gp = -I * sm_rxi.parameters.xiW.symbol * g2**2 * vev**2 / 4 * D2
    _assert_symbolic_equal(gp_mass, expected_gp)

    ghz_mass = _rule(
        _sector(L, fields.ghZ.bar, fields.ghZ, differentiated=False),
        fields.ghZ.bar,
        fields.ghZ,
    )
    expected_ghz = -I * sm_rxi.parameters.xiZ.symbol * (g1**2 + g2**2) * vev**2 / 4 * D2
    _assert_symbolic_equal(ghz_mass, expected_ghz)

    ghwp_mass = _rule(
        _sector(L, fields.ghWp.bar, fields.ghWp, differentiated=False),
        fields.ghWp.bar,
        fields.ghWp,
    )
    expected_ghwp = -I * sm_rxi.parameters.xiW.symbol * g2**2 * vev**2 / 4 * D2
    _assert_symbolic_equal(ghwp_mass, expected_ghwp)


def test_w_and_z_masses_match_the_higgs_mechanism(sm):
    L = sm.lagrangian
    fields = sm.fields
    g1 = sm.parameters.g1.symbol
    g2 = sm.parameters.g2.symbol
    vev = sm.parameters.vev.symbol
    denominator = g1**2 + g2**2

    w_mass = _rule(
        _sector(L, fields.W.bar, fields.W, differentiated=False),
        fields.W.bar,
        fields.W,
    )
    expected_w = (
        I
        * g2**2
        * vev**2
        / 4
        * lorentz_metric(S("mu1"), S("mu2"))
        * D2
    )
    assert _canon(w_mass) == _canon(expected_w)

    z_mass = _rule(
        _sector(L, fields.Z, fields.Z, differentiated=False),
        fields.Z,
        fields.Z,
    )
    expected_z = (
        I
        * denominator
        * vev**2
        / 4
        * lorentz_metric(S("mu1"), S("mu2"))
        * D2
    )
    _assert_rational_equal(z_mass, expected_z, denominator)


def test_higgs_mass_self_couplings_and_hvv_vertices(sm):
    L = sm.lagrangian
    fields = sm.fields
    g1 = sm.parameters.g1.symbol
    g2 = sm.parameters.g2.symbol
    lam = sm.parameters.lam.symbol
    vev = sm.parameters.vev.symbol
    denominator = g1**2 + g2**2

    higgs_mass = _rule(
        _sector(L, fields.H, fields.H, differentiated=False),
        fields.H,
        fields.H,
    )
    assert _canon(higgs_mass) == _canon(-2 * I * lam * vev**2 * D2)
    assert _canon(
        L.feynman_rule(fields.H, fields.H, fields.H, simplify=True, include_delta=True)
    ) == _canon(-6 * I * lam * vev * D3)
    assert _canon(
        L.feynman_rule(
            fields.H,
            fields.H,
            fields.H,
            fields.H,
            simplify=True,
            include_delta=True,
        )
    ) == _canon(-6 * I * lam * D4)

    hww = L.feynman_rule(
        fields.H,
        fields.W.bar,
        fields.W,
        simplify=True,
        include_delta=True,
    )
    expected_hww = (
        I
        * g2**2
        * vev
        / 2
        * lorentz_metric(S("mu2"), S("mu3"))
        * D3
    )
    assert _canon(hww) == _canon(expected_hww)

    hzz = L.feynman_rule(
        fields.H,
        fields.Z,
        fields.Z,
        simplify=True,
        include_delta=True,
    )
    expected_hzz = (
        I
        * denominator
        * vev
        / 2
        * lorentz_metric(S("mu2"), S("mu3"))
        * D3
    )
    _assert_rational_equal(hzz, expected_hzz, denominator)


def test_electromagnetic_and_charged_fermion_currents(sm):
    L = sm.lagrangian
    fields = sm.fields
    g1 = sm.parameters.g1.symbol
    g2 = sm.parameters.g2.symbol
    ee = sm.parameters.ee.value

    neutrino_photon = canonize_full(
        L.feynman_rule(
            fields.vl.bar,
            fields.vl,
            fields.A,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    assert _canon(neutrino_photon) == _canon(ZERO)

    lepton_photon = canonize_full(
        L.feynman_rule(
            fields.l.bar,
            fields.l,
            fields.A,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    flavor_metric = sm.indices.generation.representation.g(
        S("fl1"),
        S("fl2"),
    ).to_expression()
    projector_left = chiral_projector_left(S("k"), S("i2"))
    projector_right = chiral_projector_right(S("k"), S("i2"))
    expected_qed = (
        -I
        * ee
        * flavor_metric
        * (
            chiral_projector_right(S("i1"), S("a"))
            * gamma_matrix(S("a"), S("k"), S("mu3"))
            * projector_left
            + chiral_projector_left(S("i1"), S("a"))
            * gamma_matrix(S("a"), S("k"), S("mu3"))
            * projector_right
        )
        * D3
    )
    expected_qed = canonize_full(
        expected_qed,
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    assert _canon(lepton_photon) == _canon(expected_qed)

    charged_current = canonize_full(
        L.feynman_rule(
            fields.uq.bar,
            fields.dq,
            fields.W,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    text = _canon(charged_current)
    assert "CKM(" in text
    assert "gamma5(" in text
    assert "g2" in text


def test_ckm_orientation_and_neutral_current_unitarity(sm):
    L = sm.lagrangian
    fields = sm.fields

    charged_current = canonize_full(
        L.feynman_rule(
            fields.uq.bar,
            fields.dq,
            fields.W,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    charged_conjugate = canonize_full(
        L.feynman_rule(
            fields.dq.bar,
            fields.uq,
            fields.W.bar,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    neutral_current = canonize_full(
        L.feynman_rule(
            fields.uq.bar,
            fields.uq,
            fields.Z,
            simplify=True,
            include_delta=True,
        ),
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )

    charged_text = _canon(charged_current)
    charged_conjugate_text = _canon(charged_conjugate)
    neutral_text = _canon(neutral_current)

    assert "CKM(" in charged_text
    assert "CKMDag(" not in charged_text
    assert "CKMDag(" in charged_conjugate_text
    assert "CKM(" not in charged_conjugate_text
    assert "CKM(" not in neutral_text
    assert "CKMDag(" not in neutral_text


def test_representative_fermion_masses_and_yukawa_vertex(sm):
    L = sm.lagrangian
    fields = sm.fields
    vev = sm.parameters.vev.symbol
    electron = fields.l.class_members[0]
    up = fields.uq.class_members[0]

    electron_mass = _rule(
        _sector(L, fields.l.bar, fields.l, differentiated=False),
        electron.bar,
        electron,
        flavor_expand=True,
    )
    expected_electron_mass = (
        -I
        * INV_SQRT2
        * vev
        * S("ye1")
        * (
            chiral_projector_right(S("i1"), S("k"))
            * chiral_projector_right(S("k"), S("i2"))
            + chiral_projector_left(S("i1"), S("k"))
            * chiral_projector_left(S("k"), S("i2"))
        )
        * D2
    )
    electron_mass = canonize_full(
        electron_mass,
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    expected_electron_mass = canonize_full(
        expected_electron_mass,
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    assert _canon(electron_mass) == _canon(expected_electron_mass)

    electron_higgs = _rule(
        _sector(L, fields.l.bar, fields.l, fields.H, differentiated=False),
        electron.bar,
        electron,
        fields.H,
        flavor_expand=True,
    )
    expected_electron_higgs = (
        -I
        * INV_SQRT2
        * S("ye1")
        * (
            chiral_projector_right(S("i1"), S("k"))
            * chiral_projector_right(S("k"), S("i2"))
            + chiral_projector_left(S("i1"), S("k"))
            * chiral_projector_left(S("k"), S("i2"))
        )
        * D3
    )
    electron_higgs = canonize_full(
        electron_higgs,
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    expected_electron_higgs = canonize_full(
        expected_electron_higgs,
        infer_indices=True,
        field_heads=tuple(sm.model.fields),
        run_color=False,
    )
    assert _canon(electron_higgs) == _canon(expected_electron_higgs)

    up_mass = _rule(
        _sector(L, fields.uq.bar, fields.uq, differentiated=False),
        up.bar,
        up,
        flavor_expand=True,
    )
    text = _canon(up_mass)
    assert "yu1" in text
    assert "vev" in text
    assert "cof(3" in text


def test_three_and_four_gauge_boson_vertices(sm):
    L = sm.lagrangian
    fields = sm.fields
    ee = sm.parameters.ee.value

    wwa = L.feynman_rule(
        fields.W.bar,
        fields.W,
        fields.A,
        simplify=True,
        include_delta=True,
    )
    expected_wwa = I * ee * (
        lorentz_metric(S("mu1"), S("mu2")) * pcomp(q1, S("mu3"))
        - lorentz_metric(S("mu1"), S("mu2")) * pcomp(q2, S("mu3"))
        - lorentz_metric(S("mu1"), S("mu3")) * pcomp(q1, S("mu2"))
        + lorentz_metric(S("mu1"), S("mu3")) * pcomp(q3, S("mu2"))
        + lorentz_metric(S("mu2"), S("mu3")) * pcomp(q2, S("mu1"))
        - lorentz_metric(S("mu2"), S("mu3")) * pcomp(q3, S("mu1"))
    ) * D3
    assert _canon(wwa) == _canon(expected_wwa)

    quartic_tensor = (
        2
        * lorentz_metric(S("mu1"), S("mu2"))
        * lorentz_metric(S("mu3"), S("mu4"))
        - lorentz_metric(S("mu1"), S("mu3"))
        * lorentz_metric(S("mu2"), S("mu4"))
        - lorentz_metric(S("mu1"), S("mu4"))
        * lorentz_metric(S("mu2"), S("mu3"))
    )
    wwaa = L.feynman_rule(
        fields.W.bar,
        fields.W,
        fields.A,
        fields.A,
        simplify=True,
        include_delta=True,
    )
    assert _canon(wwaa) == _canon(-I * ee**2 * quartic_tensor * D4)


def test_representative_qcd_vertex_is_present(sm):
    rule = sm.lagrangian.feynman_rule(
        sm.fields.uq.bar,
        sm.fields.uq,
        sm.fields.G,
        simplify=True,
        include_delta=True,
    )
    text = _canon(rule)
    assert "g3" in text
    assert "gamma(" in text
    assert "::t(" in text


def test_feynman_gauge_ghost_masses_and_interactions(sm):
    L = sm.lagrangian
    fields = sm.fields
    g1 = sm.parameters.g1.symbol
    g2 = sm.parameters.g2.symbol
    vev = sm.parameters.vev.symbol
    denominator = g1**2 + g2**2

    charged = _collapse_internal_labels(
        L.feynman_rule(
            fields.ghWp.bar,
            fields.ghWp,
            simplify=True,
            include_delta=True,
        )
    )
    ghost_momentum = _internal_momentum_square(charged)
    expected_charged = -I * (
        ghost_momentum + g2**2 * vev**2 / 4
    ) * D2
    assert _canon(charged) == _canon(expected_charged)

    neutral = _collapse_internal_labels(
        L.feynman_rule(
            fields.ghZ.bar,
            fields.ghZ,
            simplify=True,
            include_delta=True,
        )
    )
    ghost_momentum = _internal_momentum_square(neutral)
    expected_neutral = -I * (
        ghost_momentum + denominator * vev**2 / 4
    ) * D2
    _assert_rational_equal(neutral, expected_neutral, denominator)

    photon = _collapse_internal_labels(
        L.feynman_rule(
            fields.ghA.bar,
            fields.ghA,
            simplify=True,
            include_delta=True,
        )
    )
    ghost_momentum = _internal_momentum_square(photon)
    _assert_rational_equal(
        photon,
        -I * ghost_momentum * D2,
        denominator,
    )

    charged_goldstone = L.feynman_rule(
        fields.ghWp.bar,
        fields.ghWp,
        fields.G0,
        simplify=True,
        include_delta=True,
    )
    assert _canon(charged_goldstone) == _canon(g2**2 * vev / 4 * D3)

    charged_photon = L.feynman_rule(
        fields.ghWp.bar,
        fields.A,
        fields.ghWp,
        simplify=True,
        include_delta=True,
    )
    expected_charged_photon = (
        -I
        * sm.parameters.ee.value
        * pcomp(q1, S("mu2"))
        * D3
    )
    assert _canon(charged_photon) == _canon(expected_charged_photon)
