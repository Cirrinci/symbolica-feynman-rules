from __future__ import annotations

from symbolica import Expression, S

from feynpy import COLOR_ADJ_INDEX, Field, GhostField, LORENTZ_INDEX
from symbolic.spenso_structures import (
    COLOR_ADJ,
    LORENTZ,
    gauge_generator,
    structure_constant,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.tensor_canonicalization import (
    CanonicalTensorMonomial,
    _canonicalize_commuting_partial_derivatives,
    _contract_plain_metric_heads,
    _infer_index_groups_from_expression,
    _jacobi_reduce_structure_constant_products,
    canonical_external_index_set,
    canonical_tensor_monomial_report,
    canonical_tensor_monomial_map,
    canonize_full,
)
from symbolic.vertex_engine import pcomp


def _canon(expr):
    return expr.expand().to_canonical_string()


def test_typed_spenso_slots_infer_color_and_weak_groups_separately():
    a, b, c = S("a"), S("b"), S("c")
    i, j = S("i"), S("j")
    aw, bw, cw = S("aw"), S("bw"), S("cw")
    iw, jw = S("iw"), S("jw")
    expr = (
        structure_constant(a, b, c)
        * gauge_generator(a, i, j)
        * weak_structure_constant(aw, bw, cw)
        * weak_gauge_generator(aw, iw, jw)
    )

    inferred = _infer_index_groups_from_expression(expr)
    labels = {
        kind: {item.to_canonical_string() for item in items}
        for kind, items in inferred.items()
    }

    assert labels["color_adj"] == {
        item.to_canonical_string() for item in (a, b, c)
    }
    assert labels["color_fund"] == {
        item.to_canonical_string() for item in (i, j)
    }
    assert labels["weak_adj"] == {
        item.to_canonical_string() for item in (aw, bw, cw)
    }
    assert labels["weak_fund"] == {
        item.to_canonical_string() for item in (iw, jw)
    }


def test_default_inference_matches_explicit_mixed_adjoint_groups():
    a, b, c, d, middle = (S(name) for name in ("a", "b", "c", "d", "middle"))
    aw, bw, cw, dw, weak_middle = (
        S(name) for name in ("aw", "bw", "cw", "dw", "weak_middle")
    )
    expr = (
        structure_constant(a, b, middle)
        * structure_constant(c, d, middle)
        * weak_structure_constant(aw, bw, weak_middle)
        * weak_structure_constant(cw, dw, weak_middle)
    )

    inferred = canonize_full(expr, run_gamma=False)
    explicit = canonize_full(
        expr,
        adjoint_indices=(a, b, c, d, middle),
        weak_adj_indices=(aw, bw, cw, dw, weak_middle),
        run_gamma=False,
        infer_indices=False,
    )

    assert _canon(inferred) == _canon(explicit)


def test_plain_metric_contracts_into_g_lorentz_slot():
    mu, nu, a = S("mu"), S("nu"), S("a")
    G = S("G")
    expr = LORENTZ.g(mu, nu).to_expression() * G(nu, a)
    assert _canon(_contract_plain_metric_heads(expr)) == _canon(G(mu, a))


def test_plain_metric_contracts_into_alpha_adjoint_slot():
    a, b = S("a"), S("b")
    alpha = S("alpha")
    expr = COLOR_ADJ.g(a, b).to_expression() * alpha(b)
    assert _canon(_contract_plain_metric_heads(expr)) == _canon(alpha(a))


def test_plain_metric_contracts_into_partiald_of_g():
    mu, nu, rho, a = S("mu"), S("nu"), S("rho"), S("a")
    partial = S("PartialD")
    G = S("G")
    expr = LORENTZ.g(mu, nu).to_expression() * partial(G(nu, a), rho)
    assert _canon(_contract_plain_metric_heads(expr)) == _canon(partial(G(mu, a), rho))


def test_plain_metric_contracts_into_generic_vector_head_when_field_metadata_is_supplied():
    mu, nu = S("mu"), S("nu")
    photon = Field("A", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,))
    expr = LORENTZ.g(mu, nu).to_expression() * photon(nu).species(nu)
    assert _canon(_contract_plain_metric_heads(expr, field_heads=(photon,))) == _canon(
        photon(mu).species(mu)
    )


def test_plain_metric_contracts_into_generic_ghost_head_when_field_metadata_is_supplied():
    a, b = S("a"), S("b")
    gluon = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    ghost = GhostField(
        "c",
        ghost_of=gluon,
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
    )
    expr = COLOR_ADJ.g(a, b).to_expression() * ghost(b).species(b)
    assert _canon(_contract_plain_metric_heads(expr, field_heads=(ghost,))) == _canon(
        ghost(a).species(a)
    )


def test_canonicalize_commuting_partiald_second_derivatives_on_alpha():
    mu, nu, a = S("mu"), S("nu"), S("a")
    partial = S("PartialD")
    alpha = S("alpha")
    expr = partial(partial(alpha(a), mu), nu) - partial(partial(alpha(a), nu), mu)
    canon = canonize_full(
        expr,
        lorentz_indices=(mu, nu),
        adjoint_indices=(a,),
        run_gamma=False,
        run_color=False,
        run_jacobi_reduction=False,
        run_yang_mills_antisymmetric_zero_drop=False,
    )
    assert _canon(canon) == _canon(Expression.num(0))


def test_canonicalize_commuting_partiald_third_derivatives():
    mu, nu, rho, a = S("mu"), S("nu"), S("rho"), S("a")
    partial = S("PartialD")
    alpha = S("alpha")
    lhs = partial(partial(partial(alpha(a), mu), nu), rho)
    rhs = partial(partial(partial(alpha(a), rho), mu), nu)
    canon = canonize_full(
        lhs - rhs,
        lorentz_indices=(mu, nu, rho),
        adjoint_indices=(a,),
        run_gamma=False,
        run_color=False,
        run_jacobi_reduction=False,
        run_yang_mills_antisymmetric_zero_drop=False,
    )
    assert _canon(canon) == _canon(Expression.num(0))


def test_canonicalize_commuting_partiald_on_field_head():
    mu, nu, a = S("mu"), S("nu"), S("a")
    partial = S("PartialD")
    G = S("G")
    lhs = partial(partial(G(mu, a), nu), mu)
    rhs = partial(partial(G(mu, a), mu), nu)
    canon = canonize_full(
        lhs - rhs,
        lorentz_indices=(mu, nu),
        adjoint_indices=(a,),
        run_gamma=False,
        run_color=False,
        run_jacobi_reduction=False,
        run_yang_mills_antisymmetric_zero_drop=False,
    )
    assert _canon(canon) == _canon(Expression.num(0))


def test_non_partiald_heads_are_not_reordered():
    mu, nu, a = S("mu"), S("nu"), S("a")
    covd = S("CovD")
    alpha = S("alpha")
    expr = covd(covd(alpha(a), mu), nu)
    assert _canon(_canonicalize_commuting_partial_derivatives(expr)) == _canon(expr)


def test_canonize_full_drops_identical_plain_odd_field_squares():
    ghost = GhostField(
        "cA",
        ghost_of="A",
        self_conjugate=False,
    )
    canon = canonize_full(
        ghost.symbol * ghost.symbol,
        run_gamma=False,
        run_color=False,
        field_heads=(ghost,),
    )
    assert _canon(canon) == _canon(Expression.num(0))


def test_jacobi_reduction_basic_combination_vanishes():
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")
    p12 = structure_constant(a, b, e) * structure_constant(c, d, e)
    p13 = structure_constant(a, c, e) * structure_constant(b, d, e)
    p14 = structure_constant(a, d, e) * structure_constant(b, c, e)
    reduced = _jacobi_reduce_structure_constant_products(p12 - p13 + p14)
    assert _canon(reduced) == _canon(Expression.num(0))


def test_jacobi_reduction_opposite_sign_combination_vanishes():
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")
    p12 = structure_constant(a, b, e) * structure_constant(c, d, e)
    p13 = structure_constant(a, c, e) * structure_constant(b, d, e)
    p14 = structure_constant(a, d, e) * structure_constant(b, c, e)
    reduced = _jacobi_reduce_structure_constant_products(-p12 + p13 - p14)
    assert _canon(reduced) == _canon(Expression.num(0))


def test_jacobi_reduction_handles_antisymmetric_slot_permutations():
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")
    f = structure_constant
    expr = -f(b, a, e) * f(c, d, e) - f(c, a, e) * f(d, b, e) - f(d, a, e) * f(b, c, e)
    reduced = _jacobi_reduce_structure_constant_products(expr)
    assert _canon(reduced) == _canon(Expression.num(0))


def test_non_jacobi_structure_constant_products_are_preserved():
    a, b, c, d, e1, e2 = S("a"), S("b"), S("c"), S("d"), S("e1"), S("e2")
    expr = structure_constant(a, b, e1) * structure_constant(c, d, e2)
    reduced = _jacobi_reduce_structure_constant_products(expr)
    assert _canon(reduced) == _canon(expr)


def test_canonical_tensor_monomial_report_merges_dummy_indices_inside_pcomp_powers():
    mu_a, mu_b, q2 = S("mu_a"), S("mu_b"), S("q2")
    expr = pcomp(q2, mu_a) ** 2 + pcomp(q2, mu_b) ** 2

    report = canonical_tensor_monomial_report(
        expr,
        external_indices=canonical_external_index_set(),
        max_dummy_permutations=1000,
    )

    assert report.raw_terms == 2
    assert report.canonical_terms == 1
    assert len(report.map) == 1
    assert next(iter(report.map.values())).to_canonical_string() == "2"


def test_symmetric_derivative_times_antisymmetric_f_is_dropped():
    mu, nu = S("mu"), S("nu")
    a, b, c = S("a"), S("b"), S("c")
    partial = S("PartialD")
    G = S("G")
    alpha = S("alpha")

    x_ab = partial(G(mu, a), nu) * partial(partial(alpha(b), mu), nu)
    x_ba = partial(G(mu, b), nu) * partial(partial(alpha(a), mu), nu)
    expr = structure_constant(a, b, c) * (x_ab + x_ba)

    canon = canonize_full(
        expr,
        lorentz_indices=(mu, nu),
        adjoint_indices=(a, b, c),
        run_gamma=False,
        run_color=False,
        run_jacobi_reduction=False,
    )
    assert _canon(canon) == _canon(Expression.num(0))


def test_similar_antisymmetric_term_is_not_dropped():
    a, b, c = S("a"), S("b"), S("c")
    X = S("X")
    expr = structure_constant(a, b, c) * X(a)

    canon = canonize_full(
        expr,
        adjoint_indices=(a, b, c),
        run_gamma=False,
        run_color=False,
        run_jacobi_reduction=False,
    )
    assert _canon(canon) != _canon(Expression.num(0))


def test_jacobi_pass_can_be_disabled_in_canonize_full():
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")
    p12 = structure_constant(a, b, e) * structure_constant(c, d, e)
    p13 = structure_constant(a, c, e) * structure_constant(b, d, e)
    p14 = structure_constant(a, d, e) * structure_constant(b, c, e)
    expr = p12 - p13 + p14

    with_jacobi = canonize_full(
        expr,
        adjoint_indices=(a, b, c, d, e),
        run_gamma=False,
        run_color=False,
        run_commuting_partial_derivatives=False,
        run_yang_mills_antisymmetric_zero_drop=False,
    )
    without_jacobi = canonize_full(
        expr,
        adjoint_indices=(a, b, c, d, e),
        run_gamma=False,
        run_color=False,
        run_commuting_partial_derivatives=False,
        run_jacobi_reduction=False,
        run_yang_mills_antisymmetric_zero_drop=False,
    )
    assert _canon(with_jacobi) == _canon(Expression.num(0))
    assert _canon(without_jacobi) != _canon(Expression.num(0))


def _canonical_map(expr, *, external_indices=frozenset(), **kwargs):
    return canonical_tensor_monomial_map(
        expr,
        external_indices=external_indices,
        **kwargs,
    )


def _assert_empty_canonical_map(expr, *, external_indices=frozenset(), **kwargs):
    assert _canonical_map(
        expr,
        external_indices=external_indices,
        **kwargs,
    ) == {}


def test_monomial_map_canonicalizes_symmetric_metric_arguments():
    mu, nu = S("mu"), S("nu")
    _assert_empty_canonical_map(LORENTZ.g(mu, nu).to_expression() - LORENTZ.g(nu, mu).to_expression())


def test_monomial_map_canonicalizes_antisymmetric_tensor_signs():
    a, b, c = S("a"), S("b"), S("c")
    _assert_empty_canonical_map(structure_constant(a, b, c) + structure_constant(b, a, c))
    _assert_empty_canonical_map(structure_constant(a, b, c) - structure_constant(b, c, a))


def test_monomial_map_drops_repeated_antisymmetric_indices():
    a, c = S("a"), S("c")
    _assert_empty_canonical_map(structure_constant(a, a, c))


def test_monomial_map_uses_global_dummy_contraction_graph():
    a1, a2, a3, a4, a5 = (S(f"a{i}") for i in range(1, 6))
    x, y = S("x"), S("y")
    u, v = S("u"), S("v")
    lhs = (
        structure_constant(a1, x, y)
        * structure_constant(x, a2, a3)
        * structure_constant(y, a4, a5)
    )
    rhs = (
        structure_constant(a1, u, v)
        * structure_constant(u, a2, a3)
        * structure_constant(v, a4, a5)
    )
    externals = canonical_external_index_set(
        color_adjoint=(a1, a2, a3, a4, a5),
    )
    _assert_empty_canonical_map(lhs - rhs, external_indices=externals)


def test_monomial_map_sorts_commuting_factors_and_collects():
    mu1, mu2, mu3 = S("mu1"), S("mu2"), S("mu3")
    lhs = pcomp(S("q1"), mu1) * LORENTZ.g(mu2, mu3).to_expression()
    rhs = LORENTZ.g(mu2, mu3).to_expression() * pcomp(S("q1"), mu1)
    _assert_empty_canonical_map(lhs - rhs)


def test_monomial_map_preserves_declared_ordered_factors():
    ordered = S("Ordered")
    x, y = S("x"), S("y")
    result = _canonical_map(
        ordered(x, y) - ordered(y, x),
        noncommuting_heads=("Ordered",),
    )
    assert len(result) == 2


def test_monomial_map_keeps_external_indices_fixed():
    a1, a2, a3, a4, a5 = (S(f"a{i}") for i in range(1, 6))
    x, y = S("x"), S("y")
    term = (
        structure_constant(a1, x, y)
        * structure_constant(x, a2, a3)
        * structure_constant(y, a4, a5)
    )
    externals = canonical_external_index_set(
        color_adjoint=(a1, a2, a3, a4, a5),
    )
    result = _canonical_map(term, external_indices=externals)
    assert len(result) == 1
    key = next(iter(result))
    assert isinstance(key, CanonicalTensorMonomial)
    assert "E:A:a1" in repr(key)
    assert "D:A:1" in repr(key)


def test_monomial_map_does_not_apply_jacobi_identity():
    a, b, c, d, e = S("a"), S("b"), S("c"), S("d"), S("e")
    p12 = structure_constant(a, b, e) * structure_constant(c, d, e)
    p13 = structure_constant(a, c, e) * structure_constant(b, d, e)
    p14 = structure_constant(a, d, e) * structure_constant(b, c, e)
    assert len(_canonical_map(p12 - p13 + p14)) == 3


def test_monomial_map_does_not_apply_momentum_conservation():
    mu = S("mu")
    expr = pcomp(S("q1"), mu) + pcomp(S("q2"), mu) + pcomp(S("q3"), mu)
    assert len(_canonical_map(expr, external_indices={("lorentz", "mu")})) == 3
