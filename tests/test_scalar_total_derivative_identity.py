from symbolica import S

from model import Model, PartialD, scalar_field


def test_scalar_total_derivative_current_is_representable():
    mu = S("mu")
    phi = scalar_field("Phi", self_conjugate=True)

    lagrangian = Model(
        fields=(phi,),
        lagrangian_decl=phi * phi * PartialD(phi, mu),
    ).lagrangian()

    assert len(lagrangian.terms) == 1
    term = lagrangian.terms[0]
    assert [occ.field.name for occ in term.fields] == ["Phi", "Phi", "Phi"]
    assert [
        (action.target, str(action.lorentz_index))
        for action in term.derivatives
    ] == [(2, "mu")]


def test_scalar_total_derivative_manual_expansion_matches_l1_minus_l2_export():
    mu = S("mu")
    c = S("c")
    phi = scalar_field("Phi", self_conjugate=True)

    expanded_divergence = Model(
        fields=(phi,),
        lagrangian_decl=(
            2 * phi * PartialD(phi, mu) * PartialD(phi, mu)
            + phi * phi * PartialD(PartialD(phi, mu), mu)
        ),
    ).lagrangian()

    l1_minus_l2 = Model(
        fields=(phi,),
        lagrangian_decl=(
            c * phi * phi * PartialD(PartialD(phi, mu), mu)
            + 2 * c * phi * PartialD(phi, mu) * PartialD(phi, mu)
        ),
    ).lagrangian()

    assert (
        l1_minus_l2.to_symbolica().expand().to_canonical_string()
        == (c * expanded_divergence.to_symbolica()).expand().to_canonical_string()
    )
