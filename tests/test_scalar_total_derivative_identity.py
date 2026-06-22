import pytest

from symbolica import S

from feynpy import Model, PartialD, dirac_field, scalar_field


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


def test_scalar_total_derivative_ibp_normal_form_vanishes_on_divergence():
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

    assert expanded_divergence.ibp_normal_form().terms == ()
    assert l1_minus_l2.ibp_normal_form().terms == ()


def test_scalar_total_derivative_ibp_normal_form_drops_one_derivative_total_derivative():
    mu = S("mu")
    phi = scalar_field("Phi", self_conjugate=True)

    lagrangian = Model(
        fields=(phi,),
        lagrangian_decl=phi * PartialD(phi, mu),
    ).lagrangian()

    normal = lagrangian.ibp_normal_form()
    assert normal.terms == ()


def test_ibp_normal_form_rejects_mixed_scalar_species():
    mu = S("mu")
    phi = scalar_field("Phi", self_conjugate=True)
    chi = scalar_field("Chi", self_conjugate=True)

    lagrangian = Model(
        fields=(phi, chi),
        lagrangian_decl=phi * PartialD(chi, mu),
    ).lagrangian()

    with pytest.raises(ValueError, match="mixed scalar species"):
        lagrangian.ibp_normal_form()


def test_ibp_normal_form_rejects_fermions():
    mu = S("mu")
    psi = dirac_field("psi", indices=())

    lagrangian = Model(
        fields=(psi,),
        lagrangian_decl=psi.bar() * PartialD(psi, mu),
    ).lagrangian()

    with pytest.raises(ValueError, match="Dirac bilinears|only scalar terms"):
        lagrangian.ibp_normal_form()
