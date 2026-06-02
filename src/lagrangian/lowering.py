"""Helpers for lowering declarative Lagrangian monomials.

This module is intentionally model-agnostic: callers provide concrete factor
and term classes from the model layer.
"""

from __future__ import annotations


def expr_equal(left, right) -> bool:
    left_expr = left.expand() if hasattr(left, "expand") else left
    right_expr = right.expand() if hasattr(right, "expand") else right
    if hasattr(left_expr, "to_canonical_string") and hasattr(right_expr, "to_canonical_string"):
        return left_expr.to_canonical_string() == right_expr.to_canonical_string()
    return left_expr == right_expr


def lower_dirac_monomial(
    term,
    *,
    field_factor_cls,
    gamma_factor_cls,
    covariant_derivative_factor_cls,
    dirac_kinetic_term_cls,
    expression_module,
):
    field_factors = [factor for factor in term.factors if isinstance(factor, field_factor_cls)]
    gamma_factors = [factor for factor in term.factors if isinstance(factor, gamma_factor_cls)]
    covd_factors = [factor for factor in term.factors if isinstance(factor, covariant_derivative_factor_cls)]
    if len(term.factors) != 3 or len(field_factors) != 1 or len(gamma_factors) != 1 or len(covd_factors) != 1:
        return None

    field_factor = field_factors[0]
    gamma_factor = gamma_factors[0]
    covd_factor = covd_factors[0]
    if field_factor.field is not covd_factor.field:
        return None
    if field_factor.field.kind != "fermion":
        return None
    if not field_factor.conjugated or covd_factor.conjugated:
        return None
    if gamma_factor.lorentz_index != covd_factor.lorentz_index:
        return None

    normalized = term.coefficient / expression_module.I
    if not expr_equal(expression_module.I * normalized, term.coefficient):
        return None
    return dirac_kinetic_term_cls(
        field=field_factor.field,
        coefficient=normalized,
    )


def lower_scalar_covd_monomial(
    term,
    *,
    covariant_derivative_factor_cls,
    complex_scalar_kinetic_term_cls,
):
    covd_factors = [factor for factor in term.factors if isinstance(factor, covariant_derivative_factor_cls)]
    if len(term.factors) != 2 or len(covd_factors) != 2:
        return None
    left, right = covd_factors
    if left.field is not right.field:
        return None
    if left.field.kind != "scalar" or left.field.self_conjugate:
        return None
    if left.lorentz_index != right.lorentz_index:
        return None
    if {left.conjugated, right.conjugated} != {False, True}:
        return None
    return complex_scalar_kinetic_term_cls(
        field=left.field,
        coefficient=term.coefficient,
    )


def expand_field_strength_factor(
    *,
    gauge_field,
    abelian,
    lorentz_slot,
    adjoint_slot,
    left_index,
    right_index,
    adjoint_index,
    coupling,
    structure_constant_builder,
    fresh_adjoint_label,
    field_factor_cls,
    partial_derivative_factor_cls,
    declared_monomial_cls,
    expression_module,
):
    """Expand one ``FieldStrength(...)`` factor into additive local monomials.

    Returns a tuple of ``declared_monomial_cls`` pieces summing to the field
    strength in the convention

    ``F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + g f^{abc} A^b_mu A^c_nu``

    (abelian groups drop the structure-constant piece). Each piece is a plain
    local monomial built from ``field_factor_cls`` / ``partial_derivative_factor_cls``
    so the existing local-monomial lowering can compile it unchanged. The
    structure constant for non-abelian groups is folded into the piece
    coefficient using ``structure_constant_builder`` so the correct
    representation (e.g. color vs weak adjoint) is preserved.
    """

    def packed(lorentz_label, adjoint_label):
        slot_map = {lorentz_slot: lorentz_label}
        if adjoint_slot is not None and adjoint_label is not None:
            slot_map[adjoint_slot] = adjoint_label
        return gauge_field.pack_slot_labels(slot_map)

    one = expression_module.num(1)
    pieces = [
        declared_monomial_cls(
            coefficient=one,
            factors=(
                partial_derivative_factor_cls(
                    field=gauge_field,
                    lorentz_indices=(left_index,),
                    labels=packed(right_index, adjoint_index),
                ),
            ),
        ),
        declared_monomial_cls(
            coefficient=-one,
            factors=(
                partial_derivative_factor_cls(
                    field=gauge_field,
                    lorentz_indices=(right_index,),
                    labels=packed(left_index, adjoint_index),
                ),
            ),
        ),
    ]
    if abelian:
        return tuple(pieces)

    b_label = fresh_adjoint_label()
    c_label = fresh_adjoint_label()
    pieces.append(
        declared_monomial_cls(
            coefficient=coupling * structure_constant_builder(adjoint_index, b_label, c_label),
            factors=(
                field_factor_cls(gauge_field, labels=packed(left_index, b_label)),
                field_factor_cls(gauge_field, labels=packed(right_index, c_label)),
            ),
        )
    )
    return tuple(pieces)
