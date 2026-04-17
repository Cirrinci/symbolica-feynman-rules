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


def lower_field_strength_monomial(
    term,
    *,
    field_strength_factor_cls,
    gauge_kinetic_term_cls,
    expression_module,
):
    fs_factors = [factor for factor in term.factors if isinstance(factor, field_strength_factor_cls)]
    if len(term.factors) != 2 or len(fs_factors) != 2:
        return None
    left, right = fs_factors
    if left.gauge_group != right.gauge_group:
        return None
    if left.left_index != right.left_index or left.right_index != right.right_index:
        return None

    normalized = -expression_module.num(4) * term.coefficient
    return gauge_kinetic_term_cls(
        gauge_group=left.gauge_group,
        coefficient=normalized,
    )

