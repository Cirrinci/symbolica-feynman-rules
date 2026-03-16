"""
CAS-driven (Symbolica/Spenso) utilities to extract interaction monomials from a
Symbolica expression and build vertices without manually writing `factors=[...]`.

Milestone scope:
- bosonic polynomial interactions
- derivative operators represented as nested `del(field, mu)` calls
- no fermionic Grassmann signs yet

This module is intentionally independent from `examples_scalar.py` and can be
used on any Lagrangian expression once fields are registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from symbolica.core import AtomType, Expression

from model import (
    Field,
    LagrangianTerm,
    OperatorFactor,
    fast_bosonic_vertex,
)


def _as_expr(x) -> Expression:
    """Convert Spenso TensorIndices (or Expression) to Expression."""
    return x.to_expression() if hasattr(x, "to_expression") else x


@dataclass(frozen=True)
class RegisteredField:
    """
    Information needed to recognize a field occurrence inside a Symbolica Expr.

    - For scalar fields (`Field.indexed=False`): occurrences are `AtomType.Var`
      with the same `.get_name()` as `field()`.
    - For indexed fields (`Field.indexed=True`): occurrences are `AtomType.Fn`
      with `.get_name()` matching the tensor head (arguments are indices).
    """

    field: Field
    atom_type: AtomType
    name: str


class FieldRegistry:
    def __init__(
        self,
        fields: Sequence[Field],
        *,
        known_slots: Sequence[object] = (),
        del_function_suffix: str = "::del",
    ):
        self.fields = list(fields)
        self.del_function_suffix = del_function_suffix
        self._slot_by_canon: Dict[str, object] = {}

        self._by_var_name: Dict[str, Field] = {}
        self._by_fn_name: Dict[str, Field] = {}

        for sl in known_slots:
            if not hasattr(sl, "to_expression"):
                continue
            se = _as_expr(sl.to_expression())
            self._slot_by_canon[se.to_canonical_string()] = sl

        for fld in self.fields:
            if fld.indexed:
                # Tensor heads appear as Fn once called with indices. The name does
                # not depend on the specific indices.
                # We register using the head's function name.
                # Use a representative call only if user has indices elsewhere;
                # for name extraction, the head itself can be called with zero
                # args in Symbolica, yielding a TensorIndices/Expression Fn.
                head_expr = _as_expr(fld.head().to_expression()) if hasattr(fld.head, "__call__") else None
                # Fallback: if the above fails, we can still register by creating a
                # simple Expression from the head name (not ideal). For now, assume
                # heads are callable (Spenso TensorName).
                if head_expr is None:
                    raise ValueError(f"Cannot register indexed field head for {fld.name}")
                if head_expr.get_type() != AtomType.Fn:
                    raise ValueError(f"Expected Fn for indexed field {fld.name}, got {head_expr.get_type()}")
                self._by_fn_name[head_expr.get_name()] = fld
            else:
                var_expr = fld()
                var_expr = _as_expr(var_expr)
                if var_expr.get_type() != AtomType.Var:
                    raise ValueError(f"Expected Var for scalar field {fld.name}, got {var_expr.get_type()}")
                self._by_var_name[var_expr.get_name()] = fld

    def match_field(self, expr: Expression) -> Optional[Tuple[Field, Tuple]]:
        """
        If expr is a field occurrence, return (Field, indices_tuple), else None.
        Indices are returned as raw Symbolica arguments (Slots, etc.).
        """
        expr = _as_expr(expr)
        t = expr.get_type()
        if t == AtomType.Var:
            fld = self._by_var_name.get(expr.get_name())
            if fld is None:
                return None
            return fld, ()
        if t == AtomType.Fn:
            fld = self._by_fn_name.get(expr.get_name())
            if fld is None:
                return None
            # Fn args are indices
            return fld, tuple(expr)
        return None

    def is_del(self, expr: Expression) -> bool:
        expr = _as_expr(expr)
        return expr.get_type() == AtomType.Fn and expr.get_name().endswith(self.del_function_suffix)

    def slot_from_expr(self, expr: Expression) -> Optional[object]:
        """
        Try to map an index expression like `mink(4,mu)` back to a Spenso Slot
        object, if it was provided in `known_slots`.
        """
        expr = _as_expr(expr)
        return self._slot_by_canon.get(expr.to_canonical_string())


def _flatten_add(expr: Expression) -> List[Expression]:
    expr = _as_expr(expr)
    if expr.get_type() == AtomType.Add:
        return [_as_expr(a) for a in expr]
    return [expr]


def _flatten_mul(expr: Expression) -> List[Expression]:
    expr = _as_expr(expr)
    if expr.get_type() == AtomType.Mul:
        return [_as_expr(a) for a in expr]
    return [expr]


def _expand_pow(expr: Expression) -> List[Expression]:
    """
    Expand simple integer powers into repeated factors.
    If power is not a positive integer literal, return [expr] unchanged.
    """
    expr = _as_expr(expr)
    if expr.get_type() != AtomType.Pow:
        return [expr]
    base, exp = list(expr)
    base = _as_expr(base)
    exp = _as_expr(exp)
    if exp.get_type() == AtomType.Num and exp.is_integer():
        n = int(str(exp))
        if n >= 0:
            return [base] * n
    return [expr]


def _unwrap_del_chain(expr: Expression, reg: FieldRegistry) -> Optional[Tuple[Expression, Tuple[Expression, ...]]]:
    """
    If expr is a nested del(...) chain, return (base_expr, derivative_index_exprs).
    Example:
      del(del(phi, nu), mu) -> (phi, (mu, nu))  in OperatorFactor order
    """
    expr = _as_expr(expr)
    ders: List[object] = []
    cur = expr
    while reg.is_del(cur):
        args = list(cur)
        if len(args) != 2:
            return None
        inner, dind = _as_expr(args[0]), _as_expr(args[1])
        ders.append(reg.slot_from_expr(dind) or dind)
        cur = inner
    if not ders:
        return None
    # In our OperatorFactor convention, derivative_indices=(mu, nu) means d_mu d_nu.
    # Our loop collected outer-to-inner, so reverse to get application order.
    return cur, tuple(reversed(ders))


def split_coefficient_and_factors(monomial: Expression, reg: FieldRegistry) -> Tuple[Expression, List[OperatorFactor]]:
    """
    Given one monomial expression, return (coefficient_expr, operator_factors).

    - Fields are detected using the registry.
    - Derivatives are detected as `del(field_expr, mu)` chains.
    - Everything else is considered part of the coefficient.
    """
    monomial = _as_expr(monomial)

    coeff: Expression | int = 1
    factors: List[OperatorFactor] = []

    # Break product into factors; expand simple powers.
    raw_factors: List[Expression] = []
    for f in _flatten_mul(monomial):
        raw_factors.extend(_expand_pow(f))

    for f in raw_factors:
        f = _as_expr(f)

        # Derivative-wrapped field?
        unwrapped = _unwrap_del_chain(f, reg)
        if unwrapped is not None:
            base, ders = unwrapped
            m = reg.match_field(base)
            if m is not None:
                fld, idx = m
                factors.append(OperatorFactor(fld, indices=idx, derivative_indices=ders, conjugated=False))
                continue

        # Plain field?
        m = reg.match_field(f)
        if m is not None:
            fld, idx = m
            factors.append(OperatorFactor(fld, indices=idx, conjugated=False))
            continue

        # Not a field -> part of coefficient
        coeff = coeff * f

    return _as_expr(coeff), factors


def lagrangian_terms_from_expr(
    L_expr: Expression,
    reg: FieldRegistry,
    *,
    name_prefix: str = "auto",
) -> List[LagrangianTerm]:
    """
    Expand a Lagrangian expression into monomials and convert each to a LagrangianTerm
    with automatically inferred `coefficient` and `factors`.
    """
    L_expr = _as_expr(L_expr).expand()
    monomials = _flatten_add(L_expr)

    out: List[LagrangianTerm] = []
    for k, mono in enumerate(monomials):
        coeff, factors = split_coefficient_and_factors(mono, reg)
        if not factors:
            continue
        out.append(
            LagrangianTerm(
                name=f"{name_prefix}_{k}",
                expr=mono,
                fields=tuple(f.short_name() for f in factors),
                coefficient=coeff,
                factors=factors,
            )
        )
    return out


def cas_vertex_bosonic(
    L_expr: Expression,
    reg: FieldRegistry,
    external_legs,
    *,
    name_prefix: str = "auto",
) -> Expression | None:
    """
    Convenience: build auto terms from L_expr and sum their bosonic vertices for given legs.

    This mimics `canonical_vertices_from_lagrangian` but works directly from a CAS expression.
    """
    total = 0
    for term in lagrangian_terms_from_expr(L_expr, reg, name_prefix=name_prefix):
        v = fast_bosonic_vertex(term, external_legs)
        if v is not None:
            total = total + v
    return total if total != 0 else None

