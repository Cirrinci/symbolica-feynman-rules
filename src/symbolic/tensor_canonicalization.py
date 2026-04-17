"""
Generic tensor canonicalization helpers.

This module is intentionally separate from the gauge compiler:

- it provides a generic tensor-head canonicalization pass
- it does not know any physics-specific compact identities

The intended layering is:

1. canonize tensor heads and dummy indices here
2. apply gauge-/physics-specific compact rewrites elsewhere when desired
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from symbolica import Expression, S
from .spenso_structures import lorentz_metric

pcomp = S("pcomp")


@dataclass(frozen=True)
class TensorHeadSpec:
    """One tensor head mapping used during canonicalization.

    ``raw_name`` is the head that appears in the live expression.
    ``canonical_name`` is a temporary head with extra symmetry metadata.
    """

    raw_name: str
    canonical_name: str
    arity: int
    head_kwargs: dict

    @property
    def raw_head(self):
        return S(self.raw_name)

    @property
    def canonical_head(self):
        return S(self.canonical_name, **self.head_kwargs)


SPENSO_TENSOR_HEAD_SPECS = (
    TensorHeadSpec(
        raw_name="spenso::g",
        canonical_name="canon::g",
        arity=2,
        head_kwargs={"is_symmetric": True},
    ),
    TensorHeadSpec(
        raw_name="spenso::t",
        canonical_name="canon::t",
        arity=3,
        head_kwargs={},
    ),
    TensorHeadSpec(
        raw_name="spenso::f",
        canonical_name="canon::f",
        arity=3,
        head_kwargs={"is_antisymmetric": True},
    ),
)


def _wildcards(arity: int):
    return tuple(S(*(f"canon_arg_{arity}_{slot}_" for slot in range(arity))))


def _replace_tensor_heads(expr, specs: Sequence[TensorHeadSpec], *, reverse: bool = False):
    """Swap live tensor heads for canonical temporary heads, or back again."""
    result = expr
    for spec in specs:
        args = _wildcards(spec.arity)
        source = spec.canonical_head if reverse else spec.raw_head
        target = spec.raw_head if reverse else spec.canonical_head
        result = result.replace(source(*args), target(*args))
    return result


def _dummy_name(group, slot: int) -> str:
    group_text = str(group).replace(" ", "_").replace(":", "_")
    return f"canon_dummy_{group_text}_{slot}"


def _standardize_dummy_indices(expr, dummy_indices):
    """Rename dummy indices deterministically for stable comparisons."""
    result = expr
    standardized = []
    for slot, (index, group) in enumerate(dummy_indices, start=1):
        target = S(_dummy_name(group, slot))
        result = result.replace(index, target)
        standardized.append((target, group))
    return result, standardized


def canonize_tensor_expression(
    expr,
    *,
    contracted_indices: Sequence[tuple[object, object]],
    head_specs: Sequence[TensorHeadSpec] = SPENSO_TENSOR_HEAD_SPECS,
    standardize_dummy_names: bool = True,
):
    """Canonize tensor heads and dummy indices in an expression.

    The returned expression is mapped back onto the original live tensor names.
    This lets the rest of the code stay in the usual Spenso naming scheme while
    still using Symbolica's tensor canonization machinery internally.
    """

    remapped = _replace_tensor_heads(expr, head_specs, reverse=False)
    canonical_expr, external_indices, dummy_indices = remapped.canonize_tensors(contracted_indices)
    if standardize_dummy_names:
        canonical_expr, dummy_indices = _standardize_dummy_indices(canonical_expr, dummy_indices)
    canonical_expr = _replace_tensor_heads(canonical_expr, head_specs, reverse=True)
    return canonical_expr, external_indices, dummy_indices


def canonize_spenso_tensors(
    expr,
    *,
    lorentz_indices: Iterable[object] = (),
    adjoint_indices: Iterable[object] = (),
    color_fund_indices: Iterable[object] = (),
    spinor_indices: Iterable[object] = (),
    extra_index_groups: Iterable[tuple[object, object]] = (),
    standardize_dummy_names: bool = True,
):
    """Canonize the Spenso tensor structures used in this repository.

    Index groups are kept explicit so that dummy indices from different
    representations cannot be merged accidentally.  This is the main helper
    used by the current regression checks when gauge-heavy outputs should be
    compared modulo dummy-index renaming.
    """

    contracted_indices = []
    contracted_indices.extend((idx, 0) for idx in lorentz_indices)
    contracted_indices.extend((idx, 1) for idx in adjoint_indices)
    contracted_indices.extend((idx, 2) for idx in color_fund_indices)
    contracted_indices.extend((idx, 3) for idx in spinor_indices)
    contracted_indices.extend(extra_index_groups)

    return canonize_tensor_expression(
        expr,
        contracted_indices=contracted_indices,
        standardize_dummy_names=standardize_dummy_names,
    )


def contract_spenso_lorentz_metrics(expr, *, max_passes: int = 8):
    """Contract explicit Lorentz metrics against momenta and other metrics.

    This is a lightweight local simplification pass for the Spenso tensor
    structures used in the gauge-sector outputs. It intentionally does not try
    to perform full tensor algebra; it only contracts patterns of the form

    - ``g(mu, nu) * pcomp(p, nu) -> pcomp(p, mu)``
    - ``g(mu, nu) * g(rho, nu) -> g(mu, rho)``

    including the symmetric slot/order variants that appear in expanded
    Lagrangian-API vertices.
    """

    wildcard_a = S("canon_lorentz_a_")
    wildcard_b = S("canon_lorentz_b_")
    wildcard_c = S("canon_lorentz_c_")
    wildcard_p = S("canon_momentum_")

    rewrites = (
        (lorentz_metric(wildcard_a, wildcard_b) * pcomp(wildcard_p, wildcard_b), pcomp(wildcard_p, wildcard_a)),
        (pcomp(wildcard_p, wildcard_b) * lorentz_metric(wildcard_a, wildcard_b), pcomp(wildcard_p, wildcard_a)),
        (lorentz_metric(wildcard_a, wildcard_b) * pcomp(wildcard_p, wildcard_a), pcomp(wildcard_p, wildcard_b)),
        (pcomp(wildcard_p, wildcard_a) * lorentz_metric(wildcard_a, wildcard_b), pcomp(wildcard_p, wildcard_b)),
        (lorentz_metric(wildcard_a, wildcard_b) * lorentz_metric(wildcard_c, wildcard_b), lorentz_metric(wildcard_a, wildcard_c)),
        (lorentz_metric(wildcard_c, wildcard_b) * lorentz_metric(wildcard_a, wildcard_b), lorentz_metric(wildcard_a, wildcard_c)),
        (lorentz_metric(wildcard_a, wildcard_b) * lorentz_metric(wildcard_a, wildcard_c), lorentz_metric(wildcard_b, wildcard_c)),
        (lorentz_metric(wildcard_a, wildcard_c) * lorentz_metric(wildcard_a, wildcard_b), lorentz_metric(wildcard_b, wildcard_c)),
    )

    result = expr.expand() if hasattr(expr, "expand") else expr
    for _ in range(max_passes):
        previous = (
            result.to_canonical_string()
            if hasattr(result, "to_canonical_string")
            else str(result)
        )
        for pattern, replacement in rewrites:
            result = result.replace(pattern, replacement)
        if hasattr(result, "expand"):
            result = result.expand()
        current = (
            result.to_canonical_string()
            if hasattr(result, "to_canonical_string")
            else str(result)
        )
        if current == previous:
            break
    return result
