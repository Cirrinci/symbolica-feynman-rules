"""
Tensor canonicalization helpers for Symbolica/Spenso expressions.

The module exposes two layers:

- a generic tensor-head canonicalization pass (`canonize_spenso_tensors`)
- optional physics-aware passes used by `canonize_full` (commuting flat
  partial derivatives, local Jacobi reduction in the ``f`` basis, and
  Yang-Mills-specific antisymmetric zero-term detection)

The tensor canonicalization works by *swapping* every Spenso tensor head for a
plain Symbolica symbol that carries the right ``is_symmetric`` /
``is_antisymmetric`` attribute, running ``Expression.canonize_tensors`` on the
result, and swapping back.  This gives Symbolica direct access to tensor
symmetry metadata while keeping expressions in the usual Spenso naming scheme
elsewhere in the pipeline.
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Iterable, Mapping, Optional, Sequence

from symbolica import AtomType, Expression, S

pcomp = S("pcomp")
_PARTIAL_DERIVATIVE_NAME = S("PartialD").get_name()
_REPRESENTATION_ANSI = re.compile(r"\x1b\[[0-9;]*m")

LORENTZ_KIND = "lorentz"
COLOR_FUND_KIND = "color_fund"
COLOR_ADJ_KIND = "color_adj"
SPINOR_KIND = "spinor"
WEAK_FUND_KIND = "weak_fund"
WEAK_ADJ_KIND = "weak_adj"

_DUMMY_GROUP_STEMS = {
    0: "mu_mid",
    1: "a_mid",
    2: "c_mid",
    3: "i_mid",
    4: "w_mid",
    5: "aw_mid",
}

_PLAIN_HEAD_SLOT_KINDS: dict[str, dict[int, str]] = {
    # Exported gauge-variation heads used in the current YM pipeline.
    "G": {0: LORENTZ_KIND, 1: "adjoint"},
    "alpha": {0: "adjoint"},
    # Compact spinor projectors produced by the transformation post-process.
    "PL": {
        0: SPINOR_KIND,
        1: SPINOR_KIND,
    },
    "PR": {
        0: SPINOR_KIND,
        1: SPINOR_KIND,
    },
    # Dirac tensors exported by Spenso.
    "spenso::gamma": {
        0: SPINOR_KIND,
        1: SPINOR_KIND,
        2: LORENTZ_KIND,
    },
    "spenso::gamma5": {
        0: SPINOR_KIND,
        1: SPINOR_KIND,
    },
}

_PLAIN_HEAD_INDEX_KINDS = (
    LORENTZ_KIND,
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    SPINOR_KIND,
    WEAK_FUND_KIND,
    WEAK_ADJ_KIND,
)

_SPENSO_SLOT_KIND_BY_SIGNATURE = {
    ("spenso::mink", "4"): LORENTZ_KIND,
    ("spenso::coad", "8"): COLOR_ADJ_KIND,
    ("spenso::cof", "3"): COLOR_FUND_KIND,
    ("spenso::bis", "4"): SPINOR_KIND,
    ("spenso::cof", "2"): WEAK_FUND_KIND,
    ("spenso::coad", "3"): WEAK_ADJ_KIND,
}

_ADJOINT_ALIAS_KIND = "adjoint"


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


# The Spenso built-in tensors live in the ``spenso::`` namespace, while
# tensors created from Python via ``TensorName("...")`` land in
# ``spenso_python::``.  Both are matched by the swap pattern below.
SPENSO_TENSOR_HEAD_SPECS = (
    # Built-in HEP heads.
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
    TensorHeadSpec(
        raw_name="spenso::gamma",
        canonical_name="canon::gamma",
        arity=3,
        head_kwargs={},
    ),
    TensorHeadSpec(
        raw_name="spenso::gamma5",
        canonical_name="canon::gamma5",
        arity=2,
        head_kwargs={},
    ),
    # Project-defined invariant tensors.  See ``spenso_structures`` for the
    # corresponding ``TensorName`` declarations.  We register these here so
    # Symbolica knows their symmetry during ``canonize_tensors``.
    TensorHeadSpec(
        raw_name="spenso_python::weak_eps2",
        canonical_name="canon::weak_eps2",
        arity=2,
        head_kwargs={"is_antisymmetric": True},
    ),
    TensorHeadSpec(
        raw_name="spenso_python::lor_levi_civita",
        canonical_name="canon::lor_levi_civita",
        arity=4,
        head_kwargs={"is_antisymmetric": True},
    ),
    TensorHeadSpec(
        raw_name="spenso_python::color_eps3",
        canonical_name="canon::color_eps3",
        arity=3,
        head_kwargs={"is_antisymmetric": True},
    ),
    TensorHeadSpec(
        raw_name="spenso_python::color_d",
        canonical_name="canon::color_d",
        arity=3,
        head_kwargs={"is_symmetric": True},
    ),
    TensorHeadSpec(
        raw_name="spenso_python::dirac_C",
        canonical_name="canon::dirac_C",
        arity=2,
        head_kwargs={"is_antisymmetric": True},
    ),
)

_TYPED_TENSOR_HEAD_NAMES = frozenset(
    spec.raw_name for spec in SPENSO_TENSOR_HEAD_SPECS
)


def _wildcards(arity: int):
    return tuple(S(*(f"canon_arg_{arity}_{slot}_" for slot in range(arity))))


def _symbol_name_text(symbol) -> str:
    if isinstance(symbol, Expression):
        return symbol.get_name()
    return str(symbol)


def _plain_slot_kinds_compatible(left: str, right: str) -> bool:
    if left == right:
        return True
    return {left, right} <= {_ADJOINT_ALIAS_KIND, COLOR_ADJ_KIND, WEAK_ADJ_KIND}


def _merge_plain_slot_kind(left: str, right: str) -> str:
    if left == right:
        return left
    if left == _ADJOINT_ALIAS_KIND:
        return right
    if right == _ADJOINT_ALIAS_KIND:
        return left
    raise ValueError(
        f"Conflicting plain-head slot kinds {left!r} and {right!r}."
    )


def _merge_plain_head_slot_kind_maps(
    *mappings: dict[str, dict[int, str]],
) -> dict[str, dict[int, str]]:
    merged: dict[str, dict[int, str]] = {}
    for mapping in mappings:
        for head_name, slot_kinds in mapping.items():
            current = dict(merged.get(head_name, {}))
            for slot, kind in slot_kinds.items():
                existing = current.get(slot)
                if existing is None:
                    current[slot] = kind
                    continue
                if not _plain_slot_kinds_compatible(existing, kind):
                    raise ValueError(
                        f"Plain head {head_name!r} has incompatible slot-kind metadata "
                        f"for slot {slot + 1}: {existing!r} vs {kind!r}."
                    )
                current[slot] = _merge_plain_slot_kind(existing, kind)
            merged[head_name] = current
    return merged


def _validate_field_head(field: object) -> None:
    """Validate the structural field metadata used by this symbolic layer.

    Importing ``feynpy.metadata.Field`` here would create a dependency cycle:
    the metadata layer itself depends on the symbolic tensor definitions.  The
    canonicalizer only needs this small, stable interface, so validate that
    interface directly instead of depending on the concrete class.
    """

    required = ("indices", "self_conjugate", "species_for", "statistics")
    missing = tuple(name for name in required if not hasattr(field, name))
    if missing:
        details = ", ".join(missing)
        raise TypeError(
            "field_heads must contain Field-compatible metadata objects; "
            f"missing attribute(s): {details}."
        )


def _canonical_plain_index_kind(index: object) -> Optional[str]:
    """Map field metadata onto one canonicalization group when possible."""

    kind = getattr(index, "kind", None)
    if kind in _PLAIN_HEAD_INDEX_KINDS:
        return kind

    # Custom names for Lorentz and spinor indices are supported throughout the
    # model API.  Their Spenso representation families are unambiguous, unlike
    # generic fundamental/adjoint representations, so they can be normalized
    # safely here as well.
    representation = _REPRESENTATION_ANSI.sub(
        "", str(getattr(index, "representation", ""))
    )
    if representation.startswith("mink("):
        return LORENTZ_KIND
    if representation.startswith("bis("):
        return SPINOR_KIND
    return kind


def _field_plain_head_slot_kinds(field_heads: Iterable[object]) -> dict[str, dict[int, str]]:
    mapping: dict[str, dict[int, str]] = {}
    for field in field_heads:
        _validate_field_head(field)
        slot_kinds = {}
        for slot, index in enumerate(field.indices):
            kind = _canonical_plain_index_kind(index)
            if kind is not None:
                slot_kinds[slot] = kind
        head_names = {_symbol_name_text(field.species_for(False))}
        if not field.self_conjugate:
            head_names.add(_symbol_name_text(field.species_for(True)))
        field_mapping = {
            head_name: dict(slot_kinds)
            for head_name in head_names
        }
        mapping = _merge_plain_head_slot_kind_maps(mapping, field_mapping)
    return mapping


def _plain_head_slot_kinds(*, field_heads: Iterable[object] = ()) -> dict[str, dict[int, str]]:
    return _merge_plain_head_slot_kind_maps(
        _PLAIN_HEAD_SLOT_KINDS,
        _field_plain_head_slot_kinds(field_heads),
    )


def _field_plain_odd_head_names(field_heads: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for field in field_heads:
        _validate_field_head(field)
        if field.statistics != "fermion":
            continue
        names.add(_symbol_name_text(field.species_for(False)))
        if not field.self_conjugate:
            names.add(_symbol_name_text(field.species_for(True)))
    return names


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
    if group in _DUMMY_GROUP_STEMS:
        stem = _DUMMY_GROUP_STEMS[group]
        return stem if slot == 1 else f"{stem}_{slot}"
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
    canonical_expr = _canonicalize_antisymmetric_tensor_arguments(
        canonical_expr,
        head_specs,
    )
    return canonical_expr, external_indices, dummy_indices


def canonize_spenso_tensors(
    expr,
    *,
    lorentz_indices: Iterable[object] = (),
    adjoint_indices: Iterable[object] = (),
    color_fund_indices: Iterable[object] = (),
    spinor_indices: Iterable[object] = (),
    weak_fund_indices: Iterable[object] = (),
    weak_adj_indices: Iterable[object] = (),
    extra_index_groups: Iterable[tuple[object, object]] = (),
    standardize_dummy_names: bool = True,
):
    """Canonize the Spenso tensor structures used in this repository.

    Index groups are kept explicit so that dummy indices from different
    representations cannot be merged accidentally.  This is the main helper
    used by the current regression checks when gauge-heavy outputs should be
    compared modulo dummy-index renaming.

    Each kwarg corresponds to one representation kind (Lorentz, color
    adjoint, color fundamental, spinor, weak fundamental, weak adjoint).
    Indices coming from different kinds are placed in distinct groups so
    that ``canonize_tensors`` never collapses, e.g., a Lorentz dummy with a
    color dummy.  Anything that does not fit the standard catalogue can be
    routed through ``extra_index_groups`` as ``(index, group_marker)``
    pairs.
    """

    contracted_indices = []
    contracted_indices.extend((idx, 0) for idx in lorentz_indices)
    contracted_indices.extend((idx, 1) for idx in adjoint_indices)
    contracted_indices.extend((idx, 2) for idx in color_fund_indices)
    contracted_indices.extend((idx, 3) for idx in spinor_indices)
    contracted_indices.extend((idx, 4) for idx in weak_fund_indices)
    contracted_indices.extend((idx, 5) for idx in weak_adj_indices)
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
    from .spenso_structures import lorentz_metric

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


def _contract_plain_metric_heads(
    expr,
    *,
    field_heads: Iterable[object] = (),
    max_passes: int = 8,
):
    """Contract Lorentz/adjoint metrics against plain exported field heads.

    The Symbolica export uses ordinary commutative function heads such as
    ``G(mu, a)``, ``alpha(a)``, and ``PartialD(...)``.  ``simplify_metrics``
    already contracts typed Spenso tensors like ``f`` and ``t``, but it does
    not know that these plain heads carry Lorentz/adjoint labels in specific
    argument slots.  This lightweight pass covers the patterns that appear in
    the lowered gauge-variation expressions without introducing a separate
    field-aware tensor system.
    """

    wildcard_a = S("canon_plain_metric_a_")
    wildcard_b = S("canon_plain_metric_b_")
    wildcard_x = S("canon_plain_metric_x_")
    wildcard_y = S("canon_plain_metric_y_")
    partial = S("PartialD")
    from .spenso_structures import COLOR_ADJ, LORENTZ, WEAK_ADJ
    metrics_by_kind = {
        LORENTZ_KIND: LORENTZ.g(wildcard_a, wildcard_b).to_expression(),
        COLOR_ADJ_KIND: COLOR_ADJ.g(wildcard_a, wildcard_b).to_expression(),
        WEAK_ADJ_KIND: WEAK_ADJ.g(wildcard_a, wildcard_b).to_expression(),
    }
    plain_head_slot_kinds = _plain_head_slot_kinds(field_heads=field_heads)

    rewrites = []

    # Ordinary partial derivatives carry one Lorentz derivative slot by design.
    lorentz_metric = metrics_by_kind[LORENTZ_KIND]
    rewrites.extend(
        [
            (lorentz_metric * partial(wildcard_x, wildcard_b), partial(wildcard_x, wildcard_a)),
            (partial(wildcard_x, wildcard_b) * lorentz_metric, partial(wildcard_x, wildcard_a)),
            (
                lorentz_metric * partial(partial(wildcard_x, wildcard_b), wildcard_y),
                partial(partial(wildcard_x, wildcard_a), wildcard_y),
            ),
            (
                partial(partial(wildcard_x, wildcard_b), wildcard_y) * lorentz_metric,
                partial(partial(wildcard_x, wildcard_a), wildcard_y),
            ),
            (
                lorentz_metric * partial(partial(wildcard_x, wildcard_y), wildcard_b),
                partial(partial(wildcard_x, wildcard_y), wildcard_a),
            ),
            (
                partial(partial(wildcard_x, wildcard_y), wildcard_b) * lorentz_metric,
                partial(partial(wildcard_x, wildcard_y), wildcard_a),
            ),
        ]
    )

    for head_name, slot_kinds in plain_head_slot_kinds.items():
        if not slot_kinds:
            continue
        head = S(head_name)
        arity = max(slot_kinds) + 1
        head_args = [S(f"canon_plain_metric_head_{head_name}_{slot}_") for slot in range(arity)]
        for slot, kind in slot_kinds.items():
            if kind == _ADJOINT_ALIAS_KIND:
                metrics = [
                    metrics_by_kind[COLOR_ADJ_KIND],
                    metrics_by_kind[WEAK_ADJ_KIND],
                ]
            else:
                metric = metrics_by_kind.get(kind)
                if metric is None:
                    continue
                metrics = [metric]
            for metric in metrics:
                args_before = list(head_args)
                args_before[slot] = wildcard_b
                args_after = list(head_args)
                args_after[slot] = wildcard_a
                base_before = head(*args_before)
                base_after = head(*args_after)
                rewrites.extend(
                    [
                        (metric * base_before, base_after),
                        (base_before * metric, base_after),
                        (metric * partial(base_before, wildcard_x), partial(base_after, wildcard_x)),
                        (partial(base_before, wildcard_x) * metric, partial(base_after, wildcard_x)),
                        (
                            metric * partial(partial(base_before, wildcard_x), wildcard_y),
                            partial(partial(base_after, wildcard_x), wildcard_y),
                        ),
                        (
                            partial(partial(base_before, wildcard_x), wildcard_y) * metric,
                            partial(partial(base_after, wildcard_x), wildcard_y),
                        ),
                    ]
                )

                # Cover the common "derivative index outermost" nesting:
                # metric * PartialD(PartialD(H(...), x), b) -> ...a
                rewrites.extend(
                    [
                        (
                            metric * partial(partial(base_before, wildcard_x), wildcard_b),
                            partial(partial(base_after, wildcard_x), wildcard_a),
                        ),
                        (
                            partial(partial(base_before, wildcard_x), wildcard_b) * metric,
                            partial(partial(base_after, wildcard_x), wildcard_a),
                        ),
                    ]
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


def _plain_odd_factor_key(atom: Expression, odd_head_names: set[str]) -> Optional[str]:
    if not isinstance(atom, Expression):
        return None
    atom_type = atom.get_type()
    if atom_type == AtomType.Var:
        return atom.to_canonical_string() if atom.get_name() in odd_head_names else None
    if atom_type == AtomType.Pow:
        base, exponent = tuple(atom)
        base_key = _plain_odd_factor_key(base, odd_head_names)
        if base_key is None:
            return None
        exponent_text = exponent.to_canonical_string()
        try:
            exponent_value = int(exponent_text)
        except ValueError:
            return None
        return atom.to_canonical_string() if exponent_value >= 2 else base_key
    if atom_type != AtomType.Fn:
        return None

    name = atom.get_name()
    bare_name = name.rsplit("::", 1)[-1]
    if bare_name == "PartialD":
        args = tuple(atom)
        if len(args) != 2:
            return None
        return atom.to_canonical_string() if _plain_odd_factor_key(args[0], odd_head_names) is not None else None

    if name in odd_head_names or bare_name in odd_head_names:
        return atom.to_canonical_string()
    return None


def _drop_plain_odd_squares(
    expr,
    *,
    field_heads: Iterable[object] = (),
):
    """Drop monomials containing repeated identical plain odd factors.

    The Symbolica export is commutative, so it cannot encode Grassmann
    nilpotency on its own.  For the BRST checks we only need a narrow local
    repair: if a monomial contains the *same* exported odd field factor
    twice (including identical ``PartialD`` wrappers), that monomial is zero.
    """

    odd_head_names = _field_plain_odd_head_names(field_heads)
    if not odd_head_names:
        return expr

    terms = tuple(expr) if isinstance(expr, Expression) and expr.get_type() == AtomType.Add else (expr,)
    kept_terms: list[Expression] = []
    for term in terms:
        seen: set[str] = set()
        zero_term = False
        for factor in _term_factors(term):
            if isinstance(factor, Expression) and factor.get_type() == AtomType.Pow:
                base, exponent = tuple(factor)
                if _plain_odd_factor_key(base, odd_head_names) is not None:
                    try:
                        exponent_value = int(exponent.to_canonical_string())
                    except ValueError:
                        exponent_value = None
                    if exponent_value is not None and exponent_value >= 2:
                        zero_term = True
                        break
            factor_key = _plain_odd_factor_key(factor, odd_head_names)
            if factor_key is None:
                continue
            if factor_key in seen:
                zero_term = True
                break
            seen.add(factor_key)
        if not zero_term:
            kept_terms.append(term)

    total = Expression.num(0)
    for term in kept_terms:
        total += term
    return total.expand() if hasattr(total, "expand") else total


def _build_function_expression(name: str, args: Sequence[Expression]) -> Expression:
    """Rebuild a function atom from its fully qualified head name and args."""

    if not args:
        return Expression.parse(name)
    args_text = ",".join(arg.to_canonical_string() for arg in args)
    return Expression.parse(f"{name}({args_text})")


def _canonicalize_commuting_partial_derivatives(expr):
    """Sort nested ``PartialD`` chains so flat partial derivatives commute.

    This turns, e.g., ``PartialD(PartialD(alpha(a), mu), nu)`` into the same
    canonical nested call as ``PartialD(PartialD(alpha(a), nu), mu)``.  The
    pass is intentionally local: it only reorders repeated ``PartialD`` calls
    on the same subexpression and leaves the rest of the term structure alone.
    """

    atom_type = expr.get_type() if isinstance(expr, Expression) else None
    if atom_type == AtomType.Add:
        total = Expression.num(0)
        for term in expr:
            total += _canonicalize_commuting_partial_derivatives(term)
        return total
    if atom_type == AtomType.Mul:
        total = Expression.num(1)
        for factor in expr:
            total *= _canonicalize_commuting_partial_derivatives(factor)
        return total
    if atom_type != AtomType.Fn:
        return expr

    args = tuple(_canonicalize_commuting_partial_derivatives(arg) for arg in expr)
    name = expr.get_name()
    if name != _PARTIAL_DERIVATIVE_NAME:
        return _build_function_expression(name, args)

    if len(args) != 2:
        # Keep malformed / non-standard PartialD calls untouched.
        return _build_function_expression(name, args)

    base, derivative_index = args
    derivative_indices = [derivative_index]
    while (
        isinstance(base, Expression)
        and base.get_type() == AtomType.Fn
        and base.get_name() == _PARTIAL_DERIVATIVE_NAME
    ):
        inner_base, inner_index = tuple(base)
        base = inner_base
        derivative_indices.append(inner_index)

    result = base
    partial = S("PartialD")
    for ordered_index in sorted(
        derivative_indices,
        key=lambda item: item.to_canonical_string(),
    ):
        result = partial(result, ordered_index)
    return result


def _term_factors(term) -> tuple[Expression, ...]:
    if isinstance(term, Expression) and term.get_type() == AtomType.Mul:
        return tuple(term)
    return (term,)


def _terms(expression) -> tuple[Expression, ...]:
    if not isinstance(expression, Expression):
        return (expression,)
    if _expression_is_zero(expression):
        return ()
    if expression.get_type() == AtomType.Add:
        return tuple(expression)
    return (expression,)


def _term_structure_constants(term) -> tuple[Expression, ...]:
    return tuple(
        factor
        for factor in _term_factors(term)
        if isinstance(factor, Expression)
        and factor.get_type() == AtomType.Fn
        and factor.get_name() == "spenso::f"
    )


def _swap_symbols(expr, swaps: Sequence[tuple[Expression, Expression]]):
    """Swap several symbol pairs in one expression using temporary placeholders."""

    result = expr
    placeholders: list[tuple[Expression, Expression]] = []
    for slot, (left, right) in enumerate(swaps):
        placeholder = S(f"canon_swap_placeholder_{slot}")
        result = result.replace(left, placeholder)
        result = result.replace(right, left)
        placeholders.append((placeholder, right))
    for placeholder, right in placeholders:
        result = result.replace(placeholder, right)
    return result


def _present_indices(term, index_pool: Sequence[object]) -> tuple[Expression, ...]:
    symbols = tuple(term.get_all_symbols())
    return tuple(
        index
        for index in index_pool
        if any(symbol == index for symbol in symbols)
    )


def _perm_sign(current: Sequence[Expression], target: Sequence[Expression]) -> int:
    working = list(current)
    sign = 1
    for slot, desired in enumerate(target):
        current_slot = working.index(desired, slot)
        while current_slot > slot:
            working[current_slot], working[current_slot - 1] = (
                working[current_slot - 1],
                working[current_slot],
            )
            sign *= -1
            current_slot -= 1
    return sign


def _canonicalize_antisymmetric_tensor_arguments(
    expr,
    specs: Sequence[TensorHeadSpec],
):
    """Give antisymmetric tensor heads a process-independent slot order.

    Symbolica's tensor canonicalizer uses the symbol interning order when it
    chooses between equivalent external-index orientations.  That order
    depends on which tests or models created a symbol first, so an expression
    could canonize to either ``eps(i, j)`` or ``-eps(j, i)`` across runs.
    Reordering the live tensor heads lexically after ``canonize_tensors`` keeps
    the algebraic sign while making the returned representation deterministic.
    """

    antisymmetric_arities = {
        spec.raw_name: spec.arity
        for spec in specs
        if spec.head_kwargs.get("is_antisymmetric") is True
    }

    def canonicalize(atom):
        if not isinstance(atom, Expression):
            return atom
        atom_type = atom.get_type()
        if atom_type == AtomType.Add:
            total = Expression.num(0)
            for term in atom:
                total += canonicalize(term)
            return total
        if atom_type == AtomType.Mul:
            total = Expression.num(1)
            for factor in atom:
                total *= canonicalize(factor)
            return total
        if atom_type == AtomType.Pow:
            base, exponent = tuple(atom)
            return canonicalize(base) ** canonicalize(exponent)
        if atom_type != AtomType.Fn:
            return atom

        name = atom.get_name()
        args = tuple(canonicalize(arg) for arg in atom)
        arity = antisymmetric_arities.get(name)
        if arity is None or len(args) != arity:
            return _build_function_expression(name, args)

        keys = tuple(arg.to_canonical_string() for arg in args)
        if len(set(keys)) != len(keys):
            return Expression.num(0)
        ordered = tuple(
            arg
            for _key, arg in sorted(
                zip(keys, args),
                key=lambda item: item[0],
            )
        )
        sign = _perm_sign(args, ordered)
        return sign * _build_function_expression(name, ordered)

    return canonicalize(expr)


def _rebuild_function_like(template: Expression, args: Sequence[Expression]) -> Expression:
    return _build_function_expression(template.get_name(), args)


def _shared_structure_constant_index(pair: tuple[Expression, Expression], rest: Expression):
    left, right = pair
    left_args = tuple(left)
    right_args = tuple(right)
    shared = [arg for arg in left_args if any(arg == candidate for candidate in right_args)]
    if len(shared) != 1:
        return None
    candidate = shared[0]
    if candidate.to_canonical_string() in rest.to_canonical_string():
        return None
    return candidate


def _jacobi_reduce_structure_constant_products(expr):
    """Reduce ``f f`` products to a deterministic Jacobi basis.

    The standard Jacobi identity,

        ``f(a,b,e)f(c,d,e) - f(a,c,e)f(b,d,e) + f(a,d,e)f(b,c,e) = 0``,

    spans three commutative ``f f`` pairings of the same four free adjoint
    indices and one shared contracted adjoint index.  We use it to eliminate
    the middle pairing in favour of a fixed two-element basis.  This keeps the
    color structure in the ``f`` basis instead of expanding to generators.
    """

    terms = tuple(expr) if isinstance(expr, Expression) and expr.get_type() == AtomType.Add else (expr,)
    untouched: list[Expression] = []
    groups: dict[tuple[str, tuple[str, ...], str], dict] = {}

    for term in terms:
        structure_constants = _term_structure_constants(term)
        selected_pair = None
        for left_slot, left in enumerate(structure_constants):
            for right in structure_constants[left_slot + 1:]:
                rest = term.coefficient(left * right)
                shared = _shared_structure_constant_index((left, right), rest)
                if shared is None:
                    continue
                selected_pair = (left, right, shared, rest)
                break
            if selected_pair is not None:
                break

        if selected_pair is None:
            untouched.append(term)
            continue

        left, right, shared, rest = selected_pair
        left_free = tuple(arg for arg in left if arg != shared)
        right_free = tuple(arg for arg in right if arg != shared)
        free = tuple(
            sorted(
                left_free + right_free,
                key=lambda item: item.to_canonical_string(),
            )
        )
        if len({item.to_canonical_string() for item in free}) != 4:
            untouched.append(term)
            continue

        a, b, c, d = free
        targets = {
            "p12": ((a, b, shared), (c, d, shared)),
            "p13": ((a, c, shared), (b, d, shared)),
            "p14": ((a, d, shared), (b, c, shared)),
        }

        actual_pair_sets = [
            frozenset(item.to_canonical_string() for item in left_free),
            frozenset(item.to_canonical_string() for item in right_free),
        ]

        kind = None
        sign = None
        for name, (target_left, target_right) in targets.items():
            left_pair = frozenset(item.to_canonical_string() for item in target_left[:2])
            right_pair = frozenset(item.to_canonical_string() for item in target_right[:2])
            if actual_pair_sets == [left_pair, right_pair]:
                kind = name
                sign = _perm_sign(tuple(left), target_left) * _perm_sign(tuple(right), target_right)
                break
            if actual_pair_sets == [right_pair, left_pair]:
                kind = name
                sign = _perm_sign(tuple(left), target_right) * _perm_sign(tuple(right), target_left)
                break

        if kind is None:
            untouched.append(term)
            continue

        key = (
            rest.to_canonical_string(),
            tuple(item.to_canonical_string() for item in free),
            shared.to_canonical_string(),
        )
        entry = groups.setdefault(
            key,
            {
                "rest": rest,
                "free": free,
                "shared": shared,
                "coefficients": {
                    "p12": Expression.num(0),
                    "p13": Expression.num(0),
                    "p14": Expression.num(0),
                },
                "templates": (left, right),
            },
        )
        entry["coefficients"][kind] += sign

    total = Expression.num(0)
    for term in untouched:
        total += term

    for entry in groups.values():
        a, b, c, d = entry["free"]
        shared = entry["shared"]
        rest = entry["rest"]
        left_template, right_template = entry["templates"]
        coefficients = entry["coefficients"]

        pairing_12 = _rebuild_function_like(left_template, (a, b, shared)) * _rebuild_function_like(
            right_template, (c, d, shared)
        )
        pairing_14 = _rebuild_function_like(left_template, (a, d, shared)) * _rebuild_function_like(
            right_template, (b, c, shared)
        )

        # Jacobi: p12 - p13 + p14 = 0  =>  p13 = p12 + p14.
        # Eliminating p13 gives:
        #   c12*p12 + c13*p13 + c14*p14
        # = (c12 + c13)*p12 + (c14 + c13)*p14.
        reduced_coeff_12 = coefficients["p12"] + coefficients["p13"]
        reduced_coeff_14 = coefficients["p14"] + coefficients["p13"]
        total += rest * (
            reduced_coeff_12 * pairing_12
            + reduced_coeff_14 * pairing_14
        )

    return total.expand() if hasattr(total, "expand") else total


def _canonize_term_for_swap_check(
    expr,
    *,
    lorentz_indices: Sequence[object],
    adjoint_indices: Sequence[object],
    color_fund_indices: Sequence[object],
    spinor_indices: Sequence[object],
    weak_fund_indices: Sequence[object],
    weak_adj_indices: Sequence[object],
    extra_index_groups: Sequence[tuple[object, object]],
):
    normalized = _canonicalize_commuting_partial_derivatives(expr)
    canonical, external_indices, dummy_indices = canonize_spenso_tensors(
        normalized,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )
    canonical = canonical.expand() if hasattr(canonical, "expand") else canonical
    return canonical, tuple(external_indices), tuple(dummy_indices)


def _contains_external_symbol(expr: Expression, external_names: set[str]) -> bool:
    return any(
        symbol.to_canonical_string() in external_names
        for symbol in expr.get_all_symbols()
    )


def _external_symbol_names(external_indices) -> set[str]:
    names: set[str] = set()
    for item in external_indices:
        symbol = item[0] if isinstance(item, tuple) else item
        if hasattr(symbol, "to_canonical_string"):
            names.add(symbol.to_canonical_string())
    return names


def _append_symbol_group(
    target: dict[str, Expression],
    atom: Expression,
):
    for symbol in atom.get_all_symbols():
        target.setdefault(symbol.to_canonical_string(), symbol)


def _typed_spenso_slot_kind_and_label(
    atom: Expression,
) -> Optional[tuple[str, Expression]]:
    """Return the canonical kind and label encoded by one Spenso slot.

    Spenso tensor arguments carry their representation explicitly, for example
    ``coad(8, a)`` for an SU(3) adjoint slot and ``coad(3, aw)`` for an SU(2)
    adjoint slot.  Reading that metadata is both more general and safer than
    assigning a kind from the outer tensor head (``f``, ``t``, ``g``, ...).
    Wrapper functions such as a lowered/dual index are traversed recursively.
    """

    if not isinstance(atom, Expression) or atom.get_type() != AtomType.Fn:
        return None

    name = atom.get_name()
    args = tuple(atom)
    if len(args) >= 2:
        dimension = args[0].to_canonical_string()
        kind = _SPENSO_SLOT_KIND_BY_SIGNATURE.get((name, dimension))
        if kind is not None:
            return kind, args[-1]

    for child in args:
        typed = _typed_spenso_slot_kind_and_label(child)
        if typed is not None:
            return typed
    return None


def _inferred_groups_without_explicit_labels(
    inferred: dict[str, tuple[Expression, ...]],
    *,
    explicit_groups: Sequence[Sequence[object]],
    extra_index_groups: Sequence[tuple[object, object]],
) -> dict[str, tuple[Expression, ...]]:
    """Make explicit index declarations authoritative over inference."""

    explicit_names = {
        item.to_canonical_string()
        for group in explicit_groups
        for item in group
        if hasattr(item, "to_canonical_string")
    }
    explicit_names.update(
        item.to_canonical_string()
        for item, _ in extra_index_groups
        if hasattr(item, "to_canonical_string")
    )

    filtered = {
        kind: tuple(
            item
            for item in items
            if item.to_canonical_string() not in explicit_names
        )
        for kind, items in inferred.items()
    }

    assignments: dict[str, set[str]] = {}
    for kind, items in filtered.items():
        for item in items:
            key = item.to_canonical_string()
            assignments.setdefault(key, set()).add(kind)

    # The exported mixed-gauge pipeline may reuse one bare Symbolica label in
    # slots with different typed representations. ``canonize_tensors`` accepts
    # only one group marker per bare label, so choosing either representation
    # would be incorrect. Conservatively leave such labels uninferred; their
    # typed slots remain intact, and a caller can assign a group explicitly if
    # the label is known to be local to one representation in their expression.
    ambiguous_names = {
        key for key, kinds in assignments.items() if len(kinds) > 1
    }
    if not ambiguous_names:
        return filtered
    return {
        kind: tuple(
            item
            for item in items
            if item.to_canonical_string() not in ambiguous_names
        )
        for kind, items in filtered.items()
    }


def _infer_index_groups_from_expression(
    expr,
    *,
    field_heads: Iterable[object] = (),
):
    """Infer index pools from typed tensors and exported field metadata."""

    groups = {
        LORENTZ_KIND: {},
        COLOR_ADJ_KIND: {},
        COLOR_FUND_KIND: {},
        SPINOR_KIND: {},
        WEAK_FUND_KIND: {},
        WEAK_ADJ_KIND: {},
    }
    ambiguous_adjoint: dict[str, Expression] = {}
    plain_head_slot_kinds = _plain_head_slot_kinds(field_heads=field_heads)

    def append(kind: str, atom: Expression):
        if kind == _ADJOINT_ALIAS_KIND:
            _append_symbol_group(ambiguous_adjoint, atom)
            return
        target = groups.get(kind)
        if target is None:
            return
        _append_symbol_group(target, atom)

    def walk(atom):
        if not isinstance(atom, Expression):
            return
        atom_type = atom.get_type()
        if atom_type == AtomType.Fn:
            name = atom.get_name()
            args = tuple(atom)
            bare_name = name.rsplit("::", 1)[-1]
            if bare_name == "PartialD" and len(args) == 2:
                append(LORENTZ_KIND, args[1])
            elif name in _TYPED_TENSOR_HEAD_NAMES:
                for arg in args:
                    typed = _typed_spenso_slot_kind_and_label(arg)
                    if typed is not None:
                        kind, label = typed
                        append(kind, label)

            slot_kinds = plain_head_slot_kinds.get(name)
            if slot_kinds is None:
                slot_kinds = plain_head_slot_kinds.get(bare_name)
            if slot_kinds is not None:
                for slot, kind in slot_kinds.items():
                    if slot < len(args):
                        append(kind, args[slot])

        if atom_type in (AtomType.Add, AtomType.Mul, AtomType.Fn):
            for child in atom:
                walk(child)

    walk(expr)

    # Plain ``G``/``alpha`` heads historically use an ``adjoint`` alias
    # because their function arguments carry no representation metadata. Match
    # those labels to concrete typed slots when the surrounding expression
    # provides them. If it does not, retain the historical SU(3) fallback.
    for key, item in ambiguous_adjoint.items():
        if (
            key not in groups[COLOR_ADJ_KIND]
            and key not in groups[WEAK_ADJ_KIND]
        ):
            groups[COLOR_ADJ_KIND].setdefault(key, item)

    return {
        kind: tuple(group[key] for key in sorted(group))
        for kind, group in groups.items()
    }


def _merge_index_groups(explicit: Sequence[object], inferred: Sequence[object]) -> tuple[object, ...]:
    merged: dict[str, object] = {}
    for item in explicit:
        if hasattr(item, "to_canonical_string"):
            merged[item.to_canonical_string()] = item
    for item in inferred:
        if hasattr(item, "to_canonical_string"):
            merged.setdefault(item.to_canonical_string(), item)
    return tuple(merged[key] for key in sorted(merged))


def _drop_yang_mills_antisymmetric_zero_terms(
    expr,
    *,
    lorentz_indices: Sequence[object],
    adjoint_indices: Sequence[object],
    color_fund_indices: Sequence[object],
    spinor_indices: Sequence[object],
    weak_fund_indices: Sequence[object],
    weak_adj_indices: Sequence[object],
    extra_index_groups: Sequence[tuple[object, object]],
):
    """Drop terms that are odd under legal YM dummy-index relabelings.

    This pass is intentionally conservative: swap checks are only attempted on
    symbols that canonization has identified as dummy labels. This avoids
    accidentally dropping terms that are odd under swaps involving free/external
    indices.
    """

    terms = (
        tuple(expr)
        if isinstance(expr, Expression) and expr.get_type() == AtomType.Add
        else (expr,)
    )
    kept_terms: list[Expression] = []

    for term in terms:
        canonical_term, external_indices, _dummy_indices = _canonize_term_for_swap_check(
            term,
            lorentz_indices=lorentz_indices,
            adjoint_indices=adjoint_indices,
            color_fund_indices=color_fund_indices,
            spinor_indices=spinor_indices,
            weak_fund_indices=weak_fund_indices,
            weak_adj_indices=weak_adj_indices,
            extra_index_groups=extra_index_groups,
        )
        external_names = _external_symbol_names(external_indices)

        zero_by_swap = False
        present_lorentz = _present_indices(canonical_term, tuple(lorentz_indices))
        lorentz_swaps = [
            pair
            for pair in combinations(present_lorentz, 2)
            if pair[0].to_canonical_string() not in external_names
            and pair[1].to_canonical_string() not in external_names
        ]
        lorentz_swaps = [()] + lorentz_swaps

        for structure_constant in _term_structure_constants(canonical_term):
            args = tuple(structure_constant)
            for left_slot, right_slot in combinations(range(len(args)), 2):
                left = args[left_slot]
                right = args[right_slot]
                if (
                    _contains_external_symbol(left, external_names)
                    or _contains_external_symbol(right, external_names)
                ):
                    continue

                for lorentz_swap in lorentz_swaps:
                    swaps = [(left, right)]
                    if lorentz_swap:
                        swaps.append(lorentz_swap)
                    swapped = _swap_symbols(canonical_term, swaps)
                    swapped_canonical, _, _ = _canonize_term_for_swap_check(
                        swapped,
                        lorentz_indices=lorentz_indices,
                        adjoint_indices=adjoint_indices,
                        color_fund_indices=color_fund_indices,
                        spinor_indices=spinor_indices,
                        weak_fund_indices=weak_fund_indices,
                        weak_adj_indices=weak_adj_indices,
                        extra_index_groups=extra_index_groups,
                    )
                    trial = (
                        canonical_term + swapped_canonical
                    ).expand()
                    if trial.to_canonical_string() == "0":
                        zero_by_swap = True
                        break
                if zero_by_swap:
                    break
            if zero_by_swap:
                break

        if not zero_by_swap:
            kept_terms.append(canonical_term)

    total = Expression.num(0)
    for term in kept_terms:
        total += term
    return total.expand() if hasattr(total, "expand") else total


def _drop_zero_terms_from_antisymmetric_structure_constants(
    expr,
    *,
    lorentz_indices: Sequence[object],
    adjoint_indices: Sequence[object],
    color_fund_indices: Sequence[object],
    spinor_indices: Sequence[object],
    weak_fund_indices: Sequence[object],
    weak_adj_indices: Sequence[object],
    extra_index_groups: Sequence[tuple[object, object]],
):
    """Backward-compatible wrapper for YM antisymmetric zero-term dropping."""

    return _drop_yang_mills_antisymmetric_zero_terms(
        expr,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )


@dataclass(frozen=True)
class CanonicalTensorMonomial:
    """Canonical key for one tensor/momentum monomial."""

    commuting_factors: tuple[object, ...]
    ordered_factors: tuple[object, ...] = ()


@dataclass(frozen=True)
class CanonicalMonomialReport:
    """Raw/canonical term counts plus the canonical coefficient map."""

    raw_terms: int
    canonical_terms: int
    map: Mapping[CanonicalTensorMonomial, Expression]


class TensorMonomialCanonicalizationError(ValueError):
    """Raised when exact dummy relabeling would be too expensive."""


_CANONICAL_INDEX_GROUP_KEYS = {
    LORENTZ_KIND: "L",
    COLOR_ADJ_KIND: "A",
    COLOR_FUND_KIND: "C",
    SPINOR_KIND: "S",
    WEAK_FUND_KIND: "W",
    WEAK_ADJ_KIND: "AW",
}


def canonical_external_index_set(
    *,
    lorentz: Iterable[object] = (),
    color_adjoint: Iterable[object] = (),
    color_fund: Iterable[object] = (),
    spinor: Iterable[object] = (),
    weak_fund: Iterable[object] = (),
    weak_adjoint: Iterable[object] = (),
) -> frozenset[tuple[str, str]]:
    """Build fixed external-index labels for canonical monomial maps."""

    groups = (
        (LORENTZ_KIND, lorentz),
        (COLOR_ADJ_KIND, color_adjoint),
        (COLOR_FUND_KIND, color_fund),
        (SPINOR_KIND, spinor),
        (WEAK_FUND_KIND, weak_fund),
        (WEAK_ADJ_KIND, weak_adjoint),
    )
    return frozenset(
        (kind, _canonical_symbol_label(index))
        for kind, indices in groups
        for index in indices
    )


def canonical_tensor_monomial_map(
    expression: Expression,
    *,
    coefficient: object | None = None,
    external_indices: Iterable[tuple[str, object]] = (),
    tensor_head_specs: Sequence[TensorHeadSpec] = SPENSO_TENSOR_HEAD_SPECS,
    noncommuting_heads: Iterable[str] = (),
    max_dummy_permutations: int = 50_000,
) -> dict[CanonicalTensorMonomial, Expression]:
    """Return ``canonical monomial -> exact coefficient``.

    The normalization uses only intrinsic tensor symmetries, global dummy-index
    relabeling, deterministic ordering of commuting factors, and exact
    coefficient collection. It intentionally does not apply Jacobi identities,
    momentum conservation, Schouten identities, equations of motion, integration
    by parts, Fierz identities, or dimension-specific reductions.
    """

    if coefficient is not None:
        coefficient_expr = S(coefficient) if isinstance(coefficient, str) else coefficient
        expression = expression.coefficient(coefficient_expr)

    tensor_specs = {spec.raw_name: spec for spec in tensor_head_specs}
    noncommuting = {_canonical_head_name(head) for head in noncommuting_heads}
    fixed_indices = {
        (kind, _canonical_symbol_label(index))
        for kind, index in external_indices
    }

    collected: dict[CanonicalTensorMonomial, Expression] = {}
    for term in _terms(expression.cancel().expand()):
        key, coeff = _canonicalize_tensor_monomial_term(
            term,
            tensor_specs=tensor_specs,
            fixed_indices=fixed_indices,
            noncommuting_heads=noncommuting,
            max_dummy_permutations=max_dummy_permutations,
        )
        if key is None or _expression_is_zero(coeff):
            continue
        collected[key] = (
            collected.get(key, Expression.num(0)) + coeff
        ).cancel().expand()

    return {
        key: coeff
        for key, coeff in sorted(collected.items(), key=lambda item: repr(item[0]))
        if not _expression_is_zero(coeff)
    }


def canonical_tensor_monomial_report(
    expression: Expression,
    **kwargs,
) -> CanonicalMonomialReport:
    """Return raw and canonical term counts with the canonical map."""

    if kwargs.get("coefficient") is not None:
        coefficient = kwargs["coefficient"]
        coefficient_expr = S(coefficient) if isinstance(coefficient, str) else coefficient
        counted_expression = expression.coefficient(coefficient_expr)
    else:
        counted_expression = expression
    counted_expression = counted_expression.cancel().expand()
    monomial_map = canonical_tensor_monomial_map(expression, **kwargs)
    return CanonicalMonomialReport(
        raw_terms=len(_terms(counted_expression)),
        canonical_terms=len(monomial_map),
        map=monomial_map,
    )


def _expression_is_zero(expression: Expression) -> bool:
    return expression.cancel().expand().to_canonical_string() == "0"


def _canonical_head_name(head: object) -> str:
    if isinstance(head, Expression):
        text = head.get_name()
    else:
        text = str(head)
    return text.rsplit("::", 1)[-1]


def _canonical_symbol_label(symbol: object) -> str:
    text = symbol.to_canonical_string() if isinstance(symbol, Expression) else str(symbol)
    return (
        text.replace("python::{}::", "")
        .replace("python::", "")
        .replace("spenso::", "")
        .replace("spenso_python::{}::", "")
    )


def _canonical_slot_kind_label(atom: Expression) -> tuple[str, str] | None:
    typed = _typed_spenso_slot_kind_and_label(atom)
    if typed is None:
        return None
    kind, label = typed
    return kind, _canonical_symbol_label(label)


def _factor_index_occurrences(factor: Expression) -> tuple[tuple[str, str], ...]:
    if not isinstance(factor, Expression):
        return ()

    atom_type = factor.get_type()
    if atom_type == AtomType.Fn:
        name = factor.get_name()
        args = tuple(factor)
        if _canonical_head_name(name) == "pcomp" and len(args) == 2:
            return ((LORENTZ_KIND, _canonical_symbol_label(args[1])),)

        occurrences: list[tuple[str, str]] = []
        for arg in args:
            slot = _canonical_slot_kind_label(arg)
            if slot is not None:
                occurrences.append(slot)
            else:
                occurrences.extend(_factor_index_occurrences(arg))
        return tuple(occurrences)

    if atom_type in (AtomType.Add, AtomType.Mul, AtomType.Pow):
        return tuple(
            occurrence
            for child in factor
            for occurrence in _factor_index_occurrences(child)
        )
    return ()


def _is_canonical_scalar_factor(factor: Expression) -> bool:
    if factor.get_type() in (AtomType.Num, AtomType.Var):
        return True
    return not _factor_index_occurrences(factor)


def _is_ordered_monomial_factor(
    factor: Expression,
    noncommuting_heads: set[str],
) -> bool:
    return (
        isinstance(factor, Expression)
        and factor.get_type() == AtomType.Fn
        and _canonical_head_name(factor.get_name()) in noncommuting_heads
    )


def _canonicalize_tensor_monomial_term(
    term: Expression,
    *,
    tensor_specs: Mapping[str, TensorHeadSpec],
    fixed_indices: set[tuple[str, str]],
    noncommuting_heads: set[str],
    max_dummy_permutations: int,
) -> tuple[CanonicalTensorMonomial | None, Expression]:
    coefficient = Expression.num(1)
    scalar_factors: list[Expression] = []
    tensor_factors: list[Expression] = []

    for factor in _term_factors(term):
        if factor.get_type() == AtomType.Num:
            coefficient *= factor
        elif (
            not _is_ordered_monomial_factor(factor, noncommuting_heads)
            and _is_canonical_scalar_factor(factor)
        ):
            scalar_factors.append(factor)
        else:
            tensor_factors.append(factor)

    index_counts = Counter(
        occurrence
        for factor in tensor_factors
        for occurrence in _factor_index_occurrences(factor)
    )
    dummy_indices: dict[str, list[str]] = defaultdict(list)
    for index, count in index_counts.items():
        kind, label = index
        if index in fixed_indices or count == 1:
            continue
        dummy_indices[kind].append(label)

    candidates = _dummy_relabeling_candidates(
        dummy_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    all_indices = set(index_counts)
    best_key: CanonicalTensorMonomial | None = None
    best_key_text = ""
    best_signs: list[int] = []

    for candidate in candidates:
        index_map = {
            index: candidate.get(index, _canonical_external_index_key(index))
            for index in all_indices
        }
        sign = 1
        commuting: list[object] = []
        ordered: list[object] = []
        zero = False

        for factor in tensor_factors:
            factor_sign, factor_key = _canonical_monomial_factor_key(
                factor,
                index_map=index_map,
                tensor_specs=tensor_specs,
            )
            if factor_sign == 0:
                zero = True
                break
            sign *= factor_sign
            if _is_ordered_monomial_factor(factor, noncommuting_heads):
                ordered.append(factor_key)
            else:
                commuting.append(factor_key)

        if zero:
            continue

        key = CanonicalTensorMonomial(
            commuting_factors=tuple(sorted(commuting, key=repr)),
            ordered_factors=tuple(ordered),
        )
        key_text = repr(key)
        if best_key is None or key_text < best_key_text:
            best_key = key
            best_key_text = key_text
            best_signs = [sign]
        elif key_text == best_key_text:
            best_signs.append(sign)

    if best_key is None:
        return None, Expression.num(0)

    if 1 in best_signs and -1 in best_signs:
        return None, Expression.num(0)

    for scalar in sorted(
        scalar_factors,
        key=lambda item: item.to_canonical_string(),
    ):
        coefficient *= scalar
    if best_signs and best_signs[0] < 0:
        coefficient *= -1
    return best_key, coefficient.cancel().expand()


def _dummy_relabeling_candidates(
    dummy_indices: Mapping[str, Sequence[str]],
    *,
    max_dummy_permutations: int,
) -> tuple[dict[tuple[str, str], str], ...]:
    total = 1
    for labels in dummy_indices.values():
        total *= math.factorial(len(set(labels)))
    if total > max_dummy_permutations:
        raise TensorMonomialCanonicalizationError(
            "Canonical dummy relabeling would require "
            f"{total} permutations, exceeding max_dummy_permutations="
            f"{max_dummy_permutations}."
        )

    candidates: list[dict[tuple[str, str], str]] = [{}]
    for kind, labels in sorted(dummy_indices.items()):
        unique_labels = sorted(set(labels))
        group = _CANONICAL_INDEX_GROUP_KEYS.get(kind, kind)
        expanded: list[dict[tuple[str, str], str]] = []
        for permutation in permutations(range(1, len(unique_labels) + 1)):
            relabeling = {
                (kind, label): f"D:{group}:{slot}"
                for label, slot in zip(unique_labels, permutation)
            }
            for candidate in candidates:
                merged = dict(candidate)
                merged.update(relabeling)
                expanded.append(merged)
        candidates = expanded
    return tuple(candidates)


def _canonical_external_index_key(index: tuple[str, str]) -> str:
    kind, label = index
    group = _CANONICAL_INDEX_GROUP_KEYS.get(kind, kind)
    return f"E:{group}:{label}"


def _canonical_monomial_factor_key(
    factor: Expression,
    *,
    index_map: Mapping[tuple[str, str], str],
    tensor_specs: Mapping[str, TensorHeadSpec],
) -> tuple[int, object]:
    atom_type = factor.get_type()
    if atom_type == AtomType.Fn:
        name = factor.get_name()
        bare_name = _canonical_head_name(name)
        args = tuple(factor)
        if bare_name == "pcomp" and len(args) == 2:
            momentum = _canonical_symbol_label(args[0])
            lorentz_index = (LORENTZ_KIND, _canonical_symbol_label(args[1]))
            return 1, ("pcomp", momentum, index_map[lorentz_index])

        spec = tensor_specs.get(name)
        if spec is not None and len(args) == spec.arity:
            sign, canonical_args = _canonical_tensor_monomial_arguments(
                args,
                spec=spec,
                index_map=index_map,
            )
            if sign == 0:
                return 0, None
            return sign, (
                bare_name,
                _tensor_representation_signature(args),
                canonical_args,
            )

        return 1, (
            bare_name,
            tuple(
                _canonical_generic_monomial_argument(arg, index_map=index_map)
                for arg in args
            ),
        )

    if atom_type == AtomType.Pow:
        base, exponent = tuple(factor)
        sign, base_key = _canonical_monomial_factor_key(
            base,
            index_map=index_map,
            tensor_specs=tensor_specs,
        )
        if sign == 0:
            return 0, None
        return sign, ("pow", base_key, _canonical_symbol_label(exponent))

    return 1, ("raw", _canonical_symbol_label(factor))


def _canonical_tensor_monomial_arguments(
    args: Sequence[Expression],
    *,
    spec: TensorHeadSpec,
    index_map: Mapping[tuple[str, str], str],
) -> tuple[int, tuple[object, ...]]:
    canonical_args = tuple(
        _canonical_generic_monomial_argument(arg, index_map=index_map)
        for arg in args
    )

    if spec.head_kwargs.get("is_symmetric") is True:
        return 1, tuple(sorted(canonical_args))

    if spec.head_kwargs.get("is_antisymmetric") is True:
        if len(set(canonical_args)) != len(canonical_args):
            return 0, ()
        ordered = tuple(sorted(canonical_args))
        return _perm_sign(canonical_args, ordered), ordered

    return 1, canonical_args


def _canonical_generic_monomial_argument(
    arg: Expression,
    *,
    index_map: Mapping[tuple[str, str], str],
) -> object:
    slot = _canonical_slot_kind_label(arg)
    if slot is not None:
        return index_map[slot]

    if isinstance(arg, Expression) and arg.get_type() == AtomType.Fn:
        return (
            _canonical_head_name(arg.get_name()),
            tuple(
                _canonical_generic_monomial_argument(child, index_map=index_map)
                for child in arg
            ),
        )
    return _canonical_symbol_label(arg)


def _tensor_representation_signature(args: Sequence[Expression]) -> tuple[str, ...]:
    signature = []
    for arg in args:
        slot = _canonical_slot_kind_label(arg)
        signature.append(slot[0] if slot is not None else "?")
    return tuple(signature)


# ---------------------------------------------------------------------------
# One-shot orchestrator
# ---------------------------------------------------------------------------


def canonize_full(
    expr,
    *,
    lorentz_indices: Iterable[object] = (),
    adjoint_indices: Iterable[object] = (),
    color_fund_indices: Iterable[object] = (),
    spinor_indices: Iterable[object] = (),
    weak_fund_indices: Iterable[object] = (),
    weak_adj_indices: Iterable[object] = (),
    extra_index_groups: Iterable[tuple[object, object]] = (),
    run_gamma: bool = True,
    run_color: bool = False,
    run_commuting_partial_derivatives: bool = True,
    run_jacobi_reduction: bool = True,
    run_yang_mills_antisymmetric_zero_drop: bool = True,
    infer_indices: bool = True,
    field_heads: Iterable[object] = (),
):
    """One-call simplification + canonicalisation.

    Threads the idenso simplification chain (``simplify_invariants`` from
    ``spenso_structures``), the project's own metric/momentum contraction
    pass, optional commuting-partial normalization, optional local Jacobi
    reduction, optional Yang-Mills antisymmetry zero dropping, and the
    symmetry-aware tensor canonicalisation in a single deterministic order.
    Returns just the canonical expression (the dummy bookkeeping returned by
    :func:`canonize_spenso_tensors` is dropped here for ergonomic reasons).

    By default ``run_color=False`` keeps expressions in the structure-constant
    basis so the local Jacobi reducer can act on explicit ``f f`` products
    instead of letting ``simplify_color`` expand them into generators first.

    By default ``infer_indices=True`` reads representation metadata from typed
    Spenso tensor slots and from concrete ``Field`` objects supplied through
    ``field_heads``. Explicit index arguments always take precedence. This
    lets callers canonicalize exported matter / ghost expressions without
    spelling out every dummy label explicitly. Disable inference when working
    with an unsupported or intentionally nonstandard index convention.
    """
    lorentz_indices = tuple(lorentz_indices)
    adjoint_indices = tuple(adjoint_indices)
    color_fund_indices = tuple(color_fund_indices)
    spinor_indices = tuple(spinor_indices)
    weak_fund_indices = tuple(weak_fund_indices)
    weak_adj_indices = tuple(weak_adj_indices)
    extra_index_groups = tuple(extra_index_groups)
    field_heads = tuple(field_heads)

    from .spenso_structures import simplify_invariants

    expr = simplify_invariants(expr, run_gamma=run_gamma, run_color=run_color)
    expr = contract_spenso_lorentz_metrics(expr)
    expr = _contract_plain_metric_heads(expr, field_heads=field_heads)
    expr = _drop_plain_odd_squares(expr, field_heads=field_heads)
    if infer_indices:
        inferred = _infer_index_groups_from_expression(
            expr,
            field_heads=field_heads,
        )
        inferred = _inferred_groups_without_explicit_labels(
            inferred,
            explicit_groups=(
                lorentz_indices,
                adjoint_indices,
                color_fund_indices,
                spinor_indices,
                weak_fund_indices,
                weak_adj_indices,
            ),
            extra_index_groups=extra_index_groups,
        )
        lorentz_indices = _merge_index_groups(lorentz_indices, inferred[LORENTZ_KIND])
        adjoint_indices = _merge_index_groups(adjoint_indices, inferred[COLOR_ADJ_KIND])
        color_fund_indices = _merge_index_groups(
            color_fund_indices,
            inferred[COLOR_FUND_KIND],
        )
        spinor_indices = _merge_index_groups(spinor_indices, inferred[SPINOR_KIND])
        weak_fund_indices = _merge_index_groups(
            weak_fund_indices,
            inferred[WEAK_FUND_KIND],
        )
        weak_adj_indices = _merge_index_groups(
            weak_adj_indices,
            inferred[WEAK_ADJ_KIND],
        )

    canonical, _, _ = canonize_spenso_tensors(
        expr,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )
    reduced = canonical
    if run_jacobi_reduction:
        reduced = _jacobi_reduce_structure_constant_products(reduced)
    if run_commuting_partial_derivatives:
        reduced = _canonicalize_commuting_partial_derivatives(reduced)
    if hasattr(reduced, "expand"):
        reduced = reduced.expand()
    reduced = _drop_plain_odd_squares(reduced, field_heads=field_heads)
    canonical, _, _ = canonize_spenso_tensors(
        reduced,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )
    reduced = canonical
    if run_yang_mills_antisymmetric_zero_drop:
        reduced = _drop_yang_mills_antisymmetric_zero_terms(
            canonical,
            lorentz_indices=lorentz_indices,
            adjoint_indices=adjoint_indices,
            color_fund_indices=color_fund_indices,
            spinor_indices=spinor_indices,
            weak_fund_indices=weak_fund_indices,
            weak_adj_indices=weak_adj_indices,
            extra_index_groups=extra_index_groups,
        )
    canonical, _, _ = canonize_spenso_tensors(
        reduced,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )
    return canonical


def canonize_structure_constant_products(
    expr,
    *,
    lorentz_indices: Iterable[object] = (),
    adjoint_indices: Iterable[object] = (),
    color_fund_indices: Iterable[object] = (),
    spinor_indices: Iterable[object] = (),
    weak_fund_indices: Iterable[object] = (),
    weak_adj_indices: Iterable[object] = (),
    extra_index_groups: Iterable[tuple[object, object]] = (),
):
    """Canonize commutative tensor products built from ``f`` / ``g`` factors.

    This is the comparison layer used when matching gauge-sector outputs to
    external tools such as FeynRules:

    - each antisymmetric structure constant is canonized with the correct sign
    - Lorentz metrics are canonized using their symmetry
    - contracted dummy indices are renamed deterministically to readable typed
      stems such as ``a_mid`` or ``mu_mid``
    - commutative products are sorted and like terms are collected

    Unlike :func:`canonize_full`, this helper does not run the Yang-Mills
    derivative/Jacobi/antisymmetry-specific passes. It is intended for direct
    comparison of fully expanded tensor expressions.
    """

    expanded = expr.expand() if hasattr(expr, "expand") else expr
    canonical, _, _ = canonize_spenso_tensors(
        expanded,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_fund_indices,
        spinor_indices=spinor_indices,
        weak_fund_indices=weak_fund_indices,
        weak_adj_indices=weak_adj_indices,
        extra_index_groups=extra_index_groups,
    )
    return canonical.expand() if hasattr(canonical, "expand") else canonical
