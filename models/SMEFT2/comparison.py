"""Regenerate the SMEFT2 FeynRules/FeynPy comparison artifacts.

The FeynRules reference JSON is a full tensor-rule export. This script performs
the reproducible comparison currently supported for SMEFT2: signature coverage,
coefficient-head content, and raw coefficient-head multiplicity diagnostics
after normalizing field names to the FeynRules convention. It also exports the
local FeynPy vertex rules so individual rows can be inspected against the
reference JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feynrules.comparison import (
    CanonicalCoefficientComparison,
    FeynRulesVertex,
    compare_feynrules_bosonic_vertices,
    compare_canonical_coefficient_maps,
    load_feynrules_json,
)
from models.SMEFT2 import build_smeft_green_bpreserving
from symbolic.tensor_canonicalization import (
    CanonicalMonomialReport,
    CanonicalTensorMonomial,
    canonical_external_index_set,
    canonical_tensor_monomial_report,
    canonical_tensor_monomial_map,
)
from symbolica import AtomType, Expression, S

from symbolic.spenso_structures import (
    COLOR_ADJ,
    COLOR_FUND,
    WEAK_ADJ,
    WEAK_FUND,
    gamma_matrix,
    gauge_generator,
    dirac_charge_conjugation,
    lorentz_levi_civita,
    lorentz_metric,
    spinor_metric,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I, pcomp


MODEL_DIR = Path(__file__).resolve().parent
REFERENCE = MODEL_DIR / "reference" / "Ltot_SMEFT_FeynRules.json"
FEYNPY_VERTICES = MODEL_DIR / "feynpy_vertices.json"
COMPARISON_JSON = MODEL_DIR / "vertex_comparison_report.json"
COMPARISON_MD = MODEL_DIR / "COMPARISON.md"
WEINBERG_VERTICES = MODEL_DIR / "weinberg_vertices.json"
WEINBERG_COMPARISON_JSON = MODEL_DIR / "weinberg_comparison_report.json"

FIELD_NAME_MAP = {
    "LL": "lL",
    "LL.bar": "lLbar",
    "LR": "eR",
    "LR.bar": "eRbar",
    "QL": "qL",
    "QL.bar": "qLbar",
    "UR": "uR",
    "UR.bar": "uRbar",
    "DR": "dR",
    "DR.bar": "dRbar",
    "Phi": "Phi",
    "Phi.bar": "Phibar",
    "B": "B",
    "Wi": "Wi",
    "G": "G",
}

GENERIC_PARAMETER_NAMES = frozenset({"g1", "g2", "g3", "muH", "lam", "yl", "yu", "yd"})

OMITTED_COEFFICIENT_HEADS = frozenset()

REFERENCE_FERMION_NAMES = frozenset(
    {
        "qL",
        "qLbar",
        "uR",
        "uRbar",
        "dR",
        "dRbar",
        "lL",
        "lLbar",
        "eR",
        "eRbar",
    }
)

DUAL_FS_ANTISYMMETRY = "DUAL_FS_ANTISYMMETRY"
DUMMY_LORENTZ_MERGE = "DUMMY_LORENTZ_MERGE"
EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE = "EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE"
PINNED_CC_CANONICAL_EQUIVALENCE = "PINNED_CC_CANONICAL_EQUIVALENCE"

BENIGN_HEAD_COUNT_REASON_TEXT = {
    DUAL_FS_ANTISYMMETRY: (
        "FeynPy prints the two antisymmetric branches from "
        "`Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already "
        "collapsed them with epsilon antisymmetry."
    ),
    DUMMY_LORENTZ_MERGE: (
        "FeynPy leaves the two `alphaRqD` derivative-order branches as "
        "separate dummy-Lorentz contractions; FeynRules merges the identical "
        "contraction into one term with a doubled coefficient."
    ),
    EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE: (
        "The direct exact symbolic comparison proves canonical tensor-map "
        "equality for this row; the raw occurrence-count difference is a "
        "printer/expansion multiplicity, not an operator-content mismatch."
    ),
    PINNED_CC_CANONICAL_EQUIVALENCE: (
        "The pinned charge-conjugation packaging comparison proves canonical "
        "tensor-map equality for this row; the literal-signature raw count "
        "differs because the same operator is packaged under the CC partner."
    ),
}

BENIGN_HEAD_COUNT_DELTAS = {
    ("B|Phi|qL|uRbar", "alphaEuB"): DUAL_FS_ANTISYMMETRY,
    ("B|Phibar|qLbar|uR", "alphaEuB"): DUAL_FS_ANTISYMMETRY,
    ("B|Phi|dR|qLbar", "alphaEdB"): DUAL_FS_ANTISYMMETRY,
    ("B|Phibar|dRbar|qL", "alphaEdB"): DUAL_FS_ANTISYMMETRY,
    ("B|Phi|eR|lLbar", "alphaEeB"): DUAL_FS_ANTISYMMETRY,
    ("B|Phibar|eRbar|lL", "alphaEeB"): DUAL_FS_ANTISYMMETRY,
    ("B|qL|qLbar", "alphaEBq"): DUAL_FS_ANTISYMMETRY,
    ("B|qL|qLbar", "alphaEBqtp"): DUAL_FS_ANTISYMMETRY,
    ("B|qL|qLbar", "alphaRBqtp"): DUAL_FS_ANTISYMMETRY,
    ("B|qL|qLbar", "alphaRqD"): DUMMY_LORENTZ_MERGE,
    ("B|qL|qLbar", "g1"): DUMMY_LORENTZ_MERGE,
    ("G|qL|qLbar", "alphaRqD"): DUMMY_LORENTZ_MERGE,
    ("G|qL|qLbar", "g3"): DUMMY_LORENTZ_MERGE,
    ("Wi|qL|qLbar", "alphaRqD"): DUMMY_LORENTZ_MERGE,
    ("Wi|qL|qLbar", "g2"): DUMMY_LORENTZ_MERGE,
}

CANONICAL_EXTERNAL_INDEX_GROUP_BY_KIND = {
    "lorentz": "lorentz",
    "color_adj": "color_adjoint",
    "color_fund": "color_fund",
    "spinor": "spinor",
    "weak_fund": "weak_fund",
    "weak_adj": "weak_adjoint",
}

FEYNRULES_INDEX_PREFIX = {
    "Lorentz": "mu",
    "Spin": "i",
    "Colour": "c",
    "Gluon": "a",
    "SU2D": "w",
    "SU2W": "aw",
    "Generation": "f",
}

FEYNRULES_GREEK_ASCII = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "μ": "mu",
    "ν": "nu",
    "ρ": "rho",
    "σ": "sigma",
}


def _feynrules_ascii_label(label: str) -> str:
    result = label.strip().replace("$", "_")
    for greek, ascii_name in FEYNRULES_GREEK_ASCII.items():
        result = result.replace(greek, ascii_name)
    result = re.sub(r"[^A-Za-z0-9_]+", "_", result)
    if result and result[0].isdigit():
        result = f"idx_{result}"
    return result


def _find_matching_square(text: str, open_position: int) -> int:
    depth = 0
    for position in range(open_position, len(text)):
        character = text[position]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return position
    raise ValueError(f"Unbalanced FeynRules brackets near {text[open_position:]!r}")


def _split_top_level_commas(text: str) -> tuple[str, ...]:
    parts = []
    start = 0
    depth = 0
    for position, character in enumerate(text):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(text[start:position].strip())
            start = position + 1
    parts.append(text[start:].strip())
    return tuple(part for part in parts if part)


def _rewrite_feynrules_indices(text: str) -> str:
    for kind, prefix in FEYNRULES_INDEX_PREFIX.items():
        text = re.sub(
            rf"Index\[{kind},\s*Ext\[(\d+)\]\]",
            lambda match, prefix=prefix: f"{prefix}{match.group(1)}",
            text,
        )
    for kind, prefix in FEYNRULES_INDEX_PREFIX.items():
        text = re.sub(
            rf"Index\[{kind},\s*([^\]]+)\]",
            lambda match, prefix=prefix: (
                f"{prefix}_feynrules_dummy_"
                f"{_feynrules_ascii_label(match.group(1))}"
            ),
            text,
        )
    return text


def _rewrite_feynrules_indexed_parameters(text: str) -> str:
    parameter_call = r"\b((?:alpha[A-Za-z0-9]+|y[ldu]))\[([^\[\]]+)\]"
    text = re.sub(
        parameter_call,
        lambda match: f"{match.group(1)}({match.group(2)})",
        text,
    )
    text = re.sub(
        r"Conjugate\[((?:alpha[A-Za-z0-9]+|y[ldu])\([^\[\]]+\))\]",
        lambda match: f"conj({match.group(1)})",
        text,
    )
    return text


def _metric_for_index_delta(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if left.startswith("aw") and right.startswith("aw"):
        return WEAK_ADJ.g(S(left), S(right)).to_expression().to_canonical_string()
    if left.startswith("a") and right.startswith("a"):
        return COLOR_ADJ.g(S(left), S(right)).to_expression().to_canonical_string()
    if left.startswith("w") and right.startswith("w"):
        return WEAK_FUND.g(S(left), S(right)).to_expression().to_canonical_string()
    if left.startswith("i") and right.startswith("i"):
        return spinor_metric(S(left), S(right)).to_canonical_string()
    if (left.startswith("c") and right.startswith("c")) or (
        left.startswith("f") and right.startswith("f")
    ):
        return COLOR_FUND.g(S(left), S(right)).to_expression().to_canonical_string()
    raise ValueError(f"Unsupported FeynRules IndexDelta labels: {left}, {right}")


def _replace_scalar_product(match: re.Match[str]) -> str:
    _replace_scalar_product.counter += 1
    dummy = S(f"mu_feynrules_dummy_sp_{_replace_scalar_product.counter}")
    return (
        pcomp(S(f"q{match.group(1)}"), dummy)
        * pcomp(S(f"q{match.group(2)}"), dummy)
    ).to_canonical_string()


_replace_scalar_product.counter = 0


def _spinor_chain_item_factor(
    item: str,
    left,
    right,
    *,
    chain_id: int,
    item_id: int,
) -> Expression:
    ga_match = re.fullmatch(r"Ga\[([^\[\]]+)\]", item)
    if ga_match:
        return gamma_matrix(left, right, S(ga_match.group(1).strip()))

    slashed_match = re.fullmatch(r"SlashedP\[(\d+)\]", item)
    if slashed_match:
        lorentz = S(f"mu_feynrules_slash_{chain_id}_{item_id}")
        return (
            pcomp(S(f"q{slashed_match.group(1)}"), lorentz)
            * gamma_matrix(left, right, lorentz)
        )

    raise ValueError(f"Unsupported FeynRules spinor-chain item: {item!r}")


def _spinor_chain_replacement(
    items: tuple[str, ...],
    left: str,
    right: str,
    *,
    chain_id: int,
) -> str:
    matrix_items = tuple(item for item in items if item not in {"ProjM", "ProjP"})
    if not matrix_items:
        return spinor_metric(S(left), S(right)).to_canonical_string()

    result = Expression.num(1)
    current = S(left)
    for item_id, item in enumerate(matrix_items, start=1):
        target = (
            S(right)
            if item_id == len(matrix_items)
            else S(f"i_feynrules_chain_{chain_id}_{item_id}")
        )
        result *= _spinor_chain_item_factor(
            item,
            current,
            target,
            chain_id=chain_id,
            item_id=item_id,
        )
        current = target
    return result.to_canonical_string()


def _bare_symbolica_name(name: str) -> str:
    return name.rsplit("::", 1)[-1]


def _is_zero_expression(expression: Expression) -> bool:
    return expression.cancel().expand().to_canonical_string() == "0"


def _terms(expression: Expression) -> tuple[Expression, ...]:
    expression = expression.cancel().expand()
    if _is_zero_expression(expression):
        return ()
    if expression.get_type() == AtomType.Add:
        return tuple(expression)
    return (expression,)


def _term_factors(term: Expression) -> tuple[Expression, ...]:
    if term.get_type() == AtomType.Mul:
        return tuple(term)
    return (term,)


def _contains_coefficient_head(expression: Expression, coefficient: str) -> bool:
    atom_type = expression.get_type()
    if atom_type == AtomType.Var:
        return _bare_symbolica_name(expression.get_name()) == coefficient
    if atom_type == AtomType.Fn:
        if _bare_symbolica_name(expression.get_name()) == coefficient:
            return True
        return any(
            _contains_coefficient_head(argument, coefficient)
            for argument in expression
        )
    if atom_type in (AtomType.Add, AtomType.Mul, AtomType.Pow):
        return any(
            _contains_coefficient_head(argument, coefficient)
            for argument in expression
        )
    return False


def _filter_terms_by_coefficient_head(
    expression: Expression,
    coefficient: str,
) -> Expression:
    """Return terms containing ``coefficient`` as a variable/function head.

    Symbolica's bare ``expression.coefficient(S("alpha"))`` extraction works
    for scalar heads such as ``alphaO3G`` but not for indexed Wilson functions
    such as ``alphaKl(f1, f2)`` or ``conj(alphaWeinberg(f1, f2))``.  SMEFT2
    fermion rows are dominated by indexed Wilson coefficients, so exact row
    comparison filters terms by coefficient head and leaves the full indexed
    coefficient factor in the scalar coefficient.  This keeps flavor order and
    complex conjugation visible to the equality test.
    """

    total = Expression.num(0)
    for term in _terms(expression):
        if any(
            _contains_coefficient_head(factor, coefficient)
            for factor in _term_factors(term)
        ):
            total += term
    return total.cancel().expand()


_WEAK_T_FACTOR_SIGNATURE = ("weak_adj", "weak_fund", "weak_fund")
_WEAK_EPS2_FACTOR_SIGNATURE = ("weak_fund", "weak_fund")
_COLOR_T_FACTOR_SIGNATURE = ("color_adj", "color_fund", "color_fund")
_GENERATOR_T_FACTOR_SIGNATURES = {
    _COLOR_T_FACTOR_SIGNATURE,
    _WEAK_T_FACTOR_SIGNATURE,
}
_ADJOINT_F_SIGNATURE_BY_T_SIGNATURE = {
    _COLOR_T_FACTOR_SIGNATURE: ("color_adj", "color_adj", "color_adj"),
    _WEAK_T_FACTOR_SIGNATURE: ("weak_adj", "weak_adj", "weak_adj"),
}
_ADJOINT_F_SIGNATURES = frozenset(_ADJOINT_F_SIGNATURE_BY_T_SIGNATURE.values())
_DUMMY_FUND_PREFIX_BY_T_SIGNATURE = {
    _COLOR_T_FACTOR_SIGNATURE: "D:C:",
    _WEAK_T_FACTOR_SIGNATURE: "D:W:",
}
_DUMMY_ADJOINT_PREFIX_BY_T_SIGNATURE = {
    _COLOR_T_FACTOR_SIGNATURE: "D:A:",
    _WEAK_T_FACTOR_SIGNATURE: "D:AW:",
}
_DUMMY_ADJOINT_PREFIX_BY_F_SIGNATURE = {
    f_signature: _DUMMY_ADJOINT_PREFIX_BY_T_SIGNATURE[t_signature]
    for t_signature, f_signature in _ADJOINT_F_SIGNATURE_BY_T_SIGNATURE.items()
}


def _is_dummy_weak_label(label: object) -> bool:
    return isinstance(label, str) and label.startswith("D:W:")


def _is_weak_t_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "t"
        and factor[1] == _WEAK_T_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_weak_eps2_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "weak_eps2"
        and factor[1] == _WEAK_EPS2_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 2
    )


def _is_generator_t_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "t"
        and factor[1] in _GENERATOR_T_FACTOR_SIGNATURES
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_structure_constant_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "f"
        and factor[1] in _ADJOINT_F_SIGNATURES
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _factor_labels(factor: object) -> tuple[object, ...]:
    if isinstance(factor, tuple) and len(factor) == 3 and isinstance(factor[2], tuple):
        return factor[2]
    return ()


def _label_occurrences(factors: Iterable[object], label: object) -> int:
    return sum(1 for factor in factors for item in _factor_labels(factor) if item == label)


def _fresh_dummy_adjoint_label(
    factors: Iterable[object],
    signature: tuple[str, str, str],
) -> str:
    prefix = _DUMMY_ADJOINT_PREFIX_BY_T_SIGNATURE[signature]
    used = {item for factor in factors for item in _factor_labels(factor)}
    candidate = 1
    while f"{prefix}{candidate}" in used:
        candidate += 1
    return f"{prefix}{candidate}"


def _generator_chain_from_pair(
    factors: list[object],
    first_index: int,
    second_index: int,
) -> tuple[
    tuple[str, str, str],
    object,
    object,
    object,
    object,
    object,
] | None:
    first = factors[first_index]
    second = factors[second_index]
    if not _is_generator_t_factor(first) or not _is_generator_t_factor(second):
        return None
    signature = first[1]
    if second[1] != signature:
        return None

    first_adj, first_left, first_right = first[2]
    second_adj, second_left, second_right = second[2]
    dummy_prefix = _DUMMY_FUND_PREFIX_BY_T_SIGNATURE[signature]
    candidates = []
    if (
        first_right == second_left
        and isinstance(first_right, str)
        and first_right.startswith(dummy_prefix)
    ):
        candidates.append(
            (signature, first_left, first_right, second_right, first_adj, second_adj)
        )
    if (
        second_right == first_left
        and isinstance(second_right, str)
        and second_right.startswith(dummy_prefix)
    ):
        candidates.append(
            (signature, second_left, second_right, first_right, second_adj, first_adj)
        )
    if len(candidates) != 1:
        return None
    signature, open_left, shared, open_right, left_adj, right_adj = candidates[0]
    if _label_occurrences(factors, shared) != 2:
        return None
    return signature, open_left, shared, open_right, left_adj, right_adj


def _canonical_key_with_commuting_factors(
    key: CanonicalTensorMonomial,
    factors: Iterable[object],
) -> CanonicalTensorMonomial:
    return CanonicalTensorMonomial(
        commuting_factors=tuple(sorted(factors, key=repr)),
        ordered_factors=key.ordered_factors,
    )


def _permutation_sign(current: tuple[object, ...], target: tuple[object, ...]) -> int:
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


def _antisymmetric_factor_with_sorted_labels(
    head: str,
    signature: tuple[str, ...],
    labels: tuple[object, ...],
) -> tuple[int, object]:
    if len(set(labels)) != len(labels):
        return 0, None
    ordered = tuple(sorted(labels, key=repr))
    return _permutation_sign(labels, ordered), (head, signature, ordered)


def _expand_one_generator_product_ordering(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Sort one adjacent generator chain and emit the commutator term.

    FeynPy's covariant-derivative expansion may leave ``T^b T^a`` while
    FeynRules rewrites it as ``T^a T^b - i f^{abc} T^c``.  Applying that
    identity to both sides gives a strict common Lie-algebra basis.
    """

    factors = list(key.commuting_factors)
    for first_index, first in enumerate(factors):
        if not _is_generator_t_factor(first):
            continue
        for second_index in range(first_index + 1, len(factors)):
            chain = _generator_chain_from_pair(factors, first_index, second_index)
            if chain is None:
                continue
            signature, open_left, shared, open_right, left_adj, right_adj = chain
            if repr(left_adj) <= repr(right_adj):
                continue

            sorted_left_adj, sorted_right_adj = right_adj, left_adj
            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {first_index, second_index}
            ]
            product_factors = [
                *rest,
                ("t", signature, (sorted_left_adj, open_left, shared)),
                ("t", signature, (sorted_right_adj, shared, open_right)),
            ]
            dummy_adj = _fresh_dummy_adjoint_label(rest, signature)
            adjoint_signature = _ADJOINT_F_SIGNATURE_BY_T_SIGNATURE[signature]
            commutator_factors = [
                *rest,
                ("f", adjoint_signature, (dummy_adj, sorted_left_adj, sorted_right_adj)),
                ("t", signature, (dummy_adj, open_left, open_right)),
            ]
            return True, [
                (
                    Expression.num(1),
                    _canonical_key_with_commuting_factors(key, product_factors),
                ),
                (
                    -I,
                    _canonical_key_with_commuting_factors(key, commutator_factors),
                ),
            ]
    return False, [(Expression.num(1), key)]


def _normalize_generator_product_order_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(12):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = _expand_one_generator_product_ordering(term_key)
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("generator-product normalization did not converge")


def _normalize_generator_product_order_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_generator_product_order_key(key):
            normalized_coefficient = (
                coefficient * multiplier
            ).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _canonical_structure_constant_pair_factors(
    signature: tuple[str, ...],
    left_labels: tuple[object, object, object],
    right_labels: tuple[object, object, object],
) -> tuple[int, tuple[object, object]] | None:
    left_sign, left_factor = _antisymmetric_factor_with_sorted_labels(
        "f",
        signature,
        left_labels,
    )
    right_sign, right_factor = _antisymmetric_factor_with_sorted_labels(
        "f",
        signature,
        right_labels,
    )
    if left_sign == 0 or right_sign == 0:
        return None
    return left_sign * right_sign, (left_factor, right_factor)


def _structure_constant_jacobi_basis(
    signature: tuple[str, ...],
    free_labels: tuple[object, object, object, object],
    shared_label: object,
    basis_name: str,
) -> tuple[int, tuple[object, object]] | None:
    a_label, b_label, c_label, d_label = free_labels
    basis_labels = {
        "p12": ((a_label, b_label, shared_label), (c_label, d_label, shared_label)),
        "p14": ((a_label, d_label, shared_label), (b_label, c_label, shared_label)),
    }
    left_labels, right_labels = basis_labels[basis_name]
    return _canonical_structure_constant_pair_factors(
        signature,
        left_labels,
        right_labels,
    )


def _structure_constant_jacobi_pair(
    factors: list[object],
    left_index: int,
    right_index: int,
) -> tuple[tuple[str, ...], object, tuple[object, object, object, object], str, int] | None:
    left = factors[left_index]
    right = factors[right_index]
    if not _is_structure_constant_factor(left) or not _is_structure_constant_factor(
        right
    ):
        return None
    signature = left[1]
    if right[1] != signature:
        return None

    left_labels = left[2]
    right_labels = right[2]
    shared = tuple(label for label in left_labels if label in right_labels)
    if len(shared) != 1:
        return None
    shared_label = shared[0]
    dummy_prefix = _DUMMY_ADJOINT_PREFIX_BY_F_SIGNATURE[signature]
    if not (isinstance(shared_label, str) and shared_label.startswith(dummy_prefix)):
        return None
    if _label_occurrences(factors, shared_label) != 2:
        return None

    left_free = tuple(label for label in left_labels if label != shared_label)
    right_free = tuple(label for label in right_labels if label != shared_label)
    free_labels = tuple(sorted(left_free + right_free, key=repr))
    if len(set(free_labels)) != 4:
        return None

    a_label, b_label, c_label, d_label = free_labels
    targets = {
        "p12": ((a_label, b_label, shared_label), (c_label, d_label, shared_label)),
        "p13": ((a_label, c_label, shared_label), (b_label, d_label, shared_label)),
        "p14": ((a_label, d_label, shared_label), (b_label, c_label, shared_label)),
    }
    actual_pair_sets = [
        frozenset(left_free),
        frozenset(right_free),
    ]
    for name, (target_left, target_right) in targets.items():
        left_pair = frozenset(target_left[:2])
        right_pair = frozenset(target_right[:2])
        if actual_pair_sets == [left_pair, right_pair]:
            sign = _permutation_sign(left_labels, target_left) * _permutation_sign(
                right_labels,
                target_right,
            )
            return signature, shared_label, free_labels, name, sign
        if actual_pair_sets == [right_pair, left_pair]:
            sign = _permutation_sign(left_labels, target_right) * _permutation_sign(
                right_labels,
                target_left,
            )
            return signature, shared_label, free_labels, name, sign
    return None


def _expand_one_structure_constant_jacobi_pair(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Reduce one ``f*f`` product to the deterministic Jacobi basis.

    The only identity applied is

        ``f(a,b,e) f(c,d,e) - f(a,c,e) f(b,d,e)
        + f(a,d,e) f(b,c,e) = 0``.

    The middle pairing is replaced by the fixed ``p12`` and ``p14`` pairings.
    This is narrower than a general color simplification: all external labels
    and every non-``f`` factor remain untouched, and the shared adjoint label
    must be a genuine dummy occurring only in the selected ``f*f`` pair.
    """

    factors = list(key.commuting_factors)
    for left_index, left in enumerate(factors):
        if not _is_structure_constant_factor(left):
            continue
        for right_index in range(left_index + 1, len(factors)):
            pair = _structure_constant_jacobi_pair(factors, left_index, right_index)
            if pair is None:
                continue
            signature, shared_label, free_labels, kind, source_sign = pair
            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {left_index, right_index}
            ]

            target_basis = ("p12", "p14") if kind == "p13" else (kind,)
            expanded = []
            for basis_name in target_basis:
                target = _structure_constant_jacobi_basis(
                    signature,
                    free_labels,
                    shared_label,
                    basis_name,
                )
                if target is None:
                    continue
                target_sign, target_factors = target
                expanded.append(
                    (
                        Expression.num(source_sign * target_sign),
                        _canonical_key_with_commuting_factors(
                            key,
                            [*rest, *target_factors],
                        ),
                    )
                )
            if expanded:
                return kind == "p13", expanded
    return False, [(Expression.num(1), key)]


def _normalize_structure_constant_jacobi_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(8):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = _expand_one_structure_constant_jacobi_pair(
                term_key
            )
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("structure-constant Jacobi normalization did not converge")


def _normalize_structure_constant_jacobi_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_structure_constant_jacobi_key(
            key
        ):
            normalized_coefficient = (coefficient * multiplier).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _normalize_one_weak_t_epsilon_pair(
    factors: list[object],
) -> tuple[int, bool]:
    """Apply the SU(2) pseudoreality identity to one canonical-map pair.

    The identity is used only after the tensor canonicalizer has marked the
    contracted weak-fundamental index as dummy.  We keep the two generator
    endpoint orientations distinct, so this pass identifies
    ``t(D, i) eps(D, j)`` with ``t(D, j) eps(D, i)`` and
    ``t(i, D) eps(D, j)`` with ``t(j, D) eps(D, i)``, but it does not identify
    those two classes with each other.
    """

    for t_index, t_factor in enumerate(factors):
        if not _is_weak_t_factor(t_factor):
            continue
        t_adj, t_left, t_right = t_factor[2]
        t_fund = (t_left, t_right)
        for eps_index, eps_factor in enumerate(factors):
            if t_index == eps_index or not _is_weak_eps2_factor(eps_factor):
                continue
            eps_left, eps_right = eps_factor[2]
            eps_fund = (eps_left, eps_right)
            shared = {
                label
                for label in t_fund
                if label in eps_fund and _is_dummy_weak_label(label)
            }
            if len(shared) != 1:
                continue
            shared_label = next(iter(shared))
            t_positions = [
                position for position, label in enumerate(t_fund) if label == shared_label
            ]
            eps_positions = [
                position for position, label in enumerate(eps_fund) if label == shared_label
            ]
            if len(t_positions) != 1 or len(eps_positions) != 1:
                continue

            t_shared_position = t_positions[0]
            eps_shared_position = eps_positions[0]
            t_open = t_fund[1 - t_shared_position]
            eps_open = eps_fund[1 - eps_shared_position]
            composite_head = (
                "weak_t_eps_left_contract"
                if t_shared_position == 0
                else "weak_t_eps_right_contract"
            )
            sign = 1 if eps_shared_position == 0 else -1
            open_labels = tuple(sorted((t_open, eps_open), key=repr))
            composite_factor = (
                composite_head,
                _WEAK_T_FACTOR_SIGNATURE,
                (t_adj, *open_labels),
            )
            factors.pop(max(t_index, eps_index))
            factors.pop(min(t_index, eps_index))
            factors.append(composite_factor)
            factors.sort(key=repr)
            return sign, True
    return 1, False


def _normalize_weak_t_epsilon_key(
    key: CanonicalTensorMonomial,
) -> tuple[int, CanonicalTensorMonomial]:
    sign = 1
    factors = list(key.commuting_factors)
    while True:
        pair_sign, changed = _normalize_one_weak_t_epsilon_pair(factors)
        if not changed:
            break
        sign *= pair_sign
    return sign, CanonicalTensorMonomial(
        commuting_factors=tuple(factors),
        ordered_factors=key.ordered_factors,
    )


def _normalize_weak_t_epsilon_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        sign, normalized_key = _normalize_weak_t_epsilon_key(key)
        normalized_coefficient = (
            coefficient * Expression.num(sign)
        ).cancel().expand()
        normalized[normalized_key] = (
            normalized.get(normalized_key, Expression.num(0))
            + normalized_coefficient
        ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


_WEAK_T_EPS_COMPOSITE_HEADS = frozenset(
    {"weak_t_eps_left_contract", "weak_t_eps_right_contract"}
)


def _is_weak_t_eps_composite_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] in _WEAK_T_EPS_COMPOSITE_HEADS
        and factor[1] == _WEAK_T_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _weak_t_eps_pair_from_factors(
    factors: list[object],
    t_index: int,
    composite_index: int,
) -> tuple[
    str,
    object,
    object,
    object,
    object,
    object,
    int,
    int,
] | None:
    t_factor = factors[t_index]
    composite_factor = factors[composite_index]
    if not _is_weak_t_factor(t_factor) or not _is_weak_t_eps_composite_factor(
        composite_factor
    ):
        return None

    t_adj, t_left, t_right = t_factor[2]
    composite_adj, composite_left, composite_right = composite_factor[2]
    t_fund = (t_left, t_right)
    composite_fund = (composite_left, composite_right)
    shared = [
        label
        for label in t_fund
        if label in composite_fund and _is_dummy_weak_label(label)
    ]
    if len(shared) != 1:
        return None

    shared_label = shared[0]
    if _label_occurrences(factors, shared_label) != 2:
        return None

    t_shared_position = t_fund.index(shared_label)
    composite_shared_position = composite_fund.index(shared_label)
    t_open = t_fund[1 - t_shared_position]
    composite_open = composite_fund[1 - composite_shared_position]
    if not (
        isinstance(t_open, str)
        and isinstance(composite_open, str)
        and t_open.startswith("E:W:")
        and composite_open.startswith("E:W:")
    ):
        return None

    return (
        composite_factor[0],
        t_adj,
        composite_adj,
        t_open,
        composite_open,
        shared_label,
        t_shared_position,
        composite_shared_position,
    )


def _replace_weak_t_open_label(
    factor: object,
    *,
    shared_label: object,
    new_open: object,
) -> object:
    adjoint, left, right = factor[2]
    labels = [left, right]
    if labels[0] == shared_label:
        labels[1] = new_open
    elif labels[1] == shared_label:
        labels[0] = new_open
    else:  # pragma: no cover - guarded by the caller.
        raise ValueError("weak generator does not contain the shared label")
    return ("t", _WEAK_T_FACTOR_SIGNATURE, (adjoint, *labels))


def _replace_weak_t_eps_open_label(
    factor: object,
    *,
    shared_label: object,
    new_open: object,
) -> object:
    adjoint, left, right = factor[2]
    labels = [left, right]
    if labels[0] == shared_label:
        labels[1] = new_open
    elif labels[1] == shared_label:
        labels[0] = new_open
    else:  # pragma: no cover - guarded by the caller.
        raise ValueError("weak T-epsilon factor does not contain the shared label")
    return (factor[0], _WEAK_T_FACTOR_SIGNATURE, (adjoint, *labels))


def _expand_one_weak_doublet_generator_epsilon_pair(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Put SU(2) ``T*T*epsilon`` products in a single pseudoreal basis.

    After the one-generator ``T*epsilon`` contraction has been made explicit,
    the two-covariant-derivative Higgs-tilde rows can still differ by which
    external weak doublet index sits on the remaining generator.  For SU(2),

        ``P(j,i) = -P(i,j) + i f(a,b,c) C(c,i,j)``

    where ``P`` is the product of the remaining generator with the contracted
    ``T*epsilon`` composite and ``C`` is the same composite with the adjoint
    commutator index.  This is the Pauli-matrix pseudoreality relation for two
    generators acting on ``epsilon * Phi.bar``.  The pass only fires when the
    two factors share one dummy weak-fundamental index and the two open weak
    indices are external.
    """

    factors = list(key.commuting_factors)
    for t_index, t_factor in enumerate(factors):
        if not _is_weak_t_factor(t_factor):
            continue
        for composite_index, composite_factor in enumerate(factors):
            if t_index == composite_index:
                continue
            pair = _weak_t_eps_pair_from_factors(factors, t_index, composite_index)
            if pair is None:
                continue
            (
                composite_head,
                t_adj,
                composite_adj,
                t_open,
                composite_open,
                shared_label,
                _t_shared_position,
                _composite_shared_position,
            ) = pair
            if repr(t_open) <= repr(composite_open):
                continue

            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {t_index, composite_index}
            ]
            swapped_product_factors = [
                *rest,
                _replace_weak_t_open_label(
                    t_factor,
                    shared_label=shared_label,
                    new_open=composite_open,
                ),
                _replace_weak_t_eps_open_label(
                    composite_factor,
                    shared_label=shared_label,
                    new_open=t_open,
                ),
            ]

            dummy_adj = _fresh_dummy_adjoint_label(rest, _WEAK_T_FACTOR_SIGNATURE)
            commutator_adjoint_labels = (
                (dummy_adj, composite_adj, t_adj)
                if composite_head == "weak_t_eps_left_contract"
                else (dummy_adj, t_adj, composite_adj)
            )
            f_sign, f_factor = _antisymmetric_factor_with_sorted_labels(
                "f",
                _ADJOINT_F_SIGNATURE_BY_T_SIGNATURE[_WEAK_T_FACTOR_SIGNATURE],
                commutator_adjoint_labels,
            )
            if f_sign == 0:
                continue
            composite_open_labels = tuple(sorted((composite_open, t_open), key=repr))
            commutator_factors = [
                *rest,
                f_factor,
                (
                    composite_head,
                    _WEAK_T_FACTOR_SIGNATURE,
                    (dummy_adj, *composite_open_labels),
                ),
            ]
            return True, [
                (
                    Expression.num(-1),
                    _canonical_key_with_commuting_factors(key, swapped_product_factors),
                ),
                (
                    I * Expression.num(f_sign),
                    _canonical_key_with_commuting_factors(key, commutator_factors),
                ),
            ]
    return False, [(Expression.num(1), key)]


def _normalize_weak_doublet_generator_epsilon_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(8):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = (
                _expand_one_weak_doublet_generator_epsilon_pair(term_key)
            )
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("weak doublet generator-epsilon normalization did not converge")


def _normalize_weak_doublet_generator_epsilon_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_weak_doublet_generator_epsilon_key(
            key
        ):
            normalized_coefficient = (coefficient * multiplier).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _canonical_report_for_coefficient_head(
    expression: Expression,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
):
    report = canonical_tensor_monomial_report(
        _filter_terms_by_coefficient_head(expression, coefficient),
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    report = _normalize_generator_product_order_report(report)
    report = _normalize_weak_t_epsilon_report(report)
    report = _normalize_weak_doublet_generator_epsilon_report(report)
    return _normalize_structure_constant_jacobi_report(report)


_DIRAC_C_FACTOR_SIGNATURE = ("spinor", "spinor")
_GAMMA_FACTOR_SIGNATURE = ("spinor", "spinor", "lorentz")
_SPINOR_METRIC_FACTOR_SIGNATURE = ("spinor", "spinor")


@dataclass(frozen=True)
class _ChargeConjugationArm:
    external: object
    lorentz_ext_to_c: tuple[object, ...]
    used_gamma_indices: frozenset[int]
    c_outgoing: bool | None


@dataclass(frozen=True)
class _EcPartnerPackagingRule:
    partner_key: str
    phase: int
    antisymmetric_duplicates: bool
    source: str


_EC_PARTNER_PACKAGING_RULES: dict[tuple[str, str], _EcPartnerPackagingRule] = {
    (
        "dRbar|eR|lLbar|qL",
        "alphaEcqedl",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|eR|lL|qLbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="LEvCCLRRL alphaEcqedl + HC; one C-arm transposition",
    ),
    (
        "dRbar|eR|lLbar|qL",
        "alphaEcqedlthree",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|eR|lL|qLbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="LEvCCLRRL alphaEcqedlthree + HC; one C-arm transposition",
    ),
    (
        "dR|eRbar|lL|qLbar",
        "alphaEcqedl",
    ): _EcPartnerPackagingRule(
        partner_key="dR|eRbar|lLbar|qL",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCLRRL alphaEcqedl",
    ),
    (
        "dR|eRbar|lL|qLbar",
        "alphaEcqedlthree",
    ): _EcPartnerPackagingRule(
        partner_key="dR|eRbar|lLbar|qL",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCLRRL alphaEcqedlthree",
    ),
    (
        "eRbar|lL|qL|uRbar",
        "alphaEcuelq",
    ): _EcPartnerPackagingRule(
        partner_key="eRbar|lL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcuelq direct packaging",
    ),
    (
        "eRbar|lL|qL|uRbar",
        "alphaEcuelqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="eRbar|lL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcuelqtwo direct packaging",
    ),
    (
        "eR|lLbar|qLbar|uR",
        "alphaEcuelq",
    ): _EcPartnerPackagingRule(
        partner_key="eR|lLbar|qL|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcuelq",
    ),
    (
        "eR|lLbar|qLbar|uR",
        "alphaEcuelqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="eR|lLbar|qL|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcuelqtwo",
    ),
    (
        "dRbar|qL|qL|uRbar",
        "alphaEcudqq",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|qL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=True,
        source="LEvCCRRLL alphaEcudqq; duplicate qL assignments antisymmetrized",
    ),
    (
        "dRbar|qL|qL|uRbar",
        "alphaEcudqqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|qL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcudqqtwo; gamma2 structure fixes symmetric duplicate sum",
    ),
    (
        "dR|qLbar|qLbar|uR",
        "alphaEcudqq",
    ): _EcPartnerPackagingRule(
        partner_key="dR|qL|qLbar|uRbar",
        phase=1,
        antisymmetric_duplicates=True,
        source="Hermitian conjugate of LEvCCRRLL alphaEcudqq",
    ),
    (
        "dR|qLbar|qLbar|uR",
        "alphaEcudqqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="dR|qL|qLbar|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcudqqtwo",
    ),
}


def _is_dirac_c_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "dirac_C"
        and factor[1] == _DIRAC_C_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 2
    )


def _is_gamma_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "gamma"
        and factor[1] == _GAMMA_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_external_spinor_label(label: object) -> bool:
    return isinstance(label, str) and label.startswith("E:S:")


def _external_leg_number(label: object) -> int | None:
    if not isinstance(label, str) or not label.startswith("E:"):
        return None
    match = re.search(r"(\d+)$", label)
    return int(match.group(1)) if match is not None else None


def _trace_charge_conjugation_arm(
    start: object,
    gamma_factors: list[object],
) -> _ChargeConjugationArm | None:
    adjacency: dict[object, list[tuple[int, object, object, object, bool]]] = defaultdict(
        list
    )
    for index, factor in enumerate(gamma_factors):
        left, right, lorentz = factor[2]
        adjacency[left].append((index, left, right, lorentz, True))
        adjacency[right].append((index, right, left, lorentz, False))

    if _is_external_spinor_label(start) and not adjacency[start]:
        return _ChargeConjugationArm(
            external=start,
            lorentz_ext_to_c=(),
            used_gamma_indices=frozenset(),
            c_outgoing=None,
        )

    current = start
    previous = None
    used: set[int] = set()
    path: list[tuple[object, object, object]] = []
    first_edge_outgoing = None
    for _ in range(len(gamma_factors) + 2):
        candidates = [
            edge
            for edge in adjacency[current]
            if edge[0] not in used and edge[2] != previous
        ]
        if len(candidates) != 1:
            return None
        index, left, right, lorentz, outgoing = candidates[0]
        if first_edge_outgoing is None:
            first_edge_outgoing = outgoing
        used.add(index)
        path.append((left, right, lorentz))
        previous = current
        current = right
        if _is_external_spinor_label(current):
            return _ChargeConjugationArm(
                external=current,
                lorentz_ext_to_c=tuple(edge[2] for edge in reversed(path)),
                used_gamma_indices=frozenset(used),
                c_outgoing=first_edge_outgoing,
            )
    return None


def _fresh_dummy_spinor_label(used_labels: set[object]) -> str:
    candidate = 1
    while True:
        label = f"D:S:{candidate}"
        if label not in used_labels:
            used_labels.add(label)
            return label
        candidate += 1


def _spinor_chain_factors(
    start: object,
    end: object,
    lorentz_sequence: tuple[object, ...],
    used_labels: set[object],
) -> list[object]:
    if not lorentz_sequence:
        return [("g", _SPINOR_METRIC_FACTOR_SIGNATURE, (start, end))]

    spinors = [start]
    for _ in range(len(lorentz_sequence) - 1):
        spinors.append(_fresh_dummy_spinor_label(used_labels))
    spinors.append(end)
    return [
        (
            "gamma",
            _GAMMA_FACTOR_SIGNATURE,
            (spinors[index], spinors[index + 1], lorentz),
        )
        for index, lorentz in enumerate(lorentz_sequence)
    ]


def _charge_conjugation_bilinear_factors(
    first: _ChargeConjugationArm,
    second: _ChargeConjugationArm,
    used_labels: set[object],
) -> list[object] | None:
    if first.lorentz_ext_to_c and second.lorentz_ext_to_c:
        return None

    if first.lorentz_ext_to_c:
        if first.c_outgoing:
            start, end = second.external, first.external
            sequence = first.lorentz_ext_to_c
        else:
            start, end = first.external, second.external
            sequence = tuple(reversed(first.lorentz_ext_to_c))
    elif second.lorentz_ext_to_c:
        if second.c_outgoing:
            start, end = first.external, second.external
            sequence = second.lorentz_ext_to_c
        else:
            start, end = second.external, first.external
            sequence = tuple(reversed(second.lorentz_ext_to_c))
    else:
        start, end = first.external, second.external
        sequence = ()

    return _spinor_chain_factors(start, end, sequence, used_labels)


def _replace_external_nonspinor_leg_label(
    label: object,
    first_leg: int,
    second_leg: int,
) -> object:
    if not isinstance(label, str) or not label.startswith("E:"):
        return label
    parts = label.split(":")
    if len(parts) != 3 or parts[1] == "S":
        return label
    leg = _external_leg_number(label)
    if leg == first_leg:
        return re.sub(r"\d+$", str(second_leg), label)
    if leg == second_leg:
        return re.sub(r"\d+$", str(first_leg), label)
    return label


def _replace_factor_external_nonspinor_legs(
    factor: object,
    first_leg: int,
    second_leg: int,
) -> object:
    if not (
        isinstance(factor, tuple)
        and len(factor) == 3
        and isinstance(factor[2], tuple)
    ):
        return factor
    return (
        factor[0],
        factor[1],
        tuple(
            _replace_external_nonspinor_leg_label(label, first_leg, second_leg)
            for label in factor[2]
        ),
    )


def _ec_charge_conjugation_key(
    key: CanonicalTensorMonomial,
    *,
    mode: str,
    phase: int,
) -> tuple[int, CanonicalTensorMonomial] | None:
    factors = list(key.commuting_factors)
    c_indices = [
        index for index, factor in enumerate(factors) if _is_dirac_c_factor(factor)
    ]
    if len(c_indices) != 2:
        return None

    gamma_factors = [factor for factor in factors if _is_gamma_factor(factor)]
    first_c, second_c = (factors[index] for index in c_indices)

    arms: list[_ChargeConjugationArm] = []
    used_gamma_indices: set[int] = set()
    for label in first_c[2] + second_c[2]:
        arm = _trace_charge_conjugation_arm(label, gamma_factors)
        if arm is None:
            return None
        arms.append(arm)
        used_gamma_indices.update(arm.used_gamma_indices)
    if len(used_gamma_indices) != len(gamma_factors):
        return None

    first_leg = second_leg = None
    if mode == "crossed":
        first_leg = _external_leg_number(first_c[2][1])
        second_leg = _external_leg_number(second_c[2][1])
    elif mode != "direct":
        raise ValueError(f"Unsupported Ec charge-conjugation mode {mode!r}.")

    rest = []
    for index, factor in enumerate(factors):
        if index in c_indices or _is_gamma_factor(factor):
            continue
        if first_leg is not None and second_leg is not None:
            factor = _replace_factor_external_nonspinor_legs(
                factor,
                first_leg,
                second_leg,
            )
        rest.append(factor)

    used_labels = {
        label
        for factor in rest
        for label in (
            factor[2]
            if isinstance(factor, tuple)
            and len(factor) == 3
            and isinstance(factor[2], tuple)
            else ()
        )
    }
    used_labels.update(arm.external for arm in arms)

    pairings = {
        "crossed": ((arms[0], arms[3]), (arms[1], arms[2])),
        "direct": ((arms[0], arms[1]), (arms[2], arms[3])),
    }[mode]
    bilinear_factors: list[object] = []
    for first, second in pairings:
        factors_for_pair = _charge_conjugation_bilinear_factors(
            first,
            second,
            used_labels,
        )
        if factors_for_pair is None:
            return None
        bilinear_factors.extend(factors_for_pair)

    return phase, _canonical_key_with_commuting_factors(
        key,
        [*rest, *bilinear_factors],
    )


def _swap_ec_coefficient_boundary_args(
    expression: Expression,
    coefficient: str,
) -> Expression:
    first, second, third, fourth = S("ec_first_", "ec_second_", "ec_third_", "ec_fourth_")
    return expression.replace(
        S(coefficient)(first, second, third, fourth),
        S(coefficient)(fourth, second, third, first),
    ).cancel().expand()


def _symbol_from_canonical_label(label: object):
    if not isinstance(label, str):
        return S(str(label))
    if label.startswith("E:"):
        return S(label.split(":", 2)[2])
    if label.startswith("D:"):
        _dummy, group, name = label.split(":", 2)
        prefix = {
            "S": "i",
            "L": "mu",
            "C": "c",
            "W": "w",
            "A": "a",
            "AW": "aw",
        }[group]
        return S(f"{prefix}_cc_dummy_{name}")
    return S(label)


def _expression_from_canonical_factor(factor: object) -> Expression:
    if isinstance(factor, tuple) and len(factor) == 3 and factor[0] == "pcomp":
        return pcomp(
            _symbol_from_canonical_label(factor[1]),
            _symbol_from_canonical_label(factor[2]),
        )
    if not (
        isinstance(factor, tuple)
        and len(factor) == 3
        and isinstance(factor[2], tuple)
    ):
        raise ValueError(f"Unsupported canonical factor in Ec rewrite: {factor!r}")

    head, signature, labels = factor
    args = tuple(_symbol_from_canonical_label(label) for label in labels)
    if head == "g":
        if signature == ("lorentz", "lorentz"):
            return lorentz_metric(*args)
        if signature == ("spinor", "spinor"):
            return spinor_metric(*args)
        if signature == ("color_fund", "color_fund"):
            return COLOR_FUND.g(*args).to_expression()
        if signature == ("color_adj", "color_adj"):
            return COLOR_ADJ.g(*args).to_expression()
        if signature == ("weak_fund", "weak_fund"):
            return WEAK_FUND.g(*args).to_expression()
        if signature == ("weak_adj", "weak_adj"):
            return WEAK_ADJ.g(*args).to_expression()
    if head == "gamma":
        return gamma_matrix(*args)
    if head == "dirac_C":
        return dirac_charge_conjugation(*args)
    if head == "weak_eps2":
        return weak_eps2(*args)
    if head == "lor_levi_civita":
        return lorentz_levi_civita(*args)
    if head == "t":
        if signature == _COLOR_T_FACTOR_SIGNATURE:
            return gauge_generator(*args)
        if signature == _WEAK_T_FACTOR_SIGNATURE:
            return weak_gauge_generator(*args)
    if head == "f":
        if signature == ("color_adj", "color_adj", "color_adj"):
            return structure_constant(*args)
        if signature == ("weak_adj", "weak_adj", "weak_adj"):
            return weak_structure_constant(*args)

    raise ValueError(f"Unsupported canonical factor in Ec rewrite: {factor!r}")


def _expression_from_canonical_map(
    mapping: dict[CanonicalTensorMonomial, Expression],
) -> Expression:
    total = Expression.num(0)
    for key, coefficient in mapping.items():
        term = coefficient
        for factor in key.commuting_factors:
            term *= _expression_from_canonical_factor(factor)
        for factor in key.ordered_factors:
            term *= _expression_from_canonical_factor(factor)
        total += term
    return total.cancel().expand()


def _normalize_ec_charge_conjugation_report(
    report: CanonicalMonomialReport,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
    mode: str,
    phase: int,
) -> CanonicalMonomialReport:
    transformed: dict[CanonicalTensorMonomial, Expression] = {}
    changed = False
    for key, coefficient_expression in report.map.items():
        replacement = _ec_charge_conjugation_key(key, mode=mode, phase=phase)
        if replacement is None:
            transformed_key = key
            transformed_coefficient = coefficient_expression
        else:
            multiplier, transformed_key = replacement
            transformed_coefficient = (
                coefficient_expression * Expression.num(multiplier)
            ).cancel().expand()
            if mode == "crossed":
                transformed_coefficient = _swap_ec_coefficient_boundary_args(
                    transformed_coefficient,
                    coefficient,
                )
            changed = True
        transformed[transformed_key] = (
            transformed.get(transformed_key, Expression.num(0))
            + transformed_coefficient
        ).cancel().expand()

    if not changed:
        return report

    recanonicalized = canonical_tensor_monomial_report(
        _expression_from_canonical_map(transformed),
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=recanonicalized.canonical_terms,
        map=recanonicalized.map,
    )


def _coefficient_comparison_from_reports(
    coefficient: str,
    feynpy_report: CanonicalMonomialReport,
    feynrules_report: CanonicalMonomialReport,
) -> CanonicalCoefficientComparison:
    feynpy_keys = set(feynpy_report.map)
    feynrules_keys = set(feynrules_report.map)
    shared_keys = feynpy_keys & feynrules_keys
    coefficient_mismatches = {
        key: (feynpy_report.map[key], feynrules_report.map[key])
        for key in shared_keys
        if feynpy_report.map[key].cancel().expand().to_canonical_string()
        != feynrules_report.map[key].cancel().expand().to_canonical_string()
    }
    return CanonicalCoefficientComparison(
        coefficient=coefficient,
        feynpy_raw_terms=feynpy_report.raw_terms,
        feynrules_raw_terms=feynrules_report.raw_terms,
        feynpy_canonical_terms=feynpy_report.canonical_terms,
        feynrules_canonical_terms=feynrules_report.canonical_terms,
        feynpy_only={
            key: feynpy_report.map[key]
            for key in sorted(feynpy_keys - feynrules_keys, key=repr)
        },
        feynrules_only={
            key: feynrules_report.map[key]
            for key in sorted(feynrules_keys - feynpy_keys, key=repr)
        },
        coefficient_mismatches=coefficient_mismatches,
    )


def _bar_insensitive_name(name: str) -> str:
    return name[:-3] if name.endswith("bar") else name


def _bar_insensitive_field_key(fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(_bar_insensitive_name(name) for name in fields))


def _charge_conjugation_candidate_orders(
    reference_fields: tuple[str, ...],
    partner_fields: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    orders: list[tuple[str, ...]] = []

    def visit(slot: int, remaining: tuple[str, ...], current: tuple[str, ...]) -> None:
        if slot == len(reference_fields):
            orders.append(current)
            return
        target_base = _bar_insensitive_name(reference_fields[slot])
        for index, candidate in enumerate(remaining):
            if _bar_insensitive_name(candidate) != target_base:
                continue
            visit(
                slot + 1,
                remaining[:index] + remaining[index + 1 :],
                (*current, candidate),
            )

    visit(0, tuple(partner_fields), ())
    return tuple(orders)


def _duplicate_assignment_sign(
    order: tuple[str, ...],
    reference_fields: tuple[str, ...],
) -> int:
    sign = 1
    for base in sorted({_bar_insensitive_name(name) for name in reference_fields}):
        slots = [
            slot
            for slot, name in enumerate(reference_fields)
            if _bar_insensitive_name(name) == base
        ]
        if len(slots) < 2:
            continue
        current = tuple(order[slot] for slot in slots)
        target = tuple(sorted(current))
        sign *= _permutation_sign(current, target)
    return sign


def _candidate_order_rule_sum(
    *,
    lagrangian,
    field_map: dict[str, object],
    reference_fields: tuple[str, ...],
    candidate_orders: tuple[tuple[str, ...], ...],
    antisymmetric_duplicates: bool,
) -> Expression:
    total = Expression.num(0)
    for order in candidate_orders:
        sign = (
            _duplicate_assignment_sign(order, reference_fields)
            if antisymmetric_duplicates
            else 1
        )
        total += Expression.num(sign) * lagrangian.feynman_rule(
            *(field_map[name] for name in order),
            simplify=True,
        )
    return total.cancel().expand()


def _ec_partner_packaging_comparison(
    *,
    reference: FeynRulesVertex,
    coefficient: str,
    feynrules_report: CanonicalMonomialReport,
    local_vertices: Iterable["LocalVertex"],
    lagrangian,
    field_map: dict[str, object],
    external_indices,
    max_dummy_permutations: int,
) -> tuple[CanonicalCoefficientComparison, str] | None:
    """Apply a pinned Ec partner-packaging rule for one coefficient sector."""

    reference_key = _name_key(reference.fields)
    rule = _EC_PARTNER_PACKAGING_RULES.get((reference_key, coefficient))
    if rule is None:
        return None

    candidates = [
        vertex
        for vertex in local_vertices
        if vertex.key == rule.partner_key
        and coefficient in dict(vertex.head_counts)
    ]
    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    candidate_orders = _charge_conjugation_candidate_orders(
        reference.fields,
        candidate.local_names,
    )
    if not candidate_orders:
        return None

    local_rule = _candidate_order_rule_sum(
        lagrangian=lagrangian,
        field_map=field_map,
        reference_fields=reference.fields,
        candidate_orders=candidate_orders,
        antisymmetric_duplicates=rule.antisymmetric_duplicates,
    )
    local_report = _canonical_report_for_coefficient_head(
        local_rule,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    transformed_report = _normalize_ec_charge_conjugation_report(
        local_report,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
        mode="direct",
        phase=rule.phase,
    )
    comparison = _coefficient_comparison_from_reports(
        coefficient,
        transformed_report,
        feynrules_report,
    )
    if not comparison.matches:
        return None

    duplicate_mode = (
        "antisymmetric duplicate-leg sum"
        if rule.antisymmetric_duplicates
        else "symmetric duplicate-leg sum"
    )
    return (
        comparison,
        (
            f"{coefficient} matched via pinned charge-conjugation partner "
            f"`{candidate.key}` using direct CC packaging, phase {rule.phase:+d}, "
            f"and {duplicate_mode} ({rule.source})."
        ),
    )



def _compare_smeft2_canonical_coefficient_maps(
    feynpy_rule: Expression | str,
    feynrules_rule: Expression | str,
    *,
    coefficients: Iterable[str],
    external_indices,
    max_dummy_permutations: int = 50_000,
) -> dict[str, CanonicalCoefficientComparison]:
    """Compare SMEFT2 rows by coefficient-head-filtered canonical maps."""

    feynpy_expression = (
        Expression.parse(feynpy_rule)
        if isinstance(feynpy_rule, str)
        else feynpy_rule
    )
    feynrules_expression = (
        Expression.parse(feynrules_rule)
        if isinstance(feynrules_rule, str)
        else feynrules_rule
    )

    comparisons = {}
    for coefficient in coefficients:
        feynpy_report = _canonical_report_for_coefficient_head(
            feynpy_expression,
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=max_dummy_permutations,
        )
        if coefficient.startswith("alphaEc"):
            feynpy_report = _normalize_ec_charge_conjugation_report(
                feynpy_report,
                coefficient=coefficient,
                external_indices=external_indices,
                max_dummy_permutations=max_dummy_permutations,
                mode="crossed",
                phase=-1,
            )
        feynrules_report = _canonical_report_for_coefficient_head(
            feynrules_expression,
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=max_dummy_permutations,
        )
        comparisons[coefficient] = _coefficient_comparison_from_reports(
            coefficient,
            feynpy_report,
            feynrules_report,
        )
    return comparisons


def _replace_tensdot_chains(text: str) -> str:
    output = []
    position = 0
    chain_id = 0
    while True:
        start = text.find("TensDot[", position)
        if start == -1:
            output.append(text[position:])
            return "".join(output)

        output.append(text[position:start])
        inner_open = start + len("TensDot")
        inner_close = _find_matching_square(text, inner_open)
        if inner_close + 1 >= len(text) or text[inner_close + 1] != "[":
            output.append(text[start : inner_close + 1])
            position = inner_close + 1
            continue

        spin_open = inner_close + 1
        spin_close = _find_matching_square(text, spin_open)
        chain_id += 1
        items = _split_top_level_commas(text[inner_open + 1 : inner_close])
        spin_args = _split_top_level_commas(text[spin_open + 1 : spin_close])
        if len(spin_args) != 2:
            raise ValueError(f"Unsupported FeynRules TensDot spin args: {spin_args}")
        output.append(
            _spinor_chain_replacement(
                items,
                spin_args[0],
                spin_args[1],
                chain_id=chain_id,
            )
        )
        position = spin_close + 1


def parse_smeft2_matter_rule(
    rule: str,
    *,
    projector_as_dirac_c: bool = False,
) -> Expression:
    """Parse SMEFT2 two-fermion FeynRules tensor syntax into native tensors.

    The SMEFT2 model represents chirality in the field class itself, while the
    FeynRules export prints explicit ``ProjM``/``ProjP`` factors. This parser
    therefore removes those projectors from gamma chains and maps bare
    projector bilinears to the spinor metric.  The Weinberg charge-conjugation
    packaging comparison is the exception: there FeynRules' same-chirality
    projector row is compared to FeynPy's mixed ``LLbar C LL`` packaging, so
    bare projectors are mapped to the antisymmetric Dirac charge-conjugation
    tensor.
    """

    text = _rewrite_feynrules_indices(rule)
    text = _rewrite_feynrules_indexed_parameters(text)
    _replace_scalar_product.counter = 0

    text = _replace_tensdot_chains(text)
    projector_tensor = (
        dirac_charge_conjugation if projector_as_dirac_c else spinor_metric
    )
    text = re.sub(
        r"Proj[MP]\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: projector_tensor(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ga\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gamma_matrix(
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(1).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"IndexDelta\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: _metric_for_index_delta(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"ME\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_metric(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"FV\[(\d+),\s*([^\[\]]+)\]",
        lambda match: pcomp(
            S(f"q{match.group(1)}"),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(r"SP\[(\d+),\s*(\d+)\]", _replace_scalar_product, text)
    text = re.sub(
        r"T\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ta\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"(?:f|fsu3)\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"fsu2\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_levi_civita(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(4).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_eps2(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = text.replace("Sqrt[2]", "(2)^(1/2)")
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported SMEFT2 FeynRules matter syntax remains after parsing: "
            f"{text}"
        )

    return Expression.parse(text).cancel().expand()


@dataclass(frozen=True)
class LocalVertex:
    key: str
    signature: tuple[str, ...]
    local_names: tuple[str, ...]
    feynpy_names: tuple[str, ...]
    arity: int
    term_count: int
    sectors: tuple[str, ...]
    heads: tuple[str, ...]
    head_counts: tuple[tuple[str, int], ...]
    rule: str


def _comparison_field_map(bundle) -> dict[str, object]:
    field_map = {}
    for local_name, reference_name in FIELD_NAME_MAP.items():
        if local_name.endswith(".bar"):
            field_map[reference_name] = bundle.fields[local_name[:-4]].bar
        else:
            field_map[reference_name] = bundle.fields[local_name]
    return field_map


def _exact_symbolic_family(fields: Iterable[str]) -> str:
    fermion_count = sum(name in REFERENCE_FERMION_NAMES for name in fields)
    return {
        0: "BOSONIC",
        2: "TWO_FERMION",
        4: "FOUR_FERMION",
    }.get(fermion_count, "UNCLASSIFIED")


def _unsupported_exact_symbolic_detail(family: str) -> str:
    return {
        "TWO_FERMION": (
            "Exact symbolic comparison is enabled for shared SMEFT2 two-fermion "
            "rows; this row has no literal local signature or falls outside that "
            "shared-signature layer."
        ),
        "FOUR_FERMION": (
            "Exact symbolic comparison is enabled for shared SMEFT2 four-fermion "
            "rows; this row has no literal local signature or falls outside that "
            "shared-signature layer."
        ),
        "UNCLASSIFIED": (
            "Exact symbolic comparison is not enabled for this field-content class."
        ),
        "BOSONIC": (
            "Bosonic exact symbolic comparison should have been attempted for this row."
        ),
    }[family]


def _bosonic_exact_symbolic_rows(
    references: Iterable[FeynRulesVertex],
    bundle,
) -> dict[str, dict[str, str]]:
    bosonic_references = tuple(
        reference
        for reference in references
        if _exact_symbolic_family(reference.fields) == "BOSONIC"
    )
    if not bosonic_references:
        return {}

    report = compare_feynrules_bosonic_vertices(
        bundle.model.lagrangian(),
        bosonic_references,
        field_map=_comparison_field_map(bundle),
        feynpy_name_aliases=FIELD_NAME_MAP,
    )
    status_map = {
        "MATCH": "EXACT_MATCH",
        "MISMATCH": "EXACT_MISMATCH",
        "MISSING_FEYNPY": "EXACT_NO_LOCAL_SIGNATURE",
        "MISSING_FIELD_MAP": "EXACT_ERROR",
        "ERROR": "EXACT_ERROR",
    }
    return {
        _name_key(row.reference.fields): {
            "family": "BOSONIC",
            "status": status_map[row.status],
            "detail": row.detail,
        }
        for row in report.rows
    }


def _fermion_exact_symbolic_row(
    *,
    reference: FeynRulesVertex,
    local: LocalVertex | None,
    local_vertices: Iterable[LocalVertex],
    reference_heads: set[str],
    local_heads: set[str],
    head_count_status: str,
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, str] | None:
    family = _exact_symbolic_family(reference.fields)
    if family not in {"TWO_FERMION", "FOUR_FERMION"}:
        return None
    if local is None:
        return None

    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": "Could not infer external indices for fermion row.",
        }

    coefficients = tuple(
        sorted(
            head
            for head in reference_heads | local_heads
            if head.startswith("alpha")
        )
    )
    if not coefficients:
        return None

    try:
        local_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in reference.fields),
            simplify=True,
        )
        reference_rule = parse_smeft2_matter_rule(reference.rule)
        comparisons = _compare_smeft2_canonical_coefficient_maps(
            local_rule,
            reference_rule,
            coefficients=coefficients,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
        pinned_partner_details = []
        unresolved_ec = []
        mismatched_ec = [
            coefficient
            for coefficient, comparison in comparisons.items()
            if coefficient.startswith("alphaEc") and not comparison.matches
        ]
        if mismatched_ec:
            reference_expression = parse_smeft2_matter_rule(reference.rule)
            for coefficient in mismatched_ec:
                feynrules_report = _canonical_report_for_coefficient_head(
                    reference_expression,
                    coefficient=coefficient,
                    external_indices=external_indices,
                    max_dummy_permutations=2_000_000,
                )
                partner_match = _ec_partner_packaging_comparison(
                    reference=reference,
                    coefficient=coefficient,
                    feynrules_report=feynrules_report,
                    local_vertices=local_vertices,
                    lagrangian=lagrangian,
                    field_map=field_map,
                    external_indices=external_indices,
                    max_dummy_permutations=2_000_000,
                )
                if partner_match is None:
                    unresolved_ec.append(coefficient)
                    continue
                comparisons[coefficient], detail = partner_match
                pinned_partner_details.append(detail)
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if all(comparison.matches for comparison in comparisons.values()):
        if pinned_partner_details:
            return {
                "family": family,
                "status": "MATCH_MODULO_CC_PACKAGING",
                "detail": (
                    "Canonical tensor-monomial maps agree after pinned Ec "
                    "charge-conjugation partner packaging for "
                    f"{len(pinned_partner_details)} coefficient sector(s). "
                    "No phase or duplicate-leg symmetry was searched at "
                    "acceptance time: each transform came from the explicit "
                    "Ec packaging rule table. "
                    + " ".join(pinned_partner_details)
                    + " Raw head-count status before exact-proof "
                    f"classification was {head_count_status}."
                ),
            }
        return {
            "family": family,
            "status": "EXACT_MATCH",
            "detail": (
                "Canonical tensor-monomial maps agree for all "
                f"{len(comparisons)} coefficient sector(s); raw head-count "
                "status before exact-proof classification was "
                f"{head_count_status}."
            ),
        }

    mismatched = tuple(
        coefficient
        for coefficient, comparison in comparisons.items()
        if not comparison.matches
    )
    if (
        unresolved_ec
        and mismatched
        and all(coefficient.startswith("alphaEc") for coefficient in mismatched)
    ):
        return {
            "family": family,
            "status": "UNRESOLVED_CC_PACKAGING",
            "detail": (
                "Direct same-signature canonical maps disagree for "
                f"{', '.join(mismatched)}. The comparison has no successful "
                "pinned Ec partner-packaging rule for "
                f"{', '.join(unresolved_ec)}, so the row remains unresolved "
                "rather than being accepted through a searched phase/symmetry "
                "choice. Raw head-count status before exact-proof "
                f"classification was {head_count_status}."
            ),
        }

    return {
        "family": family,
        "status": "EXACT_MISMATCH",
        "detail": (
            "Canonical tensor-monomial maps differ for coefficient sector(s): "
            + ", ".join(mismatched)
        ),
    }


def _weinberg_packaged_field_orders(
    reference_fields: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if sorted(reference_fields) == ["Phi", "Phi", "lL", "lL"]:
        fermion_name = "lL"
        scalar_name = "Phi"
    elif sorted(reference_fields) == ["Phibar", "Phibar", "lLbar", "lLbar"]:
        fermion_name = "lLbar"
        scalar_name = "Phibar"
    else:
        return None

    fermion_slots = [
        slot for slot, name in enumerate(reference_fields) if name == fermion_name
    ]
    if len(fermion_slots) != 2:
        return None
    if any(
        name != scalar_name
        for slot, name in enumerate(reference_fields)
        if slot not in fermion_slots
    ):
        return None

    first_assignment = list(reference_fields)
    first_assignment[fermion_slots[0]] = "lLbar"
    first_assignment[fermion_slots[1]] = "lL"

    second_assignment = list(reference_fields)
    second_assignment[fermion_slots[0]] = "lL"
    second_assignment[fermion_slots[1]] = "lLbar"

    return tuple(first_assignment), tuple(second_assignment)


def _weinberg_cc_exact_symbolic_row(
    *,
    reference: FeynRulesVertex,
    reference_heads: set[str],
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, str] | None:
    family = _exact_symbolic_family(reference.fields)
    if family != "TWO_FERMION" or reference_heads != {"alphaWeinberg"}:
        return None

    packaged_orders = _weinberg_packaged_field_orders(reference.fields)
    if packaged_orders is None:
        return None

    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": "Could not infer external indices for Weinberg row.",
        }

    try:
        first_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[0]),
            simplify=True,
        )
        second_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[1]),
            simplify=True,
        )
        # Swapping which external lepton is represented by the charge-conjugate
        # FeynPy leg swaps the two C-matrix spinor slots.  Since C is
        # antisymmetric, the same-chirality FeynRules rule corresponds to the
        # antisymmetrized packaged local rule.
        local_rule = (first_rule - second_rule).cancel().expand()
        reference_rule = parse_smeft2_matter_rule(
            reference.rule,
            projector_as_dirac_c=True,
        )
        comparisons = _compare_smeft2_canonical_coefficient_maps(
            local_rule,
            reference_rule,
            coefficients=("alphaWeinberg",),
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    comparison = comparisons["alphaWeinberg"]
    if comparison.matches:
        return {
            "family": family,
            "status": "MATCH_MODULO_CC_PACKAGING",
            "detail": (
                "Weinberg charge-conjugation packaging match (sign pinned): the "
                "same-chirality FeynRules row equals the antisymmetrized "
                "FeynPy mixed `lLbar,lL` assignment pair `FeynPy(lLbar,lL) - "
                "FeynPy(lL,lLbar)`, with `ProjM/ProjP` mapped to the "
                "antisymmetric Dirac charge-conjugation tensor. The relative "
                "minus sign is derived from the antisymmetry of the "
                "charge-conjugation matrix, not searched, so the canonical-map "
                "equality is exact modulo the explicitly tracked CC packaging."
            ),
        }

    return {
        "family": family,
        "status": "EXACT_MISMATCH",
        "detail": (
            "Weinberg charge-conjugation packaging canonical maps differ for "
            "`alphaWeinberg`."
        ),
    }


def _parse_weinberg_fermion_flow_rule(rule: str) -> Expression:
    """Parse a Weinberg FeynRules row into the sidecar fermion-flow basis."""

    text = _rewrite_feynrules_indices(rule)
    text = _rewrite_feynrules_indexed_parameters(text)
    text = re.sub(
        r"ProjM\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: S("PL")(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjP\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: S("PR")(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_eps2(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported Weinberg FeynRules syntax remains after parsing: "
            f"{text}"
        )
    return Expression.parse(text).cancel().expand()


def _weinberg_projector_head(reference_fields: tuple[str, ...]) -> str:
    if sorted(reference_fields) == ["Phi", "Phi", "lL", "lL"]:
        return "PL"
    if sorted(reference_fields) == ["Phibar", "Phibar", "lLbar", "lLbar"]:
        return "PR"
    raise ValueError(f"Unsupported Weinberg reference fields: {reference_fields!r}")


def _replace_dirac_c_with_weinberg_flow(
    expression: Expression,
    *,
    projector_head: str,
) -> Expression:
    left, right = S("weinberg_spin_left_", "weinberg_spin_right_")
    return expression.replace(
        dirac_charge_conjugation(left, right),
        S(projector_head)(left, right),
    ).cancel().expand()


def _weinberg_external_indices(reference: FeynRulesVertex, field_map: dict[str, object]):
    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        raise ValueError(f"Could not infer external indices for {reference.fields!r}.")
    return external_indices


def _reconstructed_weinberg_flow_rule(
    *,
    reference: FeynRulesVertex,
    lagrangian,
    field_map: dict[str, object],
    sign: int = -1,
) -> Expression:
    """Return the Weinberg sidecar rule in the common same-chirality order.

    ``sign=-1`` gives the charge-conjugation antisymmetric combination
    ``first - second``. ``sign=+1`` is kept for regression tests; it must not
    match the same-chirality FeynRules row.
    """

    packaged_orders = _weinberg_packaged_field_orders(reference.fields)
    if packaged_orders is None:
        raise ValueError(f"Unsupported Weinberg fields: {reference.fields!r}")
    if sign not in {-1, 1}:
        raise ValueError(f"Unsupported Weinberg reconstruction sign {sign!r}.")

    first_rule = lagrangian.feynman_rule(
        *(field_map[name] for name in packaged_orders[0]),
        simplify=True,
    )
    second_rule = lagrangian.feynman_rule(
        *(field_map[name] for name in packaged_orders[1]),
        simplify=True,
    )
    combined = (first_rule + Expression.num(sign) * second_rule).cancel().expand()

    # Canonicalize while the spinor pair is still the antisymmetric C tensor.
    # This maps the second mixed ordering into the same external spinor order
    # and produces the transposed flavor coefficient without assuming symmetry.
    canonical_c_report = _canonical_report_for_coefficient_head(
        combined,
        coefficient="alphaWeinberg",
        external_indices=_weinberg_external_indices(reference, field_map),
        max_dummy_permutations=2_000_000,
    )
    canonical_c_rule = _expression_from_canonical_map(canonical_c_report.map)
    return _replace_dirac_c_with_weinberg_flow(
        canonical_c_rule,
        projector_head=_weinberg_projector_head(reference.fields),
    )


def _weinberg_canonical_report(
    expression: Expression,
    *,
    external_indices,
) -> CanonicalMonomialReport:
    return canonical_tensor_monomial_report(
        expression.cancel().expand(),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )


def _weinberg_canonical_zero(
    expression: Expression,
    *,
    external_indices,
) -> bool:
    return not _weinberg_canonical_report(
        expression,
        external_indices=external_indices,
    ).map


def _weinberg_coefficient_expression(text: str) -> Expression:
    if text.startswith("conj(") and text.endswith(")"):
        inner = text[len("conj(") : -1]
        return S("conj")(Expression.parse(inner))
    return Expression.parse(text)


def _weinberg_coefficient_checks(
    *,
    feynpy_rule: Expression,
    feynrules_rule: Expression,
    coefficient_texts: tuple[str, ...],
    external_indices,
) -> list[dict[str, object]]:
    checks = []
    for coefficient_text in coefficient_texts:
        coefficient = _weinberg_coefficient_expression(coefficient_text)
        feynpy_coefficient = feynpy_rule.coefficient(coefficient).cancel().expand()
        feynrules_coefficient = (
            feynrules_rule.coefficient(coefficient).cancel().expand()
        )
        difference = (feynpy_coefficient - feynrules_coefficient).cancel().expand()
        checks.append(
            {
                "coefficient": coefficient_text,
                "matches": _weinberg_canonical_zero(
                    difference,
                    external_indices=external_indices,
                ),
                "feynpy_coefficient": feynpy_coefficient.to_canonical_string(),
                "feynrules_coefficient": feynrules_coefficient.to_canonical_string(),
                "difference": difference.to_canonical_string(),
            }
        )
    return checks


def _weinberg_reference_vertices(
    reference_path: Path = REFERENCE,
) -> tuple[FeynRulesVertex, FeynRulesVertex]:
    references_by_key = {
        _name_key(reference.fields): reference
        for reference in load_feynrules_json(reference_path)
    }
    return (
        references_by_key["Phi|Phi|lL|lL"],
        references_by_key["Phibar|Phibar|lLbar|lLbar"],
    )


def compare_reconstructed_weinberg(
    reference_path: Path = REFERENCE,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build and directly compare the Weinberg sidecar export.

    This is deliberately separate from the existing SMEFT2 aggregate comparison.
    It only reconstructs the two same-chirality Weinberg rows from the two
    ordered mixed FeynPy calls and compares them to the corresponding
    FeynRules rows in a compact fermion-flow basis.
    """

    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)

    vertices = []
    comparison_rows = []
    for reference in _weinberg_reference_vertices(reference_path):
        key = _name_key(reference.fields)
        external_indices = _weinberg_external_indices(reference, field_map)
        feynpy_rule = _reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=-1,
        )
        wrong_sign_rule = _reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=1,
        )
        feynrules_rule = _parse_weinberg_fermion_flow_rule(reference.rule)
        difference = (feynpy_rule - feynrules_rule).cancel().expand()
        wrong_sign_difference = (wrong_sign_rule - feynrules_rule).cancel().expand()
        coefficient_texts = (
            (
                "alphaWeinberg(f1,f2)",
                "alphaWeinberg(f2,f1)",
            )
            if key == "Phi|Phi|lL|lL"
            else (
                "conj(alphaWeinberg(f1,f2))",
                "conj(alphaWeinberg(f2,f1))",
            )
        )
        coefficient_checks = _weinberg_coefficient_checks(
            feynpy_rule=feynpy_rule,
            feynrules_rule=feynrules_rule,
            coefficient_texts=coefficient_texts,
            external_indices=external_indices,
        )
        matches = _weinberg_canonical_zero(
            difference,
            external_indices=external_indices,
        )
        wrong_sign_matches = _weinberg_canonical_zero(
            wrong_sign_difference,
            external_indices=external_indices,
        )
        packaged_orders = _weinberg_packaged_field_orders(reference.fields)
        vertices.append(
            {
                "key": key,
                "fields": list(reference.fields),
                "source_orders": {
                    "first": list(packaged_orders[0]),
                    "second": list(packaged_orders[1]),
                    "combination": "first - second",
                },
                "spinor_representation": _weinberg_projector_head(
                    reference.fields
                ),
                "flavor_structures": list(coefficient_texts),
                "rule": feynpy_rule.to_canonical_string(),
            }
        )
        comparison_rows.append(
            {
                "key": key,
                "fields": list(reference.fields),
                "matches": matches,
                "wrong_sign_matches": wrong_sign_matches,
                "coefficient_checks": coefficient_checks,
                "difference": difference.to_canonical_string(),
                "wrong_sign_difference": wrong_sign_difference.to_canonical_string(),
                "feynpy_rule": feynpy_rule.to_canonical_string(),
                "feynrules_rule": feynrules_rule.to_canonical_string(),
            }
        )

    report = {
        "generated_on": date.today().isoformat(),
        "comparison_level": (
            "Weinberg-only sidecar comparison. FeynPy mixed charge-conjugation "
            "orders are reconstructed as first - second in the common "
            "same-chirality external-leg order, then compared directly to the "
            "FeynRules lL lL Phi Phi and lLbar lLbar Phibar Phibar rows in a "
            "compact PL/PR fermion-flow basis."
        ),
        "summary": {
            "reference_vertices": len(comparison_rows),
            "direct_matches": sum(row["matches"] for row in comparison_rows),
            "wrong_sign_matches": sum(
                row["wrong_sign_matches"] for row in comparison_rows
            ),
            "coefficient_checks": sum(
                len(row["coefficient_checks"]) for row in comparison_rows
            ),
            "coefficient_matches": sum(
                sum(check["matches"] for check in row["coefficient_checks"])
                for row in comparison_rows
            ),
        },
        "vertices": comparison_rows,
    }
    return report, vertices


def _name_key(names: Iterable[str]) -> str:
    return "|".join(sorted(names))


def _normalize_local_name(name: str) -> str:
    try:
        return FIELD_NAME_MAP[name]
    except KeyError as exc:
        raise ValueError(f"No FeynRules name mapping for local field {name!r}") from exc


def _parameter_head_counts_from_text(
    text: str,
    parameter_names: Iterable[str],
) -> dict[str, int]:
    counts = Counter(re.findall(r"\balpha[A-Za-z0-9]+(?=\[|\(|\b)", text))
    for name in parameter_names:
        if name.startswith("alpha"):
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            counts[name] += len(
                re.findall(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                    text,
                )
            )
    return dict(sorted((head, count) for head, count in counts.items() if count))


def _parameter_heads_from_text(text: str, parameter_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(_parameter_head_counts_from_text(text, parameter_names))


# Field metadata index kinds map onto the canonical index groups. Generation
# (flavor) indices are carried on the ``cof(3)`` representation just like color
# fundamentals, so they are kept fixed in the ``color_fund`` group; their labels
# (``f1``, ``f2``, ...) never collide with the color labels (``c1``, ...).
_EXTERNAL_INDEX_GROUP_BY_KIND = {
    "lorentz": "lorentz",
    "color_adj": "color_adjoint",
    "color_fund": "color_fund",
    "spinor": "spinor",
    "weak_fund": "weak_fund",
    "weak_adj": "weak_adjoint",
    "generation": "color_fund",
}


def _external_index_set_from_fields(fields):
    """Return the canonical external-index set from field metadata.

    The external (leg) indices are fixed by the field content and their leg
    position: leg ``k`` contributes ``prefix{k}`` for each of its indices, which
    matches the labelling that :meth:`feynman_rule` emits for the same field
    order. Using metadata (rather than guessing free indices from a single term)
    is robust to term ordering, so canonical collection never accidentally
    renames a genuine external index and over-cancels distinct terms.
    """

    groups: dict[str, list] = {}
    for slot, field in enumerate(fields, start=1):
        base = field.field if hasattr(field, "field") else field
        for index in getattr(base, "indices", ()):
            kind = getattr(index, "kind", None)
            prefix = getattr(index, "prefix", None)
            group = _EXTERNAL_INDEX_GROUP_BY_KIND.get(kind)
            if group is None or not prefix:
                return None
            groups.setdefault(group, []).append(S(f"{prefix}{slot}"))
    return canonical_external_index_set(
        lorentz=tuple(groups.get("lorentz", ())),
        color_adjoint=tuple(groups.get("color_adjoint", ())),
        color_fund=tuple(groups.get("color_fund", ())),
        spinor=tuple(groups.get("spinor", ())),
        weak_fund=tuple(groups.get("weak_fund", ())),
        weak_adjoint=tuple(groups.get("weak_adjoint", ())),
    )


def _canonical_head_counts_from_rule(
    rule,
    parameter_names: Iterable[str],
    external_indices,
) -> dict[str, int]:
    """Coefficient-head multiplicity after canonical (algebraic) collection.

    The raw FeynPy rule keeps terms that only vanish once tensor identities are
    applied (e.g. ``epsilon^{mu nu rho sigma} B_rho B_sigma = 0`` from a double
    covariant-derivative expansion). Counting heads from the raw text therefore
    reports spurious coefficient heads (``alphaEuH``/``alphaEdH``/``alphaEeH``)
    that carry an identically-zero coefficient. Canonicalizing the rule
    term-by-term and collecting like terms drops those zero coefficients, so the
    surviving heads reflect the genuine algebraic operator content.
    """

    expression = rule.cancel().expand()
    monomial_map = canonical_tensor_monomial_map(
        expression,
        external_indices=external_indices,
    )
    canonical_text = " ".join(
        coefficient.to_canonical_string() for coefficient in monomial_map.values()
    )
    return _parameter_head_counts_from_text(canonical_text, parameter_names)


def _reference_heads(
    reference: FeynRulesVertex,
    parameter_names: Iterable[str],
) -> tuple[str, ...]:
    return _parameter_heads_from_text(reference.rule, parameter_names)


def _reference_head_counts(
    reference: FeynRulesVertex,
    parameter_names: Iterable[str],
) -> dict[str, int]:
    return _parameter_head_counts_from_text(reference.rule, parameter_names)


def _head_count_delta(
    reference_counts: dict[str, int],
    local_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    delta = {}
    for head in sorted(set(reference_counts) | set(local_counts)):
        reference_count = reference_counts.get(head, 0)
        local_count = local_counts.get(head, 0)
        if reference_count != local_count:
            delta[head] = {
                "reference": reference_count,
                "feynpy": local_count,
            }
    return delta


def _benign_head_count_delta_reasons(
    key: str,
    head_count_delta: dict[str, dict[str, int]],
) -> dict[str, str]:
    reasons = {}
    for head, counts in head_count_delta.items():
        reason = BENIGN_HEAD_COUNT_DELTAS.get((key, head))
        if reason is None:
            continue
        if counts["feynpy"] <= counts["reference"]:
            continue
        reasons[head] = reason
    return reasons


def _exact_symbolic_head_count_delta_reasons(
    *,
    exact_symbolic_status: str,
    head_count_delta: dict[str, dict[str, int]],
    existing_reasons: dict[str, str],
) -> dict[str, str]:
    reason = {
        "EXACT_MATCH": EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE,
        "MATCH_MODULO_CC_PACKAGING": PINNED_CC_CANONICAL_EQUIVALENCE,
    }.get(exact_symbolic_status)
    if reason is None:
        return {}
    return {
        head: reason
        for head in head_count_delta
        if head not in existing_reasons
    }


def _head_count_status(
    *,
    has_local_signature: bool,
    head_count_delta: dict[str, dict[str, int]],
    benign_reasons: dict[str, str],
) -> str:
    if not has_local_signature:
        return "NO_LOCAL_SIGNATURE"
    if not head_count_delta:
        return "COUNT_MATCH"
    if len(benign_reasons) == len(head_count_delta):
        return "COUNT_BENIGN_EXPANSION"
    if benign_reasons:
        return "COUNT_MIXED_BENIGN_AND_UNEXPLAINED"
    return "COUNT_MISMATCH"


def _canonical_map_external_indices(
    fields: tuple[str, ...],
    *,
    field_map: dict[str, object],
) -> frozenset[tuple[str, str]] | None:
    if _exact_symbolic_family(fields) != "BOSONIC":
        return None

    grouped: dict[str, list[object]] = {}
    for slot, name in enumerate(fields, start=1):
        field = field_map.get(name)
        if field is None:
            return None
        base = field.field if hasattr(field, "field") else field
        for index in getattr(base, "indices", ()):
            kind = getattr(index, "kind", None)
            group = CANONICAL_EXTERNAL_INDEX_GROUP_BY_KIND.get(kind)
            prefix = getattr(index, "prefix", None)
            if group is None or not prefix:
                return None
            grouped.setdefault(group, []).append(S(f"{prefix}{slot}"))

    return canonical_external_index_set(
        lorentz=tuple(grouped.get("lorentz", ())),
        color_adjoint=tuple(grouped.get("color_adjoint", ())),
        color_fund=tuple(grouped.get("color_fund", ())),
        spinor=tuple(grouped.get("spinor", ())),
        weak_fund=tuple(grouped.get("weak_fund", ())),
        weak_adjoint=tuple(grouped.get("weak_adjoint", ())),
    )


def _canonical_map_coefficients(
    reference_heads: set[str],
    local_heads: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            head
            for head in reference_heads & local_heads
            if head.startswith("alpha")
        )
    )


def _first_canonical_map_item(mapping) -> dict[str, str] | None:
    if not mapping:
        return None
    key = next(iter(mapping))
    return {
        "monomial": repr(key),
        "coefficient": mapping[key].cancel().expand().to_canonical_string(),
    }


def _first_coefficient_mismatch(mapping) -> dict[str, str] | None:
    if not mapping:
        return None
    key = next(iter(mapping))
    feynpy_coefficient, feynrules_coefficient = mapping[key]
    return {
        "monomial": repr(key),
        "feynpy_coefficient": feynpy_coefficient.cancel().expand().to_canonical_string(),
        "feynrules_coefficient": feynrules_coefficient.cancel().expand().to_canonical_string(),
    }


def _canonical_map_diagnostic(
    *,
    reference: FeynRulesVertex,
    local: LocalVertex | None,
    reference_heads: set[str],
    local_heads: set[str],
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, object]:
    external_indices = _canonical_map_external_indices(
        reference.fields,
        field_map=field_map,
    )
    if local is None or external_indices is None:
        return {
            "status": "CANONICAL_MAP_UNSUPPORTED",
            "coefficients": {},
            "error": "",
        }

    coefficients = _canonical_map_coefficients(reference_heads, local_heads)
    if not coefficients:
        return {
            "status": "CANONICAL_MAP_UNSUPPORTED",
            "coefficients": {},
            "error": "",
        }

    try:
        local_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in reference.fields),
            simplify=True,
        )
        comparisons = compare_canonical_coefficient_maps(
            local_rule,
            reference.rule,
            coefficients=coefficients,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "status": "CANONICAL_MAP_ERROR",
            "coefficients": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    coefficient_payload = {}
    for coefficient, comparison in comparisons.items():
        coefficient_payload[coefficient] = {
            "matches": comparison.matches,
            "feynpy_raw_terms": comparison.feynpy_raw_terms,
            "feynrules_raw_terms": comparison.feynrules_raw_terms,
            "feynpy_canonical_terms": comparison.feynpy_canonical_terms,
            "feynrules_canonical_terms": comparison.feynrules_canonical_terms,
            "feynpy_only_count": len(comparison.feynpy_only),
            "feynrules_only_count": len(comparison.feynrules_only),
            "coefficient_mismatch_count": len(comparison.coefficient_mismatches),
            "first_feynpy_only": _first_canonical_map_item(comparison.feynpy_only),
            "first_feynrules_only": _first_canonical_map_item(
                comparison.feynrules_only
            ),
            "first_coefficient_mismatch": _first_coefficient_mismatch(
                comparison.coefficient_mismatches
            ),
        }

    return {
        "status": (
            "CANONICAL_MAP_MATCH"
            if all(comparison.matches for comparison in comparisons.values())
            else "CANONICAL_MAP_MISMATCH"
        ),
        "coefficients": coefficient_payload,
        "error": "",
    }


def _local_vertices(parameter_names: Iterable[str]) -> tuple[LocalVertex, ...]:
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    rows: list[LocalVertex] = []
    for signature in lagrangian.vertex_signatures():
        if not 3 <= signature.arity <= 6:
            continue
        rule = lagrangian.feynman_rule(*signature.fields, simplify=True)
        rule_text = rule.cancel().expand().to_canonical_string()
        # Raw-text multiplicities feed the (apples-to-apples) raw head-count
        # diagnostic against the FeynRules reference text.
        raw_head_counts = _parameter_head_counts_from_text(rule_text, parameter_names)
        # The coefficient-head *set* (which drives the operator-content match)
        # is taken after canonical collection so that heads whose coefficient is
        # only zero by a tensor identity are not counted as genuine content.
        external_indices = _external_index_set_from_fields(signature.fields)
        if external_indices is None:
            canonical_head_counts = raw_head_counts
        else:
            try:
                canonical_head_counts = _canonical_head_counts_from_rule(
                    rule, parameter_names, external_indices
                )
            except Exception:
                canonical_head_counts = raw_head_counts
        local_names = tuple(_normalize_local_name(name) for name in signature.names)
        normalized_signature = tuple(sorted(local_names))
        rows.append(
            LocalVertex(
                key=_name_key(local_names),
                signature=normalized_signature,
                local_names=local_names,
                feynpy_names=signature.names,
                arity=signature.arity,
                term_count=signature.term_count,
                sectors=signature.sectors,
                heads=tuple(canonical_head_counts),
                head_counts=tuple(raw_head_counts.items()),
                rule=rule_text,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.arity, row.key)))


def _reason_for_status(status: str) -> str:
    return {
        "SHARED_HEADS_MATCH": (
            "Shared field multiset and identical coefficient-head set at the "
            "operator-content comparison level."
        ),
        "MISSING_SIGNATURE_OMITTED_DERIVATIVE_SECTORS": (
            "Reference signature is driven by derivative-sector coefficient "
            "families that are explicitly not lowered in local SMEFT2."
        ),
        "MISSING_SIGNATURE_WEINBERG_PACKAGING": (
            "Reference has same-chirality Weinberg signatures while the local "
            "model packages that operator with mixed bar/conjugation signatures."
        ),
        "MISSING_SIGNATURE": "Reference signature is absent from the local FeynPy output.",
        "SHARED_MISSING_OMITTED_HEADS": (
            "Shared field multiset, but the reference contains omitted "
            "derivative-sector coefficient heads."
        ),
        "SHARED_MISSING_OMITTED_HEADS_PLUS_LOCAL_EXTRA": (
            "Local rule misses omitted derivative-sector heads from the reference "
            "and also has additional local heads."
        ),
        "SHARED_LOCAL_PP_EXTRA": (
            "Local translation has extra pp-coefficient heads not present in the "
            "FeynRules reference for this signature."
        ),
        "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH": (
            "Difference is concentrated in charge-conjugated operator packaging."
        ),
        "SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH": (
            "Operator content matches once the reference's charge-conjugated "
            "four-fermion head is credited to the FeynPy vertex that carries the "
            "same operator under the charge-conjugate (bar-flipped) bilinear "
            "packaging."
        ),
        "MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING": (
            "Reference Weinberg signature is matched by the FeynPy vertex that "
            "carries the same operator under the charge-conjugate (bar-flipped) "
            "packaging."
        ),
        "SHARED_MIXED_OPERATOR_CONTENT": (
            "Shared field multiset, but both sides contain coefficient heads absent "
            "from the other."
        ),
        "SHARED_REFERENCE_EXTRA_HEADS": (
            "Shared field multiset, but FeynRules has coefficient heads absent "
            "from local FeynPy."
        ),
        "SHARED_LOCAL_EXTRA_HEADS": (
            "Shared field multiset, but local FeynPy has coefficient heads absent "
            "from FeynRules."
        ),
        "FEYNPY_ONLY_WEINBERG_PACKAGING": (
            "Local model emits a Weinberg bar/conjugation signature not present "
            "as a separate FeynRules signature."
        ),
        "FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING": (
            "Local model emits a bar/charge-conjugation packaging not present "
            "as a separate FeynRules signature."
        ),
        "FEYNPY_ONLY_SIGNATURE": "Local signature is absent from the FeynRules reference.",
        "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER": (
            "FeynPy-only signature that is the charge-conjugate (bar-flipped) "
            "packaging of a FeynRules reference operator; it is cross-linked to "
            "that reference row and is not an unexplained residual."
        ),
        "FEYNPY_ONLY_ALGEBRAICALLY_ZERO": (
            "FeynPy-only signature whose canonical coefficient-head set is empty: "
            "the rule cancels to zero under canonical tensor identities, so it is "
            "a zero-signature artifact rather than residual operator content."
        ),
    }[status]


def _shared_status(reference_heads: set[str], local_heads: set[str]) -> str:
    if reference_heads == local_heads:
        return "SHARED_HEADS_MATCH"

    reference_extra = reference_heads - local_heads
    local_extra = local_heads - reference_heads
    if reference_extra & OMITTED_COEFFICIENT_HEADS:
        return (
            "SHARED_MISSING_OMITTED_HEADS_PLUS_LOCAL_EXTRA"
            if local_extra
            else "SHARED_MISSING_OMITTED_HEADS"
        )
    if reference_extra or local_extra:
        differing = reference_extra | local_extra
        if any(head.startswith("alphaEc") for head in differing):
            return "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH"
        if (
            local_extra
            and not reference_extra
            and all(head.endswith("pp") for head in local_extra)
        ):
            return "SHARED_LOCAL_PP_EXTRA"
        if reference_extra and local_extra:
            return "SHARED_MIXED_OPERATOR_CONTENT"
        if reference_extra:
            return "SHARED_REFERENCE_EXTRA_HEADS"
        return "SHARED_LOCAL_EXTRA_HEADS"
    raise AssertionError("unreachable shared comparison state")


def _missing_reference_status(reference_heads: set[str]) -> str:
    if "alphaWeinberg" in reference_heads:
        return "MISSING_SIGNATURE_WEINBERG_PACKAGING"
    if reference_heads & OMITTED_COEFFICIENT_HEADS:
        return "MISSING_SIGNATURE_OMITTED_DERIVATIVE_SECTORS"
    return "MISSING_SIGNATURE"


def _feynpy_only_status(local_heads: set[str]) -> str:
    if "alphaWeinberg" in local_heads:
        return "FEYNPY_ONLY_WEINBERG_PACKAGING"
    if "alphaOHud" in local_heads or any(head.startswith("alphaEc") for head in local_heads):
        return "FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING"
    return "FEYNPY_ONLY_SIGNATURE"


def compare(reference_path: Path = REFERENCE) -> tuple[dict[str, object], tuple[LocalVertex, ...]]:
    references = load_feynrules_json(reference_path)
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)
    parameter_names = set(bundle.parameters) | GENERIC_PARAMETER_NAMES

    local_vertices = _local_vertices(parameter_names)
    exact_symbolic_by_key = _bosonic_exact_symbolic_rows(references, bundle)
    local_by_key = {vertex.key: vertex for vertex in local_vertices}
    reference_keys = {_name_key(reference.fields) for reference in references}

    reference_rows = []
    status_counts: Counter[str] = Counter()
    for reference in sorted(
        references,
        key=lambda item: (_name_key(item.fields), item.identifier or 0),
    ):
        key = _name_key(reference.fields)
        reference_heads = set(_reference_heads(reference, parameter_names))
        reference_head_counts = _reference_head_counts(reference, parameter_names)
        local = local_by_key.get(key)
        if local is None:
            status = _missing_reference_status(reference_heads)
            local_heads: set[str] = set()
            local_head_counts: dict[str, int] = {}
            local_names: tuple[str, ...] = ()
            feynpy_names: tuple[str, ...] = ()
            sectors: tuple[str, ...] = ()
            term_count = 0
        else:
            # ``local.heads`` is the coefficient-head set after canonical
            # collection (spurious heads whose coefficient is only zero by a
            # tensor identity are removed). That collection is reliable for
            # dropping genuinely feynpy-only heads (e.g. the ``epsilon B B = 0``
            # heads from the double covariant-derivative expansion), but the
            # open-spinor canonicalization can be field-order dependent for
            # sigma-dipole rows, so we never let it drop a head that FeynRules
            # also carries. Shared heads are therefore always retained from the
            # raw multiplicity map, which protects against false
            # "reference-extra" verdicts.
            raw_local_heads = set(dict(local.head_counts))
            local_heads = set(local.heads) | (raw_local_heads & reference_heads)
            local_head_counts = dict(local.head_counts)
            status = _shared_status(reference_heads, local_heads)
            local_names = local.local_names
            feynpy_names = local.feynpy_names
            sectors = local.sectors
            term_count = local.term_count

        head_count_delta = _head_count_delta(reference_head_counts, local_head_counts)
        benign_head_count_delta_reasons = _benign_head_count_delta_reasons(
            key,
            head_count_delta,
        )
        pre_exact_head_count_status = _head_count_status(
            has_local_signature=local is not None,
            head_count_delta=head_count_delta,
            benign_reasons=benign_head_count_delta_reasons,
        )
        canonical_map = _canonical_map_diagnostic(
            reference=reference,
            local=local,
            reference_heads=reference_heads,
            local_heads=local_heads,
            lagrangian=lagrangian,
            field_map=field_map,
        )
        exact_symbolic_family = _exact_symbolic_family(reference.fields)
        exact_symbolic = exact_symbolic_by_key.get(key)
        if exact_symbolic is None:
            exact_symbolic = _weinberg_cc_exact_symbolic_row(
                reference=reference,
                reference_heads=reference_heads,
                lagrangian=lagrangian,
                field_map=field_map,
            )
        if exact_symbolic is None:
            exact_symbolic = _fermion_exact_symbolic_row(
                reference=reference,
                local=local,
                local_vertices=local_vertices,
                reference_heads=reference_heads,
                local_heads=local_heads,
                head_count_status=pre_exact_head_count_status,
                lagrangian=lagrangian,
                field_map=field_map,
            )
        if exact_symbolic is None:
            exact_symbolic = {
                "family": exact_symbolic_family,
                "status": "EXACT_UNSUPPORTED",
                "detail": _unsupported_exact_symbolic_detail(exact_symbolic_family),
            }
        benign_head_count_delta_reasons.update(
            _exact_symbolic_head_count_delta_reasons(
                exact_symbolic_status=exact_symbolic["status"],
                head_count_delta=head_count_delta,
                existing_reasons=benign_head_count_delta_reasons,
            )
        )
        unexplained_head_count_delta = {
            head: counts
            for head, counts in head_count_delta.items()
            if head not in benign_head_count_delta_reasons
        }
        head_count_status = _head_count_status(
            has_local_signature=local is not None,
            head_count_delta=head_count_delta,
            benign_reasons=benign_head_count_delta_reasons,
        )

        status_counts[status] += 1
        reference_rows.append(
            {
                "id": reference.identifier,
                "key": key,
                "fields": list(reference.fields),
                "legs": list(reference.legs),
                "signature": sorted(reference.fields),
                "arity": len(reference.fields),
                "reference_heads": sorted(reference_heads),
                "feynpy_heads": sorted(local_heads),
                "reference_head_counts": reference_head_counts,
                "feynpy_head_counts": local_head_counts,
                "head_count_delta": head_count_delta,
                "benign_head_count_delta_reasons": benign_head_count_delta_reasons,
                "unexplained_head_count_delta": unexplained_head_count_delta,
                "head_count_status": head_count_status,
                "canonical_map_status": canonical_map["status"],
                "canonical_map_coefficients": canonical_map["coefficients"],
                "canonical_map_error": canonical_map["error"],
                "exact_symbolic_family": exact_symbolic["family"],
                "exact_symbolic_status": exact_symbolic["status"],
                "exact_symbolic_detail": exact_symbolic["detail"],
                "feynrules_extra_heads": sorted(reference_heads - local_heads),
                "feynpy_extra_heads": sorted(local_heads - reference_heads),
                "local_names": list(local_names),
                "feynpy_names": list(feynpy_names),
                "local_term_count": term_count,
                "sectors": list(sectors),
                "status": status,
                "reason": _reason_for_status(status),
            }
        )

    # A FeynPy-only signature whose *canonical* coefficient-head set is empty is
    # algebraically zero: every raw monomial cancels under dummy relabeling and
    # intrinsic tensor symmetries (e.g. the U(1) piece of the ``O_Hud`` covariant
    # derivative in ``B|Phi|Phi|dR|uRbar``). These are not residual unmatched
    # operator content -- FeynRules correctly omits them -- so they are recorded
    # separately as zero-signature artifacts rather than counted as FeynPy-only
    # residuals.
    feynpy_only_rows = []
    feynpy_only_zero_rows = []
    for local in local_vertices:
        if local.key in reference_keys:
            continue
        row = {
            "key": local.key,
            "signature": list(local.signature),
            "local_names": list(local.local_names),
            "feynpy_names": list(local.feynpy_names),
            "arity": local.arity,
            "term_count": local.term_count,
            "sectors": list(local.sectors),
            "feynpy_heads": list(local.heads),
            "feynpy_head_counts": dict(local.head_counts),
        }
        if not local.heads:
            row["status"] = "FEYNPY_ONLY_ALGEBRAICALLY_ZERO"
            row["reason"] = _reason_for_status(row["status"])
            feynpy_only_zero_rows.append(row)
            continue
        row["status"] = _feynpy_only_status(set(local.heads))
        row["reason"] = _reason_for_status(row["status"])
        feynpy_only_rows.append(row)

    # ------------------------------------------------------------------
    # Charge-conjugation packaging reconciliation.
    #
    # FeynRules and FeynPy sometimes assign the bar (particle vs antiparticle
    # leg) differently for the *same* operator. FeynRules keeps the Weinberg
    # operator as ``Phi Phi lL lL`` and the four-fermion ``Ec`` operators with a
    # given bilinear bar assignment, whereas FeynPy packages the same operators
    # with the charge-conjugate bilinear (both members' bars flipped), e.g.
    # ``Phi Phi lL lLbar``. Such a pair carries the *same* operator head and a
    # bar-insensitive field content, and is the same vertex related by charge
    # conjugation ``psi1bar Gamma psi2 = psi2bar Gamma' psi1^C``. We pair each
    # such reference row with the FeynPy-only vertex that carries the matching
    # head under the charge-conjugate field content, so the operator is credited
    # as an algebraic match rather than a spurious reference-extra + FeynPy-only
    # split. This is applied only as a fallback for the charge-conjugation and
    # Weinberg mismatch buckets; it never touches already-matched rows.
    # ------------------------------------------------------------------
    # This pass is deliberately an *annotation overlay*: it records that a
    # reference operator is present in FeynPy under the charge-conjugate
    # packaging, but it does NOT alter the literal signature-coverage metrics
    # (``shared_signatures`` / ``reference_only_signatures`` /
    # ``feynpy_only_signatures``), which stay defined by exact field-multiset
    # overlap. The Weinberg reference rows still have no exact local signature
    # and remain reference-only; their FeynPy charge-conjugate partners still
    # have no exact reference signature and remain FeynPy-only. The overlay adds
    # a ``charge_conjugation_partner`` cross-link on both sides and a separate
    # ``charge_conjugation_packaging_matches`` metric for operator-content
    # matching modulo charge conjugation.
    def _cc_field_key(fields: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted(name[:-3] if name.endswith("bar") else name for name in fields)
        )

    cc_local_index: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in feynpy_only_rows:
        cc_local_index.setdefault(_cc_field_key(row["signature"]), []).append(row)

    consumed_feynpy_only_ids: set[int] = set()
    for row in reference_rows:
        if row["status"] not in (
            "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH",
            "MISSING_SIGNATURE_WEINBERG_PACKAGING",
        ):
            continue
        cc_extra = {
            head
            for head in row["feynrules_extra_heads"]
            if head.startswith("alphaEc") or head == "alphaWeinberg"
        }
        if not cc_extra:
            continue
        cc_key = _cc_field_key(row["fields"])
        partner = None
        for candidate in cc_local_index.get(cc_key, ()):
            if id(candidate) in consumed_feynpy_only_ids:
                continue
            if cc_extra <= set(candidate["feynpy_head_counts"]):
                partner = candidate
                break
        if partner is None:
            continue
        consumed_feynpy_only_ids.add(id(partner))
        # Cross-link both sides; keep the literal per-row head sets untouched.
        row["charge_conjugation_partner"] = partner["key"]
        row["charge_conjugation_matched_heads"] = sorted(cc_extra)
        row["operator_content_resolved_via_charge_conjugation"] = True
        row["status"] = (
            "MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING"
            if row["status"] == "MISSING_SIGNATURE_WEINBERG_PACKAGING"
            else "SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH"
        )
        row["reason"] = _reason_for_status(row["status"])
        partner["charge_conjugation_partner"] = row["key"]
        partner["status"] = "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
        partner["reason"] = _reason_for_status(partner["status"])

    # Status tallies over the final annotated rows (reference rows plus the
    # literal FeynPy-only rows and the separate zero-signature artifacts).
    status_counts = Counter(row["status"] for row in reference_rows)
    for row in feynpy_only_rows:
        status_counts[row["status"]] += 1
    for row in feynpy_only_zero_rows:
        status_counts[row["status"]] += 1

    charge_conjugation_matches = (
        status_counts["SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH"]
        + status_counts["MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING"]
    )

    # Literal signature coverage: a reference row is "shared" iff FeynPy emits an
    # exact same-field-multiset signature for it (``local is not None``, i.e. its
    # head-count status is not ``NO_LOCAL_SIGNATURE``). This is independent of the
    # charge-conjugation overlay above.
    shared = sum(
        1
        for row in reference_rows
        if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    )
    head_count_matches = sum(
        1
        for row in reference_rows
        if row["head_count_status"] == "COUNT_MATCH"
    )
    head_count_status_counts = Counter(
        row["head_count_status"]
        for row in reference_rows
        if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    )
    canonical_map_rows = [
        row
        for row in reference_rows
        if row["canonical_map_status"] != "CANONICAL_MAP_UNSUPPORTED"
    ]
    canonical_map_status_counts = Counter(
        row["canonical_map_status"]
        for row in canonical_map_rows
    )
    canonical_map_sector_count = sum(
        len(row["canonical_map_coefficients"])
        for row in canonical_map_rows
    )
    canonical_map_equal_sector_count = sum(
        sum(
            1
            for coefficient in row["canonical_map_coefficients"].values()
            if coefficient["matches"]
        )
        for row in canonical_map_rows
    )
    # Raw-output redundancy diagnostic: how many raw FeynPy monomials collapse
    # once dummy indices are canonically relabeled and intrinsic tensor
    # symmetries applied. This quantifies the "long output" problem on the
    # sectors where an exact canonical monomial count is available. The gap
    # (raw minus canonical) is entirely redundant surface form that FeynRules
    # already writes collected; it is not extra physics.
    canonical_map_feynpy_raw_terms = sum(
        coefficient["feynpy_raw_terms"]
        for row in canonical_map_rows
        for coefficient in row["canonical_map_coefficients"].values()
    )
    canonical_map_feynpy_canonical_terms = sum(
        coefficient["feynpy_canonical_terms"]
        for row in canonical_map_rows
        for coefficient in row["canonical_map_coefficients"].values()
    )
    exact_symbolic_rows = [
        row
        for row in reference_rows
        if row["exact_symbolic_status"] != "EXACT_UNSUPPORTED"
    ]
    exact_symbolic_status_counts = Counter(
        row["exact_symbolic_status"] for row in reference_rows
    )
    exact_symbolic_family_counts = Counter(
        row["exact_symbolic_family"] for row in reference_rows
    )
    shared_reference_rows = [
        row for row in reference_rows if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    ]
    benign_head_count_delta_heads = sum(
        len(row["benign_head_count_delta_reasons"])
        for row in shared_reference_rows
    )
    unexplained_head_count_delta_heads = sum(
        len(row["unexplained_head_count_delta"])
        for row in shared_reference_rows
    )
    matched = status_counts["SHARED_HEADS_MATCH"]
    report = {
        "generated_on": date.today().isoformat(),
        "reference": str(reference_path.relative_to(ROOT)),
        "local_model": str((MODEL_DIR / "SMEFT2.py").relative_to(ROOT)),
        "comparison_level": (
            "Signature coverage, coefficient-head content, and raw "
            "coefficient-head multiplicity diagnostics, plus exact symbolic "
            "comparison for all 184 FeynRules reference rows. Fermion exact "
            "comparison filters by indexed Wilson-coefficient head and keeps "
            "flavor order/conjugation in the canonical scalar coefficient, so "
            "it cannot pass vacuously for function-valued coefficients. "
            "Exact-symbolic rows are graded honestly: `EXACT_MATCH` means "
            "direct canonical-map equality with no row-specific packaging "
            "assumption; `MATCH_MODULO_CC_PACKAGING` means equality only after "
            "a charge-conjugation packaging transform whose sign/symmetry is "
            "derived (pinned), e.g. the antisymmetrized Weinberg rows; and "
            "`UNRESOLVED_CC_PACKAGING` means no pinned packaging rule is known "
            "or the pinned transform failed. The separate canonical tensor-map "
            "diagnostic remains the gauge-sector per-coefficient map for "
            "supported bosonic coefficient sectors."
        ),
        "summary": {
            "reference_vertex_count": len(references),
            "feynpy_signature_count_3_to_6": len(local_vertices),
            # Literal exact field-multiset signature coverage.
            "shared_signatures": shared,
            "reference_only_signatures": len(references) - shared,
            "feynpy_only_signatures": len(feynpy_only_rows),
            "feynpy_only_zero_signatures": len(feynpy_only_zero_rows),
            "feynpy_only_charge_conjugation_partners": sum(
                1
                for row in feynpy_only_rows
                if row["status"] == "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
            ),
            "feynpy_only_unexplained_signatures": sum(
                1
                for row in feynpy_only_rows
                if row["status"] != "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
            ),
            # Operator-content matching (coefficient-head set).
            "shared_head_matches": matched,
            "charge_conjugation_packaging_matches": charge_conjugation_matches,
            "operator_content_matches_including_cc": matched + charge_conjugation_matches,
            "shared_head_count_matches": head_count_matches,
            "shared_head_count_mismatches": shared - head_count_matches,
            "shared_head_count_benign_expansions": head_count_status_counts[
                "COUNT_BENIGN_EXPANSION"
            ],
            "shared_head_count_mixed_benign_unexplained": head_count_status_counts[
                "COUNT_MIXED_BENIGN_AND_UNEXPLAINED"
            ],
            "shared_head_count_unexplained_mismatches": (
                head_count_status_counts["COUNT_MISMATCH"]
                + head_count_status_counts["COUNT_MIXED_BENIGN_AND_UNEXPLAINED"]
            ),
            "canonical_map_supported_vertices": len(canonical_map_rows),
            "canonical_map_equal_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_MATCH"
            ],
            "canonical_map_unequal_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_MISMATCH"
            ],
            "canonical_map_error_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_ERROR"
            ],
            "canonical_map_supported_coefficient_sectors": canonical_map_sector_count,
            "canonical_map_equal_coefficient_sectors": canonical_map_equal_sector_count,
            "canonical_map_feynpy_raw_terms": canonical_map_feynpy_raw_terms,
            "canonical_map_feynpy_canonical_terms": canonical_map_feynpy_canonical_terms,
            "canonical_map_feynpy_redundant_terms": (
                canonical_map_feynpy_raw_terms - canonical_map_feynpy_canonical_terms
            ),
            "canonical_map_unequal_coefficient_sectors": (
                canonical_map_sector_count - canonical_map_equal_sector_count
            ),
            "canonical_map_status_counts": dict(
                sorted(canonical_map_status_counts.items())
            ),
            "exact_symbolic_supported_vertices": len(exact_symbolic_rows),
            "exact_symbolic_equal_vertices": exact_symbolic_status_counts[
                "EXACT_MATCH"
            ],
            "exact_symbolic_direct_match_vertices": exact_symbolic_status_counts[
                "EXACT_MATCH"
            ],
            "cc_packaging_pinned_match_vertices": exact_symbolic_status_counts[
                "MATCH_MODULO_CC_PACKAGING"
            ],
            "cc_packaging_unresolved_vertices": exact_symbolic_status_counts[
                "UNRESOLVED_CC_PACKAGING"
            ],
            "exact_symbolic_unequal_vertices": exact_symbolic_status_counts[
                "EXACT_MISMATCH"
            ],
            "exact_symbolic_missing_local_vertices": exact_symbolic_status_counts[
                "EXACT_NO_LOCAL_SIGNATURE"
            ],
            "exact_symbolic_error_vertices": exact_symbolic_status_counts[
                "EXACT_ERROR"
            ],
            "exact_symbolic_status_counts": dict(
                sorted(exact_symbolic_status_counts.items())
            ),
            "exact_symbolic_family_counts": dict(
                sorted(exact_symbolic_family_counts.items())
            ),
            "benign_head_count_delta_heads": benign_head_count_delta_heads,
            "unexplained_head_count_delta_heads": unexplained_head_count_delta_heads,
            "head_count_status_counts": dict(sorted(head_count_status_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "comparison_basis": {
                "reference_ltot": "EFT-only FeynRules Ltot",
                "local_ltot": "EFT-only FeynPy Ltot",
                "local_sm_plus_eft_lagrangian": "Lfull",
                "omitted_sectors": list(bundle.omitted_sectors),
            },
        },
        "reference_vertices": reference_rows,
        "feynpy_only_signatures": feynpy_only_rows,
        "feynpy_only_zero_signatures": feynpy_only_zero_rows,
    }
    return report, local_vertices


def _vertex_payload(local_vertices: Iterable[LocalVertex]) -> list[dict[str, object]]:
    return [
        {
            "key": vertex.key,
            "signature": list(vertex.signature),
            "local_names": list(vertex.local_names),
            "feynpy_names": list(vertex.feynpy_names),
            "arity": vertex.arity,
            "term_count": vertex.term_count,
            "sectors": list(vertex.sectors),
            "heads": list(vertex.heads),
            "head_counts": dict(vertex.head_counts),
            "rule": vertex.rule,
        }
        for vertex in local_vertices
    ]


def write_outputs(
    report: dict[str, object],
    local_vertices: Iterable[LocalVertex],
    *,
    comparison_json: Path = COMPARISON_JSON,
    comparison_md: Path = COMPARISON_MD,
    feynpy_vertices: Path = FEYNPY_VERTICES,
) -> None:
    comparison_json.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    feynpy_vertices.write_text(
        json.dumps(_vertex_payload(local_vertices), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    comparison_md.write_text(_markdown_report(report), encoding="utf-8")


def write_weinberg_outputs(
    report: dict[str, object],
    vertices: list[dict[str, object]],
    *,
    comparison_json: Path = WEINBERG_COMPARISON_JSON,
    feynpy_vertices: Path = WEINBERG_VERTICES,
) -> None:
    comparison_json.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    feynpy_vertices.write_text(
        json.dumps(vertices, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    counts = summary["status_counts"]
    basis = summary["comparison_basis"]
    omitted_sectors = ", ".join(basis["omitted_sectors"]) or "none"
    lines = [
        "# SMEFT2 FeynRules/FeynPy Comparison",
        "",
        f"Generated on `{report['generated_on']}` by `models/SMEFT2/comparison.py`.",
        "",
        "## Scope",
        "",
        str(report["comparison_level"]),
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Reference vertices | {summary['reference_vertex_count']} |",
        f"| FeynPy 3-6 point signatures | {summary['feynpy_signature_count_3_to_6']} |",
        f"| Shared signatures (exact field multiset) | {summary['shared_signatures']} |",
        f"| Reference-only signatures (exact field multiset) | {summary['reference_only_signatures']} |",
        f"| FeynPy-only signatures (exact field multiset) | {summary['feynpy_only_signatures']} |",
        "| — of which charge-conjugation partners | "
        f"{summary['feynpy_only_charge_conjugation_partners']} |",
        "| — of which unexplained | "
        f"{summary['feynpy_only_unexplained_signatures']} |",
        "| FeynPy-only zero-signature artifacts (dropped) | "
        f"{summary['feynpy_only_zero_signatures']} |",
        f"| Shared coefficient-head matches | {summary['shared_head_matches']} |",
        "| Charge-conjugation packaging matches (modulo CC) | "
        f"{summary['charge_conjugation_packaging_matches']} |",
        "| Operator-content matches (incl. charge conjugation) | "
        f"{summary['operator_content_matches_including_cc']} |",
        f"| Shared raw head-count matches | {summary['shared_head_count_matches']} |",
        f"| Shared raw head-count mismatches | {summary['shared_head_count_mismatches']} |",
        f"| Shared raw head-count benign expansions | {summary['shared_head_count_benign_expansions']} |",
        "| Shared raw head-count mismatches with unexplained deltas | "
        f"{summary['shared_head_count_unexplained_mismatches']} |",
        f"| Exact symbolic supported vertices | {summary['exact_symbolic_supported_vertices']} |",
        f"| Direct exact symbolic matches | {summary['exact_symbolic_direct_match_vertices']} |",
        "| Exact modulo pinned CC packaging | "
        f"{summary['cc_packaging_pinned_match_vertices']} |",
        "| Unresolved CC packaging (existence only) | "
        f"{summary['cc_packaging_unresolved_vertices']} |",
        f"| Exact symbolic unequal vertices | {summary['exact_symbolic_unequal_vertices']} |",
        f"| Exact symbolic error vertices | {summary['exact_symbolic_error_vertices']} |",
        (
            "| Headline split | "
            f"direct exact: {summary['exact_symbolic_direct_match_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"modulo pinned CC: {summary['cc_packaging_pinned_match_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"unresolved CC: {summary['cc_packaging_unresolved_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"operator content: {summary['operator_content_matches_including_cc']}/"
            f"{summary['reference_vertex_count']} |"
        ),
        "| Compatibility alias `exact_symbolic_equal_vertices` (direct only) | "
        f"{summary['exact_symbolic_equal_vertices']} |",
        f"| Canonical tensor-map supported vertices | {summary['canonical_map_supported_vertices']} |",
        f"| Canonical tensor-map equal vertices | {summary['canonical_map_equal_vertices']} |",
        f"| Canonical tensor-map unequal vertices | {summary['canonical_map_unequal_vertices']} |",
        f"| Canonical tensor-map error vertices | {summary['canonical_map_error_vertices']} |",
        "| Canonical tensor-map equal coefficient sectors | "
        f"{summary['canonical_map_equal_coefficient_sectors']} |",
        "| Canonical tensor-map unequal coefficient sectors | "
        f"{summary['canonical_map_unequal_coefficient_sectors']} |",
        "| Canonical-map FeynPy raw monomials | "
        f"{summary['canonical_map_feynpy_raw_terms']} |",
        "| Canonical-map FeynPy canonical monomials | "
        f"{summary['canonical_map_feynpy_canonical_terms']} |",
        "| Canonical-map FeynPy redundant monomials (raw - canonical) | "
        f"{summary['canonical_map_feynpy_redundant_terms']} |",
        f"| Explained benign head-count deltas | {summary['benign_head_count_delta_heads']} |",
        f"| Unexplained head-count deltas | {summary['unexplained_head_count_delta_heads']} |",
        "",
        "## Basis",
        "",
        f"- Reference: `{basis['reference_ltot']}`.",
        f"- Local default model: `{basis['local_ltot']}`.",
        f"- Local SM plus EFT model: `{basis['local_sm_plus_eft_lagrangian']}`.",
        f"- Omitted sectors: `{omitted_sectors}`.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")

    exact_rows = [
        row
        for row in report["reference_vertices"]
        if row["exact_symbolic_status"] != "EXACT_UNSUPPORTED"
    ]
    lines.extend(
        [
            "",
            "## Exact Symbolic Comparison",
            "",
            "This layer is enabled for every FeynRules reference row. Bosonic "
            "rows use the native bosonic comparator. Fermion rows parse the "
            "full FeynRules tensor rule into native tensors, filter terms by "
            "indexed Wilson-coefficient head, keep flavor order and complex "
            "conjugation in the scalar coefficient, and compare canonical "
            "tensor-monomial maps. Statuses are graded honestly: "
            "`EXACT_MATCH` is direct same-signature canonical equality; "
            "`MATCH_MODULO_CC_PACKAGING` is equality after a pinned "
            "charge-conjugation packaging transform (Weinberg or Ec partner "
            "rows); `UNRESOLVED_CC_PACKAGING` means no pinned packaging rule "
            "is known or the pinned transform failed.",
            "",
            "| Signature | Status |",
            "| --- | --- |",
        ]
    )
    for row in exact_rows:
        lines.append(
            f"| `{row['key']}` | `{row['exact_symbolic_status']}` |"
        )

    canonical_rows = [
        row
        for row in report["reference_vertices"]
        if row["canonical_map_status"] != "CANONICAL_MAP_UNSUPPORTED"
    ]
    lines.extend(
        [
            "",
            "## Canonical Tensor-Map Gauge Comparison",
            "",
            "This comparison is currently enabled for pure nonabelian gauge "
            "vertices (`G^n` and `Wi^n`). It parses FeynRules `ME`, `FV`, "
            "`SP`, `Eps`, `fsu3`, and `fsu2` into native tensors, then "
            "compares canonical monomial maps per Wilson coefficient. It uses "
            "intrinsic tensor symmetries, dummy-index relabeling, commuting "
            "factor ordering, and exact coefficient collection; it does not "
            "use Jacobi, momentum conservation, EOM, IBP, or 4D reductions.",
            "",
            "| Signature | Status | Coefficient sectors |",
            "| --- | --- | --- |",
        ]
    )
    for row in canonical_rows:
        sector_summaries = []
        for coefficient, diagnostic in sorted(
            row["canonical_map_coefficients"].items()
        ):
            status = "match" if diagnostic["matches"] else "mismatch"
            sector_summaries.append(
                f"`{coefficient}` {status}: raw "
                f"{diagnostic['feynpy_raw_terms']}/"
                f"{diagnostic['feynrules_raw_terms']} -> canonical "
                f"{diagnostic['feynpy_canonical_terms']}/"
                f"{diagnostic['feynrules_canonical_terms']}"
            )
        lines.append(
            f"| `{row['key']}` | `{row['canonical_map_status']}` | "
            f"{'; '.join(sector_summaries)} |"
        )

    missing_heads = Counter()
    local_extra_heads = Counter()
    unexplained_head_count_deltas = Counter()
    benign_head_count_deltas = []
    for row in report["reference_vertices"]:
        # Heads resolved via the charge-conjugation overlay are genuine content
        # (present in FeynPy under the charge-conjugate signature), so they are
        # excluded from the "reference-side head gap" aggregate, which is meant
        # to surface only unexplained missing operator content.
        cc_matched = set(row.get("charge_conjugation_matched_heads", ()))
        missing_heads.update(
            head for head in row["feynrules_extra_heads"] if head not in cc_matched
        )
        local_extra_heads.update(row["feynpy_extra_heads"])
        if row["head_count_status"] == "NO_LOCAL_SIGNATURE":
            continue
        for head, reason in row["benign_head_count_delta_reasons"].items():
            counts_for_head = row["head_count_delta"][head]
            benign_head_count_deltas.append(
                (
                    row["key"],
                    head,
                    counts_for_head["reference"],
                    counts_for_head["feynpy"],
                    reason,
                )
            )
        for head, counts_for_head in row["unexplained_head_count_delta"].items():
            unexplained_head_count_deltas[head] += abs(
                counts_for_head["reference"] - counts_for_head["feynpy"]
            )

    lines.extend(
        [
            "",
            "## Largest Reference-Side Head Gaps",
            "",
            "| Head | Count |",
            "| --- | ---: |",
        ]
    )
    for head, count in missing_heads.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Largest Local Extra Heads",
            "",
            "| Head | Count |",
            "| --- | ---: |",
        ]
    )
    for head, count in local_extra_heads.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Explained Benign Raw Head-Count Deltas",
            "",
            "These are raw coefficient-head occurrence-count diagnostics. They catch "
            "some missing or duplicated content, but they are not tensor-rule equality "
            "proofs because equivalent algebra can be printed with different occurrence "
            "counts.",
            "",
            "| Signature | Head | Reference | FeynPy | Reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for key, head, reference_count, feynpy_count, reason in sorted(benign_head_count_deltas):
        lines.append(
            f"| `{key}` | `{head}` | {reference_count} | {feynpy_count} | "
            f"{BENIGN_HEAD_COUNT_REASON_TEXT[reason]} |"
        )

    lines.extend(
        [
            "",
            "## Largest Unexplained Raw Head-Count Deltas",
            "",
            "These exclude the explicit benign expansions listed above. The large "
            "pure-gauge raw deltas can remain large even where the canonical "
            "tensor-map comparison above proves equality.",
            "",
            "| Head | Total absolute delta |",
            "| --- | ---: |",
        ]
    )
    for head, count in unexplained_head_count_deltas.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `vertex_comparison_report.json` contains every reference row "
            "and FeynPy-only signature.",
            "- `feynpy_vertices.json` contains the regenerated local FeynPy "
            "rules and coefficient heads.",
            "- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle "
            "used for the comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE,
        help="FeynRules JSON reference to compare against.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write files; return nonzero unless operator-content coverage "
            "is complete and every supported row is a direct `EXACT_MATCH` "
            "(strict exact symbolic). Use `--allow-cc-packaging` to also accept "
            "pinned `MATCH_MODULO_CC_PACKAGING` rows. Unresolved CC packaging "
            "rows never pass `--check`."
        ),
    )
    parser.add_argument(
        "--allow-cc-packaging",
        action="store_true",
        help=(
            "With --check, accept pinned `MATCH_MODULO_CC_PACKAGING` rows "
            "(Weinberg and pinned Ec partner rows). Does not accept "
            "`UNRESOLVED_CC_PACKAGING` rows."
        ),
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help=(
            "With --check, also require matching raw coefficient-head occurrence "
            "counts for every shared signature."
        ),
    )
    args = parser.parse_args(argv)

    report, local_vertices = compare(args.reference)
    weinberg_report, weinberg_vertices = compare_reconstructed_weinberg(
        args.reference
    )
    if not args.check:
        write_outputs(report, local_vertices)
        write_weinberg_outputs(weinberg_report, weinberg_vertices)

    summary = report["summary"]
    weinberg_summary = weinberg_report["summary"]
    exact_supported = summary["exact_symbolic_supported_vertices"]
    direct_exact = summary["exact_symbolic_direct_match_vertices"]
    pinned_cc = summary["cc_packaging_pinned_match_vertices"]
    unresolved_cc = summary["cc_packaging_unresolved_vertices"]
    exact_unequal = summary["exact_symbolic_unequal_vertices"]
    exact_missing = summary["exact_symbolic_missing_local_vertices"]
    exact_error = summary["exact_symbolic_error_vertices"]
    exact_accounted = (
        direct_exact
        + pinned_cc
        + unresolved_cc
        + exact_unequal
        + exact_missing
        + exact_error
    )
    accepted_exact = direct_exact + (pinned_cc if args.allow_cc_packaging else 0)
    weinberg_check_failed = (
        weinberg_summary["reference_vertices"] != 2
        or (
            weinberg_summary["direct_matches"]
            != weinberg_summary["reference_vertices"]
        )
        or (
            weinberg_summary["coefficient_matches"]
            != weinberg_summary["coefficient_checks"]
        )
        or weinberg_summary["wrong_sign_matches"]
    )
    print(
        "SMEFT2 comparison: "
        f"{summary['operator_content_matches_including_cc']}/"
        f"{summary['reference_vertex_count']} "
        "reference vertices match at operator-content level "
        f"({summary['shared_head_matches']} direct + "
        f"{summary['charge_conjugation_packaging_matches']} via charge-conjugation "
        "packaging); "
        "exact symbolic split="
        f"direct {direct_exact}/{exact_supported}, "
        f"modulo pinned CC {pinned_cc}/{exact_supported}, "
        f"unresolved CC {unresolved_cc}/{exact_supported}; "
        f"raw-head-count matches={summary['shared_head_count_matches']}/"
        f"{summary['shared_signatures']}; "
        "canonical tensor-map matches="
        f"{summary['canonical_map_equal_vertices']}/"
        f"{summary['canonical_map_supported_vertices']} supported vertices "
        f"({summary['canonical_map_equal_coefficient_sectors']}/"
        f"{summary['canonical_map_supported_coefficient_sectors']} sectors); "
        "Weinberg reconstructed sidecar="
        f"{weinberg_summary['direct_matches']}/"
        f"{weinberg_summary['reference_vertices']} direct, "
        f"{weinberg_summary['coefficient_matches']}/"
        f"{weinberg_summary['coefficient_checks']} coefficient checks, "
        f"wrong-sign matches={weinberg_summary['wrong_sign_matches']}; "
        f"reference-only={summary['reference_only_signatures']}; "
        f"feynpy-only={summary['feynpy_only_signatures']}."
    )
    if args.check and (
        summary["operator_content_matches_including_cc"]
        != summary["reference_vertex_count"]
        or summary["feynpy_only_unexplained_signatures"]
        or summary["exact_symbolic_supported_vertices"]
        != summary["reference_vertex_count"]
        or exact_accounted != exact_supported
        or accepted_exact != exact_supported
        or exact_unequal
        or exact_missing
        or exact_error
        or unresolved_cc
        or (
            pinned_cc
            and not args.allow_cc_packaging
        )
        or summary["canonical_map_unequal_vertices"]
        or summary["canonical_map_error_vertices"]
        or (args.strict_counts and summary["shared_head_count_mismatches"])
        or weinberg_check_failed
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
