"""Regenerate the SMEFT FeynRules/FeynPy comparison artifacts.

The FeynRules reference JSON is a full tensor-rule export. This script performs
the reproducible comparison currently supported for SMEFT: signature coverage,
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

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feynrules.comparison import (
    CanonicalCoefficientComparison,
    ChiralityMismatch,
    FeynRulesVertex,
    compare_feynrules_bosonic_vertices,
    compare_canonical_coefficient_maps,
    feynrules_ascii_label as _feynrules_ascii_label,
    find_matching_square as _find_matching_square,
    load_feynrules_json,
    require_trailing_projector,
    split_top_level_commas as _split_top_level_commas,
    validate_feynrules_projector_chirality,
)
from models.SMEFT import build_smeft_green_bpreserving
from symbolic.tensor_canonicalization import (
    CanonicalMonomialReport,
    CanonicalTensorMonomial,
    SPENSO_TENSOR_HEAD_SPECS,
    TensorHeadSpec,
    canonical_external_index_set,
    canonical_tensor_monomial_report,
    canonical_tensor_monomial_map,
)
from symbolica import AtomType, Expression, S
from symbolica.community.spenso import TensorName

from symbolic.spenso_structures import (
    COLOR_ADJ,
    COLOR_FUND,
    WEAK_ADJ,
    WEAK_FUND,
    bispinor_index,
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


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR.parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
REFERENCE = MODEL_DIR / "reference" / "Ltot_SMEFT_FeynRules.json"
FEYNPY_VERTICES = ARTIFACT_DIR / "feynpy_vertices.json"
COMPARISON_JSON = ARTIFACT_DIR / "vertex_comparison_report.json"
COMPARISON_MD = MODEL_DIR / "COMPARISON.md"
WEINBERG_VERTICES = ARTIFACT_DIR / "weinberg_vertices.json"
WEINBERG_COMPARISON_JSON = ARTIFACT_DIR / "weinberg_comparison_report.json"
EC_CC_VERTICES = ARTIFACT_DIR / "ec_charge_conjugation_vertices.json"
EC_CC_COMPARISON_JSON = ARTIFACT_DIR / "ec_charge_conjugation_comparison_report.json"


def _name_key(names: Iterable[str]) -> str:
    return "|".join(sorted(names))


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


# Chirality of the unbroken-basis matter fields. A trailing ``bar`` marks the
# barred field, so only the unbarred names are listed.
SMEFT_FERMION_CHIRALITY = {
    "lL": "L",
    "qL": "L",
    "eR": "R",
    "uR": "R",
    "dR": "R",
}


def validate_smeft_projector_chirality(rule: str, fields: Iterable[str]) -> int:
    """Verify every FeynRules projector against SMEFT field chirality.

    The SMEFT parser drops ``ProjM``/``ProjP`` because chirality lives in the
    FeynPy field class, which makes the projector redundant in the unbroken
    basis. That is only sound when the projector actually agrees with the
    fields it sits between, so the agreement is verified rather than assumed.
    The rule itself is model independent and lives in the shared FeynRules
    comparison layer.
    """

    return validate_feynrules_projector_chirality(
        rule,
        fields,
        SMEFT_FERMION_CHIRALITY,
    )


def _spinor_chain_replacement(
    items: tuple[str, ...],
    left: str,
    right: str,
    *,
    chain_id: int,
) -> str:
    # Chirality is not lost silently here: the caller validates every
    # projector against the field chirality via
    # ``validate_feynrules_projector_chirality`` before this drop happens.
    # Dropping the projector is only faithful if it trails the gamma product.
    require_trailing_projector(items)
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


_EC_CC_PL = TensorName("PL")
_EC_CC_PR = TensorName("PR")


def _ec_typed_projector(projector_head: str, left_spinor, right_spinor) -> Expression:
    if projector_head == "PL":
        tensor = _EC_CC_PL
    elif projector_head == "PR":
        tensor = _EC_CC_PR
    else:
        raise ValueError(f"Unsupported EC projector head {projector_head!r}.")
    return tensor(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


def _spinor_chain_replacement_with_projector_label(
    items: tuple[str, ...],
    left: str,
    right: str,
    *,
    chain_id: int,
) -> str:
    """Parse a FeynRules spinor chain and retain chirality as PL/PR."""

    projector_items = tuple(item for item in items if item in {"ProjM", "ProjP"})
    if len(projector_items) != 1:
        raise ValueError(f"Unsupported FeynRules projector chain: {items!r}")

    projector_head = "PL" if projector_items[0] == "ProjM" else "PR"
    result = Expression.num(1)
    current = S(left)
    for item_id, item in enumerate(items, start=1):
        target = (
            S(right)
            if item_id == len(items)
            else S(f"i_feynrules_chain_{chain_id}_{item_id}")
        )
        if item == "ProjM" or item == "ProjP":
            result *= _ec_typed_projector(projector_head, current, target)
        else:
            result *= _spinor_chain_item_factor(
                item,
                current,
                target,
                chain_id=chain_id,
                item_id=item_id,
            )
        current = target
    return result.cancel().expand().to_canonical_string()


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
    such as ``alphaKl(f1, f2)`` or ``conj(alphaWeinberg(f1, f2))``.  SMEFT
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

__all__ = [name for name in globals() if not name.startswith("__")]
