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
from collections import Counter
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
from symbolic.vertex_engine import pcomp


MODEL_DIR = Path(__file__).resolve().parent
REFERENCE = MODEL_DIR / "reference" / "Ltot_SMEFT_FeynRules.json"
FEYNPY_VERTICES = MODEL_DIR / "feynpy_vertices.json"
COMPARISON_JSON = MODEL_DIR / "vertex_comparison_report.json"
COMPARISON_MD = MODEL_DIR / "COMPARISON.md"

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


def _canonical_report_for_coefficient_head(
    expression: Expression,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
):
    return canonical_tensor_monomial_report(
        _filter_terms_by_coefficient_head(expression, coefficient),
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
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
        feynrules_report = _canonical_report_for_coefficient_head(
            feynrules_expression,
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=max_dummy_permutations,
        )
        feynpy_keys = set(feynpy_report.map)
        feynrules_keys = set(feynrules_report.map)
        shared_keys = feynpy_keys & feynrules_keys
        coefficient_mismatches = {
            key: (feynpy_report.map[key], feynrules_report.map[key])
            for key in shared_keys
            if feynpy_report.map[key].cancel().expand().to_canonical_string()
            != feynrules_report.map[key].cancel().expand().to_canonical_string()
        }
        comparisons[coefficient] = CanonicalCoefficientComparison(
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
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if all(comparison.matches for comparison in comparisons.values()):
        return {
            "family": family,
            "status": "EXACT_MATCH",
            "detail": (
                "Canonical tensor-monomial maps agree for all "
                f"{len(comparisons)} coefficient sector(s); raw head-count "
                f"status was {head_count_status}."
            ),
        }

    mismatched = tuple(
        coefficient
        for coefficient, comparison in comparisons.items()
        if not comparison.matches
    )
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
            "status": "EXACT_MATCH",
            "detail": (
                "Weinberg charge-conjugation packaging matches exactly: the "
                "same-chirality FeynRules row equals the antisymmetrized "
                "FeynPy mixed `lLbar,lL` assignment pair, with `ProjM/ProjP` "
                "mapped to the antisymmetric Dirac charge-conjugation tensor."
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
                reference_heads=reference_heads,
                local_heads=local_heads,
                head_count_status=head_count_status,
                lagrangian=lagrangian,
                field_map=field_map,
            )
        if exact_symbolic is None:
            exact_symbolic = {
                "family": exact_symbolic_family,
                "status": "EXACT_UNSUPPORTED",
                "detail": _unsupported_exact_symbolic_detail(exact_symbolic_family),
            }

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
            "comparison for all bosonic rows, shared two-/four-fermion rows, "
            "and the two Weinberg charge-conjugation packaging rows. Fermion "
            "exact comparison filters by indexed Wilson-coefficient head and "
            "keeps flavor order/conjugation in the canonical scalar "
            "coefficient, so it cannot pass vacuously for function-valued "
            "coefficients. The separate canonical tensor-map diagnostic is "
            "still the gauge-sector per-coefficient map for supported bosonic "
            "coefficient sectors."
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
        f"| Exact symbolic equal vertices | {summary['exact_symbolic_equal_vertices']} |",
        f"| Exact symbolic unequal vertices | {summary['exact_symbolic_unequal_vertices']} |",
        f"| Exact symbolic error vertices | {summary['exact_symbolic_error_vertices']} |",
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
            "tensor-monomial maps. The two Weinberg rows have no literal "
            "FeynPy signature, so they are compared to the antisymmetrized "
            "charge-conjugation packaged FeynPy partner rule.",
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
            "Do not write files; return nonzero if the comparison is not a "
            "full head-level match."
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
    if not args.check:
        write_outputs(report, local_vertices)

    summary = report["summary"]
    print(
        "SMEFT2 comparison: "
        f"{summary['operator_content_matches_including_cc']}/"
        f"{summary['reference_vertex_count']} "
        "reference vertices match at operator-content level "
        f"({summary['shared_head_matches']} direct + "
        f"{summary['charge_conjugation_packaging_matches']} via charge-conjugation "
        "packaging); "
        "exact symbolic matches="
        f"{summary['exact_symbolic_equal_vertices']}/"
        f"{summary['exact_symbolic_supported_vertices']} supported vertices; "
        f"raw-head-count matches={summary['shared_head_count_matches']}/"
        f"{summary['shared_signatures']}; "
        "canonical tensor-map matches="
        f"{summary['canonical_map_equal_vertices']}/"
        f"{summary['canonical_map_supported_vertices']} supported vertices "
        f"({summary['canonical_map_equal_coefficient_sectors']}/"
        f"{summary['canonical_map_supported_coefficient_sectors']} sectors); "
        f"reference-only={summary['reference_only_signatures']}; "
        f"feynpy-only={summary['feynpy_only_signatures']}."
    )
    if args.check and (
        summary["operator_content_matches_including_cc"]
        != summary["reference_vertex_count"]
        or summary["feynpy_only_unexplained_signatures"]
        or summary["exact_symbolic_supported_vertices"]
        != summary["reference_vertex_count"]
        or summary["exact_symbolic_unequal_vertices"]
        or summary["exact_symbolic_error_vertices"]
        or summary["canonical_map_unequal_vertices"]
        or summary["canonical_map_error_vertices"]
        or (args.strict_counts and summary["shared_head_count_mismatches"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
