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
    FeynRulesVertex,
    compare_feynrules_bosonic_vertices,
    compare_canonical_coefficient_maps,
    load_feynrules_json,
)
from models.SMEFT2 import build_smeft_green_bpreserving
from symbolic.tensor_canonicalization import canonical_external_index_set
from symbolica import S


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
            "Exact symbolic comparison is not yet enabled for SMEFT2 two-fermion "
            "rows; they still rely on signature and coefficient-head diagnostics."
        ),
        "FOUR_FERMION": (
            "Exact symbolic comparison is not yet enabled for SMEFT2 four-fermion "
            "rows; they still rely on signature and coefficient-head diagnostics."
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
        head_counts = _parameter_head_counts_from_text(rule_text, parameter_names)
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
                heads=tuple(head_counts),
                head_counts=tuple(head_counts.items()),
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
            local_heads = set(local.heads)
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

    feynpy_only_rows = []
    for local in local_vertices:
        if local.key in reference_keys:
            continue
        status = _feynpy_only_status(set(local.heads))
        status_counts[status] += 1
        feynpy_only_rows.append(
            {
                "key": local.key,
                "signature": list(local.signature),
                "local_names": list(local.local_names),
                "feynpy_names": list(local.feynpy_names),
                "arity": local.arity,
                "term_count": local.term_count,
                "sectors": list(local.sectors),
                "feynpy_heads": list(local.heads),
                "feynpy_head_counts": dict(local.head_counts),
                "status": status,
                "reason": _reason_for_status(status),
            }
        )

    shared = sum(1 for row in reference_rows if row["feynpy_heads"])
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
            "comparison for supported bosonic rows and canonical tensor-"
            "monomial equality for supported pure nonabelian gauge vertices. "
            "Full tensor-rule equality is not claimed globally."
        ),
        "summary": {
            "reference_vertex_count": len(references),
            "feynpy_signature_count_3_to_6": len(local_vertices),
            "shared_signatures": shared,
            "reference_only_signatures": len(references) - shared,
            "feynpy_only_signatures": len(feynpy_only_rows),
            "shared_head_matches": matched,
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
        f"| Shared signatures | {summary['shared_signatures']} |",
        f"| Reference-only signatures | {summary['reference_only_signatures']} |",
        f"| FeynPy-only signatures | {summary['feynpy_only_signatures']} |",
        f"| Shared coefficient-head matches | {summary['shared_head_matches']} |",
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
            "This layer is currently enabled for bosonic SMEFT2 rows. It "
            "parses the full FeynRules tensor rule into native tensors, "
            "canonicalizes index structure, and checks whether the exact "
            "symbolic difference is zero. Two-fermion and four-fermion rows "
            "still fall back to the signature/head diagnostics above.",
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
        missing_heads.update(row["feynrules_extra_heads"])
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
        f"{summary['shared_head_matches']}/{summary['reference_vertex_count']} "
        "reference vertices match at coefficient-head set level; "
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
        summary["shared_head_matches"] != summary["reference_vertex_count"]
        or summary["feynpy_only_signatures"]
        or (args.strict_counts and summary["shared_head_count_mismatches"])
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
