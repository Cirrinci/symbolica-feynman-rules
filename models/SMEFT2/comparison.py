"""Regenerate the SMEFT2 FeynRules/FeynPy comparison artifacts.

The FeynRules reference JSON is a full tensor-rule export. This script performs
the reproducible comparison currently supported for SMEFT2: signature coverage
and coefficient-head content after normalizing field names to the FeynRules
convention. It also exports the local FeynPy vertex rules so individual rows
can be inspected against the reference JSON.
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

from feynrules.comparison import FeynRulesVertex, load_feynrules_json
from models.SMEFT2 import build_smeft_green_bpreserving


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

OMITTED_COEFFICIENT_HEADS = frozenset(
    {
        "alphaR2G",
        "alphaR2W",
        "alphaR2B",
        "alphaRWDH",
        "alphaRBDH",
        "alphaRDH",
        "alphaRqD",
        "alphaRuD",
        "alphaRdD",
        "alphaRlD",
        "alphaReD",
        "alphaRuHD1",
        "alphaRuHD2",
        "alphaRuHD3",
        "alphaRuHD4",
        "alphaRdHD1",
        "alphaRdHD2",
        "alphaRdHD3",
        "alphaRdHD4",
        "alphaReHD1",
        "alphaReHD2",
        "alphaReHD3",
        "alphaReHD4",
        "alphaRGq",
        "alphaRGqp",
        "alphaRGqtp",
        "alphaRWq",
        "alphaRWqp",
        "alphaRWqtp",
        "alphaRBq",
        "alphaRBqp",
        "alphaRBqtp",
        "alphaRGu",
        "alphaRGup",
        "alphaRGutp",
        "alphaRBu",
        "alphaRBup",
        "alphaRButp",
        "alphaRGd",
        "alphaRGdp",
        "alphaRGdtp",
        "alphaRBd",
        "alphaRBdp",
        "alphaRBdtp",
        "alphaRWl",
        "alphaRWlp",
        "alphaRWltp",
        "alphaRBl",
        "alphaRBlp",
        "alphaRBltp",
        "alphaRBe",
        "alphaRBep",
        "alphaRBetp",
        "alphaEGq",
        "alphaEGqp",
        "alphaEGqtp",
        "alphaEWq",
        "alphaEWqp",
        "alphaEWqtp",
        "alphaEBq",
        "alphaEBqp",
        "alphaEBqtp",
        "alphaEGu",
        "alphaEGup",
        "alphaEGutp",
        "alphaEBu",
        "alphaEBup",
        "alphaEButp",
        "alphaEGd",
        "alphaEGdp",
        "alphaEGdtp",
        "alphaEBd",
        "alphaEBdp",
        "alphaEBdtp",
        "alphaEWl",
        "alphaEWlp",
        "alphaEWltp",
        "alphaEBl",
        "alphaEBlp",
        "alphaEBltp",
        "alphaEBe",
        "alphaEBep",
        "alphaEBetp",
    }
)


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
    rule: str


def _name_key(names: Iterable[str]) -> str:
    return "|".join(sorted(names))


def _normalize_local_name(name: str) -> str:
    try:
        return FIELD_NAME_MAP[name]
    except KeyError as exc:
        raise ValueError(f"No FeynRules name mapping for local field {name!r}") from exc


def _parameter_heads_from_text(text: str, parameter_names: Iterable[str]) -> tuple[str, ...]:
    heads = set(re.findall(r"\balpha[A-Za-z0-9]+(?=\[|\(|\b)", text))
    for name in parameter_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            heads.add(name)
    return tuple(sorted(heads))


def _reference_heads(
    reference: FeynRulesVertex,
    parameter_names: Iterable[str],
) -> tuple[str, ...]:
    return _parameter_heads_from_text(reference.rule, parameter_names)


def _local_vertices(parameter_names: Iterable[str]) -> tuple[LocalVertex, ...]:
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    rows: list[LocalVertex] = []
    for signature in lagrangian.vertex_signatures():
        if not 3 <= signature.arity <= 6:
            continue
        rule = lagrangian.feynman_rule(*signature.fields, simplify=True)
        rule_text = rule.cancel().expand().to_canonical_string()
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
                heads=_parameter_heads_from_text(rule_text, parameter_names),
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
    parameter_names = set(bundle.parameters) | GENERIC_PARAMETER_NAMES

    local_vertices = _local_vertices(parameter_names)
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
        local = local_by_key.get(key)
        if local is None:
            status = _missing_reference_status(reference_heads)
            local_heads: set[str] = set()
            local_names: tuple[str, ...] = ()
            feynpy_names: tuple[str, ...] = ()
            sectors: tuple[str, ...] = ()
            term_count = 0
        else:
            local_heads = set(local.heads)
            status = _shared_status(reference_heads, local_heads)
            local_names = local.local_names
            feynpy_names = local.feynpy_names
            sectors = local.sectors
            term_count = local.term_count

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
                "status": status,
                "reason": _reason_for_status(status),
            }
        )

    shared = sum(1 for row in reference_rows if row["feynpy_heads"])
    matched = status_counts["SHARED_HEADS_MATCH"]
    report = {
        "generated_on": date.today().isoformat(),
        "reference": str(reference_path.relative_to(ROOT)),
        "local_model": str((MODEL_DIR / "SMEFT2.py").relative_to(ROOT)),
        "comparison_level": (
            "Signature coverage and coefficient-head content. Full tensor-rule "
            "equality is not claimed by this SMEFT2 report."
        ),
        "summary": {
            "reference_vertex_count": len(references),
            "feynpy_signature_count_3_to_6": len(local_vertices),
            "shared_signatures": shared,
            "reference_only_signatures": len(references) - shared,
            "feynpy_only_signatures": len(feynpy_only_rows),
            "shared_head_matches": matched,
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
        "",
        "## Basis",
        "",
        f"- Reference: `{basis['reference_ltot']}`.",
        f"- Local default model: `{basis['local_ltot']}`.",
        f"- Local SM plus EFT model: `{basis['local_sm_plus_eft_lagrangian']}`.",
        f"- Omitted sectors: `{', '.join(basis['omitted_sectors'])}`.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")

    missing_heads = Counter()
    local_extra_heads = Counter()
    for row in report["reference_vertices"]:
        missing_heads.update(row["feynrules_extra_heads"])
        local_extra_heads.update(row["feynpy_extra_heads"])

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
    args = parser.parse_args(argv)

    report, local_vertices = compare(args.reference)
    if not args.check:
        write_outputs(report, local_vertices)

    summary = report["summary"]
    print(
        "SMEFT2 comparison: "
        f"{summary['shared_head_matches']}/{summary['reference_vertex_count']} "
        "reference vertices match at coefficient-head level; "
        f"reference-only={summary['reference_only_signatures']}; "
        f"feynpy-only={summary['feynpy_only_signatures']}."
    )
    if args.check and (
        summary["shared_head_matches"] != summary["reference_vertex_count"]
        or summary["feynpy_only_signatures"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
