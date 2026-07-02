"""Generate and compare all UnbrokenSM_BFM interaction vertices."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symbolica import AtomType, Expression, S

from feynrules.comparison import (
    FeynRulesVertex,
    load_feynrules_json,
    reduce_fermion_currents,
    reduce_yukawa_bilinears,
)
from models.UnbrokenSM_BFM import build_unbroken_sm_bfm
from symbolic.spenso_structures import (
    COLOR_FUND,
    WEAK_FUND,
    chiral_projector_left,
    chiral_projector_right,
    gauge_generator,
    gamma_matrix,
    lorentz_metric,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.tensor_canonicalization import canonize_full
from symbolic.vertex_engine import pcomp


REFERENCE_DIR = Path(__file__).with_name("reference") / "feynrules"
REFERENCE = REFERENCE_DIR / "LSM_full_FeynRules.json"
FEYNPY_OUTPUT = Path(__file__).with_name("feynpy_vertices.json")
COMPARISON_OUTPUT = Path(__file__).with_name("comparison_report.json")


@dataclass(frozen=True)
class ComparisonResult:
    total: int
    matched: int
    mismatches: tuple[dict[str, str], ...]
    feynrules_only: tuple[str, ...]
    feynpy_only: tuple[str, ...]

    @property
    def all_match(self) -> bool:
        return (
            self.total == self.matched
            and not self.mismatches
            and not self.feynrules_only
            and not self.feynpy_only
        )


def field_map(fields) -> dict[str, object]:
    result = {
        name: getattr(fields, name)
        for name in (
            "B",
            "BQuantum",
            "Wi",
            "WiQuantum",
            "G",
            "GQuantum",
            "ghWi",
            "ghG",
            "lL",
            "eR",
            "qL",
            "uR",
            "dR",
            "Phi",
        )
    }
    result.update(
        {
            "ghWibar": fields.ghWi.bar,
            "ghGbar": fields.ghG.bar,
            "lLbar": fields.lL.bar,
            "eRbar": fields.eR.bar,
            "qLbar": fields.qL.bar,
            "uRbar": fields.uR.bar,
            "dRbar": fields.dR.bar,
            "Phibar": fields.Phi.bar,
        }
    )
    return result


def _replace_external_indices(text: str) -> str:
    prefixes = {
        "Lorentz": "mu",
        "Spin": "i",
        "Gluon": "a",
        "Colour": "c",
        "SU2W": "aw",
        "SU2D": "w",
        "Generation": "f",
    }
    for kind, prefix in prefixes.items():
        text = re.sub(
            rf"Index\[{kind},\s*Ext\[(\d+)\]\]",
            lambda match, p=prefix: f"{p}{match.group(1)}",
            text,
        )
        text = re.sub(
            rf"Index\[{kind},\s*{kind}\$(\d+)\]",
            lambda match, p=prefix: f"{p}_feynrules_dummy_{match.group(1)}",
            text,
        )
    return text


def parse_feynrules_rule(rule: str) -> Expression:
    """Parse every tensor used by the UnbrokenSM_BFM JSON export."""

    text = _replace_external_indices(rule)
    text = re.sub(
        r"ME\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: lorentz_metric(S(m.group(1)), S(m.group(2))).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"FV\[(\d+),\s*([^\]]+)\]",
        lambda m: pcomp(S(f"q{m.group(1)}"), S(m.group(2))).to_canonical_string(),
        text,
    )

    delta_builders = {
        "c": COLOR_FUND.g,
        "f": COLOR_FUND.g,
        "w": WEAK_FUND.g,
    }

    def replace_delta(match: re.Match[str]) -> str:
        left, right = match.group(1).strip(), match.group(2).strip()
        builder = delta_builders[left[0]]
        return builder(S(left), S(right)).to_expression().to_canonical_string()

    text = re.sub(
        r"IndexDelta\[([^,\]]+),\s*([^\]]+)\]",
        replace_delta,
        text,
    )
    text = re.sub(
        r"(?:f|fsu3)\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]",
        lambda m: structure_constant(
            S(m.group(1)), S(m.group(2)), S(m.group(3))
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"fsu2\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]",
        lambda m: weak_structure_constant(
            S(m.group(1)), S(m.group(2)), S(m.group(3))
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"T\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]",
        lambda m: gauge_generator(
            S(m.group(1)), S(m.group(2)), S(m.group(3))
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ta\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]",
        lambda m: weak_gauge_generator(
            S(m.group(1)), S(m.group(2)), S(m.group(3))
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: weak_eps2(S(m.group(1)), S(m.group(2))).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ga\[([^\]]+)\]",
        lambda m: f"GA({m.group(1)})",
        text,
    )
    text = re.sub(
        r"TensDot\[GA\(([^)]+)\),\s*(ProjM|ProjP)\]"
        r"\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: (
            gamma_matrix(S(m.group(3)), S("i_feynrules_chain"), S(m.group(1)))
            * (
                chiral_projector_left(S("i_feynrules_chain"), S(m.group(4)))
                if m.group(2) == "ProjM"
                else chiral_projector_right(S("i_feynrules_chain"), S(m.group(4)))
            )
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjM\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: chiral_projector_left(S(m.group(1)), S(m.group(2))).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjP\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: chiral_projector_right(S(m.group(1)), S(m.group(2))).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"\b(yl|yu|yd)\[([^,\]]+),\s*([^\]]+)\]",
        lambda m: f"{m.group(1)}({m.group(2)},{m.group(3)})",
        text,
    )
    text = re.sub(r"Conjugate\[([^\]]+)\]", r"conj(\1)", text)
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(f"Unsupported FeynRules syntax remains: {text}")
    return Expression.parse(text).cancel().expand()


def _walk(expression: Expression):
    yield expression
    if expression.get_type() in (AtomType.Add, AtomType.Mul, AtomType.Pow, AtomType.Fn):
        for child in expression:
            yield from _walk(child)


def _representation_labels(
    expression: Expression, *, head: str, dimension: int
) -> tuple[Expression, ...]:
    labels: dict[str, Expression] = {}
    for node in _walk(expression):
        if node.get_type() != AtomType.Fn or node.get_name() != head:
            continue
        arguments = tuple(node)
        if len(arguments) != 2 or arguments[0].to_canonical_string() != str(dimension):
            continue
        label = arguments[1]
        labels.setdefault(label.to_canonical_string(), label)
    return tuple(labels.values())


def canonicalize_rule(expression: Expression, fields: tuple[object, ...]) -> Expression:
    fermions = sum(
        (item.field if hasattr(item, "field") else item).kind == "fermion"
        for item in fields
    )
    scalars = sum(
        (item.field if hasattr(item, "field") else item).kind == "scalar"
        for item in fields
    )
    expression = expression.cancel().expand()
    if fermions == 2:
        expression = (
            reduce_yukawa_bilinears(expression)
            if scalars
            else reduce_fermion_currents(expression)
        )
    return canonize_full(
        expression,
        lorentz_indices=_representation_labels(
            expression, head="spenso::mink", dimension=4
        ),
        adjoint_indices=_representation_labels(
            expression, head="spenso::coad", dimension=8
        ),
        color_fund_indices=_representation_labels(
            expression, head="spenso::cof", dimension=3
        ),
        spinor_indices=_representation_labels(
            expression, head="spenso::bis", dimension=4
        ),
        weak_fund_indices=_representation_labels(
            expression, head="spenso::cof", dimension=2
        ),
        weak_adj_indices=_representation_labels(
            expression, head="spenso::coad", dimension=3
        ),
        run_color=False,
        field_heads=tuple(item.field if hasattr(item, "field") else item for item in fields),
    ).cancel().expand()


def _signature_key(names: tuple[str, ...]) -> str:
    aliases = {
        "ghWi.bar": "ghWibar",
        "ghG.bar": "ghGbar",
        "lL.bar": "lLbar",
        "eR.bar": "eRbar",
        "qL.bar": "qLbar",
        "uR.bar": "uRbar",
        "dR.bar": "dRbar",
        "Phi.bar": "Phibar",
    }
    return "|".join(sorted(aliases.get(name, name) for name in names))


def compare(reference_path: Path = REFERENCE) -> tuple[ComparisonResult, list[dict[str, object]]]:
    theory = build_unbroken_sm_bfm()
    mapping = field_map(theory.fields)
    references = load_feynrules_json(reference_path)
    feynrules_keys = {
        "|".join(sorted(reference.fields)) for reference in references
    }
    feynpy_keys = {
        _signature_key(signature.names)
        for signature in theory.lagrangian.vertex_signatures()
        if signature.arity in (3, 4)
    }

    rows: list[dict[str, object]] = []
    mismatches: list[dict[str, str]] = []
    matched = 0
    for reference in references:
        ordered_fields = tuple(mapping[name] for name in reference.fields)
        raw_feynpy = theory.lagrangian.feynman_rule(
            *ordered_fields,
            simplify=True,
            include_delta=False,
        )
        parsed_reference = parse_feynrules_rule(reference.rule)
        feynpy_rule = canonicalize_rule(raw_feynpy, ordered_fields)
        feynrules_rule = canonicalize_rule(parsed_reference, ordered_fields)
        difference = (feynpy_rule - feynrules_rule).cancel().expand()
        status = "MATCH" if difference.to_canonical_string() == "0" else "MISMATCH"
        matched += status == "MATCH"
        rows.append(
            {
                "id": reference.identifier,
                "key": reference.key,
                "fields": list(reference.fields),
                "rule": raw_feynpy.cancel().expand().to_canonical_string(),
                "canonical_rule": feynpy_rule.to_canonical_string(),
            }
        )
        if status != "MATCH":
            mismatches.append(
                {
                    "key": reference.key,
                    "feynpy": feynpy_rule.to_canonical_string(),
                    "feynrules": feynrules_rule.to_canonical_string(),
                    "difference": difference.to_canonical_string(),
                }
            )

    result = ComparisonResult(
        total=len(references),
        matched=matched,
        mismatches=tuple(mismatches),
        feynrules_only=tuple(sorted(feynrules_keys - feynpy_keys)),
        feynpy_only=tuple(sorted(feynpy_keys - feynrules_keys)),
    )
    return result, rows


def write_outputs(reference_path: Path = REFERENCE) -> ComparisonResult:
    result, vertices = compare(reference_path)
    FEYNPY_OUTPUT.write_text(json.dumps(vertices, indent=2) + "\n", encoding="utf-8")
    report = {
        "reference": str(reference_path.relative_to(ROOT)),
        "total": result.total,
        "matched": result.matched,
        "all_match": result.all_match,
        "feynrules_only": list(result.feynrules_only),
        "feynpy_only": list(result.feynpy_only),
        "mismatches": list(result.mismatches),
    }
    COMPARISON_OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = write_outputs()
    print(
        f"UnbrokenSM_BFM: {result.matched}/{result.total} matched; "
        f"FeynRules-only={len(result.feynrules_only)}; "
        f"FeynPy-only={len(result.feynpy_only)}"
    )
    if not result.all_match:
        raise SystemExit(1)
