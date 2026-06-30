"""Build, display and export the UnbrokenSM_BFM model.

Run from the repository root with:

    .venv/bin/python models/UnbrokenSM_BFM/export_model.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feynrules.comparison import load_feynrules_json
from symbolica import AtomType, Expression
from models.UnbrokenSM_BFM import build_unbroken_sm_bfm
from models.UnbrokenSM_BFM.comparison import (
    REFERENCE,
    canonicalize_rule,
    field_map,
    parse_feynrules_rule,
    write_outputs,
)


ANSI = re.compile(r"\x1b\[[0-9;]*m")
EXAMPLE_VERTICES = (
    ("G", "GQuantum", "G", "GQuantum"),
    ("GQuantum", "GQuantum", "G", "G"),
    ("GQuantum", "GQuantum", "GQuantum", "GQuantum"),
    ("Wi", "WiQuantum", "Wi", "WiQuantum"),
    ("WiQuantum", "WiQuantum", "Wi", "Wi"),
    ("WiQuantum", "WiQuantum", "WiQuantum", "WiQuantum"),
)


def _plain(value) -> str:
    return (
        ANSI.sub("", str(value))
        .replace("1𝑖", "I")
        .replace("𝑖", "I")
        .replace(" + -", " - ")
    )


def _short_name(name: str) -> str:
    short = name.rsplit("::", 1)[-1]
    dummy = re.match(r"canon_dummy_\d+_(\d+)$", short)
    return f"x{dummy.group(1)}" if dummy else short


def _slot(expression: Expression) -> str:
    arguments = tuple(expression)
    return pretty_expression(arguments[1]) if len(arguments) == 2 else pretty_expression(expression)


def pretty_expression(expression: Expression, *, parent_precedence: int = 0) -> str:
    """Render the tensor structures used in this example without namespaces."""

    kind = expression.get_type()
    if kind == AtomType.Num:
        return (
            expression.to_canonical_string()
            .replace("1𝑖", "I")
            .replace("𝑖", "I")
        )
    if kind == AtomType.Var:
        return _short_name(expression.get_name())
    if kind == AtomType.Add:
        rendered = " + ".join(
            pretty_expression(term, parent_precedence=1) for term in expression
        ).replace(" + -", " - ")
        return f"({rendered})" if parent_precedence > 1 else rendered
    if kind == AtomType.Mul:
        rendered = " * ".join(
            pretty_expression(factor, parent_precedence=2) for factor in expression
        )
        return f"({rendered})" if parent_precedence > 2 else rendered
    if kind == AtomType.Pow:
        base, exponent = tuple(expression)
        return (
            f"{pretty_expression(base, parent_precedence=3)}^"
            f"{pretty_expression(exponent, parent_precedence=3)}"
        )
    if kind != AtomType.Fn:
        return _plain(expression.to_canonical_string())

    name = expression.get_name()
    arguments = tuple(expression)
    if name in ("spenso::coad", "spenso::cof", "spenso::bis", "spenso::mink"):
        return _slot(expression)
    if name == "spenso::t":
        dimension = tuple(arguments[0])[0].to_canonical_string()
        head = "T_W" if dimension == "3" else "T_C"
        return f"{head}({', '.join(_slot(argument) for argument in arguments)})"
    if name == "spenso::f":
        dimension = tuple(arguments[0])[0].to_canonical_string()
        head = "f_W" if dimension == "3" else "f_C"
        return f"{head}({', '.join(_slot(argument) for argument in arguments)})"
    if name == "spenso::g":
        representation = arguments[0]
        rep_name = representation.get_name()
        dimension = tuple(representation)[0].to_canonical_string()
        labels = tuple(_slot(argument) for argument in arguments)
        if rep_name == "spenso::mink":
            head = "eta"
        elif rep_name == "spenso::bis":
            head = "deltaSpin"
        elif rep_name == "spenso::cof" and dimension == "2":
            head = "deltaWeak"
        elif rep_name == "spenso::cof" and any(label.startswith("f") for label in labels):
            head = "deltaGen"
        elif rep_name == "spenso::cof":
            head = "deltaColor"
        else:
            head = "deltaAdj"
        return f"{head}({', '.join(labels)})"
    if name == "spenso::gamma":
        left, right, lorentz = (_slot(argument) for argument in arguments)
        return f"gamma({lorentz}; {left}, {right})"
    if name == "spenso::gamma5":
        left, right = (_slot(argument) for argument in arguments)
        return f"gamma5({left}, {right})"

    clean_name = _short_name(name)
    aliases = {
        "DiracVector": "V",
        "DiracAxial": "A",
        "DiracScalar": "S",
        "DiracPseudoscalar": "P",
        "pcomp": "p",
        "weak_eps2": "epsWeak",
    }
    clean_name = aliases.get(clean_name, clean_name)
    return f"{clean_name}({', '.join(pretty_expression(arg) for arg in arguments)})"


def _reference_rule_in_order(reference, requested_fields: tuple[str, ...]) -> str:
    """Relabel FeynRules ``Ext``/``FV`` legs into a requested field order."""

    old_positions: dict[str, list[int]] = {}
    for old_position, name in enumerate(reference.fields, start=1):
        old_positions.setdefault(name, []).append(old_position)

    old_to_new: dict[int, int] = {}
    for new_position, name in enumerate(requested_fields, start=1):
        positions = old_positions.get(name)
        if not positions:
            raise ValueError(
                f"Requested fields {requested_fields} do not match {reference.fields}"
            )
        old_to_new[positions.pop(0)] = new_position

    text = re.sub(
        r"Ext\[(\d+)\]",
        lambda match: f"Ext[{old_to_new[int(match.group(1))]}]",
        reference.rule,
    )
    return re.sub(
        r"FV\[(\d+),",
        lambda match: f"FV[{old_to_new[int(match.group(1))]},",
        text,
    )


def main() -> None:
    theory = build_unbroken_sm_bfm()

    print("=== Declared Lagrangian sectors ===")
    for name in (
        "LGauge",
        "LFermions",
        "LHiggs",
        "LYukawa",
        "LGaugeFixing",
        "LGhost",
    ):
        print(f"\n{name} =")
        print(_plain(theory.lagrangians[name]))
    print("\nLSM = LGauge + LFermions + LHiggs + LYukawa + LGhost + LGaugeFixing")
    print(f"Compiled LSM terms: {len(theory.lagrangian.terms)}")

    references = load_feynrules_json(REFERENCE)
    mapping = field_map(theory.fields)
    example_matches = True
    print("\n=== Selected gauge vertices ===")
    for requested_names in EXAMPLE_VERTICES:
        signature = tuple(sorted(requested_names))
        reference = next(vertex for vertex in references if vertex.signature == signature)
        ordered_fields = tuple(mapping[name] for name in requested_names)
        reordered_reference_rule = _reference_rule_in_order(reference, requested_names)

        feynpy_raw = theory.lagrangian.feynman_rule(
            *ordered_fields,
            simplify=True,
            include_delta=False,
        )
        feynpy_canonical = canonicalize_rule(feynpy_raw, ordered_fields)
        feynrules_canonical = canonicalize_rule(
            parse_feynrules_rule(reordered_reference_rule),
            ordered_fields,
        )
        difference = (feynpy_canonical - feynrules_canonical).cancel().expand()
        matches = difference.to_canonical_string() == "0"
        example_matches = example_matches and matches

        print(f"\n--- {' / '.join(requested_names)} ---")
        print("FeynPy:")
        print(pretty_expression(feynpy_canonical))
        print("FeynRules (external legs relabelled to this order):")
        print(reordered_reference_rule)
        print(
            "Comparison after tensor canonicalization: "
            f"{'MATCH' if matches else 'MISMATCH'} "
            f"(difference = {pretty_expression(difference)})"
        )

    result = write_outputs()
    print("\n=== JSON export ===")
    print("models/UnbrokenSM_BFM/feynpy_vertices.json")
    print("models/UnbrokenSM_BFM/comparison_report.json")
    print(f"Complete comparison: {result.matched}/{result.total}")

    if not example_matches or not result.all_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
