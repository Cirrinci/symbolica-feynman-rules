#!/usr/bin/env python3
"""Simple helper to compare vertex exports from FeynRules and this Python project.

This utility intentionally performs lightweight normalization and token checks.
It is not a symbolic equivalence checker.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NAME_MAP = {
    "qL.bar": "qLbar",
    "uR.bar": "uRbar",
    "dR.bar": "dRbar",
    "lL.bar": "lLbar",
    "eR.bar": "eRbar",
    "Phi.bar": "Phibar",
}

RULE_REPLACEMENTS = {
    "qL.bar": "qLbar",
    "uR.bar": "uRbar",
    "dR.bar": "dRbar",
    "lL.bar": "lLbar",
    "eR.bar": "eRbar",
    "Phi.bar": "Phibar",
    "gamma": "Ga",
    "weak_eps2": "Eps",
    "Yu": "yu",
    "Yd": "yd",
    "Ye": "yl",
    "YuDag": "Conjugate[yu]",
    "YdDag": "Conjugate[yd]",
    "YeDag": "Conjugate[yl]",
}


@dataclass(frozen=True)
class Vertex:
    signature_raw: tuple[str, ...]
    signature_norm: tuple[str, ...]
    signature_key: tuple[str, ...]
    rule_raw: str
    rule_norm: str


@dataclass(frozen=True)
class FactorCheck:
    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class VertexCheckResult:
    signature: tuple[str, ...]
    status: str
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FeynRules vertex export vs Python vertex export with light normalization."
        )
    )
    parser.add_argument("--feynrules", required=True, help="Path to FeynRules export (txt or json)")
    parser.add_argument("--ours", required=True, help="Path to Python export (txt or json)")
    parser.add_argument(
        "--write-aligned-dir",
        help=(
            "Optional output directory. If provided, writes normalized/aligned text files "
            "for exact per-vertex inspection."
        ),
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    value = name.strip().strip("'\"")
    return NAME_MAP.get(value, value)


def normalize_rule(rule: str) -> str:
    value = rule
    for old, new in RULE_REPLACEMENTS.items():
        value = value.replace(old, new)

    # Normalize t(...) to T(...) while leaving words like "Ta" untouched.
    value = re.sub(r"(?<![A-Za-z0-9_])t\(", "T(", value)
    return value


def signature_key(signature_norm: tuple[str, ...]) -> tuple[str, ...]:
    # Compare as multisets: ordering can differ between exports.
    return tuple(sorted(signature_norm))


def split_top_level_csv(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def parse_signature_text(sig_text: str) -> tuple[str, ...]:
    s = sig_text.strip()
    if not s:
        return tuple()

    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return tuple(str(item).strip().strip("'\"") for item in parsed)
        except Exception:
            pass

    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return tuple()
        return tuple(p.strip().strip("'\"") for p in split_top_level_csv(inner))

    return tuple(p.strip().strip("'\"") for p in split_top_level_csv(s))


def parse_text_vertices(text: str) -> list[Vertex]:
    entries: list[tuple[tuple[str, ...], str]] = []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Vertex:"):
            sig_text = line.split(":", 1)[1].strip()
            signature = parse_signature_text(sig_text)
            i += 1
            rule = ""
            while i < len(lines):
                probe = lines[i].strip()
                if probe.startswith("Rule:"):
                    rule = probe.split(":", 1)[1].strip()
                    i += 1
                    break
                if probe.startswith("Vertex:") or probe.startswith("Signature:"):
                    break
                i += 1
            entries.append((signature, rule))
            continue

        if line.startswith("Signature:"):
            sig_text = line.split(":", 1)[1].strip()
            signature = parse_signature_text(sig_text)
            i += 1
            rule = ""
            while i < len(lines):
                probe = lines[i].strip()
                if probe.startswith("Rule:"):
                    rule = probe.split(":", 1)[1].strip()
                    i += 1
                    break
                if probe.startswith("Signature:") or probe.startswith("Vertex "):
                    break
                i += 1
            entries.append((signature, rule))
            continue

        i += 1

    return [build_vertex(sig, rule) for sig, rule in entries]


def _extract_vertices_from_json_payload(payload: Any) -> list[tuple[tuple[str, ...], str]]:
    out: list[tuple[tuple[str, ...], str]] = []

    if isinstance(payload, dict):
        if "vertices" in payload and isinstance(payload["vertices"], list):
            out.extend(_extract_vertices_from_json_payload(payload["vertices"]))

        if "signature" in payload and "rule" in payload:
            signature = payload["signature"]
            rule = payload["rule"]
            if isinstance(signature, str):
                sig_tuple = parse_signature_text(signature)
            elif isinstance(signature, (list, tuple)):
                sig_tuple = tuple(str(x) for x in signature)
            else:
                sig_tuple = tuple()
            out.append((sig_tuple, str(rule)))

        # Support map-style export: {"('qL.bar','qL','B')": "...rule..."}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                key_strip = key.strip()
                if key_strip.startswith(("(", "[", "{")):
                    sig_tuple = parse_signature_text(key_strip)
                    out.append((sig_tuple, value))

    elif isinstance(payload, list):
        for item in payload:
            out.extend(_extract_vertices_from_json_payload(item))

    return out


def parse_json_vertices(text: str) -> list[Vertex]:
    payload = json.loads(text)
    entries = _extract_vertices_from_json_payload(payload)
    return [build_vertex(sig, rule) for sig, rule in entries]


def parse_vertices(path: Path) -> list[Vertex]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        vertices = parse_json_vertices(text)
        if vertices:
            return vertices

    # Fallback to text parser for txt or if JSON did not match expected shape.
    return parse_text_vertices(text)


def build_vertex(signature_raw: tuple[str, ...], rule_raw: str) -> Vertex:
    sig_norm = tuple(normalize_name(x) for x in signature_raw)
    rule_norm = normalize_rule(rule_raw)
    return Vertex(
        signature_raw=signature_raw,
        signature_norm=sig_norm,
        signature_key=signature_key(sig_norm),
        rule_raw=rule_raw,
        rule_norm=rule_norm,
    )


def has_any(rule: str, needles: list[str]) -> bool:
    lowered = rule.lower()
    return any(n.lower() in lowered for n in needles)


def has_fraction(rule: str, numerator: int, denominator: int) -> bool:
    # Handle plain forms (1/6, -1/3, I/6, 1𝑖/6, etc.) by allowing an optional
    # multiplicative token before the slash fraction.
    escaped = rf"(?<!\d){numerator}\s*(?:\*?\s*[Ii𝑖])?\s*/\s*{denominator}(?!\d)"
    if re.search(escaped, rule) is not None:
        return True

    # Also handle parenthesized forms often seen in Mathematica exports:
    # (I/6), ((2*I)/3), (-1/2*I), etc.
    compact = re.sub(r"\s+", "", rule)
    if numerator >= 0:
        frac = rf"\(?{numerator}(?:\*[Ii𝑖])?/{denominator}\)?"
    else:
        abs_num = abs(numerator)
        frac = rf"\(-?{abs_num}(?:\*[Ii𝑖])?/{denominator}\)|-{abs_num}(?:\*[Ii𝑖])?/{denominator}"
    if re.search(frac, compact) is not None:
        return True

    # Mathematica often writes factors as I/6 or (2*I)/3. Accept these as
    # matching the expected rational hypercharge factors.
    abs_num = abs(numerator)
    if numerator == 1 and re.search(rf"\(?[Ii𝑖]\)?/{denominator}(?!\d)", compact):
        return True
    if re.search(rf"\({abs_num}\*[Ii𝑖]\)/{denominator}(?!\d)", compact):
        return True
    if numerator < 0 and re.search(rf"-\({abs_num}\*[Ii𝑖]\)/{denominator}(?!\d)", compact):
        return True

    return False


def has_weak_delta(rule: str) -> bool:
    return has_any(rule, ["IndexDelta[Index[SU2D", "g(cof(2,"])


def has_color_delta(rule: str) -> bool:
    return has_any(rule, ["IndexDelta[Index[Colour", "g(cof(3, c"])


def has_generation_delta(rule: str) -> bool:
    return has_any(rule, ["IndexDelta[Index[Generation", "g(cof(3, fl"])


def has_spin_delta(rule: str) -> bool:
    return has_any(rule, ["IndexDelta[Index[Spin", "g(bis(4,"])


def has_gamma(rule: str) -> bool:
    return has_any(rule, ["Ga(", "Ga[", "gamma(", "gamma["])


def has_su2_generator(rule: str) -> bool:
    return has_any(rule, ["Ta[Index[SU2W", "T(coad(3,"])


def has_su3_generator(rule: str) -> bool:
    return has_any(rule, ["T[Index[Gluon", "T(coad(8,"])


def has_weak_epsilon(rule: str) -> bool:
    return has_any(rule, ["Eps[", "Eps(", "weak_eps2("])


def check_expected_factors(rule: str, expected: list[tuple[str, str]]) -> list[FactorCheck]:
    checks: list[FactorCheck] = []

    for label, kind in expected:
        ok = False
        detail = ""

        if kind == "g1":
            ok = has_any(rule, ["g1"])
            detail = "contains g1"
        elif kind == "g2":
            ok = has_any(rule, ["g2"])
            detail = "contains g2"
        elif kind == "g3":
            ok = has_any(rule, ["g3"])
            detail = "contains g3"
        elif kind == "frac_1_6":
            ok = has_fraction(rule, 1, 6)
            detail = "contains 1/6"
        elif kind == "frac_2_3":
            ok = has_fraction(rule, 2, 3)
            detail = "contains 2/3"
        elif kind == "frac_minus_1_3":
            ok = has_fraction(rule, -1, 3) or has_any(rule, ["(-1/3", "-1𝑖/3", "-I/3"])
            detail = "contains -1/3"
        elif kind == "frac_minus_1_2":
            ok = has_fraction(rule, -1, 2) or has_any(rule, ["(-1/2", "-1𝑖/2", "-I/2"])
            detail = "contains -1/2"
        elif kind == "minus_one":
            ok = has_any(rule, ["(-I)", "-1𝑖", "-I*"])
            detail = "contains -1"
        elif kind == "gamma":
            ok = has_gamma(rule)
            detail = "contains gamma / Ga"
        elif kind == "weak_delta":
            ok = has_weak_delta(rule)
            detail = "contains weak-index delta"
        elif kind == "color_delta":
            ok = has_color_delta(rule)
            detail = "contains color delta"
        elif kind == "generation_delta":
            ok = has_generation_delta(rule)
            detail = "contains generation delta"
        elif kind == "spin_delta":
            ok = has_spin_delta(rule)
            detail = "contains spin delta"
        elif kind == "su2_generator":
            ok = has_su2_generator(rule)
            detail = "contains SU2 generator"
        elif kind == "su3_generator":
            ok = has_su3_generator(rule)
            detail = "contains SU3 generator"
        elif kind == "g1g2":
            ok = has_any(rule, ["g1"]) and has_any(rule, ["g2"])
            detail = "contains g1 and g2"
        elif kind == "yd":
            ok = has_any(rule, ["yd(", "yd["])
            detail = "contains yd"
        elif kind == "yl":
            ok = has_any(rule, ["yl(", "yl["])
            detail = "contains yl"
        elif kind == "yu":
            ok = has_any(rule, ["yu(", "yu["])
            detail = "contains yu"
        elif kind == "weak_epsilon":
            ok = has_weak_epsilon(rule)
            detail = "contains weak epsilon"
        else:
            detail = f"unknown check kind: {kind}"

        checks.append(FactorCheck(label=label, ok=ok, detail=detail))

    return checks


def check_hc_yukawa(rule: str, base: str) -> tuple[str, str]:
    base_tokens = {
        "yu": ["yu(", "yu["],
        "yd": ["yd(", "yd["],
        "yl": ["yl(", "yl["],
    }[base]
    conj_tokens = {
        "yu": ["Conjugate[yu", "yuDag"],
        "yd": ["Conjugate[yd", "ydDag"],
        "yl": ["Conjugate[yl", "ylDag"],
    }[base]

    if has_any(rule, conj_tokens):
        return "PASS", f"contains conjugate {base}"
    if has_any(rule, base_tokens):
        return "WARNING", f"contains non-conjugate {base} (known mismatch convention)"
    return "FAIL", f"missing {base} and conjugate {base}"


def group_by_signature(vertices: list[Vertex]) -> dict[tuple[str, ...], list[Vertex]]:
    grouped: dict[tuple[str, ...], list[Vertex]] = {}
    for vertex in vertices:
        grouped.setdefault(vertex.signature_key, []).append(vertex)
    return grouped


def first_rule_for(grouped: dict[tuple[str, ...], list[Vertex]], key: tuple[str, ...]) -> str | None:
    entries = grouped.get(key)
    if not entries:
        return None
    return entries[0].rule_norm


def format_sig(sig_key: tuple[str, ...]) -> str:
    return "{" + ", ".join(sig_key) + "}"


def compare_signatures(
    fr_grouped: dict[tuple[str, ...], list[Vertex]],
    our_grouped: dict[tuple[str, ...], list[Vertex]],
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]]]:
    fr_set = set(fr_grouped)
    our_set = set(our_grouped)
    both = sorted(fr_set & our_set)
    fr_only = sorted(fr_set - our_set)
    our_only = sorted(our_set - fr_set)
    return both, fr_only, our_only


def selected_vertex_specs() -> list[tuple[tuple[str, ...], list[tuple[str, str]], str | None]]:
    # (signature, expected_factors, optional h.c. Yukawa base)
    specs = [
        (
            tuple(sorted(("qLbar", "qL", "B"))),
            [
                ("g1", "g1"),
                ("1/6", "frac_1_6"),
                ("gamma", "gamma"),
                ("weak delta", "weak_delta"),
                ("color delta", "color_delta"),
                ("generation delta", "generation_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("uRbar", "uR", "B"))),
            [
                ("g1", "g1"),
                ("2/3", "frac_2_3"),
                ("gamma", "gamma"),
                ("color delta", "color_delta"),
                ("generation delta", "generation_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("dRbar", "dR", "B"))),
            [
                ("g1", "g1"),
                ("-1/3", "frac_minus_1_3"),
                ("gamma", "gamma"),
                ("color delta", "color_delta"),
                ("generation delta", "generation_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("lLbar", "lL", "B"))),
            [
                ("g1", "g1"),
                ("-1/2", "frac_minus_1_2"),
                ("gamma", "gamma"),
                ("weak delta", "weak_delta"),
                ("generation delta", "generation_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("eRbar", "eR", "B"))),
            [
                ("g1", "g1"),
                ("-1", "minus_one"),
                ("gamma", "gamma"),
                ("generation delta", "generation_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("qLbar", "qL", "Wi"))),
            [
                ("g2", "g2"),
                ("gamma", "gamma"),
                ("SU2 generator", "su2_generator"),
            ],
            None,
        ),
        (
            tuple(sorted(("qLbar", "qL", "G"))),
            [
                ("g3", "g3"),
                ("gamma", "gamma"),
                ("SU3 generator", "su3_generator"),
            ],
            None,
        ),
        (
            tuple(sorted(("Phibar", "Phi", "B", "Wi"))),
            [
                ("g1*g2", "g1g2"),
                ("SU2 generator", "su2_generator"),
            ],
            None,
        ),
        (
            tuple(sorted(("qLbar", "dR", "Phi"))),
            [
                ("Yd", "yd"),
                ("spin delta", "spin_delta"),
                ("color delta", "color_delta"),
                ("weak delta", "weak_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("lLbar", "eR", "Phi"))),
            [
                ("Ye", "yl"),
                ("spin delta", "spin_delta"),
                ("weak delta", "weak_delta"),
            ],
            None,
        ),
        (
            tuple(sorted(("qLbar", "uR", "Phibar"))),
            [
                ("Yu", "yu"),
                ("spin delta", "spin_delta"),
                ("color delta", "color_delta"),
                ("weak epsilon", "weak_epsilon"),
            ],
            None,
        ),
        (
            tuple(sorted(("dRbar", "qL", "Phibar"))),
            [
                ("spin delta", "spin_delta"),
                ("color delta", "color_delta"),
                ("weak delta", "weak_delta"),
            ],
            "yd",
        ),
        (
            tuple(sorted(("eRbar", "lL", "Phibar"))),
            [
                ("spin delta", "spin_delta"),
                ("weak delta", "weak_delta"),
            ],
            "yl",
        ),
        (
            tuple(sorted(("uRbar", "qL", "Phi"))),
            [
                ("spin delta", "spin_delta"),
                ("color delta", "color_delta"),
                ("weak epsilon", "weak_epsilon"),
            ],
            "yu",
        ),
    ]
    return specs


def evaluate_selected_vertices(
    fr_grouped: dict[tuple[str, ...], list[Vertex]],
    our_grouped: dict[tuple[str, ...], list[Vertex]],
) -> list[VertexCheckResult]:
    results: list[VertexCheckResult] = []

    for sig, expected, hc_base in selected_vertex_specs():
        fr_rule = first_rule_for(fr_grouped, sig)
        our_rule = first_rule_for(our_grouped, sig)

        if fr_rule is None and our_rule is None:
            results.append(
                VertexCheckResult(
                    signature=sig,
                    status="WARNING",
                    details="missing in both exports",
                )
            )
            continue

        if fr_rule is None:
            results.append(
                VertexCheckResult(
                    signature=sig,
                    status="FAIL",
                    details="missing in FeynRules export",
                )
            )
            continue

        if our_rule is None:
            results.append(
                VertexCheckResult(
                    signature=sig,
                    status="FAIL",
                    details="missing in Python export",
                )
            )
            continue

        details: list[str] = []
        hard_fail = False
        has_warning = False

        fr_checks = check_expected_factors(fr_rule, expected)
        our_checks = check_expected_factors(our_rule, expected)

        for check in fr_checks:
            if not check.ok:
                hard_fail = True
                details.append(f"FR missing {check.label}")

        for check in our_checks:
            if not check.ok:
                hard_fail = True
                details.append(f"OURS missing {check.label}")

        if hc_base is not None:
            fr_hc_status, fr_hc_msg = check_hc_yukawa(fr_rule, hc_base)
            our_hc_status, our_hc_msg = check_hc_yukawa(our_rule, hc_base)

            if fr_hc_status == "FAIL" or our_hc_status == "FAIL":
                hard_fail = True
            if fr_hc_status == "WARNING" or our_hc_status == "WARNING":
                has_warning = True

            if fr_hc_status != "PASS":
                details.append(f"FR h.c. Yukawa: {fr_hc_msg}")
            if our_hc_status != "PASS":
                details.append(f"OURS h.c. Yukawa: {our_hc_msg}")

        if hard_fail:
            status = "FAIL"
        elif has_warning:
            status = "WARNING"
        else:
            status = "PASS"

        if not details:
            details.append("all expected factors found")

        results.append(
            VertexCheckResult(
                signature=sig,
                status=status,
                details="; ".join(details),
            )
        )

    return results


def print_signature_section(
    both: list[tuple[str, ...]],
    fr_only: list[tuple[str, ...]],
    our_only: list[tuple[str, ...]],
) -> None:
    print("=== Stage 1: Signature Comparison ===")
    print(f"In both outputs ({len(both)}):")
    for sig in both:
        print(f"  - {format_sig(sig)}")

    print(f"Only in FeynRules ({len(fr_only)}):")
    for sig in fr_only:
        print(f"  - {format_sig(sig)}")

    print(f"Only in Python output ({len(our_only)}):")
    for sig in our_only:
        print(f"  - {format_sig(sig)}")
    print()


def print_selected_vertex_section(results: list[VertexCheckResult]) -> None:
    print("=== Stage 2: Expected-Factor Checks (Selected SM Vertices) ===")
    for result in results:
        print(f"[{result.status}] {format_sig(result.signature)}")
        print(f"  {result.details}")
    print()


def print_summary(results: list[VertexCheckResult]) -> None:
    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    warn_count = sum(1 for r in results if r.status == "WARNING")

    print("=== Stage 3: Summary ===")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"WARNING: {warn_count}")


def write_aligned_exports(
    fr_grouped: dict[tuple[str, ...], list[Vertex]],
    our_grouped: dict[tuple[str, ...], list[Vertex]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_keys = sorted(set(fr_grouped) | set(our_grouped))
    fr_lines: list[str] = []
    our_lines: list[str] = []
    aligned_lines: list[str] = []

    for idx, key in enumerate(all_keys, start=1):
        fr_rule = first_rule_for(fr_grouped, key)
        our_rule = first_rule_for(our_grouped, key)

        fr_rule_text = fr_rule if fr_rule is not None else "<MISSING>"
        our_rule_text = our_rule if our_rule is not None else "<MISSING>"

        fr_lines.append(f"Vertex {idx}")
        fr_lines.append(f"Signature: {format_sig(key)}")
        fr_lines.append(f"Rule: {fr_rule_text}")
        fr_lines.append("")

        our_lines.append(f"Vertex {idx}")
        our_lines.append(f"Signature: {format_sig(key)}")
        our_lines.append(f"Rule: {our_rule_text}")
        our_lines.append("")

        if fr_rule is not None and our_rule is not None:
            status = "BOTH"
        elif fr_rule is not None:
            status = "ONLY_FEYNRULES"
        else:
            status = "ONLY_OURS"

        aligned_lines.append("=" * 80)
        aligned_lines.append(f"Vertex {idx}")
        aligned_lines.append(f"Signature: {format_sig(key)}")
        aligned_lines.append(f"Status: {status}")
        aligned_lines.append("FeynRules Rule:")
        aligned_lines.append(fr_rule_text)
        aligned_lines.append("Python Rule:")
        aligned_lines.append(our_rule_text)
        aligned_lines.append("")

    (output_dir / "normalized_feynrules.txt").write_text(
        "\n".join(fr_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "normalized_python.txt").write_text(
        "\n".join(our_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "aligned_vertex_comparison.txt").write_text(
        "\n".join(aligned_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    fr_path = Path(args.feynrules)
    our_path = Path(args.ours)

    fr_vertices = parse_vertices(fr_path)
    our_vertices = parse_vertices(our_path)

    fr_grouped = group_by_signature(fr_vertices)
    our_grouped = group_by_signature(our_vertices)

    both, fr_only, our_only = compare_signatures(fr_grouped, our_grouped)
    selected_results = evaluate_selected_vertices(fr_grouped, our_grouped)

    if args.write_aligned_dir:
        write_aligned_exports(fr_grouped, our_grouped, Path(args.write_aligned_dir))

    print_signature_section(both, fr_only, our_only)
    print_selected_vertex_section(selected_results)
    print_summary(selected_results)

    return 1 if any(r.status == "FAIL" for r in selected_results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
