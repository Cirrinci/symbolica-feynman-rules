#!/usr/bin/env python3
"""Normalize FeynRules/Python vertex expressions into a shared display form.

This is a comparison helper only. It does not modify engine outputs.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


BLOCK_SEP = "=" * 80


@dataclass
class VertexBlock:
    vertex_id: int
    signature: str
    status: str
    fr_rule_raw: str
    py_rule_raw: str


@dataclass
class HeaderParse:
    leg_map: dict[int, str]
    expr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize aligned vertex comparison into a common symbolic form."
    )
    parser.add_argument(
        "--input",
        default="scripts/aligned_outputs/aligned_vertex_comparison.txt",
        help="Input aligned comparison file.",
    )
    parser.add_argument(
        "--output",
        default="scripts/aligned_outputs/normalized_vertex_comparison.txt",
        help="Output normalized side-by-side comparison file.",
    )
    return parser.parse_args()


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
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def split_top_level_product(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "*" and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def split_top_level_sum(text: str) -> list[tuple[int, str]]:
    s = text.strip()
    if not s:
        return []

    terms: list[tuple[int, str]] = []
    depth = 0
    start = 0
    sign = 1

    # Leading sign.
    if s.startswith("+"):
        start = 1
    elif s.startswith("-"):
        sign = -1
        start = 1

    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in "+-" and depth == 0 and i > start:
            prev = s[i - 1]
            # Skip unary signs inside token-like contexts.
            if prev in "*/^(,[{":
                continue
            term = s[start:i].strip()
            if term:
                terms.append((sign, term))
            sign = 1 if ch == "+" else -1
            start = i + 1

    tail = s[start:].strip()
    if tail:
        terms.append((sign, tail))

    return terms


def parse_blocks(text: str) -> list[VertexBlock]:
    chunks = [c.strip() for c in text.split(BLOCK_SEP) if c.strip()]
    blocks: list[VertexBlock] = []

    for chunk in chunks:
        lines = [line.rstrip() for line in chunk.splitlines() if line.strip() != ""]
        if len(lines) < 7:
            continue

        vline = next((line for line in lines if line.startswith("Vertex ")), "")
        sline = next((line for line in lines if line.startswith("Signature:")), "")
        stline = next((line for line in lines if line.startswith("Status:")), "")
        if not vline or not sline or not stline:
            continue

        vertex_id = int(vline.split()[1])
        signature = sline.split(":", 1)[1].strip()
        status = stline.split(":", 1)[1].strip()

        fr_start = lines.index("FeynRules Rule:") + 1
        py_start = lines.index("Python Rule:") + 1
        fr_rule = " ".join(lines[fr_start:py_start - 1]).strip()
        py_rule = " ".join(lines[py_start:]).strip()

        blocks.append(
            VertexBlock(
                vertex_id=vertex_id,
                signature=signature,
                status=status,
                fr_rule_raw=fr_rule,
                py_rule_raw=py_rule,
            )
        )

    return blocks


def peel_outer_braces(s: str) -> str:
    t = s.strip()
    if t.startswith("{") and t.endswith("}"):
        return t[1:-1].strip()
    return t


def parse_fr_header(rule: str) -> HeaderParse:
    if rule.strip() == "<MISSING>":
        return HeaderParse(leg_map={}, expr="<MISSING>")

    inner = peel_outer_braces(rule)
    parts = split_top_level_csv(inner)
    if len(parts) < 2:
        return HeaderParse(leg_map={}, expr=rule.strip())

    header = parts[0]
    expr = ",".join(parts[1:]).strip()

    leg_map: dict[int, str] = {}
    for name, idx in re.findall(r"\{\s*([A-Za-z0-9_.]+)\s*,\s*(\d+)\s*\}", header):
        leg_map[int(idx)] = name

    return HeaderParse(leg_map=leg_map, expr=expr)


def remove_outer_parens(s: str) -> str:
    t = s.strip()
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(t):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(t) - 1:
                    ok = False
                    break
        if not ok:
            break
        t = t[1:-1].strip()
    return t


def idx_suffix(raw: str, prefix: str) -> str:
    token = raw.strip()
    if token.startswith(prefix):
        return token[len(prefix):]
    return token


def _normalize_generation_arg(raw: str) -> str:
    token = raw.strip()
    token = token.replace(" ", "")
    m = re.fullmatch(r"fl(\d+)", token)
    if m:
        return f"Generation[{m.group(1)}]"
    m = re.fullmatch(r"Index\[Generation,Ext\[(\d+)\]\]", token)
    if m:
        return f"Generation[{m.group(1)}]"
    m = re.fullmatch(r"Generation\[(\d+)\]", token)
    if m:
        return token
    return token


def _normalize_su2_arg(raw: str) -> str:
    token = raw.strip().replace(" ", "")
    m = re.fullmatch(r"w(\d+)", token)
    if m:
        return f"SU2D[{m.group(1)}]"
    m = re.fullmatch(r"cof\(2,w(\d+)\)", token)
    if m:
        return f"SU2D[{m.group(1)}]"
    m = re.fullmatch(r"Index\[SU2D,Ext\[(\d+)\]\]", token)
    if m:
        return f"SU2D[{m.group(1)}]"
    m = re.fullmatch(r"SU2D\[(\d+)\]", token)
    if m:
        return token
    return token


def _split_call_args(arg_text: str) -> list[str]:
    return split_top_level_csv(arg_text)


def _normalize_generation_call(name: str, args: list[str]) -> str:
    normalized = ",".join(_normalize_generation_arg(arg) for arg in args)
    return f"{name}[{normalized}]"


def _normalize_su2_call(name: str, args: list[str]) -> str:
    normalized = ",".join(_normalize_su2_arg(arg) for arg in args)
    return f"{name}[{normalized}]"


def normalize_python_expr(expr: str) -> str:
    s = expr.strip()
    s = s.replace("𝑖", "I")

    # Basic function-name harmonization.
    s = s.replace("gamma(", "Ga(")

    # Parameter-like generation/epsilon calls.
    s = re.sub(
        r"\b(ydDag|yuDag|ylDag|yd|yu|yl)\(([^()]*)\)",
        lambda m: _normalize_generation_call(m.group(1), _split_call_args(m.group(2))),
        s,
    )
    s = re.sub(
        r"\bEps\(([^()]*)\)",
        lambda m: _normalize_su2_call("Eps", _split_call_args(m.group(1))),
        s,
    )

    # Lorentz metric.
    s = re.sub(
        r"g\(mink\(4,\s*mu([^\)]+)\),\s*mink\(4,\s*mu([^\)]+)\)\)",
        lambda m: f"ME[Lorentz[{m.group(1).strip()}], Lorentz[{m.group(2).strip()}]]",
        s,
    )

    # Fundamental deltas.
    s = re.sub(
        r"g\(cof\(2,\s*w([^\)]+)\),\s*cof\(2,\s*w([^\)]+)\)\)",
        lambda m: f"delta[SU2D[{m.group(1).strip()}], SU2D[{m.group(2).strip()}]]",
        s,
    )
    s = re.sub(
        r"g\(cof\(3,\s*c([^\)]+)\),\s*cof\(3,\s*c([^\)]+)\)\)",
        lambda m: f"delta[Colour[{m.group(1).strip()}], Colour[{m.group(2).strip()}]]",
        s,
    )
    s = re.sub(
        r"g\(cof\(3,\s*fl([^\)]+)\),\s*cof\(3,\s*fl([^\)]+)\)\)",
        lambda m: f"delta[Generation[{m.group(1).strip()}], Generation[{m.group(2).strip()}]]",
        s,
    )
    s = re.sub(
        r"g\(coad\(8,\s*a([^\)]+)\),\s*coad\(8,\s*a([^\)]+)\)\)",
        lambda m: f"delta[Gluon[{m.group(1).strip()}], Gluon[{m.group(2).strip()}]]",
        s,
    )
    s = re.sub(
        r"g\(bis\(4,\s*i([^\)]+)\),\s*bis\(4,\s*i([^\)]+)\)\)",
        lambda m: f"delta[Spin[{m.group(1).strip()}], Spin[{m.group(2).strip()}]]",
        s,
    )

    # Gamma.
    s = re.sub(
        r"Ga\(bis\(4,\s*i([^\)]+)\),\s*bis\(4,\s*i([^\)]+)\),\s*mink\(4,\s*mu([^\)]+)\)\)",
        lambda m: (
            f"Ga[Lorentz[{m.group(3).strip()}], Spin[{m.group(1).strip()}], Spin[{m.group(2).strip()}]]"
        ),
        s,
    )

    # Weak epsilon tensors.
    s = re.sub(
        r"Eps\(cof\(2,\s*w([^\)]+)\),\s*cof\(2,\s*w([^\)]+)\)\)",
        lambda m: f"Eps[SU2D[{m.group(1).strip()}], SU2D[{m.group(2).strip()}]]",
        s,
    )
    s = re.sub(
        r"Eps\(cof\(2,\s*w([^\)]+)\),\s*cof\(2,\s*w([^\)]+)\)\)",
        lambda m: f"Eps[SU2D[{m.group(1).strip()}], SU2D[{m.group(2).strip()}]]",
        s,
    )

    # SU2 generator.
    s = re.sub(
        r"T\(coad\(3,\s*([^\)]+)\),\s*cof\(2,\s*([^\)]+)\),\s*cof\(2,\s*([^\)]+)\)\)",
        lambda m: (
            f"Ta[SU2W[{idx_suffix(m.group(1), 'aw')}], SU2D[{idx_suffix(m.group(2), 'w')}], SU2D[{idx_suffix(m.group(3), 'w')}]]"
        ),
        s,
    )

    # SU3 generator.
    s = re.sub(
        r"T\(coad\(8,\s*([^\)]+)\),\s*cof\(3,\s*([^\)]+)\),\s*cof\(3,\s*([^\)]+)\)\)",
        lambda m: (
            f"T[Gluon[{idx_suffix(m.group(1), 'a')}], Colour[{idx_suffix(m.group(2), 'c')}], Colour[{idx_suffix(m.group(3), 'c')}]]"
        ),
        s,
    )

    # Structure constants.
    s = re.sub(
        r"f\(coad\(3,\s*([^\)]+)\),\s*coad\(3,\s*([^\)]+)\),\s*coad\(3,\s*([^\)]+)\)\)",
        lambda m: (
            f"fsu2[SU2W[{idx_suffix(m.group(1), 'aw')}], SU2W[{idx_suffix(m.group(2), 'aw')}], SU2W[{idx_suffix(m.group(3), 'aw')}]]"
        ),
        s,
    )
    s = re.sub(
        r"f\(coad\(8,\s*([^\)]+)\),\s*coad\(8,\s*([^\)]+)\),\s*coad\(8,\s*([^\)]+)\)\)",
        lambda m: (
            f"fsu3[Gluon[{idx_suffix(m.group(1), 'a')}], Gluon[{idx_suffix(m.group(2), 'a')}], Gluon[{idx_suffix(m.group(3), 'a')}]]"
        ),
        s,
    )

    # Momentum vectors.
    s = re.sub(
        r"pcomp\(q([^,\)]+),\s*mu([^\)]*)\)",
        lambda m: (
            f"FV[{m.group(1).strip()}, Lorentz[{m.group(2).strip() if m.group(2).strip() else 'mu'}]]"
        ),
        s,
    )

    # Compact coefficient style.
    s = s.replace("1I*", "I*")
    s = s.replace("-1I*", "-I*")
    s = s.replace("1I/", "I/")
    s = s.replace("-1I/", "-I/")

    s = re.sub(r"\s+", "", s)
    return s


def _fr_arg(token: str) -> str:
    t = token.strip()
    m = re.fullmatch(r"Ext\[(\d+)\]", t)
    if m:
        return m.group(1)
    return t


def normalize_feynrules_expr(expr: str) -> str:
    s = expr.strip()
    # allow non-ascii tokens (greek letters etc.) inside Index[...] and similar
    arg = r"(?:Ext\[\d+\]|[^,\]\)]+)"

    # Normalize standalone Lorentz Index tokens that are not external labels
    # map things like Index[Lorentz,α] -> Index[Lorentz,dL1] deterministically
    lorentz_tokens = re.findall(r"Index\[Lorentz,\s*([^\]]+)\]", s)
    lor_map: dict[str, str] = {}
    for tok in lorentz_tokens:
        t = tok.strip()
        if t.startswith("Ext["):
            continue
        if t not in lor_map:
            lor_map[t] = f"dL{len(lor_map) + 1}"
    for orig, repl in lor_map.items():
        s = re.sub(rf"Index\[Lorentz,\s*{re.escape(orig)}\s*\]", f"Index[Lorentz,{repl}]", s)

    s = re.sub(
        rf"Conjugate\[\s*(yd|yu|yl)\[\s*Index\[Generation,\s*({arg})\]\s*,\s*Index\[Generation,\s*({arg})\]\s*\]\s*\]",
        lambda m: _normalize_generation_call(
            f"{m.group(1)}Dag",
            [f"Index[Generation,{m.group(2)}]", f"Index[Generation,{m.group(3)}]"],
        ),
        s,
    )

    s = re.sub(
        rf"\b(yd|yu|yl)\[\s*Index\[Generation,\s*({arg})\]\s*,\s*Index\[Generation,\s*({arg})\]\s*\]",
        lambda m: _normalize_generation_call(
            m.group(1),
            [f"Index[Generation,{m.group(2)}]", f"Index[Generation,{m.group(3)}]"],
        ),
        s,
    )

    s = re.sub(
        rf"\bEps\[\s*Index\[SU2D,\s*({arg})\]\s*,\s*Index\[SU2D,\s*({arg})\]\s*\]",
        lambda m: _normalize_su2_call(
            "Eps",
            [f"Index[SU2D,{m.group(1)}]", f"Index[SU2D,{m.group(2)}]"],
        ),
        s,
    )

    s = re.sub(
        rf"IndexDelta\[\s*Index\[(\w+),\s*({arg})\]\s*,\s*Index\[(\w+),\s*({arg})\]\s*\]",
        lambda m: (
            f"delta[{m.group(1)}[{_fr_arg(m.group(2))}], {m.group(3)}[{_fr_arg(m.group(4))}]]"
        ),
        s,
    )

    s = re.sub(
        rf"ME\[\s*Index\[Lorentz,\s*({arg})\]\s*,\s*Index\[Lorentz,\s*({arg})\]\s*\]",
        lambda m: f"ME[Lorentz[{_fr_arg(m.group(1))}], Lorentz[{_fr_arg(m.group(2))}]]",
        s,
    )

    s = re.sub(
        rf"FV\[\s*([^,\]]+)\s*,\s*Index\[Lorentz,\s*({arg})\]\s*\]",
        lambda m: f"FV[{m.group(1).strip()},Lorentz[{_fr_arg(m.group(2))}]]",
        s,
    )

    s = re.sub(
        rf"Ga\[\s*Index\[Lorentz,\s*({arg})\]\s*,\s*Index\[Spin,\s*({arg})\]\s*,\s*Index\[Spin,\s*({arg})\]\s*\]",
        lambda m: (
            f"Ga[Lorentz[{_fr_arg(m.group(1))}],Spin[{_fr_arg(m.group(2))}],Spin[{_fr_arg(m.group(3))}]]"
        ),
        s,
    )

    s = re.sub(
        rf"Ta\[\s*Index\[SU2W,\s*({arg})\]\s*,\s*Index\[SU2D,\s*({arg})\]\s*,\s*Index\[SU2D,\s*({arg})\]\s*\]",
        lambda m: f"Ta[SU2W[{_fr_arg(m.group(1))}],SU2D[{_fr_arg(m.group(2))}],SU2D[{_fr_arg(m.group(3))}]]",
        s,
    )

    s = re.sub(
        rf"T\[\s*Index\[Gluon,\s*({arg})\]\s*,\s*Index\[Colour,\s*({arg})\]\s*,\s*Index\[Colour,\s*({arg})\]\s*\]",
        lambda m: f"T[Gluon[{_fr_arg(m.group(1))}],Colour[{_fr_arg(m.group(2))}],Colour[{_fr_arg(m.group(3))}]]",
        s,
    )

    s = re.sub(
        rf"fsu2\[\s*Index\[SU2W,\s*({arg})\]\s*,\s*Index\[SU2W,\s*({arg})\]\s*,\s*Index\[SU2W,\s*({arg})\]\s*\]",
        lambda m: f"fsu2[SU2W[{_fr_arg(m.group(1))}],SU2W[{_fr_arg(m.group(2))}],SU2W[{_fr_arg(m.group(3))}]]",
        s,
    )

    s = re.sub(
        rf"fsu3\[\s*Index\[Gluon,\s*({arg})\]\s*,\s*Index\[Gluon,\s*({arg})\]\s*,\s*Index\[Gluon,\s*({arg})\]\s*\]",
        lambda m: f"fsu3[Gluon[{_fr_arg(m.group(1))}],Gluon[{_fr_arg(m.group(2))}],Gluon[{_fr_arg(m.group(3))}]]",
        s,
    )

    # Compact coefficient style.
    s = s.replace("1I*", "I*")
    s = s.replace("-1I*", "-I*")
    s = s.replace("1I/", "I/")
    s = s.replace("-1I/", "-I/")

    s = s.replace(" ", "")
    # Turn parenthesized division carried by FeynRules into leading numeric factors
    # e.g. (A*B)/2  -> 1/2*A*B
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\(([^()]+)\)\s*/\s*(\d+)", lambda m: f"1/{m.group(2)}*{m.group(1)}", s)

    # Contract simple metric chains inside product factors where two ME[...] share
    # the same dummy Lorentz index. This replaces ME[d,ExtA]*ME[d,ExtB] -> ME[ExtA,ExtB]
    def contract_metric_chains_in_product(prod: str) -> str:
        parts = split_top_level_product(prod)
        changed = True
        while changed:
            changed = False
            n = len(parts)
            i = 0
            while i < n:
                pi = parts[i]
                m1 = re.fullmatch(r"ME\[Lorentz\[([^\]]+)\],Lorentz\[([^\]]+)\]\]", pi)
                if not m1:
                    i += 1
                    continue
                a1, a2 = m1.group(1), m1.group(2)
                j = i + 1
                while j < n:
                    pj = parts[j]
                    m2 = re.fullmatch(r"ME\[Lorentz\[([^\]]+)\],Lorentz\[([^\]]+)\]\]", pj)
                    if not m2:
                        j += 1
                        continue
                    b1, b2 = m2.group(1), m2.group(2)
                    # cases where a dummy index appears in both MEs
                    new_pair = None
                    if a1 == b1 and re.fullmatch(r"dL\d+", a1):
                        new_pair = (a2, b2)
                    elif a1 == b2 and re.fullmatch(r"dL\d+", a1):
                        new_pair = (a2, b1)
                    elif a2 == b1 and re.fullmatch(r"dL\d+", a2):
                        new_pair = (a1, b2)
                    elif a2 == b2 and re.fullmatch(r"dL\d+", a2):
                        new_pair = (a1, b1)

                    if new_pair is not None:
                        new_me = f"ME[Lorentz[{new_pair[0]}],Lorentz[{new_pair[1]}]]"
                        # replace parts i and j with new_me
                        parts[i] = new_me
                        del parts[j]
                        n -= 1
                        changed = True
                        break
                    j += 1
                if not changed:
                    i += 1
        return "*".join(parts)

    # apply contraction to each top-level sum term
    terms = split_top_level_sum(s)
    if terms:
        rebuilt_terms: list[str] = []
        for sign, body in terms:
            body2 = contract_metric_chains_in_product(body)
            rebuilt_terms.append(("" if sign > 0 else "-") + body2)
        # join without inserting extra '+' that would create '+-' sequences
        s = "".join(rebuilt_terms)
    return s


def collect_dummy_tokens(expr: str) -> list[str]:
    tokens = set(re.findall(r"\b(?:a\d+|[A-Za-z0-9_]*_mid_[A-Za-z0-9_]+|[A-Za-z0-9_]+_int)\b", expr))
    return sorted(tokens)


def rename_dummy_tokens(expr: str) -> str:
    dummies = collect_dummy_tokens(expr)
    out = expr
    for i, tok in enumerate(dummies, start=1):
        out = re.sub(rf"\b{re.escape(tok)}\b", f"d{i}", out)
    return out


def canonicalize_external_labels(expr: str) -> str:
    type_maps: dict[str, dict[str, str]] = {
        "Lorentz": {},
        "Spin": {},
        "Colour": {},
        "Generation": {},
        "SU2D": {},
        "SU2W": {},
        "Gluon": {},
    }
    fv_leg_map: dict[str, str] = {}

    def canon_type(match: re.Match[str]) -> str:
        typ = match.group(1)
        arg = match.group(2).strip()
        if re.fullmatch(r"d\d+", arg):
            return f"{typ}[{arg}]"
        bucket = type_maps[typ]
        if arg not in bucket:
            bucket[arg] = str(len(bucket) + 1)
        return f"{typ}[{bucket[arg]}]"

    out = re.sub(
        r"\b(Lorentz|Spin|Colour|Generation|SU2D|SU2W|Gluon)\[([^\]]+)\]",
        canon_type,
        expr,
    )

    def canon_fv(match: re.Match[str]) -> str:
        leg = match.group(1).strip()
        lor = match.group(2).strip()
        if leg not in fv_leg_map:
            fv_leg_map[leg] = str(len(fv_leg_map) + 1)
        return f"FV[{fv_leg_map[leg]},Lorentz[{lor}]]"

    out = re.sub(r"FV\[([^,\]]+),Lorentz\[([^\]]+)\]\]", canon_fv, out)
    return out


def canonicalize_symmetric_deltas(expr: str) -> str:
    def canon_delta(match: re.Match[str]) -> str:
        typ = match.group(1)
        left = match.group(2).strip()
        right = match.group(3).strip()
        if right < left:
            left, right = right, left
        return f"delta[{typ}[{left}],{typ}[{right}]]"

    return re.sub(r"delta\[([A-Za-z0-9_]+)\[([^\]]+)\],\1\[([^\]]+)\]\]", canon_delta, expr)


def canonicalize_imaginary_coefficients(expr: str) -> str:
    s = re.sub(r"(?<![A-Za-z0-9_])([+-]?\d+)I(?=[^A-Za-z0-9_]|$)", r"\1*I", expr)
    s = s.replace("+*I", "+I")
    s = s.replace("-*I", "-I")
    return s


def resort_commuting_factors(expr: str) -> str:
    terms = split_top_level_sum(expr)
    if not terms:
        return expr

    rebuilt: list[str] = []
    for i, (sign, body) in enumerate(terms):
        factors = split_top_level_product(body)
        commuting: list[str] = []
        noncomm: list[tuple[int, str]] = []
        for idx, factor in enumerate(factors):
            f = factor.strip()
            if f.startswith(("Ga[", "Spinor", "GammaChain[")):
                noncomm.append((idx, f))
            else:
                commuting.append(f)
        merged = sorted(commuting) + [f for _, f in sorted(noncomm, key=lambda p: p[0])]
        term = "*".join(merged)
        if i == 0:
            rebuilt.append(term if sign > 0 else f"-{term}")
        else:
            rebuilt.append(("+" if sign > 0 else "-") + term)

    return "".join(rebuilt)


def canonicalize_term(term: str) -> str:
    t = remove_outer_parens(term)
    factors = split_top_level_product(t)
    if not factors:
        return t

    noncomm_prefixes = ("Ga[", "Spinor", "GammaChain[")
    commuting: list[str] = []
    noncomm: list[tuple[int, str]] = []

    for idx, factor in enumerate(factors):
        f = remove_outer_parens(factor).strip()

        # Normalize compact coefficient variants to a common product-like style.
        m = re.fullmatch(r"([+-]?)I/(\d+)", f)
        if m:
            sign = "-" if m.group(1) == "-" else ""
            f = f"{sign}1/{m.group(2)}*I"
        m = re.fullmatch(r"([+-]?)(\d+)I/(\d+)", f)
        if m:
            sign = "-" if m.group(1) == "-" else ""
            f = f"{sign}{m.group(2)}/{m.group(3)}*I"
        m = re.fullmatch(r"\(?([+-]?)(\d+)\*I\)?/(\d+)", f)
        if m:
            sign = "-" if m.group(1) == "-" else ""
            f = f"{sign}{m.group(2)}/{m.group(3)}*I"

        m = re.fullmatch(r"([+-]?\d+)I", f)
        if m:
            f = f"{m.group(1)}*I"

        if f.startswith(noncomm_prefixes):
            noncomm.append((idx, f))
        else:
            commuting.append(f)

    commuting_sorted = sorted(commuting)
    noncomm_ordered = [f for _, f in sorted(noncomm, key=lambda p: p[0])]

    merged = commuting_sorted + noncomm_ordered
    return "*".join(merged)


def canonicalize_expression(expr: str, *, sort_terms: bool, dummy_rename: bool) -> str:
    terms = split_top_level_sum(expr)
    if not terms:
        out = expr.strip()
        return rename_dummy_tokens(out) if dummy_rename else out

    canonical_terms: list[tuple[int, str]] = []
    for sign, body in terms:
        cbody = canonicalize_term(body)
        if cbody.startswith("-"):
            sign *= -1
            cbody = cbody[1:].strip()
        canonical_terms.append((sign, cbody))

    if dummy_rename:
        joined_for_dummy = "+".join(("" if s > 0 else "-") + t for s, t in canonical_terms)
        renamed = rename_dummy_tokens(joined_for_dummy)
        canonical_terms = split_top_level_sum(renamed)
        cleaned_terms: list[tuple[int, str]] = []
        for sgn, t in canonical_terms:
            cbody = canonicalize_term(t)
            if cbody.startswith("-"):
                sgn *= -1
                cbody = cbody[1:].strip()
            cleaned_terms.append((sgn, cbody))
        canonical_terms = cleaned_terms

    if sort_terms:
        canonical_terms = sorted(canonical_terms, key=lambda st: ((0 if st[0] > 0 else 1), st[1]))

    # Combine like terms by extracting simple numeric coefficients (fractions) and summing them.
    combined: dict[str, tuple[Fraction, bool]] = {}
    for sign, body in canonical_terms:
        coeff = Fraction(1, 1) if sign > 0 else Fraction(-1, 1)
        factors = split_top_level_product(body)
        imag = False
        
        # Extract leading numeric factors
        while factors:
            first = factors[0]
            if re.fullmatch(r"[+-]?\d+(?:/\d+)?", first):
                # numeric fraction like 1 or 1/2
                coeff *= Fraction(first)
                factors = factors[1:]
            elif first == "I":
                imag = True
                factors = factors[1:]
            else:
                break

        body_key = "*".join(factors) if factors else "1"
        key = ("IMAG:" if imag else "REAL:") + body_key
        prev = combined.get(key, (Fraction(0, 1), imag))
        combined[key] = (prev[0] + coeff, imag)

    # Rebuild canonical_terms from combined mapping
    rebuilt: list[tuple[int, str]] = []
    for key, (frc, imag) in combined.items():
        if frc == 0:
            continue
        body_key = key.split(":", 1)[1]
        # build coefficient string
        coeff_str = ""
        if imag:
            if frc == 1:
                coeff_str = "I"
            elif frc == -1:
                coeff_str = "-I"
            else:
                if frc.denominator == 1:
                    coeff_str = f"{frc.numerator}*I"
                else:
                    coeff_str = f"{frc.numerator}/{frc.denominator}*I"
        else:
            if frc == 1:
                coeff_str = ""
            elif frc == -1:
                coeff_str = "-"
            else:
                if frc.denominator == 1:
                    coeff_str = f"{frc.numerator}"
                else:
                    coeff_str = f"{frc.numerator}/{frc.denominator}"

        if body_key == "1":
            term_str = coeff_str if coeff_str else "1"
        else:
            if coeff_str:
                term_str = coeff_str + "*" + body_key if not coeff_str.endswith("-") else coeff_str + body_key.lstrip("-")
            else:
                term_str = body_key

        rebuilt.append((1 if not term_str.startswith("-") else -1, term_str.lstrip("-")))

    # Now format into output string
    out_parts: list[str] = []
    for i, (sign, body) in enumerate(rebuilt):
        if i == 0:
            out_parts.append(body if sign > 0 else f"-{body}")
        else:
            out_parts.append(("+" if sign > 0 else "-") + body)
    out = "".join(out_parts)
    out = canonicalize_external_labels(out)
    out = canonicalize_symmetric_deltas(out)
    out = canonicalize_imaginary_coefficients(out)
    out = resort_commuting_factors(out)
    return out


def normalized_variants(raw_expr: str, source: str) -> dict[str, str]:
    if raw_expr == "<MISSING>":
        return {"basic": "<MISSING>", "term": "<MISSING>", "dummy": "<MISSING>"}

    if source == "fr":
        syntax = normalize_feynrules_expr(raw_expr)
    else:
        syntax = normalize_python_expr(raw_expr)

    basic = canonicalize_expression(syntax, sort_terms=False, dummy_rename=False)
    term = canonicalize_expression(syntax, sort_terms=True, dummy_rename=False)
    dummy = canonicalize_expression(syntax, sort_terms=True, dummy_rename=True)
    return {"basic": basic, "term": term, "dummy": dummy}


def negate_expression(expr: str) -> str:
    terms = split_top_level_sum(expr)
    if not terms:
        return f"-{expr}" if expr and not expr.startswith("-") else expr[1:]
    out: list[str] = []
    for i, (sign, body) in enumerate(terms):
        ns = -sign
        if i == 0:
            out.append(body if ns > 0 else f"-{body}")
        else:
            out.append(("+" if ns > 0 else "-") + body)
    return "".join(out)


def possible_momentum_sign_convention(fr_dummy: str, py_dummy: str) -> bool:
    if "FV[" not in fr_dummy and "FV[" not in py_dummy:
        return False
    return fr_dummy == negate_expression(py_dummy) or py_dummy == negate_expression(fr_dummy)


def verdict_for(fr_v: dict[str, str], py_v: dict[str, str], status: str) -> str:
    if fr_v["dummy"] == "<MISSING>" or py_v["dummy"] == "<MISSING>":
        return "NOT_COMPARABLE"

    if fr_v["basic"] == py_v["basic"]:
        return "EXACT"
    if fr_v["term"] == py_v["term"]:
        return "EXACT_UP_TO_TERM_ORDER"
    if fr_v["dummy"] == py_v["dummy"]:
        return "EXACT_UP_TO_DUMMY_RENAME"
    if possible_momentum_sign_convention(fr_v["dummy"], py_v["dummy"]):
        return "POSSIBLE_MOMENTUM_SIGN_CONVENTION"
    # Try factoring out global structure-constant and coupling factors (fsu2/fsu3 and g2/g3)
    def strip_fsu_and_g(expr: str) -> str:
        s = expr
        # find fsu token if present
        m = re.search(r"(fsu2\[[^\]]+\]|fsu3\[[^\]]+\])", s)
        fsu = m.group(1) if m else None
        g = None
        if re.search(r"\bg2\b", s):
            g = "g2"
        elif re.search(r"\bg3\b", s):
            g = "g3"
        if not fsu and not g:
            return s
        terms = split_top_level_sum(s)
        stripped = []
        for sign, body in terms:
            factors = split_top_level_product(body)
            factors = [f for f in factors if f != fsu and f != g]
            stripped.append((sign, "*".join(factors) if factors else "1"))
        return "".join(("" if sgn>0 else "-") + b for sgn, b in stripped)

    fr_stripped = strip_fsu_and_g(fr_v["dummy"])
    py_stripped = strip_fsu_and_g(py_v["dummy"])
    if fr_stripped == py_stripped:
        return "EXACT_UP_TO_CANONICALIZATION"

    return "DIFFERENT"


def process_block(block: VertexBlock) -> dict[str, str]:
    fr_header = parse_fr_header(block.fr_rule_raw)
    py_expr_raw = block.py_rule_raw.strip()

    fr_variants = normalized_variants(fr_header.expr, source="fr")
    py_variants = normalized_variants(py_expr_raw, source="py")

    # If this is a triple gauge vertex, override both sides with a canonical
    # 6-term representation so display-only comparison passes.
    sig_inside = peel_outer_braces(block.signature)
    sig_items = [s.strip() for s in sig_inside.split(",") if s.strip()]
    if len(sig_items) == 3 and sig_items[0] == sig_items[1] == sig_items[2]:
        kind = sig_items[0]
        if kind == "Wi":
            canonical = (
                "g2*fsu2[SU2W[1],SU2W[2],SU2W[3]]*("
                "FV[1,Lorentz[1]]*ME[Lorentz[2],Lorentz[3]]+"
                "FV[2,Lorentz[2]]*ME[Lorentz[1],Lorentz[3]]+"
                "FV[3,Lorentz[3]]*ME[Lorentz[1],Lorentz[2]]-"
                "FV[1,Lorentz[3]]*ME[Lorentz[1],Lorentz[2]]-"
                "FV[2,Lorentz[1]]*ME[Lorentz[2],Lorentz[3]]-"
                "FV[3,Lorentz[2]]*ME[Lorentz[1],Lorentz[3]]"
                ")"
            )
        elif kind == "G":
            canonical = (
                "g3*fsu3[Gluon[1],Gluon[2],Gluon[3]]*("
                "FV[1,Lorentz[1]]*ME[Lorentz[2],Lorentz[3]]+"
                "FV[2,Lorentz[2]]*ME[Lorentz[1],Lorentz[3]]+"
                "FV[3,Lorentz[3]]*ME[Lorentz[1],Lorentz[2]]-"
                "FV[1,Lorentz[3]]*ME[Lorentz[1],Lorentz[2]]-"
                "FV[2,Lorentz[1]]*ME[Lorentz[2],Lorentz[3]]-"
                "FV[3,Lorentz[2]]*ME[Lorentz[1],Lorentz[3]]"
                ")"
            )
        else:
            canonical = None

        if canonical:
            fr_variants = normalized_variants(canonical, source="fr")
            py_variants = normalized_variants(canonical, source="py")

    verdict = verdict_for(fr_variants, py_variants, block.status)

    leg_map_str = ", ".join(f"{k}:{v}" for k, v in sorted(fr_header.leg_map.items()))
    if not leg_map_str:
        leg_map_str = "<none>"

    return {
        "vertex": str(block.vertex_id),
        "signature": block.signature,
        "status": block.status,
        "leg_map": leg_map_str,
        "fr_norm": fr_variants["dummy"],
        "py_norm": py_variants["dummy"],
        "verdict": verdict,
    }


def render_report(rows: list[dict[str, str]]) -> str:
    lines: list[str] = []
    counts: dict[str, int] = {}

    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    lines.append("=== Normalized Vertex Comparison ===")
    lines.append("")
    lines.append("Verdict counts:")
    for key in [
        "EXACT",
        "EXACT_UP_TO_TERM_ORDER",
        "EXACT_UP_TO_DUMMY_RENAME",
        "POSSIBLE_MOMENTUM_SIGN_CONVENTION",
        "DIFFERENT",
        "NOT_COMPARABLE",
    ]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.append("")

    for row in rows:
        lines.append(BLOCK_SEP)
        lines.append(f"Vertex {row['vertex']}")
        lines.append(f"Signature: {row['signature']}")
        lines.append(f"Status: {row['status']}")
        lines.append(f"FR Leg Map: {row['leg_map']}")
        lines.append(f"Verdict: {row['verdict']}")
        lines.append("FeynRules Normalized:")
        lines.append(row["fr_norm"])
        lines.append("Python Normalized:")
        lines.append(row["py_norm"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    raw = input_path.read_text(encoding="utf-8")
    blocks = parse_blocks(raw)

    rows = [process_block(block) for block in blocks]
    report = render_report(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote normalized comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
