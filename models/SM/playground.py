"""Minimal Standard Model playground.

Run with:
    .venv/bin/python models/SM/playground.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from symbolica import AtomType, Expression

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from feynpy import *  # noqa: F401,F403
from models.SM import *  # noqa: F401,F403
from symbolic.tensor_canonicalization import canonize_full

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def clean_text(value) -> str:
    return (
        ANSI_ESCAPE_RE.sub("", str(value))
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
    """Render Symbolica expressions without internal namespace noise."""

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
        factors = tuple(expression)
        if len(factors) > 1:
            first = factors[0].to_canonical_string()
            if first == "1":
                factors = factors[1:]
            elif first == "-1":
                rendered = " * ".join(
                    pretty_expression(factor, parent_precedence=2)
                    for factor in factors[1:]
                )
                rendered = f"-{rendered}"
                return f"({rendered})" if parent_precedence > 2 else rendered

        rendered = " * ".join(
            pretty_expression(factor, parent_precedence=2) for factor in factors
        )
        return f"({rendered})" if parent_precedence > 2 else rendered
    if kind == AtomType.Pow:
        base, exponent = tuple(expression)
        return (
            f"{pretty_expression(base, parent_precedence=3)}^"
            f"{pretty_expression(exponent, parent_precedence=3)}"
        )
    if kind != AtomType.Fn:
        return clean_text(expression.to_canonical_string())

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
        elif rep_name == "spenso::coad" and dimension == "3":
            head = "deltaWeakAdj"
        else:
            head = "deltaColorAdj"
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


def format_rule(model, expression, *, canonicalize: bool = False) -> str:
    formatted = expression
    if canonicalize:
        formatted = canonize_full(
            expression,
            infer_indices=True,
            field_heads=tuple(model.fields),
            run_color=False,
        )
    return pretty_expression(formatted)


def show(title: str, result, *, model=None, canonicalize: bool = False) -> None:
    print("==========")
    print(title)
    if isinstance(result, dict):
        print(f"{len(result)} vertex signature(s)")
        print()
        for signature, expression in result.items():
            print("Vertex:", " / ".join(signature))
            if model is None:
                print("Rule:", clean_text(expression))
            else:
                print("Rule:", format_rule(model, expression, canonicalize=canonicalize))
            print()
        return

    if model is None:
        print(clean_text(result))
    else:
        print(format_rule(model, result, canonicalize=canonicalize))
    print()


def show_model(model, *fields, canonicalize: bool = False) -> None:
    source_terms = model.lagrangian_decl.source_terms
    if source_terms:
        lagrangian_source = (
            sum(source_terms[1:], source_terms[0])
            if len(source_terms) > 1
            else source_terms[0]
        )
        show("Lagrangian", lagrangian_source)
    else:
        show("Lagrangian", "<empty>")

    lagrangian = model.lagrangian()
    if fields:
        show(
            "Feynman Rule",
            lagrangian.feynman_rule(*fields, include_delta=False),
            model=model,
            canonicalize=canonicalize,
        )
    else:
        show(
            "Feynman Rules",
            lagrangian.feynman_rule(include_delta=False),
            model=model,
            canonicalize=canonicalize,
        )


L_gauge_model = sm_model(L_gauge, name="SM gauge sector")
L_fermions_model = sm_model(L_fermions, name="SM fermion sector")
L_higgs_model = sm_model(L_higgs, name="SM Higgs sector")
L_yukawa_model = sm_model(L_yukawa, name="SM Yukawa sector")
L_gauge_fixing_model = sm_model(L_gauge_fixing, name="SM gauge-fixing sector")
L_ghost_model = sm_model(L_ghost, name="SM ghost sector")
L_tot_model = sm_model(L_tot, name="SM total lagrangian")

SECTOR_MODELS = {
    "Gauge Sector": L_gauge_model,
    "Fermion Sector": L_fermions_model,
    "Higgs Sector": L_higgs_model,
    "Yukawa Sector": L_yukawa_model,
    "Gauge-Fixing Sector": L_gauge_fixing_model,
    "Ghost Sector": L_ghost_model,
    "Total Lagrangian": L_tot_model,
}

custom_yukawa_model = sm_model(
    -Yd(f2, f3)
    * CKM(f1, f2)
    * QL.bar(spinor, weak_left, f1, colour)
    * dR(spinor, f3, colour)
    * Phi(weak_left),
    name="Custom Yukawa term",
)


if __name__ == "__main__":
    for title, model in SECTOR_MODELS.items():
        print(f"===== {title} =====")
        show_model(model)
        print()

    print("===== Custom Yukawa Example =====")
    show_model(custom_yukawa_model)
