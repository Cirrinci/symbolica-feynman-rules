"""Display helpers for terminal-friendly model and vertex inspection."""

from __future__ import annotations

import re

from symbolica import AtomType

from symbolic.tensor_canonicalization import canonize_full

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

_FUNCTION_ALIASES = {
    "DiracVector": "V",
    "DiracAxial": "A",
    "DiracScalar": "S",
    "DiracPseudoscalar": "P",
    "pcomp": "p",
    "weak_eps2": "epsWeak",
}


def _normalize_sum_text(text: str) -> str:
    return re.sub(r"\s*\+\s*-", " - ", text).replace(" + -", " - ")


def clean_text(value) -> str:
    """Strip ANSI / internal unicode formatting noise from a stringable value."""

    return _normalize_sum_text(
        ANSI_ESCAPE_RE.sub("", str(value))
        .replace("1𝑖", "I")
        .replace("𝑖", "I")
    )


def _short_name(name: str) -> str:
    short = name.rsplit("::", 1)[-1]
    dummy = re.match(r"canon_dummy_\d+_(\d+)$", short)
    return f"x{dummy.group(1)}" if dummy else short


def _slot(expression) -> str:
    arguments = tuple(expression)
    return pretty_expression(arguments[1]) if len(arguments) == 2 else pretty_expression(expression)


def _delta_head(representation, labels: tuple[str, ...]) -> str:
    rep_name = representation.get_name()
    dimension = tuple(representation)[0].to_canonical_string()
    if rep_name == "spenso::mink":
        return "eta"
    if rep_name == "spenso::bis":
        return "deltaSpin"
    if rep_name == "spenso::cof" and dimension == "2":
        return "deltaWeak"
    if rep_name == "spenso::cof":
        return "deltaGen" if any(label.startswith("f") for label in labels) else "deltaColor"
    if rep_name == "spenso::coad" and dimension == "3":
        return "deltaWeakAdj"
    return "deltaColorAdj"


def _pretty_product(expression, *, parent_precedence: int) -> str:
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
            text = f"-{rendered}"
            return f"({text})" if parent_precedence > 2 else text

    rendered = " * ".join(
        pretty_expression(factor, parent_precedence=2) for factor in factors
    )
    return f"({rendered})" if parent_precedence > 2 else rendered


def pretty_expression(expression, *, parent_precedence: int = 0) -> str:
    """Render a Symbolica expression without ``spenso::`` / ``python::`` noise."""

    kind = expression.get_type()
    if kind == AtomType.Num:
        return clean_text(expression.to_canonical_string())
    if kind == AtomType.Var:
        return _short_name(expression.get_name())
    if kind == AtomType.Add:
        rendered = _normalize_sum_text(
            " + ".join(
                pretty_expression(term, parent_precedence=1)
                for term in expression
            )
        )
        return f"({rendered})" if parent_precedence > 1 else rendered
    if kind == AtomType.Mul:
        return _pretty_product(expression, parent_precedence=parent_precedence)
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
        head = "T_W" if tuple(arguments[0])[0].to_canonical_string() == "3" else "T_C"
        return f"{head}({', '.join(_slot(argument) for argument in arguments)})"
    if name == "spenso::f":
        head = "f_W" if tuple(arguments[0])[0].to_canonical_string() == "3" else "f_C"
        return f"{head}({', '.join(_slot(argument) for argument in arguments)})"
    if name == "spenso::g":
        labels = tuple(_slot(argument) for argument in arguments)
        return f"{_delta_head(arguments[0], labels)}({', '.join(labels)})"
    if name == "spenso::gamma":
        left, right, lorentz = (_slot(argument) for argument in arguments)
        return f"gamma({lorentz}; {left}, {right})"
    if name == "spenso::gamma5":
        left, right = (_slot(argument) for argument in arguments)
        return f"gamma5({left}, {right})"

    clean_name = _FUNCTION_ALIASES.get(_short_name(name), _short_name(name))
    return f"{clean_name}({', '.join(pretty_expression(arg) for arg in arguments)})"


def format_rule(model, expression, *, canonicalize: bool = False) -> str:
    """Return one rule in a terminal-friendly text form."""

    formatted = expression
    if canonicalize:
        formatted = canonize_full(
            expression,
            infer_indices=True,
            field_heads=tuple(model.fields),
            run_color=False,
        )
    if hasattr(formatted, "get_type"):
        return pretty_expression(formatted)
    return clean_text(formatted)


def show_result(title: str, result, *, model=None, canonicalize: bool = False) -> None:
    """Print one formatted expression or a mapping of vertex rules."""

    print("==========")
    print(title)
    if isinstance(result, dict):
        print(f"{len(result)} vertex signature(s)")
        print()
        for signature, expression in result.items():
            print("Vertex:", " / ".join(signature))
            rendered = (
                format_rule(model, expression, canonicalize=canonicalize)
                if model is not None
                else clean_text(expression)
            )
            print("Rule:", rendered)
            print()
        return

    rendered = (
        format_rule(model, result, canonicalize=canonicalize)
        if model is not None
        else clean_text(result)
    )
    print(rendered)
    print()


def show_model(model, *fields, canonicalize: bool = False, include_delta: bool = False) -> None:
    """Print the declared source and extracted rules for one model."""

    source_terms = model.lagrangian_decl.source_terms
    if source_terms:
        lagrangian_source = (
            sum(source_terms[1:], source_terms[0])
            if len(source_terms) > 1
            else source_terms[0]
        )
        show_result("Lagrangian", lagrangian_source)
    else:
        show_result("Lagrangian", "<empty>")

    lagrangian = model.lagrangian()
    if fields:
        show_result(
            "Feynman Rule",
            lagrangian.feynman_rule(*fields, include_delta=include_delta),
            model=model,
            canonicalize=canonicalize,
        )
        return

    show_result(
        "Feynman Rules",
        lagrangian.feynman_rule(include_delta=include_delta),
        model=model,
        canonicalize=canonicalize,
    )
