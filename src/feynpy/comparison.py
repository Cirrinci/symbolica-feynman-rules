"""Symbolic comparison helpers for FeynRules vertex exports.

The comparison pipeline is deliberately split into two layers:

* an input adapter parses one external format into native Symbolica tensors;
* the format-independent layer aligns signatures, canonicalizes tensor
  indices, and proves equality by simplifying the symbolic difference.

The adapters implemented here cover the compact gauge- and matter-sector JSON
exports used by the Standard Model notebook. Additional sectors can extend
the parser without changing the signature alignment or reporting machinery.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sympy
from symbolica import AtomType, Expression, S

from symbolic.spenso_structures import (
    COLOR_FUND,
    chiral_projector_left,
    chiral_projector_right,
    gamma_matrix,
    gauge_generator,
    lorentz_metric,
    structure_constant,
)
from symbolic.tensor_canonicalization import canonize_structure_constant_products
from symbolic.vertex_engine import pcomp


@dataclass(frozen=True)
class FeynRulesVertex:
    """One vertex from a FeynRules JSON export."""

    fields: tuple[str, ...]
    rule: str
    identifier: int | str | None = None
    key: str = ""
    legs: tuple[str, ...] = ()

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(self.fields))


@dataclass(frozen=True)
class VertexComparison:
    """Result of comparing one aligned FeynRules/FeynPy vertex."""

    reference: FeynRulesVertex
    status: str
    feynpy_rule: Expression | None = None
    feynrules_rule: Expression | None = None
    difference: Expression | None = None
    detail: str = ""

    @property
    def matches(self) -> bool:
        return self.status == "MATCH"


@dataclass(frozen=True)
class VertexComparisonReport:
    """Complete signature coverage and rule-level comparison report."""

    rows: tuple[VertexComparison, ...]
    feynrules_only: tuple[tuple[str, ...], ...] = ()
    feynpy_only: tuple[tuple[str, ...], ...] = ()

    @property
    def matched(self) -> int:
        return sum(row.matches for row in self.rows)

    @property
    def mismatched(self) -> int:
        return sum(not row.matches for row in self.rows)

    @property
    def all_match(self) -> bool:
        return (
            self.mismatched == 0
            and not self.feynrules_only
            and not self.feynpy_only
        )


def load_feynrules_json(path: str | Path) -> tuple[FeynRulesVertex, ...]:
    """Load the list-style FeynRules vertex JSON format."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("FeynRules vertex JSON must contain a top-level list")

    vertices: list[FeynRulesVertex] = []
    for position, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Vertex {position} is not a JSON object")
        fields = item.get("fields")
        rule = item.get("rule")
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            raise ValueError(f"Vertex {position} has no valid 'fields' list")
        if not isinstance(rule, str):
            raise ValueError(f"Vertex {position} has no string 'rule'")
        vertices.append(
            FeynRulesVertex(
                identifier=item.get("id"),
                key=str(item.get("key", "")),
                fields=tuple(fields),
                legs=tuple(str(leg) for leg in item.get("legs", ())),
                rule=rule,
            )
        )
    return tuple(vertices)


def _as_expression(value) -> Expression:
    if isinstance(value, Expression):
        return value
    if hasattr(value, "symbol"):
        return value.symbol
    if isinstance(value, int):
        return Expression.num(value)
    raise TypeError(f"Expected a Symbolica expression or Parameter, got {value!r}")


def parse_feynrules_gauge_rule(
    rule: str,
    *,
    parameter_substitutions: Mapping[str, object] | None = None,
) -> Expression:
    """Parse the gauge tensors used by the FeynRules JSON export.

    Supported external heads are ``ME`` (Lorentz metric), ``FV`` (incoming
    momentum), and the SU(3) structure constant ``f``. The resulting
    expression uses the same native Spenso tensor heads as FeynPy.
    """

    text = rule
    text = re.sub(
        r"Index\[Lorentz,\s*Ext\[(\d+)\]\]",
        lambda match: f"mu{match.group(1)}",
        text,
    )
    text = re.sub(
        r"Index\[Gluon,\s*Ext\[(\d+)\]\]",
        lambda match: f"a{match.group(1)}",
        text,
    )
    text = re.sub(
        r"Index\[Gluon,\s*Gluon\$(\d+)\]",
        lambda match: f"a_feynrules_dummy_{match.group(1)}",
        text,
    )
    text = re.sub(
        r"ME\[([^,\]]+),\s*([^\]]+)\]",
        lambda match: lorentz_metric(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"FV\[(\d+),\s*([^\]]+)\]",
        lambda match: pcomp(
            S(f"q{match.group(1)}"),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"f\[([^,\]]+),\s*([^,\]]+),\s*([^\]]+)\]",
        lambda match: structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported FeynRules gauge syntax remains after parsing: "
            f"{text}"
        )

    expression = Expression.parse(text)
    for name, replacement in (parameter_substitutions or {}).items():
        expression = expression.replace(S(name), _as_expression(replacement))
    return expression.cancel().expand()


def parse_feynrules_matter_rule(
    rule: str,
    *,
    parameter_substitutions: Mapping[str, object] | None = None,
) -> Expression:
    """Parse the fermion-gauge tensors used by the matter JSON export.

    Supported external heads are ``Ga``, ``TensDot[Ga, ProjM/ProjP]``,
    ``IndexDelta`` on color indices, and the SU(3) fundamental generator
    ``T``. Chiral products are expanded into native gamma, gamma5, and
    spinor-metric tensors before canonical reduction.
    """

    text = rule
    projector_counter = 0

    projector_pattern = re.compile(
        r"TensDot\["
        r"Ga\[Index\[Lorentz,\s*Ext\[(\d+)\]\]\],\s*"
        r"(ProjM|ProjP)"
        r"\]\["
        r"Index\[Spin,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Spin,\s*Ext\[(\d+)\]\]"
        r"\]"
    )

    def replace_projector(match: re.Match[str]) -> str:
        nonlocal projector_counter
        projector_counter += 1
        mu = S(f"mu{match.group(1)}")
        left = S(f"i{match.group(3)}")
        right = S(f"i{match.group(4)}")
        dummy = S(f"i_feynrules_dummy_{projector_counter}")
        projector = (
            chiral_projector_left(dummy, right)
            if match.group(2) == "ProjM"
            else chiral_projector_right(dummy, right)
        )
        return (
            gamma_matrix(left, dummy, mu) * projector
        ).to_canonical_string()

    text = projector_pattern.sub(replace_projector, text)
    text = re.sub(
        r"Ga\["
        r"Index\[Lorentz,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Spin,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Spin,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: gamma_matrix(
            S(f"i{match.group(2)}"),
            S(f"i{match.group(3)}"),
            S(f"mu{match.group(1)}"),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"IndexDelta\["
        r"Index\[Colour,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Colour,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: COLOR_FUND.g(
            S(f"c{match.group(1)}"),
            S(f"c{match.group(2)}"),
        ).to_expression().to_canonical_string(),
        text,
    )
    text = re.sub(
        r"T\["
        r"Index\[Gluon,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Colour,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Colour,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: gauge_generator(
            S(f"a{match.group(1)}"),
            S(f"c{match.group(2)}"),
            S(f"c{match.group(3)}"),
        ).to_canonical_string(),
        text,
    )
    text = text.replace("Sqrt[2]", "(2)^(1/2)")
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported FeynRules matter syntax remains after parsing: "
            f"{text}"
        )

    expression = Expression.parse(text)
    for name, replacement in (parameter_substitutions or {}).items():
        expression = expression.replace(S(name), _as_expression(replacement))
    return expression.cancel().expand()


def parse_feynrules_yukawa_rule(
    rule: str,
    *,
    diagonal_yukawa_names: Mapping[str, str] | None = None,
    real_diagonal_yukawas: bool = True,
) -> Expression:
    """Parse the diagonal physical-basis Yukawa export.

    ``diagonal_yukawa_names`` maps FeynRules heads to local component prefixes;
    the Standard Model notebook uses ``yl -> ye``. The reference export uses
    explicit complex conjugation, while the notebook declares diagonal real
    component symbols, so ``real_diagonal_yukawas=True`` identifies each
    conjugate component with the same local symbol.
    """

    names = {"yd": "yd", "yu": "yu", "yl": "yl"}
    names.update(diagonal_yukawa_names or {})
    text = rule

    diagonal_call = (
        r"(yd|yu|yl)\["
        r"Index\[Generation,\s*(\d+)\],\s*"
        r"Index\[Generation,\s*(\d+)\]"
        r"\]"
    )

    def component_symbol(match: re.Match[str]) -> str:
        left = int(match.group(2))
        right = int(match.group(3))
        if left != right:
            return "0"
        return f"{names[match.group(1)]}{left}"

    conjugate_pattern = re.compile(rf"Conjugate\[{diagonal_call}\]")

    def conjugate_component(match: re.Match[str]) -> str:
        component = component_symbol(match)
        if component == "0" or real_diagonal_yukawas:
            return component
        return f"conj({component})"

    text = conjugate_pattern.sub(conjugate_component, text)
    text = re.sub(diagonal_call, component_symbol, text)
    text = re.sub(
        r"ProjM\["
        r"Index\[Spin,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Spin,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: chiral_projector_left(
            S(f"i{match.group(1)}"),
            S(f"i{match.group(2)}"),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjP\["
        r"Index\[Spin,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Spin,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: chiral_projector_right(
            S(f"i{match.group(1)}"),
            S(f"i{match.group(2)}"),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"IndexDelta\["
        r"Index\[Colour,\s*Ext\[(\d+)\]\],\s*"
        r"Index\[Colour,\s*Ext\[(\d+)\]\]"
        r"\]",
        lambda match: COLOR_FUND.g(
            S(f"c{match.group(1)}"),
            S(f"c{match.group(2)}"),
        ).to_expression().to_canonical_string(),
        text,
    )
    text = text.replace("Sqrt[2]", "(2)^(1/2)")
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported FeynRules Yukawa syntax remains after parsing: "
            f"{text}"
        )
    return Expression.parse(text).cancel().expand()


def _walk_expression(expression: Expression):
    yield expression
    if expression.get_type() in (
        AtomType.Add,
        AtomType.Mul,
        AtomType.Pow,
        AtomType.Fn,
    ):
        for child in expression:
            yield from _walk_expression(child)


def _representation_labels(
    expression: Expression,
    *,
    head: str,
    dimension: int,
) -> tuple[Expression, ...]:
    labels: dict[str, Expression] = {}
    for node in _walk_expression(expression):
        if node.get_type() != AtomType.Fn or node.get_name() != head:
            continue
        args = tuple(node)
        if len(args) != 2 or args[0].to_canonical_string() != str(dimension):
            continue
        label = args[1]
        labels.setdefault(label.to_canonical_string(), label)
    return tuple(labels.values())


def _factors(expression: Expression) -> tuple[Expression, ...]:
    if expression.get_type() == AtomType.Mul:
        return tuple(expression)
    return (expression,)


def _terms(expression: Expression) -> tuple[Expression, ...]:
    if expression.get_type() == AtomType.Add:
        return tuple(expression)
    return (expression,)


def _slot_label(slot: Expression, *, representation: str) -> Expression | None:
    if slot.get_type() != AtomType.Fn or slot.get_name() != representation:
        return None
    args = tuple(slot)
    if len(args) != 2:
        return None
    return args[1]


def _spinor_matrix_edge(
    factor: Expression,
) -> tuple[str, Expression, Expression, Expression | None] | None:
    if factor.get_type() != AtomType.Fn:
        return None
    name = factor.get_name()
    args = tuple(factor)

    if name == "spenso::gamma" and len(args) == 3:
        left = _slot_label(args[0], representation="spenso::bis")
        right = _slot_label(args[1], representation="spenso::bis")
        lorentz = _slot_label(args[2], representation="spenso::mink")
        if left is not None and right is not None and lorentz is not None:
            return ("gamma", left, right, lorentz)

    if name == "spenso::gamma5" and len(args) == 2:
        left = _slot_label(args[0], representation="spenso::bis")
        right = _slot_label(args[1], representation="spenso::bis")
        if left is not None and right is not None:
            return ("gamma5", left, right, None)

    if name == "spenso::g" and len(args) == 2:
        left = _slot_label(args[0], representation="spenso::bis")
        right = _slot_label(args[1], representation="spenso::bis")
        if left is not None and right is not None:
            return ("identity", left, right, None)
    return None


def _reduce_one_fermion_current(
    term: Expression,
    *,
    left: Expression,
    right: Expression,
) -> Expression:
    matrix_edges: list[
        tuple[str, Expression, Expression, Expression | None]
    ] = []
    scalar_factors: list[Expression] = []
    for factor in _factors(term):
        edge = _spinor_matrix_edge(factor)
        if edge is None:
            scalar_factors.append(factor)
        else:
            matrix_edges.append(edge)

    gamma_edges = [edge for edge in matrix_edges if edge[0] == "gamma"]
    if len(gamma_edges) != 1:
        raise ValueError(
            "Matter comparison expects exactly one gamma matrix per term; "
            f"found {len(gamma_edges)} in {term.to_canonical_string()}"
        )

    adjacency: dict[str, list[tuple[int, Expression]]] = {}
    for position, (_kind, edge_left, edge_right, _lorentz) in enumerate(
        matrix_edges
    ):
        adjacency.setdefault(edge_left.to_canonical_string(), []).append(
            (position, edge_right)
        )
        adjacency.setdefault(edge_right.to_canonical_string(), []).append(
            (position, edge_left)
        )

    current = left
    used: set[int] = set()
    ordered_kinds: list[str] = []
    lorentz = gamma_edges[0][3]
    while current != right:
        options = [
            (position, other)
            for position, other in adjacency.get(
                current.to_canonical_string(),
                (),
            )
            if position not in used
        ]
        if len(options) != 1:
            raise ValueError(
                "Fermion-current spinor chain is not a unique path from "
                f"{left} to {right}: {term.to_canonical_string()}"
            )
        position, other = options[0]
        used.add(position)
        kind = matrix_edges[position][0]
        if kind != "identity":
            ordered_kinds.append(kind)
        current = other

    if len(used) != len(matrix_edges):
        raise ValueError(
            "Disconnected spinor tensors in matter rule: "
            f"{term.to_canonical_string()}"
        )

    if ordered_kinds.count("gamma") != 1:
        raise ValueError(f"Unsupported spinor chain: {ordered_kinds}")
    gamma_position = ordered_kinds.index("gamma")
    gamma5_count = ordered_kinds.count("gamma5")
    sign = -1 if gamma_position % 2 else 1
    current_head = S("DiracAxial" if gamma5_count % 2 else "DiracVector")
    current = current_head(lorentz, left, right)

    scalar = Expression.num(sign)
    for factor in scalar_factors:
        scalar *= factor
    return scalar * current


def reduce_fermion_currents(
    expression: Expression,
    *,
    left: Expression = S("i1"),
    right: Expression = S("i2"),
) -> Expression:
    """Reduce one-gamma spinor chains to vector/axial current basis.

    This applies the Clifford identities needed by the FeynRules matter
    export, including ``gamma5 gamma = -gamma gamma5`` and
    ``gamma5 gamma gamma5 = -gamma``. It removes representation artifacts
    from expanded chiral projectors while preserving every scalar and color
    factor.
    """

    total = Expression.num(0)
    for term in _terms(expression.expand()):
        total += _reduce_one_fermion_current(term, left=left, right=right)
    return total.cancel().expand()


def _expand_compact_projectors(expression: Expression) -> Expression:
    left = S("comparison_projector_left_")
    right = S("comparison_projector_right_")
    result = expression.replace(
        S("PL")(left, right),
        chiral_projector_left(left, right),
    )
    result = result.replace(
        S("PR")(left, right),
        chiral_projector_right(left, right),
    )
    return result.expand()


def _reduce_one_yukawa_bilinear(
    term: Expression,
    *,
    left: Expression,
    right: Expression,
) -> Expression:
    matrix_edges: list[
        tuple[str, Expression, Expression, Expression | None]
    ] = []
    scalar_factors: list[Expression] = []
    for factor in _factors(term):
        edge = _spinor_matrix_edge(factor)
        if edge is None:
            scalar_factors.append(factor)
        else:
            matrix_edges.append(edge)

    if any(edge[0] == "gamma" for edge in matrix_edges):
        raise ValueError(
            "Yukawa comparison does not accept Lorentz gamma matrices: "
            f"{term.to_canonical_string()}"
        )

    adjacency: dict[str, list[tuple[int, Expression]]] = {}
    for position, (_kind, edge_left, edge_right, _lorentz) in enumerate(
        matrix_edges
    ):
        adjacency.setdefault(edge_left.to_canonical_string(), []).append(
            (position, edge_right)
        )
        adjacency.setdefault(edge_right.to_canonical_string(), []).append(
            (position, edge_left)
        )

    current = left
    used: set[int] = set()
    gamma5_count = 0
    while current != right:
        options = [
            (position, other)
            for position, other in adjacency.get(
                current.to_canonical_string(),
                (),
            )
            if position not in used
        ]
        if len(options) != 1:
            raise ValueError(
                "Yukawa spinor chain is not a unique path from "
                f"{left} to {right}: {term.to_canonical_string()}"
            )
        position, other = options[0]
        used.add(position)
        gamma5_count += matrix_edges[position][0] == "gamma5"
        current = other

    if len(used) != len(matrix_edges):
        raise ValueError(
            "Disconnected spinor tensors in Yukawa rule: "
            f"{term.to_canonical_string()}"
        )

    bilinear = S(
        "DiracPseudoscalar" if gamma5_count % 2 else "DiracScalar"
    )(left, right)
    scalar = Expression.num(1)
    for factor in scalar_factors:
        scalar *= factor
    return scalar * bilinear


def reduce_yukawa_bilinears(
    expression: Expression,
    *,
    left: Expression = S("i1"),
    right: Expression = S("i2"),
) -> Expression:
    """Reduce scalar/projector chains to scalar and pseudoscalar bilinears."""

    expression = _expand_compact_projectors(expression)
    total = Expression.num(0)
    for term in _terms(expression):
        total += _reduce_one_yukawa_bilinear(
            term,
            left=left,
            right=right,
        )
    return total.cancel().expand()


def canonicalize_gauge_rule(expression: Expression) -> Expression:
    """Canonicalize a parsed or generated gauge vertex for exact comparison."""

    expression = expression.cancel().expand()
    lorentz_indices = _representation_labels(
        expression,
        head="spenso::mink",
        dimension=4,
    )
    adjoint_indices = _representation_labels(
        expression,
        head="spenso::coad",
        dimension=8,
    )
    canonical = canonize_structure_constant_products(
        expression,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
    )
    return canonical.cancel().expand()


def canonicalize_matter_rule(expression: Expression) -> Expression:
    """Canonicalize one fermion-gauge vertex in vector/axial current basis."""

    expression = reduce_fermion_currents(expression.cancel().expand())
    lorentz_indices = _representation_labels(
        expression,
        head="spenso::mink",
        dimension=4,
    )
    adjoint_indices = _representation_labels(
        expression,
        head="spenso::coad",
        dimension=8,
    )
    color_indices = _representation_labels(
        expression,
        head="spenso::cof",
        dimension=3,
    )
    canonical = canonize_structure_constant_products(
        expression,
        lorentz_indices=lorentz_indices,
        adjoint_indices=adjoint_indices,
        color_fund_indices=color_indices,
    )
    return canonical.cancel().expand()


def canonicalize_yukawa_rule(expression: Expression) -> Expression:
    """Canonicalize one physical Yukawa vertex."""

    expression = reduce_yukawa_bilinears(expression.cancel().expand())
    color_indices = _representation_labels(
        expression,
        head="spenso::cof",
        dimension=3,
    )
    canonical = canonize_structure_constant_products(
        expression,
        color_fund_indices=color_indices,
    )
    return canonical.cancel().expand()


def _apply_expression_substitutions(
    expression: Expression,
    substitutions: Mapping[object, object] | None,
) -> Expression:
    result = expression
    for old, new in (substitutions or {}).items():
        old_expression = S(old) if isinstance(old, str) else _as_expression(old)
        result = result.replace(old_expression, _as_expression(new))
    return result.cancel().expand()


def _apply_momentum_conservation(
    expression: Expression,
    *,
    arity: int,
    eliminate_leg: int = 1,
) -> Expression:
    lorentz = S("comparison_momentum_lorentz_")
    replacement = Expression.num(0)
    for leg in range(1, arity + 1):
        if leg == eliminate_leg:
            continue
        replacement -= pcomp(S(f"q{leg}"), lorentz)
    return expression.replace(
        pcomp(S(f"q{eliminate_leg}"), lorentz),
        replacement,
    ).expand()


def _contains_tensor_function(expression: Expression) -> bool:
    tensor_heads = {
        "spenso::f",
        "spenso::g",
        "spenso::gamma",
        "spenso::gamma5",
        "spenso::t",
        "python::pcomp",
        "python::DiracAxial",
        "python::DiracVector",
        "python::DiracScalar",
        "python::DiracPseudoscalar",
    }
    return any(
        node.get_type() == AtomType.Fn and node.get_name() in tensor_heads
        for node in _walk_expression(expression)
    )


def _scalar_to_sympy(expression: Expression):
    text = expression.to_canonical_string()
    text = re.sub(
        r"(?:python|spenso)::(?:\{[^}]*\}::)?",
        "",
        text,
    )
    text = re.sub(r"\{[^}]*\}::", "", text)
    text = re.sub(r"(?<=\d)𝑖", "*I", text)
    text = text.replace("𝑖", "I").replace("^", "**")
    return sympy.sympify(text)


def _tensor_coefficient_groups(
    expression: Expression,
) -> dict[str, tuple[Expression, object]]:
    grouped: dict[str, tuple[Expression, object]] = {}
    for term in _terms(expression.expand()):
        tensor = Expression.num(1)
        scalar = Expression.num(1)
        for factor in _factors(term):
            if _contains_tensor_function(factor):
                tensor *= factor
            else:
                scalar *= factor
        key = tensor.to_canonical_string()
        scalar_sympy = _scalar_to_sympy(scalar)
        if key in grouped:
            old_tensor, old_scalar = grouped[key]
            grouped[key] = (old_tensor, old_scalar + scalar_sympy)
        else:
            grouped[key] = (tensor, scalar_sympy)
    return grouped


def _exact_difference(
    left: Expression,
    right: Expression,
    *,
    momentum_arity: int | None = None,
) -> Expression:
    """Return zero when tensor coefficients are exactly symbolically equal."""

    difference = (left - right).expand()
    if momentum_arity is not None:
        difference = _apply_momentum_conservation(
            difference,
            arity=momentum_arity,
        )
    groups = _tensor_coefficient_groups(difference)
    if all(sympy.simplify(coefficient) == 0 for _tensor, coefficient in groups.values()):
        return Expression.num(0)
    return difference.cancel().expand()


def _normalized_feynpy_signature(
    names: Iterable[str],
    aliases: Mapping[str, str],
) -> tuple[str, ...]:
    return tuple(sorted(aliases.get(name, name) for name in names))


def _field_spin(field) -> object:
    base = field.field if hasattr(field, "field") else field
    return getattr(base, "spin", None)


def _field_kind(field) -> str | None:
    base = field.field if hasattr(field, "field") else field
    return getattr(base, "kind", None)


def compare_feynrules_gauge_vertices(
    lagrangian,
    references: Sequence[FeynRulesVertex],
    *,
    field_map: Mapping[str, object],
    parameter_substitutions: Mapping[str, object] | None = None,
    feynpy_name_aliases: Mapping[str, str] | None = None,
) -> VertexComparisonReport:
    """Compare a FeynPy gauge Lagrangian against aligned FeynRules vertices.

    FeynPy rules are extracted in the exact external-leg order stored in each
    reference vertex. Signature coverage is checked as a multiset, while the
    rule comparison preserves leg order so momentum and open-index labels are
    meaningful.
    """

    aliases = dict(feynpy_name_aliases or {})
    reference_by_signature: dict[tuple[str, ...], FeynRulesVertex] = {}
    for reference in references:
        if reference.signature in reference_by_signature:
            raise ValueError(
                f"Duplicate FeynRules signature: {reference.signature}"
            )
        reference_by_signature[reference.signature] = reference

    reference_names = {
        field
        for reference in references
        for field in reference.fields
    }
    reference_arities = {len(reference.fields) for reference in references}
    feynpy_signatures = {
        normalized
        for signature in lagrangian.vertex_signatures()
        if signature.arity in reference_arities
        for normalized in (
            _normalized_feynpy_signature(signature.names, aliases),
        )
        if all(name in reference_names for name in normalized)
    }
    reference_signatures = set(reference_by_signature)
    feynrules_only = tuple(sorted(reference_signatures - feynpy_signatures))
    feynpy_only = tuple(sorted(feynpy_signatures - reference_signatures))

    rows: list[VertexComparison] = []
    for reference in references:
        if reference.signature not in feynpy_signatures:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FEYNPY",
                    detail="No FeynPy vertex with this field multiset",
                )
            )
            continue

        missing_fields = [
            name for name in reference.fields if name not in field_map
        ]
        if missing_fields:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FIELD_MAP",
                    detail=f"Missing field mapping: {missing_fields}",
                )
            )
            continue

        try:
            feynrules_rule = parse_feynrules_gauge_rule(
                reference.rule,
                parameter_substitutions=parameter_substitutions,
            )
            feynpy_rule = lagrangian.feynman_rule(
                *(field_map[name] for name in reference.fields),
                simplify=True,
                include_delta=False,
            )
            feynrules_rule = canonicalize_gauge_rule(feynrules_rule)
            feynpy_rule = canonicalize_gauge_rule(feynpy_rule)
            difference = (feynpy_rule - feynrules_rule).cancel().expand()
            status = (
                "MATCH"
                if difference.to_canonical_string() == "0"
                else "MISMATCH"
            )
            rows.append(
                VertexComparison(
                    reference=reference,
                    status=status,
                    feynpy_rule=feynpy_rule,
                    feynrules_rule=feynrules_rule,
                    difference=difference,
                    detail=(
                        "Canonical symbolic difference is zero"
                        if status == "MATCH"
                        else "Canonical symbolic difference is non-zero"
                    ),
                )
            )
        except Exception as error:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="ERROR",
                    detail=f"{type(error).__name__}: {error}",
                )
            )

    return VertexComparisonReport(
        rows=tuple(rows),
        feynrules_only=feynrules_only,
        feynpy_only=feynpy_only,
    )


def compare_feynrules_matter_vertices(
    lagrangian,
    references: Sequence[FeynRulesVertex],
    *,
    field_map: Mapping[str, object],
    parameter_substitutions: Mapping[str, object] | None = None,
    feynpy_substitutions: Mapping[object, object] | None = None,
    feynpy_name_aliases: Mapping[str, str] | None = None,
) -> VertexComparisonReport:
    """Compare flavor-expanded FeynPy matter vertices with FeynRules.

    Candidate FeynPy signatures are evaluated after ``feynpy_substitutions``.
    This is needed when a reference export fixes a flavor convention, such as
    an identity CKM matrix, while the source model keeps a symbolic rotation.
    Signatures whose aligned rule becomes exactly zero are excluded from the
    FeynPy coverage set.
    """

    aliases = dict(feynpy_name_aliases or {})
    reference_by_signature: dict[tuple[str, ...], FeynRulesVertex] = {}
    for reference in references:
        if reference.signature in reference_by_signature:
            raise ValueError(
                f"Duplicate FeynRules signature: {reference.signature}"
            )
        reference_by_signature[reference.signature] = reference

    reference_names = {
        field
        for reference in references
        for field in reference.fields
    }
    reference_arities = {len(reference.fields) for reference in references}
    feynpy_rules_by_signature: dict[tuple[str, ...], Expression] = {}
    for signature in lagrangian.vertex_signatures(flavor_expand=True):
        if signature.arity not in reference_arities:
            continue
        normalized = _normalized_feynpy_signature(signature.names, aliases)
        if not all(name in reference_names for name in normalized):
            continue
        if sum(
            _field_spin(field_map[name]) == 1 / 2
            for name in normalized
            if name in field_map
        ) != 2:
            continue
        rule = lagrangian.feynman_rule(
            *signature.fields,
            simplify=True,
            include_delta=False,
            flavor_expand=True,
        )
        rule = _apply_expression_substitutions(rule, feynpy_substitutions)
        if rule.to_canonical_string() == "0":
            continue
        if normalized in feynpy_rules_by_signature:
            raise ValueError(f"Duplicate FeynPy signature: {normalized}")
        feynpy_rules_by_signature[normalized] = rule

    reference_signatures = set(reference_by_signature)
    feynpy_signatures = set(feynpy_rules_by_signature)
    feynrules_only = tuple(sorted(reference_signatures - feynpy_signatures))
    feynpy_only = tuple(sorted(feynpy_signatures - reference_signatures))

    rows: list[VertexComparison] = []
    for reference in references:
        if reference.signature not in feynpy_rules_by_signature:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FEYNPY",
                    detail="No non-zero FeynPy vertex with this field multiset",
                )
            )
            continue

        missing_fields = [
            name for name in reference.fields if name not in field_map
        ]
        if missing_fields:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FIELD_MAP",
                    detail=f"Missing field mapping: {missing_fields}",
                )
            )
            continue

        try:
            feynrules_rule = parse_feynrules_matter_rule(
                reference.rule,
                parameter_substitutions=parameter_substitutions,
            )
            feynpy_rule = lagrangian.feynman_rule(
                *(field_map[name] for name in reference.fields),
                simplify=True,
                include_delta=False,
                flavor_expand=True,
            )
            feynpy_rule = _apply_expression_substitutions(
                feynpy_rule,
                feynpy_substitutions,
            )
            feynrules_rule = canonicalize_matter_rule(feynrules_rule)
            feynpy_rule = canonicalize_matter_rule(feynpy_rule)
            difference = (feynpy_rule - feynrules_rule).cancel().expand()
            status = (
                "MATCH"
                if difference.to_canonical_string() == "0"
                else "MISMATCH"
            )
            rows.append(
                VertexComparison(
                    reference=reference,
                    status=status,
                    feynpy_rule=feynpy_rule,
                    feynrules_rule=feynrules_rule,
                    difference=difference,
                    detail=(
                        "Canonical symbolic difference is zero"
                        if status == "MATCH"
                        else "Canonical symbolic difference is non-zero"
                    ),
                )
            )
        except Exception as error:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="ERROR",
                    detail=f"{type(error).__name__}: {error}",
                )
            )

    return VertexComparisonReport(
        rows=tuple(rows),
        feynrules_only=feynrules_only,
        feynpy_only=feynpy_only,
    )


def compare_feynrules_bosonic_vertices(
    lagrangian,
    references: Sequence[FeynRulesVertex],
    *,
    field_map: Mapping[str, object],
    parameter_substitutions: Mapping[str, object] | None = None,
    feynpy_substitutions: Mapping[object, object] | None = None,
    feynpy_name_aliases: Mapping[str, str] | None = None,
    minimum_ghost_fields: int = 0,
    minimum_scalar_fields: int = 0,
    use_momentum_conservation: bool = False,
) -> VertexComparisonReport:
    """Compare scalar/vector/ghost tensor vertices with FeynRules."""

    aliases = dict(feynpy_name_aliases or {})
    reference_by_signature = {
        reference.signature: reference
        for reference in references
    }
    if len(reference_by_signature) != len(references):
        raise ValueError("Duplicate FeynRules bosonic signature")

    reference_names = {
        field
        for reference in references
        for field in reference.fields
    }
    reference_arities = {len(reference.fields) for reference in references}
    feynpy_rules_by_signature: dict[tuple[str, ...], Expression] = {}
    for signature in lagrangian.vertex_signatures():
        if signature.arity not in reference_arities:
            continue
        normalized = _normalized_feynpy_signature(signature.names, aliases)
        if not all(name in reference_names for name in normalized):
            continue
        if sum(
            _field_kind(field_map[name]) == "ghost"
            for name in normalized
            if name in field_map
        ) < minimum_ghost_fields:
            continue
        if sum(
            _field_kind(field_map[name]) == "scalar"
            for name in normalized
            if name in field_map
        ) < minimum_scalar_fields:
            continue
        rule = lagrangian.feynman_rule(
            *signature.fields,
            simplify=True,
            include_delta=False,
        )
        rule = _apply_expression_substitutions(rule, feynpy_substitutions)
        if rule.to_canonical_string() == "0":
            continue
        if normalized in feynpy_rules_by_signature:
            raise ValueError(f"Duplicate FeynPy signature: {normalized}")
        feynpy_rules_by_signature[normalized] = rule

    reference_signatures = set(reference_by_signature)
    feynpy_signatures = set(feynpy_rules_by_signature)
    feynrules_only = tuple(sorted(reference_signatures - feynpy_signatures))
    feynpy_only = tuple(sorted(feynpy_signatures - reference_signatures))

    rows: list[VertexComparison] = []
    for reference in references:
        if reference.signature not in feynpy_rules_by_signature:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FEYNPY",
                    detail="No non-zero FeynPy vertex with this field multiset",
                )
            )
            continue
        try:
            feynrules_rule = parse_feynrules_gauge_rule(
                reference.rule,
                parameter_substitutions=parameter_substitutions,
            )
            feynpy_rule = lagrangian.feynman_rule(
                *(field_map[name] for name in reference.fields),
                simplify=True,
                include_delta=False,
            )
            feynpy_rule = _apply_expression_substitutions(
                feynpy_rule,
                feynpy_substitutions,
            )
            feynrules_rule = canonicalize_gauge_rule(feynrules_rule)
            feynpy_rule = canonicalize_gauge_rule(feynpy_rule)
            difference = _exact_difference(
                feynpy_rule,
                feynrules_rule,
                momentum_arity=(
                    len(reference.fields)
                    if use_momentum_conservation
                    else None
                ),
            )
            status = (
                "MATCH"
                if difference.to_canonical_string() == "0"
                else "MISMATCH"
            )
            rows.append(
                VertexComparison(
                    reference=reference,
                    status=status,
                    feynpy_rule=feynpy_rule,
                    feynrules_rule=feynrules_rule,
                    difference=difference,
                    detail=(
                        "Canonical symbolic difference is zero"
                        if status == "MATCH"
                        else "Canonical symbolic difference is non-zero"
                    ),
                )
            )
        except Exception as error:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="ERROR",
                    detail=f"{type(error).__name__}: {error}",
                )
            )

    return VertexComparisonReport(
        rows=tuple(rows),
        feynrules_only=feynrules_only,
        feynpy_only=feynpy_only,
    )


def compare_feynrules_yukawa_vertices(
    lagrangian,
    references: Sequence[FeynRulesVertex],
    *,
    field_map: Mapping[str, object],
    diagonal_yukawa_names: Mapping[str, str] | None = None,
    feynpy_substitutions: Mapping[object, object] | None = None,
    feynpy_name_aliases: Mapping[str, str] | None = None,
) -> VertexComparisonReport:
    """Compare flavor-expanded physical Yukawa vertices with FeynRules."""

    aliases = dict(feynpy_name_aliases or {})
    reference_by_signature = {
        reference.signature: reference
        for reference in references
    }
    if len(reference_by_signature) != len(references):
        raise ValueError("Duplicate FeynRules Yukawa signature")

    reference_names = {
        field
        for reference in references
        for field in reference.fields
    }
    feynpy_rules_by_signature: dict[tuple[str, ...], Expression] = {}
    for signature in lagrangian.vertex_signatures(flavor_expand=True):
        if signature.arity != 3:
            continue
        normalized = _normalized_feynpy_signature(signature.names, aliases)
        if not all(name in reference_names for name in normalized):
            continue
        if sum(
            _field_spin(field_map[name]) == 1 / 2
            for name in normalized
            if name in field_map
        ) != 2:
            continue
        rule = lagrangian.feynman_rule(
            *signature.fields,
            simplify=True,
            include_delta=False,
            flavor_expand=True,
        )
        rule = _apply_expression_substitutions(rule, feynpy_substitutions)
        if rule.to_canonical_string() == "0":
            continue
        feynpy_rules_by_signature[normalized] = rule

    reference_signatures = set(reference_by_signature)
    feynpy_signatures = set(feynpy_rules_by_signature)
    rows: list[VertexComparison] = []
    for reference in references:
        if reference.signature not in feynpy_rules_by_signature:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="MISSING_FEYNPY",
                    detail="No non-zero FeynPy vertex with this field multiset",
                )
            )
            continue
        try:
            feynrules_rule = parse_feynrules_yukawa_rule(
                reference.rule,
                diagonal_yukawa_names=diagonal_yukawa_names,
            )
            feynpy_rule = lagrangian.feynman_rule(
                *(field_map[name] for name in reference.fields),
                simplify=True,
                include_delta=False,
                flavor_expand=True,
            )
            feynpy_rule = _apply_expression_substitutions(
                feynpy_rule,
                feynpy_substitutions,
            )
            feynrules_rule = canonicalize_yukawa_rule(feynrules_rule)
            feynpy_rule = canonicalize_yukawa_rule(feynpy_rule)
            difference = (feynpy_rule - feynrules_rule).cancel().expand()
            status = (
                "MATCH"
                if difference.to_canonical_string() == "0"
                else "MISMATCH"
            )
            rows.append(
                VertexComparison(
                    reference=reference,
                    status=status,
                    feynpy_rule=feynpy_rule,
                    feynrules_rule=feynrules_rule,
                    difference=difference,
                    detail=(
                        "Canonical symbolic difference is zero"
                        if status == "MATCH"
                        else "Canonical symbolic difference is non-zero"
                    ),
                )
            )
        except Exception as error:
            rows.append(
                VertexComparison(
                    reference=reference,
                    status="ERROR",
                    detail=f"{type(error).__name__}: {error}",
                )
            )

    return VertexComparisonReport(
        rows=tuple(rows),
        feynrules_only=tuple(
            sorted(reference_signatures - feynpy_signatures)
        ),
        feynpy_only=tuple(
            sorted(feynpy_signatures - reference_signatures)
        ),
    )


__all__ = (
    "FeynRulesVertex",
    "VertexComparison",
    "VertexComparisonReport",
    "canonicalize_gauge_rule",
    "canonicalize_matter_rule",
    "canonicalize_yukawa_rule",
    "compare_feynrules_bosonic_vertices",
    "compare_feynrules_gauge_vertices",
    "compare_feynrules_matter_vertices",
    "compare_feynrules_yukawa_vertices",
    "load_feynrules_json",
    "parse_feynrules_gauge_rule",
    "parse_feynrules_matter_rule",
    "parse_feynrules_yukawa_rule",
    "reduce_fermion_currents",
    "reduce_yukawa_bilinears",
)
