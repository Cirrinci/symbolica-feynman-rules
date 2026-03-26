#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def safe_unparse(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = node.__class__.__name__
    return " ".join(text.split())


def short(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def node_id(i: int) -> str:
    return f"n{i}"


class FunctionFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: List[str] = []
        self.functions: Dict[str, ast.AST] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        name = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.functions[name] = node
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        name = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.functions[name] = node
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


@dataclass
class GNode:
    id: str
    label: str
    kind: str
    lineno: int = 0


@dataclass
class GEdge:
    src: str
    dst: str
    label: str = ""


@dataclass
class Graph:
    name: str
    nodes: Dict[str, GNode] = field(default_factory=dict)
    edges: List[GEdge] = field(default_factory=list)


class RichFlowBuilder:
    def __init__(self, name: str):
        self.name = name
        self.graph = Graph(name=name)
        self.counter = 0
        self.loop_stack: List[Tuple[str, str]] = []

    def new_node(self, label: str, kind: str, lineno: int = 0) -> str:
        nid = node_id(self.counter)
        self.counter += 1
        self.graph.nodes[nid] = GNode(nid, short(label), kind, lineno)
        return nid

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        self.graph.edges.append(GEdge(src, dst, label))

    def build(self, fn_node: ast.AST) -> Graph:
        start = self.new_node(f"START\\n{self.name}", "start", getattr(fn_node, "lineno", 0))
        end = self.new_node(f"END\\n{self.name}", "end", getattr(fn_node, "end_lineno", 0) or 0)
        exits = self._block(getattr(fn_node, "body", []), [start], end)
        for n in exits:
            self.add_edge(n, end)
        return self.graph

    def _block(self, body: List[ast.stmt], incoming: List[str], end_id: str) -> List[str]:
        cur = incoming
        for stmt in body:
            cur = self._stmt(stmt, cur, end_id)
            if not cur:
                break
        return cur

    def _expr_kind(self, expr: ast.AST) -> str:
        if isinstance(expr, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            return "math"
        if isinstance(expr, ast.Call):
            return "call"
        return "stmt"

    def _assignment_label(self, stmt: ast.stmt) -> str:
        if isinstance(stmt, ast.Assign):
            targets = ", ".join(safe_unparse(t) for t in stmt.targets)
            return f"ASSIGN\\n{targets} = {safe_unparse(stmt.value)}"
        if isinstance(stmt, ast.AnnAssign):
            return f"ASSIGN\\n{safe_unparse(stmt.target)} = {safe_unparse(stmt.value) if stmt.value else '...'}"
        if isinstance(stmt, ast.AugAssign):
            return f"UPDATE\\n{safe_unparse(stmt.target)} {stmt.op.__class__.__name__}= {safe_unparse(stmt.value)}"
        return safe_unparse(stmt)

    def _simple(self, stmt: ast.stmt, incoming: List[str]) -> List[str]:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            label = self._assignment_label(stmt)
            kind = "math" if self._contains_math(stmt) else "assign"
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            label = f"CALL\\n{safe_unparse(stmt.value)}"
            kind = "call"
        elif isinstance(stmt, ast.Return):
            label = f"RETURN\\n{safe_unparse(stmt.value) if stmt.value else ''}"
            kind = "return"
        else:
            label = safe_unparse(stmt)
            kind = "stmt"
        nid = self.new_node(label, kind, getattr(stmt, "lineno", 0))
        for src in incoming:
            self.add_edge(src, nid)
        return [nid]

    def _contains_math(self, stmt: ast.AST) -> bool:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
                return True
        return False

    def _stmt(self, stmt: ast.stmt, incoming: List[str], end_id: str) -> List[str]:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Import, ast.ImportFrom, ast.Assert, ast.Delete, ast.Pass)):
            return self._simple(stmt, incoming)
        if isinstance(stmt, ast.Return):
            nid = self._simple(stmt, incoming)[0]
            self.add_edge(nid, end_id, "return")
            return []
        if isinstance(stmt, ast.If):
            test_id = self.new_node(f"IF\\n{safe_unparse(stmt.test)}", "if", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, test_id)
            then_exits = self._block(stmt.body, [test_id], end_id) if stmt.body else [test_id]
            else_exits = self._block(stmt.orelse, [test_id], end_id) if stmt.orelse else [test_id]
            self._label_first_edge_from(test_id, stmt.body, "True")
            self._label_first_edge_from(test_id, stmt.orelse, "False")
            return then_exits + else_exits
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            head = self.new_node(f"FOR\\n{safe_unparse(stmt.target)} in {safe_unparse(stmt.iter)}", "loop", getattr(stmt, "lineno", 0))
            after = self.new_node("AFTER FOR", "join", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, head)
            self.loop_stack.append((head, after))
            body_exits = self._block(stmt.body, [head], end_id) if stmt.body else [head]
            self.loop_stack.pop()
            for b in body_exits:
                self.add_edge(b, head, "loop")
            self.add_edge(head, after, "exit")
            return self._block(stmt.orelse, [after], end_id) if stmt.orelse else [after]
        if isinstance(stmt, ast.While):
            head = self.new_node(f"WHILE\\n{safe_unparse(stmt.test)}", "loop", getattr(stmt, "lineno", 0))
            after = self.new_node("AFTER WHILE", "join", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, head)
            self.loop_stack.append((head, after))
            body_exits = self._block(stmt.body, [head], end_id) if stmt.body else [head]
            self.loop_stack.pop()
            for b in body_exits:
                self.add_edge(b, head, "loop")
            self.add_edge(head, after, "exit")
            return self._block(stmt.orelse, [after], end_id) if stmt.orelse else [after]
        if isinstance(stmt, ast.Break):
            nid = self.new_node("BREAK", "break", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, nid)
            if self.loop_stack:
                self.add_edge(nid, self.loop_stack[-1][1], "break")
            return []
        if isinstance(stmt, ast.Continue):
            nid = self.new_node("CONTINUE", "continue", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, nid)
            if self.loop_stack:
                self.add_edge(nid, self.loop_stack[-1][0], "continue")
            return []
        if isinstance(stmt, ast.Try):
            tid = self.new_node("TRY", "try", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, tid)
            body_exits = self._block(stmt.body, [tid], end_id)
            handler_exits: List[str] = []
            for h in stmt.handlers:
                hid = self.new_node(f"EXCEPT\\n{safe_unparse(h.type) if h.type else 'Exception'}", "except", getattr(h, 'lineno', 0))
                self.add_edge(tid, hid, "except")
                handler_exits.extend(self._block(h.body, [hid], end_id))
            merged = body_exits + handler_exits
            if stmt.finalbody:
                return self._block(stmt.finalbody, merged, end_id)
            return merged
        return self._simple(stmt, incoming)

    def _label_first_edge_from(self, src: str, body: List[ast.stmt], label: str) -> None:
        if not body:
            return
        first_line = getattr(body[0], "lineno", None)
        if first_line is None:
            return
        for nid, node in self.graph.nodes.items():
            if node.lineno == first_line:
                for edge in self.graph.edges:
                    if edge.src == src and edge.dst == nid and not edge.label:
                        edge.label = label
                        return


def resolve_focus(name: str, known: List[str]) -> Optional[str]:
    if name in known:
        return name
    matches = [k for k in known if k.split(".")[-1] == name]
    if len(matches) == 1:
        return matches[0]
    return None


def render_dot(graph: Graph, out_base: Path, fmt: Optional[str]) -> List[Path]:
    colors = {
        "start": "#d9ead3",
        "end": "#f4cccc",
        "if": "#fff2cc",
        "math": "#d9eaf7",
        "assign": "#eaeaea",
        "call": "#eadcf8",
        "return": "#fce5cd",
        "loop": "#ffe599",
        "join": "#efefef",
        "try": "#f3f3f3",
        "except": "#f4cccc",
        "stmt": "white",
        "break": "#f4cccc",
        "continue": "#cfe2f3",
    }
    dot_path = out_base.with_suffix('.dot')
    lines = [
        'digraph G {',
        'rankdir="TB";',
        'graph [overlap=false, splines=true];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica"];',
    ]
    for n in graph.nodes.values():
        label = n.label.replace('"', r'\\"')
        fill = colors.get(n.kind, 'white')
        lines.append(f'{n.id} [label="{label}", fillcolor="{fill}"];')
    for e in graph.edges:
        if e.label:
            lines.append(f'{e.src} -> {e.dst} [label="{e.label}"];')
        else:
            lines.append(f'{e.src} -> {e.dst};')
    lines.append('}')
    dot_path.write_text('\n'.join(lines), encoding='utf-8')
    out = [dot_path]
    if fmt:
        rendered = out_base.with_suffix(f'.{fmt}')
        try:
            subprocess.run(['dot', f'-T{fmt}', str(dot_path), '-o', str(rendered)], check=True)
            out.append(rendered)
        except Exception:
            pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='Generate a richer control-flow diagram with explicit IFs and math-like steps.')
    p.add_argument('source', help='Python source file')
    p.add_argument('--focus', required=True, help='Function or method to analyze')
    p.add_argument('--out-dir', default='diagrammi_math_flow', help='Output directory')
    p.add_argument('--format', choices=['pdf', 'png', 'svg'], default='pdf')
    args = p.parse_args()

    source = Path(args.source).resolve()
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    finder = FunctionFinder()
    finder.visit(tree)
    fn_name = resolve_focus(args.focus, sorted(finder.functions))
    if not fn_name:
        raise SystemExit('Funzione non trovata. Disponibili:\n - ' + '\n - '.join(sorted(finder.functions)))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = RichFlowBuilder(fn_name)
    graph = builder.build(finder.functions[fn_name])
    base = out_dir / f'math_flow_{fn_name.replace('.', '__')}'
    files = render_dot(graph, base, args.format)

    readme = out_dir / 'README_math_flow.txt'
    readme.write_text(
        '\n'.join([
            f'Source: {source}',
            f'Function: {fn_name}',
            '',
            'This diagram emphasizes:',
            '- IF / FOR / WHILE nodes explicitly',
            '- assignments and updates',
            '- math-like expressions in assignments',
            '- returns and branching labels',
            '',
            'Generated files:',
            *[f'- {f}' for f in files],
        ]),
        encoding='utf-8'
    )
    files.append(readme)

    print('Creati:')
    for f in files:
        print(' -', f)


if __name__ == '__main__':
    main()
