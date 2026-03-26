#!/usr/bin/env python3
"""
Generate lightweight diagrams for a Python source file.

What it produces:
1. A function call graph for all top-level functions/methods in the file.
2. One control-flow graph (CFG) per function/method.

Designed for files like `model_symbolica.py`.

Usage:
    python generate_model_diagrams.py model_symbolica.py
    python generate_model_diagrams.py model_symbolica.py --format svg
    python generate_model_diagrams.py model_symbolica.py --only-calls

Requirements:
    - Python 3.10+
    - graphviz Python package + Graphviz system binaries for rendered output
      If unavailable, DOT files are still generated.
"""
from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------- Optional Graphviz support ----------
try:
    from graphviz import Digraph
except Exception:  # pragma: no cover
    Digraph = None


# ---------- Helpers ----------
def safe_unparse(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = node.__class__.__name__
    text = " ".join(text.split())
    return text


def short_label(text: str, max_len: int = 80) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def node_name(prefix: str, index: int) -> str:
    return f"{prefix}_{index}"


# ---------- Function / class discovery ----------
class DefinitionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: List[str] = []
        self.functions: Dict[str, ast.AST] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        fullname = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.functions[fullname] = node
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        fullname = ".".join(self.stack + [node.name]) if self.stack else node.name
        self.functions[fullname] = node
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


class CallCollector(ast.NodeVisitor):
    def __init__(self, known_names: Set[str]) -> None:
        self.known_names = known_names
        self.called: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_name(node.func)
        if name and name in self.known_names:
            self.called.add(name)
        self.generic_visit(node)

    def _resolve_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: List[str] = []
            current: ast.AST = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None


# ---------- CFG representation ----------
@dataclass
class CFGNode:
    id: str
    label: str
    lineno: int = 0


@dataclass
class CFGEdge:
    src: str
    dst: str
    label: str = ""


@dataclass
class FunctionCFG:
    name: str
    nodes: Dict[str, CFGNode] = field(default_factory=dict)
    edges: List[CFGEdge] = field(default_factory=list)
    start_id: str = ""
    end_id: str = ""


class CFGBuilder:
    def __init__(self, func_name: str):
        self.func_name = func_name
        self.counter = 0
        self.cfg = FunctionCFG(name=func_name)
        self.loop_stack: List[Tuple[str, str]] = []  # continue_target, break_target

    def new_node(self, label: str, lineno: int = 0) -> str:
        nid = node_name("n", self.counter)
        self.counter += 1
        self.cfg.nodes[nid] = CFGNode(nid, short_label(label), lineno)
        return nid

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        self.cfg.edges.append(CFGEdge(src, dst, label))

    def build(self, node: ast.AST) -> FunctionCFG:
        start = self.new_node(f"START {self.func_name}")
        end = self.new_node(f"END {self.func_name}")
        self.cfg.start_id = start
        self.cfg.end_id = end

        body = getattr(node, "body", [])
        exits = self._build_block(body, [start], end)
        for last in exits:
            self.add_edge(last, end)
        return self.cfg

    def _build_block(self, statements: Iterable[ast.stmt], incoming: List[str], func_end: str) -> List[str]:
        current = incoming
        for stmt in statements:
            current = self._build_stmt(stmt, current, func_end)
            if not current:
                break
        return current

    def _simple_stmt(self, stmt: ast.stmt, incoming: List[str]) -> List[str]:
        nid = self.new_node(safe_unparse(stmt), getattr(stmt, "lineno", 0))
        for src in incoming:
            self.add_edge(src, nid)
        return [nid]

    def _build_stmt(self, stmt: ast.stmt, incoming: List[str], func_end: str) -> List[str]:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Import, ast.ImportFrom, ast.Assert, ast.Delete, ast.Global, ast.Nonlocal, ast.Raise)):
            return self._simple_stmt(stmt, incoming)

        if isinstance(stmt, ast.Pass):
            return self._simple_stmt(stmt, incoming)

        if isinstance(stmt, ast.Return):
            nid = self.new_node(safe_unparse(stmt), getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, nid)
            self.add_edge(nid, func_end, "return")
            return []

        if isinstance(stmt, ast.If):
            test_id = self.new_node(f"if {safe_unparse(stmt.test)}", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, test_id)

            then_exits = self._build_block(stmt.body, [test_id], func_end) if stmt.body else [test_id]
            else_exits = self._build_block(stmt.orelse, [test_id], func_end) if stmt.orelse else [test_id]

            if stmt.body:
                first_then = self._first_node_of_block(stmt.body)
                if first_then:
                    self._relabel_edge(test_id, first_then, "True")
            if stmt.orelse:
                first_else = self._first_node_of_block(stmt.orelse)
                if first_else:
                    self._relabel_edge(test_id, first_else, "False")
            return then_exits + else_exits

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            head_id = self.new_node(
                f"for {safe_unparse(stmt.target)} in {safe_unparse(stmt.iter)}",
                getattr(stmt, "lineno", 0),
            )
            after_id = self.new_node("after for", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, head_id)

            self.loop_stack.append((head_id, after_id))
            body_exits = self._build_block(stmt.body, [head_id], func_end) if stmt.body else [head_id]
            self.loop_stack.pop()

            for b in body_exits:
                self.add_edge(b, head_id, "loop")
            self.add_edge(head_id, after_id, "exit")

            if stmt.orelse:
                else_exits = self._build_block(stmt.orelse, [after_id], func_end)
                return else_exits
            return [after_id]

        if isinstance(stmt, ast.While):
            head_id = self.new_node(f"while {safe_unparse(stmt.test)}", getattr(stmt, "lineno", 0))
            after_id = self.new_node("after while", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, head_id)

            self.loop_stack.append((head_id, after_id))
            body_exits = self._build_block(stmt.body, [head_id], func_end) if stmt.body else [head_id]
            self.loop_stack.pop()

            for b in body_exits:
                self.add_edge(b, head_id, "loop")
            self.add_edge(head_id, after_id, "exit")

            if stmt.orelse:
                else_exits = self._build_block(stmt.orelse, [after_id], func_end)
                return else_exits
            return [after_id]

        if isinstance(stmt, ast.Break):
            nid = self.new_node("break", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, nid)
            if self.loop_stack:
                _, break_target = self.loop_stack[-1]
                self.add_edge(nid, break_target, "break")
            return []

        if isinstance(stmt, ast.Continue):
            nid = self.new_node("continue", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, nid)
            if self.loop_stack:
                continue_target, _ = self.loop_stack[-1]
                self.add_edge(nid, continue_target, "continue")
            return []

        if isinstance(stmt, ast.Try):
            try_id = self.new_node("try", getattr(stmt, "lineno", 0))
            for src in incoming:
                self.add_edge(src, try_id)
            body_exits = self._build_block(stmt.body, [try_id], func_end)
            handler_exits: List[str] = []
            for h in stmt.handlers:
                h_label = "except" if h.type is None else f"except {safe_unparse(h.type)}"
                h_id = self.new_node(h_label, getattr(h, "lineno", 0))
                self.add_edge(try_id, h_id, "except")
                handler_exits.extend(self._build_block(h.body, [h_id], func_end))
            merged = body_exits + handler_exits
            if stmt.finalbody:
                return self._build_block(stmt.finalbody, merged, func_end)
            if stmt.orelse:
                return self._build_block(stmt.orelse, body_exits, func_end) + handler_exits
            return merged

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # definitions don't participate in runtime control flow inside the enclosing function body
            return self._simple_stmt(stmt, incoming)

        # Fallback for uncommon statements (match, with, etc.)
        return self._simple_stmt(stmt, incoming)

    def _first_node_of_block(self, body: List[ast.stmt]) -> Optional[str]:
        if not body:
            return None
        target_line = getattr(body[0], "lineno", None)
        if target_line is None:
            return None
        for nid, node in self.cfg.nodes.items():
            if node.lineno == target_line:
                return nid
        return None

    def _relabel_edge(self, src: str, dst: str, label: str) -> None:
        for edge in self.cfg.edges:
            if edge.src == src and edge.dst == dst:
                edge.label = label
                return


# ---------- Rendering ----------
def render_cfg(cfg: FunctionCFG, out_dir: Path, fmt: str = "svg") -> Tuple[Path, Optional[Path]]:
    dot_path = out_dir / f"cfg_{cfg.name.replace('.', '__')}.dot"

    lines = ["digraph G {", 'rankdir="TB";', 'node [shape=box];']
    for node in cfg.nodes.values():
        label = node.label.replace('"', r'\"')
        lines.append(f'{node.id} [label="{label}"];')
    for edge in cfg.edges:
        if edge.label:
            elabel = edge.label.replace('"', r'\"')
            lines.append(f'{edge.src} -> {edge.dst} [label="{elabel}"];')
        else:
            lines.append(f"{edge.src} -> {edge.dst};")
    lines.append("}")
    dot_path.write_text("\n".join(lines), encoding="utf-8")

    rendered_path = None
    if Digraph is not None:
        graph = Digraph(comment=cfg.name, format=fmt)
        graph.attr(rankdir="TB")
        graph.attr("node", shape="box")
        for node in cfg.nodes.values():
            graph.node(node.id, label=node.label)
        for edge in cfg.edges:
            graph.edge(edge.src, edge.dst, label=edge.label)
        output_base = out_dir / f"cfg_{cfg.name.replace('.', '__')}"
        try:
            rendered = Path(graph.render(str(output_base), cleanup=True))
            rendered_path = rendered
        except Exception:
            rendered_path = None
    return dot_path, rendered_path


def render_call_graph(edges: Dict[str, Set[str]], out_dir: Path, fmt: str = "svg") -> Tuple[Path, Optional[Path]]:
    dot_path = out_dir / "call_graph.dot"
    all_nodes = sorted(set(edges) | {dst for targets in edges.values() for dst in targets})
    lines = ["digraph G {", 'rankdir="LR";', 'node [shape=box];']
    for name in all_nodes:
        label = name.replace('"', r'\"')
        node_id = name.replace(".", "__")
        lines.append(f'{node_id} [label="{label}"];')
    for src, targets in sorted(edges.items()):
        src_id = src.replace(".", "__")
        for dst in sorted(targets):
            dst_id = dst.replace(".", "__")
            lines.append(f"{src_id} -> {dst_id};")
    lines.append("}")
    dot_path.write_text("\n".join(lines), encoding="utf-8")

    rendered_path = None
    if Digraph is not None:
        graph = Digraph(comment="Call Graph", format=fmt)
        graph.attr(rankdir="LR")
        graph.attr("node", shape="box")
        for name in all_nodes:
            graph.node(name.replace(".", "__"), label=name)
        for src, targets in sorted(edges.items()):
            for dst in sorted(targets):
                graph.edge(src.replace(".", "__"), dst.replace(".", "__"))
        output_base = out_dir / "call_graph"
        try:
            rendered = Path(graph.render(str(output_base), cleanup=True))
            rendered_path = rendered
        except Exception:
            rendered_path = None
    return dot_path, rendered_path


# ---------- Main pipeline ----------
def build_call_graph(tree: ast.Module) -> Tuple[Dict[str, ast.AST], Dict[str, Set[str]]]:
    defs = DefinitionCollector()
    defs.visit(tree)
    names = set(defs.functions.keys())
    short_to_full: Dict[str, str] = {name.split(".")[-1]: name for name in names}

    edges: Dict[str, Set[str]] = {name: set() for name in names}
    for fullname, node in defs.functions.items():
        collector = CallCollector(set(names) | set(short_to_full.keys()))
        collector.visit(node)
        resolved: Set[str] = set()
        for called in collector.called:
            resolved.add(short_to_full.get(called, called))
        edges[fullname] = {c for c in resolved if c in names and c != fullname}
    return defs.functions, edges


def generate_all(source_file: Path, out_dir: Path, fmt: str, only_calls: bool) -> List[str]:
    src = source_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(source_file))

    out_dir.mkdir(parents=True, exist_ok=True)
    functions, call_edges = build_call_graph(tree)

    created: List[str] = []
    call_dot, call_rendered = render_call_graph(call_edges, out_dir, fmt=fmt)
    created.append(str(call_dot))
    if call_rendered:
        created.append(str(call_rendered))

    if not only_calls:
        for name, node in functions.items():
            cfg = CFGBuilder(name).build(node)
            dot_path, rendered_path = render_cfg(cfg, out_dir, fmt=fmt)
            created.append(str(dot_path))
            if rendered_path:
                created.append(str(rendered_path))

    summary_path = out_dir / "README_generated.txt"
    lines = [
        f"Source: {source_file}",
        f"Output directory: {out_dir}",
        "",
        "Generated files:",
        *[f"- {p}" for p in created],
        "",
        "Notes:",
        "- call_graph.* shows which functions/methods call each other inside the file.",
        "- cfg_<name>.* shows a lightweight control-flow graph for one function.",
        "- If only .dot files appear, install graphviz Python package and Graphviz binaries.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    created.append(str(summary_path))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagrams for a Python source file.")
    parser.add_argument("source", help="Path to the Python file, e.g. model_symbolica.py")
    parser.add_argument("--out-dir", default="diagrammi_model_symbolica", help="Output directory")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"], help="Rendered format")
    parser.add_argument("--only-calls", action="store_true", help="Generate only the function call graph")
    args = parser.parse_args()

    source_file = Path(args.source).resolve()
    if not source_file.exists():
        raise SystemExit(f"File non trovato: {source_file}")
    if source_file.suffix != ".py":
        raise SystemExit("Passa un file Python (.py)")

    out_dir = Path(args.out_dir).resolve()
    created = generate_all(source_file, out_dir, args.format, args.only_calls)
    print("Creati:")
    for path in created:
        print(" -", path)


if __name__ == "__main__":
    main()
