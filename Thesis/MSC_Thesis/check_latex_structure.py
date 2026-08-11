#!/usr/bin/env python3
"""Structural sanity checks for the thesis LaTeX sources.

This is not a substitute for compiling, but it catches the errors that are
easiest to introduce while editing prose: unbalanced environments, references
to labels that do not exist, and duplicate labels.

Usage (from the repository root):

    .venv/bin/python Thesis/MSC_Thesis/check_latex_structure.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

THESIS = Path(__file__).resolve().parent

BEGIN = re.compile(r"\\begin\{([A-Za-z*]+)\}")
END = re.compile(r"\\end\{([A-Za-z*]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:Cref|cref|ref|autoref|eqref)\{([^}]+)\}")
COMMENT = re.compile(r"(?<!\\)%.*$")

# Environments whose bodies are verbatim, so \begin/\end inside them is text.
VERBATIM = {"verbatim", "minted", "lstlisting"}


def strip_comments(line: str) -> str:
    return COMMENT.sub("", line)


def check_environments(path: Path, lines: list[str]) -> list[str]:
    problems: list[str] = []
    stack: list[tuple[str, int]] = []
    verbatim_env: str | None = None

    for number, raw in enumerate(lines, start=1):
        if verbatim_env is not None:
            if re.search(rf"\\end\{{{verbatim_env}\}}", raw):
                verbatim_env = None
            continue

        line = strip_comments(raw)

        for match in re.finditer(r"\\(begin|end)\{([A-Za-z*]+)\}", line):
            kind, name = match.group(1), match.group(2)
            if kind == "begin":
                if name in VERBATIM:
                    verbatim_env = name
                    break
                stack.append((name, number))
            else:
                if not stack:
                    problems.append(
                        f"{path.name}:{number}: \\end{{{name}}} with nothing open"
                    )
                    continue
                open_name, open_line = stack.pop()
                if open_name != name:
                    problems.append(
                        f"{path.name}:{number}: \\end{{{name}}} closes "
                        f"\\begin{{{open_name}}} from line {open_line}"
                    )

    for name, number in stack:
        problems.append(f"{path.name}:{number}: \\begin{{{name}}} never closed")
    return problems


def main() -> int:
    sources = sorted(THESIS.glob("*.tex")) + sorted(
        THESIS.glob("chapters/*.tex")
    ) + sorted(THESIS.glob("appendices/*.tex"))

    labels: Counter[str] = Counter()
    refs: dict[str, list[str]] = {}
    problems: list[str] = []

    for path in sources:
        lines = path.read_text(encoding="utf-8").splitlines()
        problems.extend(check_environments(path, lines))
        body = "\n".join(strip_comments(line) for line in lines)
        for label in LABEL.findall(body):
            labels[label] += 1
        for ref in REF.findall(body):
            for single in ref.split(","):
                refs.setdefault(single.strip(), []).append(path.name)

    for label, count in sorted(labels.items()):
        if count > 1:
            problems.append(f"duplicate label: {label} ({count} definitions)")

    for ref, where in sorted(refs.items()):
        if ref not in labels:
            problems.append(
                f"undefined reference: {ref} (used in {', '.join(sorted(set(where)))})"
            )

    print(f"scanned {len(sources)} files, {len(labels)} labels, {len(refs)} refs")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print("  ", problem)
        return 1
    print("no structural problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
