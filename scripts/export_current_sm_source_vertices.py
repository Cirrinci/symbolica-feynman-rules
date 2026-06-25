#!/usr/bin/env python3
"""Export the current gauge-basis SM source vertices in FeynRules-style text.

This bridges the maintained ``build_standard_model()`` implementation to the
existing comparison scripts that were originally built around older ad hoc
exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from theories import build_standard_model


CURRENT_TO_FEYNRULES_NAMES = {
    "QL": "qL",
    "LL": "lL",
    "lR": "eR",
}


def feynrules_style_name(name: str) -> str:
    if name.endswith(".bar"):
        base = name[:-4]
        return CURRENT_TO_FEYNRULES_NAMES.get(base, base) + ".bar"
    return CURRENT_TO_FEYNRULES_NAMES.get(name, name)


def build_source_vertex_export_text() -> str:
    sm = build_standard_model(
        include_ghosts=False,
        include_gauge_fixing=False,
    )
    source = sm.source_model.lagrangian()

    lines: list[str] = []
    for signature in sorted(
        source.vertex_signatures(),
        key=lambda item: tuple(feynrules_style_name(name) for name in item.names),
    ):
        rule = source.feynman_rule(
            *signature.fields,
            simplify=True,
            include_delta=False,
        )
        lines.append(
            f"Vertex: {tuple(feynrules_style_name(name) for name in signature.names)!r}"
        )
        lines.append(f"Rule: {rule.expand().to_canonical_string()}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the current source-basis Standard Model vertices in the "
            "same lightweight text format used by the FeynRules comparison "
            "helpers."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the text export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.write_text(build_source_vertex_export_text(), encoding="utf-8")
    print(f"Wrote current SM source export: {output}")


if __name__ == "__main__":
    main()
