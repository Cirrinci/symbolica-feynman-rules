"""Generate the SMEFT2 FeynPy Feynman rules and report runtime.

This script is intentionally separate from the comparison package. It imports the
SMEFT2 model, computes FeynPy rules directly, writes the complete printout to a
JSON file by default, and prints timing information.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.SMEFT2 import build_smeft_green_bpreserving


MODEL_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = MODEL_DIR / "comparison" / "artifacts"
DEFAULT_OUTPUT = ARTIFACT_DIR / "feynpy_rules_printout.json"


def _rule_payload(signature, rule_text: str) -> dict[str, object]:
    return {
        "key": "|".join(sorted(signature.names)),
        "fields": list(signature.names),
        "arity": signature.arity,
        "term_count": signature.term_count,
        "sectors": list(signature.sectors),
        "rule": rule_text,
    }


def generate_rules(*, min_arity: int | None, max_arity: int | None) -> list[dict[str, object]]:
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()

    vertices = []
    for signature in lagrangian.vertex_signatures():
        if min_arity is not None and signature.arity < min_arity:
            continue
        if max_arity is not None and signature.arity > max_arity:
            continue
        rule = lagrangian.feynman_rule(*signature.fields, simplify=True)
        rule_text = rule.cancel().expand().to_canonical_string()
        vertices.append(_rule_payload(signature, rule_text))

    return sorted(vertices, key=lambda row: (row["arity"], row["key"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SMEFT2 FeynPy Feynman rules and print runtime."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the generated JSON to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--min-arity",
        type=int,
        default=None,
        help="Optional minimum vertex arity.",
    )
    parser.add_argument(
        "--max-arity",
        type=int,
        default=None,
        help="Optional maximum vertex arity.",
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    vertices = generate_rules(min_arity=args.min_arity, max_arity=args.max_arity)
    elapsed = time.perf_counter() - start

    payload = {
        "model": "SMEFT2",
        "vertex_count": len(vertices),
        "elapsed_seconds": elapsed,
        "vertices": vertices,
    }

    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.print:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(vertices)} FeynPy rules to {args.output}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
