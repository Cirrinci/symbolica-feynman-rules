#!/usr/bin/env python3
"""Generate the SMEFT2 per-row comparison appendix from the report artifact.

The appendix table is derived from
``models/SMEFT2/comparison/artifacts/vertex_comparison_report.json`` rather than
maintained by hand, so the thesis cannot drift out of step with the comparison.

Usage (from the repository root):

    .venv/bin/python Thesis/MSC_Thesis/generate_comparison_appendix.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    REPO_ROOT
    / "models"
    / "SMEFT2"
    / "comparison"
    / "artifacts"
    / "vertex_comparison_report.json"
)
OUTPUT = Path(__file__).resolve().parent / "appendices" / "appendix_smeft2_comparison.tex"

# Short labels keep the table readable; the long status names live in the
# chapter text and in the generated Markdown report.
STATUS_LABEL = {
    "EXACT_MATCH": "direct",
    "MATCH_MODULO_EC_CC_CONVENTION": "global Ec",
    "MATCH_MODULO_CC_PACKAGING": "pinned CC",
    "UNRESOLVED_CC_PACKAGING": "\\textbf{unresolved}",
    "EXACT_MISMATCH": "\\textbf{mismatch}",
    "EXACT_ERROR": "\\textbf{error}",
    "EXACT_NO_LOCAL_SIGNATURE": "\\textbf{missing}",
    "EXACT_UNSUPPORTED": "unsupported",
}

FAMILY_LABEL = {
    "BOSONIC": "bosonic",
    "TWO_FERMION": "2-fermion",
    "FOUR_FERMION": "4-fermion",
}


def latex_escape(text: str) -> str:
    """Escape a signature string for use inside \\texttt{}."""

    return text.replace("|", "\\textbar{}").replace("_", "\\_")


def build() -> str:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rows = payload["reference_vertices"]
    summary = payload["summary"]

    ordered = sorted(rows, key=lambda row: (row["arity"], row["key"]))
    status_counts = Counter(row["exact_symbolic_status"] for row in ordered)

    lines: list[str] = []
    add = lines.append

    add("% ============================================================")
    add("%  appendix_smeft2_comparison.tex")
    add("%  GENERATED FILE -- do not edit by hand.")
    add("%  Regenerate with:")
    add("%    .venv/bin/python Thesis/MSC_Thesis/generate_comparison_appendix.py")
    add("% ============================================================")
    add("")
    add("\\chapter{SMEFT2 Per-Row Comparison Results}")
    add("\\label{app:smeft2-rows}")
    add("")
    add(
        "This appendix lists the exact-symbolic verdict for every FeynRules "
        "reference vertex in the SMEFT2 comparison of "
        "\\Cref{sec:validation-smeft2}. It is generated directly from "
        "\\path{models/SMEFT2/comparison/artifacts/vertex_comparison_report.json}, "
        "so it cannot drift out of step with the comparison itself."
    )
    add("")
    add("The verdict column uses the abbreviations of "
        "\\Cref{tab:smeft2-final-comparison}: \\emph{direct} is canonical "
        "tensor-map equality with no packaging assumption; \\emph{global Ec} "
        "additionally uses the single global evanescent charge-conjugation "
        "convention of \\Cref{sec:validation-ec}; and \\emph{pinned CC} "
        "additionally uses one explicitly tabulated row-specific transform.")
    add("")
    add(
        f"The {len(ordered)} rows break down as "
        + ", ".join(
            f"{count} {STATUS_LABEL.get(status, status)}"
            for status, count in sorted(
                status_counts.items(), key=lambda item: -item[1]
            )
        )
        + "."
    )
    add("")
    # A longtable must not be wrapped in ``center``: it centres itself and
    # needs to remain at the outer level to break across pages correctly.
    add("\\begingroup")
    add("\\footnotesize")
    add("\\begin{longtable}{r l l l}")
    add("\\toprule")
    add("Arity & External fields & Family & Verdict \\\\")
    add("\\midrule")
    add("\\endfirsthead")
    add("\\toprule")
    add("Arity & External fields & Family & Verdict \\\\")
    add("\\midrule")
    add("\\endhead")
    add("\\midrule")
    add("\\multicolumn{4}{r}{\\emph{continued on next page}} \\\\")
    add("\\endfoot")
    add("\\bottomrule")
    add("\\endlastfoot")

    for row in ordered:
        signature = latex_escape(row["key"])
        family = FAMILY_LABEL.get(
            row["exact_symbolic_family"], row["exact_symbolic_family"]
        )
        verdict = STATUS_LABEL.get(
            row["exact_symbolic_status"], row["exact_symbolic_status"]
        )
        add(
            f"{row['arity']} & \\texttt{{{signature}}} & {family} & {verdict} \\\\"
        )

    add("\\end{longtable}")
    add("\\endgroup")
    add("")
    add(
        "\\noindent Consistency check against "
        "\\Cref{tab:smeft2-final-comparison}: "
        f"{summary['exact_symbolic_direct_match_vertices']} direct, "
        f"{summary['ec_cc_convention_match_vertices']} global Ec, "
        f"{summary['cc_packaging_pinned_match_vertices']} pinned CC, "
        f"{summary['cc_packaging_unresolved_vertices']} unresolved, out of "
        f"{summary['exact_symbolic_supported_vertices']} supported rows."
    )
    add("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
