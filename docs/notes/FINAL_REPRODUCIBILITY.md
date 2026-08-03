# Final Reproducibility Checklist

This note records the commands and artifacts needed to reproduce the final
validated state of the project.

## Environment

From the repository root:

```bash
bash setup_env.sh
source .venv/bin/activate
```

## Full Test Suite

```bash
.venv/bin/python -m pytest -q
```

Expected terminal summary:

```text
503 passed
```

The runtime suffix printed by pytest depends on the machine.

## SMEFT2 Acceptance Gate

The accepted SMEFT2 comparison is the explicit charge-conjugation packaging
gate:

```bash
.venv/bin/python -m models.SMEFT2.comparison --check --allow-cc-packaging
```

Latest checked result:

- `184/184` operator-content matches
- `176/184` direct exact symbolic matches
- `8/184` exact matches modulo pinned charge-conjugation packaging
- `0/184` unresolved charge-conjugation packaging rows
- `0` exact symbolic unequal rows
- `0` exact symbolic error rows

Expected terminal summary:

```text
SMEFT2 comparison: 184/184 reference vertices match at operator-content level (176 direct + 8 via charge-conjugation packaging); exact symbolic split=direct 176/184, modulo pinned CC 8/184, unresolved CC 0/184; raw-head-count matches=100/182; canonical tensor-map matches=32/32 supported vertices (93/93 sectors); Weinberg reconstructed sidecar=2/2 direct, 4/4 coefficient checks, wrong-sign matches=0; EC CC sidecar=12/12 coefficient sectors, wrong-combination matches=0; reference-only=2; feynpy-only=8.
```

The direct-only audit command is:

```bash
.venv/bin/python -m models.SMEFT2.comparison --check
```

It is intentionally stricter than the thesis acceptance gate and currently
fails because the eight documented charge-conjugation rows are not direct
same-signature matches.

## Regenerating SMEFT2 Artifacts

```bash
.venv/bin/python -m models.SMEFT2.comparison
```

This regenerates:

- `models/SMEFT2/COMPARISON.md`
- `models/SMEFT2/comparison/artifacts/vertex_comparison_report.json`
- `models/SMEFT2/comparison/artifacts/feynpy_vertices.json`

The maintained method description is
`models/SMEFT2/COMPARISON_METHOD.md`.

## Comparison Scope

The SMEFT2 comparison scope is the bundled EFT-only FeynRules `Ltot` export:

- reference file: `models/SMEFT2/reference/Ltot_SMEFT_FeynRules.json`
- local model: `models/SMEFT2/SMEFT2.py`
- local basis: EFT-only `Ltot`
- SM-plus-EFT local model: `Lfull`, not used for the comparison gate
- reference rows: 184 flavor-expanded three- to six-point vertices
- sectors omitted from local `Ltot`: none
- accepted non-direct rows: two Weinberg rows and six pinned `alphaEc*`
  charge-conjugation packaging rows
- unresolved comparison failures accepted by the thesis gate: none

## Acceptance Statement

For the current thesis scope, SMEFT2 is accepted as complete against the
bundled FeynRules EFT-only `Ltot` reference. The remaining non-direct rows are
documented packaging differences, not unresolved tensor, coefficient or sign
mismatches.
