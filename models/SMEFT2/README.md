# SMEFT2 FeynPy Status

This folder contains the simple FeynPy translation of
`reference/feynrules/SMEFT_Green_Bpreserving.fr`.

The implementation lives in [SMEFT2.py](/Users/rems/Documents/MScThesis/models/SMEFT2/SMEFT2.py).
The bundled FR sources live in
[reference/feynrules](/Users/rems/Documents/MScThesis/models/SMEFT2/reference/feynrules).
The goal is to stay close to the FR file:

- same field names
- same unbroken gauge basis
- same coefficient names
- direct Lagrangian blocks written inside the model builder

The default bundled model now follows the FeynRules convention that `Ltot` is
the EFT-only Lagrangian. The local SM core is still available separately as
`LSM`, and the old SM-plus-EFT combination is available as `Lfull`.

## What Is Implemented

The file now declares the SM fields, gauge groups, and a broad parameter set,
including the coefficients needed by the FeynRules Green-basis and evanescent
sectors.

These sectors are included in the compiled `Ltot`:

- `L2Higgs`
- `L4Gauge`
- `L4Fermions`
- `L4Higgs`
- `L4Yukawa`
- `LWeinberg`
- `LX3`
- `LX2D2`
- `LX2H2`
- `LH2XD2`
- `LH2D4`
- `LH4D2` including `alphaRHDpp`
- `LH6`
- `LF2D3`
- `LF2HD2`
- `LF2XH`
- `LF2XD`
- `LF2DH2`
- `LF2H3`
- `L4q`
- `L4l`
- `L4lq`
- `LEvF2XH`
- `LEvF2HD2`
- `LEvF2XD`
- `LEv4q`
- `LEv4l`
- `LEv4lq`
- `LEvCCLLLL`
- `LEvCCRRRR`
- `LEvCCLRRL`
- `LEvCCRRLL`

## Remaining Caveats

No Green-basis sectors are currently omitted from the compiled `Ltot`.

The nested derivative API supports the core structures these blocks need,
including `DC(FS(...))`, `PartialD(FS(...))`, `DC(DC(field, ...), ...)`,
`PartialD(DC(...), ...)`, and mixed monomials containing both matter
`DC(...)` factors and raw `FS(...)` factors.

The remaining non-direct comparison rows are charge-conjugation packaging
differences, not unresolved coefficient or tensor mismatches. The strict
same-signature gate intentionally reports those rows as not direct; the
accepted physics gate is `--check --allow-cc-packaging`.

`LEvF2HD2` is now implemented by expanding every first covariant derivative
term-by-term into `PartialD(...)` and gauge-field pieces before building the
sigma-matrix chain. A direct `DC(...)` rewrite compiles for simpler Higgs
operators, but this block still needs local-lowering support for preserving
the sigma-chain fermion pairing through generic covariant branches.

`LF2DH2` centralizes the compact triplet Higgs-derivative convention in local
helpers inside `SMEFT2.py`. The `alphaRHq3pp` / `alphaRHl3pp` terms still use
compact covariant derivatives, but the helpers force explicit weak labels on
the differentiated Higgs fields. This avoids the former fragile source pattern
where writing an unlabeled `DC(Phi, mu)` branch could reintroduce a spurious
two-fermion `B`-vertex head mismatch.

## Comparison

The reproducible comparison entry point is:

```bash
.venv/bin/python -m models.SMEFT2.comparison
```

It regenerates:

- `COMPARISON.md` — human-readable summary.
- `comparison/artifacts/vertex_comparison_report.json` — per-signature comparison rows.
- `comparison/artifacts/feynpy_vertices.json` — local FeynPy 3-6 point vertex rules.

The maintained method report is
`COMPARISON_METHOD.md`: it explains the comparison layers, the exact
canonicalization identities, and the charge-conjugation packaging checks.

The comparison checks signature coverage and coefficient-head content after
normalizing field names to the FeynRules convention. Exact symbolic comparison
is graded honestly across all 184 reference rows:

- `176/184` direct `EXACT_MATCH` (same-signature canonical tensor maps)
- `8/184` `MATCH_MODULO_CC_PACKAGING` (two Weinberg rows plus six pinned
  `alphaEc*` four-fermion partner rows)
- `0/184` `UNRESOLVED_CC_PACKAGING`

Operator-content coverage remains `184/184` including the charge-conjugation
overlay. Raw head-count deltas remain visible diagnostics for expansion form.

The default gate requires direct exact matches only:

```bash
.venv/bin/python -m models.SMEFT2.comparison --check
```

Pinned CC packaging can be accepted explicitly with `--allow-cc-packaging`.
Unresolved CC packaging rows never pass `--check`.

## Final Acceptance

The thesis acceptance gate is the explicit charge-conjugation packaging gate:

```bash
.venv/bin/python -m models.SMEFT2.comparison --check --allow-cc-packaging
```

Expected terminal summary:

```text
SMEFT2 comparison: 184/184 reference vertices match at operator-content level (176 direct + 8 via charge-conjugation packaging); exact symbolic split=direct 176/184, modulo pinned CC 8/184, unresolved CC 0/184; raw-head-count matches=100/182; canonical tensor-map matches=32/32 supported vertices (93/93 sectors); Weinberg reconstructed sidecar=2/2 direct, 4/4 coefficient checks, wrong-sign matches=0; EC CC sidecar=12/12 coefficient sectors, wrong-combination matches=0; reference-only=2; feynpy-only=8.
```

This accepts only the pinned Weinberg and `alphaEc*` packaging rows described
in `COMPARISON_METHOD.md`. It still fails on any unresolved packaging row,
operator-content miss, exact symbolic inequality, exact symbolic error, or
unexplained FeynPy-only signature.

## Check

```bash
.venv/bin/python -m pytest -q models/SMEFT2/tests/test_smeft2.py
```
