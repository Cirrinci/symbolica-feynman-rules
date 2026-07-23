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
including the coefficients needed by the implemented and still-omitted FR
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

## What Is Still Omitted

No Green-basis sectors are currently omitted from the compiled `Ltot`.

The nested derivative API supports the core structures these blocks need,
including `DC(FS(...))`, `PartialD(FS(...))`, `DC(DC(field, ...), ...)`,
`PartialD(DC(...), ...)`, and mixed monomials containing both matter
`DC(...)` factors and raw `FS(...)` factors.

`LEvF2HD2` is now implemented by expanding every first covariant derivative
term-by-term into `PartialD(...)` and gauge-field pieces before building the
sigma-matrix chain. A direct `DC(...)` rewrite compiles for simpler Higgs
operators, but this block still needs local-lowering support for preserving
the sigma-chain fermion pairing through generic covariant branches.

`LF2DH2` also keeps the Higgs-derivative `pp` structures in their expanded
`PartialD(...)` form for now. Rewriting the first `Phibar` derivative in the
triplet `alphaRHq3pp` / `alphaRHl3pp` terms to compact `DC(...)` regresses the
current two-fermion comparison and test baselines.

## Comparison

The reproducible comparison entry point is:

```bash
.venv/bin/python models/SMEFT2/comparison.py
```

It regenerates:

- `COMPARISON.md` — human-readable summary.
- `vertex_comparison_report.json` — per-signature comparison rows.
- `feynpy_vertices.json` — local FeynPy 3-6 point vertex rules.

The comparison currently checks signature coverage and coefficient-head content
after normalizing field names to the FeynRules convention. It does not claim
full tensor-rule equality for SMEFT2 yet; the omitted derivative sectors remain
the dominant gap.

## Check

```bash
.venv/bin/python -m pytest models/SMEFT2/tests/test_smeft2.py
```
