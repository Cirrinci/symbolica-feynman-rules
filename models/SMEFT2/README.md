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

`LF2DH2` still needs care around compact Higgs covariant derivatives. The
triplet `alphaRHq3pp` / `alphaRHl3pp` terms now use the compact form on both
Higgs factors, but only with their explicit weak labels kept as
`DC(Phi.bar(w1), mu)` and `DC(Phi(w2), mu)`. Leaving the `Phi` derivative
unlabeled reintroduces the spurious two-fermion `B`-vertex head mismatch in the
comparison.

## Comparison

The reproducible comparison entry point is:

```bash
.venv/bin/python models/SMEFT2/comparison.py
```

It regenerates:

- `COMPARISON.md` — human-readable summary.
- `vertex_comparison_report.json` — per-signature comparison rows.
- `feynpy_vertices.json` — local FeynPy 3-6 point vertex rules.

The maintained method report is
`COMPARISON_METHOD.md`: it explains the comparison layers, the exact
canonicalization identities, and the charge-conjugation packaging checks.

The comparison checks signature coverage and coefficient-head content after
normalizing field names to the FeynRules convention. It also attempts strict
exact canonical tensor-rule equality for all 184 reference rows. The current
strict result is 184 exact matches out of 184 supported reference rows:
32 bosonic rows, 131 two-fermion rows, and 21 four-fermion rows.

The two literal Weinberg reference signatures have no same-field FeynPy
signature, but their operator content is now proven exactly against FEYNPy's
mixed `lLbar,lL` charge-conjugation packaging by comparing the antisymmetrized
partner-rule pair. The `alphaEc*` four-fermion charge-conjugation rows are
also proven at exact tensor-map level by controlled packaging transforms that
must reduce to identical canonical coefficient-sector maps. Raw head-count
deltas remain visible in `COMPARISON.md` and `vertex_comparison_report.json`,
but they are diagnostics for expansion form, not acceptance criteria.

The strict gate now passes:

```bash
.venv/bin/python models/SMEFT2/comparison.py --check
```

## Check

```bash
.venv/bin/python -m pytest models/SMEFT2/tests/test_smeft2.py
```
