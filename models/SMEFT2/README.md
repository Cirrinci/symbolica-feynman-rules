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
- `LX2H2`
- `LH4D2` including `alphaRHDpp`
- `LH6`
- `LF2XH`
- `LF2DH2`
- `LF2H3`
- `L4q`
- `L4l`
- `L4lq`
- `LEvF2XH`
- `LEvF2HD2`
- `LEv4q`
- `LEv4l`
- `LEv4lq`
- `LEvCCLLLL`
- `LEvCCRRRR`
- `LEvCCLRRL`
- `LEvCCRRLL`

## What Is Still Omitted

There are two different reasons for omissions.

Blocked by the current declarative lowering, because they need true
`D_mu F^{mu nu}` operators or genuine nested covariant derivatives:

- `LX2D2`
- `LH2XD2`
- `LH2D4`
- `LF2D3`
- `LF2HD2`
- `LF2XD`
- `LEvF2XD`

`LEvF2HD2` is now implemented by expanding every first covariant derivative
term-by-term into `PartialD(...)` and gauge-field pieces before building the
sigma-matrix chain. The same strategy does not fully rescue `LF2HD2`, because
that FR block also contains genuine `DC[DC[...]]` terms, and the current API
does not yet give an exact path for those or for derivatives acting on
`FS(...)`.

## Check

```bash
.venv/bin/python -m pytest models/SMEFT2/tests/test_smeft2.py
```
