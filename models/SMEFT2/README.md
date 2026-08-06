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

## Implementation Architecture

The SMEFT2 model is implemented as a direct declarative translation, not as a
separate code generator. The main entry point is:

```python
from models.SMEFT2 import build_smeft_green_bpreserving

bundle = build_smeft_green_bpreserving()
model = bundle.model
```

The returned `SMEFT2Bundle` exposes:

- `model`: the default FeynPy model, with EFT-only `Ltot` as its active
  Lagrangian.
- `fields`: the unbroken-basis fields `B`, `Wi`, `G`, `LL`, `LR`, `QL`, `UR`,
  `DR`, and `Phi`.
- `parameters`: SM parameters and all supported Wilson coefficients. Flavor
  coefficients are declared as indexed complex parameters, so expressions such
  as `alphaKq(f1, f2)` and `conj(alphaWeinberg(f2, f1))` remain visible in
  Feynman rules and in the comparison.
- `gauge_groups`: `U1Y`, `SU2L`, and `SU3C`, including couplings,
  representations, generators, and structure constants.
- `lagrangians`: each named source block plus `LSM`, EFT-only `Ltot`, and
  `Lfull = LSM + Ltot`.

The fields and gauge groups follow the unbroken FeynRules convention:

- `B` is the hypercharge gauge field.
- `Wi` is the weak-adjoint gauge field.
- `G` is the color-adjoint gauge field.
- `LL`, `LR`, `QL`, `UR`, and `DR` are the lepton and quark multiplets. The
  comparison normalizes these local names to the FeynRules print names
  `lL`, `eR`, `qL`, `uR`, and `dR`.
- `Phi` is the Higgs doublet, with `Phi.bar` printed as `Phibar`.

Gauge covariance is handled by the shared FeynPy DSL:

- `DC(field, mu)` inserts the covariant derivative appropriate to the field's
  hypercharge and nonabelian representations.
- `FS(group, mu, nu, adjoint)` inserts an explicit field-strength tensor.
- Nested objects such as `DC(DC(field, mu), nu)`,
  `DC(FS(SU3C, mu, nu, a), rho)`, and `PartialD(FS(U1Y, mu, nu), rho)` are
  lowered by the general compiler.
- Helper tensors from `symbolic.spenso_structures` provide the Lorentz epsilon,
  weak epsilon, charge-conjugation matrix, SU(2)/SU(3) generators, and
  structure constants.

The implementation deliberately keeps source blocks close to the FeynRules
Green-basis blocks. For example, `LX3` contains the cubic field-strength
operators, `LF2XH` contains dipoles with the explicit sigma chain, `LF2DH2`
contains Higgs-current times fermion-current operators, and the `LEv*` blocks
carry the evanescent sectors.

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

## Working with Lagrangians and Feynman Rules

Use `bundle.lagrangians` when you want a specific block, and construct a
temporary `Model` when you want that block to be the active Lagrangian:

```python
from feynpy import Model
from models.SMEFT2 import build_smeft_green_bpreserving

bundle = build_smeft_green_bpreserving()

lf2xh_model = Model(
    name="SMEFT2_LF2XH",
    gauge_groups=tuple(bundle.gauge_groups.values()),
    fields=tuple(bundle.fields.values()),
    parameters=tuple(bundle.parameters.values()),
    lagrangian_decl=bundle.lagrangians["LF2XH"],
)
lf2xh = lf2xh_model.lagrangian()
```

List available vertices before extracting a rule:

```python
for signature in bundle.model.vertex_signatures(arity=3):
    print(signature.names, signature.term_count, signature.sectors)
```

Extract a specific Feynman rule by passing fields in the desired external-leg
order:

```python
fields = bundle.fields
rule = bundle.model.feynman_rule(
    fields["B"],
    fields["QL"],
    fields["QL"].bar,
    simplify=True,
)
print(rule.cancel().expand().to_canonical_string())
```

The same method works on a block model:

```python
rule = lf2xh_model.feynman_rule(
    bundle.fields["B"],
    bundle.fields["Phi"],
    bundle.fields["UR"].bar,
    bundle.fields["QL"],
    simplify=True,
)
```

Useful inspection patterns:

```python
# all four-point rules, keyed by readable field names
rules4 = bundle.model.feynman_rule(arity=4)

# signatures containing a Higgs doublet
for sig in bundle.model.vertex_signatures(contains_fields=(fields["Phi"],)):
    print(sig.names)

# full compiled Lagrangian as one Symbolica expression
expr = bundle.model.to_symbolica()

# expand explicit flavor components when needed
rules_flavor = bundle.model.feynman_rule(arity=3, flavor_expand=True)
```

For bulk export, use:

```bash
.venv/bin/python models/SMEFT2/generate_feynpy_rules.py --min-arity 3 --max-arity 6
```

## Choices and Questions

- Field helpers: `dr(...)`, `ql(...)`, `ll(...)`, `ur(...)`,`lr(...)`

  ```python
  dr(sp=sp1, f=f2, c=c1, bar=True)
  ```

  This returns `DR.bar` with spinor label `sp1`, generation label `f2`, and
  color label `c1`.

- `phitilde(target, source)` is the FeynPy shorthand for the conjugated Higgs
  doublet used in the up-type Yukawa structures:

  ```python
  def phitilde(target, source):
      return weak_eps2(target, source) * Phi.bar(source)
  ```

  It represents `epsilon[target, source] Phibar[source]`. These helpers appear in
  `L4Yukawa`, `LF2XH`, and related Hermitian-conjugate dipole/Yukawa blocks.

- In `LF2HD2`, the up-type `alphaRuHD*` terms show `weak_eps2(...)`
  explicitly because FeynRules hides the same tensor inside `Phitilde`.
  The FeynRules support model defines `Phitilde[i] :> Eps[i,j] Phibar[j]`,
  while FeynPy writes this as
  `weak_eps2(w1, w_ru_hd) * Phi.bar(w_ru_hd)` or its covariant derivatives.

- The `LF2XD` helpers `f2xd_fs_current`,
  `f2xd_derivative_current`, `f2xd_color_derivative_current`, and
  `f2xd_weak_derivative_current` are local source-code builders for repeated
  `Psi^2 X D` patterns. They encode FeynRules structures such as
  `fermionbar gamma_mu fermion D_nu X_{mu nu}` and the antisymmetric current
  `(i/2) (fermionbar gamma_mu D_nu fermion - D_nu fermionbar gamma_mu fermion)
  X_{mu nu}`, with optional SU(3) or SU(2) generators. They are correct but
  make the source less transparent; a planned cleanup is to change this API so
  the `LF2XD` block can be written closer to the mathematical/FeynRules form.

- The helper `weak_t(adjoint, left, right)` includes the factor of two used by
  FeynRules when it writes `2 Ta[aa, ii, jj]`. In Python,
  `weak_gauge_generator(...)` corresponds to `Ta[...]`, while
  `weak_t(...) = 2 * weak_gauge_generator(...)`. note `LH2XD2`
  `alphaRWDH` term has the same SU(2) normalization as the FeynRules line,
  even though the factor `2` is not written at the call site.

- In `LH4D2`, FeynRules writes compact derivatives of products, for example
  `del[del[Phibar[jj] Phi[jj], mu], mu]`. FeynPy currently writes the
  Leibniz-expanded form explicitly:
  `(del del Phibar) Phi + 2 (del Phibar)(del Phi) + Phibar (del del Phi)`.
  A useful next API improvement is to allow
  derivatives to act directly on composite products.

- FeynRules charge conjugation, written as `CC[...]`, is represented explicitly
  in FeynPy with the charge-conjugation tensor. In `LWeinberg` FeynRules writes
  the same-chirality lepton contraction using `CC[LLbar].LL`. FeynPy writes:

  ```python
  ll(sp=sp1, w=w1, f=f1, bar=True)
  * dirac_charge_conjugation(sp1, sp2)
  * ll(sp=sp2, w=w2, f=f2)
  ```

  This is the explicit tensor contraction `LLbar(sp1) C(sp1, sp2) LL(sp2)`.
  The same charge-conjugation packaging also matters in the evanescent
  `LEvCC*` four-fermion sectors.

- The matrix `C` is implemented as a symbolic Spenso tensor named `dirac_C`,
  created by `dirac_charge_conjugation(i, j)`. Its slots are typed as Dirac
  bispinor indices, and the canonicalization layer registers it as
  antisymmetric:

  ```text
  C(i, j) = -C(j, i)
  ```

  We do not choose a concrete numeric gamma-matrix basis for `C` in the SMEFT2
  comparison. The property needed for the Weinberg and `LEvCC*` packaging
  checks is the antisymmetry of `C`, plus the pinned charge-conjugation
  packaging rules described in `COMPARISON_METHOD.md`.

- `_dual_fs(group, mu, nu, rho, sigma, adjoint)` is the explicit dual field
  strength convention:

  ```python
  1/2 * lorentz_levi_civita(mu, nu, rho, sigma) * FS(group, rho, sigma, adjoint)
  ```

  For example `_dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)` is
  `Gtilde^a_{mu nu} = 1/2 epsilon_{mu nu rho sigma} G^a_{rho sigma}`. 

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
