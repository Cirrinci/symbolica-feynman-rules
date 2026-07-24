# SMEFT2 Comparison Method Report

Generated for the SMEFT2 FeynRules/FeynPy comparison on 2026-07-24.

## Current Result

The comparison now has two distinct conclusions:

- Operator-content coverage is complete: `184/184` FeynRules reference vertices
  are present in FeynPy after coefficient-head matching and the controlled
  charge-conjugation packaging overlay.
- Strict exact symbolic equality is not complete: `78/184` reference vertices
  currently have matching canonical tensor-monomial maps, while `106/184`
  remain strict mismatches.

The important change from the previous state is that exact fermion comparison
is no longer allowed to pass vacuously. Indexed Wilson coefficients such as
`alphaKq(f1,f2)` and `conj(alphaWeinberg(f2,f1))` are function-valued
coefficients. Symbolica's bare `coefficient(S("alphaKq"))` extraction does not
select those terms. The SMEFT2 comparison now filters terms structurally by
coefficient head and keeps the full indexed coefficient factor in the scalar
coefficient. This preserves flavor order and complex conjugation.

## Input Basis

The reference side is `reference/Ltot_SMEFT_FeynRules.json`, exported from the
EFT-only FeynRules `Ltot`. The local side is the EFT-only FeynPy `Ltot` from
`SMEFT2.py`. `Lfull` still exists for SM-plus-EFT use, but it is not the
comparison basis.

Field names are normalized to the FeynRules convention before matching:

- `LL` / `LL.bar` become `lL` / `lLbar`.
- `LR` / `LR.bar` become `eR` / `eRbar`.
- `QL`, `UR`, `DR` and their bars become `qL`, `uR`, `dR` and their bars.
- `Phi.bar` becomes `Phibar`.
- Gauge fields keep the FeynRules names `B`, `Wi`, and `G`.

## Comparison Layers

The first layer is literal signature coverage. Each vertex is keyed by the
sorted field multiset, for example `B|qL|qLbar`. This gives `182` shared
reference signatures, `2` reference-only signatures, and `8` nonzero FeynPy-only
signatures. The two reference-only rows are the Weinberg same-chirality
signatures; the nonzero FeynPy-only rows are charge-conjugation packaged
partners.

The second layer is operator-content matching by coefficient head. For a shared
signature, the comparison checks whether both sides contain the same Wilson
coefficient heads after dropping algebraically zero local heads through
canonical collection. For nonliteral charge-conjugation packaging, a controlled
overlay pairs a reference row with a FeynPy-only partner if the bar-insensitive
field content and the relevant `alphaEc*` or `alphaWeinberg` head agree. This is
why operator-content coverage is `184/184`.

The third layer is raw head-count diagnostics. This counts textual occurrences
of each coefficient head. It is useful for spotting suspicious over-expansion,
but it is not an equality proof. Known benign raw deltas include dual field
strength expansion, where FeynPy prints the two antisymmetric branches of
`Dual[FS] = 1/2 epsilon.FS` separately, and derivative-order branches that
FeynRules has already merged.

The fourth layer is strict exact symbolic comparison. Both sides are converted
to Symbolica expressions with native Spenso tensor heads, then mapped to
canonical tensor monomials. Equality requires the same monomial keys and exactly
equal scalar coefficients.

## Canonicalization Used

The canonical tensor maps use:

- intrinsic tensor symmetries: symmetric metrics, antisymmetric structure
  constants, weak epsilon, Lorentz epsilon, color epsilon, and `dirac_C`;
- dummy-index relabeling within representation groups;
- deterministic ordering of commuting tensor factors;
- exact scalar coefficient collection.

The comparison does not use equations of motion, integration by parts, momentum
conservation, Schouten identities, Fierz identities, or broad gamma-matrix
reductions beyond the explicitly represented tensor products. When a match is
reported at strict exact level, it is a direct canonical tensor-map equality
under the listed identities.

## Bosonic Sector

The bosonic sector is fully proven:

- rows: `32`
- strict exact matches: `32`
- strict mismatches: `0`
- canonical gauge/Higgs coefficient sectors: `93/93` matching

FeynRules tensor syntax such as `ME`, `FV`, `SP`, `Eps`, `fsu3`, and `fsu2` is
parsed into native metric, momentum, Levi-Civita, and structure-constant
tensors. The main work here is canonicalizing dummy Lorentz/adjoint labels and
using the intrinsic antisymmetry of epsilon and structure constants. Raw
FeynPy output can be much longer than FeynRules output, but the canonical maps
collapse to the same terms for all supported bosonic coefficient sectors.

## Two-Fermion Sector

The two-fermion sector is partially proven:

- rows: `131`
- strict exact matches: `46`
- strict mismatches: `85`

For ordinary two-fermion rows, the FeynRules parser rewrites:

- `TensDot[Ga[...], ...][i,j]` to native gamma-chain tensors;
- `SlashedP[n]` to `pcomp(qn,mu) * gamma(mu)`;
- `ProjM` and `ProjP` to the spinor metric, because FeynPy encodes chirality in
  the field class rather than as an explicit projector tensor;
- `T`, `Ta`, `f`, `fsu2`, `Eps`, `IndexDelta`, `ME`, `FV`, and `SP` to native
  tensors.

The two Weinberg rows are now strict exact matches even though they have no
literal local signature:

- `Phi|Phi|lL|lL`
- `Phibar|Phibar|lLbar|lLbar`

FeynRules emits these as same-chirality rows from `CC[LLbar].LL` plus the
Hermitian conjugate. FeynPy packages the same operator as a mixed
`lLbar,lL` bilinear with `dirac_C`. The comparison handles this by parsing the
FeynRules Weinberg projectors as `dirac_C` and comparing against:

```text
FeynPy(lLbar, lL, H, H) - FeynPy(lL, lLbar, H, H)
```

The minus sign is required because the second assignment swaps the two spinor
slots of the antisymmetric charge-conjugation tensor. The canonical map keeps
both `alphaWeinberg(f1,f2)` and `alphaWeinberg(f2,f1)`, so the flavor transpose
is checked rather than assumed.

The remaining two-fermion strict mismatches are not hidden. Representative
classes are:

- explicit generation metrics in FeynPy monomials where the FeynRules export
  carries the flavor structure only on the Wilson coefficient;
- sigma/gamma phase-convention differences, for example sectors where the
  strict coefficient comparison sees `1/2` versus `i/2`;
- expanded derivative and gauge-field branches whose spinor-chain identities
  are not yet normalized by the comparison layer.

## Four-Fermion Sector

The four-fermion sector is matched at operator-content level but not yet at
strict exact tensor-map level:

- rows: `21`
- strict exact matches: `0`
- strict mismatches: `21`

The current exact parser can read the FeynRules four-fermion syntax and keeps
indexed Wilson coefficients visible. However, the strict tensor maps still
differ systematically because the two sides package charge-conjugated bilinears
differently. FeynPy uses explicit `dirac_C` tensors in the closed bilinear
packaging, while the FeynRules export often presents the corresponding
structures through projector/metric and gamma-chain forms after `CC[...]`
processing.

The existing charge-conjugation overlay is therefore still an
operator-content overlay for the `Ec` rows, not a full exact proof. It pairs
the reference and FeynPy-only signatures using bar-insensitive field content
and matching `alphaEc*` heads, but it does not rewrite the full spinor tensor
structure into an exact canonical form.

## Charge Conjugation

Charge conjugation is used in two places, with different strength:

- Operator-content overlay: used for `alphaEc*` and `alphaWeinberg` packaging
  differences. This proves the same coefficient heads exist under the
  charge-conjugate FeynPy signature, but it does not by itself prove exact
  tensor equality.
- Weinberg strict exact comparison: used as an actual tensor-level proof. The
  FeynRules same-chirality row is transformed to `dirac_C`, the two FeynPy
  mixed assignments are antisymmetrized, and the full canonical map is checked.

No broad charge-conjugation rewrite is applied to all rows. This avoids making
rows match unless the required tensor identity has been explicitly encoded and
verified.

## Meaning of `--check`

`models/SMEFT2/comparison.py --check` now fails if strict exact mismatches are
present. With the stricter indexed-coefficient comparison, the current report
therefore fails `--check` because `106` exact mismatches remain. The
operator-content result is still complete at `184/184`, but exact symbolic
equivalence is currently proven only for `78/184` rows.

The next work should not be to relax the check. It should be to resolve the
remaining strict mismatch classes one at a time: flavor-index convention,
sigma/gamma phase conventions, derivative-order normalization, and full
four-fermion charge-conjugated bilinear identities.
