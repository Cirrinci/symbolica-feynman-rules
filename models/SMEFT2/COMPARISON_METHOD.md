# SMEFT2 Comparison Method Report

Updated for the SMEFT2 FeynRules/FeynPy comparison on 2026-07-27.

## Current Result

Exact symbolic comparison is graded in three tiers:

- Reference vertices: `184`
- Operator-content matches: `184/184`
- Direct exact symbolic matches (`EXACT_MATCH`): `176/184`
- Exact modulo pinned CC packaging (`MATCH_MODULO_CC_PACKAGING`): `8/184`
  (two Weinberg rows plus six pinned `alphaEc*` partner rows)
- Unresolved CC packaging (`UNRESOLVED_CC_PACKAGING`): `0/184`
- Exact symbolic unequal vertices: `0`
- Exact symbolic error vertices: `0`
- Sector split: `32` bosonic, `131` two-fermion, `21` four-fermion

Headline form:

```text
direct exact: 176/184; modulo pinned CC: 8/184; unresolved CC: 0/184; operator content: 184/184
```

The direct signature accounting remains useful but is not the final criterion.
There are `182` exact field-multiset signatures shared literally, `2`
FeynRules-only Weinberg signatures, and `8` FeynPy-only charge-conjugation
partner signatures.

The raw coefficient-head occurrence count remains a diagnostic:
`100/182` shared rows have identical raw counts, while the other shared rows
still show expansion-form differences. These raw deltas are visible in
`COMPARISON.md` and `vertex_comparison_report.json`, but they are not accepted
or rejected by the normal `--check` gate. `--strict-counts` can still be used
when the diagnostic count itself is the target.

The normal gate is:

```bash
.venv/bin/python models/SMEFT2/comparison.py --check
```

It requires every supported row to be a direct `EXACT_MATCH`. Pinned CC
packaging can be accepted with `--allow-cc-packaging`. Unresolved CC packaging
rows never pass `--check`.

## Basis

The reference side is `reference/Ltot_SMEFT_FeynRules.json`, exported from the
EFT-only FeynRules `Ltot`. The local side is the EFT-only FeynPy `Ltot` from
`SMEFT2.py`. `Lfull` still exists for SM-plus-EFT use, but it is not the
comparison basis.

Field names are normalized to the FeynRules convention before any matching:

- `LL` / `LL.bar` become `lL` / `lLbar`.
- `LR` / `LR.bar` become `eR` / `eRbar`.
- `QL`, `UR`, `DR` and their bars become `qL`, `uR`, `dR` and their bars.
- `Phi.bar` becomes `Phibar`.
- Gauge fields keep the FeynRules names `B`, `Wi`, and `G`.

## Comparison Layers

The comparison has four layers.

1. Literal signature coverage. Vertices are keyed by sorted field multiset,
   for example `B|qL|qLbar`. This tells us whether FeynRules and FeynPy expose
   the same external fields literally.

2. Operator-content matching. For each row the comparison checks the Wilson
   coefficient heads. Direct rows require the same heads under the same
   signature. Charge-conjugation rows may be paired with a controlled partner
   signature if the field content and coefficient head identify the same
   operator packaging.

3. Raw head-count diagnostics. This counts textual occurrences of each Wilson
   head. It catches suspicious over-expansion, but it is not an equality proof.
   Dual field-strength expansion and derivative-order expansion can change the
   raw count while leaving the canonical tensor rule unchanged.

4. Exact symbolic comparison. Both sides are converted to native tensor
   expressions, filtered by indexed Wilson-coefficient head, canonicalized into
   tensor-monomial maps, and compared coefficient-sector by coefficient-sector.
   This is the strict proof used by `--check`.

## Exact Tensor-Map Comparison

The exact layer does not compare printed strings. It compares canonical tensor
maps.

On the FeynRules side, the parser rewrites exported constructs into the same
native tensor vocabulary used by FeynPy. This includes `ME`, `FV`, `SP`, `Eps`,
`f`, `fsu2`, `fsu3`, `T`, `Ta`, `IndexDelta`, gamma chains from `TensDot`,
projectors, and `SlashedP`.

On the FeynPy side, the local vertex rule is already lowered into explicit
tensor factors. The comparison then filters terms structurally by coefficient
head. This is essential for function-valued Wilson coefficients such as
`alphaKq(f1,f2)` or `conj(alphaWeinberg(f2,f1))`: the indexed coefficient
factor, flavor order, and complex conjugation are kept inside the scalar
coefficient. The comparison cannot pass by merely extracting a bare symbol such
as `alphaKq`.

Canonicalization uses:

- intrinsic tensor symmetries: symmetric metrics, antisymmetric structure
  constants, weak epsilon, Lorentz epsilon, color epsilon, and `dirac_C`;
- dummy-index relabeling within each representation class;
- deterministic ordering of commuting tensor factors;
- exact scalar coefficient collection;
- narrow gauge identities explicitly encoded in the comparison layer.

The comparison does not use equations of motion, integration by parts,
momentum conservation, Schouten identities, Fierz identities, or broad
gamma-matrix reductions. If a row is marked `EXACT_MATCH`, the two
coefficient-sector tensor maps are identical after only the canonicalizations
and identities described here.

## Bosonic Sector

Result: `32/32` exact symbolic matches.

The bosonic vertices were already closest to FeynRules because both sides use
the same unbroken gauge basis. They are not generally literal string matches:
FeynPy may expand dummy labels and field-strength branches more explicitly.
The canonical tensor map collapses those forms by using metric symmetry,
Levi-Civita antisymmetry, structure-constant antisymmetry, and deterministic
dummy relabeling.

The nonabelian gauge rows also need narrow group identities:

- generator product ordering, using the commutator relation
  `[T^a,T^b] = i f^{abc} T^c`;
- a Jacobi reducer for products of two structure constants sharing one dummy
  adjoint index.

Dual field-strength rows are handled through the explicit epsilon-tensor
definition and epsilon antisymmetry. This is why some raw head counts still
look different even though the canonical maps are equal.

## Two-Fermion Sector

Result: `129/131` direct `EXACT_MATCH`, plus `2/131`
`MATCH_MODULO_CC_PACKAGING` (Weinberg).

The first reliable target was the set of two-fermion rows whose raw
coefficient-head counts already matched. Those rows established that the
indexed Wilson-coefficient filter, flavor ordering, and basic gamma-chain
parser were working. The remaining rows then reduced to specific mismatch
classes rather than a broad comparison failure.

The FeynRules two-fermion parser normalizes:

- `TensDot[Ga[...], ...][i,j]` to native gamma-chain tensors;
- `SlashedP[n]` to momentum components times gamma matrices;
- `ProjM` and `ProjP` to the spinor metric, because FeynPy encodes chirality
  in the field class rather than as an explicit projector tensor;
- `T`, `Ta`, `f`, `fsu2`, `Eps`, `IndexDelta`, `ME`, `FV`, and `SP` to native
  tensor heads.

Several mismatch classes required source-side fixes in the FeynPy model:

- labelled Dirac `Gamma` plus `CovD` monomials now go through generic
  covariant derivative lowering instead of a too-special local branch;
- flavor source labels in kinetic-core lowering are preserved;
- `LF2HD2` and `LEvF2HD2` phases and weak-generator factors were aligned with
  the FeynRules convention;
- `LF2XH` and `LEvF2XH` up-type `Wi` dipoles and Hermitian-conjugate color/weak
  orientations were fixed;
- `LF2XD` derivative-current terms were expanded through explicit color and
  weak derivative-current helpers;
- compact Higgs derivative labels were kept where dropping the label changed a
  two-fermion `B` vertex head.

The comparison layer also needed controlled tensor identities:

- non-ASCII Greek dummy labels from FeynRules are mapped without collisions;
- dual field-strength branches are normalized through epsilon antisymmetry;
- derivative-order differences are compared after tensor canonicalization;
- SU(2) pseudoreality identities for `T epsilon` and `T T epsilon` products
  resolve rows such as `alphaRuHD1`;
- generator ordering and the narrow Jacobi reducer resolve the triple
  nonabelian derivative/dual classes.

The two Weinberg rows are `MATCH_MODULO_CC_PACKAGING` (sign pinned) even though
they have no literal local same-chirality signature:

- `Phi|Phi|lL|lL`
- `Phibar|Phibar|lLbar|lLbar`

FeynRules emits these through `CC[...]` same-chirality packaging. FeynPy stores
the same operator as a mixed `lLbar,lL` bilinear with explicit `dirac_C`. The
comparison parses the FeynRules projectors as `dirac_C` and compares the
reference row against the antisymmetrized local pair:

```text
FeynPy(lLbar, lL, H, H) - FeynPy(lL, lLbar, H, H)
```

The minus sign is fixed by the antisymmetry of the charge-conjugation tensor
when the two spinor slots are exchanged. The canonical scalar coefficient keeps
both `alphaWeinberg(f1,f2)` and the transposed flavor structure, so the flavor
transpose is checked explicitly. This is equality modulo an explicitly tracked
CC packaging transform, not a direct same-signature `EXACT_MATCH`.

## Four-Fermion Sector

Result: `15/21` direct `EXACT_MATCH`, plus `6/21`
`MATCH_MODULO_CC_PACKAGING` (`alphaEc*` partner rows with pinned rules).

The non-`Ec` four-fermion rows needed ordinary source corrections, not relaxed
comparison rules. The weak-triplet currents in `alphaOqq3` and `alphaOlq3`
now use the correct second weak endpoint. The conjugate color-octet branches
for `alphaOquqd8` and `alphaEquqdtwo8` now reverse both color-generator matrix
orientations. These were real orientation mismatches; the comparison now
matches them directly after canonicalization.

The `Ec` rows are different. Their mismatch is a packaging problem: FeynPy
keeps explicit `dirac_C` tensors in closed bilinears, while FeynRules exports
the same charge-conjugated structures through its `CC[...]` processing.

Same-signature `Ec` sectors may use a crossed charge-conjugation transform
with a fixed crossing phase. Partner-signature `Ec` rows use a pinned rule
table keyed by reference signature and coefficient head. Each rule specifies
exactly one FeynPy partner signature, one phase, and one duplicate-leg symmetry
mode. The comparison no longer searches over phase or over symmetric versus
antisymmetric duplicate-leg sums at acceptance time.

| Reference packaging | Coefficient heads | Current grade |
| --- | --- | --- |
| `dRbar|eR|lLbar|qL` | `alphaEcqedl`, `alphaEcqedlthree` | pinned modulo CC, phase `-1`, symmetric duplicate sum |
| `dR|eRbar|lL|qLbar` | `alphaEcqedl`, `alphaEcqedlthree` | pinned modulo CC, phase `-1`, symmetric duplicate sum |
| `eRbar|lL|qL|uRbar` | `alphaEcuelq`, `alphaEcuelqtwo` | pinned modulo CC, phase `+1`, symmetric duplicate sum |
| `eR|lLbar|qLbar|uR` | `alphaEcuelq`, `alphaEcuelqtwo` | pinned modulo CC, phase `-1`, symmetric duplicate sum |
| `dRbar|qL|qL|uRbar` | `alphaEcudqq`, `alphaEcudqqtwo` | pinned modulo CC; `alphaEcudqq` antisymmetric, `alphaEcudqqtwo` symmetric |
| `dR|qLbar|qLbar|uR` | `alphaEcudqq`, `alphaEcudqqtwo` | pinned modulo CC; `alphaEcudqq` antisymmetric, `alphaEcudqqtwo` symmetric |

If a listed pinned transform fails canonical-map equality, the row is not
accepted. If no pinned rule exists for a future `Ec` mismatch, it is reported as
`UNRESOLVED_CC_PACKAGING`.

## Charge Conjugation Usage

Charge conjugation is used in three places, with different strength.

First, operator-content accounting uses it to explain why the two Weinberg
reference-only signatures and the eight FeynPy-only `Ec` partner signatures are
not missing physics. This accounting explains coverage, but it is not by itself
a tensor-level proof.

Second, the Weinberg rows use charge conjugation at pinned packaging level by
comparing the same-chirality FeynRules row against the antisymmetrized mixed
FeynPy packaging. The relative minus sign is derived from `C` antisymmetry.
These rows are `MATCH_MODULO_CC_PACKAGING`.

Third, the `Ec` four-fermion partner rows use the pinned rule table described
above. The table makes the partner signature, phase, and duplicate-leg
symmetry explicit; the row passes only if that single transform gives exact
canonical-map equality.

## Meaning of `--check`

`models/SMEFT2/comparison.py --check` returns nonzero if any of these are true:

- operator-content coverage is not `184/184`;
- any FeynPy-only signature remains unexplained;
- exact symbolic support does not cover all reference vertices;
- any exact symbolic row is unequal or errors;
- any row is `UNRESOLVED_CC_PACKAGING`;
- any row is `MATCH_MODULO_CC_PACKAGING` unless `--allow-cc-packaging` is set;
- any supported canonical gauge-sector map is unequal or errors.

Raw coefficient-head count mismatches are not part of the normal gate because
they are expansion diagnostics. They become part of the gate only with
`--strict-counts`.

The current state fails the default `--check` gate because eight rows are
modulo pinned CC packaging rather than direct `EXACT_MATCH`. With
`--allow-cc-packaging`, those pinned packaging rows are accepted.
