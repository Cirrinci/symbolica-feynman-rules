# SMEFT2 Comparison Method Report

Updated for the SMEFT2 FeynRules/FeynPy comparison on 2026-07-24.

## Current Result

The strict comparison is complete.

- Reference vertices: `184`
- Operator-content matches: `184/184`
- Exact symbolic supported vertices: `184/184`
- Exact symbolic equal vertices: `184/184`
- Exact symbolic unequal vertices: `0`
- Exact symbolic error vertices: `0`
- Sector split: `32` bosonic, `131` two-fermion, `21` four-fermion

The direct signature accounting is still useful but it is not the final
criterion. There are `182` exact field-multiset signatures shared literally,
`2` FeynRules-only Weinberg signatures, and `8` FeynPy-only
charge-conjugation partner signatures. Those nonliteral rows are now resolved
by exact tensor-level packaging comparisons.

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

It now passes without relaxing the check. The change was to resolve the
mismatch classes at the symbolic layer.

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

Result: `131/131` exact symbolic matches.

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

The two Weinberg rows are exact matches even though they have no literal local
same-chirality signature:

- `Phi|Phi|lL|lL`
- `Phibar|Phibar|lLbar|lLbar`

FeynRules emits these through `CC[...]` same-chirality packaging. FeynPy stores
the same operator as a mixed `lLbar,lL` bilinear with explicit `dirac_C`. The
exact comparison parses the FeynRules projectors as `dirac_C` and compares the
reference row against the antisymmetrized local pair:

```text
FeynPy(lLbar, lL, H, H) - FeynPy(lL, lLbar, H, H)
```

The minus sign is fixed by the antisymmetry of the charge-conjugation tensor
when the two spinor slots are exchanged. The canonical scalar coefficient keeps
both `alphaWeinberg(f1,f2)` and the transposed flavor structure, so the flavor
transpose is checked explicitly.

## Four-Fermion Sector

Result: `21/21` exact symbolic matches.

The non-`Ec` four-fermion rows needed ordinary source corrections, not relaxed
comparison rules. The weak-triplet currents in `alphaOqq3` and `alphaOlq3`
now use the correct second weak endpoint. The conjugate color-octet branches
for `alphaOquqd8` and `alphaEquqdtwo8` now reverse both color-generator matrix
orientations. These were real orientation mismatches; the comparison now
matches them directly after canonicalization.

The `Ec` rows were different. Their mismatch was a packaging problem:
FeynPy keeps explicit `dirac_C` tensors in closed bilinears, while FeynRules
exports the same charge-conjugated structures through its `CC[...]`
processing. The comparison therefore formalizes that packaging equivalence
instead of merely annotating it.

There are two exact `Ec` modes.

Same-signature `Ec` rows use a crossed charge-conjugation transform. The
comparison traces each `dirac_C` arm, replaces the two explicit `C` factors by
the FeynRules bilinear packaging, swaps the non-spinor external labels at the
two charge-conjugation boundary endpoints, swaps the first and fourth
arguments of the `alphaEc*` coefficient head, applies the fixed crossing phase,
reconstructs the tensor expression, and recanonicalizes it. The row is accepted
only if the resulting coefficient-sector map is exactly identical to the
FeynRules map.

Partner-signature `Ec` rows use a direct charge-conjugation transform. The
comparison constructs candidate FeynPy-only partner orders in the reference
leg order by bar-insensitive field matching. If two identical fields can be
assigned in more than one way, both symmetric and antisymmetric duplicate-leg
sums are tested as appropriate. The possible overall phases are tested, but a
phase is accepted only when the final canonical map is exactly equal.

The resolved partner classes are:

| Reference packaging | Coefficient heads | Partner handling |
| --- | --- | --- |
| `dRbar|eR|lLbar|qL` | `alphaEcqedl`, `alphaEcqedlthree` | direct partner, phase `-1`, symmetric duplicate sum |
| `dR|eRbar|lL|qLbar` | `alphaEcqedl`, `alphaEcqedlthree` | direct partner, phase `-1`, symmetric duplicate sum |
| `eRbar|lL|qL|uRbar` | `alphaEcuelq`, `alphaEcuelqtwo` | direct partner, phase `+1` |
| `eR|lLbar|qLbar|uR` | `alphaEcuelq`, `alphaEcuelqtwo` | direct partner, phase `-1` |
| `dRbar|qL|qL|uRbar` | `alphaEcudqq`, `alphaEcudqqtwo` | duplicate-leg sums, symmetric or antisymmetric by head |
| `dR|qLbar|qLbar|uR` | `alphaEcudqq`, `alphaEcudqqtwo` | duplicate-leg sums, symmetric or antisymmetric by head |

The key safety property is that the charge-conjugation machinery is not a
general row matcher. It is restricted to `alphaEc*` coefficient sectors, it
must find the appropriate charge-conjugation arms, and it must end in exact
canonical tensor-map equality.

## Charge Conjugation Usage

Charge conjugation is used in three places.

First, operator-content accounting uses it to explain why the two Weinberg
reference-only signatures and the eight FeynPy-only `Ec` partner signatures are
not missing physics. This accounting explains coverage, but it is not by itself
the strict proof.

Second, the Weinberg rows use charge conjugation at exact level by comparing
the same-chirality FeynRules row against the antisymmetrized mixed FeynPy
packaging. This checks the full tensor map, including flavor transposition.

Third, the `Ec` four-fermion rows use the crossed or direct packaging
transforms described above. These transforms formalize the closed-bilinear sign
and field-order conventions already present in the engine, but the comparison
layer now makes them explicit and machine-checkable.

No row passes because it merely has the right coefficient head. It must also
pass the exact canonical tensor-map comparison.

## Meaning of `--check`

`models/SMEFT2/comparison.py --check` returns nonzero if any of these are true:

- operator-content coverage is not `184/184`;
- any FeynPy-only signature remains unexplained;
- exact symbolic support does not cover all reference vertices;
- any exact symbolic row is unequal or errors;
- any supported canonical gauge-sector map is unequal or errors.

Raw coefficient-head count mismatches are not part of the normal gate because
they are expansion diagnostics. They become part of the gate only with
`--strict-counts`.

The current state passes the normal strict gate because all `184` supported
reference vertices are exact symbolic matches.
