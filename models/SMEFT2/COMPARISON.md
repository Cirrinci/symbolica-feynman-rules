# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-07-23` by `models/SMEFT2/comparison.py`.

## Scope

Signature coverage, coefficient-head content, and raw coefficient-head multiplicity diagnostics, plus exact symbolic comparison for supported bosonic rows and canonical tensor-monomial equality for supported pure nonabelian gauge vertices. Full tensor-rule equality is not claimed globally.

| Item | Value |
| --- | ---: |
| Reference vertices | 184 |
| FeynPy 3-6 point signatures | 192 |
| Shared signatures (exact field multiset) | 182 |
| Reference-only signatures (exact field multiset) | 2 |
| FeynPy-only signatures (exact field multiset) | 8 |
| — of which charge-conjugation partners | 8 |
| — of which unexplained | 0 |
| FeynPy-only zero-signature artifacts (dropped) | 2 |
| Shared coefficient-head matches | 176 |
| Charge-conjugation packaging matches (modulo CC) | 8 |
| Operator-content matches (incl. charge conjugation) | 184 |
| Shared raw head-count matches | 90 |
| Shared raw head-count mismatches | 92 |
| Shared raw head-count benign expansions | 9 |
| Shared raw head-count mismatches with unexplained deltas | 83 |
| Exact symbolic supported vertices | 32 |
| Exact symbolic equal vertices | 32 |
| Exact symbolic unequal vertices | 0 |
| Exact symbolic error vertices | 0 |
| Canonical tensor-map supported vertices | 32 |
| Canonical tensor-map equal vertices | 32 |
| Canonical tensor-map unequal vertices | 0 |
| Canonical tensor-map error vertices | 0 |
| Canonical tensor-map equal coefficient sectors | 93 |
| Canonical tensor-map unequal coefficient sectors | 0 |
| Canonical-map FeynPy raw monomials | 10472 |
| Canonical-map FeynPy canonical monomials | 3779 |
| Canonical-map FeynPy redundant monomials (raw - canonical) | 6693 |
| Explained benign head-count deltas | 15 |
| Unexplained head-count deltas | 295 |

## Basis

- Reference: `EFT-only FeynRules Ltot`.
- Local default model: `EFT-only FeynPy Ltot`.
- Local SM plus EFT model: `Lfull`.
- Omitted sectors: `none`.

## Status Counts

| Status | Count |
| --- | ---: |
| `FEYNPY_ONLY_ALGEBRAICALLY_ZERO` | 2 |
| `FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER` | 8 |
| `MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING` | 2 |
| `SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH` | 6 |
| `SHARED_HEADS_MATCH` | 176 |

## Exact Symbolic Comparison

This layer is currently enabled for bosonic SMEFT2 rows. It parses the full FeynRules tensor rule into native tensors, canonicalizes index structure, and checks whether the exact symbolic difference is zero. Two-fermion and four-fermion rows still fall back to the signature/head diagnostics above.

| Signature | Status |
| --- | --- |
| `B|B|B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|B|Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `B|B|Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `B|Phi|Phibar` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi|Wi|Wi` | `EXACT_MATCH` |
| `B|Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `B|Phi|Phi|Phibar|Phibar|Wi` | `EXACT_MATCH` |
| `G|G|G` | `EXACT_MATCH` |
| `G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|Phi|Phibar` | `EXACT_MATCH` |
| `G|G|G|Phi|Phibar` | `EXACT_MATCH` |
| `G|G|Phi|Phibar` | `EXACT_MATCH` |
| `Phi|Phibar|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar|Wi` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phi|Phi|Phibar|Phibar|Phibar` | `EXACT_MATCH` |
| `Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |

## Canonical Tensor-Map Gauge Comparison

This comparison is currently enabled for pure nonabelian gauge vertices (`G^n` and `Wi^n`). It parses FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu3`, and `fsu2` into native tensors, then compares canonical monomial maps per Wilson coefficient. It uses intrinsic tensor symmetries, dummy-index relabeling, commuting factor ordering, and exact coefficient collection; it does not use Jacobi, momentum conservation, EOM, IBP, or 4D reductions.

| Signature | Status | Coefficient sectors |
| --- | --- | --- |
| `B|B|B|B|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 3/3 -> canonical 3/3 |
| `B|B|B|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 6/6 -> canonical 6/6 |
| `B|B|B|Phi|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 3/3 -> canonical 3/3 |
| `B|B|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaKH` match: raw 1/1 -> canonical 1/1; `alphaOHB` match: raw 2/2 -> canonical 2/2; `alphaOHBt` match: raw 8/2 -> canonical 1/1; `alphaRBDH` match: raw 4/4 -> canonical 4/4; `alphaRDH` match: raw 9/9 -> canonical 9/9 |
| `B|B|Phi|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 6/6 -> canonical 6/6 |
| `B|B|Phi|Phibar|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 14/6 -> canonical 6/6 |
| `B|B|Phi|Phi|Phibar|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHD` match: raw 2/2 -> canonical 2/2; `alphaRHDp` match: raw 2/2 -> canonical 2/2 |
| `B|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaRBDH` match: raw 4/4 -> canonical 4/4; `alphaRDH` match: raw 4/4 -> canonical 4/4 |
| `B|Phi|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaKH` match: raw 1/1 -> canonical 1/1; `alphaOHWB` match: raw 2/2 -> canonical 2/2; `alphaOHWBt` match: raw 4/4 -> canonical 1/1; `alphaRBDH` match: raw 2/2 -> canonical 2/2; `alphaRDH` match: raw 9/9 -> canonical 9/9; `alphaRWDH` match: raw 2/2 -> canonical 2/2 |
| `B|Phi|Phibar|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHWB` match: raw 4/2 -> canonical 2/2; `alphaOHWBt` match: raw 4/2 -> canonical 1/1; `alphaRDH` match: raw 24/12 -> canonical 12/12; `alphaRWDH` match: raw 8/6 -> canonical 6/6 |
| `B|Phi|Phibar|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaRDH` match: raw 24/12 -> canonical 12/12; `alphaRWDH` match: raw 6/6 -> canonical 6/6 |
| `B|Phi|Phi|Phibar|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHD` match: raw 8/8 -> canonical 8/8; `alphaRHDp` match: raw 8/8 -> canonical 8/8; `alphaRHDpp` match: raw 10/10 -> canonical 10/10 |
| `B|Phi|Phi|Phibar|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaRHDp` match: raw 4/4 -> canonical 4/4 |
| `G|G|G` | `CANONICAL_MAP_MATCH` | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 14/8 -> canonical 8/8; `alphaO3Gt` match: raw 21/42 -> canonical 12/12; `alphaR2G` match: raw 54/36 -> canonical 36/36 |
| `G|G|G|G` | `CANONICAL_MAP_MATCH` | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 144/48 -> canonical 48/48; `alphaO3Gt` match: raw 138/186 -> canonical 72/72; `alphaR2G` match: raw 204/156 -> canonical 156/156 |
| `G|G|G|G|G` | `CANONICAL_MAP_MATCH` | `alphaO3G` match: raw 720/240 -> canonical 120/120; `alphaO3Gt` match: raw 720/420 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| `G|G|G|G|G|G` | `CANONICAL_MAP_MATCH` | `alphaO3G` match: raw 720/720 -> canonical 120/120; `alphaO3Gt` match: raw 720/360 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| `G|G|G|G|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHG` match: raw 12/6 -> canonical 6/6; `alphaOHGt` match: raw 24/3 -> canonical 3/3 |
| `G|G|G|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHG` match: raw 12/6 -> canonical 6/6; `alphaOHGt` match: raw 24/12 -> canonical 3/3 |
| `G|G|Phi|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHG` match: raw 2/2 -> canonical 2/2; `alphaOHGt` match: raw 8/8 -> canonical 1/1 |
| `Phi|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaRDH` match: raw 4/4 -> canonical 4/4; `alphaRWDH` match: raw 4/4 -> canonical 4/4 |
| `Phi|Phibar|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaOHW` match: raw 2/2 -> canonical 2/2; `alphaOHWt` match: raw 8/8 -> canonical 1/1; `alphaRDH` match: raw 20/12 -> canonical 12/12; `alphaRWDH` match: raw 24/20 -> canonical 20/20 |
| `Phi|Phibar|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHW` match: raw 12/6 -> canonical 6/6; `alphaOHWt` match: raw 24/12 -> canonical 3/3; `alphaRDH` match: raw 36/24 -> canonical 24/24; `alphaRWDH` match: raw 60/48 -> canonical 48/48 |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHW` match: raw 12/6 -> canonical 6/6; `alphaOHWt` match: raw 24/3 -> canonical 3/3; `alphaRDH` match: raw 24/24 -> canonical 24/24; `alphaRWDH` match: raw 48/48 -> canonical 48/48 |
| `Phi|Phi|Phibar|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOHBox` match: raw 12/12 -> canonical 12/12; `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaOlambda` match: raw 2/2 -> canonical 2/2; `alphaRHDp` match: raw 4/4 -> canonical 4/4; `alphaRHDpp` match: raw 8/8 -> canonical 8/8 |
| `Phi|Phi|Phibar|Phibar|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHD` match: raw 8/8 -> canonical 8/8; `alphaRHDp` match: raw 8/8 -> canonical 8/8; `alphaRHDpp` match: raw 12/12 -> canonical 12/12 |
| `Phi|Phi|Phibar|Phibar|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaRHDp` match: raw 8/8 -> canonical 8/8 |
| `Phi|Phi|Phi|Phibar|Phibar|Phibar` | `CANONICAL_MAP_MATCH` | `alphaOH` match: raw 6/6 -> canonical 6/6 |
| `Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 14/8 -> canonical 8/8; `alphaO3Wt` match: raw 21/42 -> canonical 12/12; `alphaR2W` match: raw 54/36 -> canonical 36/36 |
| `Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 144/48 -> canonical 48/48; `alphaO3Wt` match: raw 138/186 -> canonical 72/72; `alphaR2W` match: raw 204/156 -> canonical 156/156 |
| `Wi|Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaO3W` match: raw 720/240 -> canonical 120/120; `alphaO3Wt` match: raw 720/420 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaO3W` match: raw 720/720 -> canonical 120/120; `alphaO3Wt` match: raw 720/360 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |

## Largest Reference-Side Head Gaps

| Head | Count |
| --- | ---: |

## Largest Local Extra Heads

| Head | Count |
| --- | ---: |

## Explained Benign Raw Head-Count Deltas

These are raw coefficient-head occurrence-count diagnostics. They catch some missing or duplicated content, but they are not tensor-rule equality proofs because equivalent algebra can be printed with different occurrence counts.

| Signature | Head | Reference | FeynPy | Reason |
| --- | --- | ---: | ---: | --- |
| `B|Phibar|dRbar|qL` | `alphaEdB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phibar|eRbar|lL` | `alphaEeB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phibar|qLbar|uR` | `alphaEuB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|dR|qLbar` | `alphaEdB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|eR|lLbar` | `alphaEeB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|qL|uRbar` | `alphaEuB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaEBq` | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaEBqtp` | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaRBqtp` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `B|qL|qLbar` | `g1` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `G|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `G|qL|qLbar` | `g3` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `Wi|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `Wi|qL|qLbar` | `g2` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |

## Largest Unexplained Raw Head-Count Deltas

These exclude the explicit benign expansions listed above. The large pure-gauge raw deltas can remain large even where the canonical tensor-map comparison above proves equality.

| Head | Total absolute delta |
| --- | ---: |
| `g2` | 2260 |
| `g3` | 2218 |
| `alphaR2G` | 786 |
| `alphaR2W` | 786 |
| `alphaO3Gt` | 729 |
| `alphaO3Wt` | 729 |
| `alphaO3G` | 582 |
| `alphaO3W` | 582 |
| `g1` | 196 |
| `alphaRqD` | 72 |
| `alphaRDH` | 52 |
| `alphaOHGt` | 33 |
| `alphaOHWt` | 33 |
| `alphaRdD` | 32 |
| `alphaRuD` | 32 |
| `alphaRlD` | 32 |
| `alphaEGqp` | 24 |
| `alphaEGqtp` | 24 |
| `alphaEWqp` | 24 |
| `alphaEWqtp` | 24 |

## Files

- `vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
