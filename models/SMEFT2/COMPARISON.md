# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-07-22` by `models/SMEFT2/comparison.py`.

## Scope

Signature coverage, coefficient-head content, and raw coefficient-head multiplicity diagnostics, plus canonical tensor-monomial equality for supported pure nonabelian gauge vertices. Full tensor-rule equality is not claimed globally.

| Item | Value |
| --- | ---: |
| Reference vertices | 184 |
| FeynPy 3-6 point signatures | 192 |
| Shared signatures | 182 |
| Reference-only signatures | 2 |
| FeynPy-only signatures | 10 |
| Shared coefficient-head matches | 168 |
| Shared raw head-count matches | 83 |
| Shared raw head-count mismatches | 99 |
| Shared raw head-count benign expansions | 9 |
| Shared raw head-count mismatches with unexplained deltas | 90 |
| Canonical tensor-map supported vertices | 8 |
| Canonical tensor-map equal vertices | 4 |
| Canonical tensor-map unequal vertices | 4 |
| Canonical tensor-map error vertices | 0 |
| Canonical tensor-map equal coefficient sectors | 24 |
| Canonical tensor-map unequal coefficient sectors | 4 |
| Explained benign head-count deltas | 15 |
| Unexplained head-count deltas | 331 |

## Basis

- Reference: `EFT-only FeynRules Ltot`.
- Local default model: `EFT-only FeynPy Ltot`.
- Local SM plus EFT model: `Lfull`.
- Omitted sectors: `none`.

## Status Counts

| Status | Count |
| --- | ---: |
| `FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING` | 8 |
| `FEYNPY_ONLY_WEINBERG_PACKAGING` | 2 |
| `MISSING_SIGNATURE_WEINBERG_PACKAGING` | 2 |
| `SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH` | 6 |
| `SHARED_HEADS_MATCH` | 168 |
| `SHARED_LOCAL_EXTRA_HEADS` | 6 |
| `SHARED_LOCAL_PP_EXTRA` | 2 |

## Canonical Tensor-Map Gauge Comparison

This comparison is currently enabled for pure nonabelian gauge vertices (`G^n` and `Wi^n`). It parses FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu3`, and `fsu2` into native tensors, then compares canonical monomial maps per Wilson coefficient. It uses intrinsic tensor symmetries, dummy-index relabeling, commuting factor ordering, and exact coefficient collection; it does not use Jacobi, momentum conservation, EOM, IBP, or 4D reductions.

| Signature | Status | Coefficient sectors |
| --- | --- | --- |
| `G|G|G` | `CANONICAL_MAP_MISMATCH` | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 14/8 -> canonical 8/8; `alphaO3Gt` match: raw 21/42 -> canonical 12/12; `alphaR2G` mismatch: raw 54/36 -> canonical 54/36 |
| `G|G|G|G` | `CANONICAL_MAP_MISMATCH` | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 144/48 -> canonical 48/48; `alphaO3Gt` match: raw 138/186 -> canonical 72/72; `alphaR2G` mismatch: raw 204/156 -> canonical 156/156 |
| `G|G|G|G|G` | `CANONICAL_MAP_MATCH` | `alphaO3G` match: raw 720/240 -> canonical 120/120; `alphaO3Gt` match: raw 720/420 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| `G|G|G|G|G|G` | `CANONICAL_MAP_MATCH` | `alphaO3G` match: raw 720/720 -> canonical 120/120; `alphaO3Gt` match: raw 720/360 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| `Wi|Wi|Wi` | `CANONICAL_MAP_MISMATCH` | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 14/8 -> canonical 8/8; `alphaO3Wt` match: raw 21/42 -> canonical 12/12; `alphaR2W` mismatch: raw 54/36 -> canonical 54/36 |
| `Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MISMATCH` | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 144/48 -> canonical 48/48; `alphaO3Wt` match: raw 138/186 -> canonical 72/72; `alphaR2W` mismatch: raw 204/156 -> canonical 156/156 |
| `Wi|Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaO3W` match: raw 720/240 -> canonical 120/120; `alphaO3Wt` match: raw 720/420 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `CANONICAL_MAP_MATCH` | `alphaO3W` match: raw 720/720 -> canonical 120/120; `alphaO3Wt` match: raw 720/360 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |

## Largest Reference-Side Head Gaps

| Head | Count |
| --- | ---: |
| `alphaWeinberg` | 2 |
| `alphaEcqedl` | 2 |
| `alphaEcqedlthree` | 2 |
| `alphaEcudqq` | 2 |
| `alphaEcudqqtwo` | 2 |
| `alphaEcuelq` | 2 |
| `alphaEcuelqtwo` | 2 |

## Largest Local Extra Heads

| Head | Count |
| --- | ---: |
| `alphaEdH` | 2 |
| `alphaEeH` | 2 |
| `alphaEuH` | 2 |
| `alphaRHl3pp` | 1 |
| `alphaRHq3pp` | 1 |

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
| `g3` | 2242 |
| `g2` | 2237 |
| `alphaR2G` | 786 |
| `alphaR2W` | 786 |
| `alphaO3Gt` | 729 |
| `alphaO3Wt` | 729 |
| `alphaO3G` | 582 |
| `alphaO3W` | 582 |
| `g1` | 200 |
| `alphaRqD` | 72 |
| `alphaRDH` | 52 |
| `alphaRWDH` | 38 |
| `alphaOHGt` | 33 |
| `alphaOHWt` | 33 |
| `alphaRdD` | 32 |
| `alphaRuD` | 32 |
| `alphaRlD` | 32 |
| `alphaEGqp` | 24 |
| `alphaEGqtp` | 24 |
| `alphaEWqp` | 24 |

## Files

- `vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
