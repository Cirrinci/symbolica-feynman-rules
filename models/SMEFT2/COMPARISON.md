# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-07-21` by `models/SMEFT2/comparison.py`.

## Scope

Signature coverage, coefficient-head content, and raw coefficient-head multiplicity diagnostics. Full tensor-rule equality is not claimed by this SMEFT2 report.

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

These exclude the explicit benign expansions listed above. The large `G^5`/`W^5` deltas are still left as expansion noise until they are reduced or checked against a stronger tensor-level oracle.

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
