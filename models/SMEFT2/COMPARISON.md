# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-08-03` by `models.SMEFT2.comparison`.

## Scope

Signature coverage, coefficient-head content, and raw coefficient-head multiplicity diagnostics, plus exact symbolic comparison for all 184 FeynRules reference rows. Fermion exact comparison filters by indexed Wilson-coefficient head and keeps flavor order/conjugation in the canonical scalar coefficient, so it cannot pass vacuously for function-valued coefficients. Exact-symbolic rows are graded honestly: `EXACT_MATCH` means direct canonical-map equality with no row-specific packaging assumption; `MATCH_MODULO_CC_PACKAGING` means equality only after a charge-conjugation packaging transform whose sign/symmetry is derived (pinned), e.g. the antisymmetrized Weinberg rows; and `UNRESOLVED_CC_PACKAGING` means no pinned packaging rule is known or the pinned transform failed. The separate canonical tensor-map diagnostic remains the gauge-sector per-coefficient map for supported bosonic coefficient sectors.

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
| Shared raw head-count matches | 100 |
| Shared raw head-count mismatches | 82 |
| Shared raw head-count benign expansions | 82 |
| Shared raw head-count mismatches with unexplained deltas | 0 |
| Exact symbolic supported vertices | 184 |
| Direct exact symbolic matches | 176 |
| Exact modulo pinned CC packaging | 8 |
| Unresolved CC packaging (existence only) | 0 |
| Exact symbolic unequal vertices | 0 |
| Exact symbolic error vertices | 0 |
| Headline split | direct exact: 176/184; modulo pinned CC: 8/184; unresolved CC: 0/184; operator content: 184/184 |
| Compatibility alias `exact_symbolic_equal_vertices` (direct only) | 176 |
| Canonical tensor-map supported vertices | 32 |
| Canonical tensor-map equal vertices | 32 |
| Canonical tensor-map unequal vertices | 0 |
| Canonical tensor-map error vertices | 0 |
| Canonical tensor-map equal coefficient sectors | 93 |
| Canonical tensor-map unequal coefficient sectors | 0 |
| Canonical-map FeynPy raw monomials | 10472 |
| Canonical-map FeynPy canonical monomials | 3779 |
| Canonical-map FeynPy redundant monomials (raw - canonical) | 6693 |
| Explained benign head-count deltas | 285 |
| Unexplained head-count deltas | 0 |

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

This layer is enabled for every FeynRules reference row. Bosonic rows use the native bosonic comparator. Fermion rows parse the full FeynRules tensor rule into native tensors, filter terms by indexed Wilson-coefficient head, keep flavor order and complex conjugation in the scalar coefficient, and compare canonical tensor-monomial maps. Statuses are graded honestly: `EXACT_MATCH` is direct same-signature canonical equality; `MATCH_MODULO_CC_PACKAGING` is equality after a pinned charge-conjugation packaging transform (Weinberg or Ec partner rows); `UNRESOLVED_CC_PACKAGING` means no pinned packaging rule is known or the pinned transform failed.

| Signature | Status |
| --- | --- |
| `B|B|B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|B|B|dR|dRbar` | `EXACT_MATCH` |
| `B|B|B|eR|eRbar` | `EXACT_MATCH` |
| `B|B|B|lL|lLbar` | `EXACT_MATCH` |
| `B|B|B|qL|qLbar` | `EXACT_MATCH` |
| `B|B|B|uR|uRbar` | `EXACT_MATCH` |
| `B|B|G|dR|dRbar` | `EXACT_MATCH` |
| `B|B|G|qL|qLbar` | `EXACT_MATCH` |
| `B|B|G|uR|uRbar` | `EXACT_MATCH` |
| `B|B|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `B|B|Phibar|eRbar|lL` | `EXACT_MATCH` |
| `B|B|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `B|B|Phi|Phibar` | `EXACT_MATCH` |
| `B|B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|B|Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `B|B|Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `B|B|Phi|dR|qLbar` | `EXACT_MATCH` |
| `B|B|Phi|eR|lLbar` | `EXACT_MATCH` |
| `B|B|Phi|qL|uRbar` | `EXACT_MATCH` |
| `B|B|Wi|lL|lLbar` | `EXACT_MATCH` |
| `B|B|Wi|qL|qLbar` | `EXACT_MATCH` |
| `B|B|dR|dRbar` | `EXACT_MATCH` |
| `B|B|eR|eRbar` | `EXACT_MATCH` |
| `B|B|lL|lLbar` | `EXACT_MATCH` |
| `B|B|qL|qLbar` | `EXACT_MATCH` |
| `B|B|uR|uRbar` | `EXACT_MATCH` |
| `B|G|G|dR|dRbar` | `EXACT_MATCH` |
| `B|G|G|qL|qLbar` | `EXACT_MATCH` |
| `B|G|G|uR|uRbar` | `EXACT_MATCH` |
| `B|G|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `B|G|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `B|G|Phi|dR|qLbar` | `EXACT_MATCH` |
| `B|G|Phi|qL|uRbar` | `EXACT_MATCH` |
| `B|G|Wi|qL|qLbar` | `EXACT_MATCH` |
| `B|G|dR|dRbar` | `EXACT_MATCH` |
| `B|G|qL|qLbar` | `EXACT_MATCH` |
| `B|G|uR|uRbar` | `EXACT_MATCH` |
| `B|Phibar|Wi|dRbar|qL` | `EXACT_MATCH` |
| `B|Phibar|Wi|eRbar|lL` | `EXACT_MATCH` |
| `B|Phibar|Wi|qLbar|uR` | `EXACT_MATCH` |
| `B|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `B|Phibar|eRbar|lL` | `EXACT_MATCH` |
| `B|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `B|Phi|Phibar` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `B|Phi|Phibar|Wi|Wi|Wi` | `EXACT_MATCH` |
| `B|Phi|Phibar|dR|dRbar` | `EXACT_MATCH` |
| `B|Phi|Phibar|eR|eRbar` | `EXACT_MATCH` |
| `B|Phi|Phibar|lL|lLbar` | `EXACT_MATCH` |
| `B|Phi|Phibar|qL|qLbar` | `EXACT_MATCH` |
| `B|Phi|Phibar|uR|uRbar` | `EXACT_MATCH` |
| `B|Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `B|Phi|Phi|Phibar|Phibar|Wi` | `EXACT_MATCH` |
| `B|Phi|Wi|dR|qLbar` | `EXACT_MATCH` |
| `B|Phi|Wi|eR|lLbar` | `EXACT_MATCH` |
| `B|Phi|Wi|qL|uRbar` | `EXACT_MATCH` |
| `B|Phi|dR|qLbar` | `EXACT_MATCH` |
| `B|Phi|eR|lLbar` | `EXACT_MATCH` |
| `B|Phi|qL|uRbar` | `EXACT_MATCH` |
| `B|Wi|Wi|lL|lLbar` | `EXACT_MATCH` |
| `B|Wi|Wi|qL|qLbar` | `EXACT_MATCH` |
| `B|Wi|lL|lLbar` | `EXACT_MATCH` |
| `B|Wi|qL|qLbar` | `EXACT_MATCH` |
| `B|dR|dRbar` | `EXACT_MATCH` |
| `B|eR|eRbar` | `EXACT_MATCH` |
| `B|lL|lLbar` | `EXACT_MATCH` |
| `B|qL|qLbar` | `EXACT_MATCH` |
| `B|uR|uRbar` | `EXACT_MATCH` |
| `G|G|G` | `EXACT_MATCH` |
| `G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|G|G` | `EXACT_MATCH` |
| `G|G|G|G|Phi|Phibar` | `EXACT_MATCH` |
| `G|G|G|Phi|Phibar` | `EXACT_MATCH` |
| `G|G|G|dR|dRbar` | `EXACT_MATCH` |
| `G|G|G|qL|qLbar` | `EXACT_MATCH` |
| `G|G|G|uR|uRbar` | `EXACT_MATCH` |
| `G|G|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `G|G|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `G|G|Phi|Phibar` | `EXACT_MATCH` |
| `G|G|Phi|dR|qLbar` | `EXACT_MATCH` |
| `G|G|Phi|qL|uRbar` | `EXACT_MATCH` |
| `G|G|Wi|qL|qLbar` | `EXACT_MATCH` |
| `G|G|dR|dRbar` | `EXACT_MATCH` |
| `G|G|qL|qLbar` | `EXACT_MATCH` |
| `G|G|uR|uRbar` | `EXACT_MATCH` |
| `G|Phibar|Wi|dRbar|qL` | `EXACT_MATCH` |
| `G|Phibar|Wi|qLbar|uR` | `EXACT_MATCH` |
| `G|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `G|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `G|Phi|Phibar|dR|dRbar` | `EXACT_MATCH` |
| `G|Phi|Phibar|qL|qLbar` | `EXACT_MATCH` |
| `G|Phi|Phibar|uR|uRbar` | `EXACT_MATCH` |
| `G|Phi|Wi|dR|qLbar` | `EXACT_MATCH` |
| `G|Phi|Wi|qL|uRbar` | `EXACT_MATCH` |
| `G|Phi|dR|qLbar` | `EXACT_MATCH` |
| `G|Phi|qL|uRbar` | `EXACT_MATCH` |
| `G|Wi|Wi|qL|qLbar` | `EXACT_MATCH` |
| `G|Wi|qL|qLbar` | `EXACT_MATCH` |
| `G|dR|dRbar` | `EXACT_MATCH` |
| `G|qL|qLbar` | `EXACT_MATCH` |
| `G|uR|uRbar` | `EXACT_MATCH` |
| `Phibar|Phibar|Wi|dRbar|uR` | `EXACT_MATCH` |
| `Phibar|Phibar|dRbar|uR` | `EXACT_MATCH` |
| `Phibar|Phibar|lLbar|lLbar` | `MATCH_MODULO_CC_PACKAGING` |
| `Phibar|Wi|Wi|dRbar|qL` | `EXACT_MATCH` |
| `Phibar|Wi|Wi|eRbar|lL` | `EXACT_MATCH` |
| `Phibar|Wi|Wi|qLbar|uR` | `EXACT_MATCH` |
| `Phibar|Wi|dRbar|qL` | `EXACT_MATCH` |
| `Phibar|Wi|eRbar|lL` | `EXACT_MATCH` |
| `Phibar|Wi|qLbar|uR` | `EXACT_MATCH` |
| `Phibar|dRbar|qL` | `EXACT_MATCH` |
| `Phibar|eRbar|lL` | `EXACT_MATCH` |
| `Phibar|qLbar|uR` | `EXACT_MATCH` |
| `Phi|Phibar|Phibar|dRbar|qL` | `EXACT_MATCH` |
| `Phi|Phibar|Phibar|eRbar|lL` | `EXACT_MATCH` |
| `Phi|Phibar|Phibar|qLbar|uR` | `EXACT_MATCH` |
| `Phi|Phibar|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|dR|dRbar` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|eR|eRbar` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|lL|lLbar` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|qL|qLbar` | `EXACT_MATCH` |
| `Phi|Phibar|Wi|uR|uRbar` | `EXACT_MATCH` |
| `Phi|Phibar|dR|dRbar` | `EXACT_MATCH` |
| `Phi|Phibar|eR|eRbar` | `EXACT_MATCH` |
| `Phi|Phibar|lL|lLbar` | `EXACT_MATCH` |
| `Phi|Phibar|qL|qLbar` | `EXACT_MATCH` |
| `Phi|Phibar|uR|uRbar` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar|Wi` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|Phibar|Wi|Wi` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|dR|qLbar` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|eR|lLbar` | `EXACT_MATCH` |
| `Phi|Phi|Phibar|qL|uRbar` | `EXACT_MATCH` |
| `Phi|Phi|Phi|Phibar|Phibar|Phibar` | `EXACT_MATCH` |
| `Phi|Phi|Wi|dR|uRbar` | `EXACT_MATCH` |
| `Phi|Phi|dR|uRbar` | `EXACT_MATCH` |
| `Phi|Phi|lL|lL` | `MATCH_MODULO_CC_PACKAGING` |
| `Phi|Wi|Wi|dR|qLbar` | `EXACT_MATCH` |
| `Phi|Wi|Wi|eR|lLbar` | `EXACT_MATCH` |
| `Phi|Wi|Wi|qL|uRbar` | `EXACT_MATCH` |
| `Phi|Wi|dR|qLbar` | `EXACT_MATCH` |
| `Phi|Wi|eR|lLbar` | `EXACT_MATCH` |
| `Phi|Wi|qL|uRbar` | `EXACT_MATCH` |
| `Phi|dR|qLbar` | `EXACT_MATCH` |
| `Phi|eR|lLbar` | `EXACT_MATCH` |
| `Phi|qL|uRbar` | `EXACT_MATCH` |
| `Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `EXACT_MATCH` |
| `Wi|Wi|Wi|lL|lLbar` | `EXACT_MATCH` |
| `Wi|Wi|Wi|qL|qLbar` | `EXACT_MATCH` |
| `Wi|Wi|lL|lLbar` | `EXACT_MATCH` |
| `Wi|Wi|qL|qLbar` | `EXACT_MATCH` |
| `Wi|lL|lLbar` | `EXACT_MATCH` |
| `Wi|qL|qLbar` | `EXACT_MATCH` |
| `dRbar|eR|lLbar|qL` | `MATCH_MODULO_CC_PACKAGING` |
| `dRbar|qL|qL|uRbar` | `MATCH_MODULO_CC_PACKAGING` |
| `dR|dRbar|eR|eRbar` | `EXACT_MATCH` |
| `dR|dRbar|lL|lLbar` | `EXACT_MATCH` |
| `dR|dRbar|qL|qLbar` | `EXACT_MATCH` |
| `dR|dRbar|uR|uRbar` | `EXACT_MATCH` |
| `dR|dR|dRbar|dRbar` | `EXACT_MATCH` |
| `dR|eRbar|lL|qLbar` | `MATCH_MODULO_CC_PACKAGING` |
| `dR|qLbar|qLbar|uR` | `MATCH_MODULO_CC_PACKAGING` |
| `eRbar|lL|qL|uRbar` | `MATCH_MODULO_CC_PACKAGING` |
| `eR|eRbar|lL|lLbar` | `EXACT_MATCH` |
| `eR|eRbar|qL|qLbar` | `EXACT_MATCH` |
| `eR|eRbar|uR|uRbar` | `EXACT_MATCH` |
| `eR|eR|eRbar|eRbar` | `EXACT_MATCH` |
| `eR|lLbar|qLbar|uR` | `MATCH_MODULO_CC_PACKAGING` |
| `lL|lLbar|qL|qLbar` | `EXACT_MATCH` |
| `lL|lLbar|uR|uRbar` | `EXACT_MATCH` |
| `lL|lL|lLbar|lLbar` | `EXACT_MATCH` |
| `qL|qLbar|uR|uRbar` | `EXACT_MATCH` |
| `qL|qL|qLbar|qLbar` | `EXACT_MATCH` |
| `uR|uR|uRbar|uRbar` | `EXACT_MATCH` |

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
| `B|B|Phibar|dRbar|qL` | `alphaEdH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phibar|dRbar|qL` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phibar|eRbar|lL` | `alphaEeH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phibar|eRbar|lL` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phibar|qLbar|uR` | `alphaEuH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phibar|qLbar|uR` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|Phibar` | `alphaOHBt` | 2 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|Phibar|Wi|Wi` | `alphaRDH` | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|Phibar|Wi|Wi` | `g1` | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|Phibar|Wi|Wi` | `g2` | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|dR|qLbar` | `alphaEdH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|dR|qLbar` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|eR|lLbar` | `alphaEeH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|eR|lLbar` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|qL|uRbar` | `alphaEuH` | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|Phi|qL|uRbar` | `g1` | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|dR|dRbar` | `alphaEBdtp` | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|dR|dRbar` | `alphaRBdtp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|dR|dRbar` | `g1` | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|eR|eRbar` | `alphaEBetp` | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|eR|eRbar` | `alphaRBetp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|eR|eRbar` | `g1` | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|lL|lLbar` | `alphaEBltp` | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|lL|lLbar` | `alphaRBltp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|lL|lLbar` | `g1` | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|qL|qLbar` | `alphaEBqtp` | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|qL|qLbar` | `alphaRBqtp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|qL|qLbar` | `g1` | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|uR|uRbar` | `alphaEButp` | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|uR|uRbar` | `alphaRButp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|B|uR|uRbar` | `g1` | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `alphaEGdp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `alphaEGdtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `alphaRGdtp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `alphaRdD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `g1` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|dR|dRbar` | `g3` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `alphaEGqp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `alphaEGqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `alphaRGqtp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `alphaRqD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `g1` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|qL|qLbar` | `g3` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `alphaEGup` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `alphaEGutp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `alphaRGutp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `alphaRuD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `g1` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|G|uR|uRbar` | `g3` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|dR|dRbar` | `alphaEBdtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|dR|dRbar` | `g3` | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|qL|qLbar` | `alphaEBqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|qL|qLbar` | `g3` | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|uR|uRbar` | `alphaEButp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|G|uR|uRbar` | `g3` | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phibar|dRbar|qL` | `alphaEdB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phibar|eRbar|lL` | `alphaEeB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phibar|qLbar|uR` | `alphaEuB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|Phibar|Wi|Wi` | `alphaOHWB` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi` | `alphaOHWBt` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi` | `alphaRDH` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi` | `alphaRWDH` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi` | `g1` | 18 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi` | `g2` | 22 | 40 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi|Wi` | `alphaRDH` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi|Wi` | `g1` | 18 | 30 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|Phibar|Wi|Wi|Wi` | `g2` | 18 | 30 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Phi|dR|qLbar` | `alphaEdB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|eR|lLbar` | `alphaEeB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Phi|qL|uRbar` | `alphaEuB` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|Wi|Wi|lL|lLbar` | `alphaEWlp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|lL|lLbar` | `alphaEWltp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|lL|lLbar` | `alphaRWltp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|lL|lLbar` | `alphaRlD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|lL|lLbar` | `g1` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|lL|lLbar` | `g2` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `alphaEWqp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `alphaEWqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `alphaRWqtp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `alphaRqD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `g1` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|Wi|qL|qLbar` | `g2` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|lL|lLbar` | `alphaEBltp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|lL|lLbar` | `g2` | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|qL|qLbar` | `alphaEBqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|Wi|qL|qLbar` | `g2` | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|dR|dRbar` | `alphaEBd` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|dR|dRbar` | `alphaEBdtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|dR|dRbar` | `alphaRBdtp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|dR|dRbar` | `alphaRdD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|dR|dRbar` | `g1` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|eR|eRbar` | `alphaEBe` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|eR|eRbar` | `alphaEBetp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|eR|eRbar` | `alphaRBetp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|eR|eRbar` | `alphaReD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|eR|eRbar` | `g1` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|lL|lLbar` | `alphaEBl` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|lL|lLbar` | `alphaEBltp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|lL|lLbar` | `alphaRBltp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|lL|lLbar` | `alphaRlD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|lL|lLbar` | `g1` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|qL|qLbar` | `alphaEBq` | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaEBqtp` | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaRBqtp` | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| `B|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `B|qL|qLbar` | `g1` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `B|uR|uRbar` | `alphaEBu` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|uR|uRbar` | `alphaEButp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|uR|uRbar` | `alphaRButp` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|uR|uRbar` | `alphaRuD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `B|uR|uRbar` | `g1` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G` | `alphaO3G` | 8 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G` | `alphaO3Gt` | 42 | 21 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G` | `alphaR2G` | 36 | 54 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G` | `g3` | 42 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G` | `alphaO3G` | 48 | 144 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G` | `alphaO3Gt` | 186 | 138 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G` | `alphaR2G` | 156 | 204 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G` | `g3` | 396 | 492 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G` | `alphaO3G` | 240 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G` | `alphaO3Gt` | 420 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G` | `alphaR2G` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G` | `g3` | 1020 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G|G` | `alphaO3Gt` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G|G` | `alphaR2G` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|G|G` | `g3` | 1440 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|Phi|Phibar` | `alphaOHG` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|Phi|Phibar` | `alphaOHGt` | 3 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|G|Phi|Phibar` | `g3` | 9 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|Phi|Phibar` | `alphaOHG` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|Phi|Phibar` | `alphaOHGt` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|Phi|Phibar` | `g3` | 18 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `alphaEGd` | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `alphaEGdp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `alphaEGdtp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `alphaRGdtp` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `alphaRdD` | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|dR|dRbar` | `g3` | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `alphaEGq` | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `alphaEGqp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `alphaEGqtp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `alphaRGqtp` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `alphaRqD` | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|qL|qLbar` | `g3` | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `alphaEGu` | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `alphaEGup` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `alphaEGutp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `alphaRGutp` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `alphaRuD` | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|G|uR|uRbar` | `g3` | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|dRbar|qL` | `alphaEdG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|dRbar|qL` | `alphaOdG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|dRbar|qL` | `g3` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|qLbar|uR` | `alphaEuG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|qLbar|uR` | `alphaOuG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phibar|qLbar|uR` | `g3` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|dR|qLbar` | `alphaEdG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|dR|qLbar` | `alphaOdG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|dR|qLbar` | `g3` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|qL|uRbar` | `alphaEuG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|qL|uRbar` | `alphaOuG` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Phi|qL|uRbar` | `g3` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `alphaEGqp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `alphaEGqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `alphaRGqtp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `alphaRqD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `g2` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|Wi|qL|qLbar` | `g3` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaEGd` | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaEGdp` | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaEGdtp` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaRGd` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaRGdtp` | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `alphaRdD` | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|dR|dRbar` | `g3` | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaEGq` | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaEGqp` | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaEGqtp` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaRGq` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaRGqtp` | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `alphaRqD` | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|qL|qLbar` | `g3` | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaEGu` | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaEGup` | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaEGutp` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaRGu` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaRGutp` | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `alphaRuD` | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|G|uR|uRbar` | `g3` | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `alphaEWqp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `alphaEWqtp` | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `alphaRWqtp` | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `alphaRqD` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `g2` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|Wi|Wi|qL|qLbar` | `g3` | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|dR|dRbar` | `alphaRdD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|dR|dRbar` | `g3` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `G|qL|qLbar` | `g3` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `G|uR|uRbar` | `alphaRuD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `G|uR|uRbar` | `g3` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|dRbar|qL` | `alphaEdW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|dRbar|qL` | `alphaOdW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|dRbar|qL` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|eRbar|lL` | `alphaEeW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|eRbar|lL` | `alphaOeW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|eRbar|lL` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|qLbar|uR` | `alphaEuW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|qLbar|uR` | `alphaOuW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phibar|Wi|Wi|qLbar|uR` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi` | `alphaRDH` | 12 | 20 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi` | `alphaRWDH` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi` | `g2` | 34 | 46 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi` | `alphaOHW` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi` | `alphaOHWt` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi` | `alphaRDH` | 24 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi` | `alphaRWDH` | 48 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi` | `g2` | 90 | 132 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `alphaOHW` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `alphaOHWt` | 3 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Phibar|Wi|Wi|Wi|Wi` | `g2` | 81 | 108 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|dR|qLbar` | `alphaEdW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|dR|qLbar` | `alphaOdW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|dR|qLbar` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|eR|lLbar` | `alphaEeW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|eR|lLbar` | `alphaOeW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|eR|lLbar` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|qL|uRbar` | `alphaEuW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|qL|uRbar` | `alphaOuW` | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Phi|Wi|Wi|qL|uRbar` | `g2` | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi` | `alphaO3W` | 8 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi` | `alphaO3Wt` | 42 | 21 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi` | `alphaR2W` | 36 | 54 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi` | `g2` | 42 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi` | `alphaO3W` | 48 | 144 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi` | `alphaO3Wt` | 186 | 138 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi` | `alphaR2W` | 156 | 204 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi` | `g2` | 396 | 492 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi` | `alphaO3W` | 240 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi` | `alphaO3Wt` | 420 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi` | `alphaR2W` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi` | `g2` | 1020 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `alphaO3Wt` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `alphaR2W` | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|Wi|Wi|Wi` | `g2` | 1440 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|lL|lLbar` | `alphaEWlp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|lL|lLbar` | `alphaEWltp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|lL|lLbar` | `alphaRWltp` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|lL|lLbar` | `alphaRlD` | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|lL|lLbar` | `g2` | 102 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|qL|qLbar` | `alphaEWqp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|qL|qLbar` | `alphaEWqtp` | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|qL|qLbar` | `alphaRWqtp` | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|qL|qLbar` | `alphaRqD` | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|Wi|qL|qLbar` | `g2` | 102 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaEWl` | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaEWlp` | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaEWltp` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaRWl` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaRWltp` | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `alphaRlD` | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|lL|lLbar` | `g2` | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaEWq` | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaEWqp` | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaEWqtp` | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaRWq` | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaRWqtp` | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `alphaRqD` | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|Wi|qL|qLbar` | `g2` | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|lL|lLbar` | `alphaRlD` | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|lL|lLbar` | `g2` | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| `Wi|qL|qLbar` | `alphaRqD` | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `Wi|qL|qLbar` | `g2` | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| `dRbar|eR|lLbar|qL` | `alphaEcqedl` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dRbar|eR|lLbar|qL` | `alphaEcqedlthree` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dRbar|qL|qL|uRbar` | `alphaEcudqq` | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dRbar|qL|qL|uRbar` | `alphaEcudqqtwo` | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dR|eRbar|lL|qLbar` | `alphaEcqedl` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dR|eRbar|lL|qLbar` | `alphaEcqedlthree` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dR|qLbar|qLbar|uR` | `alphaEcudqq` | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `dR|qLbar|qLbar|uR` | `alphaEcudqqtwo` | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `eRbar|lL|qL|uRbar` | `alphaEcuelq` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `eRbar|lL|qL|uRbar` | `alphaEcuelqtwo` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `eR|lLbar|qLbar|uR` | `alphaEcuelq` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| `eR|lLbar|qLbar|uR` | `alphaEcuelqtwo` | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |

## Largest Unexplained Raw Head-Count Deltas

These exclude the explicit benign expansions listed above. The large pure-gauge raw deltas can remain large even where the canonical tensor-map comparison above proves equality.

| Head | Total absolute delta |
| --- | ---: |

## Files

- `comparison/artifacts/vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `comparison/artifacts/feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
