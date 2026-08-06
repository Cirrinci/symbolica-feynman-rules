# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-08-06` by `models.SMEFT2.comparison`.

## Scope

Signature coverage, coefficient-head content, and raw coefficient-head multiplicity diagnostics, plus exact symbolic comparison for all 184 FeynRules reference rows. Fermion exact comparison filters by indexed Wilson-coefficient head and keeps flavor order/conjugation in the canonical scalar coefficient, so it cannot pass vacuously for function-valued coefficients. Exact-symbolic rows are graded honestly: `EXACT_MATCH` means direct canonical-map equality with no row-specific packaging assumption; `MATCH_MODULO_CC_PACKAGING` means equality only after a charge-conjugation packaging transform whose sign/symmetry is derived (pinned), e.g. the antisymmetrized Weinberg rows; and `UNRESOLVED_CC_PACKAGING` means no pinned packaging rule is known or the pinned transform failed. The separate canonical tensor-map diagnostic remains the bosonic-sector per-coefficient map for supported bosonic coefficient sectors.

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

## Sector-by-Sector Reading Guide

This table explains what the comparison did to put each sector in the same mathematical form before equality was tested. Direct exact rows compare the same external-field signature. Pinned CC rows compare an explicitly listed charge-conjugation partner with a fixed phase and duplicate-leg symmetry.

| Sector family | Rows | Result | Normalization/canonicalization used |
| --- | ---: | --- | --- |
| Bosonic and Higgs/gauge | 32 | 32 direct exact | Parse FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu2`, `fsu3`; expand dual field strengths; use metric symmetry, epsilon antisymmetry, structure-constant antisymmetry, dummy-index relabeling, generator-product ordering, and the narrow `f*f` Jacobi reducer. |
| Two-fermion non-Weinberg | 129 | 129 direct exact | Parse gamma chains, slashed momenta, projectors, generators, index deltas, epsilons, and indexed Wilson functions; keep flavor order/conjugation in the scalar coefficient; canonicalize open spinor, Lorentz, color, and weak tensors; apply narrow SU(2) pseudoreality identities for Higgs-tilde/generator products. |
| Weinberg | 2 | 2 pinned CC | FeynRules emits same-chirality `Phi Phi lL lL` and HC rows; FeynPy stores mixed `lLbar,lL` rows with explicit `dirac_C`. The accepted transform is the antisymmetrized local pair `FeynPy(lLbar,lL) - FeynPy(lL,lLbar)`, with the sign fixed by `C^T = -C`. |
| Ordinary four-fermion | 15 | 15 direct exact | Preserve all four Wilson flavor slots; canonicalize color singlet/octet contractions, weak triplet currents, identical fermion dummy labels, gamma chains, and Hermitian-conjugate generator orientations. |
| Charge-conjugated evanescent four-fermion | 6 rows / 12 coefficient sectors | 6 pinned CC | Use the pinned `alphaEc*` rule table: exactly one partner signature, one phase, and one symmetric or antisymmetric duplicate-leg rule per coefficient sector; rewrite explicit `dirac_C` arms into FeynRules `CC[...]` flow and then demand canonical-map equality. |
| FeynPy-only zero-signature artifacts | 2 local signatures | dropped from residuals | Canonical coefficient-head collection proves the apparent signatures cancel to zero under tensor symmetries, so they are diagnostics rather than unmatched operator content. |

<details>
<summary>Show exact symbolic status table (184 rows)</summary>

| Signature | Status |
| --- | --- |
| <code>B&#124;B&#124;B&#124;B&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;B&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phibar&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;B&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;Wi&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;Wi&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;Wi&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phibar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Wi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Wi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;Wi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>B&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;G</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;G&#124;G&#124;G</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Phi&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phibar&#124;Wi&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phibar&#124;Wi&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;Phibar&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;Phibar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;Phibar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;Wi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;Wi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>G&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Phibar&#124;Wi&#124;dRbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Phibar&#124;dRbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Phibar&#124;lLbar&#124;lLbar</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>Phibar&#124;Wi&#124;Wi&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Wi&#124;Wi&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Wi&#124;Wi&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Wi&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Wi&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;Wi&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Phibar&#124;dRbar&#124;qL</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Phibar&#124;eRbar&#124;lL</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Phibar&#124;qLbar&#124;uR</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;Wi&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;dR&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phibar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phibar&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Phibar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;Wi&#124;dR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;dR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Phi&#124;lL&#124;lL</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>Phi&#124;Wi&#124;Wi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Wi&#124;Wi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Wi&#124;Wi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Wi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Wi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;Wi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;dR&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;eR&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Phi&#124;qL&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>Wi&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>dRbar&#124;eR&#124;lLbar&#124;qL</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>dRbar&#124;qL&#124;qL&#124;uRbar</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>dR&#124;dRbar&#124;eR&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>dR&#124;dRbar&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>dR&#124;dRbar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>dR&#124;dRbar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>dR&#124;dR&#124;dRbar&#124;dRbar</code> | <code>EXACT_MATCH</code> |
| <code>dR&#124;eRbar&#124;lL&#124;qLbar</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>dR&#124;qLbar&#124;qLbar&#124;uR</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>eRbar&#124;lL&#124;qL&#124;uRbar</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>eR&#124;eRbar&#124;lL&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>eR&#124;eRbar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>eR&#124;eRbar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>eR&#124;eR&#124;eRbar&#124;eRbar</code> | <code>EXACT_MATCH</code> |
| <code>eR&#124;lLbar&#124;qLbar&#124;uR</code> | <code>MATCH_MODULO_CC_PACKAGING</code> |
| <code>lL&#124;lLbar&#124;qL&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>lL&#124;lLbar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>lL&#124;lL&#124;lLbar&#124;lLbar</code> | <code>EXACT_MATCH</code> |
| <code>qL&#124;qLbar&#124;uR&#124;uRbar</code> | <code>EXACT_MATCH</code> |
| <code>qL&#124;qL&#124;qLbar&#124;qLbar</code> | <code>EXACT_MATCH</code> |
| <code>uR&#124;uR&#124;uRbar&#124;uRbar</code> | <code>EXACT_MATCH</code> |

</details>

## Canonical Tensor-Map Gauge Comparison

This diagnostic is enabled for supported bosonic rows. It parses FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu3`, and `fsu2` into native tensors, then compares canonical monomial maps per Wilson coefficient. It uses intrinsic tensor symmetries, dummy-index relabeling, commuting factor ordering, exact coefficient collection, generator-product ordering, SU(2) pseudoreality normalization, and the narrow `f*f` Jacobi reducer. It does not use momentum conservation, EOM, IBP, Schouten/Fierz identities, or broad 4D gamma reductions.

<details>
<summary>Show canonical tensor-map table (32 rows)</summary>

| Signature | Status | Coefficient sectors |
| --- | --- | --- |
| <code>B&#124;B&#124;B&#124;B&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 3/3 -> canonical 3/3 |
| <code>B&#124;B&#124;B&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 6/6 -> canonical 6/6 |
| <code>B&#124;B&#124;B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 3/3 -> canonical 3/3 |
| <code>B&#124;B&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKH` match: raw 1/1 -> canonical 1/1; `alphaOHB` match: raw 2/2 -> canonical 2/2; `alphaOHBt` match: raw 8/2 -> canonical 1/1; `alphaRBDH` match: raw 4/4 -> canonical 4/4; `alphaRDH` match: raw 9/9 -> canonical 9/9 |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 6/6 -> canonical 6/6 |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 14/6 -> canonical 6/6 |
| <code>B&#124;B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHD` match: raw 2/2 -> canonical 2/2; `alphaRHDp` match: raw 2/2 -> canonical 2/2 |
| <code>B&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaRBDH` match: raw 4/4 -> canonical 4/4; `alphaRDH` match: raw 4/4 -> canonical 4/4 |
| <code>B&#124;Phi&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKH` match: raw 1/1 -> canonical 1/1; `alphaOHWB` match: raw 2/2 -> canonical 2/2; `alphaOHWBt` match: raw 4/4 -> canonical 1/1; `alphaRBDH` match: raw 2/2 -> canonical 2/2; `alphaRDH` match: raw 9/9 -> canonical 9/9; `alphaRWDH` match: raw 2/2 -> canonical 2/2 |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHWB` match: raw 4/2 -> canonical 2/2; `alphaOHWBt` match: raw 4/2 -> canonical 1/1; `alphaRDH` match: raw 24/12 -> canonical 12/12; `alphaRWDH` match: raw 8/6 -> canonical 6/6 |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaRDH` match: raw 24/12 -> canonical 12/12; `alphaRWDH` match: raw 6/6 -> canonical 6/6 |
| <code>B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHD` match: raw 8/8 -> canonical 8/8; `alphaRHDp` match: raw 8/8 -> canonical 8/8; `alphaRHDpp` match: raw 10/10 -> canonical 10/10 |
| <code>B&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaRHDp` match: raw 4/4 -> canonical 4/4 |
| <code>G&#124;G&#124;G</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 14/8 -> canonical 8/8; `alphaO3Gt` match: raw 21/42 -> canonical 12/12; `alphaR2G` match: raw 54/36 -> canonical 36/36 |
| <code>G&#124;G&#124;G&#124;G</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKG` match: raw 6/6 -> canonical 6/6; `alphaO3G` match: raw 144/48 -> canonical 48/48; `alphaO3Gt` match: raw 138/186 -> canonical 72/72; `alphaR2G` match: raw 204/156 -> canonical 156/156 |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaO3G` match: raw 720/240 -> canonical 120/120; `alphaO3Gt` match: raw 720/420 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| <code>G&#124;G&#124;G&#124;G&#124;G&#124;G</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaO3G` match: raw 720/720 -> canonical 120/120; `alphaO3Gt` match: raw 720/360 -> canonical 180/180; `alphaR2G` match: raw 720/360 -> canonical 360/360 |
| <code>G&#124;G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHG` match: raw 12/6 -> canonical 6/6; `alphaOHGt` match: raw 24/3 -> canonical 3/3 |
| <code>G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHG` match: raw 12/6 -> canonical 6/6; `alphaOHGt` match: raw 24/12 -> canonical 3/3 |
| <code>G&#124;G&#124;Phi&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHG` match: raw 2/2 -> canonical 2/2; `alphaOHGt` match: raw 8/8 -> canonical 1/1 |
| <code>Phi&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaRDH` match: raw 4/4 -> canonical 4/4; `alphaRWDH` match: raw 4/4 -> canonical 4/4 |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKH` match: raw 2/2 -> canonical 2/2; `alphaOHW` match: raw 2/2 -> canonical 2/2; `alphaOHWt` match: raw 8/8 -> canonical 1/1; `alphaRDH` match: raw 20/12 -> canonical 12/12; `alphaRWDH` match: raw 24/20 -> canonical 20/20 |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHW` match: raw 12/6 -> canonical 6/6; `alphaOHWt` match: raw 24/12 -> canonical 3/3; `alphaRDH` match: raw 36/24 -> canonical 24/24; `alphaRWDH` match: raw 60/48 -> canonical 48/48 |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHW` match: raw 12/6 -> canonical 6/6; `alphaOHWt` match: raw 24/3 -> canonical 3/3; `alphaRDH` match: raw 24/24 -> canonical 24/24; `alphaRWDH` match: raw 48/48 -> canonical 48/48 |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHBox` match: raw 12/12 -> canonical 12/12; `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaOlambda` match: raw 2/2 -> canonical 2/2; `alphaRHDp` match: raw 4/4 -> canonical 4/4; `alphaRHDpp` match: raw 8/8 -> canonical 8/8 |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHD` match: raw 8/8 -> canonical 8/8; `alphaRHDp` match: raw 8/8 -> canonical 8/8; `alphaRHDpp` match: raw 12/12 -> canonical 12/12 |
| <code>Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOHD` match: raw 4/4 -> canonical 4/4; `alphaRHDp` match: raw 8/8 -> canonical 8/8 |
| <code>Phi&#124;Phi&#124;Phi&#124;Phibar&#124;Phibar&#124;Phibar</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaOH` match: raw 6/6 -> canonical 6/6 |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 14/8 -> canonical 8/8; `alphaO3Wt` match: raw 21/42 -> canonical 12/12; `alphaR2W` match: raw 54/36 -> canonical 36/36 |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaKW` match: raw 6/6 -> canonical 6/6; `alphaO3W` match: raw 144/48 -> canonical 48/48; `alphaO3Wt` match: raw 138/186 -> canonical 72/72; `alphaR2W` match: raw 204/156 -> canonical 156/156 |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaO3W` match: raw 720/240 -> canonical 120/120; `alphaO3Wt` match: raw 720/420 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>CANONICAL_MAP_MATCH</code> | `alphaO3W` match: raw 720/720 -> canonical 120/120; `alphaO3Wt` match: raw 720/360 -> canonical 180/180; `alphaR2W` match: raw 720/360 -> canonical 360/360 |

</details>

## Largest Reference-Side Head Gaps

| Head | Count |
| --- | ---: |

## Largest Local Extra Heads

| Head | Count |
| --- | ---: |

## Explained Benign Raw Head-Count Deltas

These are raw coefficient-head occurrence-count diagnostics. They catch some missing or duplicated content, but they are not tensor-rule equality proofs because equivalent algebra can be printed with different occurrence counts.

<details>
<summary>Show explained raw head-count delta table (285 rows)</summary>

| Signature | Head | Reference | FeynPy | Reason |
| --- | --- | ---: | ---: | --- |
| <code>B&#124;B&#124;Phibar&#124;dRbar&#124;qL</code> | <code>alphaEdH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phibar&#124;dRbar&#124;qL</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phibar&#124;eRbar&#124;lL</code> | <code>alphaEeH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phibar&#124;eRbar&#124;lL</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phibar&#124;qLbar&#124;uR</code> | <code>alphaEuH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phibar&#124;qLbar&#124;uR</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;Phibar</code> | <code>alphaOHBt</code> | 2 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaRDH</code> | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>g1</code> | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>g2</code> | 6 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;dR&#124;qLbar</code> | <code>alphaEdH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;dR&#124;qLbar</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;eR&#124;lLbar</code> | <code>alphaEeH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;eR&#124;lLbar</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;qL&#124;uRbar</code> | <code>alphaEuH</code> | 0 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;Phi&#124;qL&#124;uRbar</code> | <code>g1</code> | 3 | 7 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;dR&#124;dRbar</code> | <code>alphaEBdtp</code> | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;dR&#124;dRbar</code> | <code>alphaRBdtp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;dR&#124;dRbar</code> | <code>g1</code> | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;eR&#124;eRbar</code> | <code>alphaEBetp</code> | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;eR&#124;eRbar</code> | <code>alphaRBetp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;eR&#124;eRbar</code> | <code>g1</code> | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;lL&#124;lLbar</code> | <code>alphaEBltp</code> | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;lL&#124;lLbar</code> | <code>alphaRBltp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;lL&#124;lLbar</code> | <code>g1</code> | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;qL&#124;qLbar</code> | <code>alphaEBqtp</code> | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;qL&#124;qLbar</code> | <code>alphaRBqtp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;qL&#124;qLbar</code> | <code>g1</code> | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;uR&#124;uRbar</code> | <code>alphaEButp</code> | 8 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;uR&#124;uRbar</code> | <code>alphaRButp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;B&#124;uR&#124;uRbar</code> | <code>g1</code> | 31 | 41 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRGdtp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRdD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>g1</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>g3</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRGqtp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>g1</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>g3</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGup</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGutp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRGutp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRuD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>g1</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>g3</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEBdtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;dR&#124;dRbar</code> | <code>g3</code> | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEBqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;qL&#124;qLbar</code> | <code>g3</code> | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEButp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;G&#124;uR&#124;uRbar</code> | <code>g3</code> | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phibar&#124;dRbar&#124;qL</code> | <code>alphaEdB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Phibar&#124;eRbar&#124;lL</code> | <code>alphaEeB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Phibar&#124;qLbar&#124;uR</code> | <code>alphaEuB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaOHWB</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaOHWBt</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaRDH</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaRWDH</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>g1</code> | 18 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>g2</code> | 22 | 40 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaRDH</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>g1</code> | 18 | 30 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 18 | 30 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Phi&#124;dR&#124;qLbar</code> | <code>alphaEdB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Phi&#124;eR&#124;lLbar</code> | <code>alphaEeB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Phi&#124;qL&#124;uRbar</code> | <code>alphaEuB</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWlp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWltp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRWltp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRlD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>g1</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>g2</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRWqtp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g1</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEBltp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;lL&#124;lLbar</code> | <code>g2</code> | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEBqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 21 | 25 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;dR&#124;dRbar</code> | <code>alphaEBd</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;dR&#124;dRbar</code> | <code>alphaEBdtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;dR&#124;dRbar</code> | <code>alphaRBdtp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;dR&#124;dRbar</code> | <code>alphaRdD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;dR&#124;dRbar</code> | <code>g1</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;eR&#124;eRbar</code> | <code>alphaEBe</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;eR&#124;eRbar</code> | <code>alphaEBetp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;eR&#124;eRbar</code> | <code>alphaRBetp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;eR&#124;eRbar</code> | <code>alphaReD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;eR&#124;eRbar</code> | <code>g1</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;lL&#124;lLbar</code> | <code>alphaEBl</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;lL&#124;lLbar</code> | <code>alphaEBltp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;lL&#124;lLbar</code> | <code>alphaRBltp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;lL&#124;lLbar</code> | <code>alphaRlD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;lL&#124;lLbar</code> | <code>g1</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;qL&#124;qLbar</code> | <code>alphaEBq</code> | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;qL&#124;qLbar</code> | <code>alphaEBqtp</code> | 4 | 8 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;qL&#124;qLbar</code> | <code>alphaRBqtp</code> | 2 | 4 | FeynPy prints the two antisymmetric branches from `Dual[FS] = 1/2 epsilon.FS` separately; FeynRules has already collapsed them with epsilon antisymmetry. |
| <code>B&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>B&#124;qL&#124;qLbar</code> | <code>g1</code> | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>B&#124;uR&#124;uRbar</code> | <code>alphaEBu</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;uR&#124;uRbar</code> | <code>alphaEButp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;uR&#124;uRbar</code> | <code>alphaRButp</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;uR&#124;uRbar</code> | <code>alphaRuD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>B&#124;uR&#124;uRbar</code> | <code>g1</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G</code> | <code>alphaO3G</code> | 8 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G</code> | <code>alphaO3Gt</code> | 42 | 21 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G</code> | <code>alphaR2G</code> | 36 | 54 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G</code> | <code>g3</code> | 42 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G</code> | <code>alphaO3G</code> | 48 | 144 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G</code> | <code>alphaO3Gt</code> | 186 | 138 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G</code> | <code>alphaR2G</code> | 156 | 204 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G</code> | <code>g3</code> | 396 | 492 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>alphaO3G</code> | 240 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>alphaO3Gt</code> | 420 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>alphaR2G</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G</code> | <code>g3</code> | 1020 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G&#124;G</code> | <code>alphaO3Gt</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G&#124;G</code> | <code>alphaR2G</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;G&#124;G</code> | <code>g3</code> | 1440 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>alphaOHG</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>alphaOHGt</code> | 3 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>g3</code> | 9 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>alphaOHG</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>alphaOHGt</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;Phi&#124;Phibar</code> | <code>g3</code> | 18 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGd</code> | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdtp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRGdtp</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRdD</code> | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;dR&#124;dRbar</code> | <code>g3</code> | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGq</code> | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqtp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRGqtp</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;qL&#124;qLbar</code> | <code>g3</code> | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGu</code> | 30 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGup</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGutp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRGutp</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRuD</code> | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;G&#124;uR&#124;uRbar</code> | <code>g3</code> | 108 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>alphaEdG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>alphaOdG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;dRbar&#124;qL</code> | <code>g3</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>alphaEuG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>alphaOuG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phibar&#124;qLbar&#124;uR</code> | <code>g3</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;dR&#124;qLbar</code> | <code>alphaEdG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;dR&#124;qLbar</code> | <code>alphaOdG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;dR&#124;qLbar</code> | <code>g3</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;qL&#124;uRbar</code> | <code>alphaEuG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;qL&#124;uRbar</code> | <code>alphaOuG</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Phi&#124;qL&#124;uRbar</code> | <code>g3</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEGqp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEGqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRGqtp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;Wi&#124;qL&#124;qLbar</code> | <code>g3</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGd</code> | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdp</code> | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaEGdtp</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRGd</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRGdtp</code> | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>alphaRdD</code> | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;dR&#124;dRbar</code> | <code>g3</code> | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGq</code> | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqp</code> | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaEGqtp</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRGq</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRGqtp</code> | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;qL&#124;qLbar</code> | <code>g3</code> | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGu</code> | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGup</code> | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaEGutp</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRGu</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRGutp</code> | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>alphaRuD</code> | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;G&#124;uR&#124;uRbar</code> | <code>g3</code> | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqtp</code> | 4 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRWqtp</code> | 1 | 2 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g3</code> | 17 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;dR&#124;dRbar</code> | <code>alphaRdD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;dR&#124;dRbar</code> | <code>g3</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>G&#124;qL&#124;qLbar</code> | <code>g3</code> | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>G&#124;uR&#124;uRbar</code> | <code>alphaRuD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>G&#124;uR&#124;uRbar</code> | <code>g3</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;dRbar&#124;qL</code> | <code>alphaEdW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;dRbar&#124;qL</code> | <code>alphaOdW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;dRbar&#124;qL</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;eRbar&#124;lL</code> | <code>alphaEeW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;eRbar&#124;lL</code> | <code>alphaOeW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;eRbar&#124;lL</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;qLbar&#124;uR</code> | <code>alphaEuW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;qLbar&#124;uR</code> | <code>alphaOuW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phibar&#124;Wi&#124;Wi&#124;qLbar&#124;uR</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaRDH</code> | 12 | 20 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>alphaRWDH</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi</code> | <code>g2</code> | 34 | 46 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaOHW</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaOHWt</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaRDH</code> | 24 | 36 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaRWDH</code> | 48 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 90 | 132 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaOHW</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaOHWt</code> | 3 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Phibar&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 81 | 108 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;dR&#124;qLbar</code> | <code>alphaEdW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;dR&#124;qLbar</code> | <code>alphaOdW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;dR&#124;qLbar</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;eR&#124;lLbar</code> | <code>alphaEeW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;eR&#124;lLbar</code> | <code>alphaOeW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;eR&#124;lLbar</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;qL&#124;uRbar</code> | <code>alphaEuW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;qL&#124;uRbar</code> | <code>alphaOuW</code> | 2 | 4 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Phi&#124;Wi&#124;Wi&#124;qL&#124;uRbar</code> | <code>g2</code> | 6 | 10 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>alphaO3W</code> | 8 | 14 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>alphaO3Wt</code> | 42 | 21 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>alphaR2W</code> | 36 | 54 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 42 | 60 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaO3W</code> | 48 | 144 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaO3Wt</code> | 186 | 138 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaR2W</code> | 156 | 204 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 396 | 492 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaO3W</code> | 240 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaO3Wt</code> | 420 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaR2W</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 1020 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaO3Wt</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>alphaR2W</code> | 360 | 720 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi&#124;Wi</code> | <code>g2</code> | 1440 | 2160 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWlp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWltp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRWltp</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRlD</code> | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>g2</code> | 102 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqtp</code> | 12 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRWqtp</code> | 6 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 30 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 102 | 114 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWl</code> | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWlp</code> | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaEWltp</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRWl</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRWltp</code> | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>alphaRlD</code> | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;lL&#124;lLbar</code> | <code>g2</code> | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWq</code> | 26 | 32 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqp</code> | 12 | 16 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaEWqtp</code> | 20 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRWq</code> | 6 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRWqtp</code> | 10 | 12 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 18 | 24 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 104 | 128 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;lL&#124;lLbar</code> | <code>alphaRlD</code> | 7 | 8 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;lL&#124;lLbar</code> | <code>g2</code> | 8 | 9 | The direct exact symbolic comparison proves canonical tensor-map equality for this row; the raw occurrence-count difference is a printer/expansion multiplicity, not an operator-content mismatch. |
| <code>Wi&#124;qL&#124;qLbar</code> | <code>alphaRqD</code> | 7 | 8 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>Wi&#124;qL&#124;qLbar</code> | <code>g2</code> | 8 | 9 | FeynPy leaves the two `alphaRqD` derivative-order branches as separate dummy-Lorentz contractions; FeynRules merges the identical contraction into one term with a doubled coefficient. |
| <code>dRbar&#124;eR&#124;lLbar&#124;qL</code> | <code>alphaEcqedl</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dRbar&#124;eR&#124;lLbar&#124;qL</code> | <code>alphaEcqedlthree</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dRbar&#124;qL&#124;qL&#124;uRbar</code> | <code>alphaEcudqq</code> | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dRbar&#124;qL&#124;qL&#124;uRbar</code> | <code>alphaEcudqqtwo</code> | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dR&#124;eRbar&#124;lL&#124;qLbar</code> | <code>alphaEcqedl</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dR&#124;eRbar&#124;lL&#124;qLbar</code> | <code>alphaEcqedlthree</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dR&#124;qLbar&#124;qLbar&#124;uR</code> | <code>alphaEcudqq</code> | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>dR&#124;qLbar&#124;qLbar&#124;uR</code> | <code>alphaEcudqqtwo</code> | 2 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>eRbar&#124;lL&#124;qL&#124;uRbar</code> | <code>alphaEcuelq</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>eRbar&#124;lL&#124;qL&#124;uRbar</code> | <code>alphaEcuelqtwo</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>eR&#124;lLbar&#124;qLbar&#124;uR</code> | <code>alphaEcuelq</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |
| <code>eR&#124;lLbar&#124;qLbar&#124;uR</code> | <code>alphaEcuelqtwo</code> | 1 | 0 | The pinned charge-conjugation packaging comparison proves canonical tensor-map equality for this row; the literal-signature raw count differs because the same operator is packaged under the CC partner. |

</details>

## Largest Unexplained Raw Head-Count Deltas

These exclude the explicit benign expansions listed above. The large pure-gauge raw deltas can remain large even where the canonical tensor-map comparison above proves equality.

| Head | Total absolute delta |
| --- | ---: |

## Files

- `comparison/artifacts/vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `comparison/artifacts/feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
