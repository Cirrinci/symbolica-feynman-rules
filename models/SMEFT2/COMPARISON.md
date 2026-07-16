# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-07-16` by `models/SMEFT2/comparison.py`.

## Scope

Signature coverage and coefficient-head content. Full tensor-rule equality is not claimed by this SMEFT2 report.

| Item | Value |
| --- | ---: |
| Reference vertices | 184 |
| FeynPy 3-6 point signatures | 147 |
| Shared signatures | 137 |
| Reference-only signatures | 47 |
| FeynPy-only signatures | 10 |
| Shared coefficient-head matches | 51 |

## Basis

- Reference: `EFT-only FeynRules Ltot`.
- Local default model: `EFT-only FeynPy Ltot`.
- Local SM plus EFT model: `Lfull`.
- Omitted sectors: `LX2D2, LH2XD2, LH2D4, LF2D3, LF2HD2, LF2XD, LEvF2XD`.

## Status Counts

| Status | Count |
| --- | ---: |
| `FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING` | 8 |
| `FEYNPY_ONLY_WEINBERG_PACKAGING` | 2 |
| `MISSING_SIGNATURE_OMITTED_DERIVATIVE_SECTORS` | 45 |
| `MISSING_SIGNATURE_WEINBERG_PACKAGING` | 2 |
| `SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH` | 6 |
| `SHARED_HEADS_MATCH` | 51 |
| `SHARED_LOCAL_PP_EXTRA` | 2 |
| `SHARED_MISSING_OMITTED_HEADS` | 72 |
| `SHARED_MISSING_OMITTED_HEADS_PLUS_LOCAL_EXTRA` | 6 |

## Largest Reference-Side Head Gaps

| Head | Count |
| --- | ---: |
| `g1` | 33 |
| `alphaRqD` | 19 |
| `g3` | 19 |
| `g2` | 18 |
| `alphaRdHD4` | 16 |
| `alphaRuHD4` | 16 |
| `alphaRDH` | 14 |
| `alphaRdHD2` | 14 |
| `alphaRuHD2` | 14 |
| `alphaRdHD1` | 12 |
| `alphaRdHD3` | 12 |
| `alphaReHD1` | 12 |
| `alphaRuHD1` | 12 |
| `alphaRuHD3` | 12 |
| `alphaReHD4` | 10 |
| `alphaRdD` | 9 |
| `alphaRlD` | 9 |
| `alphaRuD` | 9 |
| `alphaReHD2` | 8 |
| `alphaEGqp` | 7 |

## Largest Local Extra Heads

| Head | Count |
| --- | ---: |
| `alphaEdH` | 2 |
| `alphaEeH` | 2 |
| `alphaEuH` | 2 |
| `alphaRHl3pp` | 1 |
| `alphaRHq3pp` | 1 |

## Files

- `vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
