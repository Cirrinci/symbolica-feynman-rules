# SMEFT2 FeynRules/FeynPy Comparison

Generated on `2026-07-21` by `models/SMEFT2/comparison.py`.

## Scope

Signature coverage and coefficient-head content. Full tensor-rule equality is not claimed by this SMEFT2 report.

| Item | Value |
| --- | ---: |
| Reference vertices | 184 |
| FeynPy 3-6 point signatures | 192 |
| Shared signatures | 182 |
| Reference-only signatures | 2 |
| FeynPy-only signatures | 10 |
| Shared coefficient-head matches | 168 |

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

## Files

- `vertex_comparison_report.json` contains every reference row and FeynPy-only signature.
- `feynpy_vertices.json` contains the regenerated local FeynPy rules and coefficient heads.
- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle used for the comparison.
