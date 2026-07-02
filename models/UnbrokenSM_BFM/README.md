# UnbrokenSM_BFM

Self-contained FeynPy reproduction of the FeynRules
`UnbrokenSM_BFM.fr` model.

- `unbrokenSM_BFM.py`: complete model and compiled Lagrangian.
- `feynpy_vertices.json`: all nonzero three- and four-point FeynPy rules.
- `comparison.py`: exact symbolic comparison with
  `reference/feynrules/LSM_full_FeynRules.json`.
- `comparison_report.json`: generated comparison summary.
- `reference/feynrules/UnbrokenSM_BFM.fr`: FeynRules source model used for
  the reference calculation.
- `reference/feynrules/UnbrokenSM_BFM_export.nb`: Wolfram notebook that loads
  the source model, runs `FeynmanRules[LSM, FlavorExpand -> True]`, and exports
  the JSON oracle.
- `reference/feynrules/README.md`: reference provenance, hashes, scope, and
  regeneration notes.
- `tests/`: signature-coverage, exact-comparison, and fixture-provenance tests.

Run:

```bash
.venv/bin/python models/UnbrokenSM_BFM/comparison.py
```

Result: **67/67 exact symbolic matches** (42 three-point and 25 four-point),
with no missing or extra interaction signatures.

The raw strings are intentionally different: FeynRules uses `ME`, `FV`,
`Ta`, `fsu2`, `fsu3`, `ProjM` and `ProjP`; FeynPy uses typed Spenso tensors.
The comparison translates these objects, preserves external-leg order, reduces
equivalent chiral chains, canonicalizes dummy indices and then requires the
symbolic difference to be zero.

The JSON oracle contains interaction vertices only. Two-point terms, particle
metadata, widths, PDG codes and downstream FeynRules export formats are not
part of this comparison.
