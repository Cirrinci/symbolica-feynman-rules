# UnbrokenSM_BFM

Minimal FeynPy reproduction of `papers/UnbrokenSM_BFM.fr.txt`.

- `unbrokenSM_BFM.py`: complete model and compiled Lagrangian.
- `feynpy_vertices.json`: all nonzero three- and four-point FeynPy rules.
- `comparison.py`: exact symbolic comparison with
  `sandbox/wolframnotebook/LSM_full_FeynRules.json`.
- `comparison_report.json`: generated comparison summary.
- `export_model.py`: prints all Lagrangian sectors, demonstrates one
  non-literal rule comparison, and regenerates both JSON outputs.

Run:

```bash
.venv/bin/python models/UnbrokenSM_BFM/comparison.py
```

For the verbose sector/rule demonstration:

```bash
.venv/bin/python models/UnbrokenSM_BFM/export_model.py
```

Result: **67/67 exact symbolic matches** (42 three-point and 25 four-point),
with no missing or extra interaction signatures.

The raw strings are intentionally different: FeynRules uses `ME`, `FV`,
`Ta`, `fsu2`, `fsu3`, `ProjM` and `ProjP`; FeynPy uses typed Spenso tensors.
The comparison translates these objects, preserves external-leg order, reduces
equivalent chiral chains, canonicalizes dummy indices and then requires the
symbolic difference to be zero. No numerical fitting is used.

The JSON oracle contains interaction vertices only. Two-point terms, particle
metadata, widths, PDG codes and downstream FeynRules export formats are not
part of this comparison.
