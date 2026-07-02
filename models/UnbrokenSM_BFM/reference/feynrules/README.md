# FeynRules reference material

This directory contains the complete external reference material used by the
`UnbrokenSM_BFM` comparison. Nothing under `sandbox/` or `papers/` is required
to run the tracked test suite.

## Files

- `UnbrokenSM_BFM.fr` is the FeynRules source model used to define `LSM`.
- `UnbrokenSM_BFM_export.nb` is the Wolfram notebook used to load the model,
  calculate `FeynmanRules[LSM, FlavorExpand -> True]`, sort the vertices by
  field signature, and export JSON.
- `LSM_full_FeynRules.json` is that exported 67-vertex oracle. It preserves
  the original FeynRules leg order while using a sorted field key for
  signature alignment.

## Pinned provenance

The regression suite pins the exact artifacts with SHA-256 hashes:

- FeynRules model: `f0952159e47b807d7b5daaa89f70d00666413f68fca7d0d4fe10cbe891c372d9`
- export notebook: `c66c28303da3406a9e2089e31c64e20dd2f3b0dd86616105c5bdf41167522f4b`
- vertex JSON: `fd34adaed860c50278b6dba965f8a2bc532b1f8f32336e85bfcb8d6430bc8033`

## Regeneration

Open the saved notebook from this directory in a Wolfram kernel with FeynRules
available. It sets the working directory to `NotebookDirectory[]`, loads the
bundled `UnbrokenSM_BFM.fr`, and writes `LSM_full_FeynRules.json` beside the
notebook. After regenerating the file, run:

```bash
.venv/bin/python -m pytest -q models/UnbrokenSM_BFM/tests
```

An intentional reference update must also update the pinned hash in
`tests/test_bfm_feynrules_fixture_provenance.py` and should explain why the oracle
changed.

## Scope

The oracle contains the nonzero flavor-expanded three- and four-point
interaction vertices generated from `LSM`. It does not validate two-point
terms, particle metadata, parameter-card semantics, widths, PDG identifiers,
or downstream export formats.
