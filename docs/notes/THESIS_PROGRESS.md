## Thesis Progress

### Working title

Toward a Python implementation of FeynRules using Symbolica and Spenso

### Core objective

The project aims to derive Feynman interaction vertices from model declarations
in Python, using:

- Symbolica for symbolic expressions, rewriting, simplification, and combinatorics
- Spenso for tensor structures, spinor/Lorentz objects, and gauge-ready indices

### Current architecture

The active implementation now lives in `src/`.

Main files:

- `src/model_symbolica.py`
  - contraction engine and simplification helpers
- `src/model.py`
  - model-layer declarations and bridge objects
- `src/spenso_structures.py`
  - gamma, metric, and generator wrappers
- `src/operators.py`
  - reusable operator builders
- `src/gauge_compiler.py`
  - minimal structural gauge compiler plus convention-fixed physical compiler for covered matter and pure-gauge cases
- `src/examples.py`
  - runnable examples and regression checks

Supporting notes:

- `docs/notes/PROJECT_GOAL.md`
- `docs/notes/ROADMAP.md`
- `docs/notes/RESEARCH_LOG.md`
- `docs/notes/FEYNRULES_STYLE_STRATEGY.md`

### What is already achieved

The current prototype already supports:

- scalar polynomial interactions
- derivative interactions with correct permutation-aware momentum assignment
- fermionic permutation signs
- mixed scalar-fermion workflows
- stripped and unstripped external wavefunction forms
- spinor-delta structures using Spenso bispinor metrics
- gamma-matrix and gauge-generator structures in the covered patterns
- a first model layer that can drive the same engine as the direct API
- reusable operator builders for the covered fermion/gauge structures
- a minimal compiled gauge workflow for non-abelian fermion currents and abelian complex-scalar terms
- a working covariant-derivative compiler for covered fermion and complex-scalar gauge interactions
- fixed-convention covariant outputs for QCD/QED fermions and QED/QCD complex scalars
- multi-gauge-group covariant expansion for covered matter fields
- pure-gauge kinetic compilation for:
  - abelian `-1/4 F_{mu nu} F^{mu nu}`
  - non-abelian `-1/4 F^a_{mu nu} F^{a mu nu}`
  - Yang-Mills 3-gauge and 4-gauge vertices
- direct/model agreement checks in the main example suite
- a runnable gamma/tensor validation sandbox

### Current position

The project is past the proof-of-concept stage for the symbolic core.

Reasonable summary:

- scalar sector: working
- derivative sector: working
- fermion combinatorics: working at prototype level
- spinor/Lorentz tensor structures: working in the covered patterns
- model-driven input: working and usable
- minimal gauge-model compilation: working
- covariant-derivative compilation for covered matter-sector cases: working
- ordinary pure-gauge field-strength compilation: working in the covered abelian and Yang-Mills cases
- BFM-complete support: not implemented
- full FeynRules-like usability layer: not implemented

### Main result so far

The main scientific and technical result so far is that Symbolica is already
sufficient for the core combinatoric and symbolic-rewriting tasks needed for
vertex derivation, while Spenso provides a natural representation layer for
the tensor structures required for fermion and later gauge-theory work.

In short, the architecture is viable.

### Main limitations

The prototype is not yet a complete FeynRules replacement.

Main missing or weak points:

- the model layer is still thinner than the intended end state
- conventions now work in code but still need one stable documentation/testing source of truth
- gauge support is broader but still not BFM-complete
- background-field-gauge scaffolding, gauge fixing, and ghosts are not implemented
- there is not yet a dedicated test harness beyond the main example script

### Next milestone

The next milestone is to build BFM-oriented scaffolding on top of the now-working ordinary gauge foundation.

That means:

1. keep the active conventions frozen in code/docs/tests
2. move the growing checks into a stronger regression layout
3. add background/quantum gauge-field splitting, gauge fixing, and ghosts
4. then continue into broader BFM-style model support

### Writing use

This document should stay short and stable.

It is meant to answer:

- what is the project trying to do?
- what already works?
- what is the current architecture?
- what remains to be done?

For dated progress details, use `docs/notes/RESEARCH_LOG.md`.
