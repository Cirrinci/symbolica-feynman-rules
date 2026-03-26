## Roadmap

This roadmap turns the current Symbolica/Spenso prototype into a more complete
Python analogue of FeynRules.

### Current baseline

Already working in the repository:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with permutation-aware momentum assignment
- fermion permutation signs
- amputated open-index and unamputated external fermion factors
- spinor-delta output using Spenso bispinor metrics
- hand-supplied vector-current structures such as `gamma(mu, i, j)`
- explicit remapping of open spinor labels inside coupling tensors to external
  leg spinor slots for the currently exercised fermion patterns
- current-current four-fermion operators with gamma matrices in the coupling

Current implementation entry points:

- [code/model_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/model_symbolica.py)
- [code/examples_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/examples_symbolica.py)

### Current handoff

This is the most important status snapshot to keep in mind for the next
session.

Physics conclusions reached in the current cleanup:

- `-(g/2)(psibar psi)^2` should not vanish after amputation
- the correct amputated vertex is the open-index object
  `-i g [g(i1,i2)g(i3,i4) - g(i1,i4)g(i3,i2)]`
- the unstripped result with `UF/UbarF` is only a matrix-element diagnostic
- a bare product like `psi * psibar * psi * psibar` is not a well-defined
  four-fermion scalar operator unless its spinor contractions are specified

Current support boundary in the code:

- supported:
  - scalar interactions
  - derivative interactions
  - fermion bilinears encoded by repeated dummy labels in
    `field_spinor_indices`
  - scalar-bilinear four-fermion terms like
    `field_spinor_indices=[alpha, alpha, beta, beta]`
- explicit coupling tensors with open spinor labels for the currently covered
  examples, including vector currents and a current-current four-fermion
  operator
- not yet supported in a general way:
  - arbitrary multi-fermion tensor structures beyond the currently exercised
    bilinear/current-current patterns
  - automatic extraction of those tensor structures from a higher-level model
    declaration

Most important technical limitation right now:

- the engine can now remap explicit open spinor labels in the coupling for the
  current patterns, but normalization choices and ambiguous encodings are still
  not centralized or guarded strongly enough.

### What is still missing

The prototype can already carry gamma matrices symbolically, but it does not yet
provide a proper model-language layer for them. In practice, the missing pieces are:

- first-class Dirac/Lorentz tensor objects instead of ad hoc symbolic factors
- automatic construction of spinor chains from interaction definitions
- a field/model declaration layer that can generate vertex ingredients
- gauge-index support and representation-aware gauge tensors
- gauge-field specific Lorentz structures such as field strengths and
  triple/quartic gauge-boson vertices
- a cleaner bridge from user Lagrangian input to final vertex expressions

### Recommended build order

The safest order is:

1. Stabilize gamma-matrix support.
2. Add a model-definition layer for fields and interactions.
3. Add gauge groups and gauge-field interactions.
4. Add higher-level FeynRules-style extraction APIs.

This order matters because gauge interactions for fermions and vectors depend on
having a good tensor/index representation first.

### Phase 1: Gamma matrices

Goal:
Make gamma structures first-class symbolic tensor objects, not just manual
coupling prefactors.

Deliverables:

- define canonical tensor names for `gamma`, `sigma`, identity spinor metric,
  chirality objects such as `gamma5` if needed
- represent gamma matrices with explicit Lorentz and bispinor slots using Spenso
- separate scalar coupling factors from tensor/Lorentz/spinor structures
- provide helpers to build common bilinears:
  - `psibar psi`
  - `psibar gamma^mu psi`
  - `psibar gamma^mu gamma^5 psi`
  - `psibar sigma^{mu nu} psi`
- add simplification helpers for basic spinor-metric contractions around gamma
  chains

Suggested code changes:

- add a tensor-structure module, for example `code/tensors.py`
- move gamma/spinor structure builders out of the example file and into library
  functions
- keep `model_symbolica.py` as the contraction engine, but let it consume richer
  coupling/tensor objects

Success criteria:

- the user can declare a Yukawa term and a vector/axial current term without
  manually spelling out raw `gamma(...)` expressions in every example
- tests cover at least scalar, Yukawa, vector current, axial current, and one
  sigma-tensor interaction

### Phase 2: Model layer

Goal:
Move from “call `vertex_factor(...)` with many lists” to “define a model and ask
for vertices.”

Deliverables:

- `Field`, `Parameter`, `Index`, and `InteractionTerm` data structures
- explicit field metadata:
  - statistics
  - self-conjugacy
  - Lorentz type
  - gauge representation
  - spinor index structure
- a compact interaction-specification format that expands into:
  - field ordering
  - derivative targets
  - tensor factors
  - combinatoric normalization metadata

Suggested API direction:

- `Model(fields=[...], parameters=[...], interactions=[...])`
- `derive_vertex(interaction, external_fields=[...])`
- `derive_vertices(model)`

Success criteria:

- common examples no longer require hand-built `alphas`, `betas`, `field_roles`,
  and `field_spinor_indices`
- notebooks become thin demos over importable library code

### Phase 3: Gauge fields

Goal:
Introduce gauge representations and gauge-boson interactions in a way that fits
naturally with Spenso representations.

Deliverables:

- gauge-field type metadata:
  - abelian vs non-abelian
  - adjoint/fundamental representations
  - gauge indices
- canonical gauge tensors:
  - structure constants `f^{abc}`
  - generators `T^a_{ij}`
  - adjoint and fundamental metrics if needed
- Lorentz structures for vector fields:
  - polarization/index slots
  - metric tensors
  - momentum-difference structures
- interaction builders for:
  - fermion-gauge current `psibar gamma^mu T^a psi A^a_mu`
  - scalar covariant-derivative couplings
  - triple gauge-boson vertex
  - quartic gauge-boson vertex
  - ghost terms later if needed

Recommended order inside gauge support:

1. QED-like abelian vector couplings.
2. Non-abelian fermion-gauge couplings with `T^a`.
3. Scalar gauge couplings from covariant derivatives.
4. Pure gauge self-interactions.

Success criteria:

- derive the QED fermion-photon vertex cleanly from a model declaration
- derive a simple Yang-Mills fermion-gauge vertex with explicit gauge indices
- derive the Lorentz structure of a three-gauge-boson vertex

### Phase 4: FeynRules-style usability

Goal:
Make the system feel like a small Python FeynRules environment instead of a
research prototype.

Deliverables:

- a model file format or Python DSL
- automatic symmetry/combinatoric factor helpers
- a standardized vertex object with:
  - external fields
  - coupling prefactor
  - Lorentz structure
  - gauge structure
  - momentum convention
- export helpers for readable text, LaTeX, and possibly UFO-like data later

### Immediate next tasks

These are the next concrete tasks I recommend doing in the codebase:

1. Centralize normalization and symmetry-factor conventions for fermion
   operators, especially four-fermion terms.
2. Add validation against ambiguous fermion encodings, especially repeated
   field spinor labels combined with explicit tensor endpoints in the coupling.
3. Widen supported multi-fermion structures beyond the current
   bilinear/current-current patterns.
4. Broaden `code/spenso_structures.py` from a wrapper collection into the
   central tensor vocabulary used by the engine.
5. Introduce an `InteractionTerm` object so examples stop passing parallel lists.
6. Introduce gauge-field metadata and an abelian vector-field example.
7. Add a non-abelian generator tensor `T(a,i,j)` and test a simple color current.

### Session outcome (2026-03-26)

Completed in this session:

1. Ran `./.venv/bin/python code/examples_symbolica.py --suite fermion` as the
   primary regression check.
2. Verified that `vertex_factor(...)` now remaps explicit coupling spinor
   labels per contraction permutation.
3. Locked that behavior down with runnable checks for
   `psibar gamma^mu psi A_mu` and
   `gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)`.

Recommended starting point for the next session:

1. Centralize normalization and symmetry-factor conventions.
2. Add stronger validation for ambiguous encodings that mix repeated dummy
   labels with explicit coupling endpoints.
3. Move the runnable asserts toward a dedicated test harness.

### Nice rule of thumb

For each new physics feature, implement it in this order:

1. tensor/index representation
2. interaction declaration
3. contraction/vertex extraction
4. simplification
5. regression tests

That sequence should keep the project extensible and avoid notebook-only logic.
