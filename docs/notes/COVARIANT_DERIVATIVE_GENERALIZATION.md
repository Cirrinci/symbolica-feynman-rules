## Generalizing Covariant-Derivative Expansion (Repeated Index Kinds)

### Context
This repository is a Python analogue of FeynRules built on:

- `src/model_symbolica.py`: the contraction engine (Wick-permutation sums, derivative-to-momentum replacement, open-index remapping)
- `src/model.py`: the model-layer declarations (`Field`, `GaugeGroup`, `InteractionTerm`, and covariant/kinetic term declarations)
- `src/gauge_compiler.py`: the “covariant compiler” that expands gauge-covariant kinetic terms into conventional `InteractionTerm`s
- `src/spenso_structures.py` and `src/operators.py`: Spenso-backed tensor wrappers (gamma matrices, metrics, gauge generators)

In the docs, “covariant derivative” means gauge-covariant derivatives of the form:

- Matter: `D_mu = partial_mu + i g A_mu`
- Derivatives map to momentum factors in Fourier space: `partial_mu -> -i p_mu`

### Where covariant-derivative expansion happens

#### Covariant compiler entry point
- [`src/gauge_compiler.py`](src/gauge_compiler.py)
  - `compile_covariant_terms(model)`: flattens declared covariant kinetic terms into standard `InteractionTerm`s
  - `compile_dirac_kinetic_term(model, DiracKineticTerm)`: expands the gauge part of `psibar i gamma^mu D_mu psi`
  - `compile_complex_scalar_kinetic_term(model, ComplexScalarKineticTerm)`: expands the gauge part of `(D_mu phi)^dagger (D^mu phi)` into
    - a one-gauge “current” term
    - a two-gauge “contact” term

#### Engine behavior that consumes the result
- [`src/model_symbolica.py`](src/model_symbolica.py)
  - `InteractionTerm.to_vertex_kwargs(external_legs)`: produces the parallel-list kwargs consumed by `vertex_factor(...)`
  - `vertex_factor(...)`: calls `contract_to_full_expression(...)` and applies global factors / optional stripping
  - `contract_to_full_expression(...)`:
    - inserts derivative momentum factors via:
      - `term *= (-I) * pcomp(ps[perm[tgt]], mu)` (loop over `derivative_indices` and `derivative_targets`)
    - performs “open label” remapping in the coupling tensor by:
      - identifying labels that appear exactly once across field slots (`_open_index_labels`)
      - replacing them with the corresponding external-leg labels for each permutation

### What limits generality today (specifically for repeated identical index kinds)

This section lists the concrete hard-coded assumptions that prevent “fully general” covariant-derivative expansion for fields carrying multiple indices of the same kind (e.g. a tensor field with two identical color-fundamental slots).

#### 1) Gauge representation acts on a single “selected slot”
- [`src/model.py`](src/model.py)
  - `GaugeRepresentation.slot_for(field)`
    - returns a concrete field-index slot only if exactly one slot matches the representation’s `IndexType`
    - raises if multiple slots match, unless the user disambiguates with `GaugeRepresentation(slot=...)`

Effect on covariant compilation:

- `src/gauge_compiler.py` selects a single `rep_slot` and builds generators only for that slot.
- Therefore, covariant derivative action on a multi-slot field cannot be interpreted as “sum over all tensor-product factors” unless the representation metadata explicitly enumerates/chooses an active slot.

#### 2) Scalar contact terms do not include cross-slot gauge-leg placement
- [`src/gauge_compiler.py`](src/gauge_compiler.py)
  - `compile_complex_scalar_gauge_terms(...)` constructs the two-gauge contact tensor using:
    - a single `generator_chain`:
      - `rep.build_generator(adj_mu, left, middle) * rep.build_generator(adj_nu, middle, right)`

What is missing for a general tensor field:

- If the field has multiple identical representation slots, the gauge part of `D_mu` can act on different slots for different gauge legs:
  - `A_mu` acting on `slot_i`
  - `A_nu` acting on `slot_j`
- For `slot_i != slot_j`, the correct tensor-product expression is not a single generator chain on one slot; it factorizes into two generators placed on different slots, with spectator identities on the other slots.

#### 3) Label packing for repeated kinds can become ordinal-unstable
- [`src/model.py`](src/model.py)
  - `Field.pack_slot_labels(slot_labels)`
    - groups provided slot labels by `index.kind`
    - appends labels in the iteration order of the `Field.indices` and then decides whether to pack into a tuple
    - if only a subset of slots is provided for a repeated kind, the packed tuple can be shorter than the field’s actual repeated-kind count

Engine interaction:

- [`src/model_symbolica.py`](src/model_symbolica.py)
  - `_flatten_index_labels(...)` enumerates tuples and counts each `(kind, ordinal, label)`
  - `_get_label(..., ordinal)` retrieves the label at a given ordinal

Failure mode:

- If tuple lengths do not preserve the intended ordinal positions, `_get_label` can return the wrong label for a given ordinal, and `_open_index_labels` counting/remapping can misidentify which coupling labels are “open” vs “contracted”.

### Generalization design (to support “sum over tensor factors” and cross-slot contact terms)

The goal is to make covariant-derivative extraction/expansion general in the presence of repeated identical index kinds.

There are three complementary changes: metadata semantics, covariant compiler updates, and ordinal-stable label packing.

#### A) Add a slot policy: unique-slot (current) vs sum over matching slots (general)

Proposed API change (conceptual):

- Extend `GaugeRepresentation` with a `slot_policy` or similar field:
  - `slot_policy="unique"` (default for backward compatibility): current behavior
  - `slot_policy="sum"`: when `slot` is not explicitly provided, act on all matching slots and sum contributions

Concrete compilation semantics:

- If the representation declares `slot=...`, only that slot is active.
- If `slot_policy="sum"` and `slot` is not set, then the covariant compiler should generate:
  - current terms as a sum over active slots for the single gauge leg
  - contact terms as a sum over `(slot_i, slot_j)` placement for two gauge legs

Why this is the right layer:

- This keeps `src/model_symbolica.py` generic: it only needs correct coupling tensors and correct index labels.
- It moves “tensor-product factor interpretation” into the declaration/compiler layer, where metadata belongs.

#### B) Generalize fermion current and scalar current: loop over active slots

For both:

- `compile_fermion_gauge_current(...)`
- `compile_complex_scalar_gauge_terms(...)` (current terms only)

update structure:

- Determine all matching representation slots for the gauge generator action.
- For each active slot `k`:
  - build the appropriate generator tensor using labels tied to slot `k`
  - build spectator identity factors for the remaining slots
  - sum the resulting interaction terms (or sum their couplings)

Important nuance:

- Spectator identities depend on which slots are excluded. Therefore, when active slot changes, spectator identity factors must be recomputed (or built systematically per-slot).

#### C) Generalize scalar contact term: sum over ordered placement of the two gauge legs

Replace the single-slot `generator_chain` contact construction with a two-slot decomposition.

Given:

- adjoint index for the first gauge leg: `adj_mu`
- adjoint index for the second gauge leg: `adj_nu`
- repeated representation slots: `{0, 1, ...}`

Construct:

- For each `(slot_i, slot_j)`:
  - if `slot_i == slot_j`:
    - use the existing generator chain on that slot:
      - `rep.build_generator(adj_mu, left_i, middle) * rep.build_generator(adj_nu, middle, right_i)`
  - else (`slot_i != slot_j`):
    - build two independent generators:
      - one generator tensor for `adj_mu` acting on slot `slot_i`
      - one generator tensor for `adj_nu` acting on slot `slot_j`
    - keep distinct fundamental indices for the two slots
    - multiply by spectator identities on all other slots

Then sum the contributions over all `(slot_i, slot_j)`.

This yields the full tensor-product representation of the gauge interaction in the multi-slot case.

#### D) Make packing ordinal-stable for repeated kinds (use `None` placeholders)

Proposed change to `Field.pack_slot_labels(slot_labels)`:

- For each index kind that appears `N` times in `Field.indices`, the packed representation should always output:
  - `label_or_tuple` where tuples (or lists) have length exactly `N`
  - unspecified slots produce `None` placeholders

Why this helps:

- `_flatten_index_labels` already ignores `None`, so counting/open-label remapping remains correct.
- `_get_label` then reliably maps ordinals to the intended label positions.
- This prevents “ordinal drift” when covariant compiler builds partial slot label maps (common once you start looping over active slots).

### Regression test plan (matrix to validate generality)

This section proposes what to add to ensure the covariant compiler is correct once multi-slot generalization is implemented.

#### Minimal test fixtures to reuse

The repo already has:

- `PhiBiField` as a non-self-conjugate scalar with two identical `COLOR_FUND_INDEX` slots
- `MODEL_SCALAR_QCD_BISLOT_BASE` and `MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS`
- minimal gauge compilation tests that already check spectator identities in the bislot case

New tests should cover covariant compilation of kinetic terms for these same models.

#### Test cases (covariant compiler)

1. **Bislot QCD covariant kinetic term expands without errors**
   - Input: `Model(name=..., covariant_terms=(ComplexScalarKineticTerm(field=PhiBiField),))`
   - Assert:
     - `compile_covariant_terms(model)` returns the expected number of `InteractionTerm`s
       - for a generic complex-scalar kinetic term, current behavior is “current + conjugate-current + contact”
     - each term’s index structures and couplings contain generator tensors in the expected placement pattern

2. **Current term includes sum over both active slots**
   - Expected behavior under `slot_policy="sum"`:
     - two contributions corresponding to:
       - gauge leg acting on slot 0 (with slot 1 spectator identity)
       - gauge leg acting on slot 1 (with slot 0 spectator identity)

3. **Contact term includes same-slot and cross-slot contributions**
   - Expected behavior under `slot_policy="sum"`:
     - same-slot contributions: `(slot_0,slot_0)` and `(slot_1,slot_1)`
     - cross-slot contributions: `(slot_0,slot_1)` and `(slot_1,slot_0)`
   - Assert that:
     - same-slot pieces have the expected “generator chain” structure
     - cross-slot pieces factor into two independent generator tensors in different slots

4. **Ambiguous gauge representation metadata is handled**
   - For `MODEL_SCALAR_QCD_BISLOT_AMBIGUOUS`:
     - define expected behavior depending on the chosen default `slot_policy`
       - either error (if default remains “unique”)
       - or sum over slots (if default is “sum”)

#### Validation approach

Prefer tests that compare final vertices after:

- using `vertex_factor(...)` / `_model_vertex(...)` (as in existing suites)
- applying `simplify_deltas(...)` and any relevant gamma/tensor canonicalization

Because multi-slot contact terms can grow, compare against:

- either a canonicalized tensor form (`canonize_spenso_tensors`)
- or compact “structured” expectations using `gauge_generator(...)` and `lorentz_metric(...)` helper builders.

### Next steps aligned with existing roadmap

These are the codebase hardening tasks that must happen before (or along with) generalizing covariant derivative expansion:

1. Freeze covariant/engine conventions into a stable, single source of truth (docs + tests)
2. Move covariant compiler assertions out of `src/examples.py` into a dedicated pytest-style test harness
3. Implement repeated index kind hardening by:
   - adding slot policy semantics to `GaugeRepresentation`
   - updating `src/gauge_compiler.py` to loop over active slots for current and contact terms
   - making `Field.pack_slot_labels` ordinal-stable with explicit `None` placeholders
4. Keep `src/model_symbolica.py` generic:
   - no tensor-specific branching in the engine
   - ensure all generality is expressed by correct couplings/labels coming from compiler layers

