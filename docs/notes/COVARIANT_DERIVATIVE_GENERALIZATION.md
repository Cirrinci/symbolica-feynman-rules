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

### Conceptual frame: the covariant derivative is representation-dependent

Yes: in general the covariant derivative is representation-dependent.

The stable abstraction is:

- `D_mu = partial_mu + i sum_G g_G A^a_{G,mu} T^a_R`

where the gauge-group data come from the gauge declaration, while the object
`T^a_R` is determined by the field representation `R`.

- for an abelian factor, `T_R` reduces to the field charge
- for a non-abelian factor, `T^a_R` is the generator acting in that field's representation

This is the right FeynRules-style mental model for the codebase:

- gauge groups declare the gauge boson, coupling, structure constants, and available representations
- fields declare their quantum numbers and index slots
- the covariant compiler reads that metadata and builds the gauge part appropriate to the field

Important convention note for this repository:

- the physical compiler freezes the sign convention as `D_mu = partial_mu + i g A_mu`
- the structural point here is the dependence on representation metadata, not the overall sign choice
- if the sign convention changes later, the metadata flow should stay the same

#### 1. Do not hardcode one universal `D_mu`

Treat the covariant derivative as a function of field metadata, not as one
single object reused blindly for every field.

Each field should carry at least:

- whether it is a scalar, fermion, or vector
- whether it is self-conjugate
- its abelian quantum numbers
- its non-abelian representation indices
- which gauge groups act on it

That is exactly the same split used by FeynRules:

- `QuantumNumbers` encode `U(1)`-type charges
- `Indices` encode non-abelian representation slots
- gauge groups declare the generator structure that acts on those slots

#### 2. Separate the problem into two layers

Layer A: gauge-group data

- gauge boson `A_mu^a`
- coupling `g`
- whether the group is abelian
- generator object `T^a` or abelian charge label
- structure constants for non-abelian groups

Layer B: field transformation data

- under `U(1)`: charge `q`
- under `SU(N)`: representation type such as singlet, fundamental, or adjoint
- the actual index slots the group acts on
- for repeated identical index kinds, whether the action is on one selected slot or summed over matching slots

Once those two layers exist, the derivative builder only has to read metadata
and sum the relevant gauge contributions.

#### 3. Build `D_mu phi` by summing one contribution per gauge factor

Conceptually:

- start from `partial_mu phi`
- for each gauge group `G`:
  - if `phi` is neutral or a singlet under `G`, add nothing
  - if `G = U(1)`, add `+ i g q A_mu phi`
  - if `G` is non-abelian, add `+ i g A_mu^a T^a_R phi`

That is the reusable core. The derivative builder should not care whether the
field is "the scalar case" or "the fermion case" until it needs the field kind
for the surrounding kinetic term.

#### 4. Let the representation determine the concrete formula

Examples in the repository sign convention:

Complex scalar with electric charge `q`:

- `D_mu phi = (partial_mu + i e q A_mu) phi`

Fermion in QED:

- `D_mu psi = (partial_mu + i e q A_mu) psi`

Fundamental of `SU(3)`:

- `(D_mu phi)^i = partial_mu phi^i + i g_s G_mu^a (T^a)^i{}_j phi^j`

Adjoint of `SU(N)`:

- `(D_mu X)^a = partial_mu X^a + g f^{abc} A_mu^b X^c`

The important point is that the symmetry is the same, but the concrete action
depends on the representation:

- fundamental action uses the generator matrix on a fundamental slot
- adjoint action uses the adjoint generators, equivalently the structure constants

#### 5. Treat conjugate fields as conjugate representations

This is one of the easiest places to make a wrong implementation choice.

If `phi` transforms in `R`, then `phi^dagger` transforms in the conjugate
representation `R*`.

That means:

- for abelian groups, the charge changes sign
- for non-abelian groups, the generator action becomes that of the conjugate representation

So the code should not blindly reuse the same gauge term for both a field and
its daggered occurrence. The conjugate action should be derived from metadata.

For example in scalar QED with the sign convention used in this repository:

- `D_mu phi = (partial_mu + i e q A_mu) phi`
- `D_mu phi^dagger = (partial_mu - i e q A_mu) phi^dagger`

That sign flip is essential.

### Status update (2026-04-08)

This note started as the design document for the repeated-slot generalization.
The main implementation described below is now present in the live code for the
covered compiler paths.

Implemented pieces in the current source tree:

- `GaugeRepresentation` now has explicit repeated-slot semantics through:
  - `slot`
  - `slot_policy="unique"` and `slot_policy="sum"`
  - `slots_for(...)` and related multi-slot helpers
- `GaugeGroup` and the gauge compiler now resolve all active matching slots instead of assuming only one unique slot
- `Field.pack_slot_labels(...)` now preserves ordinal stability for repeated kinds with explicit `None` placeholders
- the minimal and covariant gauge compilers now insert spectator identities consistently on inactive repeated slots
- covariant compilation of the covered bislot scalar case now expands:
  - one current pair per active slot
  - one contact contribution per ordered slot pair
- ambiguous repeated-slot cases are rejected by default and only summed when the metadata opts in with `slot_policy="sum"`
- initial dedicated `pytest` coverage exists in `tests/test_covariant_bislot_sum.py`

Remaining gaps after the current implementation:

- mixed-group complex-scalar kinetic terms still miss the cross-group two-gauge contact terms
- most of the covariant regression matrix still lives in `src/examples.py` rather than the dedicated `tests/` tree

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

### What limited generality before the current implementation (specifically for repeated identical index kinds)

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

### Generalization design (now implemented for the covered repeated-slot compiler paths)

The goal is to make covariant-derivative extraction/expansion general in the presence of repeated identical index kinds.

There are three complementary changes: metadata semantics, covariant compiler updates, and ordinal-stable label packing.

#### A) Add a slot policy: unique-slot vs sum over matching slots

Implemented API change:

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

Implemented change to `Field.pack_slot_labels(slot_labels)`:

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
3. Extend the now-working repeated-slot machinery into the still-missing mixed-group scalar contact case
4. Keep `src/model_symbolica.py` generic:
   - no tensor-specific branching in the engine
   - ensure all generality is expressed by correct couplings/labels coming from compiler layers
