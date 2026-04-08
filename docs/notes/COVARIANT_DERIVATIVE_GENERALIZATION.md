# Generalizing Covariant-Derivative Expansion for Repeated Index Kinds

## Purpose

This note explains how covariant-derivative expansion should work in this
repository, what is already implemented, and what still remains.

The central point is simple:

> The covariant derivative is not one universal object. It depends on the field
> representation.

That is the right FeynRules-style viewpoint:

- gauge groups declare the gauge data
- fields declare the transformation data
- the compiler combines both into the concrete gauge action

## Repository Map

- `src/model_symbolica.py`
  Contraction engine: Wick-permutation sums, derivative-to-momentum replacement,
  and open-index remapping.
- `src/model.py`
  Model-layer declarations: `Field`, `GaugeGroup`, `InteractionTerm`, and
  covariant/kinetic term declarations.
- `src/gauge_compiler.py`
  Covariant compiler: expands gauge-covariant kinetic terms into ordinary
  `InteractionTerm`s.
- `src/spenso_structures.py`, `src/operators.py`
  Tensor wrappers such as gamma matrices, metrics, and gauge generators.

## Conventions

A common textbook form is:

```text
D_mu = partial_mu - i sum_G g_G A^a_{G,mu} T^a_R
```

The physical compiler in this repository uses:

```text
D_mu = partial_mu + i g A_mu
```

Only the overall sign is conventional. The structural point is unchanged:

```text
D_mu = partial_mu + representation-dependent gauge action
```

Derivatives become momentum factors in Fourier space:

```text
partial_mu -> -i p_mu
```

## Core Rule

Do not hardcode one universal `D_mu`.

Instead, treat the covariant derivative as a function of field metadata:

```text
D_mu = partial_mu + i sum_G g_G A^a_{G,mu} T^a_R
```

where:

- the gauge declaration provides `g_G`, `A^a_{G,mu}`, abelian/non-abelian
  status, and generator metadata
- the field declaration provides the representation `R`, the relevant index
  slots, and abelian charges
- the compiler builds the concrete gauge action from those two pieces

This is the right mental model for the codebase:

- `QuantumNumbers` carry abelian charges
- field indices carry non-abelian representation slots
- gauge groups declare the generator structure acting on those slots

## Two-Layer Split

### Layer A: gauge-group data

- gauge boson `A_mu^a`
- coupling `g`
- abelian vs non-abelian
- generator object `T^a` or charge label
- structure constants for non-abelian groups

### Layer B: field transformation data

- scalar / fermion / vector kind
- self-conjugate vs non-self-conjugate
- abelian charges
- non-abelian representation type
- concrete field slots on which the group acts
- for repeated identical index kinds: whether to pick one slot or sum over all
  matching slots

Once those two layers exist, the derivative builder only needs to read metadata
and sum the active contributions.

## Representation-Dependent Action

For an abelian group, the generator reduces to the field charge:

```text
U(1): T_R -> q
```

For a non-abelian group, it is the generator acting in the field
representation:

```text
SU(N): T^a_R
```

So the same abstract rule gives different concrete derivatives.

Complex scalar with charge `q`:

```text
D_mu phi = (partial_mu + i e q A_mu) phi
```

Dirac fermion with charge `q`:

```text
D_mu psi = (partial_mu + i e q A_mu) psi
```

Fundamental of `SU(3)`:

```text
(D_mu phi)^i = partial_mu phi^i + i g_s G_mu^a (T^a)^i{}_j phi^j
```

Adjoint of `SU(N)`:

```text
(D_mu X)^a = partial_mu X^a + g f^{abc} A_mu^b X^c
```

Same symmetry principle, different representation action.

## Build Rule

Conceptually, the derivative builder should do this:

```text
start from partial_mu phi
for each gauge group G:
    if phi is neutral or singlet under G:
        add nothing
    elif G is abelian:
        add + i g q A_mu phi
    else:
        add + i g A_mu^a T^a_R phi
```

This is the reusable core.

The builder should not branch too early into a "scalar case" or "fermion case".
That distinction matters when the full kinetic term is assembled around `D_mu`,
not when the gauge action itself is determined.

## Conjugate Fields

This is easy to get wrong and should stay explicit.

If `phi` transforms in `R`, then `phi^dagger` transforms in the conjugate
representation `R*`.

So:

- abelian charges change sign
- non-abelian generators become those of the conjugate representation

For scalar QED in this repository's sign convention:

```text
D_mu phi        = (partial_mu + i e q A_mu) phi
D_mu phi^dagger = (partial_mu - i e q A_mu) phi^dagger
```

So the compiler must not blindly reuse the same gauge action for both a field
and its daggered occurrence.

## How Expansion Produces Interaction Terms

One covariant kinetic term expands into several ordinary interactions.

For a complex scalar,

```text
(D_mu phi)^dagger (D^mu phi)
```

produces:

- the free kinetic term
- mixed derivative-gauge terms
- a two-gauge contact term

So one covariant kinetic term naturally generates both:

- 3-point current vertices
- 4-point contact vertices

That is why the compiler output splits into current and contact contributions.

## Status as of 2026-04-08

### Implemented

- `GaugeRepresentation` supports repeated-slot semantics through:
  - `slot`
  - `slot_policy="unique"`
  - `slot_policy="sum"`
  - `slots_for(...)` and related helpers
- `GaugeGroup` and the gauge compiler resolve all active matching slots instead
  of assuming a single unique slot
- `Field.pack_slot_labels(...)` preserves ordinal stability for repeated kinds by
  keeping explicit `None` placeholders
- minimal and covariant gauge compilers insert spectator identities
  consistently on inactive repeated slots
- the covered bislot scalar case expands into:
  - one current pair per active slot
  - one contact contribution per ordered slot pair
- ambiguous repeated-slot cases are rejected by default and are only summed when
  metadata opts in via `slot_policy="sum"`
- initial dedicated `pytest` coverage exists in
  `tests/test_covariant_bislot_sum.py`

### Still Missing

- mixed-group complex-scalar kinetic terms still miss cross-group two-gauge
  contact terms
- most covariant regression coverage still lives in `src/examples.py` instead of
  the dedicated `tests/` tree

## Where Covariant Expansion Happens

### Covariant compiler: `src/gauge_compiler.py`

Key entry points:

- `compile_covariant_terms(model)`
  Flattens declared covariant kinetic terms into standard `InteractionTerm`s.
- `compile_dirac_kinetic_term(model, DiracKineticTerm)`
  Expands the gauge part of `psibar i gamma^mu D_mu psi`.
- `compile_complex_scalar_kinetic_term(model, ComplexScalarKineticTerm)`
  Expands the gauge part of `(D_mu phi)^dagger (D^mu phi)` into current and
  contact terms.

### Engine side: `src/model_symbolica.py`

- `InteractionTerm.to_vertex_kwargs(external_legs)`
  Produces the kwargs consumed by `vertex_factor(...)`.
- `vertex_factor(...)`
  Calls `contract_to_full_expression(...)` and applies global factors.
- `contract_to_full_expression(...)`
  Handles derivative-to-momentum replacement and open-label remapping.

This is where kinetic-term derivatives become momentum factors.

## What Repeated-Slot Generalization Had to Fix

The issue is not gauge invariance itself. The issue is how to represent
tensor-product actions when a field carries several slots of the same kind.

### 1. Single-slot assumption

Old behavior effectively assumed that the relevant representation acts on one
selected slot.

That fails for fields with repeated matching slots, because the correct gauge
action may be a sum over tensor-product factors.

### 2. Same-slot-only contact terms

Old scalar contact construction only covered the case where both gauge legs act
on the same slot.

For repeated slots, the full result must also include cross-slot placements:

- first gauge leg acting on slot `i`
- second gauge leg acting on slot `j`
- including `i != j`

### 3. Ordinal drift in packed labels

If repeated-kind labels are packed without preserving the full slot shape,
ordinal lookup becomes unstable.

That breaks:

- open-label detection
- remapping of coupling labels to external-leg labels

## Design Rule for Repeated Slots

There are three parts:

### A. Slot policy

`GaugeRepresentation` should express one of two behaviors:

- `slot_policy="unique"`
  Repeated matches are ambiguous unless one slot is explicitly chosen.
- `slot_policy="sum"`
  When `slot` is not given, the representation acts on all matching slots and
  the compiler sums the contributions.

### B. Current terms

For fermion and scalar currents:

- determine all active matching slots
- for each active slot:
  - build the generator on that slot
  - build spectator identities on the inactive slots
  - emit the corresponding contribution

### C. Contact terms

For scalar contact terms, sum over ordered slot pairs `(slot_i, slot_j)`:

- if `slot_i == slot_j`, use the usual chained-generator structure
- if `slot_i != slot_j`, place one generator on each slot and keep spectator
  identities on the rest

### D. Stable label packing

`Field.pack_slot_labels(...)` should always preserve the full repeated-slot
shape, using `None` placeholders for unspecified positions.

That keeps ordinals stable and open-label remapping correct.

## Recommended Next Steps

1. Freeze covariant conventions in one stable source of truth across docs and
   tests.
2. Move the main covariant regression checks out of `src/examples.py` into
   `tests/`.
3. Extend the repeated-slot machinery to the still-missing mixed-group scalar
   contact case.
4. Keep `src/model_symbolica.py` generic:
   - no tensor-specific branching in the engine
   - all generality should come from correct couplings and labels emitted by the
     compiler layer

## Bottom Line

The right abstraction is:

```text
covariant derivative = partial derivative + representation-dependent gauge action
```

and the right compiler rule is:

```text
one covariant kinetic term -> several ordinary interaction terms
```

For repeated identical index kinds, the hard part is not the physics. The hard
part is expressing the tensor-product action cleanly in metadata, compiler
output, and label bookkeeping.

That is why the repeated-slot generalization belongs in the
declaration/compiler layer, not in the contraction engine.
