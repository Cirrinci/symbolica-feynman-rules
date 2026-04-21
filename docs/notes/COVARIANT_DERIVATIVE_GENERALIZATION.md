# Covariant-Derivative Generalization

Purpose: technical rulebook for representation-dependent covariant expansion.

## Core rule

Do not use one universal hard-coded derivative. Use field metadata and gauge metadata:

`D_mu = partial_mu + i sum_G g_G A_{G,mu}^a T_R^a`

with:

- abelian case: `T_R -> q`
- non-abelian case: `T_R -> representation generator`

## Repository map

The active implementation boundary is now:

- declaration and lowering: `src/model/*`, `src/lagrangian/*`
- gauge/covariant compilation: `src/compiler/gauge.py`
- symbolic vertex extraction: `src/symbolic/vertex_engine.py`

The important split is unchanged:

- declaration/lowering decides what gauge action a field carries
- compiler emits normalized interaction terms
- symbolic engine stays generic and turns derivatives into momentum factors

## Build rule

Conceptually, covariant expansion should do this:

1. start from the ordinary derivative term
2. inspect each active gauge group on the field
3. insert the abelian charge or non-abelian generator dictated by metadata
4. emit one ordinary interaction contribution per active current/contact branch

That means the scalar/fermion distinction belongs mainly to the surrounding
kinetic term, not to the representation-dependent gauge action itself.

## Repeated-slot requirement

For fields carrying repeated slots of the same index kind, expansion must be metadata-driven:

- `slot_policy="unique"`: ambiguous repeated matches are rejected unless a slot is explicit.
- `slot_policy="sum"`: sum contributions over all active matching slots.

## Contact-term rule

For scalar contact terms, sum ordered slot pairs `(i, j)`:

- `i = j`: same-slot chained-generator branch
- `i != j`: cross-slot branch with one generator per slot

## Interaction-generation rule

One covariant kinetic term lowers to several ordinary interactions:

- fermion kinetic term produces current vertices
- complex-scalar kinetic term produces current vertices and two-gauge contact terms
- multi-group scalar cases also produce ordered mixed-group contact contributions

This is why the compiler should lower covariant declarations into ordinary
interaction objects rather than trying to keep a universal symbolic `D_mu`
alive all the way to vertex extraction.

## Conjugate-field rule

If a field is in representation `R`, the conjugate occurrence uses `R*`:

- abelian charge flips sign
- non-abelian generator is conjugate representation action

## Failure modes this note is meant to prevent

The main bugs in this area are structural, not conceptual:

1. single-slot assumption
   repeated matching slots cannot be treated as one implicit slot
2. same-slot-only contact construction
   scalar contact expansion must include cross-slot placements as well
3. unstable packed-label ordinals
   repeated-kind labels must preserve slot shape so remapping stays correct

## Current status (2026-04-21)

Implemented for covered compiler paths:

- repeated-slot resolution with explicit policy
- spectator identities on inactive repeated slots
- ordered contact expansion for bislot scalar cases
- mixed-group scalar contact contributions
- ordinal-stable repeated-slot label packing for covered lowering/compiler flows

Still to harden:

- broader regression extraction from examples into tests
- stricter validation outside core compiler entry points

## Recommended rule of thumb

Keep repeated-slot meaning in metadata and compiler output. Do not push
slot-specific physics into the generic symbolic engine.
