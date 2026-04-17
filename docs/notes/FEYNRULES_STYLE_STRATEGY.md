# FeynRules-Style Strategy for Symbolica + Spenso

## Why this strategy

The repository already has the right split in outline:

- `src/model_symbolica.py` as the engine
- `src/model.py` as the declaration layer
- `src/spenso_structures.py` as the tensor vocabulary

The problem is that this split is not complete yet. Some engine decisions still
depend on stringified symbols, and too much physics structure is still assembled
inside `src/examples.py`.

The next step is not another wave of examples. It is to make the current split
real and durable.

## Design target

The long-term target remains FeynRules-like:

1. gauge groups
2. index declarations
3. particle classes
4. parameters
5. Lagrangian terms
6. compilation into normalized interaction objects
7. vertex extraction through one engine

## Current implemented front-end

The repository now has a real declarative entry point:

- `Model(..., lagrangian_decl=...)`
- `CovD(...)`
- `Gamma(...)`
- `FieldStrength(...)`
- `GaugeFixing(...)`
- `GhostLagrangian(...)`

These source declarations are preserved for demos and notebooks, then lowered
to the existing `DiracKineticTerm` / `ComplexScalarKineticTerm` /
`GaugeKineticTerm` / `GaugeFixingTerm` / `GhostTerm` backend.

That is the right boundary for this codebase: a typed public front-end over the
current Symbolica + Spenso execution path, not a second symbolic engine.

## Recommended architecture

## 1) Keep `src/model_symbolica.py` as a pure engine

`src/model_symbolica.py` should remain responsible for:

- contraction permutations
- fermion signs
- derivative momentum factors
- open-index remapping
- low-level simplification calls

It should not absorb more model-specific branching.

Most important cleanup here:

- remove string-based species/index matching
- rely on exact symbolic objects and structural inspection instead

## 2) Strengthen `src/model.py` into a stricter declaration layer

`src/model.py` already has the right core objects:

- `Field`
- `FieldOccurrence`
- `ExternalLeg`
- `DerivativeAction`
- `InteractionTerm`
- `Model`

The next step is to make this layer carry more meaning:

- distinguish scalar, vector, and gauge-field roles properly
- make abelian charges and non-abelian representation slots first-class field metadata
- make index signatures more central to compatibility checks
- make covariant-derivative construction depend on that metadata rather than one hard-coded universal `D_mu`
- reduce the amount of parallel-list logic that leaks into calling code

## 3) Centralize operator builders

Create a dedicated operator vocabulary, either by extending
`src/spenso_structures.py` or by adding a new module such as `src/operators.py`.

That module should provide builders for common structures:

- `psi_bar_psi(...)`
- `psi_bar_gamma_psi(...)`
- `psi_bar_gamma5_psi(...)`
- gauge-current structures
- scalar current structures
- later covariant derivatives and field strengths

This avoids hand-building the same structures differently across examples.

## 4) Add a compiler layer when the current pieces are stable

After the engine/model/tensor boundary is cleaned up, add a higher-level
compiler layer:

- input: model declarations and operator builders
- output: normalized `InteractionTerm` objects

Responsibilities:

- canonicalize dummy labels
- expand covariant-derivative structures
- insert generators and other gauge tensors
- apply normalization conventions consistently

That is the correct place for FeynRules-style compilation logic.

## 5) Make index growth declarative

For every field occurrence, index slots should come from metadata, not from
hard-coded branches.

The intended rule is:

1. field declaration defines intrinsic slots
2. occurrence defines concrete labels
3. engine consumes those slots generically

If a future field or index type requires touching multiple engine conditionals,
the layering is still too weak.

## Suggested implementation order

The last hardening cycle reached the earlier ordinary-gauge goals:

1. a first real regression split now exists in `tests/`
2. one long-lived conventions reference now exists across code/docs/tests
3. repeated same-kind slot handling now works for the covered compiler paths
4. the minimal compiler remains the structural helper and the physical compiler remains the user-facing path
5. ordinary gauge-fixing and ghost sectors now compile through that physical compiler

So the next implementation order is:

1. keep widening the dedicated test split
2. tighten the remaining model/declaration validation outside the current compiler entry points
3. add background/quantum splitting
4. add BFM gauge fixing and ghosts on top of that split
5. only then continue toward broader FeynRules-style workflows

## Immediate concrete next task

The next concrete step should not be another ordinary-gauge feature.

It should be the first clean BFM design pass built on the now-working ordinary
gauge-fixed baseline.

In practice, that means:

1. draft the declaration/model API for ordinary, background, and quantum gauge fields
2. extend the pure-gauge compiler with `A -> B + Q` while preserving the ordinary path
3. add a small BFM-oriented demo/test matrix for the split pure-gauge sector
4. keep the raw-vs-compact display distinction explicit for the resulting gauge-sector output

## Priority after that

Once that hardening step is done, the next physics order should be:

1. background/quantum splitting
2. BFM gauge fixing
3. BFM ghosts
4. broader fermion and symmetry-breaking support

That order is safer because the ordinary gauge-fixed sector is already working
and should remain the base that everything else builds on top of.
