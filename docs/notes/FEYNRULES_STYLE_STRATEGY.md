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
- make index signatures more central to compatibility checks
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

1. remove string-based logic from `src/model_symbolica.py`
2. repair the stale sandbox and validation path
3. strengthen `src/model.py`
4. centralize operator builders
5. add a compiler layer
6. broaden gauge support

## Immediate concrete next task

The next concrete step should be:

Repair the object boundary between `src/model.py`, `src/spenso_structures.py`,
and `src/model_symbolica.py`.

In practice, that means:

1. eliminate stringified matching in the engine
2. give field roles/index signatures stronger semantics
3. move hand-built current structures out of `src/examples.py`

If those three pieces are done, the project will be in a much stronger position
to grow like a real FeynRules-style system instead of a growing prototype.
