# FeynRules-Style Strategy for Symbolica + Spenso

## Why this strategy

You are already moving in the right direction: `src/model_schema.py` and
`src/examples_metadata.py` show a metadata-first, object-based layer that avoids
the old parallel-list API.

The next step is to align this layer directly with the FeynRules input model so
that adding new indices or fields means adding metadata, not adding custom
conditionals.

## Design target (FeynRules-like)

Mirror FeynRules blocks as Python objects:

1. Gauge groups (`M$GaugeGroups`)
2. Index declarations (`IndexRange`, `IndexStyle`)
3. Particle classes (`M$ClassesDescription`)
4. Parameters (`M$Parameters`)
5. Lagrangian terms (`LGauge`, `LFermions`, `LHiggs`, `LYukawa`, ...)

Then compile those declarations into your existing `InteractionTerm` objects,
which are consumed by `vertex_factor(...)`.

## Key insight from `UnbrokenSM_BFM.fr.txt`

The model file mostly declares *metadata*:

- group structure (abelian/non-abelian, coupling, generators, structure constants)
- index vocabularies (SU2W, Colour, Generation, ...)
- field classes with intrinsic index signatures
- parameters with index signatures
- symbolic Lagrangian pieces

This is exactly the shape your schema layer should own.

## Recommended architecture

## 1) Keep `model_symbolica.py` as a pure engine

`src/model_symbolica.py` should remain responsible for:

- contraction permutations
- fermion signs
- derivative momentum factors
- open-index remapping
- simplification calls

No model-specific branching should be added here.

## 2) Extend `model_schema.py` into a full model declaration layer

Add these new dataclasses:

- `GaugeGroup(name, abelian, coupling, gauge_boson, structure_constant=None, representations=())`
- `IndexFamily(name, size, style, index_type)`
- `Parameter(name, indexed_by=(), complex=False, internal=True, value=None)`
- `Model(name, gauge_groups, index_families, fields, parameters, interactions)`

You already have `Field`, `FieldOccurrence`, `ExternalLeg`, `InteractionTerm`.
Use those as the core and attach model-level containers around them.

## 3) Introduce typed operator builders (replace raw tensor strings)

Create one module (for example `src/operators.py`) with builders returning
Symbolica/Spenso expressions:

- `psi_bar_psi(...)`
- `psi_bar_gamma_psi(mu, ...)`
- `psi_bar_gamma5_psi(...)`
- `covariant_derivative(field, mu, group_context)`
- `field_strength(gauge_field, mu, nu, group_context)`

This avoids ad hoc coupling strings and centralizes conventions.

## 4) Add a compiler layer from model declarations to interaction terms

Create `src/model_compile.py`:

- input: high-level model declarations and Lagrangian builders
- output: normalized `InteractionTerm` objects
- responsibilities:
  - expand covariant derivatives
  - insert generators `T^a_{ij}` and structure constants `f^{abc}`
  - canonicalize dummy labels
  - assign derivatives to target slots
  - apply normalization factors consistently

This is the main bridge to "FeynRules-like input, Symbolica engine output."

## 5) Make index growth declarative

For every field occurrence, all index slots should come from:

1. field intrinsic signature (`Field.indices`)
2. optional conjugate signature
3. occurrence-level concrete labels

No function should special-case "if color and spinor then ...".
All operations should iterate over `ConcreteIndexSlot` entries by type.

## Mapping of FeynRules concepts to your current code

- `M$GaugeGroups` -> new `GaugeGroup` model objects
- `IndexRange/IndexStyle` -> `IndexFamily` + `IndexType`
- `M$ClassesDescription` -> existing `Field` objects (+ metadata extensions)
- `M$Parameters` -> new `Parameter` objects
- `Lagrangian definitions` -> operator builders + compiler -> `InteractionTerm`

## Suggested implementation order (short, safe increments)

1. **Model container only**:
   add `Model`, `GaugeGroup`, `IndexFamily`, `Parameter` dataclasses.
2. **Operator builders**:
   move common current/gamma/gauge structures from examples into reusable helpers.
3. **Compiler MVP**:
   compile a QED-like model to `psibar gamma^mu psi A_mu`.
4. **Non-abelian fermion current**:
   compile `psibar gamma^mu T^a psi G^a_mu`.
5. **Scalar covariant derivative**:
   compile `phi^\dagger D_mu phi` interactions.
6. **Pure gauge terms**:
   compile `F^2` into triple/quartic gauge vertices.

Each step should add tests in metadata mode first, then compare to legacy where
possible.

## Immediate concrete next task

Implement `Model` + `GaugeGroup` + `Parameter` in `model_schema.py` and port the
existing gauge-ready examples in `examples_metadata.py` to be declared through a
single model object.

If that works, you can stop writing manual interaction tuples and start writing
FeynRules-style model declarations.
