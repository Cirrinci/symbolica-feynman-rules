## Symbolica + Spenso Feynman-Rule Prototype

This repository explores a Python-based analogue of FeynRules built around:

- Symbolica for symbolic expressions, rewriting, and simplification
- Spenso for tensor structures, spinor/Lorentz objects, and index-aware building blocks

The current codebase is centered on a Symbolica contraction engine plus a thin
model layer that maps FeynRules-style declarations into that engine.

The package boundary follows the FeynRules split:

- `feynpy` is the reusable toolkit/engine
- `theories` contains concrete theory definitions, analogous to model files
- `feynrules` contains external FeynRules export parsers and comparison helpers

### Repository layout

Live source code is organized as split packages rather than flat top-level files:

- `src/feynpy/`
  - public FeynPy engine API and implementation modules
  - metadata/declarations, compiled interaction objects, lowering, validation,
    and field transformations
- `src/theories/`
  - concrete theory definitions built on top of the engine
  - the gauge-basis-to-broken Standard Model builder in `standard_model.py`
- `src/feynrules/`
  - generic adapters for parsing and comparing FeynRules vertex exports
  - model-specific routing belongs in `theories`, not in the engine package
- `src/compiler/`
  - convention-fixed gauge / covariant compilation
  - public compiler entry points in `gauge.py`
  - internal helper modules for covariant cores, spectators, and matter-action builders
- `src/symbolic/`
  - Symbolica contraction engine in `vertex_engine.py`
  - post-processing helpers in `vertex_postprocessing.py`
  - Spenso-backed tensor wrappers in `spenso_structures.py`
  - tensor canonicalization helpers in `tensor_canonicalization.py`
- `src/lagrangian/`
  - reusable operator builders such as bilinears, currents, and gauge-contact structures
- `examples/`
  - runnable example/regression scripts
  - includes flavor-expansion, SU(2), electroweak, and full Standard Model examples
- `tests/`
  - the main regression suite
- `docs/notes/RESEARCH_LOG.md`
  - tracked research log; other working notes stay local and out of git

Walkthrough notebooks live under `notebooks/`.

### Canonical notebooks for the modern API

The current cleanup target is to keep only the modern public API illustrated in
the notebooks below. When older entry points or compatibility layers disagree
with these notebooks, treat the notebook workflow as the source of truth.

- `notebooks/list_lagrangians.ipynb`
- `notebooks/field_strength_operators_walkthrough.ipynb`
- `notebooks/flavor_expansion_walkthrough.ipynb`
- `notebooks/index_handling.ipynb`
- `notebooks/operator_action_and_symbolica_walkthrough.ipynb`

`notebooks/codebase_workflow_walkthrough.ipynb` is the next workflow notebook to
bring into line with that modern API surface.

### Current status

What is already solid in the active code path:

- local scalar, fermion, and mixed-species interaction lowering
- derivative interactions with explicit derivative-target bookkeeping
- fermion sign handling for the currently supported closed-bilinear structures
- stripped and unstripped external-leg output through `feynman_rule(...)`
- Spenso-backed gamma matrices, Lorentz metrics, generators, and structure constants
- declarative source syntax around:
  - `Gamma(...)`
  - `CovD(...)`
  - `FieldStrength(...)`
  - `GaugeFixing(...)`
  - `GhostLagrangian(...)`
- convention-fixed covariant compilation for:
  - `i psibar gamma^mu D_mu psi`
  - `(D_mu phi)^dagger (D^mu phi)`
  - abelian and non-abelian gauge kinetic terms
  - ordinary gauge fixing
  - ordinary Faddeev-Popov ghost terms
- broad `pytest` coverage across:
  - QED / QCD covariant compilation
  - repeated-slot gauge actions
  - mixed-group scalar contacts
  - gauge-fixing and ghost compilation
  - declarative Lagrangian lowering
  - flavor-class declarations and selective flavor expansion
  - pure-gauge canonicalization
  - general field transformations and the broken Standard Model
- simultaneous FeynRules-style field definitions with fixed-point dependency
  handling, conjugation, component decomposition, and derivative propagation
- a broken-phase Standard Model generated from gauge-basis declarations,
  including CKM, Yukawa, Higgs/Goldstone, gauge, QCD, and ghost sectors
- FeynRules-style flavor classes through
  `dirac_field(..., class_members=..., flavor_index=...)` and
  selective `flavor_expand=...`
- monomial-wide validation of explicit symbolic index labels across local
  lowering; the `Model(..., parameters=...)` path also validates indexed
  parameter labels against their declared index spaces

What is still under development:

- general multi-fermion tensor support beyond the currently supported ordered closed-bilinear structures
- broader model-validation checks in the FeynRules style
  - hermiticity
  - kinetic normalization
  - mass-diagonalization / mass-spectrum consistency
- remaining maintainability work in `src/feynpy/lowering.py`
- some integration-style assertions still live in `examples/` instead of focused tests

### Conventions and entry points

Frozen compiler conventions are summarized below and reinforced by focused tests.

User-facing entry point:

- `Model(...).lagrangian().feynman_rule(...)`
  - declare a `Model`, compile it, and extract a chosen vertex

Internal symbolic entry points (used underneath the model layer):

- `symbolic.vertex_engine.contract_to_full_expression(...)`
  - low-level contraction/permutation engine
- `symbolic.vertex_engine.vertex_factor(interaction=..., external_legs=...)`
  - vertex extraction over a compiled `InteractionTerm`
- `symbolic.vertex_engine.simplify_vertex(...)`
  - post-processing helper
- `symbolic.vertex_engine.simplify_deltas(...)`
- `symbolic.vertex_engine.simplify_spinor_indices(...)`

Important output conventions:

- `strip_externals=True` by default
  - external wavefunctions are amputated from the displayed vertex
- high-level `Lagrangian.feynman_rule(...)` / `Model(...).lagrangian().feynman_rule(...)`
  use `include_delta=False` by default
  - the returned expression omits the universal momentum-conservation factor
    unless you request it explicitly
- low-level `vertex_factor(...)` still defaults to `include_delta=True`
- `simplify_vertex(..., simplify_gamma=False)` keeps gamma-chain simplification opt-in
- use `include_delta=True` when you want to keep the universal
  `(2*pi)^d Delta(sum p)` factor in the displayed rule

Gauge/compiler conventions:

- derivatives map to `-i p_mu`
- `vertex_factor(...)` contributes the universal overall `+i`
- matter uses `D_mu = partial_mu - i g A_mu`
- pure gauge uses
  `F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu + g f^{abc} A^b_mu A^c_nu`
- ordinary gauge fixing uses
  `L_gf = -(1/2 xi) (partial.A)^2`
- ordinary non-abelian ghosts use the integrated form
  `L_gh = (partial cbar)(partial c) + g f (partial cbar) A c`

### Recommended workflow

`Model(...)` is the single source-declaration front door. It accepts both
local/already-expanded operators and metadata-dependent declarations, and
`model.lagrangian()` compiles them into the term container on which you call
`feynman_rule(...)`:

- Local operators: explicit products of fields, `PartialD(...)`, `Gamma(...)`,
  `Metric(...)`, `T(...)`, and `StructureConstant(...)`.
- Metadata-dependent declarations: `CovD(...)`, `FieldStrength(...)`, gauge
  kinetic terms, `GaugeFixing(...)`, `GhostLagrangian(...)`, and any
  declaration that needs charges, representations, gauge-boson assignments, or
  ghost-field metadata.

Field transformations are applied after metadata-dependent compilation:

```python
broken = model.transform_fields(
    FieldTransformation(B, -sw * Z + cw * A),
)
```

See `docs/FIELD_TRANSFORMATIONS.md` for simultaneous/fixed-point semantics,
conjugation, index handling, derivative propagation, and the ordering relative
to covariant-derivative expansion.

There is no separate `Lagrangian` source class. `model.lagrangian()` compiles a
declaration into a `CompiledLagrangian` term container, which is also
constructible directly as `CompiledLagrangian(terms=...)` from already-lowered
internal compiled terms.

For compact FeynRules-style index notation, plain symbols such as
`f, h, col = S("f", "h", "col")` are still valid. During lowering, each label
name is bound to one index space per monomial, so reusing one name across
incompatible slots is rejected. Full parameter-aware label checks use the
`Model(..., parameters=..., lagrangian_decl=...)` path, which carries a
parameter table.

For metadata-free local operators, pass them straight to `Model(...)`:

```python
model = Model(
    g4 * psi.bar * Gamma(mu) * psi * chi.bar * Gamma(mu) * chi
)

vertex = model.lagrangian().feynman_rule(psi.bar, psi, chi.bar, chi)
```

For model declarations that need gauge metadata, use `Model(..., lagrangian_decl=...)`:

```python
model = Model(
    gauge_groups=(SU3C,),
    fields=(q, G, ghG),
    lagrangian_decl=(
        I * q.bar * Gamma(mu) * CovD(q, mu)
        - Expression.num(1) / Expression.num(4)
        * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)
        + GaugeFixing(SU3C, xi=xiQCD)
        + GhostLagrangian(SU3C)
    ),
)

vertex = model.lagrangian().feynman_rule(q.bar, q, G)
```

### Local DSL scope

The local tensor helpers `T(...)` and `StructureConstant(...)` are currently
limited placeholders for already-expanded monomials:

- they are useful when you want to write an explicit local color / gauge tensor
  structure by hand,
- they do not by themselves select a gauge group or representation,
- they do not infer normalization conventions from `GaugeGroup` metadata,
- they should not be read as fully generic group-aware objects yet.

When the interaction should be derived from declared gauge data, prefer
`Model(..., lagrangian_decl=...)` and let the compiler build the corresponding
generator or structure-constant insertions from `GaugeGroup` metadata.

The lower-level engine `contract_to_full_expression(...)` still accepts
parallel-list contraction input (`coupling`, `alphas`, `betas`, `ps`, roles,
index labels, and derivative targets), but this is an internal interface used
underneath the model layer. User code should go through `Model(...)` and
`feynman_rule(...)`; the removed direct `vertex_factor(coupling=..., alphas=...)`
entry point now raises.

Legacy split declaration slots such as `covariant_terms`,
`gauge_kinetic_terms`, `gauge_fixing_terms`, and `ghost_terms` are still
supported for compatibility, but `lagrangian_decl=` is the preferred source
entry point.

### Setup

Create or refresh the local virtual environment:

- `bash setup_env.sh`

This creates `.venv` and installs the pinned runtime and validation
dependencies listed in `requirements.txt`.

### Usage

Run the main example/regression scripts from the repository root:

- `./.venv/bin/python -m examples.examples_flavor_expansion`
- `./.venv/bin/python examples/examples_su2.py`
- `./.venv/bin/python examples/examples_electroweak_unbroken.py`
- `./.venv/bin/python examples/examples_standard_model.py`
- `./.venv/bin/python src/symbolic/spenso_gamma_checks.py`
- `./.venv/bin/python -m pytest -q`

For notebooks, use the repository virtual environment:

- `.venv/bin/python`

### Notes and roadmap

The tracked project note is:

- `docs/notes/RESEARCH_LOG.md`

Other working notes and planning material are kept locally and are not part of
the published repository.

### Immediate priorities

Near-term themes:

1. keep conventions and validation checks aligned with the test suite
2. continue shrinking large multi-responsibility modules such as `src/feynpy/lowering.py`
3. reduce string-based equality/canonical-form workarounds where Symbolica/Spenso can do the job directly
4. extend physics-facing validation and user-facing diagnostics without changing conventions implicitly
