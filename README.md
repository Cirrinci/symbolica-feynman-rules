## Symbolica + Spenso Feynman-Rule Prototype

This repository explores a Python-based analogue of FeynRules built around:

- Symbolica for symbolic expressions, rewriting, and simplification
- Spenso for tensor structures, spinor/Lorentz objects, and index-aware building blocks

The current codebase is centered on a Symbolica contraction engine plus a thin
model layer that maps FeynRules-style declarations into that engine.

### Repository layout

Live source code is organized as split packages rather than flat top-level files:

- `src/model/`
  - model metadata and declarations
  - `Field`, `GaugeGroup`, `GaugeRepresentation`, `Model`
  - compiled interaction objects, `Lagrangian`, and lowering from declarative source terms
  - electroweak symmetry-breaking helpers in `ssb.py`
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
  - includes the general examples, the declarative Lagrangian examples, SU(2), and electroweak examples
- `tests/`
  - the main regression suite
- `docs/notes/`
  - conventions, roadmap material, and project notes

Walkthrough notebooks live under `notebooks/`.

### Current status

What is already solid in the active code path:

- local scalar, fermion, and mixed-species interaction lowering
- derivative interactions with explicit derivative-target bookkeeping
- fermion sign handling for the currently supported closed-bilinear structures
- stripped and unstripped external-leg output through `vertex_factor(...)`
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
  - pure-gauge canonicalization
  - electroweak unbroken and SSB examples

What is still under development:

- general multi-fermion tensor support beyond the currently supported ordered closed-bilinear structures
- broader model-validation checks in the FeynRules style
  - hermiticity
  - kinetic normalization
  - mass-diagonalization / mass-spectrum consistency
- remaining maintainability work in `src/model/lowering.py`
- some integration-style assertions still live in `examples/` instead of focused tests

### Conventions and entry points

Frozen compiler conventions are documented in:

- `docs/notes/CONVENTIONS.md`

Core symbolic entry points:

- `symbolic.vertex_engine.contract_to_full_expression(...)`
  - low-level contraction/permutation engine
- `symbolic.vertex_engine.vertex_factor(...)`
  - high-level vertex extraction façade
- `symbolic.vertex_engine.simplify_vertex(...)`
  - high-level post-processing helper
- `symbolic.vertex_engine.simplify_deltas(...)`
- `symbolic.vertex_engine.simplify_spinor_indices(...)`

Important output conventions:

- `strip_externals=True` by default
  - external wavefunctions are amputated from the displayed vertex
- `include_delta=True` by default
  - the returned expression keeps the overall momentum-conservation factor
    `(2*pi)^d Delta(sum p)`
- `simplify_vertex(..., simplify_gamma=False)` keeps gamma-chain simplification opt-in
- use `include_delta=False` when you want the reduced vertex with the universal momentum delta stripped

Gauge/compiler conventions:

- derivatives map to `-i p_mu`
- `vertex_factor(...)` contributes the universal overall `+i`
- matter uses `D_mu = partial_mu - i g A_mu`
- pure gauge uses
  `F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu`
- ordinary gauge fixing uses
  `L_gf = -(1/2 xi) (partial.A)^2`
- ordinary non-abelian ghosts use the integrated form
  `L_gh = (partial cbar)(partial c) - g f (partial cbar) A c`

### Recommended workflows

Choose the front door based on whether the source term is already local and
expanded, or whether it still needs model metadata:

- Use `Lagrangian(...)` for local/already-expanded operators.
  - Good fit: explicit products of fields, `PartialD(...)`, `Gamma(...)`,
    `Metric(...)`, `T(...)`, and `StructureConstant(...)`.
  - `Lagrangian(...)` does not consult `GaugeGroup` metadata and does not
    compile covariant derivatives, gauge kinetic terms, gauge fixing, or ghost
    sectors.
- Use `Model(..., lagrangian_decl=...)` for metadata-dependent declarations.
  - Good fit: `CovD(...)`, `FieldStrength(...)`, gauge kinetic terms,
    `GaugeFixing(...)`, `GhostLagrangian(...)`, and any declaration that needs
    charges, representations, gauge-boson assignments, or ghost-field
    metadata.

For metadata-free local operators, use `Lagrangian(...)` directly:

```python
L = Lagrangian(
    g4 * psi.bar * Gamma(mu) * psi * chi.bar * Gamma(mu) * chi
)

vertex = L.feynman_rule(psi.bar, psi, chi.bar, chi)
```

This path is intended for terms that are already written in local form. It is
not the right entry point for `CovD(...)`, `FieldStrength(...)`,
`GaugeFixing(...)`, or `GhostLagrangian(...)`.

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

Use the lower-level direct engine only when you explicitly want to supply the
parallel-list contraction input (`coupling`, `alphas`, `betas`, `ps`, roles,
index labels, and derivative targets) yourself.

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

- `./.venv/bin/python examples/examples.py --suite all`
- `./.venv/bin/python examples/examples.py --suite scalar`
- `./.venv/bin/python examples/examples.py --suite fermion`
- `./.venv/bin/python examples/examples.py --suite gauge`
- `./.venv/bin/python examples/examples.py --suite model`
- `./.venv/bin/python examples/examples.py --suite covariant`
- `./.venv/bin/python examples/examples.py --suite cross`
- `./.venv/bin/python examples/examples_lagrangian.py --suite all`
- `./.venv/bin/python examples/examples_lagrangian.py --suite covariant --skip-tests`
- `./.venv/bin/python examples/examples_su2.py`
- `./.venv/bin/python examples/examples_electroweak_unbroken.py`
- `./.venv/bin/python examples/examples_electroweak_ssb.py`
- `./.venv/bin/python src/symbolic/spenso_gamma_checks.py`
- `./.venv/bin/python -m pytest -q`

For notebooks, use the repository virtual environment:

- `.venv/bin/python`

### Notes and roadmap

Project notes are kept in:

- `docs/notes/CONVENTIONS.md`
- `docs/notes/FEYNRULES_STYLE_STRATEGY.md`
- `docs/notes/code_review_roadmap.md`
- `docs/notes/PROJECT_GOAL.md`
- `docs/notes/ROADMAP.md`
- `docs/notes/RESEARCH_LOG.md`
- `docs/notes/THESIS_PROGRESS.md`

The current day-to-day engineering roadmap is `docs/notes/code_review_roadmap.md`.

### Immediate priorities

The current implementation priorities are tracked in `docs/notes/code_review_roadmap.md`.

Near-term themes:

1. keep conventions and validation checks aligned with the test suite
2. continue shrinking large multi-responsibility modules such as `src/model/lowering.py`
3. reduce string-based equality/canonical-form workarounds where Symbolica/Spenso can do the job directly
4. extend physics-facing validation and user-facing diagnostics without changing conventions implicitly
