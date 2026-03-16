## Symbolica-based Feynman Rule Prototype

This repository contains a prototype implementation of a FeynRules-like
workflow in Python, built on top of **Symbolica** and **Spenso**.
The current focus is on **scalar field theories** and **derivative
interactions**, with a clear path to extend to fermions and gauge fields.

### Current capabilities

- **Field and parameter metadata**
  - `Field`: scalar vs fermion, indexed vs non-indexed, conjugation, mass.
  - `Parameter`: couplings and masses as Symbolica symbols.

- **Lagrangian terms**
  - `LagrangianTerm` objects store:
    - the raw Symbolica expression `expr`,
    - an explicit `coefficient`,
    - a list of `OperatorFactor`s describing each field insertion,
      including derivative indices.
  - `Lagrangian` collects terms and builds the total Lagrangian.

- **Bosonic interaction vertices**
  - Support for several scalar interactions:
    - mass term, `phi^4`, `phi^6`,
    - mixed `phi^2 chi^2`,
    - multi-species sextic `phi_i^2 phi_j^2 phi_k^2`.
  - **Derivative interactions**:
    - e.g. `-(g) (∂_μ φ) φ (∂^μ χ) χ` with correct momentum factors.
    - Derivatives on a factor are encoded as `OperatorFactor(..., derivative_indices=(mu,))`.
  - Two vertex extractors:
    - `canonical_vertex`: brute-force sum over all valid contractions (n!),
    - `fast_bosonic_vertex`: grouped/combinatorial version that avoids n! and
      has been tested to agree with `canonical_vertex` (after expansion) for
      the current scalar benchmarks.

- **Kinetic + mass term sanity checks**
  - The quadratic kernel from the kinetic term alone is proportional to `i p^2`
    (once momentum conservation is used, e.g. `p2 = -p1` for a 2-point function).
  - Including the mass term gives a kernel proportional to `i (p^2 - m^2)` up
    to overall sign conventions for `L_int` vs `H_int`.

- **CAS-assisted parsing (optional / experimental)**
  - `code/cas_vertex.py` provides:
    - a `FieldRegistry` to recognize field occurrences in a Symbolica
      expression,
    - `split_coefficient_and_factors` to reconstruct `OperatorFactor`s from
      a monomial,
    - `lagrangian_terms_from_expr` and `cas_vertex_bosonic` to go directly
      from a raw Lagrangian expression `L_expr` to vertices, by reusing
      `fast_bosonic_vertex`.
  - This layer is **optional**: the core vertex algorithm lives in
    `code/model.py` and works independently.

### File overview

- `code/model.py`
  - Core data structures:
    - `Field`, `Parameter`, `OperatorFactor`, `ExternalLeg`,
      `LagrangianTerm`, `Lagrangian`.
  - Vertex extraction:
    - `valid_contractions`, `derivative_factor_for_leg`,
      `ContractedTerm`, `canonical_vertex`, `fast_bosonic_vertex`.

- `code/examples_scalar.py`
  - Concrete scalar fields (`Phi`, `Chi`, `Phi_i`, …) and parameters.
  - Example Lagrangian terms: mass, `phi^4`, `phi^6`, `phi^2 chi^2`,
    multi-species sextic, Yukawa-like terms, derivative interaction.
  - Benchmarks and assertions for:
    - combinatorics (number of contractions),
    - expected vertex prefactors,
    - agreement between `canonical_vertex` and `fast_bosonic_vertex`.

- `code/cas_vertex.py` (experimental)
  - CAS-driven utilities to parse Symbolica expressions into
    `LagrangianTerm`s and call the existing vertex machinery.

- `Notebooks/test_phi4.ipynb`
  - Interactive notebook to:
    - visualize the total scalar Lagrangian,
    - inspect specific vertices (`phi^4`, `phi^2 chi^2`, derivative term),
    - check 2-point kernels for the kinetic + mass term,
    - compare manual vs CAS-based vertex extraction on a derivative example.

### Roadmap
Short term:

- Fermions and ghosts

  - Implement fermion_reordering_sign to correctly account for Grassmann minus signs in canonical_vertex and fast_bosonic_vertex.
Add simple fermionic benchmarks (e.g. Yukawa vertex) with sign-sensitive permutations.

- Vector / gauge fields

  - Introduce vector fields with Lorentz indices.
Extend index_factor in ContractedTerm to build tensor structures: metrics, epsilon tensors, gamma chains, color tensors, etc.
Derive propagators by inverting the quadratic kernel in field space (not via the interaction vertex combinatorics).

Medium term:

- More CAS automation

  - Make lagrangian_terms_from_expr robust for larger models so that most users only need to write a Symbolica Lagrangian, not explicit factors lists.
