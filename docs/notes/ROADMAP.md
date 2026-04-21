## Roadmap

Purpose: forward plan only. Historical details live in `RESEARCH_LOG.md`.

## Current baseline (already working)

- declarative `lagrangian_decl` front end
- covariant matter compilation
- pure-gauge Yang-Mills 2/3/4-point compilation
- ordinary gauge fixing and non-abelian ghosts
- dedicated regression tests for core compiler matrix
- SU(2)L example + dedicated test coverage

## Next implementation phases

### Phase 1: compiler hardening (short term)

1. Expand regression coverage for recent gauge/lowering refactors.
2. Add parity tests for full-covariant assembly paths.
3. Reduce example-driven assertions; keep `examples/` primarily demonstrative.

### Phase 2: API consistency and ergonomics

1. Add whole-Lagrangian extraction (`feynman_rules(...)`).
2. Make single-term workflows natural (term-level `feynman_rule(...)`).
3. Keep declarative syntax stable while tightening lowering diagnostics.

### Phase 3: BFM entry layer

1. Introduce background/quantum gauge-field split in declarations.
2. Extend pure-gauge compiler with split-aware expansion while preserving ordinary path.
3. Add BFM-oriented test matrix before widening physics scope.

### Phase 4: broader physics growth

1. BFM gauge fixing and ghosts.
2. wider fermion/tensor structures.
3. later SSB and electroweak extensions as thesis scope allows.

## Priority order

1. correctness and tests
2. API clarity
3. new physics features

