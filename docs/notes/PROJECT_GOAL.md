## Project Goal

Implement a Python analogue of FeynRules on top of Symbolica + Spenso.

### Core objective

From model declarations, derive Feynman vertices through one reproducible pipeline:

- Symbolica: symbolic algebra, rewriting, simplification
- Spenso: tensor/index structures (Lorentz, spinor, gauge)

### Scope in this repository

- ordinary (non-BFM) gauge-theory baseline
- declarative Lagrangian front end lowered into existing compiler back end
- reproducible tests for covariant matter, pure gauge, gauge fixing, and ghosts

### Not yet in scope as completed work

- full BFM support (background/quantum split, BFM gauge fixing, BFM ghosts)
- full electroweak/SSB pipeline
- broad multi-fermion generality comparable to mature FeynRules ecosystems

### Success criterion

A user can declare a model once, compile interactions consistently, and extract vertices with stable conventions and test-backed behavior.
