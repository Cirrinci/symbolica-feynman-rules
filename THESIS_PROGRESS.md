## Thesis Progress

### Working title

Toward a Python implementation of FeynRules using Symbolica and Spenso

### Core objective

The goal of this project is to build a Python-based system that derives
Feynman interaction vertices from field-theory model definitions, in the spirit
of FeynRules.

The intended division of labor is:

- Symbolica for symbolic expressions, rewriting, simplification, and
  combinatorics
- Spenso for tensor structures, index representations, and high-energy-physics
  objects such as spinor and gauge tensors

### Motivation

The project explores whether a modern Python workflow based on Symbolica and
Spenso can reproduce the essential symbolic tasks usually handled by
FeynRules-like tools:

- expressing interaction terms in a Lagrangian
- handling field permutations and statistics
- converting derivatives into momentum factors
- tracking Lorentz, spinor, and later gauge indices
- extracting final vertex rules in a reusable symbolic form

### What is already achieved

The current prototype already supports:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with correct permutation-aware momentum assignment
- fermionic permutation signs
- mixed scalar-fermion derivative interactions
- stripped and unstripped external wavefunction forms
- spinor-delta structures using Spenso bispinor metrics
- first Spenso-backed gamma-matrix structures
- first axial-current structures using `gamma` and `gamma5`
- explicit open-spinor remapping inside coupling tensors for current examples
- first current-current four-fermion structures with gamma matrices
- matrix-backed consistency checks through the Spenso HEP tensor library

### Current implementation structure

Main files:

- symbolic engine:
  [code/model_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/model_symbolica.py)
- Spenso tensor wrappers:
  [code/spenso_structures.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/spenso_structures.py)
- examples and regression checks:
  [code/examples_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/examples_symbolica.py)

Supporting notes:

- chronological project history:
  [RESEARCH_LOG.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/RESEARCH_LOG.md)
- project goal:
  [PROJECT_GOAL.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/PROJECT_GOAL.md)
- roadmap:
  [ROADMAP.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/ROADMAP.md)

### Current position

The project is past the proof-of-concept stage for the symbolic core.

Reasonable summary of progress:

- scalar sector: working
- derivative sector: working
- fermion combinatorics: working at prototype level
- spinor/Lorentz tensor structures: started and already functional
- explicit gamma-current and current-current examples: working in the covered patterns
- gauge-field support: not yet implemented
- full model-definition interface: not yet implemented

### Main scientific/technical result so far

The main result so far is that Symbolica is already sufficient to support the
core combinatoric and symbolic-rewriting logic needed for Feynman-rule
derivation, while Spenso provides a natural way to represent the tensor and
index structures required for fermions and, later, gauge theories.

In other words, the basic architecture appears viable.

### Main limitations at the current stage

The prototype is not yet a complete FeynRules replacement.

Main missing parts:

- gauge representations and gauge-field interactions
- automatic generation of richer Lorentz/spinor chains from model terms
- broader multi-fermion tensor support beyond the currently exercised patterns
- centralized normalization and symmetry-factor conventions
- a clean model-definition API
- broader simplification for nontrivial gamma and gauge tensor structures
- export layer for final rule sets

### Next milestone

The next milestone is to make the framework gauge-ready.

That means:

1. stabilize `gamma`, `gamma5`, and related spinor-tensor abstractions
2. add gauge generators and structure constants in the same Spenso wrapper layer
3. implement abelian and non-abelian fermion-current examples
4. then move on to pure gauge-boson interaction structures

### Writing use

This document should stay short and stable.

It is meant to help with thesis writing by answering:

- what is the project trying to do?
- what already works?
- what is the current architecture?
- what remains to be done?

For dated progress details, use:

- [RESEARCH_LOG.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/RESEARCH_LOG.md)
