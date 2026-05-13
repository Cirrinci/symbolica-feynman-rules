# Thesis outline

### Absrtract


### Chapter 1: Introduction

- State why Feynman rules are needed in particle physics and why automation is useful
- Introduce FeynRules 
- End the chapter with list of thesis contributions

### Chapter 2:  Theory Background

- Essential background only: Lagrangian field theory, vertices from interaction terms, scalar/fermion/vector/ghost fields
- Index types: Lorentz, spinor, gauge, flavor; covariant derivatives; abelian and non-abelian gauge theories; fermionic non-commutativity
- Gauge fixing and ghost fields: why required
- SM
- SMEFT

### Chapter 3: Existing Tools: FeynRules

- FeynRules workflow and model-file structure

### Chapter 4: Design Goals and Scope

- Build a transparent symbolic pipeline from model declaration to Feynman rules

### Chapter 5: Architecture of the Python Framework
- full pipeline: user syntax -> lowered objects -> extracted vertices
- Cover `Model`, `Field`, `ConjugateField`, `IndexType`, `GaugeGroup`, `Representation`, `CovD`, `PartialD`, `Gamma`, `InteractionTerm`, and the analyzed/lowered source-term layer.

### Chapter 6: Index and Tensor Handling
- Explain how index kinds and labels are represented, when labels are explicit or inferred, and where ambiguity is rejected.
- Note that explicit labels are verbose but robust, while compact notation is readable but inference-sensitive.
- Require ambiguous input to fail clearly.

### Chapter 7: Gauge Interactions
- Describe the internal representation of gauge groups and matter representations, 
- how covariant derivatives are expanded.
- Cover abelian and non-abelian cases, scalar and fermion kinetic terms, matter-gauge vertices

### Chapter 8: Fermions and Non-Commuting Products

- Grassmann parity, Dirac conjugation, fermion ordering
- Bilinear recognition, gamma matrices, chiral projectors, four-fermion terms
- Difficulties of non-canonical ordering
- Fermions as main technical challenge (order & indices)

### Chapter 9: Gauge Kinetic Terms, Gauge Fixing and Ghosts

- Field-strength tensors, pure-gauge interactions
- Gauge-fixing terms, ghost declarations, ghost-gauge interactions
- Adoc metadata -> improved reporting, validation, debugging

### Chapter 10: Feynman-Rule Extraction

- Canonical interaction terms to vertices
- Vertex signatures, external-leg labels, momentum assignment
- Derivative-to-momentum conversion
- Fermionic signs, output simplification

### Chapter 11: Validation and Comparison

- Small analytic examples, then targeted FeynRules comparisons
- tests

### Chapter 12: Results

- Representative model inputs, lowered terms, extracted vertices
- intermediate structure
- Compact feature table: implemented, partially implemented, unsupported
- Validation summary: analytic and FeynRules checks

### Chapter 13: Discussion

- What worked well and remained difficult
- Technical lessons: fermion signs, sign conventions, index inference, gauge matching
- Inspectable intermediate representations as main strength

### Chapter 14: Conclusions and Future Work

- Restate goal, summarize framework, main validated examples, current limitations
- Future:....

## Supported scope to present

- scalar fields
- Dirac fermions
- vector gauge bosons
- ghost fields
- abelian and non-abelian gauge groups
- fundamental and adjoint matter representations
- symbolic parameters and couplings
- local interactions and covariant-derivative interactions
- gauge kinetic terms
- gauge-fixing terms
- ghost terms
- partial flavor/class expansion support

## What to keep out of scope

...
