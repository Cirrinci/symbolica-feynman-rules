# Thesis outline

**Narrative arc:** motivation and context → design principles → pipeline implementation (declaration → lowering → vertices) → validation → synthesis.

Keep each implementation chapter anchored on one question: *what goes in, what comes out, and what can go wrong?*

---

### Front matter

- Abstract
- (Optional) notation and conventions table — index kinds, metric signature, fermion ordering convention

---

### Chapter 1 — Introduction

- Why Feynman rules matter; why automated, inspectable derivation is useful
- Brief role of FeynRules and of this Python/Symbolica approach (one paragraph each; details in Ch. 2)
- **Contributions** (numbered list)
- **Scope** (what is in / out of the thesis)
- Chapter guide

---

### Chapter 2 — Background and related work

*Theory and tools in one place; avoid repeating FeynRules later.*

**2.1 Field theory essentials** — Lagrangians, interaction terms, vertices; scalar / vector / Dirac / ghost fields; Lorentz, spinor, gauge, and flavor indices; abelian and non-abelian gauge structure; covariant derivatives; gauge fixing and ghosts (why they are needed).

**2.2 Reference models** — Standard Model as the main validation target; SMEFT only as motivation or outlook if not implemented.

**2.3 FeynRules workflow** — model files, conventions, outputs used for comparison.

---

### Chapter 3 — Requirements, design, and architecture

*Former Ch. 4 + 5.*

- Design goals: transparent symbolic pipeline, explicit intermediate representations, fail-fast index handling, separation of physics layers
- End-to-end pipeline diagram: declarative `Model` → analyzed source terms → lowered `InteractionTerm`s → compiler (gauge) → vertex engine → output
- Layer responsibilities: `model/`, `lagrangian/`, `compiler/`, `symbolic/` — what each may and may not know
- Core objects: `Field`, conjugation, `IndexType`, `GaugeGroup`, `Representation`, `CovD`, `PartialD`, `Gamma`, `InteractionTerm`, provenance metadata

---

### Chapter 4 — Model declaration and lowering

- Public declarative syntax and `Model` construction
- Source-term analysis: index binding, plain labels vs typed slots, ambiguity rejection
- Lowering to canonical interaction monomials (fields, `DerivativeAction`, couplings, bilinear metadata)
- Flavor / class expansion (supported subset)

---

### Chapter 5 — Gauge sector

*Merge former gauge-interaction and kinetic/fixing/ghost chapters.*

- Metadata-driven covariant expansion (abelian and non-abelian; scalar and fermion matter; repeated slots)
- Field strengths and pure-gauge interactions
- Gauge fixing and ghost Lagrangians; ghost–gauge vertices
- How provenance metadata supports validation and debugging

---

### Chapter 6 — Fermions, indices, and Feynman-rule extraction

*Fermion ordering and vertex engine belong in one chapter — they are one pipeline stage.*

- Grassmann parity, Dirac conjugation, ordered products, bilinear recognition
- Gamma matrices, projectors, four-fermion structures; ordering pitfalls
- Index/tensor infrastructure (representation, contraction, canonicalization) at the point of use
- Vertex signatures, external legs, momentum replacement, fermionic signs, simplification

---

### Chapter 7 — Validation and results

*Single evaluation chapter — no separate “results” chapter.*

- Strategy: analytic checks → targeted FeynRules parity → regression tests
- Representative examples: small models, unbroken SM (and any other flagship case)
- Show intermediate structures where they clarify correctness (lowered terms, sample vertices)
- **Feature table:** implemented / partial / not supported
- Summary of agreements and known presentation mismatches (e.g. canonicalization)

---

### Chapter 8 — Discussion

- What worked well; what remained difficult (fermion signs, conventions, index inference, gauge matching)
- Inspectable intermediates as the main methodological strength
- Limitations of scope and of comparison to FeynRules

---

### Chapter 9 — Conclusions and future work

- Restate goal and main outcomes
- Future directions (pick what you actually want to pursue):
  - electroweak symmetry breaking / BFM extensions
  - richer flavor expansion and SMEFT interfaces
  - operator calculus on lowered Lagrangians (`apply_operator`, Symbolica export) and IBP / total-derivative layer
  - tighter Symbolica ↔ `InteractionTerm` round-trip only with explicit ordering metadata

---

### Appendices (optional)

- A: Conventions and index glossary
- B: Selected vertex or model-file comparisons
- C: Short note on operator-action layer (if too detailed for the main text)

---

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
