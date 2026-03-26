## Research Log

This file is a chronological summary of the work completed so far on the
Symbolica/Spenso FeynRules-style prototype.

### Current status snapshot

As of 2026-03-25:

- current branch: `gamma_matrix-from-fermions`
- working tree: clean
- branch position relative to `main`: 12 commits ahead, 0 behind
- long-term goal: implement a Python analogue of FeynRules using Symbolica for
  symbolic rewriting and Spenso for tensor/index structures

### 2026-03-11: repository setup

What happened:

- initial repository was created
- project structure was added

What this achieved:

- established the base workspace and version control history

### 2026-03-14: repository restructuring

What happened:

- the repository was restructured to track the full MSc thesis project

What this achieved:

- created a more realistic home for the symbolic-physics prototype inside the
  broader thesis workspace

### 2026-03-16 to 2026-03-17: scalar prototype and repository cleanup

Relevant commits:

- `e9b2991` / `00136d3` scalar Feynman-rule prototype
- `9c4ddce` README for Symbolica-based prototype
- `caa6aa4` and `f997e64` documentation of working scalar/derivative vertices
- cleanup and visibility commits around `.gitignore`, tracked files, and merges

What happened:

- the first symbolic pipeline for scalar interactions was implemented
- README and prototype documentation were added
- repository hygiene improved through tracking cleanup and visibility changes

What this achieved:

- a first working proof of concept for scalar interaction vertices
- a documented baseline that could be extended rather than repeatedly rebuilt

### 2026-03-17: broader scalar-Lagrangian exploration

Relevant commit:

- `404c313` `working code for many scalar field lagrangian and derivatie`

What happened:

- the project explored broader scalar-field and derivative interaction cases

What this achieved:

- confidence that the basic combinatoric and derivative-handling approach could
  scale beyond the most trivial examples

### 2026-03-19: fermion signs and pure-Symbolica direction

Relevant commits:

- `650f778` add delta index structure and Grassmann signs
- `7e0b149` pure-Symbolica FeynRules implementation
- `7e42a1d` extend derivative examples and cleanup workflow

What happened:

- fermionic sign handling and delta-index structure were introduced
- a more explicitly Symbolica-native direction was developed in parallel
- derivative examples and workflow organization were improved

What this achieved:

- the project moved from a scalar-only proof of concept toward a realistic
  field-theory engine
- the core issue of fermionic permutation signs started being handled
- a cleaner Symbolica-first design path emerged

### 2026-03-23: first serious fermion-chain work

Relevant commits:

- `497f5b9` limited for fermions in symbolica
- `c267431` better fermion chain structure
- `19afbf7` notebook cleanup
- `4abecfe` simplified notebook

What happened:

- the symbolic engine was extended to cover fermion use cases
- fermion-chain structure was improved
- notebooks were simplified and refactored

What this achieved:

- the project stopped being “scalar plus ideas” and started becoming a genuine
  scalar+fermion prototype
- notebook experiments became easier to reuse and reason about

### 2026-03-24: scalar + fermion + derivative milestone

Relevant commits:

- `3a4a4b0` working for fermions and scalar
- `b35132a` scalar fermions and derivative mixed working
- `855672b` improvement in delta and spenso structure
- `f7ce776` better delta distinction
- `b29c7d7` clean up old version
- `bb52702` clean and move

What happened:

- scalar, fermion, and mixed derivative cases were made to work together
- delta-handling logic was improved so symbolic deltas were not over-simplified
- Spenso integration improved
- older prototype structure was cleaned and reorganized

What this achieved:

- the project reached an important milestone:
  scalar interactions, fermionic statistics, and derivative momentum factors now
  coexist in one working pipeline
- the simplification layer became more trustworthy
- the codebase moved from notebook-heavy exploration toward a reusable module

### 2026-03-24: first Spenso gamma-structure branch

Relevant commits:

- `a1e9203` first spenso and gamma matrices structure
- `502d1eb` clean

What happened:

- the branch `gamma_matrix-from-fermions` introduced the first explicit step
  toward Spenso-backed gamma-matrix structures
- the branch was cleaned afterward

What this achieved:

- the project crossed from “fermion support” to “tensor-structured fermion
  support”
- this is the natural bridge toward gauge fields and proper Lorentz/spinor
  tensor algebra

### 2026-03-25: current implementation state

What is now working in the codebase:

- scalar polynomial interactions
- multi-species scalar interactions
- derivative interactions with permutation-aware momentum assignment
- fermion permutation signs
- stripped and unstripped external fermion wavefunction factors
- spinor-delta output through Spenso bispinor metrics
- vector-current structures
- axial-current structures with `gamma` and `gamma5`
- first matrix-backed checks using the Spenso HEP tensor library

Concrete repository state:

- main symbolic engine:
  [code/model_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/model_symbolica.py)
- Spenso tensor wrappers:
  [code/spenso_structures.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/spenso_structures.py)
- runnable examples and tests:
  [code/examples_symbolica.py](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/examples_symbolica.py)
- project objective:
  [PROJECT_GOAL.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/PROJECT_GOAL.md)
- forward plan:
  [ROADMAP.md](/Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/ROADMAP.md)

### Where we are in the overall progress

Best current description:

- foundation phase: done
- scalar + derivative phase: done
- fermion-sign and mixed-interaction phase: done
- first spinor/Lorentz tensor phase: started and already working for gamma and gamma5
- gauge-field phase: not implemented yet
- full FeynRules-style model-definition layer: not implemented yet
- export/usability layer: not implemented yet

### Progress assessment

If the final target is:

"A Python FeynRules-like system that defines models and derives vertices with
Symbolica + Spenso,"

then the project appears to be roughly at this stage:

- around 55-65% through the symbolic core
- around 20-30% through the full end-user system

Reason:

- the hard symbolic backbone for scalar, derivative, and fermion combinatorics
  is already in place
- gamma-structure support has begun in the right way
- the biggest remaining blocks are:
  - gauge representations and gauge-boson interactions
  - a clean model-definition API
  - systematic tensor simplification for richer Lorentz/spinor structures

### Immediate next milestone

The next milestone should be:

"Gauge-ready tensor infrastructure"

That means:

1. stabilize gamma/gamma5/tensor-current abstractions
2. expose gauge generators `T^a_{ij}` and structure constants `f^{abc}` in the
   same wrapper layer
3. add abelian and non-abelian fermion-current examples
4. only after that, add full gauge-boson self-interaction support

### Short summary in one sentence

The project has successfully moved from a scalar Symbolica prototype to a
working scalar+fermion+derivative engine with initial Spenso-backed gamma-matrix
support, and the next major frontier is gauge-field structure plus a proper
model-definition layer.
