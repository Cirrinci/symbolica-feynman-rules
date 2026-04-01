## Research Log

This file is a chronological summary of the work completed so far on the
Symbolica/Spenso FeynRules-style prototype.

### Current status snapshot

As of 2026-04-01:

- the active source tree is `src/`
- the main runnable validation script is `src/examples.py`
- the core engine is `src/model_symbolica.py`
- the current model layer is `src/model.py`
- reusable operator builders live in `src/operators.py`
- the minimal gauge compiler lives in `src/gauge_compiler.py`
- `src/spenso_gamma_checks.py` is runnable against the current source tree
- the long-term goal remains a Python analogue of FeynRules using Symbolica for symbolic rewriting and Spenso for tensor/index structures

### 2026-03-11: repository setup

What happened:

- initial repository was created
- project structure and version control baseline were added

What this achieved:

- established the working environment for the prototype

### 2026-03-14: thesis-workspace integration

What happened:

- the prototype was folded into the broader MSc thesis workspace

What this achieved:

- gave the project a stable home inside the actual thesis repository

### 2026-03-16 to 2026-03-17: scalar prototype

What happened:

- the first symbolic pipeline for scalar interactions was implemented
- early derivative handling and repository documentation were added

What this achieved:

- proved that Symbolica could support the basic contraction and rewriting logic
- established a first working reference point for later extensions

### 2026-03-19: fermion signs and Symbolica-first direction

What happened:

- fermionic sign handling and delta/index structure were introduced
- the implementation direction became more explicitly Symbolica-native

What this achieved:

- moved the project beyond a scalar-only proof of concept
- established the first serious path toward a mixed scalar/fermion engine

### 2026-03-23 to 2026-03-24: mixed scalar, fermion, and derivative support

What happened:

- scalar, fermion, and derivative cases were made to work together
- delta handling and Spenso integration were improved
- older prototype structure was cleaned up

What this achieved:

- reached the first genuinely usable symbolic core
- improved trust in the simplification layer
- reduced dependence on notebook-only logic

### 2026-03-24 to 2026-03-26: first tensor-structured fermion work

What happened:

- gamma-matrix and related spinor structures were introduced through Spenso-backed wrappers
- explicit open-spinor remapping inside coupling tensors was added for the covered fermion patterns
- current-current and gauge-ready fermion examples were added

What this achieved:

- shifted the project from plain fermion combinatorics toward actual tensor-structured fermion vertices
- demonstrated that open-index amputated output is viable in the current design

### 2026-03-31: repository refactor review

What happened:

- the active tree now consists of `src/model_symbolica.py`, `src/model.py`,
  `src/spenso_structures.py`, and `src/examples.py`
- the main example suite was confirmed runnable
- the documentation was updated to describe the live `src/` layout instead of the archived `code/` layout
- a code review identified the main structural gaps still worth addressing

What this achieved:

- clarified the actual architecture of the current repository
- separated working validation paths from stale scripts
- established a sharper set of next engineering tasks

### 2026-04-01: engine hardening and minimal gauge-model compiler

What happened:

- role handling in the engine/model boundary was hardened around typed field roles
- reusable operator builders were moved into `src/operators.py`
- `src/spenso_gamma_checks.py` was brought back onto the live source layout
- a minimal model-driven gauge compiler was added in `src/gauge_compiler.py`
- `src/examples.py` was extended to exercise compiled gauge interactions and print them clearly in both demo and validation paths

What this achieved:

- removed the main dependency on ad hoc operator assembly inside examples for the covered structures
- made the model layer capable of generating non-abelian fermion-gauge currents from gauge metadata
- made the model layer capable of generating abelian complex-scalar current/contact terms from field charge plus gauge-group declarations
- restored a second live validation path for gamma/tensor identities
- established the first genuinely model-driven gauge workflow in the repository

### 2026-04-01 (later): covariant-derivative compiler, fixed conventions, and non-abelian matter support

What happened:

- a working covariant-derivative compiler was added on top of the model/gauge layer
- the convention was fixed consistently as
  `D_mu = partial_mu + i g A_mu`
- the covered covariant cases were checked explicitly:
  - fermion QCD
  - fermion QED
  - scalar QED 3-point
  - scalar QED 4-point
  - scalar QCD 3-point
  - scalar QCD 4-point
  - mixed QCD+QED fermion coupling
- one covariant kinetic term now expands into the sum over all gauge groups acting on the field
- non-abelian complex-scalar current/contact compilation was added, including explicit representation labels and spectator-index identities
- the output/docs were clarified so the minimal structural compiler is separated from the physical covariant compiler

What this achieved:

- the project now has a working covariant-expansion layer, not only a minimal structural gauge compiler
- the representation/generic structure layer is working for the covered matter-sector cases
- the covariant expansion layer is working for the covered matter-sector cases
- basic validation on standard abelian and non-abelian matter interactions is working
- a major source of fake sign/confvention bugs was removed by separating:
  - the minimal compiler as a generic interaction-structure layer
  - the covariant compiler as the convention-fixed `D_mu` expansion layer

Current interpretation:

- representation/generic structure layer: working
- covariant expansion layer: working
- basic validation on standard interactions: working
- convention confusion: mostly resolved

Practical next steps:

1. freeze conventions in one place across code/docs/tests
   - Fourier convention
   - derivative-to-momentum rule
   - overall vertex `i`
   - covariant-derivative sign
2. turn the current checks into stronger regression tests so refactors cannot silently flip signs or factors
3. extend benchmarks and coverage toward the next harder sector
   - pure Yang-Mills 3-gluon and 4-gluon
   - ghosts if they become relevant to the chosen scope
   - more mixed-representation / multi-gauge-group cases
   - later chiral/projector structures if needed
4. keep improving output readability with clearer labels and compact interpretations
5. decide which compiler is the public physics-facing API; the covariant compiler is the strongest candidate

### Where we are in the overall progress

Best current summary:

- foundation phase: done
- scalar + derivative phase: done
- fermion combinatorics phase: done at prototype level
- first tensor-structured fermion/gauge-ready phase: working
- model-layer phase: working and usable
- minimal gauge-compiler phase: working
- covariant-derivative compiler phase: working for covered matter-sector cases
- gauge-complete phase: not implemented
- full FeynRules-style compilation layer: not implemented
- export/usability layer: not implemented

### Current assessment

The project now has a real core, not just experiments:

- one working contraction engine
- one usable model layer
- one reusable operator vocabulary
- one minimal gauge compiler
- one working covariant-derivative compiler for the covered matter-sector cases
- two runnable validation scripts

The main remaining risks are structural rather than conceptual:

- conventions now exist in code, but still need to be frozen/documented centrally
- gauge support is broader, but still not gauge-complete, especially in the pure-gauge sector
- the model/compiler/test boundary is still too concentrated in `src/examples.py`

### Immediate next milestone

The next milestone should be:

"Convention freeze and pure-gauge extension"

That means:

1. freeze and document the active conventions once across code/docs/tests
2. move the now-growing checks out of `src/examples.py` into a dedicated test layout
3. extend the compiler into the next harder sector, starting with pure non-abelian gauge self-interactions
4. decide whether the minimal compiler remains only an internal/helper layer while the covariant compiler becomes the main public layer
