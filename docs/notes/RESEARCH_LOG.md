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

### Where we are in the overall progress

Best current summary:

- foundation phase: done
- scalar + derivative phase: done
- fermion combinatorics phase: done at prototype level
- first tensor-structured fermion/gauge-ready phase: working
- model-layer phase: working and usable
- minimal gauge-compiler phase: working
- gauge-complete phase: not implemented
- full FeynRules-style compilation layer: not implemented
- export/usability layer: not implemented

### Current assessment

The project now has a real core, not just experiments:

- one working contraction engine
- one usable model layer
- one reusable operator vocabulary
- one minimal gauge compiler
- two runnable validation scripts

The main remaining risks are structural rather than conceptual:

- there is still no real covariant-derivative compiler
- gauge support is broader, but still not gauge-complete
- the model/compiler/test boundary is still too concentrated in `src/examples.py`

### Immediate next milestone

The next milestone should be:

"Covariant-derivative compilation"

That means:

1. define model/compiler builders for `D_mu psi` and `D_mu phi`
2. expand `|D_mu phi|^2` into scalar-current and contact interactions automatically
3. expand `psibar i gamma^mu D_mu psi` into fermion-gauge current interactions automatically
4. fix and document gauge-normalization conventions before widening gauge coverage
5. move the now-growing checks out of `src/examples.py` into a dedicated test layout
