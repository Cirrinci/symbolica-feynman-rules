# FeynPy

FeynPy is a Python framework for deriving tree-level Feynman rules from
declarative field-theory models. It uses
[Symbolica](https://symbolica.io/) for symbolic algebra and Spenso-backed tensor
objects for Lorentz, spinor, gauge and flavor structures.

The project follows the broad FeynRules workflow:

1. declare parameters, fields, indices and gauge groups;
2. write a Lagrangian using ordinary Python expressions;
3. compile covariant derivatives and field strengths;
4. optionally transform from a gauge basis to a physical basis;
5. request individual vertices or enumerate the interaction set.

## Installation

From the repository root:

```bash
bash setup_env.sh
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD"
```

This creates `.venv`, installs the dependencies in `requirements.txt`, and
makes the local engine and model packages importable in the active shell.

## Minimal example

```python
from feynpy import Field, Model, Parameter

phi = Field("phi", spin=0, self_conjugate=True)
g = Parameter("g")

model = Model(
    g * phi * phi * phi,
    fields=(phi,),
    parameters=(g,),
)

vertex = model.feynman_rule(phi, phi, phi)
print(vertex)
```

For gauge theories, the declaration language also provides `GaugeGroup`,
`CovD`, `FieldStrength`, `Gamma`, `PartialD`, `GaugeFixing` and
`GhostLagrangian`.

## Standard Model

The complete Standard Model application lives in [`models/SM`](models/SM/).
It contains the implementation, notebooks, tracked FeynRules reference data,
comparison adapter, tests, documentation and example.

```python
from models.SM import build_standard_model

sm = build_standard_model()
L = sm.lagrangian

wwa = L.feynman_rule(sm.fields.W.bar, sm.fields.W, sm.fields.A)
hww = L.feynman_rule(sm.fields.H, sm.fields.W.bar, sm.fields.W)
```

The validated result is **163/163 exact symbolic matches** against the exported
nonzero flavor-expanded tree-level three- and four-point vertices of FeynRules
`SM.fr` in Feynman gauge.

Model resources:

- [`SM_feynpy.ipynb`](models/SM/notebooks/SM_feynpy.ipynb) — readable
  `SM.fr`-style FeynPy implementation;
- [`SM_comparison.ipynb`](models/SM/notebooks/SM_comparison.ipynb) — complete
  executable comparison;
- [`SM_FR_COMPARISON_REVIEW.md`](models/SM/docs/SM_FR_COMPARISON_REVIEW.md) —
  precise scope, method, evidence and limitations.

## Supported capabilities

- scalar, fermion, gauge and ghost declarations;
- indexed parameters and flavor classes;
- abelian and non-abelian covariant derivatives;
- field-strength expansion through cubic and quartic gauge interactions;
- derivative interactions with explicit derivative-target bookkeeping;
- simultaneous field transformations with component restrictions,
  conjugation and CKM/flavor rotations;
- finite weak-index expansion;
- flavor-expanded and compact vertex extraction;
- Lorentz, spinor, color and structure-constant canonicalization;
- model validation and grouped vertex reporting;
- stripped or unstripped external wavefunctions and optional momentum delta.

Important conventions:

- derivatives map to `-i p_mu`;
- vertex extraction contributes the universal overall `+i`;
- matter uses `D_mu = partial_mu - i g A_mu`;
- pure gauge uses
  `F^a_mu_nu = partial_mu A^a_nu - partial_nu A^a_mu + g f^abc A^b_mu A^c_nu`;
- high-level `feynman_rule(...)` omits the universal momentum-conservation delta
  unless `include_delta=True` is requested.

## Repository layout

```text
src/feynpy/       reusable model and Feynman-rule engine
src/compiler/     gauge and covariant compilation
src/symbolic/     contraction, tensor and canonicalization machinery
src/lagrangian/   operator lowering and Symbolica export
src/feynrules/    generic FeynRules JSON parser and symbolic comparator
models/SM/        complete Standard Model vertical slice
notebooks/        generic API walkthroughs
examples/         generic runnable examples
tests/            generic regression suite
```

Model-specific code and evidence stay together under `models/<model>/`; generic
engine functionality stays under `src/`.

## Current scope

FeynPy is not yet a complete replacement for FeynRules. Current limitations
include general multi-fermion tensor structures, broader physics validation,
loop/NLO functionality, restriction files and downstream formats such as UFO,
FeynArts or CalcHEP.

For the Standard Model specifically, the 163/163 result validates the tested
interaction vertices, not complete parameter-card semantics, numerical model
metadata or an independent two-point-function comparison.

## Validation

Run the complete suite with:

```bash
.venv/bin/python -m pytest -q
```

Run the Standard Model example with:

```bash
.venv/bin/python models/SM/examples/example_standard_model.py
```

The chronological development record is
[`docs/notes/RESEARCH_LOG.md`](docs/notes/RESEARCH_LOG.md).
