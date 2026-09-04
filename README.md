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
```

This creates `.venv` and installs the engine, model packages, test dependencies
and notebook execution dependencies in editable mode. No manual `PYTHONPATH` configuration is
required. The equivalent manual installation is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,notebook]"
```

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
FeynRules-style `DC` and `FS`, `Gamma`, `PartialD`, `GaugeFixing` and
`GhostLagrangian`. The older descriptive names `CovD` and `FieldStrength`
remain available as compatibility aliases. FeynRules' `del` is spelled
`PartialD` because `del` is a Python keyword.

## Standard Model

The complete Standard Model application lives in [`models/SM`](models/SM/).
It contains the implementation, notebooks, tracked FeynRules reference data,
comparison adapter, tests, documentation and playground.

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

## SMEFT

The SMEFT Green-basis implementation lives in [`models/SMEFT`](models/SMEFT/).
Its default `Ltot` matches the FeynRules convention: EFT-only, with the local
SM-plus-EFT combination available separately as `Lfull`.

The accepted comparison against the bundled FeynRules EFT-only `Ltot` export is
**184/184 operator-content matches**. The exact-symbolic split is **161/184
direct matches**, **15/184 matches modulo the global `Ec` charge-conjugation
convention**, and **8/184 pinned charge-conjugation packaging matches**. There
are **0 unresolved** packaging rows and **0 exact symbolic unequal/error** rows.

Run the final SMEFT acceptance gate with:

```bash
.venv/bin/python -m models.SMEFT.comparison --check --allow-cc-packaging
```

The generated report and JSON artifacts are kept under
[`models/SMEFT`](models/SMEFT/) and
[`models/SMEFT/comparison/artifacts`](models/SMEFT/comparison/artifacts/).

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
notebooks/        focused API walkthroughs
docs/             maintained documentation, diagrams and reproducibility checklist
tests/            generic regression suite
```

The generic notebooks are:

- [`notebooks/getting_started.ipynb`](notebooks/getting_started.ipynb) — declare, validate, report and extract vertices
- [`notebooks/indices.ipynb`](notebooks/indices.ipynb) — `IndexType`, labels, custom index families
- [`notebooks/flavor.ipynb`](notebooks/flavor.ipynb) — flavor classes and `flavor_expand`
- [`notebooks/field_strengths.ipynb`](notebooks/field_strengths.ipynb) — `FS`, $F^3$, $F^4$
- [`notebooks/nested_derivatives.ipynb`](notebooks/nested_derivatives.ipynb) — nested `DC`, `PartialD`, `FS`
- [`notebooks/field_transformations.ipynb`](notebooks/field_transformations.ipynb) — EWSB, mixing, projectors
- [`notebooks/compiled_operators.ipynb`](notebooks/compiled_operators.ipynb) — operators, IBP, Symbolica export
- [`notebooks/gauge_and_brst.ipynb`](notebooks/gauge_and_brst.ipynb) — gauge variation and BRST

Model-specific code and evidence stay together under `models/<model>/`; generic
engine functionality stays under `src/`.

High-level Mermaid architecture diagrams are kept in
[`docs/diagrams/README.md`](docs/diagrams/README.md).

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

Run the generic notebook smoke check with:

```bash
mkdir -p /tmp/feynpy-notebooks
.venv/bin/python -m jupyter nbconvert --to notebook --execute --output-dir /tmp/feynpy-notebooks --ExecutePreprocessor.timeout=600 notebooks/*.ipynb
```

Run the final SMEFT acceptance gate with:

```bash
.venv/bin/python -m models.SMEFT.comparison --check --allow-cc-packaging
```

The expected terminal summary starts with:

```text
SMEFT comparison: 184/184 reference vertices match at operator-content level (176 literal-signature head matches + 8 CC-packaging head matches); exact symbolic split=direct 161/184, modulo global Ec CC convention 15/184, modulo pinned CC 8/184, unresolved CC 0/184
```

Run the Standard Model playground with:

```bash
.venv/bin/python models/SM/playground.py
```

The final validation and reproducibility checklist is
[`docs/notes/FINAL_REPRODUCIBILITY.md`](docs/notes/FINAL_REPRODUCIBILITY.md).
