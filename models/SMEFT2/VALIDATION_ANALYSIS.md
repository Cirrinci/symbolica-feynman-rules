# Validation of the SMEFT2 Implementation Against FeynRules

This document records what the SMEFT2-vs-FeynRules comparison actually proves,
how it proves it, where its assumptions lie, and what remains outside its reach.
It is written to be usable directly as thesis material: every quantitative claim
below is reproducible from the commands listed in the last section.

---

## 1. Purpose and logical structure of the validation

FeynRules is taken as the mathematical oracle. The goal is not to show that
FeynPy and FeynRules produce *identical* expressions — they do not, and should
not be expected to, because the two frameworks differ in how they package
charge conjugation, expand covariant derivatives, and represent chirality.
The goal is to show that the two outputs are *equal as tensors*, i.e. that they
differ only by index relabelling, factor reordering, and applications of gauge
and Lorentz identities.

The validation therefore has three logically distinct layers, and it is
important not to conflate them:

1. **Model fidelity.** Does `SMEFT2.py` transcribe the operators of
   `SMEFT_Green_Bpreserving.fr`?
2. **Vertex equality.** Do the Feynman rules derived from the two Lagrangians
   agree as tensor expressions?
3. **Coverage.** For how much of the model does layer 2 actually run?

Layers 1 and 3 bound what layer 2 can mean. A perfect score in layer 2 over a
small subset of the model would prove very little.

---

## 2. Layer 1 — model fidelity

`SMEFT2.py` implements all 32 EFT Lagrangian blocks of
`reference/feynrules/SMEFT_Green_Bpreserving.fr`, and its `Ltot` is composed
from exactly the same 32 blocks in the same order. All 295 active Wilson
coefficients are declared and used. The two coefficients present in the
FeynRules file but absent from `SMEFT2.py` (`alphaEcdlq`, `alphaEcqeu`) are
commented out in FeynRules and appear in no FeynRules operator, so omitting them
is correct rather than a gap.

The SM core is taken from `UnbrokenSM_BFM.fr` minus the background-field ghost
and gauge-fixing sectors, which are deliberately excluded: the model is
formulated in the unbroken basis and the comparison is against the EFT-only
`Ltot`.

Structural deviations from the FeynRules source are systematic conventions
rather than per-operator adjustments:

| FeynRules | SMEFT2.py | Nature |
|---|---|---|
| `2 Ta[...]` | `weak_t = 2 * weak_gauge_generator` | named constant, 62 uses |
| `sigmamunu` | `sigma_term` / `sigma_matrix` | named helper, 38 uses |
| `Dual[FS]` | `_dual_fs` with `1/2 * epsilon` | named helper |
| `lag + HC[lag]` | explicit `.conj()` partner terms | expansion |
| `CC[...]` | `dirac_charge_conjugation(...)` | expansion |

No comparison-tuning comments, fudge factors, or per-operator sign corrections
appear in the model file. This matters for interpreting Section 4: because the
Lagrangian contains no hand-fitting, a residual disagreement between the two
sides is evidence of a *convention* difference rather than of a transcription
error.

---

## 3. Layer 2 — how vertex equality is established

### 3.1 Ingestion

The FeynRules export `Ltot_SMEFT_FeynRules.json` contains 184 vertices of arity
3 to 6, stored as Mathematica-syntax strings. These are parsed into Symbolica
expressions: `Index[Lorentz, Ext[1]]` becomes an index symbol, `IndexDelta`
becomes a metric or Kronecker delta according to index type, `TensDot[Ga[mu],
ProjM][...]` becomes a gamma chain, indexed Wilson parameters become
`alphaXXX(f1,f2)`, and field names are aligned with the FeynPy naming
(`LL`→`lL`, `LR`→`eR`, `QL`→`qL`, `UR`→`uR`, `DR`→`dR`).

The FeynPy side is generated directly by
`lagrangian.feynman_rule(*fields, simplify=True)`.

### 3.2 The canonical form

Equality is *not* decided by string or expression comparison. Both sides are
reduced to a canonical tensor-monomial map by
`src/symbolic/tensor_canonicalization.py`:

1. Live tensor heads are replaced by temporary heads carrying explicit symmetry
   metadata (`is_symmetric`, `is_antisymmetric`), so that the canonicaliser can
   exploit index symmetries.
2. `Expression.canonize_tensors` runs with contracted dummy indices grouped, so
   that dummy-index naming and placement are quotiented out.
3. Dummy names are standardised and commuting factors reordered into a canonical
   sequence.
4. The result is a map from canonical tensor monomial to *exact rational*
   coefficient.

Two rules are equal iff these maps are identical. Comparison is performed
separately for each Wilson-coefficient sector, i.e. after filtering both sides
to the terms carrying a given coefficient head, so that a cancellation between
different operators cannot mask an error in either.

On top of this sit a small number of narrow, physically justified rewriters
(`comparison/canonical.py`): the generator commutator
\([T^a, T^b] = i f^{abc} T^c\), the structure-constant Jacobi identity
\(f^{abe}f^{cde} - f^{ace}f^{bde} + f^{ade}f^{bce} = 0\), and SU(2)
pseudoreality relations. These are the "tensor identities" that allow a
non-identical but mathematically equal expression to be recognised as equal.

### 3.3 The comparison is genuine, not hardcoded

Three independent lines of evidence:

- **No operator is excused.** The set of coefficient heads exempted from
  comparison (`OMITTED_COEFFICIENT_HEADS`) is empty.
- **Mutation testing.** Perturbing the reference makes passing rows fail:
  doubling a Wilson coefficient turns `EXACT_MATCH` into a mismatch, and
  corrupting a gamma structure raises an error rather than passing.
- **Sensitivity suite on the shared core.** The same machinery under
  `models/SM` carries six deliberate-corruption tests (chirality flip, wrong
  CKM conjugation, reversed colour generator, removed imaginary unit, wrong
  derivative momentum, global sign flip), all of which are detected.

The declared non-direct cases are small and enumerated: 15 rows use the single
global evanescent charge-conjugation convention; 2 Weinberg rows use the fixed
antisymmetric `C` packaging; and 6 evanescent partner rows, covering 12
coefficient sectors, use pinned row-specific charge-conjugation packaging rules
(Section 4). Raw head-count deltas are reported separately as diagnostics:
every raw delta has a generated benign reason and none is unexplained in the
current artifact.

---

## 4. The evanescent charge-conjugation packaging convention

This is the single most important assumption in the fermionic sector and is
documented here in full.

### 4.1 The discrepancy

FeynRules writes the evanescent charge-conjugated four-fermion operators through
the `CC[...]` macro, for example

```
alphaEcll[f1,f2,f3,f4] CC[LLbar[sp1,ii,f1]].LL[sp1,jj,f2]
                       .LLbar[sp2,jj,f3].CC[LL[sp2,ii,f4]]
```

`ExpandIndices` resolves the conjugation, so the exported vertex carries **no
residual charge-conjugation matrix**: the spinor flow runs from external leg 1
to leg 4 and from leg 2 to leg 3 through ordinary spinor-metric and gamma
chains.

`SMEFT2.py` instead keeps the charge-conjugation matrices **explicit** and pairs
adjacent legs:

```
lLbar(sp1) * C(sp1,sp2) * lL(sp2) * lLbar(sp3) * C(sp3,sp4) * lL(sp4)
```

that is, flow (1,2) and (3,4) with two explicit `C` factors.

### 4.2 The transform

The two forms describe the same operator. Mapping the FeynPy packaging onto the
FeynRules packaging eliminates both `C` factors by re-pairing the four spinor
arms in **crossed** order, (arm 0, arm 3) and (arm 1, arm 2), and picks up a
single overall sign from the antisymmetry of the charge-conjugation matrix,
\(C^{T} = -C\), together with the anticommutation needed to reorder the fermion
fields into the crossed pairing.

Both the pairing and the sign are global constants, fixed once for every
`alphaEc` head:

```python
_EC_CC_CONVENTION_MODE = "crossed"
_EC_CC_CONVENTION_PHASE = -1
```

### 4.3 Why this is a derivation and not a fit

The choice is heavily overdetermined by the data. Of the four possible
(mode, phase) combinations, only one reproduces the reference, and it does so
uniformly across the sector:

| mode | phase | rows reproduced (of 21) |
|---|---|---|
| crossed | −1 | 15 (remaining 6 need the pinned rules of §4.4) |
| crossed | +1 | 0 |
| direct | −1 | 3 |
| direct | +1 | 2 |

The handful of passes in the failing rows are rows on which the transform does
not apply at all. A per-row fit would have 4²¹ degrees of freedom; what is used
here is a single global choice, and three of its four alternatives fail almost
everywhere. Disabling the transform entirely causes **all 21** four-fermion `Ec`
rows to fail, so it is load-bearing and must be stated explicitly rather than
left implicit.

### 4.4 Residual row-specific packaging

Six rows (12 coefficient sectors) require a further, row-specific
charge-conjugation partner transform on top of the global convention. These are
enumerated in an explicit table, `_EC_PARTNER_PACKAGING_RULES`, each entry
carrying a partner signature, a phase, a duplicate-leg symmetry flag, and a
provenance string naming the source operator. Nothing is searched at acceptance
time.

### 4.5 Reporting

Rows are graded so that the assumption is visible in the headline numbers:

| status | meaning |
|---|---|
| `EXACT_MATCH` | canonical-map equality with no packaging assumption |
| `MATCH_MODULO_EC_CC_CONVENTION` | equality only after the global §4.2 convention |
| `MATCH_MODULO_CC_PACKAGING` | equality only after a further pinned, row-specific transform |
| `UNRESOLVED_CC_PACKAGING` | no pinned rule known, or the pinned transform failed |

The resulting split over the 184 reference vertices is

**161 direct + 15 modulo the global convention + 8 modulo pinned packaging + 0 unresolved.**

The strict gate (`--check`) rejects anything other than the 161; the thesis gate
(`--check --allow-cc-packaging`) accepts the documented packaging rows and
rejects unresolved ones.

---

## 5. Layer 3 — coverage and its limits

### 5.1 What is covered

Of 303 parameters declared in the model, 295 appear in at least one validated
reference vertex.

### 5.2 What is not

| parameter | reason |
|---|---|
| `lam`, `muH`, `yd`, `yl`, `yu` | Standard Model parameters; correctly absent from an EFT-only `Ltot` |
| `alphaKB`, `alphaR2B`, `alphaOmuH2` | only ever appear in 2-point vertices |

The reference export covers arities 3 to 6 only. `alphaKB` and `alphaR2B`
multiply operators built from the abelian field strength, and `alphaOmuH2`
multiplies \(\mu^2|\Phi|^2\); none of the three generates a vertex with three or
more legs, so none is reachable by the present reference.

This is the *entire* coverage gap. Every other coefficient with a 2-point piece
— the gauge kinetic terms `alphaKG`, `alphaKW`, the Higgs kinetic term
`alphaKH`, and the fermion kinetic terms — also appears in validated ≥3-point
vertices, where gauge invariance ties the 2-point normalisation to the
higher-point ones. Three unvalidated coefficients out of 298 EFT coefficients
should be stated explicitly in the thesis, but they do not undermine the result.

### 5.3 A known blind spot

The SMEFT2-local parser discards `ProjM`/`ProjP` chirality projectors, on the
grounds that in the unbroken basis every fermion field (`lL`, `eR`, `qL`, `uR`,
`dR`) is already chiral and the projector is therefore redundant. This is
correct for this model, but the code does not verify that a projector agrees
with the chirality of the field it acts on, so a hypothetical chirality
disagreement between the two sides would pass unnoticed. The shared parser in
`src/feynrules/comparison.py` does translate projectors faithfully and is
demonstrably chirality-sensitive; migrating the SMEFT2 parser onto it would
close this gap.

---

## 6. Generality of the machinery

The comparison is not SMEFT2-specific infrastructure. The model-agnostic core
lives in `src/feynrules/comparison.py` and is already used by three models:

| model | result | integration |
|---|---|---|
| `models/SM` | 163/163 vertices | thin 205-line adapter over the generic sector comparators |
| `models/UnbrokenSM_BFM` | 67/67 vertices | model-local parser reusing the JSON loader and fermion reducers |
| `models/SMEFT2` | 184/184 vertices | model-local parser over the generic bosonic and canonical-map primitives |

Applying it to a further model requires four things: a parser adapter for that
model's export syntax, a field-name map, sector routing, and any
convention substitutions (CKM, diagonal Yukawa names, ghost momentum
conservation). Models with Wilson coefficients additionally need
coefficient-head filtering. The machinery is therefore general in substance but
not plug-and-play; the Standard Model is the reference integration and is by far
the smallest.

---

## 7. Summary of what the validation establishes

**Established.** The FeynPy SMEFT2 Feynman rules agree with FeynRules as tensor
expressions, sector by sector in the Wilson coefficients, for all 184 exported
vertices of arity 3 to 6, covering 295 of 303 model parameters. Agreement is
proved by exact rational equality of canonical tensor-monomial maps, is
sensitive to injected errors, and rests on no unenumerated exceptions.

**Assumed, and now documented.** One global charge-conjugation packaging
convention relating FeynRules' resolved `CC[...]` spinor flow to SMEFT2's
explicit `C` factors (§4), plus 12 pinned row-specific packaging rules. These
affect 23 of 184 rows and are reported under distinct statuses rather than being
folded into the direct-match count.

**Outside scope.** The 2-point sector, hence `alphaKB`, `alphaR2B` and
`alphaOmuH2`; and chirality consistency in the SMEFT2 parser (§5.3).

---

## 8. Reproduction

```bash
# Thesis acceptance gate (accepts documented packaging, rejects unresolved)
.venv/bin/python -m models.SMEFT2.comparison --check --allow-cc-packaging

# Strict gate: only rows with no packaging assumption at all
.venv/bin/python -m models.SMEFT2.comparison --check

# Regenerate the JSON/Markdown artifacts
.venv/bin/python -m models.SMEFT2.comparison

# Test suites
.venv/bin/python -m pytest models/SMEFT2/tests -q
.venv/bin/python -m pytest models/SM/tests models/UnbrokenSM_BFM/tests tests -q
```
