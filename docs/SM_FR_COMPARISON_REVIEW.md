# Review of the FeynPy reproduction of FeynRules `SM.fr`

Date: 2026-06-29

## Executive conclusion

FeynPy reproduces the complete exported tree-level interaction-vertex set of
FeynRules `SM.fr` within the tested scope: all 163 flavor-expanded three- and
four-point vertices have the same field content and an exactly zero symbolic
difference after documented convention transformations.

This is a strong result, but the precise claim matters:

> The packaged FeynPy Standard Model reproduces the 163 tree-level interaction
> vertices exported from FeynRules `SM.fr` in Feynman gauge, modulo explicit
> representation, algebraic, and momentum-convention transformations.

It is not correct to say that the outputs differ only in names. They also use
different tensor syntax, chiral representations, dummy-index names, and—in the
ghost sector—different derivative placements. Three ghost rules additionally
require the Standard Model identity `cw^2 + sw^2 = 1`. These transformations
are valid and explicit, so the final agreement is semantic and exact, but it is
not raw expression or byte equality.

It is also not yet correct to say that FeynPy fully reproduces `SM.fr` as a
model file or that FeynPy is a complete FeynRules replacement. The current
comparison does not validate the full parameter card, numerical parameter
definitions, particle metadata, quadratic Lagrangian, propagators, widths,
interaction-order metadata, NLO switches, restrictions, or downstream export
interfaces.

## Reference and provenance

The local FeynRules source used for the full export is `SM.fr` version 1.4.7.
Its SHA-256 is:

```text
44690e769ecc4ed649033d2f9d58c5672203d8e820c56c90a378464204c99edc
```

This agrees with the hash recorded in `sm_model_FeynRules.json`. The full
export script loads `SM.fr`, evaluates

```mathematica
FeynmanRules[
  LSM,
  MinParticles -> 3,
  MaxParticles -> 4,
  FlavorExpand -> Generation
]
```

and stores each vertex's ordered fields, external-leg labels, and analytical
rule. The five sector files are an exact partition of the full export: their
keys, field orders, leg records, and rule strings agree with the 163-entry full
file.

The seven JSON oracle files are now tracked under
`tests/fixtures/feynrules/sm/`, so the comparison is reproducible from a clean
clone. Fixture provenance tests distinguish the recorded `SM.fr` source hash
above from the SHA-256 of the full vertex-export JSON,
`01dc1b98feb6a112b65e8c1cb42aa9e005fb6ae362767850d7b5888e8689913f`, and
verify that the five sector fixtures are an exact content partition of the
full export.

## How closely the model declaration follows `SM.fr`

The physics construction follows `SM.fr` closely.

| `SM.fr` concept | FeynPy implementation | Assessment |
|---|---|---|
| `U1Y`, `SU2L`, `SU3C` gauge groups | Three `GaugeGroup` declarations with the same gauge fields, couplings, charges, generators, and structure constants | Faithful |
| Generation, weak, colour, Lorentz and spinor indices | Typed `IndexType` and built-in representation objects | Faithful for the calculation |
| Unphysical fields `B`, `Wi`, `Phi`, `LL`, `lR`, `QL`, `uR`, `dR` | Gauge-basis `Field` declarations with the corresponding indices and hypercharges | Faithful |
| Physical `A`, `Z`, `W`, `H`, Goldstones, ghosts and fermion classes | Physical fields and flavor classes with conjugation, masses, charges, ghost/Goldstone links | Faithful for rule generation |
| FeynRules `Definitions` | Simultaneous `FieldTransformation` rules for electroweak mixing, the Higgs shift, chiral projectors, CKM rotation, and ghosts | Structurally faithful |
| `LGauge` | Gauge-field strengths squared | Faithful |
| `LFermions` | Gauge-basis covariant fermion kinetic terms | Faithful |
| `LHiggs` | Covariant Higgs kinetic term and symmetry-breaking potential | Physically equivalent, not literal |
| `LYukawa` | Gauge-basis Yukawa terms plus explicit Hermitian-conjugate terms | Physically equivalent, not literal |
| `LGhost` in Feynman gauge | Generic gauge ghost terms plus an explicit electroweak scalar-ghost construction | Output-equivalent for tested interactions |

The main declaration-level differences are:

- FeynPy writes the Higgs quadratic coefficient as `lam * vev^2`; `SM.fr`
  writes `muH^2` and defines `muH = sqrt(lam * vev^2)`.
- FeynPy writes the Hermitian-conjugate Yukawa terms explicitly; `SM.fr` uses
  `yuk + HC[yuk]`.
- The packaged FeynPy model keeps a general symbolic 3-by-3 CKM matrix. The
  `SM.fr` parameter table assigns the Cabibbo-only numerical form. The exported
  analytical FeynRules vertices retain symbolic CKM components, which is why
  the rule comparison can still match. Numerical parameter-point equality has
  not been tested.
- Diagonal Yukawa components are identified with real symbols during the
  comparison. This agrees with the real mass-derived Yukawas in this `SM.fr`,
  but it is an assumption rather than a test of general complex Yukawas.
- FeynPy substitutes `g1 = ee/cw` and `gw = ee/sw` in the compiled physical
  interactions. It does not reproduce the complete `SM.fr` input-parameter
  dependency chain from `aEWM1`, `Gf`, `aS`, `MZ`, Yukawa masses and `cabi`.
- Most FeynRules presentation and generator metadata—PDG codes, widths,
  propagator styles, particle labels, Les Houches blocks, interaction orders,
  loop switches and `FR$RmDblExt`—is not represented by the packaged model.
- FeynPy adds an optional general `R_xi` gauge-fixing construction. This is a
  useful extension, not a literal part of `LSM` in this `SM.fr`.

Therefore, the implementation is FeynRules-like in model-building structure,
but it is an independent Python implementation rather than a line-for-line or
feature-for-feature port.

## How the comparison works

The comparison is a staged symbolic equivalence proof.

1. FeynRules vertices are loaded from JSON. The sorted field multiset is used
   for signature alignment, while the original external-leg order is retained
   for momenta and open tensor indices.
2. The full reference set is classified into gauge, matter, Yukawa, Higgs and
   ghost sectors from the mapped fields' spin and kind.
3. FeynPy signatures are flavor-expanded. Barred-field spelling is aligned,
   for example `W.bar -> Wbar`, `GP.bar -> GPbar` and `e.bar -> ebar`.
4. The FeynPy rule is extracted in exactly the external-leg order stored by
   FeynRules. The universal momentum-conservation delta is omitted on the
   FeynPy side because it is not present in the FeynRules rule export.
5. Sector parsers convert FeynRules syntax into native FeynPy tensor objects:
   `ME` to the Lorentz metric, `FV` to momentum components, `f` to structure
   constants, `Ga` to gamma matrices, `T` to color generators,
   `IndexDelta` to representation metrics, and `ProjM`/`ProjP` to chiral
   projectors.
6. Both sides are canonicalized independently. This includes dummy-index
   normalization, products of structure constants, color tensors, and spinor
   chains.
7. Matter rules are reduced to a common vector/axial-current basis. Yukawa
   rules are reduced to a common scalar/pseudoscalar basis. These reductions
   preserve chirality information but remove the presentation difference
   between compact FeynRules projectors and expanded FeynPy gamma/gamma5
   chains.
8. Ghost rules may use all-incoming momentum conservation to reconcile which
   ghost leg carries the derivative. Scalar coefficients may be reduced modulo
   `cw^2 + sw^2 - 1`.
9. A row is a match only if the canonical symbolic difference is exactly zero.
   The report also checks for FeynRules-only and FeynPy-only nonzero
   signatures inside each comparison sector.

This is substantially stronger than the older text-normalization scripts. The
active comparison operates on symbolic tensor expressions and subtracts the
two rules; it does not declare equality because normalized display strings
look similar.

## Measured agreement

The current workspace produces the following result:

| Sector | FeynRules vertices | Exact matches | Extra nonzero FeynPy vertices | Missing FeynPy vertices |
|---|---:|---:|---:|---:|
| Pure gauge | 8 | 8 | 0 | 0 |
| Fermion-gauge matter | 51 | 51 | 0 | 0 |
| Yukawa/Goldstone | 42 | 42 | 0 | 0 |
| Higgs/Goldstone/gauge | 38 | 38 | 0 | 0 |
| Ghost | 24 | 24 | 0 | 0 |
| **Total** | **163** | **163** | **0** | **0** |

Additional instrumentation shows how much normalization is required:

- Gauge: 7 of 8 rules are already equal after FeynRules syntax is parsed into
  native tensors; the four-gluon rule needs structure-constant and contracted
  dummy-index canonicalization.
- Matter: 0 of 51 are equal before the common spinor-current reduction; all 51
  are equal afterwards. This is expected because FeynRules and FeynPy present
  chiral gamma chains differently.
- Yukawa: 18 of 42 are equal before the common bilinear reduction; all 42 are
  equal afterwards.
- Higgs: all 38 are directly equal after common tensor parsing and
  canonicalization. Although the comparator permits the weak-angle relation,
  none of these 38 currently needs it.
- Ghost: 8 of 24 are directly equal; 13 additional rules require momentum
  conservation and 3 additional scalar ghost rules require
  `cw^2 + sw^2 = 1`.

FeynPy initially reports 171 candidate flavor-expanded signatures of arity at
least three. Eight candidates sum to the exact zero rule after all contributing
Lagrangian terms are combined. Removing those zero candidates leaves exactly
the same 163 nonzero signatures as FeynRules. They are cancellation candidates,
not eight extra physical interactions.

The dedicated comparison, provenance, global-coverage, and sensitivity tests
pass. The entire repository test suite currently gives 436 passed and zero
failed. Antisymmetric tensor heads now receive a deterministic lexical slot
order after Symbolica tensor canonicalization, removing the former
order-dependent weak-`eps2` failure.

## Are the results identical except for names?

No. The strongest accurate statement is “exactly equivalent after explicit
convention transformations.” Besides field aliases, equivalence uses:

- FeynRules-to-Spenso tensor-head translation;
- external and dummy index relabeling;
- color and structure-constant canonicalization;
- compact-projector versus expanded gamma/gamma5 reduction;
- scalar/pseudoscalar and vector/axial basis conversion;
- real diagonal Yukawa identification;
- all-incoming momentum conservation for derivative ghost vertices;
- the weak-angle identity for three ghost vertices;
- omission of the common momentum-conservation delta distribution.

None of these is an arbitrary numerical fit. They are exact symbolic identities
or declared Standard Model assumptions. Nevertheless, they are materially more
than renaming.

## What the result does not prove

The 163/163 result does not currently prove:

- equality of the gauge-basis or physical Lagrangian as a complete expression;
- equality of one- and two-point functions, tadpoles, kinetic matrices, mass
  matrices, gauge-fixing terms or propagators;
- equality of numerical parameter values or the complete parameter dependency
  graph;
- equality of particle widths, PDG metadata, names or export metadata;
- equality under `Massless.rst`, `DiagonalCKM.rst`, unitary gauge or arbitrary
  `R_xi` choices;
- loop-level or NLO behavior;
- UFO, FeynArts, CalcHEP or other interface output compatibility;
- general FeynRules feature compatibility outside the Standard Model
  declaration and tensor structures exercised here.

The stored `sm_model_FeynRules.json` contains model metadata and a quadratic
Lagrangian string, but no tracked test reads it. Its
`feynman_rules_two_point_vertex_count` is hard-coded to zero by the export
script rather than obtained from a two-point Feynman-rule comparison.

## Reliability issues in the current comparison artifacts

### 1. Reference fixtures are tracked — resolved

All seven JSON oracle files now live in tracked test fixtures. Comparison tests
no longer read from the ignored `sandbox/` directory. Provenance regressions
pin both the `SM.fr` source hash and the full vertex-export hash.

### 2. The Standard Model notebooks are current and executable — resolved

The notebook roles are now explicit. `SM_feynpy.ipynb` is a literate,
declaration-by-declaration FeynPy implementation of `SM.fr`: it defines the
indices, parameters, fields, gauge groups, source Lagrangian, field
transformations and compilation pipeline without calling the packaged builder.
It prints all 163 nonzero FeynPy interaction rules followed by the 163 original
FeynRules export strings. `SM_comparison.ipynb` uses the authoritative packaged
`build_standard_model()` implementation and tracked fixtures for validation. It
reports the current 8/51/42/38/24 sector counts, the complete 163/163 result,
all eight exact-zero cancellation candidates, untouched and canonical
side-by-side outputs, and reusable inspection helpers. Every code cell in both
notebooks has freshly executed saved output and completes without errors.

### 3. Global signature coverage is tested — resolved

A regression now evaluates every flavor-expanded FeynPy candidate of arity
three or four without a sector field-name envelope. It asserts exactly eight
zero cancellation candidates and exact set equality between the remaining 163
nonzero FeynPy signatures and the full FeynRules export.

### 4. The equivalence layer has mutation tests — resolved

Six deliberate negative mutations now verify sensitivity to chirality, CKM
conjugation, factors of `I`, color-generator ordering, ghost momentum, and a
complete sign flip. Every mutation survives normalization and is reported as a
mismatch.

### 5. Weak-`eps2` canonicalization is deterministic — resolved

The former order dependence came from Symbolica using symbol interning order to
choose an equivalent orientation of antisymmetric external indices. FeynPy now
applies a deterministic lexical slot order after tensor canonicalization while
tracking the permutation sign. A regression creates the labels in reverse
order, and the full suite passes in one process.

## Recommended next steps

1. **Resolved:** the minimal FeynRules JSON oracle is tracked under
   `tests/fixtures/feynrules/sm/`, with source and export hashes asserted.
2. **Resolved:** a provenance regression verifies that the five sector files
   remain an exact content partition of the full 163-entry export.
3. **Resolved:** global nonzero-signature coverage evaluates every FeynPy
   candidate of arity 3–4 without reference-name filtering.
4. **Resolved:** `SM_feynpy.ipynb` is the executable `SM.fr`-style model
   implementation, while `SM_comparison.ipynb` uses the packaged model and
   tracked fixtures to report 8/51/42/38/24 and 163/163. Both are saved with
   fresh outputs.
5. Add a separate two-point comparison for kinetic terms, masses,
   gauge–Goldstone mixing, ghost masses and gauge fixing. Do not use the
   hard-coded zero count as evidence.
6. Compare model declarations and parameter definitions separately from
   vertices: gauge groups, fields, masses, `sw/cw/ee`, Yukawa tables, CKM value,
   interaction orders and relevant metadata.
7. **Resolved:** mutation-based negative tests cover chirality, CKM
   conjugation, factors of `I`, color ordering, ghost momenta, and sign flips.
8. **Resolved:** weak-`eps2` and other antisymmetric tensor heads now have a
   deterministic canonical orientation; the full suite is green.

## Final assessment

The Standard Model interaction physics is replicated very well. For the exact
FeynRules source and export configuration recorded above, FeynPy has achieved
complete symbolic agreement on the full nonzero three- and four-point
tree-level interaction set: 163/163.

The broader `SM.fr` model package is only partially replicated. The core
gauge-basis construction, symmetry breaking, flavor expansion, and physical
vertices are present; much of the model metadata, parameter-card semantics,
quadratic validation, restrictions, and interface ecosystem is not. The right
claim for a thesis is therefore a precise equivalence result for the supported
tree-level interaction scope, not full FeynRules equivalence.
