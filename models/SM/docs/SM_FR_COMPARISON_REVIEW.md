# FeynPy reproduction of FeynRules `SM.fr`

## Executive conclusion

Within the tested scope, FeynPy reproduces the complete exported tree-level
interaction set of FeynRules `SM.fr`: all **163 flavor-expanded three- and
four-point vertices** have the same field content and an exactly zero symbolic
difference after documented convention transformations.

This is semantic equality, not raw text equality. The two systems use different
tensor syntax, chiral representations, dummy-index names and, for some ghost
vertices, derivative placements. These differences are reconciled by exact
symbolic transformations rather than numerical fitting.

The result does not establish complete model-file parity or make FeynPy a full
replacement for FeynRules. Parameter-card semantics, particle metadata, the
quadratic sector, restrictions, loop features and downstream export interfaces
remain outside the validated scope.

## Model declaration

The physics construction follows `SM.fr` closely.

| `SM.fr` concept | FeynPy implementation | Assessment |
|---|---|---|
| `U1Y`, `SU2L`, `SU3C` | `GaugeGroup` declarations with corresponding couplings, fields, charges, generators and structure constants | Faithful |
| Generation, weak, colour, Lorentz and spinor indices | Typed `IndexType` and representation objects | Faithful for the calculation |
| Gauge-basis fields | `B`, `Wi`, `Phi`, `LL`, `lR`, `QL`, `uR`, `dR` with the corresponding indices and hypercharges | Faithful |
| Physical fields | Gauge bosons, Higgs, Goldstones, ghosts and flavor classes with masses and quantum numbers | Faithful for rule generation |
| FeynRules `Definitions` | Simultaneous `FieldTransformation` rules for mixing, symmetry breaking, chirality, CKM rotation and ghosts | Structurally faithful |
| `LGauge`, `LFermions` | Field strengths and gauge-basis covariant kinetic terms | Faithful |
| `LHiggs`, `LYukawa` | Higgs sector and explicit Yukawa terms plus their Hermitian conjugates | Physically equivalent, not literal |
| Feynman-gauge `LGhost` | Non-abelian ghost terms and electroweak scalar–ghost interactions | Output-equivalent for tested interactions |

The principal declaration-level differences are:

- FeynPy uses `lam * vev^2` for the Higgs quadratic coefficient, while
  `SM.fr` introduces `muH^2` with `muH = sqrt(lam * vev^2)`.
- FeynPy writes Hermitian-conjugate Yukawa terms explicitly; `SM.fr` uses
  `yuk + HC[yuk]`.
- FeynPy keeps a symbolic general 3×3 CKM matrix, whereas the `SM.fr`
  parameter table assigns a Cabibbo-only numerical form. The analytical export
  retains symbolic CKM entries, so the vertex comparison remains valid.
- Diagonal Yukawa entries are treated as real, consistently with the
  mass-derived Yukawas in this `SM.fr`; general complex Yukawas are not tested.
- FeynPy rewrites `g1 = ee/cw` and `gw = ee/sw` in physical interactions but
  does not reproduce the full dependency chain from `aEWM1`, `Gf`, `aS`, `MZ`,
  Yukawa masses and `cabi`.
- PDG codes, widths, propagator styles, particle labels, Les Houches blocks,
  interaction orders and loop switches are not part of the packaged model.

FeynPy also provides optional general Rξ gauge fixing. This is an
extension rather than a literal part of `LSM` in the reference `SM.fr`.

## Symbolic comparison

The comparison is an exact symbolic equivalence check:

1. The JSON export is aligned with flavor-expanded FeynPy vertices by field
   multiset, while retaining the original FeynRules leg order for momenta and
   open indices.
2. FeynRules objects such as `ME`, `FV`, `Ga`, `T`, `f`, `IndexDelta`, `ProjM`
   and `ProjP` are parsed into the native FeynPy/Spenso tensor representation.
3. Both sides independently canonicalize dummy indices, color tensors,
   structure constants and spinor chains.
4. Matter rules are reduced to a common vector/axial basis and Yukawa rules to
   a common scalar/pseudoscalar basis. Chirality is preserved.
5. Ghost rules may use all-incoming momentum conservation to reconcile the
   differentiated ghost leg. Three scalar ghost rules also use
   `cw^2 + sw^2 = 1`.
6. A vertex matches only when the canonical symbolic difference is exactly
   zero. Global coverage separately checks for missing or extra nonzero
   signatures.

The universal momentum-conservation delta is omitted from the FeynPy side
because it is absent from the FeynRules export.

## Measured agreement

| Sector | FeynRules vertices | Exact matches | Extra FeynPy | Missing FeynPy |
|---|---:|---:|---:|---:|
| Pure gauge | 8 | 8 | 0 | 0 |
| Fermion–gauge matter | 51 | 51 | 0 | 0 |
| Yukawa/Goldstone | 42 | 42 | 0 | 0 |
| Higgs/Goldstone/gauge | 38 | 38 | 0 | 0 |
| Ghost | 24 | 24 | 0 | 0 |
| **Total** | **163** | **163** | **0** | **0** |

The amount of normalization differs by sector:

- **Gauge:** 7/8 rules match after tensor parsing; the four-gluon rule also
  needs structure-constant and dummy-index canonicalization.
- **Matter:** all 51 require the common spinor-current reduction because the
  two systems present chiral gamma chains differently.
- **Yukawa:** 18/42 match before bilinear reduction; all 42 match afterwards.
- **Higgs:** all 38 match after ordinary tensor parsing and canonicalization.
- **Ghost:** 8/24 match directly, 13 more after momentum conservation and the
  remaining 3 after applying the weak-angle identity.

FeynPy finds 171 flavor-expanded candidate signatures of arity three or four.
Eight candidates cancel to the exact zero rule after all contributing terms
are combined, leaving precisely the 163 nonzero FeynRules signatures. These are
cancellations, not additional physical vertices.

Provenance, global-coverage and mutation tests protect the result. Deliberate
changes to chirality, CKM conjugation, factors of the imaginary unit, color
ordering, ghost momentum and overall signs are all detected as mismatches. The
full repository test suite passes.

## Interpretation

 Equality requires tensor
translation, index canonicalization, chiral-basis reduction, momentum
conservation for derivative ghosts, real diagonal Yukawas and the weak-angle
identity for three ghost vertices. Each is an explicit algebraic identity or a
declared Standard Model assumption.

The strongest accurate statement is therefore:

> FeynPy and FeynRules are exactly equivalent for the exported nonzero
> flavor-expanded tree-level three- and four-point `SM.fr` interactions, after
> documented convention transformations.

## Remaining scope

The 163/163 result does not prove:

- equality of the complete gauge-basis or physical Lagrangian;
- equality of tadpoles, kinetic terms, mass matrices, gauge–Goldstone mixing,
  ghost masses, gauge-fixing terms or propagators;
- equality of numerical parameter values or the full parameter dependency
  graph;
- equality of widths, PDG information, particle names or export metadata;
- equality under `Massless.rst`, `DiagonalCKM.rst`, unitary gauge or arbitrary
  Rξ choices;
- loop/NLO behavior or UFO, FeynArts and CalcHEP compatibility;
- general FeynRules feature compatibility outside the structures exercised by
  this Standard Model calculation.

The tracked `sm_model_FeynRules.json` contains model metadata and a quadratic
Lagrangian string, but the stored two-point count is hard-coded to zero and is
not evidence of quadratic agreement.

## Recommended next step

Add an independent two-point comparison covering kinetic terms, masses,
gauge–Goldstone mixing, ghost masses and gauge fixing.

## Final assessment

FeynPy has reached complete symbolic agreement with the tested `SM.fr`
interaction sector. The defensible claim is precise tree-level vertex
equivalence within that scope, not complete FeynRules or model-file parity.
