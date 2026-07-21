# SMEFT Appendix D Audit

## Scope

This audit checked the implementation in `models/SMEFT` against Appendix D of
arXiv:2112.10787. The review covered:

* the local SMEFT model files, including uncommitted work;
* the supporting engine code that fixes conventions and lowering behavior;
* the operator inventory, helper functions, and representative Feynman rules.

The operator-by-operator register requested in the audit brief is in
`operator_audit.tsv`.

## Actual correctness issues found

Two genuine implementation mistakes were found and fixed during the audit.

### 1. `R''_{HD}` missed terms

`models/SMEFT/tensors.py` differentiated ordinary fields and partial
derivatives, but not declarative covariant-derivative factors. As a result, the
outer `∂_μ` in `R''_{HD}` did not act on `DC(H, μ)` or `DC(H.bar, μ)`, so
`_rpphd` in `operators_bosonic.py` produced only 2 terms instead of 4.

The fix was to extend `_differentiate_factor()` so `partial()` also handles:

* `CovariantDerivativeFactor -> DifferentiatedOperatorFactor`
* legacy `DifferentiatedCovariantFactor -> DifferentiatedOperatorFactor`
  with one more ordinary derivative

Regression test: `tests/test_bosonic.py::test_rpphd_outer_derivative_hits_covariant_derivative`.

### 2. Non-self-conjugate operators were missing `+ h.c.`

Several Appendix D operators are explicitly non-Hermitian. The builders already
constructed the bare operator `C O`, but the compiled Lagrangian omitted the
Hermitian conjugate. This affected:

* Table 2 dipoles, Yukawa-like operators, `OHud`, and redundant `psi^2HD2`;
* Table 3 LR scalar/tensor four-fermion operators;
* Table 4 dual dipoles and evanescent `psi^2HD2`;
* the non-self-conjugate Tables 5-6 LR/LR evanescent structures.

The fix was implemented in `registry.py` and `smeft.py` by compiling
operator-by-operator and adding `compiled + compiled^dagger` for the known
non-Hermitian set.

Regression tests:

* `tests/test_two_fermion.py::test_non_hermitian_two_fermion_operators_include_hc`
* `tests/test_two_fermion_redundant.py::test_non_hermitian_redundant_psi2hd2_include_hc`
* `tests/test_four_fermion.py::test_non_hermitian_four_fermion_include_hc`
* `tests/test_evanescent.py::test_non_hermitian_evanescent_include_hc`

## Helper audit

### `Poly`

`Poly` is a small sum-of-monomials layer over declarative factors. It is needed
because Appendix D operators are naturally written as sums whose factors must
remain in a fixed left-to-right order so fermion pairing and ordered gamma
chains survive lowering.

Verdict:

* mathematically appropriate for antisymmetrized Higgs currents, chiral
  projectors, and explicit Leibniz expansions;
* relies on private `feynpy.declared` internals (`_DeclaredMonomial`,
  `_coerce_decl_factor`), so it is correct but structurally fragile;
* does not by itself violate compiler assumptions, but it does bypass a cleaner
  public sum/product API that the engine does not currently expose.

### Explicit covariant derivatives

The doublet and fermion derivative helpers in `tensors.py` implement Eq. (D.3)
with the expected `D = ∂ - i g A` sign, and they correctly reverse the gauge
action for conjugated fields. The sign pattern agrees with the core gauge
compiler in `src/compiler/gauge.py`.

Main risk area:

* manual Leibniz expansions are maintenance-heavy;
* before the `R''_{HD}` fix, ordinary derivatives acting on covariant
  derivatives were incompletely represented.

After the fix, no additional missing-term pattern was found.

### Explicit field strengths and duals

The field-strength helpers represent `G^A_{μν}`, `W^I_{μν}`, `B_{μν}` and their
duals exactly in the Appendix D convention. Duals use
`Xtilde_{μν} = 1/2 ε_{μναβ} X^{αβ}` and compiled vertices carry the expected
Levi-Civita tensor.

Verdict: no sign or normalization discrepancy found.

### Sigma tensors and gamma-chain helpers

The implementation uses the ordered sigma and gamma-chain structures required
for the evanescent basis. The sigma convention matches the internal symbolic
checks in `src/symbolic/spenso_gamma_checks.py`. Ordered triple-gamma chains are
preserved, which is the essential requirement for Tables 4-9.

Verdict: consistent with Appendix D. No unwanted four-dimensional reduction was
seen.

### Higgs left-right derivative and `Htilde`

The Higgs singlet and triplet currents implement
`H^dag i<->D_mu H` and `H^dag i<->D^I_mu H` by explicit antisymmetrization. The
conjugate doublet `Htilde_r = eps_{rs} H*_s` follows the paper convention.

Verdict: equivalent to the paper expressions.

### Charge-conjugation helpers

The charge-conjugation helpers in `operators_evanescent.py` build the declared
`psi^T C Gamma psi` and `psibar Gamma C psibar^T` chains faithfully. The problem
is not the declared structure; it is the lowering step afterwards.

Verdict: the helper layer is correct, but the engine cannot compile the result.

## Gauge conventions

The implementation is consistent with Appendix D on the points that matter for
equivalence:

* covariant derivative sign: `D = ∂ - i g A`;
* `SU(2)` generators: Pauli matrices over 2;
* `SU(3)` generators: `lambda^A / 2`;
* structure constants and weak epsilon tensors: standard conventions;
* hypercharges: `Y_q = 1/6`, `Y_l = -1/2`, `Y_u = 2/3`, `Y_d = -1/3`,
  `Y_e = -1`, `Y_H = 1/2`;
* `Htilde` and dual field-strength normalization: matching Appendix D.

No convention mismatch was found that would require a compensating redefinition.

## Representative operator checks

Representative difficult classes were checked directly against the paper form:

* `O3G` and `OHWB`: direct agreement of index structure and coefficients.
* `R2G`: explicit `D^ν G^A_{μν}` expansion matches the intended covariant
  derivative of a field strength, including non-abelian self-interaction terms.
* `RDH`: iterated covariant derivatives on the Higgs are represented explicitly
  and compile without engine changes.
* `OHq3`: the weak-triplet current and Higgs-triplet current use matching
  `sigma^I` insertions.
* `OuG`: the dipole tensor structure is correct and now includes the required
  Hermitian conjugate.
* `RqD`: the explicit `i/2 psi-bar {D^2, /D} psi` structure is represented by
  ordered sums/products rather than hidden simplification.
* `RuHD1-4` and analogous operators: scalar/tensor `psi^2HD2` structures match
  the paper class and now include `+ h.c.`.
* `Oquqd8` and `Olequ3`: colour/octet and tensor structures are correct; both
  are non-self-conjugate and required the `+ h.c.` fix.
* `E3ll`: ordered triple-gamma chains survive compilation.
* `EuB` and `EpBq`: dual field strengths and ordered `sigma`/derivative
  placements survive compilation.

Confidence is high for these classes after the fixes above. `R''_{HD}` is now
only slightly lower confidence (`medium-high`) because it required a helper
correction during the audit.

## Evanescent operators

For Tables 4-7, the implementation respects the key Appendix D requirements:

* gamma ordering is explicit and preserved;
* projector placement follows the field chirality rather than being simplified
  away;
* Lorentz, colour, weak-isospin, and epsilon contractions are encoded
  explicitly;
* no four-dimensional identities are used to collapse the evanescent sector.

The Tables 8-9 charge-conjugation sector is different:

* declared structures are built correctly;
* compilation fails because the lowering code only supports closed Dirac chains
  with one conjugated and one unconjugated endpoint.

This is a real engine limitation, not a bookkeeping omission.

## Wilson coefficients

The Wilson-coefficient arities are correct across the implemented basis. The
main limitation is that Hermiticity and permutation symmetries are stored mostly
as metadata in `wilson.py`, not enforced by the parameter objects themselves.

Assessment:

* mathematically sufficient if the user assigns coefficients consistently;
* insufficient if one wants the model layer itself to enforce symmetry-related
  coefficient identities;
* this does not make the generated Feynman rules wrong for a fixed coefficient
  choice, but it leaves room for inconsistent external assignments.

## Independent validation status

The requested FeynRules cross-check could not be completed in this environment.

Reason:

* no `wolframscript` or FeynRules installation is available locally;
* no SMEFT-specific FeynRules export artifact is present in the repository for
  offline comparison.

So the independent-validation step remains open.

## Completeness re-evaluation

The statement

> "The complete dimension-six SMEFT Green basis has been implemented."

is false as written.

Counts from Appendix D:

* Tables 1-7: 231 operators
* Table 8: 18 operators
* Table 9: 19 operators
* total Appendix D basis: 268 operators

Current code status:

* 231 operators from Tables 1-7 are implemented and compile;
* 8 representative Table 8-9 operators are registered as declared-only blocked
  entries;
* the remaining 29 Table 8-9 operators are not yet registered.

The checklist has been rewritten accordingly.

## Final report

### A. Confirmed correct

* The implemented Tables 1-7 basis is now consistent with Appendix D at the
  operator-structure level.
* Gauge conventions, dual-field normalization, `Htilde`, and sigma/gamma-chain
  conventions are consistent with the paper.
* The evanescent non-charge-conjugation sector preserves ordered gamma chains as
  required.

### B. Possible mistakes

* The manual derivative expansions remain the highest maintenance-risk area and
  deserve re-checking if those helpers change again.
* Wilson-coefficient symmetry constraints are not enforced, only documented.
* `R''_{HD}` should be kept under explicit regression coverage because it was the
  one helper-induced omission found in the bosonic redundant sector.

### C. Confirmed limitations

* Charge-conjugation bilinears of Tables 8-9 are blocked by the engine's local
  fermion-flow lowering.
* Independent FeynRules validation is not possible in the current environment
  without a Wolfram/FeynRules setup or a saved SMEFT reference export.

### D. Recommended improvements

* Highest priority: add engine-level support for charge-conjugation fermion
  flow, then register the remaining 29 Tables 8-9 operators.
* High priority: replace private-internal `Poly` and Hermitian-conjugation
  hooks with public engine APIs.
* Medium priority: enforce Wilson-coefficient symmetry metadata at the parameter
  layer.
* Medium priority: add a stored SMEFT FeynRules comparison fixture so Appendix D
  checks are reproducible without Mathematica.
