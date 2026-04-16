# Curated review of the example vertices

This version is intentionally **not** a raw dump of the printed output.

For each example I list:
- the interaction term,
- a **worked summary** of the reported vertex,
- a short review:
  - **Correct**
  - **Correct, needs simplification**
  - **Correct rejection**

The long printed expressions were often overexpanded. Where useful, I rewrote them into a more canonical compact form while keeping the meaning of your reported result.

## Scalar sector

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| phi^4 | `lam4 * Phi * Phi * Phi * Phi` | `24 i lam4 (2π)^d Δ(q1+q2+q3+q4)` | Correct | Standard 4! combinatoric factor. |
| phi^2 chi^2 | `g * Phi * Phi * Chi * Chi` | `4 i g (2π)^d Δ(q1+q2+q3+q4)` | Correct | Correct mixed-scalar combinatorics. |
| complex scalar bilinear | `lamC * PhiC.bar * PhiC` | `i lamC (2π)^d Δ(q1+q2)` | Correct | Plain local bilinear. |
| derivative-contracted phi^4 | `gD2 * PartialD(Phi, mu) * PartialD(Phi, mu) * Phi * Phi` | `-4 i gD2 (2π)^d Δ(Σq) Σ_{i<j} p_i·p_j`  <br>(reported as the explicit sum over 6 momentum pairs) | Correct, needs simplification | Physics looks right; should be rewritten as a compact pairwise dot-product sum. |
| phi^6 | `lam6 * Phi * Phi * Phi * Phi * Phi * Phi` | `720 i lam6 (2π)^d Δ(q1+q2+q3+q4+q5+q6)` | Correct | Standard 6! combinatoric factor. |

## Fermion sector

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| Yukawa | `yF * Psi.bar * Psi * Phi` | `i yF (2π)^d δ_spin Δ(q1+q2+q3)` | Correct | Standard scalar Yukawa structure. |
| vector current | `gV * Psi.bar * Gamma(mu) * Psi * A` | `i gV (2π)^d γ^μ Δ(q1+q2+q3)` | Correct | Standard vector coupling. |
| scalar four-fermion operator | `-1/2*g_psi4 * Psi.bar * Psi * Psi.bar * Psi` | `i g_psi4 (2π)^d [ -δ12 δ34 + δ14 δ23 ] Δ(Σq)`  <br>(reported in unsimplified spinor-delta form) | Correct, needs simplification | Direct minus exchange structure is right. |

## Fermion + scalar derivative operators

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| psibar gamma^mu partial_mu psi phi chi | `yF * Psi.bar * Gamma(mu) * PartialD(Psi, mu) * Phi * Chi` | `yF (2π)^d γ^μ p_{psi,μ} Δ(Σq)` | Correct | Derivative acts on the fermion field; convention is internally consistent. |
| psibar psi (partial_mu phi)(partial^mu chi) | `yF * Psi.bar * Psi * PartialD(Phi, mu) * PartialD(Chi, mu)` | `-i yF (2π)^d δ_spin (p_φ·p_χ) Δ(Σq)` | Correct | Expected two-derivative scalar structure. |
| psibar psi (partial^2 phi) chi | `g1 * Psi.bar * Psi * PartialD(Phi, mu, mu) * Chi` | `-i g1 (2π)^d δ_spin p_φ^2 Δ(Σq)` | Correct | Correct higher-derivative local term. |

## Gauge interactions from declarative models

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| abelian fermion current from CovD | `i * PsiQED.bar * Gamma(mu) * CovD(PsiQED, mu)` | `-i eQED qPsi (2π)^d γ^μ Δ(Σq)` | Correct | Matches the standard `i ψ̄γ·D ψ` convention. |
| abelian scalar current from CovD | `CovD(PhiQED.bar, mu) * CovD(PhiQED, mu)` | `i eQED qPhi (p2 - p1)^μ (2π)^d Δ(Σq)`  <br>(reported as `-i e p1^μ + i e p2^μ`) | Correct | Standard scalar QED current. |
| abelian scalar contact from CovD | `CovD(PhiQED.bar, mu) * CovD(PhiQED, mu)` | `2 i eQED^2 qPhi^2 g^{μν} (2π)^d Δ(Σq)` | Correct | Correct two-photon contact from scalar kinetic term. |
| non-abelian fermion current from CovD | `i * q.bar * Gamma(mu) * CovD(q, mu)` | `-i gS γ^μ T^a (2π)^d Δ(Σq)` | Correct | Standard quark-gluon current. |

## Minimal gauge compiler

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| quark-gluon | `SU3C: q gauge current` | `-i gS γ^μ T^a (2π)^d Δ(Σq)` | Correct | Now convention-consistent with the CovD expansion. |
| fermion QED | `U1QED: PsiQED gauge current` | `-i eQED qPsi γ^μ (2π)^d Δ(Σq)` | Correct | Sign and overall `i` are now correct. |
| scalar QED current | `U1QED: scalar current (+) + scalar current (-)` | `i eQED qPhi (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Fixed compared with the earlier missing-`i` version. |
| scalar QED contact | `U1QED: scalar contact` | `2 i eQED^2 qPhi^2 g^{μν} (2π)^d Δ(Σq)` | Correct | Correct contact term. |
| scalar QCD current | `SU3C: scalar current (+) + SU3C: scalar current (-)` | `i gS T^a (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Correct non-abelian scalar current. |
| scalar QCD contact | `SU3C: scalar contact [slots 1,1]` | `i gS^2 g^{μν} (T^a T^b + T^b T^a) (2π)^d Δ(Σq)`  <br>(reported with explicit generator ordering) | Correct, needs simplification | Good structure; generator ordering is explicit but not canonicalized. |
| repeated-slot ambiguity | `ambiguous repeated ColorFund slots` | `rejected by compiler` | Correct rejection | This is the right behavior; ambiguity should not be silently resolved. |
| repeated-slot scalar QCD current | `SU3CBi: scalar current (+)_slot1 + scalar current (-)_slot1` | `i gS (spectator δ) T^a (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Active slot plus spectator identity is the expected result. |

## Covariant compiler: matter terms

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| qbar i gamma^mu D_mu q | `i * q.bar * Gamma(mu) * CovD(q, mu)` | `-i gS γ^μ T^a (2π)^d Δ(Σq)` | Correct | Matches the minimal compiler result. |
| PsiQEDbar i gamma^mu D_mu PsiQED | `i * PsiQED.bar * Gamma(mu) * CovD(PsiQED, mu)` | `-i eQED qPsi γ^μ (2π)^d Δ(Σq)` | Correct | Correct abelian fermion current. |
| one Dirac term over QCD+QED [gluon piece] | `i * PsiMix.bar * Gamma(mu) * CovD(PsiMix, mu)` | `-i gS γ^μ T^a (2π)^d Δ(Σq)` | Correct | Correct QCD component of the mixed covariant derivative. |
| one Dirac term over QCD+QED [photon piece] | `i * PsiMix.bar * Gamma(mu) * CovD(PsiMix, mu)` | `-i eQED qMix (spectator color δ) γ^μ (2π)^d Δ(Σq)` | Correct | Correct abelian piece with color spectator identity. |
| (D_mu phi)^dagger (D^mu phi) current | `CovD(PhiQED.bar, mu) * CovD(PhiQED, mu)` | `i eQED qPhi (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Correct scalar current. |
| (D_mu phi)^dagger (D^mu phi) contact | `CovD(PhiQED.bar, mu) * CovD(PhiQED, mu)` | `2 i eQED^2 qPhi^2 g^{μν} (2π)^d Δ(Σq)` | Correct | Correct scalar contact. |
| (D_mu PhiQCD)^dagger (D^mu PhiQCD) current | `CovD(PhiQCD.bar, mu) * CovD(PhiQCD, mu)` | `i gS T^a (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Correct non-abelian scalar current. |
| (D_mu PhiQCD)^dagger (D^mu PhiQCD) contact | `CovD(PhiQCD.bar, mu) * CovD(PhiQCD, mu)` | `i gS^2 g^{μν} (T^a T^b + T^b T^a) (2π)^d Δ(Σq)` | Correct, needs simplification | Same comment as the minimal compiler contact term. |
| one scalar term over QCD+QED [gluon current] | `CovD(PhiMix.bar, mu) * CovD(PhiMix, mu)` | `i gS T^a (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Correct QCD contribution. |
| one scalar term over QCD+QED [photon current] | `CovD(PhiMix.bar, mu) * CovD(PhiMix, mu)` | `i eQED qPhiMix (spectator color δ) (p2 - p1)^μ (2π)^d Δ(Σq)` | Correct | Correct abelian contribution. |
| one scalar term over QCD+QED [mixed contact] | `CovD(PhiMix.bar, mu) * CovD(PhiMix, mu)` | `2 i gS eQED qPhiMix g^{μν} T^a (2π)^d Δ(Σq)` | Correct | Correct mixed QCD×QED scalar contact. |
| (D_mu PhiBi)^dagger (D^mu PhiBi) [bislot, slot_policy='sum'] | `CovD(PhiBi.bar, mu) * CovD(PhiBi, mu)` | `sum over both color-fundamental slots of the scalar current` | Correct, needs simplification | Conceptually right; output is expanded over both active slots. |
| (D_mu PhiBi)^dagger (D^mu PhiBi) contact [bislot sum] | `CovD(PhiBi.bar, mu) * CovD(PhiBi, mu)` | `sum of all ordered slot-pair contact terms` | Correct, needs simplification | Structure looks right but is heavily expanded. |

## Covariant compiler: pure gauge/YM terms

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| -1/4 F_mu nu F^mu nu [abelian bilinear] | `-1/4 * FieldStrength(U1QED, mu, nu) * FieldStrength(U1QED, mu, nu)` | `i (g^{μν} p^2 - p^μ p^ν)`  <br>(reported in unsimplified momentum-expanded form) | Correct, needs simplification | Should be canonicalized to the standard transverse bilinear. |
| -1/4 G^a_mu nu G^{a mu nu} [bilinear] | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `i δ^{ab} (g^{μν} p^2 - p^μ p^ν)`  <br>(reported in unsimplified momentum-expanded form) | Correct, needs simplification | Same as above with adjoint color delta. |
| Yang-Mills 3-gauge vertex | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `gS f^{abc}[ g^{μν}(p1-p2)^ρ + g^{νρ}(p2-p3)^μ + g^{ρμ}(p3-p1)^ν ]`  <br>(reported as a 6-term expanded form) | Correct, needs simplification | Good YM structure, not yet in canonical compact form. |
| Yang-Mills 4-gauge vertex | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `standard quartic YM 3-channel structure`  <br>(reported as a highly expanded sum over metric/color structures) | Correct, needs simplification | Looks right, but should be reduced to the standard 3 independent channels. |

## Pure gauge

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| QED photon bilinear | `-1/4 * FieldStrength(U1QED, mu, nu) * FieldStrength(U1QED, mu, nu)` | `i (g^{μν} p^2 - p^μ p^ν)`  <br>(reported expanded) | Correct, needs simplification | Same physics as the covariant pure-gauge bilinear. |
| QCD gluon bilinear | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `i δ^{ab} (g^{μν} p^2 - p^μ p^ν)`  <br>(reported expanded) | Correct, needs simplification | Correct YM two-point structure. |
| QCD 3-gluon | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `standard YM 3-gluon vertex`  <br>(reported expanded) | Correct, needs simplification | Matches the expected cubic self-interaction. |
| QCD 4-gluon | `-1/4 * FieldStrength(SU3C, mu, nu) * FieldStrength(SU3C, mu, nu)` | `standard YM 4-gluon vertex`  <br>(reported expanded) | Correct, needs simplification | Overexpanded but physically consistent. |

## Ordinary gauge fixing and ghosts

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| -(1/2 xi) (partial.A)^2 [abelian] | `GaugeFixing(U1QED, xi=xiQED)` | `i (1/ξ) p^μ p^ν`  <br>(reported expanded) | Correct, needs simplification | Correct abelian gauge-fixing contribution. |
| -(1/2 xi) (partial.G)^2 [non-abelian] | `GaugeFixing(SU3C, xi=xiQCD)` | `i δ^{ab} (1/ξ) p^μ p^ν`  <br>(reported expanded) | Correct, needs simplification | Correct non-abelian gauge-fixing contribution. |
| ordinary photon bilinear | `-1/4 F^2 + GaugeFixing(U1QED, xi=xiQED)` | `i [ g^{μν} p^2 - (1 - 1/ξ) p^μ p^ν ]`  <br>(reported expanded) | Correct, needs simplification | Correct full inverse propagator structure. |
| Faddeev-Popov ghost bilinear | `GhostLagrangian(SU3C)` | `-i δ^{ab} p^2`  <br>(reported expanded with two derivative indices) | Correct, needs simplification | Should be reduced to the standard ghost kinetic form. |
| ghost-gluon interaction | `GhostLagrangian(SU3C)` | `-gS f^{abc} p^μ` | Correct | Momentum assignment is consistent with derivative on the antighost. |
| ordinary gluon bilinear | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `i δ^{ab} [ g^{μν} p^2 - (1 - 1/ξ) p^μ p^ν ]`  <br>(reported expanded) | Correct, needs simplification | Correct full gluon inverse propagator structure. |

## Ghost sector

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| ghost bilinear | `GhostLagrangian(SU3C)` | `-i δ^{ab} p^2`  <br>(reported expanded) | Correct, needs simplification | Same as above, just isolated. |
| ghost-gluon interaction | `GhostLagrangian(SU3C)` | `-gS f^{abc} p^μ` | Correct | Correct cubic FP ghost coupling. |

## Full gauge-fixed models

| Example | Interaction term | Vertex (curated from reported output) | Review | Note |
|---|---|---|---|---|
| full QCD: gluon bilinear | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `i δ^{ab} [ g^{μν} p^2 - (1 - 1/ξ) p^μ p^ν ]`  <br>(reported expanded) | Correct, needs simplification | Correct full QCD two-point structure. |
| full QCD: 3-gluon | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `standard YM 3-gluon vertex`  <br>(reported expanded) | Correct, needs simplification | Gauge fixing and ghosts do not alter the cubic YM self-coupling. |
| full QCD: 4-gluon | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `standard YM 4-gluon vertex`  <br>(reported expanded) | Correct, needs simplification | Same physical quartic YM interaction as in the pure-gauge sector. |
| full QCD: ghost bilinear | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `-i δ^{ab} p^2`  <br>(reported expanded) | Correct, needs simplification | Correct ghost kinetic term in the full model. |
| full QCD: ghost-gluon | `-1/4 G^2 + GaugeFixing(SU3C, xi=xiQCD) + GhostLagrangian(SU3C)` | `-gS f^{abc} p^μ` | Correct | Correct ghost-gauge coupling in the full model. |
| full QED: photon bilinear | `-1/4 F^2 + GaugeFixing(U1QED, xi=xiQED)` | `i [ g^{μν} p^2 - (1 - 1/ξ) p^μ p^ν ]`  <br>(reported expanded) | Correct, needs simplification | Correct full abelian inverse propagator. |

