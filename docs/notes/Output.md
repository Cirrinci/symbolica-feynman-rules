# ================================================================================
Demo: all

Note:
minimal compiler shows generic interaction structure; covariant compiler shows the convention-fixed `D_mu` expansion.
Pure-gauge covariant blocks also show a compact override, because the raw compiled vertex keeps the derivative-index metrics explicit.

# === scalar: phi^4 ===

lam4 * phi^4
alphas = [phi0, phi0, phi0, phi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
24𝑖*lam4

# === scalar: phi^6 ===

lam6 * phi^6
alphas = [phi0, phi0, phi0, phi0, phi0, phi0]
betas  = [b1, b2, b3, b4, b5, b6]
ps     = [p1, p2, p3, p4, p5, p6]

Vertex:
720𝑖*lam6

# === scalar: phi^2 chi^2 ===

g * phi^2 * chi^2
alphas = [phi0, phi0, chi0, chi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
4𝑖*g

# === scalar: complex scalar bilinear ===

lamC * phi^dagger * phi
alphas = [phiCdag0, phiC0]
betas  = [b1, b2]
ps     = [p1, p2]

Vertex:
1𝑖*lamC

# === scalar: derivative (mu,nu) * phi^4 ===

gD * (d_mu phi)(d_nu phi) phi phi
alphas = [phi0, phi0, phi0, phi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Compact override:
-1𝑖*gD*(2*pcomp(p1,mu)*pcomp(p2,nu)+2*pcomp(p1,mu)*pcomp(p3,nu)+2*pcomp(p1,mu)*pcomp(p4,nu)+2*pcomp(p1,nu)*pcomp(p2,mu)+2*pcomp(p1,nu)*pcomp(p3,mu)+2*pcomp(p1,nu)*pcomp(p4,mu)+2*pcomp(p2,mu)*pcomp(p3,nu)+2*pcomp(p2,mu)*pcomp(p4,nu)+2*pcomp(p2,nu)*pcomp(p3,mu)+2*pcomp(p2,nu)*pcomp(p4,mu)+2*pcomp(p3,mu)*pcomp(p4,nu)+2*pcomp(p3,nu)*pcomp(p4,mu))
Sum notation:
(2)! * Σ_{a, b distinct} p_{a,mu} p_{b,nu}

# === scalar: derivative (mu,mu) * phi^4 ===

gD2 * (d_mu phi)(d_mu phi) phi phi
alphas = [phi0, phi0, phi0, phi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Compact override:
-1𝑖*gD2*(4*pcomp(p1,mu)*pcomp(p2,mu)+4*pcomp(p1,mu)*pcomp(p3,mu)+4*pcomp(p1,mu)*pcomp(p4,mu)+4*pcomp(p2,mu)*pcomp(p3,mu)+4*pcomp(p2,mu)*pcomp(p4,mu)+4*pcomp(p3,mu)*pcomp(p4,mu))
Sum notation:
(2)! * Σ_{a, b distinct} p_{a,mu} p_{b,mu}

# === scalar: multi-species phi_i^2 phi_j^2 phi_k^2 ===

gijk(i,j,k) * phi_i^2 phi_j^2 phi_k^2
alphas = [i, i, j, j, k, k]
betas  = [b1, b2, b3, b4, b5, b6]
ps     = [p1, p2, p3, p4, p5, p6]

Vertex:
8𝑖*gijk(i,j,k)

# === fermion: Yukawa [amputated] ===

yF * psibar * psi * phi
alphas = [psibar0, psi0, phi0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
1𝑖*yF*g(bis(4, i1),bis(4, i2))

# === fermion: Yukawa [matrix element] ===

yF * psibar * psi * phi  [matrix element]
alphas = [psibar0, psi0, phi0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
1𝑖*yF*U(phi0,p3)*UF(psi0,p2,s2,alpha_s)*UbarF(psibar0,p1,s1,alpha_s)

# === fermion: vector current ===

gV * psibar gamma^mu psi A_mu
alphas = [psibar0, psi0, A0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
1𝑖*gV*gamma(bis(4, i1),bis(4, i2),mink(4, mu))

# === fermion: axial current ===

gV * psibar gamma^mu gamma5 psi A_mu
alphas = [psibar0, psi0, A0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
1𝑖*gV*gamma(bis(4, i1),bis(4, alpha_s),mink(4, mu))*gamma5(bis(4, alpha_s),bis(4, i2))

# === fermion: underspecified product diagnostic ===

g4F * psi * psibar * psi * psibar  [no spinor contractions]
rejected: multi-fermion operators need explicit spinor contractions

# === fermion: -(g/2)(psibar psi)^2 [amputated] ===

- (g/2)(psibar psi)^2 [amputated]
alphas = [psibar0, psi0, psibar0, psi0]
betas = [b1, b2, b3, b4]
ps = [p1, p2, p3, p4]

Vertex:
1𝑖*(-g_psi4*g(bis(4, i1),bis(4, i2))*g(bis(4, i3),bis(4, i4))+g_psi4*g(bis(4, i1),bis(4, i4))*g(bis(4, i2),bis(4, i3)))

# === fermion: -(g/2)(psibar psi)^2 [matrix element] ===

- (g/2)(psibar psi)^2 [matrix element]
alphas = [psibar0, psi0, psibar0, psi0]
betas = [b1, b2, b3, b4]
ps = [p1, p2, p3, p4]

Vertex:
1𝑖*(-1/2*g_psi4*UF(psi0,p2,s2,alpha_s)*UF(psi0,p4,s4,beta_s)*UbarF(psibar0,p1,s1,alpha_s)UbarF(psibar0,p3,s3,beta_s)+1/2g_psi4*UF(psi0,p2,s2,alpha_s)*UF(psi0,p4,s4,beta_s)*UbarF(psibar0,p1,s1,beta_s)UbarF(psibar0,p3,s3,alpha_s)+1/2g_psi4*UF(psi0,p2,s2,beta_s)*UF(psi0,p4,s4,alpha_s)*UbarF(psibar0,p1,s1,alpha_s)UbarF(psibar0,p3,s3,beta_s)-1/2g_psi4*UF(psi0,p2,s2,beta_s)*UF(psi0,p4,s4,alpha_s)*UbarF(psibar0,p1,s1,beta_s)*UbarF(psibar0,p3,s3,alpha_s))

# === fermion: -(g/2)(psibar psi)^2 [explicit spinor labels] ===

- (g/2)(psibar psi)^2 [spinor delta]
alphas = [psibar0, psi0, psibar0, psi0]
betas = [b1, b2, b3, b4]
ps = [p1, p2, p3, p4]

Vertex:
1𝑖*(-g_psi4*g(bis(4, i1),bis(4, i2))*g(bis(4, i3),bis(4, i4))+g_psi4*g(bis(4, i1),bis(4, i4))*g(bis(4, i2),bis(4, i3)))

================================================================================
-(g/2)(psibar psi)(psibar psi)  →  V = (-ig)[g(i1,i2)*g(i3,i4) - g(i1,i4)*g(i3,i2)]

# === fermion: current-current operator ===

gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]
alphas = [psibar0, psi0, psibar0, psi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Compact override:
2𝑖*gJJ*gamma(bis(4, i1),bis(4, i2),mink(4, mu))*gamma(bis(4, i3),bis(4, i4),mink(4, mu))-2𝑖*gJJ*gamma(bis(4, i1),bis(4, i4),mink(4, mu))*gamma(bis(4, i3),bis(4, i2),mink(4, mu))

Interpretation: stripped output keeps the direct minus exchange gamma structure visible.

=== fermion: operator-order diagnostics (psibar psi)^2 ===
swap-both bilinears:  == V_orig
swap-one bilinear:   == -V_orig

# === fermion+scalar: mixed derivatives ===

yF * (d_mu psibar) * psi * phi * chi
alphas = [psibar0, psi0, phi0, chi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
yF*g(bis(4, i1),bis(4, i2))*pcomp(p1,mu)

================================================================================
yF * psibar * (d_nu psi) * phi * chi
alphas = [psibar0, psi0, phi0, chi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
yF*g(bis(4, i1),bis(4, i2))*pcomp(p2,nu)

================================================================================
yF * psibar * psi * (d_mu phi) * (d_nu chi)
alphas = [psibar0, psi0, phi0, chi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
-1𝑖*yF*g(bis(4, i1),bis(4, i2))*pcomp(p3,mu)*pcomp(p4,nu)

================================================================================
g * (d_mu psibar)(d_nu psi) phi phi chi
alphas = [psibar0, psi0, phi0, phi0, chi0]
betas  = [b1, b2, b3, b4, b5]
ps     = [p1, p2, p3, p4, p5]

Vertex:
-2𝑖*g*g(bis(4, i1),bis(4, i2))*pcomp(p1,mu)*pcomp(p2,nu)

================================================================================
g1 * psibar * psi * (d^2 phi) * chi
alphas = [psibar0, psi0, phi0, chi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
-1𝑖*g1*g(bis(4, i1),bis(4, i2))*pcomp(p3,mu)^2

================================================================================
g2 * psibar * psi * (d_mu d_nu phi)(d_mu d_nu phi)
alphas = [psibar0, psi0, phi0, phi0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
2𝑖*g2*g(bis(4, i1),bis(4, i2))*pcomp(p3,mu)*pcomp(p3,nu)*pcomp(p4,mu)*pcomp(p4,nu)

# === gauge-ready: non-abelian fermion current ===

gS * psibar gamma^mu T^a psi G^a_mu
alphas = [psibar0, psi0, G0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
1𝑖*gS*gamma(bis(4, i1),bis(4, i2),mink(4, mu3))*t(coad(8, a3),cof(3, c1),cof(3, c2))

Interpretation: the coupling now remaps spinor, Lorentz, and color labels through one slot-label path.

=== gauge-ready: complex scalar current ===

gPhiA * A_mu * phi^dagger <-> d^mu phi
alphas = [phiCdag0, phiC0, A0]
betas  = [b1, b2, b3]
ps     = [p1, p2, p3]

Vertex:
-gPhiA*pcomp(p1,mu3)+gPhiA*pcomp(p2,mu3)

Interpretation: the gauge-field Lorentz slot now remaps into the derivative index as well.

# === gauge-ready: complex scalar contact ===

gPhiAA * A_mu A^mu phi^dagger phi
alphas = [phiCdag0, phiC0, A0, A0]
betas  = [b1, b2, b3, b4]
ps     = [p1, p2, p3, p4]

Vertex:
2𝑖*gPhiAA*g(mink(4, mu3),mink(4, mu4))

Interpretation: repeated gauge legs stay bosonic, while distinct scalar/scalar_dag roles keep the matter flow explicit.

# ================================================================================
Tests: all

phi^4: PASS
phi^2 chi^2: PASS
phi^dagger phi: PASS
Derivative (mu,nu): PASS
Derivative (mu,mu): PASS
Multi-species (base): PASS
Multi-species (perm): PASS
Multi-species (bad mult -> 0): PASS

Scalar+derivative tests passed.

Yukawa (amputated): PASS
Yukawa (unstripped, has UF/UbarF): PASS
Vector current: PASS
Axial current: PASS
underspecified multi-fermion operator rejected: PASS
(psibar psi)^2 amputated: PASS
(psibar psi)^2 matrix element (non-zero direct/exchange): PASS
(psibar psi)^2 spinor deltas: PASS
Current-current stripped: PASS
Current-current unstripped (non-zero): PASS
Missing fermion leg spinor index -> ValueError: PASS

Fermion tests passed.

d_mu psibar: PASS
d_nu psi: PASS
(d_mu phi)(d_nu chi): PASS
5pt mixed: PASS
g1 * psibar psi (d^2 phi) chi: PASS
g2 * psibar psi (d_mu d_nu phi)^2: PASS

Mixed fermion+scalar derivative tests passed.

Gauge-ready quark-gluon current: PASS
Complex scalar gauge current: PASS
Complex scalar gauge contact: PASS

Gauge-ready tests passed.

All selected tests passed.
(.venv) rems@student-net-hci-3769 thesis-code % /Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/t
hesis-code/.venv/bin/python /Users/rems/Library/CloudStorage/OneDrive-ETHZurich/ETHz/ETHz_FS26/MScThesis/thesis-code/code/spen
so_gamma_checks.py

# ================================================================================
Demo: spenso-gamma

== gamma identities ==
Tr(gamma^mu gamma_mu) = 16.0000000000000

{gamma^mu, gamma^nu} =
2*g(mink(4, mu),mink(4, nu))*g(bis(4, si),bis(4, sk))

== bilinears ==
gBilin * psibar gamma^mu psi
1𝑖*gBilin*gamma(bis(4, i1),bis(4, i2),mink(4, mu))

gBilin * psibar gamma5 psi
1𝑖*gBilin*gamma5(bis(4, i1),bis(4, i2))

gBilin * psibar sigma^{mu nu} psi
1𝑖*gBilin*sigma(bis(4, i1),bis(4, i2),mink(4, mu),mink(4, nu))

gBilin * psibar P_L psi
-1𝑖/2*gBilin*gamma5(bis(4, i1),bis(4, i2))+1𝑖/2*gBilin*g(bis(4, i1),bis(4, i2))

gBilin * psibar P_R psi
1𝑖/2*gBilin*gamma5(bis(4, i1),bis(4, i2))+1𝑖/2*gBilin*g(bis(4, i1),bis(4, i2))

gBilin * psibar gamma_mu psi
equivalent after metric simplification: True

gBilin * psibar gamma^mu gamma^nu psi
1𝑖*gBilin*gamma(bis(4, alpha),bis(4, i2),mink(4, nu))*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))

gBilin * psibar gamma^mu gamma^nu gamma^rho psi
raw:
1𝑖*gBilin*gamma(bis(4, alpha),bis(4, beta),mink(4, nu))*gamma(bis(4, beta),bis(4, i2),mink(4, rho))gamma(bis(4, i1),bis(4, alpha),mink(4, mu))
simplified:
1𝑖gBilin*gamma(bis(4, alpha),bis(4, i2),mink(4, rho))*gamma(bis(4, beta),bis(4, alpha),mink(4, nu))gamma(bis(4, i1),bis(4, beta),mink(4, mu))-2𝑖gBilin*g(mink(4, mu),mink(4, nu))*gamma(bis(4, i1),bis(4, i2),mink(4, rho))+2𝑖*gBilin*g(mink(4, mu),mink(4, rho))*gamma(bis(4, i1),bis(4, i2),mink(4, nu))

gBilin * psibar gamma^mu gamma^nu gamma5 psi
1𝑖*gBilin*gamma(bis(4, alpha),bis(4, beta),mink(4, nu))*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma5(bis(4, beta),bis(4, i2))

gBilin * psibar gamma^nu gamma^mu psi
-1𝑖*gBilin*gamma(bis(4, alpha),bis(4, i2),mink(4, nu))*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))+2𝑖*gBilin*g(mink(4, mu),mink(4, nu))*g(bis(4, i1),bis(4, i2))

sum of both orderings
2𝑖*gBilin*g(mink(4, mu),mink(4, nu))*g(bis(4, i1),bis(4, i2))

gBilin * psibar gamma_mu gamma_nu psi
equivalent after metric simplification: True

== current-current operator ==
gBilin * (psibar gamma^mu psi)(psibar gamma_mu psi)  [stripped]
2𝑖*gBilin*gamma(bis(4, i1),bis(4, i2),mink(4, mu))*gamma(bis(4, i3),bis(4, i4),mink(4, mu))-2𝑖*gBilin*gamma(bis(4, i1),bis(4, i4),mink(4, mu))*gamma(bis(4, i3),bis(4, i2),mink(4, mu))

gBilin * (psibar gamma^mu gamma5 psi)(psibar gamma_mu gamma5 psi)  [stripped]
1𝑖*gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, beta),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i2))gamma5(bis(4, beta),bis(4, i4))-1𝑖gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, beta),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i4))gamma5(bis(4, beta),bis(4, i2))-1𝑖gBilin*gamma(bis(4, i1),bis(4, beta),mink(4, mu))*gamma(bis(4, i3),bis(4, alpha),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i2))gamma5(bis(4, beta),bis(4, i4))+1𝑖gBilin*gamma(bis(4, i1),bis(4, beta),mink(4, mu))*gamma(bis(4, i3),bis(4, alpha),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i4))*gamma5(bis(4, beta),bis(4, i2))

gBilin * (psibar gamma^mu P_L psi)(psibar gamma_mu P_L psi)  [stripped]
1𝑖/4*gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, beta),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i2))gamma5(bis(4, beta),bis(4, i4))-1𝑖/4gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, beta),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i4))gamma5(bis(4, beta),bis(4, i2))+1𝑖/4gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, i2),mink(4, mu))gamma5(bis(4, alpha),bis(4, i4))-1𝑖/4gBilin*gamma(bis(4, i1),bis(4, alpha),mink(4, mu))*gamma(bis(4, i3),bis(4, i4),mink(4, mu))gamma5(bis(4, alpha),bis(4, i2))-1𝑖/4gBilin*gamma(bis(4, i1),bis(4, beta),mink(4, mu))*gamma(bis(4, i3),bis(4, alpha),mink(4, mu))*gamma5(bis(4, alpha),bis(4, i2))gamma5(bis(4, beta),bis(4, i4))+1𝑖/4gBilin*gamma(bis(4, i1),bis(4, beta),mink(4, mu))*gamma(bis(4, i3),bis(4, alpha),mink(4, mu))gamma5(bis(4, alpha),bis(4, i4))gamma5(bis(4, beta),bis(4, i2))+1𝑖/4gBilingamma(bis(4, i1),bis(4, beta),mink(4, mu))gamma(bis(4, i3),bis(4, i2),mink(4, mu))gamma5(bis(4, beta),bis(4, i4))-1𝑖/4gBilingamma(bis(4, i1),bis(4, beta),mink(4, mu))gamma(bis(4, i3),bis(4, i4),mink(4, mu))gamma5(bis(4, beta),bis(4, i2))-1𝑖/4gBilingamma(bis(4, i1),bis(4, i2),mink(4, mu))gamma(bis(4, i3),bis(4, alpha),mink(4, mu))gamma5(bis(4, alpha),bis(4, i4))-1𝑖/4gBilingamma(bis(4, i1),bis(4, i2),mink(4, mu))gamma(bis(4, i3),bis(4, beta),mink(4, mu))gamma5(bis(4, beta),bis(4, i4))+1𝑖/2gBilingamma(bis(4, i1),bis(4, i2),mink(4, mu))gamma(bis(4, i3),bis(4, i4),mink(4, mu))+1𝑖/4gBilin*gamma(bis(4, i1),bis(4, i4),mink(4, mu))*gamma(bis(4, i3),bis(4, alpha),mink(4, mu))gamma5(bis(4, alpha),bis(4, i2))+1𝑖/4gBilin*gamma(bis(4, i1),bis(4, i4),mink(4, mu))*gamma(bis(4, i3),bis(4, beta),mink(4, mu))gamma5(bis(4, beta),bis(4, i2))-1𝑖/2gBilin*gamma(bis(4, i1),bis(4, i4),mink(4, mu))*gamma(bis(4, i3),bis(4, i2),mink(4, mu))

# ================================================================================
Complex Scalar Structures

lamC * phi^dagger phi
1𝑖*lamC

gPhiA * A_mu * phi^dagger <-> d^mu phi
-gPhiA*pcomp(p1,mu3)+gPhiA*pcomp(p2,mu3)

gPhiAA * A_mu A^mu phi^dagger phi
2𝑖*gPhiAA*g(mink(4, mu3),mink(4, mu4))

== tests ==
Clifford anticommutator: PASS
psibar gamma^mu psi: PASS
psibar gamma5 psi: PASS
psibar sigma^{mu nu} psi: PASS
P_L + P_R = 1: PASS
P_R - P_L = gamma5: PASS
psibar gamma_mu psi: PASS
psibar gamma^mu gamma^nu psi: PASS
psibar gamma^nu gamma^mu psi: PASS
psibar gamma_mu gamma_nu psi: PASS
gamma^mu gamma^nu + gamma^nu gamma^mu: PASS
stripped current-current: PASS
raw psibar gamma^mu gamma^nu gamma^rho psi: PASS
simplified psibar gamma^mu gamma^nu gamma^rho psi: PASS
psibar gamma^mu gamma^nu gamma^rho psi: PASS
psibar gamma^mu gamma^nu gamma5 psi: PASS
phi^dagger phi: PASS
complex scalar gauge current: PASS
complex scalar gauge contact: PASS

All selected tests passed.
