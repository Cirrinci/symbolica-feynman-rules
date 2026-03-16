from model import (
    Field,
    Parameter,
    OperatorFactor,
    ExternalLeg,
    LagrangianTerm,
    Lagrangian,
    gamma,
    mu,
    delx,
    i,
    j,
    f,
    I,
    valid_contractions,
    canonical_vertex,
)
# -----------------------------------------------------------------------------
# 6) Model objects
# -----------------------------------------------------------------------------

Phi = Field("phi", kind="scalar", indexed=False, self_conjugate=True, mass="mphi")
Psi = Field("psi", kind="fermion", indexed=True, self_conjugate=False, mass="mpsi")
Chi = Field("chi", kind="scalar", indexed=False, self_conjugate=True)
Phi_i = Field("phi_i", kind="scalar", indexed=False, self_conjugate=True)
Phi_j = Field("phi_j", kind="scalar", indexed=False, self_conjugate=True)
Phi_k = Field("phi_k", kind="scalar", indexed=False, self_conjugate=True)
gpar = Parameter("gpar")
mpar = Parameter("m")
ypar = Parameter("y")
lampar = Parameter("lam")
gijk = Parameter("gijk")


# -----------------------------------------------------------------------------
# 7) Simple Lagrangian terms in native Spenso/Symbolica style
# -----------------------------------------------------------------------------

# scalar mass: -(1/2) m^2 phi^2
L_scalar_mass = LagrangianTerm(
    "scalar_mass",
    -(1/2) * mpar.symbol**2 * Phi() * Phi(),
    fields=("phi", "phi"),
    coefficient=-(1/2) * mpar.symbol**2,
    factors=[
        OperatorFactor(Phi),
        OperatorFactor(Phi),
    ],
)

# scalar quartic, phi^2 chi^2: -(g/2) phi^2 chi^2
L_phi2chi2 = LagrangianTerm(
    "phi2chi2",
    -(gpar.symbol) * Phi() * Phi() * Chi() * Chi(),
    fields=("phi", "phi", "chi", "chi"),
    coefficient=-(gpar.symbol),
    factors=[
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Chi),
        OperatorFactor(Chi),
    ],
)

# scalar quartic: -(lam/4!) phi^4
L_phi4 = LagrangianTerm(
    "phi4",
    -(lampar.symbol) * Phi() * Phi() * Phi() * Phi(),
    fields=("phi", "phi", "phi", "phi"),
    coefficient=-(lampar.symbol),
    factors=[
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
    ],
)

#scalar sextic: -(lam/6!) phi^6
L_phi6 = LagrangianTerm(
    "phi6",
    -(lampar.symbol) * Phi() * Phi() * Phi() * Phi()* Phi() * Phi(),
    fields=("phi", "phi", "phi", "phi", "phi", "phi"),
    coefficient=-(lampar.symbol),
    factors=[
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
        OperatorFactor(Phi),
    ],
)

# scalar sextic with three species: -gijk phi_i^2 phi_j^2 phi_k^2
L_phi_i2_phi_j2_phi_k2 = LagrangianTerm(
    "phi_i2_phi_j2_phi_k2",
    -gijk.symbol * Phi_i() * Phi_i() * Phi_j() * Phi_j() * Phi_k() * Phi_k(),
    fields=("phi_i", "phi_i", "phi_j", "phi_j", "phi_k", "phi_k"),
    coefficient=-gijk.symbol,
    factors=[
        OperatorFactor(Phi_i),
        OperatorFactor(Phi_i),
        OperatorFactor(Phi_j),
        OperatorFactor(Phi_j),
        OperatorFactor(Phi_k),
        OperatorFactor(Phi_k),
    ],
)

# fermion mass: -m psibar psi
L_fermion_mass = LagrangianTerm(
    "fermion_mass",
    -mpar.symbol * Psi.conjugate(i, f) * Psi(i, f),
    fields=("psibar", "psi"),
    coefficient=-mpar.symbol,
    factors=[
        OperatorFactor(Psi, indices=(i, f), conjugated=True),
        OperatorFactor(Psi, indices=(i, f), conjugated=False),
    ],
)

# Yukawa: -y phi psibar psi
L_yukawa = LagrangianTerm(
    "yukawa",
    -ypar.symbol * Phi() * Psi.conjugate(i, f) * Psi(i, f),
    fields=("phi", "psibar", "psi"),
    coefficient=-ypar.symbol,
    factors=[
        OperatorFactor(Phi),
        OperatorFactor(Psi, indices=(i, f), conjugated=True),
        OperatorFactor(Psi, indices=(i, f), conjugated=False),
    ],
)

# fermion kinetic: I psibar gamma del psi
# derivative kept as actual expr and also as metadata in factors
L_fermion_kin = LagrangianTerm(
    "fermion_kinetic",
    I * Psi.conjugate(i, f) * gamma(i, j, mu) * delx(mu) * Psi(j, f),
    fields=("psibar", "psi"),
    coefficient=I,
    factors=[
        OperatorFactor(Psi, indices=(i, f), conjugated=True),
        OperatorFactor(Psi, indices=(j, f), derivative_indices=(mu,), conjugated=False),
    ],
)


# -----------------------------------------------------------------------------
# 8) Full Lagrangian
# -----------------------------------------------------------------------------

L = Lagrangian([
    L_scalar_mass,
    L_phi2chi2,
    L_phi4,
    L_phi6,
    L_phi_i2_phi_j2_phi_k2,
    L_fermion_mass,
    L_yukawa,
    L_fermion_kin,
])


# -----------------------------------------------------------------------------
# 9) Minimal tests
# -----------------------------------------------------------------------------


def _run_tests():
    total = L.total()
    assert total is not None
    assert len(L.terms) == 8
    assert str(Phi()) == "phi"
    assert Psi(i, f) is not None
    assert gamma(i, j, mu) is not None

    # bookkeeping
    assert L_scalar_mass.fields == ("phi", "phi")
    assert L_phi4.fields == ("phi", "phi", "phi", "phi")
    assert L_phi6.fields == ("phi", "phi", "phi", "phi", "phi", "phi")
    assert L_yukawa.fields == ("phi", "psibar", "psi")

    # canonical data present
    assert L_phi4.has_canonical_data()
    assert L_yukawa.has_canonical_data()
    assert L_phi6.has_canonical_data()

    # combinatorial benchmarks (using current normalizations)
    legs_phi4 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Phi, "p3"),
        ExternalLeg(Phi, "p4"),
    ]

    contractions = valid_contractions(L_phi4, legs_phi4)
    assert len(contractions) == 24

    v4 = canonical_vertex(L_phi4, legs_phi4)
    assert v4 == -24 * I * lampar.symbol

    legs_phi2chi2 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Chi, "p3"),
        ExternalLeg(Chi, "p4"),
    ]

    contractions = valid_contractions(L_phi2chi2, legs_phi2chi2)
    assert len(contractions) == 4

    v = canonical_vertex(L_phi2chi2, legs_phi2chi2)
    assert v == -4 * I * gpar.symbol

    legs_phi6 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Phi, "p3"),
        ExternalLeg(Phi, "p4"),
        ExternalLeg(Phi, "p5"),
        ExternalLeg(Phi, "p6"),
    ]
    contractions = valid_contractions(L_phi6, legs_phi6)
    assert len(contractions) == 720

    # phi_i^2 phi_j^2 phi_k^2: expect 6! / (2! 2! 2!) = 90 contractions
    legs_phi_i2_j2_k2 = [
        ExternalLeg(Phi_i, "p1"),
        ExternalLeg(Phi_i, "p2"),
        ExternalLeg(Phi_j, "p3"),
        ExternalLeg(Phi_j, "p4"),
        ExternalLeg(Phi_k, "p5"),
        ExternalLeg(Phi_k, "p6"),
    ]
    contractions = valid_contractions(L_phi_i2_phi_j2_phi_k2, legs_phi_i2_j2_k2)
    assert len(contractions) == 8

    v_ijk = canonical_vertex(L_phi_i2_phi_j2_phi_k2, legs_phi_i2_j2_k2)
    assert v_ijk == -8 * I * gijk.symbol
    print("All basic scalar tests passed.")


# -----------------------------------------------------------------------------
# 10) Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Terms ===")
    L.show()

    print("\n=== Summary ===")
    L.summary()

    print("\n=== Canonical summary ===")
    L.canonical_summary()

    print("\n=== Total Lagrangian ===")
    print(L.total())

    print("\n=== phi^4 benchmark ===")
    legs_phi4 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Phi, "p3"),
        ExternalLeg(Phi, "p4"),
    ]
    print("valid contractions =", len(valid_contractions(L_phi4, legs_phi4)))
    print("phi4 vertex =", canonical_vertex(L_phi4, legs_phi4))

    legs_phi2chi2 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Chi, "p3"),
        ExternalLeg(Chi, "p4"),
    ]

    print("\n=== phi^2 chi^2 benchmark ===")

    contractions = valid_contractions(L_phi2chi2, legs_phi2chi2)
    print("valid contractions =", len(contractions))

    vertex = canonical_vertex(L_phi2chi2, legs_phi2chi2)
    print("vertex =", vertex)

    legs_phi6 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Phi, "p3"),
        ExternalLeg(Phi, "p4"),
        ExternalLeg(Phi, "p5"),
        ExternalLeg(Phi, "p6"),
    ]

    print("\n=== phi^6 benchmark ===")
    contractions = valid_contractions(L_phi6, legs_phi6)
    print("valid contractions =", len(contractions))

    vertex = canonical_vertex(L_phi6, legs_phi6)
    print("vertex =", vertex)

    print("\n=== phi_i^2 phi_j^2 phi_k^2 benchmark ===")
    legs_phi_i2_j2_k2 = [
        ExternalLeg(Phi_i, "p1"),
        ExternalLeg(Phi_i, "p2"),
        ExternalLeg(Phi_j, "p3"),
        ExternalLeg(Phi_j, "p4"),
        ExternalLeg(Phi_k, "p5"),
        ExternalLeg(Phi_k, "p6"),
    ]
    contractions = valid_contractions(L_phi_i2_phi_j2_phi_k2, legs_phi_i2_j2_k2)
    print("valid contractions =", len(contractions))
    vertex = canonical_vertex(L_phi_i2_phi_j2_phi_k2, legs_phi_i2_j2_k2)
    print("vertex =", vertex)


    print("\n=== Running tests ===")
    _run_tests()
    