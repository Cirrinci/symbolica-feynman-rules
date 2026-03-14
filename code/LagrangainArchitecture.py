from itertools import permutations

from symbolica import S
from symbolica.community.spenso import Representation
from symbolica.community.spenso import TensorName as N

# -----------------------------------------------------------------------------
# Minimal model layer on top of Symbolica + Spenso
# Keep tensor algebra in Spenso, keep only light physics bookkeeping here.
# -----------------------------------------------------------------------------


# 1) Representations / index spaces
lor = Representation.mink(4)
spin = Representation.bis(4)
flav = Representation("flavor", 3)


# 2) Concrete indices (slots)
mu = lor("mu")
i = spin("i")
j = spin("j")
f = flav("f")


# 3) Tensor heads / symbols
# Built-in Spenso tensor
gamma = N.gamma()

# Undefined indexed tensors
psi = N("psi")
psibar = N("psibar")
delx = N("del")

# Scalar symbols
phi = S("phi")
m = S("m")
y = S("y")
lam = S("lam")
I = S("I")


# -----------------------------------------------------------------------------
# 4) Minimal metadata layer
# -----------------------------------------------------------------------------


class Field:
    def __init__(self, name, kind, indexed=False, self_conjugate=False, mass=None):
        # name is the name
        # kind is the type of field (e.g., scalar, fermion, vector)
        # indexed indicates whether the field carries indices (e.g., fermions)
        # self_conjugate indicates whether the field is its own conjugate (e.g., real scalar)
        # mass is the mass of the field

        self.name = name
        self.kind = kind
        self.indexed = indexed
        self.self_conjugate = self_conjugate
        self.mass = mass

        if indexed:
            self.head = N(name)
            self.conjugate_head = self.head if self_conjugate else N(f"{name}bar")
        else:
            self.head = S(name)
            self.conjugate_head = self.head if self_conjugate else S(f"{name}bar")

    def __call__(self, *indices):
        if self.indexed:
            return self.head(*indices)
        if indices:
            raise ValueError(f"Scalar field '{self.name}' takes no indices.")
        return self.head

    def conjugate(self, *indices):
        if self.indexed:
            return self.conjugate_head(*indices)
        if indices:
            raise ValueError(f"Scalar field '{self.name}' takes no indices.")
        return self.conjugate_head

    def __repr__(self):
        return (
            f"Field(name={self.name!r}, kind={self.kind!r}, indexed={self.indexed}, "
            f"self_conjugate={self.self_conjugate}, mass={self.mass!r})"
        )


class Parameter:
    def __init__(self, name, value=None, is_external=True):
        # kept because FeynRules distinguishes external/internal parameters
        self.name = name
        self.value = value
        self.is_external = is_external
        self.symbol = S(name)

    def __repr__(self):
        return (
            f"Parameter(name={self.name!r}, value={self.value!r}, "
            f"is_external={self.is_external})"
        )


class OperatorFactor:
    """
    One operator-valued factor appearing in a Lagrangian term.

    field:
        Field metadata object

    indices:
        optional tuple of explicit indices for indexed fields

    derivative_indices:
        optional tuple of derivative Lorentz indices.
        For phi^4 this is empty, but later d_mu(phi) can be represented.

    conjugated:
        whether this is the conjugate field
    """

    def __init__(self, field, indices=None, derivative_indices=None, conjugated=False):
        self.field = field
        self.indices = tuple(indices) if indices else ()
        self.derivative_indices = tuple(derivative_indices) if derivative_indices else ()
        self.conjugated = conjugated

    def expr(self):
        base = self.field.conjugate(*self.indices) if self.conjugated else self.field(*self.indices)

        # For now we keep derivatives as metadata only.
        # Later this can be turned into an actual symbolic derivative structure.
        if self.derivative_indices:
            out = base
            for dind in self.derivative_indices:
                out = delx(dind) * out
            return out
        return base

    def short_name(self):
        if self.conjugated and not self.field.self_conjugate:
            return f"{self.field.name}bar"
        return self.field.name

    def __repr__(self):
        return (
            f"OperatorFactor(field={self.field.name!r}, indices={self.indices}, "
            f"derivative_indices={self.derivative_indices}, conjugated={self.conjugated})"
        )


class ExternalLeg:
    """
    One external particle attached to the vertex.

    For now:
      - field species
      - momentum label
      - optional indices
      - conjugation flag

    This is deliberately simple now, but it is the right place to extend later
    for spin, polarization, incoming/outgoing convention, etc.
    """

    def __init__(self, field, momentum, indices=None, conjugated=False):
        self.field = field
        self.momentum = momentum
        self.indices = tuple(indices) if indices else ()
        self.conjugated = conjugated

    def __repr__(self):
        return (
            f"ExternalLeg(field={self.field.name!r}, momentum={self.momentum!r}, "
            f"indices={self.indices}, conjugated={self.conjugated})"
        )


class LagrangianTerm:
    """
    Keep expr for Symbolica/Spenso inspection.
    Add coefficient + factors for canonical-quantization-style vertex extraction.
    Keep fields as lightweight metadata / summary.
    """

    def __init__(self, name, expr, fields=None, coefficient=None, factors=None):
        self.name = name
        self.expr = expr
        self.fields = tuple(fields) if fields else ()
        self.coefficient = coefficient
        self.factors = list(factors) if factors else []

    def __repr__(self):
        return f"{self.name} = {self.expr}"

    def n_factors(self):
        return len(self.factors)

    def has_canonical_data(self):
        return self.coefficient is not None and len(self.factors) > 0

    def canonical_summary(self):
        factor_names = [fac.short_name() for fac in self.factors]
        return {
            "name": self.name,
            "coefficient": self.coefficient,
            "factors": factor_names,
        }


class Lagrangian:
    def __init__(self, terms=None):
        self.terms = list(terms) if terms else []

    def add(self, term):
        self.terms.append(term)

    def total(self):
        if not self.terms:
            return S("0")
        out = self.terms[0].expr
        for term in self.terms[1:]:
            out = out + term.expr
        return out

    def show(self):
        for term in self.terms:
            print(term)

    def summary(self):
        for term in self.terms:
            print(f"{term.name}: fields={term.fields}")

    def canonical_summary(self):
        for term in self.terms:
            if term.has_canonical_data():
                print(term.canonical_summary())
            else:
                print(f"{term.name}: no canonical data")


# -----------------------------------------------------------------------------
# 5) Canonical-quantization-style helper functions
# -----------------------------------------------------------------------------

def factor_matches_leg(factor, leg):
    """
    First matching rule:
    - same field name
    - same conjugation flag

    Later this is where index compatibility, spin, vectors, ghosts, etc. go.
    """
    return (
        factor.field.name == leg.field.name
        and factor.conjugated == leg.conjugated
    )


def valid_contractions(term, external_legs):
    """
    Return all valid complete contractions between the ordered factors
    in the term and the external legs.

    For phi^4 with four identical real scalar legs this gives 4! = 24.
    """
    if not term.has_canonical_data():
        return []

    n = term.n_factors()
    if n != len(external_legs):
        return []

    out = []
    for perm in permutations(range(n)):
        ok = True
        for i_factor, i_leg in enumerate(perm):
            if not factor_matches_leg(term.factors[i_factor], external_legs[i_leg]):
                ok = False
                break
        if ok:
            out.append(perm)
    return out


def contraction_weight(term, contraction, external_legs):
    """
    Weight of one contraction.

    For the first scalar prototype:
    - each valid contraction contributes 1
    - no derivative momenta yet
    - no fermion minus signs yet

    This is intentionally the place where the real FeynRules-like complexity
    will be added later.
    """
    return 1


def canonical_vertex(term, external_legs):
    """
    Canonical-style vertex extraction for the current prototype.

    Steps:
    1. find all valid complete contractions
    2. sum their weights
    3. multiply by the term coefficient
    4. multiply by I
    """
    contractions = valid_contractions(term, external_legs)
    if not contractions:
        return None

    total_weight = 0
    for c in contractions:
        total_weight = total_weight + contraction_weight(term, c, external_legs)

    return I * term.coefficient * total_weight


def canonical_vertices_from_lagrangian(L, external_legs):
    out = []
    for term in L.terms:
        v = canonical_vertex(term, external_legs)
        if v is not None:
            out.append((term.name, v))
    return out


# -----------------------------------------------------------------------------
# 6) Model objects
# -----------------------------------------------------------------------------

Phi = Field("phi", kind="scalar", indexed=False, self_conjugate=True, mass="mphi")
Psi = Field("psi", kind="fermion", indexed=True, self_conjugate=False, mass="mpsi")
Chi = Field("chi", kind="scalar", indexed=False, self_conjugate=True)
gpar = Parameter("gpar")
mpar = Parameter("m")
ypar = Parameter("y")
lampar = Parameter("lam")


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
    L_phi4,
    L_phi2chi2,
    L_phi6,
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
    assert len(L.terms) == 7
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

    # first canonical benchmark: phi^4 -> -I*lam
    legs_phi4 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Phi, "p3"),
        ExternalLeg(Phi, "p4"),
    ]

    contractions = valid_contractions(L_phi4, legs_phi4)
    assert len(contractions) == 24

    v4 = canonical_vertex(L_phi4, legs_phi4)
    assert v4 == -24*I * lampar.symbol

    legs_phi2chi2 = [
        ExternalLeg(Phi, "p1"),
        ExternalLeg(Phi, "p2"),
        ExternalLeg(Chi, "p3"),
        ExternalLeg(Chi, "p4"),
    ]

    contractions = valid_contractions(L_phi2chi2, legs_phi2chi2)
    assert len(contractions) == 4

    v = canonical_vertex(L_phi2chi2, legs_phi2chi2)
    assert v == -4*I * gpar.symbol

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
    print("All basic tests passed.")


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


    print("\n=== Running tests ===")
    _run_tests()

