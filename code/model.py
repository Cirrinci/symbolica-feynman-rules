from itertools import permutations

from symbolica import S
from symbolica.community.spenso import Representation
from symbolica.community.spenso import TensorName as N


# -----------------------------------------------------------------------------
# Core model layer: representations, fields, parameters, Lagrangian, vertices
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
delx = N("del")

# Scalar symbol for the imaginary unit
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
            self.conjugate_head = self.head if self.self_conjugate else N(f"{name}bar")
        else:
            self.head = S(name)
            self.conjugate_head = self.head if self.self_conjugate else S(f"{name}bar")

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
# 5) Canonical-quantization-style helper functions (vertex extraction)
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

