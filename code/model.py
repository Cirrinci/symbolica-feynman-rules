from itertools import permutations
from math import factorial

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

# Momentum tensor head (symbolic): p(p_label, mu)
mom = N("p")

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

    def base_expr(self):
        return self.field.conjugate(*self.indices) if self.conjugated else self.field(*self.indices)

    def expr(self):
        """
        Symbolic expression for inspection/debugging.

        We represent derivatives as nested applications:
            del(phi, mu)
            del(del(phi, nu), mu)  for derivative_indices=(mu, nu)

        Note: Spenso's `TensorName` canonicalizes argument order, so we call
        `delx(out, mu)` (not `delx(mu, out)`).
        """
        out = self.base_expr()
        for dind in reversed(self.derivative_indices):
            out = delx(out, dind)
        return out

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
        self.momentum = S(momentum) if isinstance(momentum, str) else momentum
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
            return 0
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
        and len(factor.indices) == len(leg.indices)
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


def derivative_factor_for_leg(factor, leg):
    """
    Convert derivatives acting on a field into momentum factors.

    Convention (Fourier with exp(-i p x)):
        d_mu phi  ->  (-I) p_mu
        d_mu d_nu phi -> (-I)^2 p_mu p_nu
    """
    if not factor.derivative_indices:
        return 1

    out = 1
    for dind in factor.derivative_indices:
        out = out * (-I) * mom(leg.momentum, dind)
    return out


class ContractedTerm:
    """
    One fully contracted contribution from a term for one permutation of legs.
    """

    def __init__(self, coefficient, permutation, sign, momentum_factor, index_factor=None):
        self.coefficient = coefficient
        self.permutation = permutation
        self.sign = sign
        self.momentum_factor = momentum_factor
        self.index_factor = index_factor if index_factor is not None else 1

    def expr(self):
        return self.sign * self.coefficient * self.momentum_factor * self.index_factor

    def __repr__(self):
        return (
            f"ContractedTerm(permutation={self.permutation}, sign={self.sign}, "
            f"momentum_factor={self.momentum_factor}, index_factor={self.index_factor})"
        )


def fermion_reordering_sign(term, contraction, external_legs):
    """
    Placeholder for future Grassmann-sign logic.
    """
    return 1


def contracted_term(term, contraction, external_legs):
    """
    Build the symbolic contribution of one valid contraction.
    """
    sign = fermion_reordering_sign(term, contraction, external_legs)

    momentum_factor = 1
    index_factor = 1

    for i_factor, i_leg in enumerate(contraction):
        factor = term.factors[i_factor]
        leg = external_legs[i_leg]

        momentum_factor = momentum_factor * derivative_factor_for_leg(factor, leg)
        # Later:
        # - index_factor *= explicit index deltas / polarization structures / gamma chains

    return ContractedTerm(
        coefficient=term.coefficient,
        permutation=contraction,
        sign=sign,
        momentum_factor=momentum_factor,
        index_factor=index_factor,
    )


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

    total = 0
    for c in contractions:
        ct = contracted_term(term, c, external_legs)
        total = total + ct.expr()

    return I * total


def canonical_vertex_debug(term, external_legs):
    contractions = valid_contractions(term, external_legs)
    return [contracted_term(term, c, external_legs) for c in contractions]


def canonical_vertices_from_lagrangian(L, external_legs):
    out = []
    for term in L.terms:
        v = canonical_vertex(term, external_legs)
        if v is not None:
            out.append((term.name, v))
    return out


# -----------------------------------------------------------------------------
# 6) Faster bosonic vertex extraction (no n! permutations)
# -----------------------------------------------------------------------------


def _factor_signature(factor: OperatorFactor):
    """
    Signature used to group identical bosonic operator factors.

    This is the level of identity relevant for the current prototype:
    - same field species
    - same conjugation
    - same number of explicit indices
    - same derivative multi-index
    """
    return (
        factor.field.name,
        factor.conjugated,
        len(factor.indices),
        factor.derivative_indices,
    )


def _choose_k(seq, k):
    """
    Yield all k-element combinations from seq as tuples.
    Minimal helper to avoid importing itertools.combinations in the core file.
    """
    if k == 0:
        yield ()
        return
    if k > len(seq):
        return
    if k == 1:
        for x in seq:
            yield (x,)
        return
    first, rest = seq[0], seq[1:]
    for comb in _choose_k(rest, k - 1):
        yield (first,) + comb
    for comb in _choose_k(rest, k):
        yield comb


def fast_bosonic_vertex(term: LagrangianTerm, external_legs):
    """
    Faster vertex extraction for purely bosonic terms.

    Compared to `canonical_vertex`, this avoids enumerating n! permutations by:
    - grouping identical operator factors (same signature)
    - summing only over distinct assignments between *different* signatures
      (e.g. derivative vs non-derivative factors of the same field)
    - multiplying by prod_s factorial(count_s) for identical factors

    This matches the normalization implicit in your current `canonical_vertex`
    for the scalar prototypes.

    Limitations (by design for now):
    - does not handle fermionic Grassmann signs
    - does not yet build index factors beyond momentum from derivatives
    """
    if not term.has_canonical_data():
        return None

    if term.n_factors() != len(external_legs):
        return None

    # Only handle bosonic factors for now.
    for fac in term.factors:
        if getattr(fac.field, "kind", None) == "fermion":
            raise NotImplementedError("fast_bosonic_vertex does not handle fermions yet.")

    # Build groups of identical factor signatures.
    sig_to_count = {}
    sig_to_proto = {}
    for fac in term.factors:
        sig = _factor_signature(fac)
        sig_to_count[sig] = sig_to_count.get(sig, 0) + 1
        sig_to_proto.setdefault(sig, fac)

    signatures = list(sig_to_count.keys())

    # For each signature, precompute eligible leg indices.
    eligible = {}
    for sig in signatures:
        proto = sig_to_proto[sig]
        eligible[sig] = [
            idx for idx, leg in enumerate(external_legs) if factor_matches_leg(proto, leg)
        ]
        if len(eligible[sig]) < sig_to_count[sig]:
            return None

    # Factorial combinatorics for truly identical factors.
    identical_factor_multiplier = 1
    for sig, cnt in sig_to_count.items():
        identical_factor_multiplier *= factorial(cnt)

    # Recursively assign disjoint subsets of legs to each signature.
    def rec(sig_idx, remaining_legs_set, picked_map):
        if sig_idx == len(signatures):
            yield picked_map
            return

        sig = signatures[sig_idx]
        cnt = sig_to_count[sig]
        candidates = [i for i in eligible[sig] if i in remaining_legs_set]

        for chosen in _choose_k(candidates, cnt):
            new_remaining = set(remaining_legs_set)
            for c in chosen:
                new_remaining.remove(c)
            new_map = dict(picked_map)
            new_map[sig] = chosen
            yield from rec(sig_idx + 1, new_remaining, new_map)

    total = 0
    all_legs = set(range(len(external_legs)))
    for assignment in rec(0, all_legs, {}):
        mom_factor = 1
        for sig, chosen_legs in assignment.items():
            proto = sig_to_proto[sig]
            for leg_idx in chosen_legs:
                mom_factor *= derivative_factor_for_leg(proto, external_legs[leg_idx])
        total += mom_factor

    return I * term.coefficient * identical_factor_multiplier * total

