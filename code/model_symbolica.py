"""
Feynman vertex rule derivation via canonical quantization -- mooooreeee Symbolica (attempt :D)

The implementation follows the canonical-quantization/Wick-contraction logic,
but works directly at the symbolic contraction level rather than constructing
full operator-valued expressions explicitly.

Current scope:
    - bosonic polynomial interaction terms
    - permutation-summed Wick contractions
    - derivative interactions via momentum factors
    - fermionic permutation signs and role-aware contractions

Pipeline:
    1. Select an interaction monomial
    2. Sum over contractions with external legs
    3. Evaluate derivatives as momentum factors per contraction
    4. Replace the plane-wave factor by momentum conservation
    5. Strip external wavefunctions
    6. Multiply by i to obtain the vertex factor
"""

from collections import Counter
from itertools import permutations
from math import factorial
from typing import Literal, Optional, Sequence

from symbolica import S, Expression

# ---------------------------------------------------------------------------
# Module-level Symbolica symbols... then spenso and more indices here and gamma etc....
# ---------------------------------------------------------------------------

phi, psi, adag = S("phi", "psi", "adag")
U = S("U")
UF = S("UF")
UbarF = S("UbarF")
gamma = S("gamma")
delta = S("delta")
delta_s = S("delta_s")   # spinor Kronecker delta  δ_s(i,j)
Delta = S("Delta")
Dot = S("Dot")
pcomp = S("pcomp")
D = S("D")

I = Expression.I
pi = Expression.PI


'''
!!!NOTE!!!
----
For statistics="fermion", this implementation includes Grassmann permutation
signs and role-aware contractions, with external wavefunctions UF/UbarF.
It does not yet build full gamma/index chains automatically from a Lagrangian.
'''
Statistics = Literal["boson", "fermion"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

#very naive i know
def plane_wave(p, x):
    """exp(-i p.x)"""
    return Expression.EXP(-I * Dot(p, x))

#only for dimonstration purposes... see step 1 in notebook
def contraction_rule(alpha, beta, p, x):
    """[phi_alpha(x), a^dag_beta(p)] = delta(alpha,beta) U(beta,p) exp(-i p.x)"""
    return delta(alpha, beta) * U(beta, p) * plane_wave(p, x)


def fermion_contraction_rule(alpha, beta, p, x, *, role, spin, spinor_index):
    """Fermionic contraction building block with explicit spinor wf.

    role="psi"    -> UF(beta,p,spin,spinor_index)
    role="psibar" -> UbarF(beta,p,spin,spinor_index)
    """
    if role == "psi":
        wf = UF(beta, p, spin, spinor_index)
    elif role == "psibar":
        wf = UbarF(beta, p, spin, spinor_index)
    else:
        raise ValueError(f"Unknown fermion role '{role}'")
    return delta(alpha, beta) * wf * plane_wave(p, x)

# ---------------------------------------------------------------------------
# Fermion list position
# ---------------------------------------------------------------------------

def factor_leg_compatible(i, j, alphas, betas, field_roles=None, leg_roles=None):
    """Compatibility check for matching factor i with external leg j.

    Species compatibility is intentionally left symbolic via delta(alphas[i], betas[j]).
    This helper enforces only optional role-level constraints (e.g. psi vs psibar).
    """
    if field_roles is not None and leg_roles is not None:
        return field_roles[i] == leg_roles[j]
    return True


def fermion_contraction_sign(perm, field_roles=None):
    """Sign of a fermion contraction permutation.

    Parameters
    ----------
    perm : list of integers
    field_roles : list of strings
    """
    if field_roles is None:
        return 1

    inv = 0
    fslots = [k for k,r in enumerate(field_roles) if r in ("psi","psibar")]
    if len(fslots) <= 1:
        return 1

    assigned = [perm[k] for k in fslots]
    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            if assigned[i] > assigned[j]:
                inv += 1
    return (-1) ** inv


def _default_spin_symbol(leg_position: int):
    return S(f"s{leg_position + 1}")


def _external_factor_for_contraction(
    *,
    role,
    alpha,
    beta,
    p,
    x,
    spin,
    spinor_index,
):
    if role == "psi":
        return delta(alpha, beta) * UF(beta, p, spin, spinor_index)
    if role == "psibar":
        return delta(alpha, beta) * UbarF(beta, p, spin, spinor_index)
    return delta(alpha, beta) * U(beta, p)




# ---------------------------------------------------------------------------
# Wick contractions (permutation sum)
# ---------------------------------------------------------------------------


def permutation_parity(perm) -> int:
    """Parity of a permutation: 0 for even, 1 for odd."""
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return inv % 2


def _infer_fermion_chains_from_endpoints(field_roles, field_spinor_indices):
    """Infer simple fermion chains with ordered endpoints.

    For now we only support delta_s-only spinor structure:
      each chain corresponds to one bilinear with exactly two fermion slots:
        (psibar endpoint, psi endpoint)

    The endpoints are inferred by matching equal non-None spinor index symbols
    and requiring one slot to be "psibar" and the other to be "psi".
    """
    if field_roles is None:
        raise ValueError("field_roles must be provided to infer fermion chains")
    if field_spinor_indices is None:
        raise ValueError("field_spinor_indices must be provided to infer fermion chains")
    if len(field_roles) != len(field_spinor_indices):
        raise ValueError("field_roles and field_spinor_indices must have the same length")

    groups = {}
    for i, si in enumerate(field_spinor_indices):
        if si is None:
            continue
        groups.setdefault(str(si), []).append(i)

    chains = []
    for spinor_label, slots in groups.items():
        if len(slots) != 2:
            raise ValueError(
                "Each non-None spinor index must appear exactly twice in "
                "leg_spinor_indices mode, got "
                f"{len(slots)} occurrences for index '{spinor_label}'."
            )
        a, b = slots
        if field_roles[a] == "psibar" and field_roles[b] == "psi":
            chains.append((a, b))
        elif field_roles[a] == "psi" and field_roles[b] == "psibar":
            chains.append((b, a))
        else:
            raise ValueError(
                "Invalid fermion chain endpoints: expected one 'psibar' and one 'psi' "
                f"for the same spinor index, got roles ({field_roles[a]}, {field_roles[b]})"
            )
    return chains


def contract_to_full_expression(
    *,
    alphas: Sequence,
    betas: Sequence,
    ps: Sequence,
    x,
    derivative_indices=(),
    derivative_targets: Optional[Sequence[int]] = None,
    statistics: Statistics = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
):
    """Sum over all Wick contractions to build the full vacuum matrix element.

    For bosons the sum is a permanent (all signs +1).
    For fermions the sign of each permutation contributes (-1)^parity.

    If derivative information is provided, derivative momentum factors are
    evaluated per permutation, i.e. the derivative acting on field k picks the
    momentum of the external leg that field k contracts with in that
    permutation... this caused many errors when I was trying to use it in the vertex_factor function :(

    When leg_spinor_indices is provided, fermion external factors are replaced
    by delta_s(leg_spinor_a, leg_spinor_b) connecting legs whose fields share
    a bilinear spinor contraction (inferred from field_spinor_indices).
    Bosonic fields still produce U(beta, p) as usual.
    """

    #first check if the lengths of the sequences are the same
    n = len(alphas)
    if not (len(betas) == len(ps) == n):
        raise ValueError("Nope! alphas, betas, ps must have the same length dear...")

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)

    if len(derivative_targets) != len(derivative_indices):
        raise ValueError("Nope! derivative_targets length must match derivative_indices... this is not gonna work...")

    for tgt in derivative_targets:
        if tgt < 0 or tgt >= n:
            raise ValueError(f"Nope! derivative target index {tgt} out of range for {n} fields... this is not gonna work...")

    if field_roles is not None and len(field_roles) != n:
        raise ValueError("field_roles must have the same length as alphas")
    if leg_roles is not None and len(leg_roles) != n:
        raise ValueError("leg_roles must have the same length as betas")
    if (field_roles is None) != (leg_roles is None):
        raise ValueError("Provide both field_roles and leg_roles, or neither")
    if field_spinor_indices is not None and len(field_spinor_indices) != n:
        raise ValueError("field_spinor_indices must have the same length as alphas")
    if leg_spins is not None and len(leg_spins) != n:
        raise ValueError("leg_spins must have the same length as ps")
    if leg_spinor_indices is not None and len(leg_spinor_indices) != n:
        raise ValueError("leg_spinor_indices must have the same length as ps")

    use_spinor_deltas = leg_spinor_indices is not None
    fermion_chains = []
    if use_spinor_deltas:
        if field_roles is None:
            raise ValueError("field_roles required when leg_spinor_indices is given")
        if field_spinor_indices is None:
            raise ValueError("field_spinor_indices required when leg_spinor_indices is given")
        for i, role in enumerate(field_roles):
            if role in ("psi", "psibar") and field_spinor_indices[i] is None:
                raise ValueError(
                    "Fermion slots must carry a spinor index when "
                    "leg_spinor_indices is provided"
                )
        # Fermion chain abstraction for delta_s-only case:
        # infer ordered endpoints (psibar_slot, psi_slot).
        fermion_chains = _infer_fermion_chains_from_endpoints(field_roles, field_spinor_indices)

    #now we can start the actual computation
    total = Expression.num(0)
    for perm in permutations(range(n)):
        term = Expression.num(1)

        # Skip incompatible role assignments early (species stays symbolic via deltas).
        valid = True
        for i, j in enumerate(perm):
            if not factor_leg_compatible(
                i,
                j,
                alphas,
                betas,
                field_roles=field_roles,
                leg_roles=leg_roles,
            ):
                valid = False
                break
        if not valid:
            continue

        if statistics == "fermion":
            if field_roles is not None:
                term *= Expression.num(fermion_contraction_sign(perm, field_roles))
            elif permutation_parity(perm) == 1:
                term *= Expression.num(-1)

        #first we evaluate the derivative momentum factors with the momentum assigned by this permutation
        for mu, tgt in zip(derivative_indices, derivative_targets):
            term *= (-I) * pcomp(ps[perm[tgt]], mu)

        #now we evaluate the delta and U factors
        p_sum = Expression.num(0)
        for i, j in enumerate(perm):
            role = field_roles[i] if field_roles is not None else None

            if use_spinor_deltas and role in ("psi", "psibar"):
                term *= delta(alphas[i], betas[j])
            else:
                spin = leg_spins[j] if leg_spins is not None else _default_spin_symbol(j)
                spinor_index = (
                    field_spinor_indices[i]
                    if field_spinor_indices is not None
                    else S(f"si{i + 1}")
                )
                term *= _external_factor_for_contraction(
                    role=role,
                    alpha=alphas[i],
                    beta=betas[j],
                    p=ps[j],
                    x=x,
                    spin=spin,
                    spinor_index=spinor_index,
                )
            p_sum += ps[j]

        if use_spinor_deltas:
            for psibar_slot, psi_slot in fermion_chains:
                term *= delta_s(
                    leg_spinor_indices[perm[psibar_slot]],
                    leg_spinor_indices[perm[psi_slot]],
                )

        term *= plane_wave(p_sum, x)
        total += term

    return total


# ---------------------------------------------------------------------------
# Derivative helpers
# ---------------------------------------------------------------------------


def infer_derivative_targets(field_derivative_map):
    """Build (derivative_indices, derivative_targets) from a per-field spec.

    Parameters
    ----------
    field_derivative_map : list of (field_index, [mu1, mu2, ...]) pairs.
        Example: [(0, [mu]), (2, [mu, nu])] means field 0 has d_mu,
        field 2 has d_mu d_nu.

    Returns
    -------
    (derivative_indices, derivative_targets) : tuple of two lists
    """
    indices = []
    targets = []
    for field_idx, lorentz_indices in field_derivative_map:
        for mu in lorentz_indices:
            indices.append(mu)
            targets.append(field_idx)
    return indices, targets


# ---------------------------------------------------------------------------
# Full vertex factor pipeline
# ---------------------------------------------------------------------------


def vertex_factor(
    *,
    coupling,
    alphas: Sequence,
    betas: Sequence,
    ps: Sequence,
    x,
    derivative_indices=(),
    derivative_targets=None,
    statistics: Statistics = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    strip_externals: bool = True,
    include_delta: bool = True,
    d=None,
):
    """Compute the Feynman vertex factor from an interaction term.

    This combines all algorithm steps:
        1. Contract fields with creation operators
        2. Evaluate derivative momentum factors for each contraction
           (permutation-aware assignment)
        3. Integrate over x: exp(-i sum(p).x) -> (2pi)^d Delta(sum(p))
        4. Strip external wavefunctions U(beta, p) -> 1
        5. Multiply by i

    Parameters
    ----------
    coupling : Symbolica expression for the coupling constant/tensor
    alphas : species labels for the fields in the Lagrangian term
    betas : species labels for the external particles
    ps : momentum symbols for each external leg
    x : spacetime position symbol
    derivative_indices : Lorentz indices for derivatives
    derivative_targets : which field each derivative acts on
    statistics : "boson" or "fermion"
    strip_externals : remove U(...) factors and leftover plane waves
    include_delta : replace the x-integral with momentum delta
    d : spacetime dimension symbol (defaults to S('d'))
    leg_spinor_indices : per-leg spinor index; when provided, fermion
        external factors become delta_s connecting legs whose Lagrangian
        fields share a bilinear contraction (inferred from
        field_spinor_indices). Bosonic factors are stripped normally.
    """
    contracted = contract_to_full_expression(
        alphas=alphas,
        betas=betas,
        ps=ps,
        x=x,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        statistics=statistics,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_spinor_indices=field_spinor_indices,
        leg_spins=leg_spins,
        leg_spinor_indices=leg_spinor_indices,
    )
    full = coupling * contracted

    if include_delta:
        if d is None:
            d = S("d")
        p_sum = Expression.num(0)
        for p in ps:
            p_sum += p
        full = full.replace(plane_wave(p_sum, x), (2 * pi) ** d * Delta(p_sum))

    if strip_externals:
        beta_, p_ = S("beta_", "p_")
        full = full.replace(U(beta_, p_), 1)
        if leg_spinor_indices is None:
            spin_, si_ = S("spin_", "si_")
            full = full.replace(UF(beta_, p_, spin_, si_), 1)
            full = full.replace(UbarF(beta_, p_, spin_, si_), 1)
        q_, x_ = S("q_", "x_")
        full = full.replace(Expression.EXP(-I * Dot(q_, x_)), 1)

    return I * full


# ---------------------------------------------------------------------------
# Helpers... worst part of the code... if you know a better way to do this, please let me know...  
# maybe better solutions with more advanced Symbolica features...or other tools...
# ---------------------------------------------------------------------------

# Provvisorio arghhh
def simplify_deltas(expr, species_map=None):
    """Lightweight delta simplification helper for controlled example cases.

    This routine is intended mainly for demo/test post-processing where the species
    assignment of external legs is already known. It is not a complete symbolic
    delta simplifier.

    Parameters
    ----------
    expr : Symbolica expression
    species_map : dict mapping beta_symbol -> species_symbol.
        When provided, delta(species, beta) is replaced by 1 for matching
        pairs, and all remaining deltas involving those betas become 0.
        If None, only delta(a, a) -> 1 is applied.
    """
    a_, b_ = S("a_", "b_")

    if species_map is not None:
        for beta_sym, species_sym in species_map.items():
            expr = expr.replace(delta(species_sym, beta_sym), Expression.num(1))
            other_ = S("other_")
            expr = expr.replace(delta(other_, beta_sym), Expression.num(0))

    expr = expr.replace(delta(a_, a_), Expression.num(1))
    return expr



def _species_key(x):
    return str(x)


def derivative_momentum_sum_expression(
    *,
    ps: Sequence,
    derivative_indices,
    derivative_targets=None,
    field_species: Optional[Sequence] = None,
    leg_species: Optional[Sequence] = None,
):
    """Build a compact momentum-sum expression for general derivative patterns.

    This computes the same momentum part as permutation summation, but groups
    permutations by the derivative-assignment pattern:
      - choose leg assignments for distinct derivative target fields
      - multiply by combinatorial multiplicity of remaining species-compatible
        contractions

    Parameters
    ----------
    ps : sequence of momentum symbols for external legs
    derivative_indices : sequence of Lorentz indices (mu, nu, ...)
    derivative_targets : which field slot each derivative acts on
    field_species : species label for each field slot in the interaction term
    leg_species : species label for each external leg (same length as ps)
    """
    n = len(ps)
    m = len(derivative_indices)

    if derivative_targets is None:
        derivative_targets = [0] * m
    if len(derivative_targets) != m:
        raise ValueError("derivative_targets length must match derivative_indices")

    if field_species is not None and len(field_species) != n:
        raise ValueError("field_species must have same length as ps")
    if leg_species is not None and len(leg_species) != n:
        raise ValueError("leg_species must have same length as ps")

    # Distinct field slots that carry at least one derivative, preserving order.
    unique_targets = []
    for t in derivative_targets:
        if t < 0 or t >= n:
            raise ValueError(f"derivative target index {t} out of range for {n} fields")
        if t not in unique_targets:
            unique_targets.append(t)

    k = len(unique_targets)
    target_slot = {t: s for s, t in enumerate(unique_targets)}
    total = Expression.num(0)

    for assigned_legs in permutations(range(n), k):
        # Species compatibility for targeted fields.
        if field_species is not None and leg_species is not None:
            ok = True
            for t in unique_targets:
                slot = target_slot[t]
                leg = assigned_legs[slot]
                if _species_key(field_species[t]) != _species_key(leg_species[leg]):
                    ok = False
                    break
            if not ok:
                continue

        # Product of momentum factors for this derivative assignment.
        monomial = Expression.num(1)
        for mu, t in zip(derivative_indices, derivative_targets):
            leg = assigned_legs[target_slot[t]]
            monomial *= pcomp(ps[leg], mu)

        # Count remaining species-compatible bijections.
        if field_species is not None and leg_species is not None:
            rem_field = [i for i in range(n) if i not in set(unique_targets)]
            rem_legs = [j for j in range(n) if j not in set(assigned_legs)]

            cf = Counter(_species_key(field_species[i]) for i in rem_field)
            cl = Counter(_species_key(leg_species[j]) for j in rem_legs)
            if cf != cl:
                continue

            multiplicity = 1
            for cnt in cf.values():
                multiplicity *= factorial(cnt)
        else:
            multiplicity = factorial(n - k)

        total += multiplicity * monomial

    return total


def compact_vertex_sum_form(
    *,
    coupling,
    ps: Sequence,
    derivative_indices,
    derivative_targets=None,
    d=None,
    field_species: Optional[Sequence] = None,
    leg_species: Optional[Sequence] = None,
):
    """Compact sum-form vertex for general derivative-target patterns.

    This returns the compact expression:
      i * coupling * (-i)^m * (2*pi)^d * Delta(sum p) * S_momentum
    where S_momentum is built by derivative_momentum_sum_expression(...).
    """
    if d is None:
        d = S("d")

    p_sum = Expression.num(0)
    for p in ps:
        p_sum += p

    m = len(derivative_indices)
    phase = I * ((-I) ** m)
    mom_sum = derivative_momentum_sum_expression(
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        field_species=field_species,
        leg_species=leg_species,
    )

    return phase * coupling * (2 * pi) ** d * Delta(p_sum) * mom_sum


def compact_sum_notation(
    *,
    derivative_indices,
    derivative_targets=None,
    n_legs=None,
):
    """Human-readable sigma notation for generic derivative assignment pattern.

    Example output:
      "(n-k)! * Σ_{a,b distinct} p_{a,mu} p_{b,nu}"
    """
    if n_legs is None:
        if derivative_targets:
            n_legs = max(derivative_targets) + 1
        else:
            raise ValueError("n_legs required when derivative_targets is empty")

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)

    unique_targets = []
    for t in derivative_targets:
        if t not in unique_targets:
            unique_targets.append(t)

    symbols = "abcdefghijklmnopqrstuvwxyz"
    if len(unique_targets) > len(symbols):
        raise ValueError("Too many unique derivative targets for notation helper")

    t_to_var = {t: symbols[i] for i, t in enumerate(unique_targets)}
    vars_used = [t_to_var[t] for t in unique_targets]

    terms = []
    for mu, t in zip(derivative_indices, derivative_targets):
        v = t_to_var[t]
        terms.append(f"p_{{{v},{mu}}}")

    product = " ".join(terms) if terms else "1"
    if len(vars_used) <= 1:
        cond = vars_used[0] if vars_used else ""
    else:
        cond = ", ".join(vars_used) + " distinct"

    pref = f"({n_legs - len(unique_targets)})!"
    if cond:
        return f"{pref} * Σ_{{{cond}}} {product}"
    return f"{pref}"
