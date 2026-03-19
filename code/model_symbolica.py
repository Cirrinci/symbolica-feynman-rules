"""
Feynman vertex rule derivation via canonical quantization -- pure Symbolica.

This module implements the standard algorithm for extracting Feynman vertex
factors from Lagrangian interaction terms using Symbolica expressions.  It is
a *functional* companion to the OOP-style ``model.py`` (which uses
symbolica.community.spenso) and can be used independently or for
cross-validation.

Algorithm steps:
    1. Select an interaction term of the Lagrangian
    2. Attach creation operators for external particles
    3. Form the vacuum matrix element
    4. Contract fields with creation operators (Wick contractions)
    5. Evaluate derivatives on plane waves  ->  momentum factors
    6. Strip external wavefunctions and plane-wave factor
    7. Multiply by i  ->  vertex factor
"""

from itertools import permutations
from typing import Literal, Optional, Sequence

from symbolica import S, Expression

# ---------------------------------------------------------------------------
# Module-level Symbolica symbols
# ---------------------------------------------------------------------------

phi, psi, adag = S("phi", "psi", "adag")
U = S("U")
delta = S("delta")
Delta = S("Delta")
Dot = S("Dot")
pcomp = S("pcomp")
D = S("D")

I = Expression.I
pi = Expression.PI

Statistics = Literal["boson", "fermion"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def plane_wave(p, x):
    """exp(-i p.x)"""
    return Expression.EXP(-I * Dot(p, x))


def contraction_rule(alpha, beta, p, x):
    """[phi_alpha(x), a^dag_beta(p)] = delta(alpha,beta) U(beta,p) exp(-i p.x)"""
    return delta(alpha, beta) * U(beta, p) * plane_wave(p, x)


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


def contract_to_full_expression(
    *,
    alphas: Sequence,
    betas: Sequence,
    ps: Sequence,
    x,
    statistics: Statistics = "boson",
):
    """Sum over all Wick contractions to build the full vacuum matrix element.

    For bosons the sum is a permanent (all signs +1).
    For fermions the sign of each permutation contributes (-1)^parity.
    """
    n = len(alphas)
    if not (len(betas) == len(ps) == n):
        raise ValueError("alphas, betas, ps must have the same length")

    total = Expression.num(0)
    for perm in permutations(range(n)):
        term = Expression.num(1)

        if statistics == "fermion" and permutation_parity(perm) == 1:
            term *= Expression.num(-1)

        p_sum = Expression.num(0)
        for i, j in enumerate(perm):
            term *= delta(alphas[i], betas[j]) * U(betas[j], ps[j])
            p_sum += ps[j]

        term *= plane_wave(p_sum, x)
        total += term

    return total


# ---------------------------------------------------------------------------
# Derivative momentum factors
# ---------------------------------------------------------------------------


def derivative_factors(
    derivative_indices,
    *,
    ps_by_field: Sequence,
    derivative_targets: Optional[Sequence[int]] = None,
):
    """Product of (-i)*pcomp(p_target, mu) for each derivative index.

    Parameters
    ----------
    derivative_indices : sequence of Symbolica symbols (Lorentz indices)
    ps_by_field : momenta assigned to each field, indexed by field position
    derivative_targets : which field each derivative acts on (defaults to all
        acting on field 0)
    """
    if not derivative_indices:
        return Expression.num(1)

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)

    if len(derivative_targets) != len(derivative_indices):
        raise ValueError("derivative_targets length must match derivative_indices")

    out = Expression.num(1)
    for mu, tgt in zip(derivative_indices, derivative_targets):
        out *= (-I) * pcomp(ps_by_field[tgt], mu)
    return out


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
    strip_externals: bool = True,
    include_delta: bool = True,
    d=None,
):
    """Compute the Feynman vertex factor from an interaction term.

    This combines all algorithm steps:
        1. Contract fields with creation operators
        2. Multiply by coupling and derivative momentum factors
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
    """
    contracted = contract_to_full_expression(
        alphas=alphas,
        betas=betas,
        ps=ps,
        x=x,
        statistics=statistics,
    )

    deriv = derivative_factors(
        derivative_indices,
        ps_by_field=ps,
        derivative_targets=derivative_targets,
    )

    full = coupling * deriv * contracted

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
        q_, x_ = S("q_", "x_")
        full = full.replace(Expression.EXP(-I * Dot(q_, x_)), 1)

    return I * full


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simplify_deltas(expr, species_map=None):
    """Simplify Kronecker deltas in a vertex expression.

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
