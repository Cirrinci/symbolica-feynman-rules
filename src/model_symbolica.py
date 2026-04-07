"""
Feynman vertex rule derivation via canonical quantization -- Symbolica engine.

This module owns the contraction logic.  It accepts either:
  (a) parallel-list kwargs directly (the original interface), or
  (b) model-layer objects via InteractionTerm.to_vertex_kwargs().

Current scope:
    - bosonic polynomial interaction terms
    - permutation-summed Wick contractions
    - derivative interactions via momentum factors
    - fermionic permutation signs and role-aware contractions
    - open index remapping for ANY index kind (spinor, Lorentz, colour, ...)
"""

from collections import Counter
from itertools import permutations
from math import factorial
from typing import Literal, Optional, Sequence

from symbolica import S, Expression
from symbolica.community.spenso import Representation
from symbolica.community.idenso import simplify_metrics

from spenso_structures import SPINOR_KIND

# ---------------------------------------------------------------------------
# Module-level Symbolica symbols + Spenso bispinor representation
# ---------------------------------------------------------------------------

phi, psi, adag = S("phi", "psi", "adag")
U = S("U")
UF = S("UF")
UbarF = S("UbarF")
gamma = S("gamma")
delta = S("delta", is_symmetric=True)
bis = Representation.bis(4)
mink = Representation.mink(4)
Delta = S("Delta")
Dot = S("Dot")
pcomp = S("pcomp")
D = S("D")

I = Expression.I
pi = Expression.PI

Statistics = Literal["boson", "fermion"]


# ---------------------------------------------------------------------------
# Role helpers (duck-typed: accept both FieldRole objects and legacy strings)
# ---------------------------------------------------------------------------

_STRING_TO_FERMION = frozenset(("psi", "psibar", "ghost", "ghost_dag"))


def _role_is_fermion(role) -> bool:
    if hasattr(role, "is_fermion"):
        return role.is_fermion
    return str(role) in _STRING_TO_FERMION


def _role_is_psi(role) -> bool:
    if hasattr(role, "name"):
        return role.name == "psi"
    return str(role) == "psi"


def _role_is_psibar(role) -> bool:
    if hasattr(role, "name"):
        return role.name == "psibar"
    return str(role) == "psibar"


def _roles_compatible(field_role, leg_role) -> bool:
    if hasattr(field_role, "compatible_with"):
        return field_role.compatible_with(leg_role)
    if hasattr(leg_role, "compatible_with"):
        return leg_role.compatible_with(field_role)
    return field_role == leg_role


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def plane_wave(p, x):
    """exp(-i p.x)"""
    return Expression.EXP(-I * Dot(p, x))


# ---------------------------------------------------------------------------
# Index-label helpers (generalized from spinor-only to any kind)
# ---------------------------------------------------------------------------

def _flatten_index_labels(index_labels_dict):
    """Flatten a {kind: label_or_labels} dict into [(kind, ordinal, label), ...].

    Supports both single labels and tuples/lists for fields with multiple
    indices of the same kind.
    """
    if not index_labels_dict:
        return []
    result = []
    for kind, labels in index_labels_dict.items():
        if isinstance(labels, (list, tuple)):
            for ordinal, label in enumerate(labels):
                if label is not None:
                    result.append((kind, ordinal, label))
        elif labels is not None:
            result.append((kind, 0, labels))
    return result


def _open_index_labels(field_index_labels, field_roles=None):
    """Find index labels that appear exactly once across all field slots.

    These are 'open' labels carried by the coupling tensor (e.g. gamma(mu,i,j))
    and must be remapped to external leg labels per contraction permutation.

    Returns: list of (field_slot, kind, ordinal, label)
    """
    if field_index_labels is None:
        return []

    counts = Counter()
    for slot_labels in field_index_labels:
        for kind, _, label in _flatten_index_labels(slot_labels):
            counts[(kind, str(label))] += 1

    open_slots = []
    for slot_idx, slot_labels in enumerate(field_index_labels):
        if field_roles is not None and not _role_is_fermion(field_roles[slot_idx]):
            pass
        for kind, ordinal, label in _flatten_index_labels(slot_labels):
            if counts[(kind, str(label))] == 1:
                open_slots.append((slot_idx, kind, ordinal, label))
    return open_slots


def _get_label(index_labels_dict, kind, ordinal=0):
    """Get one label for a given kind/ordinal from an index-label dict."""
    if not index_labels_dict:
        return None
    val = index_labels_dict.get(kind)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[ordinal] if 0 <= ordinal < len(val) else None
    return val


# ---------------------------------------------------------------------------
# Fermion helpers
# ---------------------------------------------------------------------------

def factor_leg_compatible(i, j, alphas, betas, field_roles=None, leg_roles=None):
    """Compatibility check for matching factor i with external leg j.

    Uses FieldRole.compatible_with() when roles are typed objects,
    falls back to equality for legacy strings.
    """
    if field_roles is not None and leg_roles is not None:
        return _roles_compatible(field_roles[i], leg_roles[j])
    return True


def _fermion_slots_from_roles(field_roles):
    """Return positions of fermionic fields in the interaction ordering."""
    return [i for i, role in enumerate(field_roles) if _role_is_fermion(role)]


def _group_spinor_slots(field_index_labels):
    """Group field slots by spinor label for chain inference."""
    groups = {}
    for i, slot_labels in enumerate(field_index_labels):
        spinor = _get_label(slot_labels, SPINOR_KIND)
        if spinor is None:
            continue
        groups.setdefault(str(spinor), []).append(i)
    return groups


def _infer_fermion_chains(field_roles, field_index_labels):
    """Infer bilinear fermion chains from repeated spinor labels.

    Each chain is (psibar_slot, psi_slot) where both slots share the same
    spinor label.
    """
    if field_roles is None or field_index_labels is None:
        return []

    groups = _group_spinor_slots(field_index_labels)
    chains = []

    for spinor_label, slots in groups.items():
        if len(slots) == 1:
            continue
        if len(slots) != 2:
            raise ValueError(
                f"Spinor index '{spinor_label}' appears {len(slots)} times; "
                "expected exactly 2 for a bilinear chain or 1 for an open slot."
            )
        a, b = slots
        ra, rb = field_roles[a], field_roles[b]
        if _role_is_psibar(ra) and _role_is_psi(rb):
            chains.append((a, b))
        elif _role_is_psi(ra) and _role_is_psibar(rb):
            chains.append((b, a))
        else:
            raise ValueError(
                f"Invalid fermion chain: spinor index '{spinor_label}' "
                f"connects roles ({ra}, {rb}), expected (psibar, psi)."
            )
    return chains


def _all_fermion_slots_labeled(field_roles, field_index_labels):
    """Whether every fermion slot has a spinor label."""
    if field_roles is None or field_index_labels is None:
        return False
    for i, role in enumerate(field_roles):
        if _role_is_fermion(role):
            if _get_label(field_index_labels[i], SPINOR_KIND) is None:
                return False
    return True


def _fermion_sign_from_slots(perm, fermion_slots):
    """Grassmann sign from permutation restricted to fermion slots."""
    if len(fermion_slots) <= 1:
        return 1
    assigned = [perm[k] for k in fermion_slots]
    inv = 0
    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            if assigned[i] > assigned[j]:
                inv += 1
    return (-1) ** inv


def _default_spin_symbol(leg_position: int):
    return S(f"s{leg_position + 1}")


def _default_leg_index_labels(num_legs: int):
    """Generate default per-leg index labels with spinor labels i1, i2, ..."""
    return [{SPINOR_KIND: S(f"i{k + 1}")} for k in range(num_legs)]


def _external_factor_for_contraction(*, role, alpha, beta, p, spin, spinor_index):
    if _role_is_psi(role):
        return delta(alpha, beta) * UF(beta, p, spin, spinor_index)
    if _role_is_psibar(role):
        return delta(alpha, beta) * UbarF(beta, p, spin, spinor_index)
    return delta(alpha, beta) * U(beta, p)


def _validate_supported_fermion_structure(field_roles, field_index_labels, coupling=None):
    """Reject underspecified multi-fermion operators."""
    if field_roles is None:
        return
    fermion_slots = _fermion_slots_from_roles(field_roles)
    if len(fermion_slots) <= 2:
        return

    if field_index_labels is None:
        raise ValueError(
            "Multi-fermion operators require explicit index labels. "
            "Provide field_index_labels with spinor labels, e.g. "
            '[{"spinor": alpha}, {"spinor": alpha}, {"spinor": beta}, {"spinor": beta}]'
        )

    chains = _infer_fermion_chains(field_roles, field_index_labels)
    if chains:
        return

    labels = [str(_get_label(field_index_labels[i], SPINOR_KIND)) for i in fermion_slots]
    if all(_get_label(field_index_labels[i], SPINOR_KIND) is not None for i in fermion_slots) \
       and len(set(labels)) == len(labels):
        if coupling is None:
            raise ValueError(
                "Explicit open-slot multi-fermion encoding requires coupling to "
                "contain those spinor labels, but coupling was not provided."
            )
        coupling_text = coupling.to_canonical_string()
        missing = [
            str(_get_label(field_index_labels[i], SPINOR_KIND))
            for i in fermion_slots
            if str(_get_label(field_index_labels[i], SPINOR_KIND)) not in coupling_text
        ]
        if missing:
            raise ValueError(
                "Open-slot labels missing from coupling: " + ", ".join(missing)
            )
        return

    raise ValueError(
        "Unsupported multi-fermion operator: provide either repeated dummy "
        "spinor labels for bilinear chains or fully explicit open-slot labels "
        "tied to the coupling tensor."
    )


# ---------------------------------------------------------------------------
# Backward-compat helpers: old params -> new params
# ---------------------------------------------------------------------------

def _merge_legacy_spinor_params(
    field_index_labels, leg_index_labels,
    field_spinor_indices, leg_spinor_indices,
    n,
):
    """Merge old field_spinor_indices/leg_spinor_indices into the new format."""
    if field_spinor_indices is not None and field_index_labels is None:
        field_index_labels = [
            {SPINOR_KIND: si} if si is not None else {}
            for si in field_spinor_indices
        ]
    if leg_spinor_indices is not None and leg_index_labels is None:
        leg_index_labels = [
            {SPINOR_KIND: si} if si is not None else {}
            for si in leg_spinor_indices
        ]
    return field_index_labels, leg_index_labels


# ---------------------------------------------------------------------------
# Wick contractions (permutation sum)
# ---------------------------------------------------------------------------

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
    field_index_labels: Optional[Sequence[dict]] = None,
    leg_index_labels: Optional[Sequence[dict]] = None,
    field_spinor_indices: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    leg_spins: Optional[Sequence] = None,
    coupling=None,
):
    """Sum over all Wick contractions to build the full vacuum matrix element.

    For bosons the sum is a permanent (all signs +1).
    For fermions the sign of each permutation contributes (-1)^parity.

    Index labels carried by field occurrences are tracked per contraction:
    labels appearing once (open slots) are remapped to the corresponding
    external leg labels.  Labels appearing twice (bilinear chains) produce
    bispinor metrics connecting the matched legs.

    This function is the core engine.  It stays agnostic about the origin of
    the inputs: the direct API and the model layer both reduce to the same
    parallel lists before reaching this point.
    """
    n = len(alphas)
    if not (len(betas) == len(ps) == n):
        raise ValueError("alphas, betas, ps must have the same length")

    if derivative_targets is None:
        derivative_targets = [0] * len(derivative_indices)
    if len(derivative_targets) != len(derivative_indices):
        raise ValueError("derivative_targets length must match derivative_indices")
    for tgt in derivative_targets:
        if tgt < 0 or tgt >= n:
            raise ValueError(f"derivative target {tgt} out of range for {n} fields")

    if field_roles is not None and len(field_roles) != n:
        raise ValueError("field_roles must have the same length as alphas")
    if leg_roles is not None and len(leg_roles) != n:
        raise ValueError("leg_roles must have the same length as betas")
    if (field_roles is None) != (leg_roles is None):
        raise ValueError("Provide both field_roles and leg_roles, or neither")

    field_index_labels, leg_index_labels = _merge_legacy_spinor_params(
        field_index_labels, leg_index_labels,
        field_spinor_indices, leg_spinor_indices,
        n,
    )

    if field_index_labels is not None and len(field_index_labels) != n:
        raise ValueError("field_index_labels must have the same length as alphas")
    if leg_index_labels is not None and len(leg_index_labels) != n:
        raise ValueError("leg_index_labels must have the same length as betas")
    if leg_spins is not None and len(leg_spins) != n:
        raise ValueError("leg_spins must have the same length as ps")

    if statistics == "fermion":
        if field_roles is None or leg_roles is None:
            raise ValueError(
                "statistics='fermion' requires both field_roles and leg_roles"
            )
        _validate_supported_fermion_structure(
            field_roles, field_index_labels, coupling=coupling,
        )

    use_spinor_deltas = leg_index_labels is not None and any(
        _get_label(entry, SPINOR_KIND) is not None
        for entry in leg_index_labels
    )

    fermion_chains = []
    if use_spinor_deltas and field_index_labels is not None:
        fermion_chains = _infer_fermion_chains(field_roles, field_index_labels)

    fermion_slots = (
        _fermion_slots_from_roles(field_roles)
        if statistics == "fermion" else []
    )

    open_index_slots = []
    if coupling is not None and field_index_labels is not None:
        open_index_slots = _open_index_labels(field_index_labels, field_roles)

    total = Expression.num(0)

    for perm in permutations(range(n)):
        term = Expression.num(1)

        coupling_term = coupling if coupling is not None else Expression.num(1)
        if open_index_slots and leg_index_labels is not None:
            for slot_idx, kind, ordinal, label in open_index_slots:
                target_leg = perm[slot_idx]
                target_label = _get_label(leg_index_labels[target_leg], kind, ordinal)
                if target_label is not None:
                    coupling_term = coupling_term.replace(label, target_label)
        term *= coupling_term

        valid = True
        for i, j in enumerate(perm):
            if not factor_leg_compatible(
                i, j, alphas, betas,
                field_roles=field_roles, leg_roles=leg_roles,
            ):
                valid = False
                break
        if not valid:
            continue

        if fermion_slots:
            term *= Expression.num(_fermion_sign_from_slots(perm, fermion_slots))

        for mu, tgt in zip(derivative_indices, derivative_targets):
            term *= (-I) * pcomp(ps[perm[tgt]], mu)

        p_sum = Expression.num(0)
        for i, j in enumerate(perm):
            role = field_roles[i] if field_roles is not None else None

            if use_spinor_deltas and _role_is_fermion(role):
                term *= delta(alphas[i], betas[j])
            else:
                spin = leg_spins[j] if leg_spins is not None else _default_spin_symbol(j)
                spinor_index = (
                    _get_label(field_index_labels[i], SPINOR_KIND)
                    if field_index_labels is not None
                    else S(f"si{i + 1}")
                )
                term *= _external_factor_for_contraction(
                    role=role,
                    alpha=alphas[i],
                    beta=betas[j],
                    p=ps[j],
                    spin=spin,
                    spinor_index=spinor_index or S(f"si{i + 1}"),
                )
            p_sum += ps[j]

        if use_spinor_deltas:
            for psibar_slot, psi_slot in fermion_chains:
                bar_label = _get_label(leg_index_labels[perm[psibar_slot]], SPINOR_KIND)
                psi_label = _get_label(leg_index_labels[perm[psi_slot]], SPINOR_KIND)
                if bar_label is not None and psi_label is not None:
                    term *= bis.g(bar_label, psi_label).to_expression()

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
    interaction=None,
    external_legs=None,
    coupling=None,
    alphas=None,
    betas=None,
    ps=None,
    x,
    derivative_indices=(),
    derivative_targets=None,
    statistics: Statistics = "boson",
    field_roles=None,
    leg_roles=None,
    field_index_labels=None,
    leg_index_labels=None,
    field_spinor_indices=None,
    leg_spinor_indices=None,
    leg_spins=None,
    strip_externals: bool = True,
    include_delta: bool = True,
    d=None,
):
    """Compute the Feynman vertex factor from an interaction term.

    Accepts either model-layer objects (interaction + external_legs) or the
    direct parallel-list interface.  The workflow is:

    1. normalize inputs into the engine format
    2. call contract_to_full_expression(...)
    3. optionally replace the plane wave by the universal momentum delta
    4. optionally strip external wavefunctions
    """
    if interaction is not None:
        if external_legs is None:
            raise ValueError("external_legs required when interaction is provided")
        kwargs = interaction.to_vertex_kwargs(external_legs)
        coupling = kwargs["coupling"]
        alphas = kwargs["alphas"]
        betas = kwargs["betas"]
        ps = kwargs["ps"]
        statistics = kwargs["statistics"]
        field_roles = kwargs["field_roles"]
        leg_roles = kwargs["leg_roles"]
        field_index_labels = kwargs["field_index_labels"]
        leg_index_labels = kwargs["leg_index_labels"]
        leg_spins = kwargs["leg_spins"]
        derivative_indices = kwargs["derivative_indices"]
        derivative_targets = kwargs["derivative_targets"]

    if alphas is None or betas is None or ps is None:
        raise ValueError("alphas, betas, ps are required")

    n = len(ps)
    field_index_labels, leg_index_labels = _merge_legacy_spinor_params(
        field_index_labels, leg_index_labels,
        field_spinor_indices, leg_spinor_indices,
        n,
    )

    if (
        strip_externals
        and leg_index_labels is None
        and statistics == "fermion"
        and _all_fermion_slots_labeled(field_roles, field_index_labels)
    ):
        leg_index_labels = _default_leg_index_labels(n)

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
        field_index_labels=field_index_labels,
        leg_index_labels=leg_index_labels,
        leg_spins=leg_spins,
        coupling=coupling,
    )
    full = contracted

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
        if leg_index_labels is None:
            spin_, si_ = S("spin_", "si_")
            full = full.replace(UF(beta_, p_, spin_, si_), 1)
            full = full.replace(UbarF(beta_, p_, spin_, si_), 1)
        q_, x_ = S("q_", "x_")
        full = full.replace(Expression.EXP(-I * Dot(q_, x_)), 1)

    return I * full


# ---------------------------------------------------------------------------
# Simplification helpers
# ---------------------------------------------------------------------------

def _species_key(x):
    return x.to_canonical_string() if hasattr(x, 'to_canonical_string') else str(x)


def simplify_deltas(expr, species_map=None):
    """Simplify species Kronecker deltas."""
    a_ = S("a_")

    if species_map is not None:
        for beta_sym, species_sym in species_map.items():
            expr = expr.replace(beta_sym, species_sym)
        expr = expr.replace(delta(a_, a_), Expression.num(1))

        known_species = sorted(
            set(species_map.values()),
            key=lambda s: _species_key(s),
        )
        for i in range(len(known_species)):
            for j in range(i + 1, len(known_species)):
                expr = expr.replace(
                    delta(known_species[i], known_species[j]),
                    Expression.num(0),
                )
    else:
        expr = expr.replace(delta(a_, a_), Expression.num(1))

    return expr


def simplify_spinor_indices(expr):
    """Contract repeated bispinor indices using Spenso's metric simplification."""
    return simplify_metrics(expr)


def simplify_vertex(expr, species_map=None):
    """Simplify a vertex factor expression in one call."""
    expr = simplify_deltas(expr, species_map=species_map)
    expr = simplify_spinor_indices(expr)
    return expr


# ---------------------------------------------------------------------------
# Compact notation helpers
# ---------------------------------------------------------------------------

def derivative_momentum_sum_expression(
    *,
    ps: Sequence,
    derivative_indices,
    derivative_targets=None,
    field_species: Optional[Sequence] = None,
    leg_species: Optional[Sequence] = None,
):
    """Build a compact momentum-sum expression for derivative patterns."""
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

    unique_targets = []
    for t in derivative_targets:
        if t < 0 or t >= n:
            raise ValueError(f"derivative target {t} out of range for {n} fields")
        if t not in unique_targets:
            unique_targets.append(t)

    k = len(unique_targets)
    target_slot = {t: s for s, t in enumerate(unique_targets)}
    total = Expression.num(0)

    for assigned_legs in permutations(range(n), k):
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

        monomial = Expression.num(1)
        for mu, t in zip(derivative_indices, derivative_targets):
            leg = assigned_legs[target_slot[t]]
            monomial *= pcomp(ps[leg], mu)

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
    """Compact sum-form vertex for derivative-target patterns."""
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
    """Human-readable sigma notation for derivative assignment patterns."""
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
