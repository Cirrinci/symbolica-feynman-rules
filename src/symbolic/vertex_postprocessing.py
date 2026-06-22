"""Post-contraction cleanup and output-policy helpers for vertex expressions."""

from __future__ import annotations

from symbolica import Expression, S
from symbolica.community.idenso import simplify_color, simplify_metrics

from feynpy.metadata import is_lorentz_index, lorentz_slots_for
from symbolic.spenso_structures import WEAK_ADJ, simplify_gamma_chain
from symbolic.tensor_canonicalization import (
    canonize_spenso_tensors,
    contract_spenso_lorentz_metrics,
)


delta = S("delta", is_symmetric=True)


def _species_key(x):
    return x.to_canonical_string() if hasattr(x, "to_canonical_string") else str(x)


def _get_label(index_labels_dict, kind, ordinal=0):
    if not index_labels_dict:
        return None
    val = index_labels_dict.get(kind)
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return val[ordinal] if 0 <= ordinal < len(val) else None
    return val


def replace_plane_wave_with_delta(
    expr,
    *,
    ps,
    x,
    d,
    plane_wave,
    delta_symbol,
    pi_symbol,
):
    if d is None:
        d = S("d")
    p_sum = Expression.num(0)
    for p in ps:
        p_sum += p
    return expr.replace(plane_wave(p_sum, x), (2 * pi_symbol) ** d * delta_symbol(p_sum))


def strip_external_wavefunctions(
    expr,
    *,
    leg_index_labels,
    u_symbol,
    uf_symbol,
    ubarf_symbol,
    dot_symbol,
    i_symbol,
):
    beta_, p_ = S("beta_", "p_")
    expr = expr.replace(u_symbol(beta_, p_), 1)
    if leg_index_labels is None:
        spin_, si_ = S("spin_", "si_")
        expr = expr.replace(uf_symbol(beta_, p_, spin_, si_), 1)
        expr = expr.replace(ubarf_symbol(beta_, p_, spin_, si_), 1)
    q_, x_ = S("q_", "x_")
    expr = expr.replace(Expression.EXP(-i_symbol * dot_symbol(q_, x_)), 1)
    return expr


def apply_vertex_output_policy(
    expr,
    *,
    ps,
    x,
    include_delta,
    strip_externals,
    leg_index_labels,
    d,
    plane_wave,
    delta_symbol,
    pi_symbol,
    u_symbol,
    uf_symbol,
    ubarf_symbol,
    dot_symbol,
    i_symbol,
):
    full = expr
    if include_delta:
        full = replace_plane_wave_with_delta(
            full,
            ps=ps,
            x=x,
            d=d,
            plane_wave=plane_wave,
            delta_symbol=delta_symbol,
            pi_symbol=pi_symbol,
        )
    if strip_externals:
        full = strip_external_wavefunctions(
            full,
            leg_index_labels=leg_index_labels,
            u_symbol=u_symbol,
            uf_symbol=uf_symbol,
            ubarf_symbol=ubarf_symbol,
            dot_symbol=dot_symbol,
            i_symbol=i_symbol,
        )
    return full


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


def simplify_color_indices(expr):
    """Apply idenso ``simplify_color`` followed by metric clean-up.

    This is a no-op on subexpressions that do not carry color tensors, so it
    is safe to insert into the generic vertex pipeline.
    """
    expr = simplify_color(expr)
    return simplify_metrics(expr)


def _atom_type_name(expr):
    return str(expr.get_type())


def _term_factors(expr):
    if _atom_type_name(expr) == "AtomType.Mul":
        return list(expr)
    return [expr]


def _term_product(factors):
    result = Expression.num(1)
    for factor in factors:
        result *= factor
    return result


def _is_su2_adjoint_slot(slot):
    return (
        _atom_type_name(slot) == "AtomType.Fn"
        and slot.get_name() == "spenso::coad"
        and len(slot) == 2
        and str(slot[0]) == "3"
    )


def _is_su2_structure_constant(factor):
    return (
        _atom_type_name(factor) == "AtomType.Fn"
        and factor.get_name() == "spenso::f"
        and len(factor) == 3
        and all(_is_su2_adjoint_slot(slot) for slot in factor)
    )


def _slot_label(slot):
    return slot[1]


def _weak_adj_metric(left, right):
    return WEAK_ADJ.g(left, right).to_expression()


def _normalize_su2_f_dummy_third(factor, dummy_key):
    labels = [_slot_label(slot) for slot in factor]
    position = [_species_key(label) for label in labels].index(dummy_key)
    sign = Expression.num(-1) if position == 1 else Expression.num(1)
    others = [label for label in labels if _species_key(label) != dummy_key]
    return sign, others[0], others[1]


def _rewrite_su2_ff_term(term):
    factors = _term_factors(term)
    structure_positions = [
        pos for pos, factor in enumerate(factors)
        if _is_su2_structure_constant(factor)
    ]
    for i, left_pos in enumerate(structure_positions):
        for right_pos in structure_positions[i + 1:]:
            left_factor = factors[left_pos]
            right_factor = factors[right_pos]
            left_labels = [_slot_label(slot) for slot in left_factor]
            right_labels = [_slot_label(slot) for slot in right_factor]
            left_map = {_species_key(label): label for label in left_labels}
            right_keys = {_species_key(label) for label in right_labels}
            shared = [key for key in left_map if key in right_keys]
            if len(shared) != 1:
                continue
            dummy_key = shared[0]
            left_sign, a, b = _normalize_su2_f_dummy_third(left_factor, dummy_key)
            right_sign, c, d = _normalize_su2_f_dummy_third(right_factor, dummy_key)
            replacement = left_sign * right_sign * (
                _weak_adj_metric(a, c) * _weak_adj_metric(b, d)
                - _weak_adj_metric(a, d) * _weak_adj_metric(b, c)
            )
            remaining = [
                factor
                for pos, factor in enumerate(factors)
                if pos not in (left_pos, right_pos)
            ]
            return _term_product(remaining) * replacement
    return term


def simplify_su2_ff(expr):
    """Rewrite SU(2) structure-constant products into adjoint Kronecker deltas.

    This is intentionally narrow: it looks for one product of two ``f`` tensors
    in a term, where the two factors share exactly one repeated adjoint index of
    the SU(2) adjoint representation ``coad(3, ...)``. Each ``f`` is first
    normalized so that the shared dummy index sits in the third slot, with the
    corresponding antisymmetry sign tracked explicitly, then the identity

        f(a,b,m) f(c,d,m) = delta(a,c) delta(b,d) - delta(a,d) delta(b,c)

    is applied.
    """

    expanded = expr.expand() if hasattr(expr, "expand") else expr
    terms = list(expanded) if _atom_type_name(expanded) == "AtomType.Add" else [expanded]
    result = Expression.num(0)
    for term in terms:
        result += _rewrite_su2_ff_term(term)
    return result.expand() if hasattr(result, "expand") else result


def _vector_leg_lorentz_labels(external_legs):
    labels = []
    for leg in external_legs:
        field = getattr(leg, "field", None)
        if field is None:
            return ()
        lorentz_slots = lorentz_slots_for(field)
        if len(lorentz_slots) != 1:
            return ()
        lorentz_slot = lorentz_slots[0]
        lorentz_kind = field.indices[lorentz_slot].kind
        lorentz_ordinal = sum(
            1
            for index in field.indices[:lorentz_slot]
            if index.kind == lorentz_kind
        )
        label = _get_label(
            getattr(leg, "labels", None),
            lorentz_kind,
            lorentz_ordinal,
        )
        if label is None:
            return ()
        labels.append(label)
    return tuple(labels)


def _vector_leg_internal_labels(external_legs):
    labels = []
    for leg in external_legs:
        label = None
        for index in getattr(leg.field, "indices", ()):
            if is_lorentz_index(index):
                continue
            label = _get_label(getattr(leg, "labels", None), index.kind)
            if label is not None:
                labels.append(label)
                break
        if label is None:
            return ()
    return tuple(labels)


def canonicalize_vector_vertex(expr, external_legs):
    if external_legs is None or len(external_legs) not in (3, 4):
        return expr
    if not all(
        getattr(getattr(leg, "field", None), "kind", None) == "vector"
        for leg in external_legs
    ):
        return expr

    lorentz_labels = _vector_leg_lorentz_labels(external_legs)
    if not lorentz_labels:
        return expr

    internal_labels = _vector_leg_internal_labels(external_legs)
    if not internal_labels:
        return expr

    canonical_expr, _, _ = canonize_spenso_tensors(
        expr,
        lorentz_indices=lorentz_labels,
        adjoint_indices=internal_labels,
    )
    return canonical_expr


def simplify_vertex(
    expr,
    species_map=None,
    external_legs=None,
    simplify_gamma: bool = False,
    simplify_color: bool = False,
):
    """Simplify a vertex factor expression in one call.

    Steps applied (in order):

    1. species-Kronecker-delta cleanup
    2. (optional) ``simplify_gamma`` for Dirac chains
    3. ``simplify_metrics`` (via ``simplify_spinor_indices``) to contract
       bispinor and other metric pairs
    4. (optional) ``simplify_color`` for SU(N) color identities -- this is
       opt-in because the idenso pass applies Fierz-like rewrites
       (e.g. ``T^a -> f`` substitutions) that change the surface form even
       when they preserve the value, which would invalidate downstream
       structural comparisons.
    5. project-local Lorentz-metric/momentum contraction pass
    6. vector-vertex canonicalisation when ``external_legs`` is provided

    For a stronger one-call cleanup that always runs the full idenso chain,
    use :func:`symbolic.tensor_canonicalization.canonize_full` instead.
    """
    expr = simplify_deltas(expr, species_map=species_map)
    if simplify_gamma:
        expr = simplify_gamma_chain(expr)
    expr = simplify_spinor_indices(expr)
    if simplify_color:
        expr = simplify_color_indices(expr)
    expr = contract_spenso_lorentz_metrics(expr)
    expr = canonicalize_vector_vertex(expr, external_legs)
    return expr
