"""Post-contraction cleanup and output-policy helpers for vertex expressions."""

from __future__ import annotations

from symbolica import Expression, S
from symbolica.community.idenso import simplify_metrics

from symbolic.spenso_structures import LORENTZ_KIND, simplify_gamma_chain
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


def _vector_leg_lorentz_labels(external_legs):
    labels = []
    for leg in external_legs:
        label = _get_label(getattr(leg, "labels", None), LORENTZ_KIND)
        if label is None:
            return ()
        labels.append(label)
    return tuple(labels)


def _vector_leg_internal_labels(external_legs):
    labels = []
    for leg in external_legs:
        label = None
        for index in getattr(leg.field, "indices", ()):
            if index.kind == LORENTZ_KIND:
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


def simplify_vertex(expr, species_map=None, external_legs=None, simplify_gamma: bool = False):
    """Simplify a vertex factor expression in one call."""
    expr = simplify_deltas(expr, species_map=species_map)
    if simplify_gamma:
        expr = simplify_gamma_chain(expr)
    expr = simplify_spinor_indices(expr)
    expr = contract_spenso_lorentz_metrics(expr)
    expr = canonicalize_vector_vertex(expr, external_legs)
    return expr
