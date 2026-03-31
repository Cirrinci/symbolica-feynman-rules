"""
Legacy parallel-list adapter for the Symbolica vertex engine.

This module preserves the original list-based API used by the older examples
while delegating the actual contraction logic to the metadata-first engine in
``model_symbolica.py``.
"""

from typing import Optional, Sequence

from model_symbolica import (
    S,
    Expression,
    I,
    U,
    UF,
    UbarF,
    delta,
    bis,
    mink,
    Delta,
    Dot,
    pcomp,
    D,
    plane_wave,
    infer_derivative_targets,
    simplify_deltas,
    simplify_spinor_indices,
    compact_vertex_sum_form,
    compact_sum_notation,
    _prepare_slot_side,
    _build_contraction_input,
    _contract_to_full_expression_core,
    _vertex_factor_core,
)


def contract_to_full_expression(
    *,
    alphas: Sequence,
    betas: Sequence,
    ps: Sequence,
    x,
    derivative_indices=(),
    derivative_targets: Optional[Sequence[int]] = None,
    statistics: str = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    field_slot_labels=None,
    field_index_slots=None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    leg_slot_labels=None,
    leg_index_slots=None,
    coupling=None,
):
    field_slot_state = _prepare_slot_side(
        base_slot_labels=field_slot_labels,
        explicit_index_slots=field_index_slots,
        spinor_indices=field_spinor_indices,
        roles=field_roles,
        expected_length=len(alphas),
        label_name="field_slot_labels",
    )
    leg_slot_state = _prepare_slot_side(
        base_slot_labels=leg_slot_labels,
        explicit_index_slots=leg_index_slots,
        spinor_indices=leg_spinor_indices,
        roles=leg_roles,
        expected_length=len(ps),
        label_name="leg_slot_labels",
    )
    contraction_input = _build_contraction_input(
        coupling=coupling if coupling is not None else Expression.num(1),
        statistics=statistics,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_slot_labels=field_slot_state.slot_labels,
        field_index_slots=field_slot_state.index_slots,
        leg_spins=leg_spins,
        leg_slot_labels=leg_slot_state.slot_labels,
        leg_index_slots=leg_slot_state.index_slots,
    )
    return _contract_to_full_expression_core(
        contraction_input=contraction_input,
        x=x,
    )


def vertex_factor(
    *,
    coupling,
    alphas: Sequence,
    betas: Sequence,
    ps: Sequence,
    x,
    derivative_indices=(),
    derivative_targets=None,
    statistics: str = "boson",
    field_roles=None,
    leg_roles=None,
    field_spinor_indices: Optional[Sequence] = None,
    field_slot_labels=None,
    field_index_slots=None,
    leg_spins: Optional[Sequence] = None,
    leg_spinor_indices: Optional[Sequence] = None,
    leg_slot_labels=None,
    leg_index_slots=None,
    strip_externals: bool = True,
    include_delta: bool = False,
    d=None,
):
    field_slot_state = _prepare_slot_side(
        base_slot_labels=field_slot_labels,
        explicit_index_slots=field_index_slots,
        spinor_indices=field_spinor_indices,
        roles=field_roles,
        expected_length=len(alphas),
        label_name="field_slot_labels",
    )
    leg_slot_state = _prepare_slot_side(
        base_slot_labels=leg_slot_labels,
        explicit_index_slots=leg_index_slots,
        spinor_indices=leg_spinor_indices,
        roles=leg_roles,
        expected_length=len(ps),
        label_name="leg_slot_labels",
    )
    contraction_input = _build_contraction_input(
        coupling=coupling,
        statistics=statistics,
        alphas=alphas,
        betas=betas,
        ps=ps,
        derivative_indices=derivative_indices,
        derivative_targets=derivative_targets,
        field_roles=field_roles,
        leg_roles=leg_roles,
        field_slot_labels=field_slot_state.slot_labels,
        field_index_slots=field_slot_state.index_slots,
        leg_spins=leg_spins,
        leg_slot_labels=leg_slot_state.slot_labels,
        leg_index_slots=leg_slot_state.index_slots,
    )
    return _vertex_factor_core(
        contraction_input=contraction_input,
        x=x,
        strip_externals=strip_externals,
        include_delta=include_delta,
        d=d,
    )
