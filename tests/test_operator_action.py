"""Tests for the ``lagrangian.operator_action`` and ``symbolica_export`` modules.

These tests live one layer above the model layer and below the vertex
engine: they build a few small ``InteractionTerm`` objects by hand, apply
``FieldOperator``s to them, and check the resulting term lists for the
right number of summands, the right slot ordering, and the right graded
Leibniz signs for fermionic / ghost-odd operators.

The Symbolica-export tests intentionally do **not** assert anything about
fermion ordering -- the export is documented to be commutative.
"""

from __future__ import annotations

import pytest

from symbolica import Expression, S

from model import (
    CompiledLagrangian,
    Field,
    GhostField,
    Lagrangian,
    LORENTZ_INDEX,
    Parameter,
    SPINOR_INDEX,
    dirac_field,
    scalar_field,
)
from model.interactions import DerivativeAction, FieldOccurrence, InteractionTerm
from lagrangian.operator_action import (
    FieldOperator,
    OperatorAtomResult,
    OperatorSummand,
    apply_field_operator,
    apply_field_operator_to_term,
    constant_result,
    replacement_operator,
    single_field_result,
    zero_result,
)
from lagrangian.symbolica_export import (
    PARTIAL_DERIVATIVE_HEAD,
    SymbolicaFieldRegistry,
    interaction_term_to_symbolica,
    interaction_terms_to_symbolica,
    lagrangian_to_symbolica,
)


def _canon(expr):
    if hasattr(expr, "expand"):
        return expr.expand().to_canonical_string()
    return str(expr)


# ---------------------------------------------------------------------------
# Minimal field fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def phi() -> Field:
    return scalar_field("phi", self_conjugate=True)


@pytest.fixture
def chi() -> Field:
    """Replacement scalar field used as the right-hand side of O[phi]."""

    return scalar_field("chi", self_conjugate=True)


@pytest.fixture
def psi() -> Field:
    return dirac_field("psi")


@pytest.fixture
def eta() -> Field:
    """Replacement Dirac field used by the BRST-like odd-operator tests."""

    return dirac_field("eta")


@pytest.fixture
def ghost() -> Field:
    return GhostField("c", spin=0, self_conjugate=False)


# ---------------------------------------------------------------------------
# Tests: bosonic operator on a simple scalar monomial
# ---------------------------------------------------------------------------


def _phi_cubed_term(phi: Field) -> InteractionTerm:
    """Build ``phi * phi * phi`` as an ``InteractionTerm`` by hand."""

    return InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence(), phi.occurrence()),
    )


def test_replacement_operator_on_scalar_cubed_produces_three_terms(phi, chi):
    """Leibniz on ``phi*phi*phi`` gives three terms with slot k = 0, 1, 2."""

    term = _phi_cubed_term(phi)
    operator = replacement_operator("O", {phi: chi.occurrence()})

    results = apply_field_operator_to_term(term, operator)

    assert len(results) == 3
    for slot_index, result in enumerate(results):
        names = tuple(occ.field.name for occ in result.fields)
        expected = ("phi",) * slot_index + ("chi",) + ("phi",) * (2 - slot_index)
        assert names == expected, f"slot {slot_index}: got {names}, expected {expected}"
        # Coupling has no sign change for a bosonic operator on a bosonic monomial.
        assert _canon(result.coupling) == _canon(Expression.num(1))


def test_bosonic_operator_keeps_derivative_on_replacement_slot(phi, chi):
    """Derivative on slot k is re-targeted to the first replacement slot.

    Source term: ``phi * phi * phi * (partial_mu phi)``.
    Operator: ``O[phi] = chi`` (even, commutes with the partial derivative).
    Expected: four output terms; the derivative ``partial_mu`` always
    tracks the slot that was acted on by ``O``.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence(), phi.occurrence(), phi.occurrence()),
        derivatives=(DerivativeAction(target=3, lorentz_index=mu),),
    )

    operator = replacement_operator("O", {phi: chi.occurrence()})
    results = apply_field_operator_to_term(term, operator)

    assert len(results) == 4
    for slot_index, result in enumerate(results):
        names = tuple(occ.field.name for occ in result.fields)
        expected = ("phi",) * slot_index + ("chi",) + ("phi",) * (3 - slot_index)
        assert names == expected
        # The derivative always sits on the original slot 3, which is
        # remapped under splicing of a single replacement slot to slot 3 still.
        assert result.derivatives == (DerivativeAction(target=3, lorentz_index=mu),)


def test_replacement_operator_returns_none_for_unmapped_field(phi, chi):
    """Fields not in the mapping are silently left alone."""

    rho = scalar_field("rho", self_conjugate=True)
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(rho.occurrence(), phi.occurrence(), rho.occurrence()),
    )
    operator = replacement_operator("O", {phi: chi.occurrence()})

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 1
    assert tuple(occ.field.name for occ in results[0].fields) == ("rho", "chi", "rho")


# ---------------------------------------------------------------------------
# Tests: fermion ordering and graded Leibniz signs
# ---------------------------------------------------------------------------


def _psibar_psi_phi_term(psi: Field, phi: Field) -> InteractionTerm:
    """Build ``psi.bar * psi * phi`` as an ``InteractionTerm``."""

    return InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True),
            psi.occurrence(),
            phi.occurrence(),
        ),
        closed_dirac_bilinears=((0, 1),),
    )


def test_even_operator_does_not_flip_signs_on_fermion_bilinear(psi, phi, chi):
    """An even operator picks up no signs even though it crosses fermions."""

    term = _psibar_psi_phi_term(psi, phi)
    operator = replacement_operator("O", {phi: chi.occurrence()}, parity=0)

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 1
    assert tuple(occ.field.name for occ in results[0].fields) == ("psi", "psi", "chi")
    assert _canon(results[0].coupling) == _canon(Expression.num(1))


def test_odd_operator_picks_up_sign_after_two_fermions(psi, eta):
    """Odd ``s`` on a ``psibar * psi`` chain: slot 0 contributes ``+`` and
    slot 1 contributes ``-`` because the operator has to step over the
    Grassmann-odd ``psibar`` to reach slot 1.

    Concretely we use a non-trivial replacement (``psi -> eta``) so that
    the two output terms can be distinguished by their field content.
    """

    term = _psibar_psi_phi_term(psi, scalar_field("phi", self_conjugate=True))
    operator = replacement_operator(
        "s",
        {psi: eta.occurrence()},
        parity=1,
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 2

    # First slot: psibar -> eta (sign = +1)
    first = results[0]
    assert tuple(occ.field.name for occ in first.fields) == ("eta", "psi", "phi")
    assert tuple(occ.conjugated for occ in first.fields) == (False, False, False)
    assert _canon(first.coupling) == _canon(Expression.num(1))

    # Second slot: psi -> eta (sign = -1, because |O| = 1 and one fermion to the left)
    second = results[1]
    assert tuple(occ.field.name for occ in second.fields) == ("psi", "eta", "phi")
    assert tuple(occ.conjugated for occ in second.fields) == (True, False, False)
    assert _canon(second.coupling) == _canon(-Expression.num(1))


def test_odd_operator_sign_alternates_across_pure_fermion_chain(psi, eta):
    """``s`` applied to a four-fermion chain ``psibar * psi * psibar * psi``
    gives signs ``(+, -, +, -)`` for slots ``(0, 1, 2, 3)``.
    """

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True),
            psi.occurrence(),
            psi.occurrence(conjugated=True),
            psi.occurrence(),
        ),
        closed_dirac_bilinears=((0, 1), (2, 3)),
    )
    operator = replacement_operator(
        "s",
        {psi: eta.occurrence()},
        parity=1,
    )

    results = apply_field_operator_to_term(term, operator)
    expected_signs = (1, -1, 1, -1)
    assert len(results) == 4
    for slot, (result, expected_sign) in enumerate(zip(results, expected_signs)):
        # Slot k of the original term becomes ``eta`` in the output; verify
        # both the field-name structure and the picked-up sign.
        names = tuple(occ.field.name for occ in result.fields)
        expected_names = ("psi",) * slot + ("eta",) + ("psi",) * (3 - slot)
        assert names == expected_names
        assert _canon(result.coupling) == _canon(Expression.num(expected_sign))


# ---------------------------------------------------------------------------
# Ghost / BRST-like odd parity test
# ---------------------------------------------------------------------------


def test_odd_operator_treats_ghost_factor_as_grassmann_odd(ghost):
    """A ghost field counts as Grassmann-odd in the Leibniz sign sum.

    Source term: ``cbar * c`` (an ordinary FP ghost bilinear).
    Odd operator: ``s[c] = c_prime`` (just a placeholder name swap).
    Expected: slot 0 contributes sign ``+``, slot 1 contributes sign ``-``
    because the operator stepped over one Grassmann-odd factor on its way.
    """

    c_prime = GhostField("c_prime", spin=0, self_conjugate=False)
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            ghost.occurrence(conjugated=True),
            ghost.occurrence(),
        ),
    )
    operator = replacement_operator(
        "s",
        {ghost: c_prime.occurrence()},
        parity=1,
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 2

    first, second = results
    assert tuple(occ.field.name for occ in first.fields) == ("c_prime", "c")
    assert tuple(occ.conjugated for occ in first.fields) == (False, False)
    assert _canon(first.coupling) == _canon(Expression.num(1))

    assert tuple(occ.field.name for occ in second.fields) == ("c", "c_prime")
    assert tuple(occ.conjugated for occ in second.fields) == (True, False)
    assert _canon(second.coupling) == _canon(-Expression.num(1))


# ---------------------------------------------------------------------------
# Coupling factor propagation
# ---------------------------------------------------------------------------


def test_summand_coefficient_multiplies_existing_coupling(phi, chi):
    """``OperatorSummand.coefficient`` multiplies into the term coupling."""

    g = S("g")
    term = InteractionTerm(coupling=g, fields=(phi.occurrence(),))
    custom_result = OperatorAtomResult(
        summands=(OperatorSummand(coefficient=S("alpha"), replacement=(chi.occurrence(),)),),
    )
    operator = FieldOperator(
        name="O",
        parity=0,
        on_field=lambda occ: custom_result if occ.field is phi else None,
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 1
    assert _canon(results[0].coupling) == _canon(S("alpha") * g)
    assert tuple(occ.field.name for occ in results[0].fields) == ("chi",)


def test_constant_result_drops_slot_and_multiplies_coupling(phi):
    """``constant_result`` is the "scalar replacement" path."""

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence()),
    )
    operator = FieldOperator(
        name="O",
        parity=0,
        on_field=lambda occ: constant_result(S("X")) if occ.field is phi else None,
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 2
    for result in results:
        assert tuple(occ.field.name for occ in result.fields) == ("phi",)
        assert _canon(result.coupling) == _canon(S("X"))


# ---------------------------------------------------------------------------
# Engine-level invariants
# ---------------------------------------------------------------------------


def test_apply_field_operator_lifts_to_a_sequence_of_terms(phi, chi):
    """``apply_field_operator`` is just term-wise concatenation."""

    term_a = InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),))
    term_b = InteractionTerm(coupling=Expression.num(2), fields=(phi.occurrence(), phi.occurrence()))
    operator = replacement_operator("O", {phi: chi.occurrence()})

    results = apply_field_operator((term_a, term_b), operator)
    assert len(results) == 1 + 2


def test_compiled_lagrangian_apply_operator_returns_new_object(phi, chi):
    """``CompiledLagrangian.apply_operator`` preserves the public API."""

    base = Lagrangian(terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),))
    operator = replacement_operator("O", {phi: chi.occurrence()})

    out = base.apply_operator(operator)
    assert isinstance(out, CompiledLagrangian)
    assert len(out.terms) == 1
    assert tuple(occ.field.name for occ in out.terms[0].fields) == ("chi",)


def test_zero_result_keeps_term_count_consistent(phi):
    """``zero_result`` contributes no summands but still consumes the slot."""

    operator = FieldOperator(
        name="annihilator",
        parity=0,
        on_field=lambda occ: zero_result() if occ.field is phi else None,
    )
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence()),
    )
    results = apply_field_operator_to_term(term, operator)
    assert results == ()


def test_empty_replacement_with_derivative_on_slot_raises(phi):
    """Empty-replacement summand on a slot that carries a derivative is an
    explicit error: there is no replacement slot for the derivative to ride
    along with, so the engine refuses rather than silently dropping it.
    """

    operator = FieldOperator(
        name="drop_slot",
        parity=0,
        # An explicit summand with an empty replacement -- this is the
        # "the slot disappears" path, not the "zero contribution" path.
        on_field=lambda occ: OperatorAtomResult(
            summands=(OperatorSummand(coefficient=Expression.num(1), replacement=()),)
        ),
    )
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(DerivativeAction(target=0, lorentz_index=S("mu")),),
    )
    with pytest.raises(ValueError, match="carries derivative actions"):
        apply_field_operator_to_term(term, operator)


# ---------------------------------------------------------------------------
# Symbolica export
# ---------------------------------------------------------------------------


def test_interaction_term_to_symbolica_uses_field_symbols_and_labels(phi):
    """The export wraps each occurrence as ``species(labels...)`` and
    multiplies by the coupling.

    Ordering is not preserved (Symbolica is commutative), so we check for
    presence and correctness of factors, not order.
    """

    g = S("g")
    term = InteractionTerm(
        coupling=g,
        fields=(phi.occurrence(), phi.occurrence()),
    )
    exported = interaction_term_to_symbolica(term)
    rendered = _canon(exported)

    assert "phi" in rendered
    assert "g" in rendered


def test_interaction_term_to_symbolica_wraps_derivatives(phi):
    """A derivative action becomes a ``PartialD(...)`` call in the export."""

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
    )
    exported = interaction_term_to_symbolica(term)
    rendered = _canon(exported)
    assert PARTIAL_DERIVATIVE_HEAD in rendered


def test_interaction_terms_to_symbolica_sums_term_expressions(phi, chi):
    term_a = InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),))
    term_b = InteractionTerm(coupling=Expression.num(2), fields=(chi.occurrence(),))
    total = interaction_terms_to_symbolica((term_a, term_b))
    rendered = _canon(total)
    assert "phi" in rendered
    assert "chi" in rendered


def test_lagrangian_to_symbolica_delegates_to_terms(phi):
    base = Lagrangian(terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),))
    direct = lagrangian_to_symbolica(base)
    via_method = base.to_symbolica()
    assert _canon(direct) == _canon(via_method)


def test_symbolica_field_registry_reverse_pass_is_not_implemented():
    """The reverse direction is intentionally guarded with a structured error."""

    registry = SymbolicaFieldRegistry()
    with pytest.raises(NotImplementedError, match="not implemented"):
        registry.from_symbolica(Expression.num(0))


def test_symbolica_export_does_not_distinguish_fermion_ordering(psi):
    """Documented limitation: Symbolica multiplication is commutative.

    ``psibar * psi`` and ``psi * psibar`` produce the same export, so the
    test asserts that limitation directly and serves as a regression
    against accidentally over-promising in the future.
    """

    a = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )
    b = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(), psi.occurrence(conjugated=True)),
        closed_dirac_bilinears=((1, 0),),
    )
    assert _canon(interaction_term_to_symbolica(a)) == _canon(interaction_term_to_symbolica(b))
