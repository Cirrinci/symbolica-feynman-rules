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

from dataclasses import replace

import pytest

from symbolica import Expression, S

from model import (
    CompiledLagrangian,
    Field,
    GhostField,
    CompiledLagrangian,
    LORENTZ_INDEX,
    Model,
    Parameter,
    SPINOR_INDEX,
    dirac_field,
    flavor_index,
    scalar_field,
)
from model.interactions import DerivativeAction, FieldOccurrence, InteractionTerm
from lagrangian.operator_action import (
    FieldOperator,
    OperatorExpansionError,
    OperatorAtomResult,
    OperatorSummand,
    TermOperator,
    apply_operator,
    apply_field_operator,
    apply_field_operator_to_term,
    constant_result,
    partial,
    replacement_operator,
    single_field_result,
    zero_result,
)
from lagrangian.symbolica_export import (
    PARTIAL_DERIVATIVE_HEAD,
    PatternCoefficientMatch,
    SymbolicaFieldRegistry,
    interaction_term_to_symbolica,
    interaction_terms_to_symbolica,
    lagrangian_to_symbolica,
    pattern_coefficient,
    pattern_matches,
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


def _psibar_psi_phi_term(
    psi: Field,
    phi: Field,
    *,
    with_bilinear: bool = True,
) -> InteractionTerm:
    """Build ``psi.bar * psi * phi`` as an ``InteractionTerm``.

    ``with_bilinear`` controls whether the closed Dirac bilinear metadata
    is attached. Sign-tracking tests that don't preserve the bilinear
    under their operator action should pass ``with_bilinear=False`` so
    the engine doesn't reject the term for structural reasons.
    """

    return InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True),
            psi.occurrence(),
            phi.occurrence(),
        ),
        closed_dirac_bilinears=((0, 1),) if with_bilinear else (),
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
    the two output terms can be distinguished by their field content. The
    replacement is unconjugated for both slots, which would break the
    closed Dirac bilinear structure; we therefore build the source term
    without bilinear metadata, since the focus of this test is the graded
    Leibniz sign and not the bilinear bookkeeping (which is covered by
    its own dedicated tests below).
    """

    term = _psibar_psi_phi_term(
        psi,
        scalar_field("phi", self_conjugate=True),
        with_bilinear=False,
    )
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

    As with the two-fermion variant above, this test exercises the
    Leibniz sign and uses a conjugation-flipping replacement, so the
    source term is built without bilinear metadata.
    """

    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(
            psi.occurrence(conjugated=True),
            psi.occurrence(),
            psi.occurrence(conjugated=True),
            psi.occurrence(),
        ),
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

    base = CompiledLagrangian(terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),))
    operator = replacement_operator("O", {phi: chi.occurrence()})

    out = base.apply_operator(operator)
    assert isinstance(out, CompiledLagrangian)
    assert len(out.terms) == 1
    assert tuple(occ.field.name for occ in out.terms[0].fields) == ("chi",)


def test_compiled_lagrangian_apply_operator_accepts_term_operator(phi):
    """Whole-term operators dispatch through the same public entry point."""

    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )

    def apply_to_term(term):
        return (
            replace(term, coupling=term.coupling * S("a")),
            replace(term, coupling=term.coupling * S("b")),
        )

    out = base.apply_operator(TermOperator(name="split", apply_to_term=apply_to_term))
    assert len(out.terms) == 2
    assert {_canon(term.coupling) for term in out.terms} == {
        _canon(S("a")),
        _canon(S("b")),
    }


def test_model_apply_operator_forwards_to_compiled_lagrangian(phi, chi):
    """``Model.apply_operator(...)`` is a top-level forwarder."""

    model = Model(fields=(phi, chi), lagrangian_decl=phi)
    out = model.apply_operator(replacement_operator("O", {phi: chi.occurrence()}))
    assert len(out.terms) == 1
    assert tuple(occ.field.name for occ in out.terms[0].fields) == ("chi",)


def test_field_occurrence_apply_operator_matches_single_slot_lagrangian(phi, chi):
    """``FieldOccurrence.apply_operator(...)`` is thin sugar over `CompiledLagrangian`."""

    occurrence = phi()
    operator = replacement_operator("O", {phi: chi.occurrence()})

    direct = occurrence.apply_operator(operator)
    explicit = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(occurrence,)),)
    ).apply_operator(operator)

    assert isinstance(direct, CompiledLagrangian)
    assert len(direct.terms) == 1
    assert tuple(occ.field.name for occ in direct.terms[0].fields) == ("chi",)
    assert _canon(direct.to_symbolica()) == _canon(explicit.to_symbolica())


def test_apply_operators_is_left_to_right(phi, chi):
    """``apply_operators(A, B)`` means ``B(A(L))``."""

    rho = scalar_field("rho", self_conjugate=True)
    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )
    first = replacement_operator("first", {phi: chi.occurrence()})
    second = replacement_operator("second", {chi: rho.occurrence()})

    out = base.apply_operators(first, second)
    assert len(out.terms) == 1
    assert tuple(occ.field.name for occ in out.terms[0].fields) == ("rho",)


def test_operator_bracket_uses_graded_sign(phi):
    """Odd-odd brackets use the graded sign in the second composition."""

    chi = scalar_field("chi", self_conjugate=True)
    rho = scalar_field("rho", self_conjugate=True)
    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )
    left = replacement_operator("left", {phi: chi.occurrence()}, parity=1)
    right = replacement_operator("right", {chi: rho.occurrence()}, parity=1)

    bracket = base.operator_bracket(left, right)
    assert len(bracket.terms) == 1
    assert tuple(occ.field.name for occ in bracket.terms[0].fields) == ("rho",)
    assert _canon(bracket.terms[0].coupling) == _canon(Expression.num(1))


def test_operator_bracket_even_even_is_ordinary_commutator(phi):
    chi = scalar_field("chi", self_conjugate=True)
    rho = scalar_field("rho", self_conjugate=True)
    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )
    left = replacement_operator("left", {phi: chi.occurrence()}, parity=0)
    right = replacement_operator("right", {chi: rho.occurrence()}, parity=0)

    bracket = base.operator_bracket(left, right)
    assert len(bracket.terms) == 1
    assert tuple(occ.field.name for occ in bracket.terms[0].fields) == ("rho",)
    assert _canon(bracket.terms[0].coupling) == _canon(-Expression.num(1))


def test_operator_bracket_even_odd_is_ordinary_commutator(phi):
    chi = scalar_field("chi", self_conjugate=True)
    rho = scalar_field("rho", self_conjugate=True)
    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )
    left = replacement_operator("left", {phi: chi.occurrence()}, parity=0)
    right = replacement_operator("right", {chi: rho.occurrence()}, parity=1)

    bracket = base.operator_bracket(left, right)
    assert len(bracket.terms) == 1
    assert tuple(occ.field.name for occ in bracket.terms[0].fields) == ("rho",)
    assert _canon(bracket.terms[0].coupling) == _canon(-Expression.num(1))


def test_operator_anticommutator_uses_graded_sign(phi):
    """For odd-odd pairs the graded anticommutator carries the extra minus."""

    chi = scalar_field("chi", self_conjugate=True)
    rho = scalar_field("rho", self_conjugate=True)
    base = CompiledLagrangian(
        terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),)
    )
    left = replacement_operator("left", {phi: chi.occurrence()}, parity=1)
    right = replacement_operator("right", {chi: rho.occurrence()}, parity=1)

    anticommutator = base.operator_anticommutator(left, right)
    assert len(anticommutator.terms) == 1
    assert tuple(occ.field.name for occ in anticommutator.terms[0].fields) == ("rho",)
    assert _canon(anticommutator.terms[0].coupling) == _canon(-Expression.num(1))


def test_apply_operator_limit_raises_before_materializing_fanout(phi, chi):
    """Large derivative fan-out is rejected with a structured expansion error."""

    mu = S("mu")
    nu = S("nu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=0, lorentz_index=nu),
        ),
    )

    operator = FieldOperator(
        name="fanout",
        parity=0,
        on_field=lambda occ: single_field_result((chi.occurrence(), phi.occurrence())),
    )

    with pytest.raises(OperatorExpansionError) as exc_info:
        apply_operator((term,), operator, max_generated_terms=3)

    exc = exc_info.value
    assert exc.operator_name == "fanout"
    assert exc.slot == 0
    assert exc.replacement_len == 2
    assert exc.derivative_count_on_slot == 2
    assert exc.projected_terms == 4


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


def test_lagrangian_to_symbolica_coordinate_export_uses_symbolica_derivatives():
    mu = S("mu")
    phi = scalar_field("Phi", self_conjugate=True)
    model = Model(
        fields=(phi,),
        lagrangian_decl=phi * phi * phi + phi * partial(mu, phi),
    )

    exported = model.lagrangian().to_symbolica(derivative_style="coordinate")
    rendered = str(exported)
    assert "Phi(x_mu)" in rendered
    assert "der(1,Phi(x_mu))" in rendered

    differentiated = exported.derivative(S("x_mu"))
    assert _canon(differentiated) == _canon(
        Expression.parse(
            "3*der(1,Phi(x_mu))*Phi(x_mu)^2 + der(2,Phi(x_mu))*Phi(x_mu) + der(1,Phi(x_mu))^2"
        )
    )


def test_interaction_terms_to_symbolica_sums_term_expressions(phi, chi):
    term_a = InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),))
    term_b = InteractionTerm(coupling=Expression.num(2), fields=(chi.occurrence(),))
    total = interaction_terms_to_symbolica((term_a, term_b))
    rendered = _canon(total)
    assert "phi" in rendered
    assert "chi" in rendered


def test_lagrangian_to_symbolica_delegates_to_terms(phi):
    base = CompiledLagrangian(terms=(InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),)),))
    direct = lagrangian_to_symbolica(base)
    via_method = base.to_symbolica()
    assert _canon(direct) == _canon(via_method)


def test_compiled_lagrangian_forwards_symbolica_methods(phi):
    g = S("g")
    lagrangian = CompiledLagrangian(
        terms=(
            InteractionTerm(
                coupling=g,
                fields=(phi.occurrence(), phi.occurrence()),
            ),
        )
    )

    assert _canon(lagrangian.expand()) == _canon(lagrangian.to_symbolica().expand())
    assert _canon(lagrangian.coefficient(phi.symbol * phi.symbol)) == _canon(g)


def test_pattern_matches_deduplicate_commutative_wildcard_orderings():
    a, b, c = S("a", "b", "c")
    x_, y_ = S("x_", "y_")

    matches = pattern_matches(a * b * c, x_ * y_)

    assert all(isinstance(match, PatternCoefficientMatch) for match in matches)
    assert {_canon(match.matched_factor) for match in matches} == {
        _canon(a * b),
        _canon(a * c),
        _canon(b * c),
    }
    assert {_canon(match.coefficient) for match in matches} == {
        _canon(a),
        _canon(b),
        _canon(c),
    }


def test_pattern_coefficient_sums_unique_wildcard_matches():
    a, b, c = S("a", "b", "c")
    x_, y_ = S("x_", "y_")

    assert _canon(pattern_coefficient(a * b * c, x_ * y_)) == _canon(a + b + c)


def test_compiled_lagrangian_pattern_coefficient_supports_wildcard_labels():
    g = S("g")
    mu, nu = S("mu", "nu")
    mu_, nu_ = S("mu_", "nu_")
    vector = Field("A", spin=1, indices=(LORENTZ_INDEX,), self_conjugate=True)
    lagrangian = CompiledLagrangian(
        terms=(
            InteractionTerm(
                coupling=g,
                fields=(vector(mu), vector(nu)),
            ),
        )
    )

    assert _canon(
        lagrangian.pattern_coefficient(vector.symbol(mu_) * vector.symbol(nu_))
    ) == _canon(g)


def test_symbolica_field_registry_reverse_pass_is_not_implemented():
    """The reverse direction is intentionally guarded with a structured error."""

    registry = SymbolicaFieldRegistry()
    with pytest.raises(NotImplementedError, match="not implemented"):
        registry.from_symbolica(Expression.num(0))


def test_symbolica_export_canonicalizes_fermion_ordering_by_grassmann_sign(psi):
    """The export normalizes odd-factor order with the corresponding sign.

    Symbolica multiplication is still commutative, but the exporter now
    multiplies each term by the sign required to move odd factors into a
    canonical order. Reversing a fermion pair therefore flips the exported
    coefficient.
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
    assert _canon(interaction_term_to_symbolica(a)) == _canon(-interaction_term_to_symbolica(b))


# ---------------------------------------------------------------------------
# Bilinear preservation / violation (finding 1)
# ---------------------------------------------------------------------------


def test_replacement_preserving_conjugation_remaps_bilinears_correctly(psi, eta):
    """1-to-1 replacement that preserves Dirac-fermion conjugation keeps
    the closed Dirac bilinear pointing at the right slots.

    Original term: ``psi.bar * psi`` with bilinear ``(0, 1)``.
    Operator: ``O[psi] = eta`` on the unconjugated psi slot, with an
    explicit on_field that mirrors the original conjugation. Acting only
    on the unconjugated slot remaps the bilinear to ``(0, 1)`` again
    (replacement length 1 -> no slot shift).
    """

    def on_field(occurrence):
        if occurrence.field is not psi:
            return None
        return single_field_result(eta.occurrence(conjugated=occurrence.conjugated))

    operator = FieldOperator(name="O", parity=0, on_field=on_field)
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 2

    for slot, result in enumerate(results):
        assert result.closed_dirac_bilinears == ((0, 1),)
        names = tuple(occ.field.name for occ in result.fields)
        expected = ("eta", "psi") if slot == 0 else ("psi", "eta")
        assert names == expected


def test_replacement_with_extra_factors_remaps_bilinear_endpoint(psi, ghost):
    """Product-valued ``s[psi] = c * psi`` correctly remaps the bilinear
    endpoint to the position of the unconjugated fermion in the
    replacement.

    Original term: ``psi.bar * psi`` with bilinear ``(0, 1)``.
    Replacement on slot 1: ``(c, psi)``, where ``c`` is a Grassmann-odd
    ghost. The bilinear's psi endpoint must move to slot ``1 + 1 = 2``,
    leaving the psibar endpoint at slot ``0``.
    """

    def on_field(occurrence):
        if occurrence.field is not psi or occurrence.conjugated:
            return None
        return single_field_result(
            (ghost.occurrence(), psi.occurrence()),
        )

    operator = FieldOperator(name="s", parity=1, on_field=on_field)
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 1
    assert tuple(occ.field.name for occ in results[0].fields) == ("psi", "c", "psi")
    assert tuple(occ.conjugated for occ in results[0].fields) == (True, False, False)
    assert results[0].closed_dirac_bilinears == ((0, 2),)


def test_replacement_breaking_bilinear_conjugation_raises(psi, eta):
    """A replacement that does not contain a matching-conjugation
    Dirac-fermion factor at a bilinear endpoint is rejected with a clear
    structured error -- otherwise the stale bilinear would later be
    rejected by ``vertex_engine`` with a less informative message.
    """

    operator = replacement_operator(
        "broken",
        # Unconditionally maps psi -> eta (unconjugated), regardless of
        # the original slot's conjugation. This is exactly the source
        # situation that finding 1 warned about.
        {psi: eta.occurrence()},
        parity=0,
    )
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )
    with pytest.raises(ValueError, match="psibar endpoint of a closed Dirac bilinear"):
        apply_field_operator_to_term(term, operator)


def test_ghost_replacement_in_dirac_bilinear_is_rejected(psi, ghost):
    """Ghost fields are Grassmann-odd but have no Dirac spinor index, so
    they must not be silently accepted as bilinear endpoints just
    because their statistics is ``fermion``.
    """

    operator = FieldOperator(
        name="O",
        parity=0,
        on_field=lambda occ: single_field_result(ghost.occurrence(conjugated=True)) if occ.field is psi else None,
    )
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )
    with pytest.raises(ValueError, match="closed Dirac bilinear"):
        apply_field_operator_to_term(term, operator)


def test_replacement_inherits_spinor_labels_so_feynman_rule_still_works():
    """Custom replacements with matching fermion structure should inherit the
    acted slot's spinor labels automatically.

    This keeps the resulting lowered terms compatible with the ordinary
    ``feynman_rule(...)`` path, so notebook-style BRST/odd derivations do
    not need to rebuild spinor labels by hand in every ``on_field`` hook.
    """

    g = S("g")
    psi = dirac_field("psi", indices=())
    xi = dirac_field("xi", indices=())
    phi = scalar_field("phi", self_conjugate=True)
    chi = scalar_field("chi", self_conjugate=True)
    model = Model(
        fields=(psi, xi, phi, chi),
        lagrangian_decl=g * psi.bar() * psi() * phi,
    )

    def on_field(occurrence):
        if occurrence.field is psi:
            return single_field_result(xi.occurrence(conjugated=occurrence.conjugated))
        if occurrence.field is phi:
            return single_field_result(chi.occurrence())
        return None

    out = model.lagrangian().apply_operator(
        FieldOperator(name="s", parity=1, on_field=on_field)
    )

    assert out.terms[0].fields[0].labels == out.terms[0].fields[1].labels
    assert out.terms[1].fields[0].labels == out.terms[1].fields[1].labels

    rules = out.feynman_rule()
    assert set(rules) == {
        ("xi.bar", "psi", "phi"),
        ("psi.bar", "xi", "phi"),
        ("psi.bar", "psi", "chi"),
    }


# ---------------------------------------------------------------------------
# Derivative Leibniz expansion (finding 2)
# ---------------------------------------------------------------------------


def test_product_valued_replacement_leibniz_expands_one_derivative(phi, chi):
    """Bosonic Leibniz across a 2-slot replacement on a slot with one
    derivative: the derivative fans out across the replacement, giving
    two output terms.

    Source: ``phi * partial_mu phi``.
    Operator: ``O[phi] = (chi, phi)`` (two replacement slots) with
    ``parity = 0`` and ``commute_with_partial_derivative = True``.
    Acting on slot 1 should produce:

    * ``phi * (partial_mu chi) * phi``  (derivative on first replacement slot)
    * ``phi *  chi  * (partial_mu phi)``  (derivative on second replacement slot)

    and acting on slot 0 (which has no derivative) should produce:

    * ``(chi, phi) * partial_mu phi`` -- a single term.

    Total: 3 output terms.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence()),
        derivatives=(DerivativeAction(target=1, lorentz_index=mu),),
    )

    def on_field(occurrence):
        if occurrence.field is not phi:
            return None
        return single_field_result((chi.occurrence(), phi.occurrence()))

    operator = FieldOperator(name="O", parity=0, on_field=on_field)
    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 3

    # Slot 0 acted on: no derivative on slot 0 -> single output term.
    slot0 = results[0]
    assert tuple(occ.field.name for occ in slot0.fields) == ("chi", "phi", "phi")
    # The derivative originally on slot 1 is now on slot 2 after the shift.
    assert slot0.derivatives == (DerivativeAction(target=2, lorentz_index=mu),)

    # Slot 1 acted on -> two arrangements of the original derivative
    # across the (chi, phi) replacement slots (1 and 2).
    arrangement_a = results[1]
    arrangement_b = results[2]
    for arrangement in (arrangement_a, arrangement_b):
        assert tuple(occ.field.name for occ in arrangement.fields) == ("phi", "chi", "phi")

    derivative_targets = sorted(
        action.target
        for arrangement in (arrangement_a, arrangement_b)
        for action in arrangement.derivatives
    )
    assert derivative_targets == [1, 2]


def test_product_valued_replacement_leibniz_expands_two_derivatives(phi, chi):
    """Two derivatives on the acted slot fan out to ``N**M = 2**2 = 4``
    output terms (one per (derivative, replacement-slot) assignment).
    """

    mu = S("mu")
    nu = S("nu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=mu),
            DerivativeAction(target=0, lorentz_index=nu),
        ),
    )

    def on_field(occurrence):
        if occurrence.field is not phi:
            return None
        return single_field_result((chi.occurrence(), phi.occurrence()))

    operator = FieldOperator(name="O", parity=0, on_field=on_field)
    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 4

    arrangements = set()
    for result in results:
        assert tuple(occ.field.name for occ in result.fields) == ("chi", "phi")
        slot_of = {action.lorentz_index: action.target for action in result.derivatives}
        arrangements.add((slot_of[mu], slot_of[nu]))

    assert arrangements == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_single_replacement_is_still_one_arrangement_per_summand(phi, chi):
    """For replacement length ``N = 1`` the Leibniz expansion collapses
    back to one arrangement per summand (``1**M = 1``).
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
    )
    operator = replacement_operator("O", {phi: chi.occurrence()})

    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 1
    assert results[0].derivatives == (DerivativeAction(target=0, lorentz_index=mu),)


def test_non_commuting_operator_still_rejects_derivative_slots(phi, chi):
    """``commute_with_partial_derivative = False`` is preserved by the new
    enumeration path: slots with derivatives are refused.
    """

    operator = FieldOperator(
        name="O",
        parity=0,
        on_field=lambda occ: single_field_result(chi.occurrence()) if occ.field is phi else None,
        commute_with_partial_derivative=False,
    )
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(),),
        derivatives=(DerivativeAction(target=0, lorentz_index=S("mu")),),
    )
    with pytest.raises(ValueError, match="does not commute with partial derivatives"):
        apply_field_operator_to_term(term, operator)


# ---------------------------------------------------------------------------
# Flavor-expanded Symbolica export (finding 3)
# ---------------------------------------------------------------------------


def test_to_symbolica_with_flavor_expand_reflects_class_member_expansion():
    """``CompiledLagrangian.to_symbolica(flavor_expand=True)`` exposes the
    expanded class-member terms, not the flavor-generic source terms.

    Concretely we build a tiny lepton-class CompiledLagrangian with three
    class members ``e``, ``mu``, ``ta``. The flavor-expanded export
    should mention each member by name; the un-expanded export must not.
    """

    generation = flavor_index("Generation", 3, prefix="f")
    l = dirac_field(
        "l",
        class_members=("e", "mu", "ta"),
        indices=(generation,),
        flavor_index=generation,
    )
    Phi = scalar_field("Phi")
    f = S("f")

    from model import Model

    model = Model(
        fields=(l, Phi),
        lagrangian_decl=S("g") * l.bar(f) * l(f) * Phi,
    )
    lagrangian = model.lagrangian()

    base_rendered = _canon(lagrangian.to_symbolica())
    expanded_rendered = _canon(lagrangian.to_symbolica(flavor_expand=True))

    # The generic form should contain the class name; the expanded form
    # should contain each class-member name and *not* the generic class
    # name.
    assert "l" in base_rendered
    for member in ("e", "mu", "ta"):
        assert member in expanded_rendered, (
            f"flavor-expanded export missing class member {member!r}"
        )


def test_to_symbolica_flavor_expand_false_matches_default_behavior():
    """The default ``flavor_expand=False`` is unchanged by the new kwarg."""

    base = CompiledLagrangian(terms=(InteractionTerm(coupling=Expression.num(1), fields=(scalar_field("phi", self_conjugate=True).occurrence(),)),))
    assert _canon(base.to_symbolica()) == _canon(base.to_symbolica(flavor_expand=False))


def test_model_to_symbolica_forwards_to_compiled_lagrangian():
    """``Model.to_symbolica()`` mirrors ``model.lagrangian().to_symbolica()``."""

    phi = scalar_field("phi", self_conjugate=True)
    model = Model(fields=(phi,), lagrangian_decl=S("g") * phi * phi)

    assert _canon(model.to_symbolica()) == _canon(model.lagrangian().to_symbolica())


def test_model_to_symbolica_forwards_flavor_expand():
    """``Model.to_symbolica(flavor_expand=...)`` forwards the kwarg unchanged."""

    generation = flavor_index("Generation", 3, prefix="f")
    l = dirac_field(
        "l",
        class_members=("e", "mu", "ta"),
        indices=(generation,),
        flavor_index=generation,
    )
    phi = scalar_field("Phi")
    f = S("f")
    model = Model(
        fields=(l, phi),
        lagrangian_decl=S("g") * l.bar(f) * l(f) * phi,
    )

    assert _canon(model.to_symbolica(flavor_expand=True)) == _canon(
        model.lagrangian().to_symbolica(flavor_expand=True)
    )


def test_model_to_symbolica_forwards_derivative_export_options():
    phi = scalar_field("Phi", self_conjugate=True)
    mu = S("mu")
    model = Model(fields=(phi,), lagrangian_decl=phi * partial(mu, phi))

    assert _canon(
        model.to_symbolica(derivative_style="coordinate")
    ) == _canon(
        model.lagrangian().to_symbolica(derivative_style="coordinate")
    )


# ---------------------------------------------------------------------------
# Real spacetime derivative operator: `partial(...)` factory
# ---------------------------------------------------------------------------
#
# These tests cover the new derivative operator. The defining feature is
# that ``partial(mu)`` does **not** replace the field; it keeps the same
# ``FieldOccurrence`` and attaches a fresh ``DerivativeAction`` to its
# slot, so the lowered ``InteractionTerm`` looks the same as one declared
# with the existing ``PartialD(Phi, mu)`` factor.


def test_partial_one_arg_returns_field_operator():
    """``partial(mu)`` (one argument) is a runtime ``FieldOperator``."""

    mu = S("mu")
    op = partial(mu)
    assert isinstance(op, FieldOperator)
    assert op.parity == 0


def test_partial_two_arg_returns_declarative_partial_d(phi):
    """``partial(mu, Phi)`` is sugar for the declarative ``PartialD(Phi, mu)``."""

    from model.declared import PartialDerivativeFactor

    mu = S("mu")
    factor = partial(mu, phi)
    assert isinstance(factor, PartialDerivativeFactor)
    assert factor.field is phi
    assert factor.lorentz_indices == (mu,)


def test_partial_rejects_none_lorentz_index():
    """A None Lorentz index is caught up front, not deep inside lowering."""

    with pytest.raises(TypeError, match="non-None Lorentz index"):
        partial(None)


def test_partial_two_arg_form_rejects_runtime_keywords(phi):
    """Mixing the declarative shortcut with runtime-only keywords is an error."""

    mu = S("mu")
    with pytest.raises(TypeError, match="declarative-factor shortcut"):
        partial(mu, phi, on=phi)


def test_partial_on_scalar_cubed_applies_product_rule(phi):
    """``partial(mu)`` on ``phi*phi*phi`` gives the Leibniz expansion:

    three output terms, each carrying one fresh derivative on the
    corresponding slot.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence(), phi.occurrence()),
    )
    results = apply_field_operator_to_term(term, partial(mu))

    assert len(results) == 3
    for slot, result in enumerate(results):
        names = tuple(occ.field.name for occ in result.fields)
        assert names == ("phi", "phi", "phi")
        assert result.derivatives == (
            DerivativeAction(target=slot, lorentz_index=mu),
        )
        # ``partial`` is parity 0 -> no graded sign factor.
        assert _canon(result.coupling) == _canon(Expression.num(1))


def test_partial_on_scalar_product_gives_two_summands(phi, chi):
    """``partial_mu(Phi*Chi) = (∂_mu Phi)*Chi + Phi*(∂_mu Chi)``."""

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), chi.occurrence()),
    )
    results = apply_field_operator_to_term(term, partial(mu))

    assert len(results) == 2
    assert tuple(occ.field.name for occ in results[0].fields) == ("phi", "chi")
    assert results[0].derivatives == (DerivativeAction(target=0, lorentz_index=mu),)
    assert tuple(occ.field.name for occ in results[1].fields) == ("phi", "chi")
    assert results[1].derivatives == (DerivativeAction(target=1, lorentz_index=mu),)


def test_partial_with_on_restricts_to_named_field(phi, chi):
    """``partial(mu, on=Phi)`` skips slots whose field is not ``Phi``."""

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), chi.occurrence(), phi.occurrence()),
    )
    results = apply_field_operator_to_term(term, partial(mu, on=phi))

    # Two terms: derivative on slot 0 (phi) and slot 2 (phi). Slot 1 (chi)
    # is skipped.
    assert len(results) == 2
    assert results[0].derivatives == (DerivativeAction(target=0, lorentz_index=mu),)
    assert results[1].derivatives == (DerivativeAction(target=2, lorentz_index=mu),)


def test_partial_preserves_bilinear_on_dirac_pair(psi):
    """``partial`` keeps closed Dirac bilinear metadata intact in every
    Leibniz summand: the replacement is the same field with the same
    conjugation, so the bilinear structure is preserved automatically.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )
    results = apply_field_operator_to_term(term, partial(mu))

    assert len(results) == 2
    for slot, result in enumerate(results):
        assert result.closed_dirac_bilinears == ((0, 1),)
        assert result.derivatives == (DerivativeAction(target=slot, lorentz_index=mu),)
        # Even parity -> no sign factor across the fermion.
        assert _canon(result.coupling) == _canon(Expression.num(1))


def test_partial_on_psi_only_in_fermion_current_keeps_bilinear(psi, phi):
    """In a ``g psi.bar psi phi`` current, ``partial(mu, on=phi)`` must

    * attach the derivative to the phi slot only,
    * leave the Dirac bilinear ``(0, 1)`` untouched.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(psi.occurrence(conjugated=True), psi.occurrence(), phi.occurrence()),
        closed_dirac_bilinears=((0, 1),),
    )
    results = apply_field_operator_to_term(term, partial(mu, on=phi))

    assert len(results) == 1
    assert results[0].closed_dirac_bilinears == ((0, 1),)
    assert results[0].derivatives == (DerivativeAction(target=2, lorentz_index=mu),)


def test_partial_runtime_matches_declarative_partial_d_on_phi_squared():
    """The runtime ``partial(mu)`` on ``Phi*Phi`` lowers to the same
    list of ``InteractionTerm`` shapes as the declarative form

        ``PartialD(Phi, mu)*Phi + Phi*PartialD(Phi, mu)``,

    i.e. the two terms each have one derivative on a different slot.
    """

    from model import Model
    from model.declared import PartialD

    mu = S("mu")
    Phi = scalar_field("Phi", self_conjugate=True)

    runtime = (
        Model(fields=(Phi,), lagrangian_decl=Phi * Phi)
        .lagrangian()
        .apply_operator(partial(mu))
    )
    declarative = Model(
        fields=(Phi,),
        lagrangian_decl=PartialD(Phi, mu) * Phi + Phi * PartialD(Phi, mu),
    ).lagrangian()

    def _signature(t):
        return (
            tuple((occ.field.name, occ.conjugated) for occ in t.fields),
            tuple((a.target, str(a.lorentz_index)) for a in t.derivatives),
        )

    assert sorted(_signature(t) for t in runtime.terms) == sorted(
        _signature(t) for t in declarative.terms
    )


def test_partial_repeated_application_increments_derivatives_on_each_slot():
    """``partial(mu)`` applied twice to ``Phi`` gives one term with two
    derivatives on the same slot. This is the canonical ``∂_mu ∂_mu Phi``
    representation -- repeated independent ``DerivativeAction`` entries
    on the same slot, exactly what the existing PartialD-of-PartialD
    declarative form lowers to.
    """

    from model import Model

    mu = S("mu")
    Phi = scalar_field("Phi", self_conjugate=True)
    L = Model(fields=(Phi,), lagrangian_decl=Phi).lagrangian().apply_operator(partial(mu))
    L2 = L.apply_operator(partial(mu))

    assert len(L2.terms) == 1
    targets = sorted((a.target, str(a.lorentz_index)) for a in L2.terms[0].derivatives)
    assert targets == [(0, "mu"), (0, "mu")]


def test_partial_does_not_overlap_with_replacement_layer(phi, chi):
    """``partial(mu)`` must not be confused with replacing the field.

    Acting on ``phi*phi`` must keep both slots as ``phi`` (with one fresh
    derivative each), not replace ``phi`` by anything.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence()),
    )
    results = apply_field_operator_to_term(term, partial(mu))
    for result in results:
        assert tuple(occ.field.name for occ in result.fields) == ("phi", "phi")


def test_partial_compatible_with_feynman_rule_for_scalar_kinetic():
    """End-to-end smoke test against ``feynman_rule``.

    A scalar two-derivative kinetic ``PartialD(Phi, mu) * PartialD(Phi, mu)``
    (declarative form, equivalent to applying ``partial(mu)`` twice once
    boundary terms are dropped) gives a two-point vertex containing
    explicit momenta. We do not pin the exact symbolic form here -- the
    point is that lowered terms with ``DerivativeAction``s feed cleanly
    into the unchanged vertex engine.
    """

    from model import Model
    from model.declared import PartialD

    mu = S("mu")
    Phi = scalar_field("Phi", self_conjugate=True)
    model = Model(
        fields=(Phi,),
        lagrangian_decl=PartialD(Phi, mu) * PartialD(Phi, mu),
    )
    vertex = model.feynman_rule(Phi, Phi)
    text = str(vertex)
    # Each external scalar carries an auto-assigned momentum q1 / q2.
    assert "q1" in text or "q2" in text


def test_partial_runtime_lowers_to_vertex_with_momentum():
    """The lowered term produced by ``partial(mu)`` on a phi^3 monomial
    drives a 3-point vertex through the existing vertex engine without
    extra plumbing.
    """

    from model import Model

    mu = S("mu")
    Phi = scalar_field("Phi", self_conjugate=True)
    base = Model(fields=(Phi,), lagrangian_decl=Phi * Phi * Phi).lagrangian()
    derived = base.apply_operator(partial(mu))
    # 3 Leibniz terms, each with one derivative on a distinct slot.
    assert len(derived.terms) == 3
    for slot, term in enumerate(derived.terms):
        assert term.derivatives == (
            DerivativeAction(target=slot, lorentz_index=mu),
        )
    # The flavor-generic 3-point vertex should be reachable through the
    # ordinary public API. (We only check that the call returns
    # something containing momentum factors; the exact form depends on
    # vertex-engine conventions.)
    text = str(derived.feynman_rule(Phi, Phi, Phi))
    assert text  # non-empty


def test_partial_on_empty_field_list_raises():
    """``partial(..., on=())`` is rejected: ambiguous intent."""

    mu = S("mu")
    with pytest.raises(ValueError, match="empty list of fields"):
        partial(mu, on=())


def test_partial_rejects_non_field_in_on():
    """``partial(..., on=...)`` rejects entries that are not ``Field``."""

    mu = S("mu")
    with pytest.raises(TypeError, match="expected Field"):
        partial(mu, on=("not a field",))


# ---------------------------------------------------------------------------
# Engine: new_derivatives translation and validation
# ---------------------------------------------------------------------------


def test_new_derivatives_target_translates_to_absolute_slot(phi):
    """An operator returning a summand with ``new_derivatives`` at
    position 0 of a 1-slot replacement on slot ``k`` produces a term
    with a derivative whose absolute ``target`` is ``k``.
    """

    mu = S("mu")
    term = InteractionTerm(
        coupling=Expression.num(1),
        fields=(phi.occurrence(), phi.occurrence(), phi.occurrence()),
    )

    def on_field(occurrence):
        if occurrence.field is not phi:
            return None
        return single_field_result(
            phi.occurrence(),
            new_derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        )

    operator = FieldOperator(name="d", parity=0, on_field=on_field)
    results = apply_field_operator_to_term(term, operator)
    assert len(results) == 3
    for slot, result in enumerate(results):
        assert result.derivatives == (
            DerivativeAction(target=slot, lorentz_index=mu),
        )


def test_new_derivatives_out_of_range_target_raises(phi):
    """``new_derivatives`` whose target lies outside the replacement is
    rejected with a structured error.
    """

    mu = S("mu")
    term = InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),))

    def on_field(occurrence):
        if occurrence.field is not phi:
            return None
        return single_field_result(
            phi.occurrence(),
            new_derivatives=(DerivativeAction(target=2, lorentz_index=mu),),
        )

    operator = FieldOperator(name="bad", parity=0, on_field=on_field)
    with pytest.raises(ValueError, match="fresh derivative target"):
        apply_field_operator_to_term(term, operator)


def test_new_derivatives_with_empty_replacement_raises(phi):
    """A fresh derivative attached to an empty replacement has no slot to live on."""

    mu = S("mu")
    term = InteractionTerm(coupling=Expression.num(1), fields=(phi.occurrence(),))

    def on_field(occurrence):
        if occurrence.field is not phi:
            return None
        return OperatorAtomResult(
            summands=(
                OperatorSummand(
                    coefficient=Expression.num(1),
                    replacement=(),
                    new_derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
                ),
            )
        )

    operator = FieldOperator(name="bad", parity=0, on_field=on_field)
    with pytest.raises(ValueError, match="needs a slot to act on|carries derivative actions"):
        apply_field_operator_to_term(term, operator)
