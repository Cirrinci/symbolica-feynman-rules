"""Canonical tensor-map helpers for the SMEFT comparison."""

from .base import *


def _is_dummy_weak_label(label: object) -> bool:
    return isinstance(label, str) and label.startswith("D:W:")


def _is_weak_t_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "t"
        and factor[1] == _WEAK_T_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_weak_eps2_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "weak_eps2"
        and factor[1] == _WEAK_EPS2_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 2
    )


def _is_generator_t_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "t"
        and factor[1] in _GENERATOR_T_FACTOR_SIGNATURES
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_structure_constant_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "f"
        and factor[1] in _ADJOINT_F_SIGNATURES
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _factor_labels(factor: object) -> tuple[object, ...]:
    if isinstance(factor, tuple) and len(factor) == 3 and isinstance(factor[2], tuple):
        return factor[2]
    return ()


def _label_occurrences(factors: Iterable[object], label: object) -> int:
    return sum(1 for factor in factors for item in _factor_labels(factor) if item == label)


def _fresh_dummy_adjoint_label(
    factors: Iterable[object],
    signature: tuple[str, str, str],
) -> str:
    prefix = _DUMMY_ADJOINT_PREFIX_BY_T_SIGNATURE[signature]
    used = {item for factor in factors for item in _factor_labels(factor)}
    candidate = 1
    while f"{prefix}{candidate}" in used:
        candidate += 1
    return f"{prefix}{candidate}"


def _generator_chain_from_pair(
    factors: list[object],
    first_index: int,
    second_index: int,
) -> tuple[
    tuple[str, str, str],
    object,
    object,
    object,
    object,
    object,
] | None:
    first = factors[first_index]
    second = factors[second_index]
    if not _is_generator_t_factor(first) or not _is_generator_t_factor(second):
        return None
    signature = first[1]
    if second[1] != signature:
        return None

    first_adj, first_left, first_right = first[2]
    second_adj, second_left, second_right = second[2]
    dummy_prefix = _DUMMY_FUND_PREFIX_BY_T_SIGNATURE[signature]
    candidates = []
    if (
        first_right == second_left
        and isinstance(first_right, str)
        and first_right.startswith(dummy_prefix)
    ):
        candidates.append(
            (signature, first_left, first_right, second_right, first_adj, second_adj)
        )
    if (
        second_right == first_left
        and isinstance(second_right, str)
        and second_right.startswith(dummy_prefix)
    ):
        candidates.append(
            (signature, second_left, second_right, first_right, second_adj, first_adj)
        )
    if len(candidates) != 1:
        return None
    signature, open_left, shared, open_right, left_adj, right_adj = candidates[0]
    if _label_occurrences(factors, shared) != 2:
        return None
    return signature, open_left, shared, open_right, left_adj, right_adj


def _canonical_key_with_commuting_factors(
    key: CanonicalTensorMonomial,
    factors: Iterable[object],
) -> CanonicalTensorMonomial:
    return CanonicalTensorMonomial(
        commuting_factors=tuple(sorted(factors, key=repr)),
        ordered_factors=key.ordered_factors,
    )


def _permutation_sign(current: tuple[object, ...], target: tuple[object, ...]) -> int:
    working = list(current)
    sign = 1
    for slot, desired in enumerate(target):
        current_slot = working.index(desired, slot)
        while current_slot > slot:
            working[current_slot], working[current_slot - 1] = (
                working[current_slot - 1],
                working[current_slot],
            )
            sign *= -1
            current_slot -= 1
    return sign


def _antisymmetric_factor_with_sorted_labels(
    head: str,
    signature: tuple[str, ...],
    labels: tuple[object, ...],
) -> tuple[int, object]:
    if len(set(labels)) != len(labels):
        return 0, None
    ordered = tuple(sorted(labels, key=repr))
    return _permutation_sign(labels, ordered), (head, signature, ordered)


def _expand_one_generator_product_ordering(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Sort one adjacent generator chain and emit the commutator term.

    FeynPy's covariant-derivative expansion may leave ``T^b T^a`` while
    FeynRules rewrites it as ``T^a T^b - i f^{abc} T^c``.  Applying that
    identity to both sides gives a strict common Lie-algebra basis.
    """

    factors = list(key.commuting_factors)
    for first_index, first in enumerate(factors):
        if not _is_generator_t_factor(first):
            continue
        for second_index in range(first_index + 1, len(factors)):
            chain = _generator_chain_from_pair(factors, first_index, second_index)
            if chain is None:
                continue
            signature, open_left, shared, open_right, left_adj, right_adj = chain
            if repr(left_adj) <= repr(right_adj):
                continue

            sorted_left_adj, sorted_right_adj = right_adj, left_adj
            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {first_index, second_index}
            ]
            product_factors = [
                *rest,
                ("t", signature, (sorted_left_adj, open_left, shared)),
                ("t", signature, (sorted_right_adj, shared, open_right)),
            ]
            dummy_adj = _fresh_dummy_adjoint_label(rest, signature)
            adjoint_signature = _ADJOINT_F_SIGNATURE_BY_T_SIGNATURE[signature]
            commutator_factors = [
                *rest,
                ("f", adjoint_signature, (dummy_adj, sorted_left_adj, sorted_right_adj)),
                ("t", signature, (dummy_adj, open_left, open_right)),
            ]
            return True, [
                (
                    Expression.num(1),
                    _canonical_key_with_commuting_factors(key, product_factors),
                ),
                (
                    -I,
                    _canonical_key_with_commuting_factors(key, commutator_factors),
                ),
            ]
    return False, [(Expression.num(1), key)]


def _normalize_generator_product_order_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(12):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = _expand_one_generator_product_ordering(term_key)
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("generator-product normalization did not converge")


def _normalize_generator_product_order_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_generator_product_order_key(key):
            normalized_coefficient = (
                coefficient * multiplier
            ).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _canonical_structure_constant_pair_factors(
    signature: tuple[str, ...],
    left_labels: tuple[object, object, object],
    right_labels: tuple[object, object, object],
) -> tuple[int, tuple[object, object]] | None:
    left_sign, left_factor = _antisymmetric_factor_with_sorted_labels(
        "f",
        signature,
        left_labels,
    )
    right_sign, right_factor = _antisymmetric_factor_with_sorted_labels(
        "f",
        signature,
        right_labels,
    )
    if left_sign == 0 or right_sign == 0:
        return None
    return left_sign * right_sign, (left_factor, right_factor)


def _structure_constant_jacobi_basis(
    signature: tuple[str, ...],
    free_labels: tuple[object, object, object, object],
    shared_label: object,
    basis_name: str,
) -> tuple[int, tuple[object, object]] | None:
    a_label, b_label, c_label, d_label = free_labels
    basis_labels = {
        "p12": ((a_label, b_label, shared_label), (c_label, d_label, shared_label)),
        "p14": ((a_label, d_label, shared_label), (b_label, c_label, shared_label)),
    }
    left_labels, right_labels = basis_labels[basis_name]
    return _canonical_structure_constant_pair_factors(
        signature,
        left_labels,
        right_labels,
    )


def _structure_constant_jacobi_pair(
    factors: list[object],
    left_index: int,
    right_index: int,
) -> tuple[tuple[str, ...], object, tuple[object, object, object, object], str, int] | None:
    left = factors[left_index]
    right = factors[right_index]
    if not _is_structure_constant_factor(left) or not _is_structure_constant_factor(
        right
    ):
        return None
    signature = left[1]
    if right[1] != signature:
        return None

    left_labels = left[2]
    right_labels = right[2]
    shared = tuple(label for label in left_labels if label in right_labels)
    if len(shared) != 1:
        return None
    shared_label = shared[0]
    dummy_prefix = _DUMMY_ADJOINT_PREFIX_BY_F_SIGNATURE[signature]
    if not (isinstance(shared_label, str) and shared_label.startswith(dummy_prefix)):
        return None
    if _label_occurrences(factors, shared_label) != 2:
        return None

    left_free = tuple(label for label in left_labels if label != shared_label)
    right_free = tuple(label for label in right_labels if label != shared_label)
    free_labels = tuple(sorted(left_free + right_free, key=repr))
    if len(set(free_labels)) != 4:
        return None

    a_label, b_label, c_label, d_label = free_labels
    targets = {
        "p12": ((a_label, b_label, shared_label), (c_label, d_label, shared_label)),
        "p13": ((a_label, c_label, shared_label), (b_label, d_label, shared_label)),
        "p14": ((a_label, d_label, shared_label), (b_label, c_label, shared_label)),
    }
    actual_pair_sets = [
        frozenset(left_free),
        frozenset(right_free),
    ]
    for name, (target_left, target_right) in targets.items():
        left_pair = frozenset(target_left[:2])
        right_pair = frozenset(target_right[:2])
        if actual_pair_sets == [left_pair, right_pair]:
            sign = _permutation_sign(left_labels, target_left) * _permutation_sign(
                right_labels,
                target_right,
            )
            return signature, shared_label, free_labels, name, sign
        if actual_pair_sets == [right_pair, left_pair]:
            sign = _permutation_sign(left_labels, target_right) * _permutation_sign(
                right_labels,
                target_left,
            )
            return signature, shared_label, free_labels, name, sign
    return None


def _expand_one_structure_constant_jacobi_pair(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Reduce one ``f*f`` product to the deterministic Jacobi basis.

    The only identity applied is

        ``f(a,b,e) f(c,d,e) - f(a,c,e) f(b,d,e)
        + f(a,d,e) f(b,c,e) = 0``.

    The middle pairing is replaced by the fixed ``p12`` and ``p14`` pairings.
    This is narrower than a general color simplification: all external labels
    and every non-``f`` factor remain untouched, and the shared adjoint label
    must be a genuine dummy occurring only in the selected ``f*f`` pair.
    """

    factors = list(key.commuting_factors)
    for left_index, left in enumerate(factors):
        if not _is_structure_constant_factor(left):
            continue
        for right_index in range(left_index + 1, len(factors)):
            pair = _structure_constant_jacobi_pair(factors, left_index, right_index)
            if pair is None:
                continue
            signature, shared_label, free_labels, kind, source_sign = pair
            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {left_index, right_index}
            ]

            target_basis = ("p12", "p14") if kind == "p13" else (kind,)
            expanded = []
            for basis_name in target_basis:
                target = _structure_constant_jacobi_basis(
                    signature,
                    free_labels,
                    shared_label,
                    basis_name,
                )
                if target is None:
                    continue
                target_sign, target_factors = target
                expanded.append(
                    (
                        Expression.num(source_sign * target_sign),
                        _canonical_key_with_commuting_factors(
                            key,
                            [*rest, *target_factors],
                        ),
                    )
                )
            if expanded:
                return kind == "p13", expanded
    return False, [(Expression.num(1), key)]


def _normalize_structure_constant_jacobi_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(8):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = _expand_one_structure_constant_jacobi_pair(
                term_key
            )
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("structure-constant Jacobi normalization did not converge")


def _normalize_structure_constant_jacobi_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_structure_constant_jacobi_key(
            key
        ):
            normalized_coefficient = (coefficient * multiplier).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _normalize_one_weak_t_epsilon_pair(
    factors: list[object],
) -> tuple[int, bool]:
    """Apply the SU(2) pseudoreality identity to one canonical-map pair.

    The identity is used only after the tensor canonicalizer has marked the
    contracted weak-fundamental index as dummy.  We keep the two generator
    endpoint orientations distinct, so this pass identifies
    ``t(D, i) eps(D, j)`` with ``t(D, j) eps(D, i)`` and
    ``t(i, D) eps(D, j)`` with ``t(j, D) eps(D, i)``, but it does not identify
    those two classes with each other.
    """

    for t_index, t_factor in enumerate(factors):
        if not _is_weak_t_factor(t_factor):
            continue
        t_adj, t_left, t_right = t_factor[2]
        t_fund = (t_left, t_right)
        for eps_index, eps_factor in enumerate(factors):
            if t_index == eps_index or not _is_weak_eps2_factor(eps_factor):
                continue
            eps_left, eps_right = eps_factor[2]
            eps_fund = (eps_left, eps_right)
            shared = {
                label
                for label in t_fund
                if label in eps_fund and _is_dummy_weak_label(label)
            }
            if len(shared) != 1:
                continue
            shared_label = next(iter(shared))
            t_positions = [
                position for position, label in enumerate(t_fund) if label == shared_label
            ]
            eps_positions = [
                position for position, label in enumerate(eps_fund) if label == shared_label
            ]
            if len(t_positions) != 1 or len(eps_positions) != 1:
                continue

            t_shared_position = t_positions[0]
            eps_shared_position = eps_positions[0]
            t_open = t_fund[1 - t_shared_position]
            eps_open = eps_fund[1 - eps_shared_position]
            composite_head = (
                "weak_t_eps_left_contract"
                if t_shared_position == 0
                else "weak_t_eps_right_contract"
            )
            sign = 1 if eps_shared_position == 0 else -1
            open_labels = tuple(sorted((t_open, eps_open), key=repr))
            composite_factor = (
                composite_head,
                _WEAK_T_FACTOR_SIGNATURE,
                (t_adj, *open_labels),
            )
            factors.pop(max(t_index, eps_index))
            factors.pop(min(t_index, eps_index))
            factors.append(composite_factor)
            factors.sort(key=repr)
            return sign, True
    return 1, False


def _normalize_weak_t_epsilon_key(
    key: CanonicalTensorMonomial,
) -> tuple[int, CanonicalTensorMonomial]:
    sign = 1
    factors = list(key.commuting_factors)
    while True:
        pair_sign, changed = _normalize_one_weak_t_epsilon_pair(factors)
        if not changed:
            break
        sign *= pair_sign
    return sign, CanonicalTensorMonomial(
        commuting_factors=tuple(factors),
        ordered_factors=key.ordered_factors,
    )


def _normalize_weak_t_epsilon_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        sign, normalized_key = _normalize_weak_t_epsilon_key(key)
        normalized_coefficient = (
            coefficient * Expression.num(sign)
        ).cancel().expand()
        normalized[normalized_key] = (
            normalized.get(normalized_key, Expression.num(0))
            + normalized_coefficient
        ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


_WEAK_T_EPS_COMPOSITE_HEADS = frozenset(
    {"weak_t_eps_left_contract", "weak_t_eps_right_contract"}
)


def _is_weak_t_eps_composite_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] in _WEAK_T_EPS_COMPOSITE_HEADS
        and factor[1] == _WEAK_T_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _weak_t_eps_pair_from_factors(
    factors: list[object],
    t_index: int,
    composite_index: int,
) -> tuple[
    str,
    object,
    object,
    object,
    object,
    object,
    int,
    int,
] | None:
    t_factor = factors[t_index]
    composite_factor = factors[composite_index]
    if not _is_weak_t_factor(t_factor) or not _is_weak_t_eps_composite_factor(
        composite_factor
    ):
        return None

    t_adj, t_left, t_right = t_factor[2]
    composite_adj, composite_left, composite_right = composite_factor[2]
    t_fund = (t_left, t_right)
    composite_fund = (composite_left, composite_right)
    shared = [
        label
        for label in t_fund
        if label in composite_fund and _is_dummy_weak_label(label)
    ]
    if len(shared) != 1:
        return None

    shared_label = shared[0]
    if _label_occurrences(factors, shared_label) != 2:
        return None

    t_shared_position = t_fund.index(shared_label)
    composite_shared_position = composite_fund.index(shared_label)
    t_open = t_fund[1 - t_shared_position]
    composite_open = composite_fund[1 - composite_shared_position]
    if not (
        isinstance(t_open, str)
        and isinstance(composite_open, str)
        and t_open.startswith("E:W:")
        and composite_open.startswith("E:W:")
    ):
        return None

    return (
        composite_factor[0],
        t_adj,
        composite_adj,
        t_open,
        composite_open,
        shared_label,
        t_shared_position,
        composite_shared_position,
    )


def _replace_weak_t_open_label(
    factor: object,
    *,
    shared_label: object,
    new_open: object,
) -> object:
    adjoint, left, right = factor[2]
    labels = [left, right]
    if labels[0] == shared_label:
        labels[1] = new_open
    elif labels[1] == shared_label:
        labels[0] = new_open
    else:  # pragma: no cover - guarded by the caller.
        raise ValueError("weak generator does not contain the shared label")
    return ("t", _WEAK_T_FACTOR_SIGNATURE, (adjoint, *labels))


def _replace_weak_t_eps_open_label(
    factor: object,
    *,
    shared_label: object,
    new_open: object,
) -> object:
    adjoint, left, right = factor[2]
    labels = [left, right]
    if labels[0] == shared_label:
        labels[1] = new_open
    elif labels[1] == shared_label:
        labels[0] = new_open
    else:  # pragma: no cover - guarded by the caller.
        raise ValueError("weak T-epsilon factor does not contain the shared label")
    return (factor[0], _WEAK_T_FACTOR_SIGNATURE, (adjoint, *labels))


def _expand_one_weak_doublet_generator_epsilon_pair(
    key: CanonicalTensorMonomial,
) -> tuple[bool, list[tuple[Expression, CanonicalTensorMonomial]]]:
    """Put SU(2) ``T*T*epsilon`` products in a single pseudoreal basis.

    After the one-generator ``T*epsilon`` contraction has been made explicit,
    the two-covariant-derivative Higgs-tilde rows can still differ by which
    external weak doublet index sits on the remaining generator.  For SU(2),

        ``P(j,i) = -P(i,j) + i f(a,b,c) C(c,i,j)``

    where ``P`` is the product of the remaining generator with the contracted
    ``T*epsilon`` composite and ``C`` is the same composite with the adjoint
    commutator index.  This is the Pauli-matrix pseudoreality relation for two
    generators acting on ``epsilon * Phi.bar``.  The pass only fires when the
    two factors share one dummy weak-fundamental index and the two open weak
    indices are external.
    """

    factors = list(key.commuting_factors)
    for t_index, t_factor in enumerate(factors):
        if not _is_weak_t_factor(t_factor):
            continue
        for composite_index, composite_factor in enumerate(factors):
            if t_index == composite_index:
                continue
            pair = _weak_t_eps_pair_from_factors(factors, t_index, composite_index)
            if pair is None:
                continue
            (
                composite_head,
                t_adj,
                composite_adj,
                t_open,
                composite_open,
                shared_label,
                _t_shared_position,
                _composite_shared_position,
            ) = pair
            if repr(t_open) <= repr(composite_open):
                continue

            rest = [
                factor
                for index, factor in enumerate(factors)
                if index not in {t_index, composite_index}
            ]
            swapped_product_factors = [
                *rest,
                _replace_weak_t_open_label(
                    t_factor,
                    shared_label=shared_label,
                    new_open=composite_open,
                ),
                _replace_weak_t_eps_open_label(
                    composite_factor,
                    shared_label=shared_label,
                    new_open=t_open,
                ),
            ]

            dummy_adj = _fresh_dummy_adjoint_label(rest, _WEAK_T_FACTOR_SIGNATURE)
            commutator_adjoint_labels = (
                (dummy_adj, composite_adj, t_adj)
                if composite_head == "weak_t_eps_left_contract"
                else (dummy_adj, t_adj, composite_adj)
            )
            f_sign, f_factor = _antisymmetric_factor_with_sorted_labels(
                "f",
                _ADJOINT_F_SIGNATURE_BY_T_SIGNATURE[_WEAK_T_FACTOR_SIGNATURE],
                commutator_adjoint_labels,
            )
            if f_sign == 0:
                continue
            composite_open_labels = tuple(sorted((composite_open, t_open), key=repr))
            commutator_factors = [
                *rest,
                f_factor,
                (
                    composite_head,
                    _WEAK_T_FACTOR_SIGNATURE,
                    (dummy_adj, *composite_open_labels),
                ),
            ]
            return True, [
                (
                    Expression.num(-1),
                    _canonical_key_with_commuting_factors(key, swapped_product_factors),
                ),
                (
                    I * Expression.num(f_sign),
                    _canonical_key_with_commuting_factors(key, commutator_factors),
                ),
            ]
    return False, [(Expression.num(1), key)]


def _normalize_weak_doublet_generator_epsilon_key(
    key: CanonicalTensorMonomial,
) -> list[tuple[Expression, CanonicalTensorMonomial]]:
    terms = [(Expression.num(1), key)]
    for _ in range(8):
        changed = False
        next_terms = []
        for coefficient, term_key in terms:
            term_changed, expansions = (
                _expand_one_weak_doublet_generator_epsilon_pair(term_key)
            )
            changed = changed or term_changed
            for factor, expanded_key in expansions:
                next_terms.append(((coefficient * factor).cancel().expand(), expanded_key))
        terms = next_terms
        if not changed:
            return terms
    raise RuntimeError("weak doublet generator-epsilon normalization did not converge")


def _normalize_weak_doublet_generator_epsilon_report(
    report: CanonicalMonomialReport,
) -> CanonicalMonomialReport:
    normalized: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in report.map.items():
        for multiplier, normalized_key in _normalize_weak_doublet_generator_epsilon_key(
            key
        ):
            normalized_coefficient = (coefficient * multiplier).cancel().expand()
            normalized[normalized_key] = (
                normalized.get(normalized_key, Expression.num(0))
                + normalized_coefficient
            ).cancel().expand()

    normalized = {
        key: coefficient
        for key, coefficient in sorted(normalized.items(), key=lambda item: repr(item[0]))
        if coefficient.cancel().expand().to_canonical_string() != "0"
    }
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=len(normalized),
        map=normalized,
    )


def _canonical_report_for_coefficient_head(
    expression: Expression,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
):
    report = canonical_tensor_monomial_report(
        _filter_terms_by_coefficient_head(expression, coefficient),
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    report = _normalize_generator_product_order_report(report)
    report = _normalize_weak_t_epsilon_report(report)
    report = _normalize_weak_doublet_generator_epsilon_report(report)
    return _normalize_structure_constant_jacobi_report(report)

__all__ = [name for name in globals() if not name.startswith("__")]
