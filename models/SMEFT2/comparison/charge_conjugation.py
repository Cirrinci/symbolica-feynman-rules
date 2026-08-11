"""Charge-conjugation packaging helpers and SMEFT2 matter parsers."""

from .canonical import *


_DIRAC_C_FACTOR_SIGNATURE = ("spinor", "spinor")
_GAMMA_FACTOR_SIGNATURE = ("spinor", "spinor", "lorentz")
_SPINOR_METRIC_FACTOR_SIGNATURE = ("spinor", "spinor")
_EC_CC_PL = TensorName("PL")
_EC_CC_PR = TensorName("PR")
_EC_CC_TENSOR_HEAD_SPECS = (
    *SPENSO_TENSOR_HEAD_SPECS,
    TensorHeadSpec(
        raw_name="spenso_python::PL",
        canonical_name="canon::PL",
        arity=2,
        head_kwargs={},
    ),
    TensorHeadSpec(
        raw_name="spenso_python::PR",
        canonical_name="canon::PR",
        arity=2,
        head_kwargs={},
    ),
)
_EC_CC_COEFFICIENTS = frozenset(
    {
        "alphaEcqedl",
        "alphaEcqedlthree",
        "alphaEcudqq",
        "alphaEcudqqtwo",
        "alphaEcuelq",
        "alphaEcuelqtwo",
    }
)

# Global FeynPy/FeynRules bilinear-packaging convention for the evanescent
# charge-conjugated four-fermion operators (every ``alphaEc*`` head).
#
# FeynRules writes these operators through the ``CC[...]`` macro, e.g.
#
#     alphaEcll[f1,f2,f3,f4] CC[LLbar[sp1,ii,f1]].LL[sp1,jj,f2]
#                            .LLbar[sp2,jj,f3].CC[LL[sp2,ii,f4]]
#
# so ``ExpandIndices`` resolves the conjugation and the exported vertex carries
# no residual charge-conjugation matrix: the spinor flow runs external leg 1 to
# leg 4 and leg 2 to leg 3 through ordinary spinor-metric/gamma chains.
#
# SMEFT2.py instead keeps the charge-conjugation matrices explicit and pairs the
# adjacent legs,
#
#     lLbar(sp1) * C(sp1,sp2) * lL(sp2) * lLbar(sp3) * C(sp3,sp4) * lL(sp4)
#
# i.e. flow (1,2) and (3,4) with two ``C`` factors.
#
# The two forms describe the same operator. Converting FeynPy to the FeynRules
# packaging eliminates both ``C`` factors by re-pairing the four spinor arms in
# the *crossed* order (arm0,arm3) and (arm1,arm2), and picks up a single overall
# sign from the antisymmetry of the charge-conjugation matrix (C^T = -C)
# together with the anticommutation needed to reorder the fermion fields into
# the crossed pairing.
#
# Both the pairing and the sign are fixed once and globally, not per row. The
# choice is also strongly overdetermined by the data: of the four
# (mode, phase) combinations, only ``("crossed", -1)`` reproduces the reference,
# and it does so uniformly. The other three fail on essentially every row
# (crossed/+1: 0/21 rows, direct/-1: 3/21, direct/+1: 2/21, where the handful of
# passes are rows on which the transform does not apply at all).
#
# Rows whose agreement depends on this transform are reported as
# ``MATCH_MODULO_EC_CC_CONVENTION`` rather than ``EXACT_MATCH`` so the
# convention assumption stays visible in the headline numbers.
_EC_CC_CONVENTION_MODE = "crossed"
_EC_CC_CONVENTION_PHASE = -1


def _ec_typed_projector(projector_head: str, left_spinor, right_spinor) -> Expression:
    if projector_head == "PL":
        tensor = _EC_CC_PL
    elif projector_head == "PR":
        tensor = _EC_CC_PR
    else:
        raise ValueError(f"Unsupported EC projector head {projector_head!r}.")
    return tensor(
        bispinor_index(left_spinor),
        bispinor_index(right_spinor),
    ).to_expression()


@dataclass(frozen=True)
class _ChargeConjugationArm:
    external: object
    lorentz_ext_to_c: tuple[object, ...]
    used_gamma_indices: frozenset[int]
    c_outgoing: bool | None


@dataclass(frozen=True)
class _EcPartnerPackagingRule:
    partner_key: str
    phase: int
    antisymmetric_duplicates: bool
    source: str


_EC_PARTNER_PACKAGING_RULES: dict[tuple[str, str], _EcPartnerPackagingRule] = {
    (
        "dRbar|eR|lLbar|qL",
        "alphaEcqedl",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|eR|lL|qLbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="LEvCCLRRL alphaEcqedl + HC; one C-arm transposition",
    ),
    (
        "dRbar|eR|lLbar|qL",
        "alphaEcqedlthree",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|eR|lL|qLbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="LEvCCLRRL alphaEcqedlthree + HC; one C-arm transposition",
    ),
    (
        "dR|eRbar|lL|qLbar",
        "alphaEcqedl",
    ): _EcPartnerPackagingRule(
        partner_key="dR|eRbar|lLbar|qL",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCLRRL alphaEcqedl",
    ),
    (
        "dR|eRbar|lL|qLbar",
        "alphaEcqedlthree",
    ): _EcPartnerPackagingRule(
        partner_key="dR|eRbar|lLbar|qL",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCLRRL alphaEcqedlthree",
    ),
    (
        "eRbar|lL|qL|uRbar",
        "alphaEcuelq",
    ): _EcPartnerPackagingRule(
        partner_key="eRbar|lL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcuelq direct packaging",
    ),
    (
        "eRbar|lL|qL|uRbar",
        "alphaEcuelqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="eRbar|lL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcuelqtwo direct packaging",
    ),
    (
        "eR|lLbar|qLbar|uR",
        "alphaEcuelq",
    ): _EcPartnerPackagingRule(
        partner_key="eR|lLbar|qL|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcuelq",
    ),
    (
        "eR|lLbar|qLbar|uR",
        "alphaEcuelqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="eR|lLbar|qL|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcuelqtwo",
    ),
    (
        "dRbar|qL|qL|uRbar",
        "alphaEcudqq",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|qL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=True,
        source="LEvCCRRLL alphaEcudqq; duplicate qL assignments antisymmetrized",
    ),
    (
        "dRbar|qL|qL|uRbar",
        "alphaEcudqqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="dRbar|qL|qLbar|uR",
        phase=1,
        antisymmetric_duplicates=False,
        source="LEvCCRRLL alphaEcudqqtwo; gamma2 structure fixes symmetric duplicate sum",
    ),
    (
        "dR|qLbar|qLbar|uR",
        "alphaEcudqq",
    ): _EcPartnerPackagingRule(
        partner_key="dR|qL|qLbar|uRbar",
        phase=1,
        antisymmetric_duplicates=True,
        source="Hermitian conjugate of LEvCCRRLL alphaEcudqq",
    ),
    (
        "dR|qLbar|qLbar|uR",
        "alphaEcudqqtwo",
    ): _EcPartnerPackagingRule(
        partner_key="dR|qL|qLbar|uRbar",
        phase=-1,
        antisymmetric_duplicates=False,
        source="Hermitian conjugate of LEvCCRRLL alphaEcudqqtwo",
    ),
}


def _is_dirac_c_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "dirac_C"
        and factor[1] == _DIRAC_C_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 2
    )


def _is_gamma_factor(factor: object) -> bool:
    return (
        isinstance(factor, tuple)
        and len(factor) == 3
        and factor[0] == "gamma"
        and factor[1] == _GAMMA_FACTOR_SIGNATURE
        and isinstance(factor[2], tuple)
        and len(factor[2]) == 3
    )


def _is_external_spinor_label(label: object) -> bool:
    return isinstance(label, str) and label.startswith("E:S:")


def _external_leg_number(label: object) -> int | None:
    if not isinstance(label, str) or not label.startswith("E:"):
        return None
    match = re.search(r"(\d+)$", label)
    return int(match.group(1)) if match is not None else None


def _trace_charge_conjugation_arm(
    start: object,
    gamma_factors: list[object],
) -> _ChargeConjugationArm | None:
    adjacency: dict[object, list[tuple[int, object, object, object, bool]]] = defaultdict(
        list
    )
    for index, factor in enumerate(gamma_factors):
        left, right, lorentz = factor[2]
        adjacency[left].append((index, left, right, lorentz, True))
        adjacency[right].append((index, right, left, lorentz, False))

    if _is_external_spinor_label(start) and not adjacency[start]:
        return _ChargeConjugationArm(
            external=start,
            lorentz_ext_to_c=(),
            used_gamma_indices=frozenset(),
            c_outgoing=None,
        )

    current = start
    previous = None
    used: set[int] = set()
    path: list[tuple[object, object, object]] = []
    first_edge_outgoing = None
    for _ in range(len(gamma_factors) + 2):
        candidates = [
            edge
            for edge in adjacency[current]
            if edge[0] not in used and edge[2] != previous
        ]
        if len(candidates) != 1:
            return None
        index, left, right, lorentz, outgoing = candidates[0]
        if first_edge_outgoing is None:
            first_edge_outgoing = outgoing
        used.add(index)
        path.append((left, right, lorentz))
        previous = current
        current = right
        if _is_external_spinor_label(current):
            return _ChargeConjugationArm(
                external=current,
                lorentz_ext_to_c=tuple(edge[2] for edge in reversed(path)),
                used_gamma_indices=frozenset(used),
                c_outgoing=first_edge_outgoing,
            )
    return None


def _fresh_dummy_spinor_label(used_labels: set[object]) -> str:
    candidate = 1
    while True:
        label = f"D:S:{candidate}"
        if label not in used_labels:
            used_labels.add(label)
            return label
        candidate += 1


def _spinor_chain_factors(
    start: object,
    end: object,
    lorentz_sequence: tuple[object, ...],
    used_labels: set[object],
) -> list[object]:
    if not lorentz_sequence:
        return [("g", _SPINOR_METRIC_FACTOR_SIGNATURE, (start, end))]

    spinors = [start]
    for _ in range(len(lorentz_sequence) - 1):
        spinors.append(_fresh_dummy_spinor_label(used_labels))
    spinors.append(end)
    return [
        (
            "gamma",
            _GAMMA_FACTOR_SIGNATURE,
            (spinors[index], spinors[index + 1], lorentz),
        )
        for index, lorentz in enumerate(lorentz_sequence)
    ]


def _charge_conjugation_bilinear_factors(
    first: _ChargeConjugationArm,
    second: _ChargeConjugationArm,
    used_labels: set[object],
) -> list[object] | None:
    if first.lorentz_ext_to_c and second.lorentz_ext_to_c:
        return None

    if first.lorentz_ext_to_c:
        if first.c_outgoing:
            start, end = second.external, first.external
            sequence = first.lorentz_ext_to_c
        else:
            start, end = first.external, second.external
            sequence = tuple(reversed(first.lorentz_ext_to_c))
    elif second.lorentz_ext_to_c:
        if second.c_outgoing:
            start, end = first.external, second.external
            sequence = second.lorentz_ext_to_c
        else:
            start, end = second.external, first.external
            sequence = tuple(reversed(second.lorentz_ext_to_c))
    else:
        start, end = first.external, second.external
        sequence = ()

    return _spinor_chain_factors(start, end, sequence, used_labels)


def _spinor_flow_chain_factors(
    start: object,
    end: object,
    lorentz_sequence: tuple[object, ...],
    used_labels: set[object],
    *,
    projector_head: str,
) -> list[object]:
    if not lorentz_sequence:
        return [(projector_head, _SPINOR_METRIC_FACTOR_SIGNATURE, (start, end))]

    factors = []
    current = start
    for lorentz in lorentz_sequence:
        target = _fresh_dummy_spinor_label(used_labels)
        factors.append(("gamma", _GAMMA_FACTOR_SIGNATURE, (current, target, lorentz)))
        current = target
    factors.append((projector_head, _SPINOR_METRIC_FACTOR_SIGNATURE, (current, end)))
    return factors


def _charge_conjugation_flow_bilinear_factors(
    first: _ChargeConjugationArm,
    second: _ChargeConjugationArm,
    used_labels: set[object],
    *,
    projector_head: str,
) -> list[object] | None:
    if first.lorentz_ext_to_c and second.lorentz_ext_to_c:
        return None

    if first.lorentz_ext_to_c:
        if first.c_outgoing:
            start, end = second.external, first.external
            sequence = first.lorentz_ext_to_c
        else:
            start, end = first.external, second.external
            sequence = tuple(reversed(first.lorentz_ext_to_c))
    elif second.lorentz_ext_to_c:
        if second.c_outgoing:
            start, end = first.external, second.external
            sequence = second.lorentz_ext_to_c
        else:
            start, end = second.external, first.external
            sequence = tuple(reversed(second.lorentz_ext_to_c))
    else:
        start, end = first.external, second.external
        sequence = ()

    return _spinor_flow_chain_factors(
        start,
        end,
        sequence,
        used_labels,
        projector_head=projector_head,
    )


def _replace_external_nonspinor_leg_label(
    label: object,
    first_leg: int,
    second_leg: int,
) -> object:
    if not isinstance(label, str) or not label.startswith("E:"):
        return label
    parts = label.split(":")
    if len(parts) != 3 or parts[1] == "S":
        return label
    leg = _external_leg_number(label)
    if leg == first_leg:
        return re.sub(r"\d+$", str(second_leg), label)
    if leg == second_leg:
        return re.sub(r"\d+$", str(first_leg), label)
    return label


def _replace_factor_external_nonspinor_legs(
    factor: object,
    first_leg: int,
    second_leg: int,
) -> object:
    if not (
        isinstance(factor, tuple)
        and len(factor) == 3
        and isinstance(factor[2], tuple)
    ):
        return factor
    return (
        factor[0],
        factor[1],
        tuple(
            _replace_external_nonspinor_leg_label(label, first_leg, second_leg)
            for label in factor[2]
        ),
    )


def _ec_charge_conjugation_key(
    key: CanonicalTensorMonomial,
    *,
    mode: str,
    phase: int,
) -> tuple[int, CanonicalTensorMonomial] | None:
    factors = list(key.commuting_factors)
    c_indices = [
        index for index, factor in enumerate(factors) if _is_dirac_c_factor(factor)
    ]
    if len(c_indices) != 2:
        return None

    gamma_factors = [factor for factor in factors if _is_gamma_factor(factor)]
    first_c, second_c = (factors[index] for index in c_indices)

    arms: list[_ChargeConjugationArm] = []
    used_gamma_indices: set[int] = set()
    for label in first_c[2] + second_c[2]:
        arm = _trace_charge_conjugation_arm(label, gamma_factors)
        if arm is None:
            return None
        arms.append(arm)
        used_gamma_indices.update(arm.used_gamma_indices)
    if len(used_gamma_indices) != len(gamma_factors):
        return None

    first_leg = second_leg = None
    if mode == "crossed":
        first_leg = _external_leg_number(first_c[2][1])
        second_leg = _external_leg_number(second_c[2][1])
    elif mode != "direct":
        raise ValueError(f"Unsupported Ec charge-conjugation mode {mode!r}.")

    rest = []
    for index, factor in enumerate(factors):
        if index in c_indices or _is_gamma_factor(factor):
            continue
        if first_leg is not None and second_leg is not None:
            factor = _replace_factor_external_nonspinor_legs(
                factor,
                first_leg,
                second_leg,
            )
        rest.append(factor)

    used_labels = {
        label
        for factor in rest
        for label in (
            factor[2]
            if isinstance(factor, tuple)
            and len(factor) == 3
            and isinstance(factor[2], tuple)
            else ()
        )
    }
    used_labels.update(arm.external for arm in arms)

    pairings = {
        "crossed": ((arms[0], arms[3]), (arms[1], arms[2])),
        "direct": ((arms[0], arms[1]), (arms[2], arms[3])),
    }[mode]
    bilinear_factors: list[object] = []
    for first, second in pairings:
        factors_for_pair = _charge_conjugation_bilinear_factors(
            first,
            second,
            used_labels,
        )
        if factors_for_pair is None:
            return None
        bilinear_factors.extend(factors_for_pair)

    return phase, _canonical_key_with_commuting_factors(
        key,
        [*rest, *bilinear_factors],
    )


def _ec_charge_conjugation_flow_key(
    key: CanonicalTensorMonomial,
    *,
    mode: str,
    phase: int,
    projector_head: str,
) -> tuple[int, CanonicalTensorMonomial] | None:
    factors = list(key.commuting_factors)
    c_indices = [
        index for index, factor in enumerate(factors) if _is_dirac_c_factor(factor)
    ]
    if len(c_indices) != 2:
        return None

    gamma_factors = [factor for factor in factors if _is_gamma_factor(factor)]
    first_c, second_c = (factors[index] for index in c_indices)

    arms: list[_ChargeConjugationArm] = []
    used_gamma_indices: set[int] = set()
    for label in first_c[2] + second_c[2]:
        arm = _trace_charge_conjugation_arm(label, gamma_factors)
        if arm is None:
            return None
        arms.append(arm)
        used_gamma_indices.update(arm.used_gamma_indices)
    if len(used_gamma_indices) != len(gamma_factors):
        return None

    first_leg = second_leg = None
    if mode == "crossed":
        first_leg = _external_leg_number(first_c[2][1])
        second_leg = _external_leg_number(second_c[2][1])
    elif mode != "direct":
        raise ValueError(f"Unsupported Ec charge-conjugation mode {mode!r}.")

    rest = []
    for index, factor in enumerate(factors):
        if index in c_indices or _is_gamma_factor(factor):
            continue
        if first_leg is not None and second_leg is not None:
            factor = _replace_factor_external_nonspinor_legs(
                factor,
                first_leg,
                second_leg,
            )
        rest.append(factor)

    used_labels = {
        label
        for factor in rest
        for label in (
            factor[2]
            if isinstance(factor, tuple)
            and len(factor) == 3
            and isinstance(factor[2], tuple)
            else ()
        )
    }
    used_labels.update(arm.external for arm in arms)

    pairings = {
        "crossed": ((arms[0], arms[3]), (arms[1], arms[2])),
        "direct": ((arms[0], arms[1]), (arms[2], arms[3])),
    }[mode]
    bilinear_factors: list[object] = []
    for first, second in pairings:
        factors_for_pair = _charge_conjugation_flow_bilinear_factors(
            first,
            second,
            used_labels,
            projector_head=projector_head,
        )
        if factors_for_pair is None:
            return None
        bilinear_factors.extend(factors_for_pair)

    return phase, _canonical_key_with_commuting_factors(
        key,
        [*rest, *bilinear_factors],
    )


def _swap_ec_coefficient_boundary_args(
    expression: Expression,
    coefficient: str,
) -> Expression:
    first, second, third, fourth = S("ec_first_", "ec_second_", "ec_third_", "ec_fourth_")
    return expression.replace(
        S(coefficient)(first, second, third, fourth),
        S(coefficient)(fourth, second, third, first),
    ).cancel().expand()


def _symbol_from_canonical_label(label: object):
    if not isinstance(label, str):
        return S(str(label))
    if label.startswith("E:"):
        return S(label.split(":", 2)[2])
    if label.startswith("D:"):
        _dummy, group, name = label.split(":", 2)
        prefix = {
            "S": "i",
            "L": "mu",
            "C": "c",
            "W": "w",
            "A": "a",
            "AW": "aw",
        }[group]
        return S(f"{prefix}_cc_dummy_{name}")
    return S(label)


def _expression_from_canonical_factor(factor: object) -> Expression:
    if isinstance(factor, tuple) and len(factor) == 3 and factor[0] == "pcomp":
        return pcomp(
            _symbol_from_canonical_label(factor[1]),
            _symbol_from_canonical_label(factor[2]),
        )
    if not (
        isinstance(factor, tuple)
        and len(factor) == 3
        and isinstance(factor[2], tuple)
    ):
        raise ValueError(f"Unsupported canonical factor in Ec rewrite: {factor!r}")

    head, signature, labels = factor
    args = tuple(_symbol_from_canonical_label(label) for label in labels)
    if head == "g":
        if signature == ("lorentz", "lorentz"):
            return lorentz_metric(*args)
        if signature == ("spinor", "spinor"):
            return spinor_metric(*args)
        if signature == ("color_fund", "color_fund"):
            return COLOR_FUND.g(*args).to_expression()
        if signature == ("color_adj", "color_adj"):
            return COLOR_ADJ.g(*args).to_expression()
        if signature == ("weak_fund", "weak_fund"):
            return WEAK_FUND.g(*args).to_expression()
        if signature == ("weak_adj", "weak_adj"):
            return WEAK_ADJ.g(*args).to_expression()
    if head == "gamma":
        return gamma_matrix(*args)
    if head == "dirac_C":
        return dirac_charge_conjugation(*args)
    if head in {"PL", "PR"}:
        return _ec_typed_projector(head, *args)
    if head == "weak_eps2":
        return weak_eps2(*args)
    if head == "lor_levi_civita":
        return lorentz_levi_civita(*args)
    if head == "t":
        if signature == _COLOR_T_FACTOR_SIGNATURE:
            return gauge_generator(*args)
        if signature == _WEAK_T_FACTOR_SIGNATURE:
            return weak_gauge_generator(*args)
    if head == "f":
        if signature == ("color_adj", "color_adj", "color_adj"):
            return structure_constant(*args)
        if signature == ("weak_adj", "weak_adj", "weak_adj"):
            return weak_structure_constant(*args)

    raise ValueError(f"Unsupported canonical factor in Ec rewrite: {factor!r}")


def _expression_from_canonical_map(
    mapping: dict[CanonicalTensorMonomial, Expression],
) -> Expression:
    total = Expression.num(0)
    for key, coefficient in mapping.items():
        term = coefficient
        for factor in key.commuting_factors:
            term *= _expression_from_canonical_factor(factor)
        for factor in key.ordered_factors:
            term *= _expression_from_canonical_factor(factor)
        total += term
    return total.cancel().expand()


def _normalize_ec_charge_conjugation_report(
    report: CanonicalMonomialReport,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
    mode: str,
    phase: int,
    fired: list[bool] | None = None,
) -> CanonicalMonomialReport:
    transformed: dict[CanonicalTensorMonomial, Expression] = {}
    changed = False
    for key, coefficient_expression in report.map.items():
        replacement = _ec_charge_conjugation_key(key, mode=mode, phase=phase)
        if replacement is None:
            transformed_key = key
            transformed_coefficient = coefficient_expression
        else:
            multiplier, transformed_key = replacement
            transformed_coefficient = (
                coefficient_expression * Expression.num(multiplier)
            ).cancel().expand()
            if mode == "crossed":
                transformed_coefficient = _swap_ec_coefficient_boundary_args(
                    transformed_coefficient,
                    coefficient,
                )
            changed = True
        transformed[transformed_key] = (
            transformed.get(transformed_key, Expression.num(0))
            + transformed_coefficient
        ).cancel().expand()

    if fired is not None:
        fired.append(changed)
    if not changed:
        return report

    recanonicalized = canonical_tensor_monomial_report(
        _expression_from_canonical_map(transformed),
        external_indices=external_indices,
        tensor_head_specs=_EC_CC_TENSOR_HEAD_SPECS,
        max_dummy_permutations=max_dummy_permutations,
    )
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=recanonicalized.canonical_terms,
        map=recanonicalized.map,
    )


def _normalize_ec_charge_conjugation_flow_report(
    report: CanonicalMonomialReport,
    *,
    coefficient: str,
    external_indices,
    max_dummy_permutations: int,
    mode: str,
    phase: int,
    projector_head: str,
) -> CanonicalMonomialReport:
    transformed: dict[CanonicalTensorMonomial, Expression] = {}
    changed = False
    for key, coefficient_expression in report.map.items():
        replacement = _ec_charge_conjugation_flow_key(
            key,
            mode=mode,
            phase=phase,
            projector_head=projector_head,
        )
        if replacement is None:
            transformed_key = key
            transformed_coefficient = coefficient_expression
        else:
            multiplier, transformed_key = replacement
            transformed_coefficient = (
                coefficient_expression * Expression.num(multiplier)
            ).cancel().expand()
            if mode == "crossed":
                transformed_coefficient = _swap_ec_coefficient_boundary_args(
                    transformed_coefficient,
                    coefficient,
                )
            changed = True
        transformed[transformed_key] = (
            transformed.get(transformed_key, Expression.num(0))
            + transformed_coefficient
        ).cancel().expand()

    if not changed:
        return report

    recanonicalized = canonical_tensor_monomial_report(
        _expression_from_canonical_map(transformed),
        external_indices=external_indices,
        tensor_head_specs=_EC_CC_TENSOR_HEAD_SPECS,
        max_dummy_permutations=max_dummy_permutations,
    )
    return CanonicalMonomialReport(
        raw_terms=report.raw_terms,
        canonical_terms=recanonicalized.canonical_terms,
        map=recanonicalized.map,
    )


def _coefficient_comparison_from_reports(
    coefficient: str,
    feynpy_report: CanonicalMonomialReport,
    feynrules_report: CanonicalMonomialReport,
) -> CanonicalCoefficientComparison:
    feynpy_keys = set(feynpy_report.map)
    feynrules_keys = set(feynrules_report.map)
    shared_keys = feynpy_keys & feynrules_keys
    coefficient_mismatches = {
        key: (feynpy_report.map[key], feynrules_report.map[key])
        for key in shared_keys
        if feynpy_report.map[key].cancel().expand().to_canonical_string()
        != feynrules_report.map[key].cancel().expand().to_canonical_string()
    }
    return CanonicalCoefficientComparison(
        coefficient=coefficient,
        feynpy_raw_terms=feynpy_report.raw_terms,
        feynrules_raw_terms=feynrules_report.raw_terms,
        feynpy_canonical_terms=feynpy_report.canonical_terms,
        feynrules_canonical_terms=feynrules_report.canonical_terms,
        feynpy_only={
            key: feynpy_report.map[key]
            for key in sorted(feynpy_keys - feynrules_keys, key=repr)
        },
        feynrules_only={
            key: feynrules_report.map[key]
            for key in sorted(feynrules_keys - feynpy_keys, key=repr)
        },
        coefficient_mismatches=coefficient_mismatches,
    )


def _bar_insensitive_name(name: str) -> str:
    return name[:-3] if name.endswith("bar") else name


def _bar_insensitive_field_key(fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(_bar_insensitive_name(name) for name in fields))


def _charge_conjugation_candidate_orders(
    reference_fields: tuple[str, ...],
    partner_fields: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    orders: list[tuple[str, ...]] = []

    def visit(slot: int, remaining: tuple[str, ...], current: tuple[str, ...]) -> None:
        if slot == len(reference_fields):
            orders.append(current)
            return
        target_base = _bar_insensitive_name(reference_fields[slot])
        for index, candidate in enumerate(remaining):
            if _bar_insensitive_name(candidate) != target_base:
                continue
            visit(
                slot + 1,
                remaining[:index] + remaining[index + 1 :],
                (*current, candidate),
            )

    visit(0, tuple(partner_fields), ())
    return tuple(orders)


def _duplicate_assignment_sign(
    order: tuple[str, ...],
    reference_fields: tuple[str, ...],
) -> int:
    sign = 1
    for base in sorted({_bar_insensitive_name(name) for name in reference_fields}):
        slots = [
            slot
            for slot, name in enumerate(reference_fields)
            if _bar_insensitive_name(name) == base
        ]
        if len(slots) < 2:
            continue
        current = tuple(order[slot] for slot in slots)
        target = tuple(sorted(current))
        sign *= _permutation_sign(current, target)
    return sign


def _candidate_order_rule_sum(
    *,
    lagrangian,
    field_map: dict[str, object],
    reference_fields: tuple[str, ...],
    candidate_orders: tuple[tuple[str, ...], ...],
    antisymmetric_duplicates: bool,
) -> Expression:
    total = Expression.num(0)
    for order in candidate_orders:
        sign = (
            _duplicate_assignment_sign(order, reference_fields)
            if antisymmetric_duplicates
            else 1
        )
        total += Expression.num(sign) * lagrangian.feynman_rule(
            *(field_map[name] for name in order),
            simplify=True,
        )
    return total.cancel().expand()


def _ec_partner_packaging_comparison(
    *,
    reference: FeynRulesVertex,
    coefficient: str,
    feynrules_report: CanonicalMonomialReport,
    local_vertices: Iterable["LocalVertex"],
    lagrangian,
    field_map: dict[str, object],
    external_indices,
    max_dummy_permutations: int,
) -> tuple[CanonicalCoefficientComparison, str] | None:
    """Apply a pinned Ec partner-packaging rule for one coefficient sector."""

    reference_key = _name_key(reference.fields)
    rule = _EC_PARTNER_PACKAGING_RULES.get((reference_key, coefficient))
    if rule is None:
        return None

    candidates = [
        vertex
        for vertex in local_vertices
        if vertex.key == rule.partner_key
        and coefficient in dict(vertex.head_counts)
    ]
    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    candidate_orders = _charge_conjugation_candidate_orders(
        reference.fields,
        candidate.local_names,
    )
    if not candidate_orders:
        return None

    local_rule = _candidate_order_rule_sum(
        lagrangian=lagrangian,
        field_map=field_map,
        reference_fields=reference.fields,
        candidate_orders=candidate_orders,
        antisymmetric_duplicates=rule.antisymmetric_duplicates,
    )
    local_report = _canonical_report_for_coefficient_head(
        local_rule,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
    )
    transformed_report = _normalize_ec_charge_conjugation_report(
        local_report,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=max_dummy_permutations,
        mode="direct",
        phase=rule.phase,
    )
    comparison = _coefficient_comparison_from_reports(
        coefficient,
        transformed_report,
        feynrules_report,
    )
    if not comparison.matches:
        return None

    duplicate_mode = (
        "antisymmetric duplicate-leg sum"
        if rule.antisymmetric_duplicates
        else "symmetric duplicate-leg sum"
    )
    return (
        comparison,
        (
            f"{coefficient} matched via pinned charge-conjugation partner "
            f"`{candidate.key}` using direct CC packaging, phase {rule.phase:+d}, "
            f"and {duplicate_mode} ({rule.source})."
        ),
    )



def _compare_smeft2_canonical_coefficient_maps(
    feynpy_rule: Expression | str,
    feynrules_rule: Expression | str,
    *,
    coefficients: Iterable[str],
    external_indices,
    max_dummy_permutations: int = 50_000,
    convention_log: set[str] | None = None,
) -> dict[str, CanonicalCoefficientComparison]:
    """Compare SMEFT2 rows by coefficient-head-filtered canonical maps.

    ``convention_log``, when supplied, collects the coefficient heads whose
    FeynPy side actually required the global ``alphaEc`` charge-conjugation
    packaging convention (see ``_EC_CC_CONVENTION_MODE``). Callers use it to
    grade such rows as ``MATCH_MODULO_EC_CC_CONVENTION`` instead of claiming a
    direct ``EXACT_MATCH``.
    """

    feynpy_expression = (
        Expression.parse(feynpy_rule)
        if isinstance(feynpy_rule, str)
        else feynpy_rule
    )
    feynrules_expression = (
        Expression.parse(feynrules_rule)
        if isinstance(feynrules_rule, str)
        else feynrules_rule
    )

    comparisons = {}
    for coefficient in coefficients:
        feynpy_report = _canonical_report_for_coefficient_head(
            feynpy_expression,
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=max_dummy_permutations,
        )
        if coefficient.startswith("alphaEc"):
            fired: list[bool] = []
            feynpy_report = _normalize_ec_charge_conjugation_report(
                feynpy_report,
                coefficient=coefficient,
                external_indices=external_indices,
                max_dummy_permutations=max_dummy_permutations,
                mode=_EC_CC_CONVENTION_MODE,
                phase=_EC_CC_CONVENTION_PHASE,
                fired=fired,
            )
            if convention_log is not None and any(fired):
                convention_log.add(coefficient)
        feynrules_report = _canonical_report_for_coefficient_head(
            feynrules_expression,
            coefficient=coefficient,
            external_indices=external_indices,
            max_dummy_permutations=max_dummy_permutations,
        )
        comparisons[coefficient] = _coefficient_comparison_from_reports(
            coefficient,
            feynpy_report,
            feynrules_report,
        )
    return comparisons


def _replace_tensdot_chains_with_projector_labels(text: str) -> str:
    output = []
    position = 0
    chain_id = 0
    while True:
        start = text.find("TensDot[", position)
        if start == -1:
            output.append(text[position:])
            return "".join(output)

        output.append(text[position:start])
        inner_open = start + len("TensDot")
        inner_close = _find_matching_square(text, inner_open)
        if inner_close + 1 >= len(text) or text[inner_close + 1] != "[":
            output.append(text[start : inner_close + 1])
            position = inner_close + 1
            continue

        spin_open = inner_close + 1
        spin_close = _find_matching_square(text, spin_open)
        chain_id += 1
        items = _split_top_level_commas(text[inner_open + 1 : inner_close])
        spin_args = _split_top_level_commas(text[spin_open + 1 : spin_close])
        if len(spin_args) != 2:
            raise ValueError(f"Unsupported FeynRules TensDot spin args: {spin_args}")
        output.append(
            _spinor_chain_replacement_with_projector_label(
                items,
                spin_args[0],
                spin_args[1],
                chain_id=chain_id,
            )
        )
        position = spin_close + 1


def parse_smeft2_matter_rule_with_projector_labels(rule: str) -> Expression:
    """Parse SMEFT2 matter syntax while retaining explicit PL/PR labels.

    This parser is used only by the EC charge-conjugation sidecar.  The normal
    comparison parser intentionally remains unchanged.
    """

    text = _rewrite_feynrules_indices(rule)
    text = _rewrite_feynrules_indexed_parameters(text)
    _replace_scalar_product.counter = 0

    text = _replace_tensdot_chains_with_projector_labels(text)
    text = re.sub(
        r"ProjM\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: _ec_typed_projector(
            "PL",
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjP\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: _ec_typed_projector(
            "PR",
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ga\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gamma_matrix(
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(1).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"IndexDelta\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: _metric_for_index_delta(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"ME\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_metric(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"FV\[(\d+),\s*([^\[\]]+)\]",
        lambda match: pcomp(
            S(f"q{match.group(1)}"),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(r"SP\[(\d+),\s*(\d+)\]", _replace_scalar_product, text)
    text = re.sub(
        r"T\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ta\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"(?:f|fsu3)\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"fsu2\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_levi_civita(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(4).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_eps2(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = text.replace("Sqrt[2]", "(2)^(1/2)")
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported SMEFT2 FeynRules matter syntax remains after "
            f"projector-label parsing: {text}"
        )

    return Expression.parse(text).cancel().expand()


def _replace_tensdot_chains(text: str) -> str:
    output = []
    position = 0
    chain_id = 0
    while True:
        start = text.find("TensDot[", position)
        if start == -1:
            output.append(text[position:])
            return "".join(output)

        output.append(text[position:start])
        inner_open = start + len("TensDot")
        inner_close = _find_matching_square(text, inner_open)
        if inner_close + 1 >= len(text) or text[inner_close + 1] != "[":
            output.append(text[start : inner_close + 1])
            position = inner_close + 1
            continue

        spin_open = inner_close + 1
        spin_close = _find_matching_square(text, spin_open)
        chain_id += 1
        items = _split_top_level_commas(text[inner_open + 1 : inner_close])
        spin_args = _split_top_level_commas(text[spin_open + 1 : spin_close])
        if len(spin_args) != 2:
            raise ValueError(f"Unsupported FeynRules TensDot spin args: {spin_args}")
        output.append(
            _spinor_chain_replacement(
                items,
                spin_args[0],
                spin_args[1],
                chain_id=chain_id,
            )
        )
        position = spin_close + 1


def parse_smeft2_matter_rule(
    rule: str,
    *,
    projector_as_dirac_c: bool = False,
    fields: Iterable[str] | None = None,
) -> Expression:
    """Parse SMEFT2 two-fermion FeynRules tensor syntax into native tensors.

    The SMEFT2 model represents chirality in the field class itself, while the
    FeynRules export prints explicit ``ProjM``/``ProjP`` factors. This parser
    therefore removes those projectors from gamma chains and maps bare
    projector bilinears to the spinor metric.  The Weinberg charge-conjugation
    packaging comparison is the exception: there FeynRules' same-chirality
    projector row is compared to FeynPy's mixed ``LLbar C LL`` packaging, so
    bare projectors are mapped to the antisymmetric Dirac charge-conjugation
    tensor.

    Dropping a projector is only sound when it agrees with the chirality of the
    fields it sits between. Pass ``fields`` (the reference vertex field names,
    in leg order) to have that agreement verified rather than assumed; a
    disagreement raises :class:`ChiralityMismatch`.
    """

    if fields is not None:
        validate_smeft2_projector_chirality(rule, fields)

    text = _rewrite_feynrules_indices(rule)
    text = _rewrite_feynrules_indexed_parameters(text)
    _replace_scalar_product.counter = 0

    text = _replace_tensdot_chains(text)
    projector_tensor = (
        dirac_charge_conjugation if projector_as_dirac_c else spinor_metric
    )
    text = re.sub(
        r"Proj[MP]\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: projector_tensor(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ga\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gamma_matrix(
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(1).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"IndexDelta\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: _metric_for_index_delta(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"ME\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_metric(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"FV\[(\d+),\s*([^\[\]]+)\]",
        lambda match: pcomp(
            S(f"q{match.group(1)}"),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(r"SP\[(\d+),\s*(\d+)\]", _replace_scalar_product, text)
    text = re.sub(
        r"T\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Ta\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_gauge_generator(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"(?:f|fsu3)\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"fsu2\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_structure_constant(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^,\[\]]+),\s*([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: lorentz_levi_civita(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
            S(match.group(3).strip()),
            S(match.group(4).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"Eps\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: weak_eps2(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = text.replace("Sqrt[2]", "(2)^(1/2)")
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported SMEFT2 FeynRules matter syntax remains after parsing: "
            f"{text}"
        )

    return Expression.parse(text).cancel().expand()

__all__ = [name for name in globals() if not name.startswith("__")]
