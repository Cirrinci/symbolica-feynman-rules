"""Sidecar reconstructions for charge-conjugation packaged SMEFT2 rows."""

from .exact import *


def _parse_weinberg_fermion_flow_rule(rule: str) -> Expression:
    """Parse a Weinberg FeynRules row into the sidecar fermion-flow basis."""

    text = _rewrite_feynrules_indices(rule)
    text = _rewrite_feynrules_indexed_parameters(text)
    text = re.sub(
        r"ProjM\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: S("PL")(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
        ).to_canonical_string(),
        text,
    )
    text = re.sub(
        r"ProjP\[([^,\[\]]+),\s*([^\[\]]+)\]",
        lambda match: S("PR")(
            S(match.group(1).strip()),
            S(match.group(2).strip()),
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
    text = re.sub(r"\bI\b", "1𝑖", text)

    if "[" in text or "]" in text:
        raise ValueError(
            "Unsupported Weinberg FeynRules syntax remains after parsing: "
            f"{text}"
        )
    return Expression.parse(text).cancel().expand()


def _weinberg_projector_head(reference_fields: tuple[str, ...]) -> str:
    if sorted(reference_fields) == ["Phi", "Phi", "lL", "lL"]:
        return "PL"
    if sorted(reference_fields) == ["Phibar", "Phibar", "lLbar", "lLbar"]:
        return "PR"
    raise ValueError(f"Unsupported Weinberg reference fields: {reference_fields!r}")


def _replace_dirac_c_with_weinberg_flow(
    expression: Expression,
    *,
    projector_head: str,
) -> Expression:
    left, right = S("weinberg_spin_left_", "weinberg_spin_right_")
    return expression.replace(
        dirac_charge_conjugation(left, right),
        S(projector_head)(left, right),
    ).cancel().expand()


def _weinberg_external_indices(reference: FeynRulesVertex, field_map: dict[str, object]):
    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        raise ValueError(f"Could not infer external indices for {reference.fields!r}.")
    return external_indices


def _reconstructed_weinberg_flow_rule(
    *,
    reference: FeynRulesVertex,
    lagrangian,
    field_map: dict[str, object],
    sign: int = -1,
) -> Expression:
    """Return the Weinberg sidecar rule in the common same-chirality order.

    ``sign=-1`` gives the charge-conjugation antisymmetric combination
    ``first - second``. ``sign=+1`` is kept for regression tests; it must not
    match the same-chirality FeynRules row.
    """

    packaged_orders = _weinberg_packaged_field_orders(reference.fields)
    if packaged_orders is None:
        raise ValueError(f"Unsupported Weinberg fields: {reference.fields!r}")
    if sign not in {-1, 1}:
        raise ValueError(f"Unsupported Weinberg reconstruction sign {sign!r}.")

    first_rule = lagrangian.feynman_rule(
        *(field_map[name] for name in packaged_orders[0]),
        simplify=True,
    )
    second_rule = lagrangian.feynman_rule(
        *(field_map[name] for name in packaged_orders[1]),
        simplify=True,
    )
    combined = (first_rule + Expression.num(sign) * second_rule).cancel().expand()

    # Canonicalize while the spinor pair is still the antisymmetric C tensor.
    # This maps the second mixed ordering into the same external spinor order
    # and produces the transposed flavor coefficient without assuming symmetry.
    canonical_c_report = _canonical_report_for_coefficient_head(
        combined,
        coefficient="alphaWeinberg",
        external_indices=_weinberg_external_indices(reference, field_map),
        max_dummy_permutations=2_000_000,
    )
    canonical_c_rule = _expression_from_canonical_map(canonical_c_report.map)
    return _replace_dirac_c_with_weinberg_flow(
        canonical_c_rule,
        projector_head=_weinberg_projector_head(reference.fields),
    )


def _weinberg_canonical_report(
    expression: Expression,
    *,
    external_indices,
) -> CanonicalMonomialReport:
    return canonical_tensor_monomial_report(
        expression.cancel().expand(),
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )


def _weinberg_canonical_zero(
    expression: Expression,
    *,
    external_indices,
) -> bool:
    return not _weinberg_canonical_report(
        expression,
        external_indices=external_indices,
    ).map


def _weinberg_coefficient_expression(text: str) -> Expression:
    if text.startswith("conj(") and text.endswith(")"):
        inner = text[len("conj(") : -1]
        return S("conj")(Expression.parse(inner))
    return Expression.parse(text)


def _weinberg_coefficient_checks(
    *,
    feynpy_rule: Expression,
    feynrules_rule: Expression,
    coefficient_texts: tuple[str, ...],
    external_indices,
) -> list[dict[str, object]]:
    checks = []
    for coefficient_text in coefficient_texts:
        coefficient = _weinberg_coefficient_expression(coefficient_text)
        feynpy_coefficient = feynpy_rule.coefficient(coefficient).cancel().expand()
        feynrules_coefficient = (
            feynrules_rule.coefficient(coefficient).cancel().expand()
        )
        difference = (feynpy_coefficient - feynrules_coefficient).cancel().expand()
        checks.append(
            {
                "coefficient": coefficient_text,
                "matches": _weinberg_canonical_zero(
                    difference,
                    external_indices=external_indices,
                ),
                "feynpy_coefficient": feynpy_coefficient.to_canonical_string(),
                "feynrules_coefficient": feynrules_coefficient.to_canonical_string(),
                "difference": difference.to_canonical_string(),
            }
        )
    return checks


def _weinberg_reference_vertices(
    reference_path: Path = REFERENCE,
) -> tuple[FeynRulesVertex, FeynRulesVertex]:
    references_by_key = {
        _name_key(reference.fields): reference
        for reference in load_feynrules_json(reference_path)
    }
    return (
        references_by_key["Phi|Phi|lL|lL"],
        references_by_key["Phibar|Phibar|lLbar|lLbar"],
    )


def compare_reconstructed_weinberg(
    reference_path: Path = REFERENCE,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build and directly compare the Weinberg sidecar export.

    This is deliberately separate from the existing SMEFT2 aggregate comparison.
    It only reconstructs the two same-chirality Weinberg rows from the two
    ordered mixed FeynPy calls and compares them to the corresponding
    FeynRules rows in a compact fermion-flow basis.
    """

    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)

    vertices = []
    comparison_rows = []
    for reference in _weinberg_reference_vertices(reference_path):
        key = _name_key(reference.fields)
        external_indices = _weinberg_external_indices(reference, field_map)
        feynpy_rule = _reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=-1,
        )
        wrong_sign_rule = _reconstructed_weinberg_flow_rule(
            reference=reference,
            lagrangian=lagrangian,
            field_map=field_map,
            sign=1,
        )
        feynrules_rule = _parse_weinberg_fermion_flow_rule(reference.rule)
        difference = (feynpy_rule - feynrules_rule).cancel().expand()
        wrong_sign_difference = (wrong_sign_rule - feynrules_rule).cancel().expand()
        coefficient_texts = (
            (
                "alphaWeinberg(f1,f2)",
                "alphaWeinberg(f2,f1)",
            )
            if key == "Phi|Phi|lL|lL"
            else (
                "conj(alphaWeinberg(f1,f2))",
                "conj(alphaWeinberg(f2,f1))",
            )
        )
        coefficient_checks = _weinberg_coefficient_checks(
            feynpy_rule=feynpy_rule,
            feynrules_rule=feynrules_rule,
            coefficient_texts=coefficient_texts,
            external_indices=external_indices,
        )
        matches = _weinberg_canonical_zero(
            difference,
            external_indices=external_indices,
        )
        wrong_sign_matches = _weinberg_canonical_zero(
            wrong_sign_difference,
            external_indices=external_indices,
        )
        packaged_orders = _weinberg_packaged_field_orders(reference.fields)
        vertices.append(
            {
                "key": key,
                "fields": list(reference.fields),
                "source_orders": {
                    "first": list(packaged_orders[0]),
                    "second": list(packaged_orders[1]),
                    "combination": "first - second",
                },
                "spinor_representation": _weinberg_projector_head(
                    reference.fields
                ),
                "flavor_structures": list(coefficient_texts),
                "rule": feynpy_rule.to_canonical_string(),
            }
        )
        comparison_rows.append(
            {
                "key": key,
                "fields": list(reference.fields),
                "matches": matches,
                "wrong_sign_matches": wrong_sign_matches,
                "coefficient_checks": coefficient_checks,
                "difference": difference.to_canonical_string(),
                "wrong_sign_difference": wrong_sign_difference.to_canonical_string(),
                "feynpy_rule": feynpy_rule.to_canonical_string(),
                "feynrules_rule": feynrules_rule.to_canonical_string(),
            }
        )

    report = {
        "generated_on": date.today().isoformat(),
        "comparison_level": (
            "Weinberg-only sidecar comparison. FeynPy mixed charge-conjugation "
            "orders are reconstructed as first - second in the common "
            "same-chirality external-leg order, then compared directly to the "
            "FeynRules lL lL Phi Phi and lLbar lLbar Phibar Phibar rows in a "
            "compact PL/PR fermion-flow basis."
        ),
        "summary": {
            "reference_vertices": len(comparison_rows),
            "direct_matches": sum(row["matches"] for row in comparison_rows),
            "wrong_sign_matches": sum(
                row["wrong_sign_matches"] for row in comparison_rows
            ),
            "coefficient_checks": sum(
                len(row["coefficient_checks"]) for row in comparison_rows
            ),
            "coefficient_matches": sum(
                sum(check["matches"] for check in row["coefficient_checks"])
                for row in comparison_rows
            ),
        },
        "vertices": comparison_rows,
    }
    return report, vertices


def _canonical_difference_expression(
    feynpy_report: CanonicalMonomialReport,
    feynrules_report: CanonicalMonomialReport,
) -> Expression:
    difference: dict[CanonicalTensorMonomial, Expression] = {}
    for key, coefficient in feynpy_report.map.items():
        difference[key] = (
            difference.get(key, Expression.num(0)) + coefficient
        ).cancel().expand()
    for key, coefficient in feynrules_report.map.items():
        difference[key] = (
            difference.get(key, Expression.num(0)) - coefficient
        ).cancel().expand()
    return _expression_from_canonical_map(
        {
            key: coefficient
            for key, coefficient in difference.items()
            if coefficient.cancel().expand().to_canonical_string() != "0"
        }
    )


def _alpha_heads_in_expression(expression: Expression) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                re.findall(
                    r"(?:^|::)(alpha[A-Za-z0-9]+)\(",
                    expression.cancel().expand().to_canonical_string(),
                )
            )
        )
    )


def _ec_flow_canonical_report(
    expression: Expression,
    *,
    coefficient: str,
    external_indices,
) -> CanonicalMonomialReport:
    report = canonical_tensor_monomial_report(
        _filter_terms_by_coefficient_head(expression, coefficient),
        external_indices=external_indices,
        tensor_head_specs=_EC_CC_TENSOR_HEAD_SPECS,
        max_dummy_permutations=2_000_000,
    )
    report = _normalize_generator_product_order_report(report)
    report = _normalize_weak_t_epsilon_report(report)
    report = _normalize_weak_doublet_generator_epsilon_report(report)
    return _normalize_structure_constant_jacobi_report(report)


def _ec_projector_head(filtered_feynrules_rule: Expression) -> str:
    compact_text = filtered_feynrules_rule.cancel().expand().to_canonical_string()
    heads = set(re.findall(r"(?:^|::)(P[LR])\(", compact_text))
    if len(heads) != 1:
        raise ValueError(
            "Expected one EC projector head in filtered FeynRules rule, got "
            f"{sorted(heads)!r}."
        )
    return next(iter(heads))


def _ec_reference_vertices_by_key(
    reference_path: Path,
) -> dict[str, FeynRulesVertex]:
    return {
        _name_key(reference.fields): reference
        for reference in load_feynrules_json(reference_path)
    }


def _ec_order_pair(
    *,
    reference_fields: tuple[str, ...],
    candidate_orders: tuple[tuple[str, ...], ...],
    antisymmetric_duplicates: bool,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if len(candidate_orders) == 1:
        return candidate_orders[0], None
    if len(candidate_orders) != 2:
        raise ValueError(
            "Expected one or two EC candidate orders, got "
            f"{candidate_orders!r}."
        )
    if not antisymmetric_duplicates:
        return candidate_orders[0], candidate_orders[1]

    orders_by_sign = {
        _duplicate_assignment_sign(order, reference_fields): order
        for order in candidate_orders
    }
    if set(orders_by_sign) != {-1, 1}:
        raise ValueError(
            "Could not identify antisymmetric duplicate-leg EC order pair: "
            f"{candidate_orders!r}."
        )
    return orders_by_sign[1], orders_by_sign[-1]


def _ec_ordered_rule(lagrangian, field_map: dict[str, object], order: tuple[str, ...]):
    return lagrangian.feynman_rule(
        *(field_map[name] for name in order),
        simplify=True,
    )


def _ec_candidate_raw_rules(
    *,
    lagrangian,
    field_map: dict[str, object],
    first_order: tuple[str, ...],
    second_order: tuple[str, ...] | None,
) -> dict[str, Expression]:
    first_rule = _ec_ordered_rule(lagrangian, field_map, first_order)
    if second_order is None:
        return {"first": first_rule}

    second_rule = _ec_ordered_rule(lagrangian, field_map, second_order)
    return {
        "first - second": (first_rule - second_rule).cancel().expand(),
        "first + second": (first_rule + second_rule).cancel().expand(),
    }


def _ec_source_contributions(
    *,
    first_order: tuple[str, ...],
    second_order: tuple[str, ...] | None,
    combination: str | None,
    phase: int | None,
) -> list[dict[str, object]]:
    if combination is None or phase is None:
        return []

    combination_signs = {
        "first": (("first", 1),),
        "first - second": (("first", 1), ("second", -1)),
        "first + second": (("first", 1), ("second", 1)),
    }
    if combination not in combination_signs:
        raise ValueError(f"Unsupported EC source combination {combination!r}.")

    orders = {"first": first_order, "second": second_order}
    contributions = []
    for label, sign in combination_signs[combination]:
        order = orders[label]
        if order is None:
            continue
        contributions.append(
            {
                "label": label,
                "order": list(order),
                "weight": phase * sign,
            }
        )
    return contributions


def _ec_raw_projector_flow_heads() -> frozenset[str]:
    return frozenset({"alphaEcudqq", "alphaEcuelq"})


def _replace_dirac_c_with_projector_label(
    expression: Expression,
    *,
    projector_head: str,
) -> Expression:
    left, right = S("ec_raw_c_left_", "ec_raw_c_right_")
    return expression.replace(
        dirac_charge_conjugation(left, right),
        _ec_typed_projector(projector_head, left, right),
    ).cancel().expand()


def _ec_flow_report_for_local_candidate(
    *,
    raw_rule: Expression,
    coefficient: str,
    external_indices,
    phase: int,
    projector_head: str,
) -> CanonicalMonomialReport:
    local_report = _canonical_report_for_coefficient_head(
        raw_rule,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
    )
    return _normalize_ec_charge_conjugation_flow_report(
        local_report,
        coefficient=coefficient,
        external_indices=external_indices,
        max_dummy_permutations=2_000_000,
        mode="direct",
        phase=phase,
        projector_head=projector_head,
    )


def _ec_flow_report_for_raw_projector_candidate(
    *,
    raw_rule: Expression,
    coefficient: str,
    external_indices,
    phase: int,
    projector_head: str,
) -> CanonicalMonomialReport:
    flow_rule = _replace_dirac_c_with_projector_label(
        raw_rule,
        projector_head=projector_head,
    )
    flow_rule = (Expression.num(phase) * flow_rule).cancel().expand()
    return _ec_flow_canonical_report(
        flow_rule,
        coefficient=coefficient,
        external_indices=external_indices,
    )


def _ec_comparison_row(
    *,
    reference: FeynRulesVertex,
    coefficient: str,
    lagrangian,
    field_map: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    reference_key = _name_key(reference.fields)
    rule = _EC_PARTNER_PACKAGING_RULES[(reference_key, coefficient)]
    external_indices = _weinberg_external_indices(reference, field_map)

    parsed_feynrules = parse_smeft2_matter_rule_with_projector_labels(reference.rule)
    filtered_feynrules = _filter_terms_by_coefficient_head(
        parsed_feynrules,
        coefficient,
    )
    feynrules_report = _ec_flow_canonical_report(
        filtered_feynrules,
        coefficient=coefficient,
        external_indices=external_indices,
    )
    projector_head = _ec_projector_head(filtered_feynrules)

    partner_fields = tuple(rule.partner_key.split("|"))
    candidate_orders = _charge_conjugation_candidate_orders(
        reference.fields,
        partner_fields,
    )
    first_order, second_order = _ec_order_pair(
        reference_fields=reference.fields,
        candidate_orders=candidate_orders,
        antisymmetric_duplicates=rule.antisymmetric_duplicates,
    )
    candidates = _ec_candidate_raw_rules(
        lagrangian=lagrangian,
        field_map=field_map,
        first_order=first_order,
        second_order=second_order,
    )

    tested_combinations = {}
    matching_combinations = []
    candidate_reports = {}
    use_raw_projector_flow = coefficient in _ec_raw_projector_flow_heads()
    phases = (1, -1) if use_raw_projector_flow or second_order is None else (rule.phase,)
    for combination, raw_rule in candidates.items():
        for phase in phases:
            tested_name = f"{combination}; phase {phase:+d}"
            if use_raw_projector_flow:
                feynpy_report = _ec_flow_report_for_raw_projector_candidate(
                    raw_rule=raw_rule,
                    coefficient=coefficient,
                    external_indices=external_indices,
                    phase=phase,
                    projector_head=projector_head,
                )
            else:
                feynpy_report = _ec_flow_report_for_local_candidate(
                    raw_rule=raw_rule,
                    coefficient=coefficient,
                    external_indices=external_indices,
                    phase=phase,
                    projector_head=projector_head,
                )
            candidate_reports[tested_name] = feynpy_report
            comparison = _coefficient_comparison_from_reports(
                coefficient,
                feynpy_report,
                feynrules_report,
            )
            difference = _canonical_difference_expression(
                feynpy_report,
                feynrules_report,
            )
            tested_combinations[tested_name] = {
                "combination": combination,
                "phase": phase,
                "matches": comparison.matches,
                "canonical_difference": difference.to_canonical_string(),
                "feynpy_canonical_terms": comparison.feynpy_canonical_terms,
                "feynrules_canonical_terms": comparison.feynrules_canonical_terms,
            }
            if comparison.matches:
                matching_combinations.append(tested_name)

    status = "exact_match" if len(matching_combinations) == 1 else "mismatch"
    chosen_candidate = matching_combinations[0] if status == "exact_match" else None
    chosen_combination = (
        tested_combinations[chosen_candidate]["combination"]
        if chosen_candidate is not None
        else None
    )
    chosen_phase = (
        tested_combinations[chosen_candidate]["phase"]
        if chosen_candidate is not None
        else None
    )
    chosen_report = (
        candidate_reports[chosen_candidate]
        if chosen_candidate is not None
        else next(iter(candidate_reports.values()))
    )
    canonical_difference = _canonical_difference_expression(
        chosen_report,
        feynrules_report,
    )
    feynpy_expression = _expression_from_canonical_map(chosen_report.map)
    feynrules_expression = _expression_from_canonical_map(feynrules_report.map)
    second_order_payload = list(second_order) if second_order is not None else None
    tested_identical_leg_mappings = (
        [list(order) for order in candidate_orders]
        if len(candidate_orders) == 2
        else []
    )
    source_contributions = _ec_source_contributions(
        first_order=first_order,
        second_order=second_order,
        combination=chosen_combination,
        phase=chosen_phase,
    )

    vertex = {
        "key": reference_key,
        "fields": list(reference.fields),
        "coefficient": coefficient,
        "source_orders": {
            "first": list(first_order),
            "second": second_order_payload,
            "combination": chosen_combination,
            "contributions": source_contributions,
        },
        "source_contributions": source_contributions,
        "heads": [coefficient],
        "rule": feynpy_expression.to_canonical_string(),
    }
    row = {
        "feynrules_key": reference_key,
        "feynrules_id": reference.identifier,
        "fields": list(reference.fields),
        "coefficient": coefficient,
        "feynpy_partner_key": rule.partner_key,
        "feynpy_source_orders": [
            list(order)
            for order in (first_order, second_order)
            if order is not None
        ],
        "tested_identical_leg_mappings": tested_identical_leg_mappings,
        "feynpy_source_contributions": source_contributions,
        "chosen_combination": chosen_combination,
        "chosen_candidate": chosen_candidate,
        "projector_head": projector_head,
        "canonical_difference": canonical_difference.to_canonical_string(),
        "status": status,
        "filtered_feynrules_heads": list(_alpha_heads_in_expression(filtered_feynrules)),
        "reconstructed_feynpy_heads": list(_alpha_heads_in_expression(feynpy_expression)),
        "feynpy_expression": feynpy_expression.to_canonical_string(),
        "filtered_feynrules_expression": feynrules_expression.to_canonical_string(),
        "tested_ordered_combinations": tested_combinations,
    }
    if status != "exact_match":
        row["canonical_nonzero_difference"] = canonical_difference.to_canonical_string()
    return row, vertex


def compare_ec_charge_conjugation_reconstruction(
    reference_path: Path = REFERENCE,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)
    references_by_key = _ec_reference_vertices_by_key(reference_path)

    rows = []
    vertices = []
    for reference_key, coefficient in sorted(
        _EC_PARTNER_PACKAGING_RULES,
        key=lambda item: (
            references_by_key[item[0]].identifier or 0,
            item[1],
        ),
    ):
        if coefficient not in _EC_CC_COEFFICIENTS:
            continue
        row, vertex = _ec_comparison_row(
            reference=references_by_key[reference_key],
            coefficient=coefficient,
            lagrangian=lagrangian,
            field_map=field_map,
        )
        rows.append(row)
        vertices.append(vertex)

    report = {
        "generated_on": date.today().isoformat(),
        "comparison_level": (
            "EC charge-conjugation four-fermion sidecar comparison. The six "
            "problematic FeynRules rows are split by alphaEc coefficient head, "
            "the FeynPy charge-conjugation partner is reconstructed in the "
            "FeynRules external-field order, and each coefficient sector is "
            "compared independently in a compact PL/PR fermion-flow basis."
        ),
        "summary": {
            "coefficient_sectors": len(rows),
            "exact_matches": sum(row["status"] == "exact_match" for row in rows),
            "wrong_combination_matches": sum(
                sum(
                    payload["matches"]
                    for candidate, payload in row[
                        "tested_ordered_combinations"
                    ].items()
                    if candidate != row["chosen_candidate"]
                )
                for row in rows
            ),
            "duplicate_leg_assignment_sectors": sum(
                bool(row["tested_identical_leg_mappings"]) for row in rows
            ),
        },
        "vertices": rows,
    }
    return report, vertices

__all__ = [name for name in globals() if not name.startswith("__")]
