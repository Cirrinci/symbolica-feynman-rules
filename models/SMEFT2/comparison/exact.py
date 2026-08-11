"""Exact symbolic row classification for SMEFT2 comparison reports."""

from .vertices import *


def _bosonic_exact_symbolic_rows(
    references: Iterable[FeynRulesVertex],
    bundle,
) -> dict[str, dict[str, str]]:
    bosonic_references = tuple(
        reference
        for reference in references
        if _exact_symbolic_family(reference.fields) == "BOSONIC"
    )
    if not bosonic_references:
        return {}

    report = compare_feynrules_bosonic_vertices(
        bundle.model.lagrangian(),
        bosonic_references,
        field_map=_comparison_field_map(bundle),
        feynpy_name_aliases=FIELD_NAME_MAP,
    )
    status_map = {
        "MATCH": "EXACT_MATCH",
        "MISMATCH": "EXACT_MISMATCH",
        "MISSING_FEYNPY": "EXACT_NO_LOCAL_SIGNATURE",
        "MISSING_FIELD_MAP": "EXACT_ERROR",
        "ERROR": "EXACT_ERROR",
    }
    return {
        _name_key(row.reference.fields): {
            "family": "BOSONIC",
            "status": status_map[row.status],
            "detail": row.detail,
        }
        for row in report.rows
    }


def _fermion_exact_symbolic_row(
    *,
    reference: FeynRulesVertex,
    local: LocalVertex | None,
    local_vertices: Iterable[LocalVertex],
    reference_heads: set[str],
    local_heads: set[str],
    head_count_status: str,
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, str] | None:
    family = _exact_symbolic_family(reference.fields)
    if family not in {"TWO_FERMION", "FOUR_FERMION"}:
        return None
    if local is None:
        return None

    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": "Could not infer external indices for fermion row.",
        }

    coefficients = tuple(
        sorted(
            head
            for head in reference_heads | local_heads
            if head.startswith("alpha")
        )
    )
    if not coefficients:
        return None

    try:
        local_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in reference.fields),
            simplify=True,
        )
        reference_rule = parse_smeft2_matter_rule(reference.rule)
        ec_convention_log: set[str] = set()
        comparisons = _compare_smeft2_canonical_coefficient_maps(
            local_rule,
            reference_rule,
            coefficients=coefficients,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
            convention_log=ec_convention_log,
        )
        pinned_partner_details = []
        unresolved_ec = []
        mismatched_ec = [
            coefficient
            for coefficient, comparison in comparisons.items()
            if coefficient.startswith("alphaEc") and not comparison.matches
        ]
        if mismatched_ec:
            reference_expression = parse_smeft2_matter_rule(reference.rule)
            for coefficient in mismatched_ec:
                feynrules_report = _canonical_report_for_coefficient_head(
                    reference_expression,
                    coefficient=coefficient,
                    external_indices=external_indices,
                    max_dummy_permutations=2_000_000,
                )
                partner_match = _ec_partner_packaging_comparison(
                    reference=reference,
                    coefficient=coefficient,
                    feynrules_report=feynrules_report,
                    local_vertices=local_vertices,
                    lagrangian=lagrangian,
                    field_map=field_map,
                    external_indices=external_indices,
                    max_dummy_permutations=2_000_000,
                )
                if partner_match is None:
                    unresolved_ec.append(coefficient)
                    continue
                comparisons[coefficient], detail = partner_match
                pinned_partner_details.append(detail)
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    if all(comparison.matches for comparison in comparisons.values()):
        if pinned_partner_details:
            return {
                "family": family,
                "status": "MATCH_MODULO_CC_PACKAGING",
                "detail": (
                    "Canonical tensor-monomial maps agree after pinned Ec "
                    "charge-conjugation partner packaging for "
                    f"{len(pinned_partner_details)} coefficient sector(s). "
                    "No phase or duplicate-leg symmetry was searched at "
                    "acceptance time: each transform came from the explicit "
                    "Ec packaging rule table. This row also sits on top of the "
                    "global evanescent charge-conjugation packaging convention "
                    f"(mode {_EC_CC_CONVENTION_MODE!r}, phase "
                    f"{_EC_CC_CONVENTION_PHASE:+d}). "
                    + " ".join(pinned_partner_details)
                    + " Raw head-count status before exact-proof "
                    f"classification was {head_count_status}."
                ),
            }
        if ec_convention_log:
            applied = ", ".join(sorted(ec_convention_log))
            return {
                "family": family,
                "status": "MATCH_MODULO_EC_CC_CONVENTION",
                "detail": (
                    "Canonical tensor-monomial maps agree for all "
                    f"{len(comparisons)} coefficient sector(s), but "
                    f"{len(ec_convention_log)} of them ({applied}) required the "
                    "global FeynPy/FeynRules evanescent charge-conjugation "
                    "packaging convention: FeynRules resolves `CC[...]` into a "
                    "crossed spinor flow with no residual charge-conjugation "
                    "matrix, while SMEFT2.py keeps two explicit `C` factors "
                    "pairing adjacent legs. The transform re-pairs the four "
                    "spinor arms in crossed order with a single overall sign "
                    f"({_EC_CC_CONVENTION_PHASE:+d}) from the antisymmetry of "
                    "`C`. Mode and sign are global constants, not per-row "
                    "fits. Raw head-count status before exact-proof "
                    f"classification was {head_count_status}."
                ),
            }
        return {
            "family": family,
            "status": "EXACT_MATCH",
            "detail": (
                "Canonical tensor-monomial maps agree for all "
                f"{len(comparisons)} coefficient sector(s); raw head-count "
                "status before exact-proof classification was "
                f"{head_count_status}."
            ),
        }

    mismatched = tuple(
        coefficient
        for coefficient, comparison in comparisons.items()
        if not comparison.matches
    )
    if (
        unresolved_ec
        and mismatched
        and all(coefficient.startswith("alphaEc") for coefficient in mismatched)
    ):
        return {
            "family": family,
            "status": "UNRESOLVED_CC_PACKAGING",
            "detail": (
                "Direct same-signature canonical maps disagree for "
                f"{', '.join(mismatched)}. The comparison has no successful "
                "pinned Ec partner-packaging rule for "
                f"{', '.join(unresolved_ec)}, so the row remains unresolved "
                "rather than being accepted through a searched phase/symmetry "
                "choice. Raw head-count status before exact-proof "
                f"classification was {head_count_status}."
            ),
        }

    return {
        "family": family,
        "status": "EXACT_MISMATCH",
        "detail": (
            "Canonical tensor-monomial maps differ for coefficient sector(s): "
            + ", ".join(mismatched)
        ),
    }


def _weinberg_packaged_field_orders(
    reference_fields: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    if sorted(reference_fields) == ["Phi", "Phi", "lL", "lL"]:
        fermion_name = "lL"
        scalar_name = "Phi"
    elif sorted(reference_fields) == ["Phibar", "Phibar", "lLbar", "lLbar"]:
        fermion_name = "lLbar"
        scalar_name = "Phibar"
    else:
        return None

    fermion_slots = [
        slot for slot, name in enumerate(reference_fields) if name == fermion_name
    ]
    if len(fermion_slots) != 2:
        return None
    if any(
        name != scalar_name
        for slot, name in enumerate(reference_fields)
        if slot not in fermion_slots
    ):
        return None

    first_assignment = list(reference_fields)
    first_assignment[fermion_slots[0]] = "lLbar"
    first_assignment[fermion_slots[1]] = "lL"

    second_assignment = list(reference_fields)
    second_assignment[fermion_slots[0]] = "lL"
    second_assignment[fermion_slots[1]] = "lLbar"

    return tuple(first_assignment), tuple(second_assignment)


def _weinberg_cc_exact_symbolic_row(
    *,
    reference: FeynRulesVertex,
    reference_heads: set[str],
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, str] | None:
    family = _exact_symbolic_family(reference.fields)
    if family != "TWO_FERMION" or reference_heads != {"alphaWeinberg"}:
        return None

    packaged_orders = _weinberg_packaged_field_orders(reference.fields)
    if packaged_orders is None:
        return None

    external_indices = _external_index_set_from_fields(
        tuple(field_map[name] for name in reference.fields)
    )
    if external_indices is None:
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": "Could not infer external indices for Weinberg row.",
        }

    try:
        first_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[0]),
            simplify=True,
        )
        second_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in packaged_orders[1]),
            simplify=True,
        )
        # Swapping which external lepton is represented by the charge-conjugate
        # FeynPy leg swaps the two C-matrix spinor slots.  Since C is
        # antisymmetric, the same-chirality FeynRules rule corresponds to the
        # antisymmetrized packaged local rule.
        local_rule = (first_rule - second_rule).cancel().expand()
        reference_rule = parse_smeft2_matter_rule(
            reference.rule,
            projector_as_dirac_c=True,
        )
        comparisons = _compare_smeft2_canonical_coefficient_maps(
            local_rule,
            reference_rule,
            coefficients=("alphaWeinberg",),
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "family": family,
            "status": "EXACT_ERROR",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    comparison = comparisons["alphaWeinberg"]
    if comparison.matches:
        return {
            "family": family,
            "status": "MATCH_MODULO_CC_PACKAGING",
            "detail": (
                "Weinberg charge-conjugation packaging match (sign pinned): the "
                "same-chirality FeynRules row equals the antisymmetrized "
                "FeynPy mixed `lLbar,lL` assignment pair `FeynPy(lLbar,lL) - "
                "FeynPy(lL,lLbar)`, with `ProjM/ProjP` mapped to the "
                "antisymmetric Dirac charge-conjugation tensor. The relative "
                "minus sign is derived from the antisymmetry of the "
                "charge-conjugation matrix, not searched, so the canonical-map "
                "equality is exact modulo the explicitly tracked CC packaging."
            ),
        }

    return {
        "family": family,
        "status": "EXACT_MISMATCH",
        "detail": (
            "Weinberg charge-conjugation packaging canonical maps differ for "
            "`alphaWeinberg`."
        ),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
