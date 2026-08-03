"""Local/reference vertex indexing and row-status helpers."""

from .charge_conjugation import *


@dataclass(frozen=True)
class LocalVertex:
    key: str
    signature: tuple[str, ...]
    local_names: tuple[str, ...]
    feynpy_names: tuple[str, ...]
    arity: int
    term_count: int
    sectors: tuple[str, ...]
    heads: tuple[str, ...]
    head_counts: tuple[tuple[str, int], ...]
    rule: str


def _comparison_field_map(bundle) -> dict[str, object]:
    field_map = {}
    for local_name, reference_name in FIELD_NAME_MAP.items():
        if local_name.endswith(".bar"):
            field_map[reference_name] = bundle.fields[local_name[:-4]].bar
        else:
            field_map[reference_name] = bundle.fields[local_name]
    return field_map


def _exact_symbolic_family(fields: Iterable[str]) -> str:
    fermion_count = sum(name in REFERENCE_FERMION_NAMES for name in fields)
    return {
        0: "BOSONIC",
        2: "TWO_FERMION",
        4: "FOUR_FERMION",
    }.get(fermion_count, "UNCLASSIFIED")


def _unsupported_exact_symbolic_detail(family: str) -> str:
    return {
        "TWO_FERMION": (
            "Exact symbolic comparison is enabled for shared SMEFT2 two-fermion "
            "rows; this row has no literal local signature or falls outside that "
            "shared-signature layer."
        ),
        "FOUR_FERMION": (
            "Exact symbolic comparison is enabled for shared SMEFT2 four-fermion "
            "rows; this row has no literal local signature or falls outside that "
            "shared-signature layer."
        ),
        "UNCLASSIFIED": (
            "Exact symbolic comparison is not enabled for this field-content class."
        ),
        "BOSONIC": (
            "Bosonic exact symbolic comparison should have been attempted for this row."
        ),
    }[family]


def _normalize_local_name(name: str) -> str:
    try:
        return FIELD_NAME_MAP[name]
    except KeyError as exc:
        raise ValueError(f"No FeynRules name mapping for local field {name!r}") from exc


def _parameter_head_counts_from_text(
    text: str,
    parameter_names: Iterable[str],
) -> dict[str, int]:
    counts = Counter(re.findall(r"\balpha[A-Za-z0-9]+(?=\[|\(|\b)", text))
    for name in parameter_names:
        if name.startswith("alpha"):
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
            counts[name] += len(
                re.findall(
                    rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                    text,
                )
            )
    return dict(sorted((head, count) for head, count in counts.items() if count))


def _parameter_heads_from_text(text: str, parameter_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(_parameter_head_counts_from_text(text, parameter_names))


# Field metadata index kinds map onto the canonical index groups. Generation
# (flavor) indices are carried on the ``cof(3)`` representation just like color
# fundamentals, so they are kept fixed in the ``color_fund`` group; their labels
# (``f1``, ``f2``, ...) never collide with the color labels (``c1``, ...).
_EXTERNAL_INDEX_GROUP_BY_KIND = {
    "lorentz": "lorentz",
    "color_adj": "color_adjoint",
    "color_fund": "color_fund",
    "spinor": "spinor",
    "weak_fund": "weak_fund",
    "weak_adj": "weak_adjoint",
    "generation": "color_fund",
}


def _external_index_set_from_fields(fields):
    """Return the canonical external-index set from field metadata.

    The external (leg) indices are fixed by the field content and their leg
    position: leg ``k`` contributes ``prefix{k}`` for each of its indices, which
    matches the labelling that :meth:`feynman_rule` emits for the same field
    order. Using metadata (rather than guessing free indices from a single term)
    is robust to term ordering, so canonical collection never accidentally
    renames a genuine external index and over-cancels distinct terms.
    """

    groups: dict[str, list] = {}
    for slot, field in enumerate(fields, start=1):
        base = field.field if hasattr(field, "field") else field
        for index in getattr(base, "indices", ()):
            kind = getattr(index, "kind", None)
            prefix = getattr(index, "prefix", None)
            group = _EXTERNAL_INDEX_GROUP_BY_KIND.get(kind)
            if group is None or not prefix:
                return None
            groups.setdefault(group, []).append(S(f"{prefix}{slot}"))
    return canonical_external_index_set(
        lorentz=tuple(groups.get("lorentz", ())),
        color_adjoint=tuple(groups.get("color_adjoint", ())),
        color_fund=tuple(groups.get("color_fund", ())),
        spinor=tuple(groups.get("spinor", ())),
        weak_fund=tuple(groups.get("weak_fund", ())),
        weak_adjoint=tuple(groups.get("weak_adjoint", ())),
    )


def _canonical_head_counts_from_rule(
    rule,
    parameter_names: Iterable[str],
    external_indices,
) -> dict[str, int]:
    """Coefficient-head multiplicity after canonical (algebraic) collection.

    The raw FeynPy rule keeps terms that only vanish once tensor identities are
    applied (e.g. ``epsilon^{mu nu rho sigma} B_rho B_sigma = 0`` from a double
    covariant-derivative expansion). Counting heads from the raw text therefore
    reports spurious coefficient heads (``alphaEuH``/``alphaEdH``/``alphaEeH``)
    that carry an identically-zero coefficient. Canonicalizing the rule
    term-by-term and collecting like terms drops those zero coefficients, so the
    surviving heads reflect the genuine algebraic operator content.
    """

    expression = rule.cancel().expand()
    monomial_map = canonical_tensor_monomial_map(
        expression,
        external_indices=external_indices,
    )
    canonical_text = " ".join(
        coefficient.to_canonical_string() for coefficient in monomial_map.values()
    )
    return _parameter_head_counts_from_text(canonical_text, parameter_names)


def _reference_heads(
    reference: FeynRulesVertex,
    parameter_names: Iterable[str],
) -> tuple[str, ...]:
    return _parameter_heads_from_text(reference.rule, parameter_names)


def _reference_head_counts(
    reference: FeynRulesVertex,
    parameter_names: Iterable[str],
) -> dict[str, int]:
    return _parameter_head_counts_from_text(reference.rule, parameter_names)


def _head_count_delta(
    reference_counts: dict[str, int],
    local_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    delta = {}
    for head in sorted(set(reference_counts) | set(local_counts)):
        reference_count = reference_counts.get(head, 0)
        local_count = local_counts.get(head, 0)
        if reference_count != local_count:
            delta[head] = {
                "reference": reference_count,
                "feynpy": local_count,
            }
    return delta


def _benign_head_count_delta_reasons(
    key: str,
    head_count_delta: dict[str, dict[str, int]],
) -> dict[str, str]:
    reasons = {}
    for head, counts in head_count_delta.items():
        reason = BENIGN_HEAD_COUNT_DELTAS.get((key, head))
        if reason is None:
            continue
        if counts["feynpy"] <= counts["reference"]:
            continue
        reasons[head] = reason
    return reasons


def _exact_symbolic_head_count_delta_reasons(
    *,
    exact_symbolic_status: str,
    head_count_delta: dict[str, dict[str, int]],
    existing_reasons: dict[str, str],
) -> dict[str, str]:
    reason = {
        "EXACT_MATCH": EXACT_SYMBOLIC_CANONICAL_EQUIVALENCE,
        "MATCH_MODULO_CC_PACKAGING": PINNED_CC_CANONICAL_EQUIVALENCE,
    }.get(exact_symbolic_status)
    if reason is None:
        return {}
    return {
        head: reason
        for head in head_count_delta
        if head not in existing_reasons
    }


def _head_count_status(
    *,
    has_local_signature: bool,
    head_count_delta: dict[str, dict[str, int]],
    benign_reasons: dict[str, str],
) -> str:
    if not has_local_signature:
        return "NO_LOCAL_SIGNATURE"
    if not head_count_delta:
        return "COUNT_MATCH"
    if len(benign_reasons) == len(head_count_delta):
        return "COUNT_BENIGN_EXPANSION"
    if benign_reasons:
        return "COUNT_MIXED_BENIGN_AND_UNEXPLAINED"
    return "COUNT_MISMATCH"


def _canonical_map_external_indices(
    fields: tuple[str, ...],
    *,
    field_map: dict[str, object],
) -> frozenset[tuple[str, str]] | None:
    if _exact_symbolic_family(fields) != "BOSONIC":
        return None

    grouped: dict[str, list[object]] = {}
    for slot, name in enumerate(fields, start=1):
        field = field_map.get(name)
        if field is None:
            return None
        base = field.field if hasattr(field, "field") else field
        for index in getattr(base, "indices", ()):
            kind = getattr(index, "kind", None)
            group = CANONICAL_EXTERNAL_INDEX_GROUP_BY_KIND.get(kind)
            prefix = getattr(index, "prefix", None)
            if group is None or not prefix:
                return None
            grouped.setdefault(group, []).append(S(f"{prefix}{slot}"))

    return canonical_external_index_set(
        lorentz=tuple(grouped.get("lorentz", ())),
        color_adjoint=tuple(grouped.get("color_adjoint", ())),
        color_fund=tuple(grouped.get("color_fund", ())),
        spinor=tuple(grouped.get("spinor", ())),
        weak_fund=tuple(grouped.get("weak_fund", ())),
        weak_adjoint=tuple(grouped.get("weak_adjoint", ())),
    )


def _canonical_map_coefficients(
    reference_heads: set[str],
    local_heads: set[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            head
            for head in reference_heads & local_heads
            if head.startswith("alpha")
        )
    )


def _first_canonical_map_item(mapping) -> dict[str, str] | None:
    if not mapping:
        return None
    key = next(iter(mapping))
    return {
        "monomial": repr(key),
        "coefficient": mapping[key].cancel().expand().to_canonical_string(),
    }


def _first_coefficient_mismatch(mapping) -> dict[str, str] | None:
    if not mapping:
        return None
    key = next(iter(mapping))
    feynpy_coefficient, feynrules_coefficient = mapping[key]
    return {
        "monomial": repr(key),
        "feynpy_coefficient": feynpy_coefficient.cancel().expand().to_canonical_string(),
        "feynrules_coefficient": feynrules_coefficient.cancel().expand().to_canonical_string(),
    }


def _canonical_map_diagnostic(
    *,
    reference: FeynRulesVertex,
    local: LocalVertex | None,
    reference_heads: set[str],
    local_heads: set[str],
    lagrangian,
    field_map: dict[str, object],
) -> dict[str, object]:
    external_indices = _canonical_map_external_indices(
        reference.fields,
        field_map=field_map,
    )
    if local is None or external_indices is None:
        return {
            "status": "CANONICAL_MAP_UNSUPPORTED",
            "coefficients": {},
            "error": "",
        }

    coefficients = _canonical_map_coefficients(reference_heads, local_heads)
    if not coefficients:
        return {
            "status": "CANONICAL_MAP_UNSUPPORTED",
            "coefficients": {},
            "error": "",
        }

    try:
        local_rule = lagrangian.feynman_rule(
            *(field_map[name] for name in reference.fields),
            simplify=True,
        )
        comparisons = compare_canonical_coefficient_maps(
            local_rule,
            reference.rule,
            coefficients=coefficients,
            external_indices=external_indices,
            max_dummy_permutations=2_000_000,
        )
    except Exception as exc:  # pragma: no cover - reported in JSON/Markdown.
        return {
            "status": "CANONICAL_MAP_ERROR",
            "coefficients": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    coefficient_payload = {}
    for coefficient, comparison in comparisons.items():
        coefficient_payload[coefficient] = {
            "matches": comparison.matches,
            "feynpy_raw_terms": comparison.feynpy_raw_terms,
            "feynrules_raw_terms": comparison.feynrules_raw_terms,
            "feynpy_canonical_terms": comparison.feynpy_canonical_terms,
            "feynrules_canonical_terms": comparison.feynrules_canonical_terms,
            "feynpy_only_count": len(comparison.feynpy_only),
            "feynrules_only_count": len(comparison.feynrules_only),
            "coefficient_mismatch_count": len(comparison.coefficient_mismatches),
            "first_feynpy_only": _first_canonical_map_item(comparison.feynpy_only),
            "first_feynrules_only": _first_canonical_map_item(
                comparison.feynrules_only
            ),
            "first_coefficient_mismatch": _first_coefficient_mismatch(
                comparison.coefficient_mismatches
            ),
        }

    return {
        "status": (
            "CANONICAL_MAP_MATCH"
            if all(comparison.matches for comparison in comparisons.values())
            else "CANONICAL_MAP_MISMATCH"
        ),
        "coefficients": coefficient_payload,
        "error": "",
    }


def _local_vertices(parameter_names: Iterable[str]) -> tuple[LocalVertex, ...]:
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    rows: list[LocalVertex] = []
    for signature in lagrangian.vertex_signatures():
        if not 3 <= signature.arity <= 6:
            continue
        rule = lagrangian.feynman_rule(*signature.fields, simplify=True)
        rule_text = rule.cancel().expand().to_canonical_string()
        # Raw-text multiplicities feed the (apples-to-apples) raw head-count
        # diagnostic against the FeynRules reference text.
        raw_head_counts = _parameter_head_counts_from_text(rule_text, parameter_names)
        # The coefficient-head *set* (which drives the operator-content match)
        # is taken after canonical collection so that heads whose coefficient is
        # only zero by a tensor identity are not counted as genuine content.
        external_indices = _external_index_set_from_fields(signature.fields)
        if external_indices is None:
            canonical_head_counts = raw_head_counts
        else:
            try:
                canonical_head_counts = _canonical_head_counts_from_rule(
                    rule, parameter_names, external_indices
                )
            except Exception:
                canonical_head_counts = raw_head_counts
        local_names = tuple(_normalize_local_name(name) for name in signature.names)
        normalized_signature = tuple(sorted(local_names))
        rows.append(
            LocalVertex(
                key=_name_key(local_names),
                signature=normalized_signature,
                local_names=local_names,
                feynpy_names=signature.names,
                arity=signature.arity,
                term_count=signature.term_count,
                sectors=signature.sectors,
                heads=tuple(canonical_head_counts),
                head_counts=tuple(raw_head_counts.items()),
                rule=rule_text,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.arity, row.key)))


def _reason_for_status(status: str) -> str:
    return {
        "SHARED_HEADS_MATCH": (
            "Shared field multiset and identical coefficient-head set at the "
            "operator-content comparison level."
        ),
        "MISSING_SIGNATURE_OMITTED_DERIVATIVE_SECTORS": (
            "Reference signature is driven by derivative-sector coefficient "
            "families that are explicitly not lowered in local SMEFT2."
        ),
        "MISSING_SIGNATURE_WEINBERG_PACKAGING": (
            "Reference has same-chirality Weinberg signatures while the local "
            "model packages that operator with mixed bar/conjugation signatures."
        ),
        "MISSING_SIGNATURE": "Reference signature is absent from the local FeynPy output.",
        "SHARED_MISSING_OMITTED_HEADS": (
            "Shared field multiset, but the reference contains omitted "
            "derivative-sector coefficient heads."
        ),
        "SHARED_MISSING_OMITTED_HEADS_PLUS_LOCAL_EXTRA": (
            "Local rule misses omitted derivative-sector heads from the reference "
            "and also has additional local heads."
        ),
        "SHARED_LOCAL_PP_EXTRA": (
            "Local translation has extra pp-coefficient heads not present in the "
            "FeynRules reference for this signature."
        ),
        "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH": (
            "Difference is concentrated in charge-conjugated operator packaging."
        ),
        "SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH": (
            "Operator content matches once the reference's charge-conjugated "
            "four-fermion head is credited to the FeynPy vertex that carries the "
            "same operator under the charge-conjugate (bar-flipped) bilinear "
            "packaging."
        ),
        "MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING": (
            "Reference Weinberg signature is matched by the FeynPy vertex that "
            "carries the same operator under the charge-conjugate (bar-flipped) "
            "packaging."
        ),
        "SHARED_MIXED_OPERATOR_CONTENT": (
            "Shared field multiset, but both sides contain coefficient heads absent "
            "from the other."
        ),
        "SHARED_REFERENCE_EXTRA_HEADS": (
            "Shared field multiset, but FeynRules has coefficient heads absent "
            "from local FeynPy."
        ),
        "SHARED_LOCAL_EXTRA_HEADS": (
            "Shared field multiset, but local FeynPy has coefficient heads absent "
            "from FeynRules."
        ),
        "FEYNPY_ONLY_WEINBERG_PACKAGING": (
            "Local model emits a Weinberg bar/conjugation signature not present "
            "as a separate FeynRules signature."
        ),
        "FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING": (
            "Local model emits a bar/charge-conjugation packaging not present "
            "as a separate FeynRules signature."
        ),
        "FEYNPY_ONLY_SIGNATURE": "Local signature is absent from the FeynRules reference.",
        "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER": (
            "FeynPy-only signature that is the charge-conjugate (bar-flipped) "
            "packaging of a FeynRules reference operator; it is cross-linked to "
            "that reference row and is not an unexplained residual."
        ),
        "FEYNPY_ONLY_ALGEBRAICALLY_ZERO": (
            "FeynPy-only signature whose canonical coefficient-head set is empty: "
            "the rule cancels to zero under canonical tensor identities, so it is "
            "a zero-signature artifact rather than residual operator content."
        ),
    }[status]


def _shared_status(reference_heads: set[str], local_heads: set[str]) -> str:
    if reference_heads == local_heads:
        return "SHARED_HEADS_MATCH"

    reference_extra = reference_heads - local_heads
    local_extra = local_heads - reference_heads
    if reference_extra & OMITTED_COEFFICIENT_HEADS:
        return (
            "SHARED_MISSING_OMITTED_HEADS_PLUS_LOCAL_EXTRA"
            if local_extra
            else "SHARED_MISSING_OMITTED_HEADS"
        )
    if reference_extra or local_extra:
        differing = reference_extra | local_extra
        if any(head.startswith("alphaEc") for head in differing):
            return "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH"
        if (
            local_extra
            and not reference_extra
            and all(head.endswith("pp") for head in local_extra)
        ):
            return "SHARED_LOCAL_PP_EXTRA"
        if reference_extra and local_extra:
            return "SHARED_MIXED_OPERATOR_CONTENT"
        if reference_extra:
            return "SHARED_REFERENCE_EXTRA_HEADS"
        return "SHARED_LOCAL_EXTRA_HEADS"
    raise AssertionError("unreachable shared comparison state")


def _missing_reference_status(reference_heads: set[str]) -> str:
    if "alphaWeinberg" in reference_heads:
        return "MISSING_SIGNATURE_WEINBERG_PACKAGING"
    if reference_heads & OMITTED_COEFFICIENT_HEADS:
        return "MISSING_SIGNATURE_OMITTED_DERIVATIVE_SECTORS"
    return "MISSING_SIGNATURE"


def _feynpy_only_status(local_heads: set[str]) -> str:
    if "alphaWeinberg" in local_heads:
        return "FEYNPY_ONLY_WEINBERG_PACKAGING"
    if "alphaOHud" in local_heads or any(head.startswith("alphaEc") for head in local_heads):
        return "FEYNPY_ONLY_CHARGE_CONJUGATION_OR_BAR_PACKAGING"
    return "FEYNPY_ONLY_SIGNATURE"

__all__ = [name for name in globals() if not name.startswith("__")]
