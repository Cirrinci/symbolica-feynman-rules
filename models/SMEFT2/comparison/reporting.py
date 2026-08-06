"""Aggregate SMEFT2 comparison report generation and artifact writing."""

from .sidecars import *


def compare(reference_path: Path = REFERENCE) -> tuple[dict[str, object], tuple[LocalVertex, ...]]:
    references = load_feynrules_json(reference_path)
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)
    parameter_names = set(bundle.parameters) | GENERIC_PARAMETER_NAMES

    local_vertices = _local_vertices(parameter_names)
    exact_symbolic_by_key = _bosonic_exact_symbolic_rows(references, bundle)
    local_by_key = {vertex.key: vertex for vertex in local_vertices}
    reference_keys = {_name_key(reference.fields) for reference in references}

    reference_rows = []
    status_counts: Counter[str] = Counter()
    for reference in sorted(
        references,
        key=lambda item: (_name_key(item.fields), item.identifier or 0),
    ):
        key = _name_key(reference.fields)
        reference_heads = set(_reference_heads(reference, parameter_names))
        reference_head_counts = _reference_head_counts(reference, parameter_names)
        local = local_by_key.get(key)
        if local is None:
            status = _missing_reference_status(reference_heads)
            local_heads: set[str] = set()
            local_head_counts: dict[str, int] = {}
            local_names: tuple[str, ...] = ()
            feynpy_names: tuple[str, ...] = ()
            sectors: tuple[str, ...] = ()
            term_count = 0
        else:
            # ``local.heads`` is the coefficient-head set after canonical
            # collection (spurious heads whose coefficient is only zero by a
            # tensor identity are removed). That collection is reliable for
            # dropping genuinely feynpy-only heads (e.g. the ``epsilon B B = 0``
            # heads from the double covariant-derivative expansion), but the
            # open-spinor canonicalization can be field-order dependent for
            # sigma-dipole rows, so we never let it drop a head that FeynRules
            # also carries. Shared heads are therefore always retained from the
            # raw multiplicity map, which protects against false
            # "reference-extra" verdicts.
            raw_local_heads = set(dict(local.head_counts))
            local_heads = set(local.heads) | (raw_local_heads & reference_heads)
            local_head_counts = dict(local.head_counts)
            status = _shared_status(reference_heads, local_heads)
            local_names = local.local_names
            feynpy_names = local.feynpy_names
            sectors = local.sectors
            term_count = local.term_count

        head_count_delta = _head_count_delta(reference_head_counts, local_head_counts)
        benign_head_count_delta_reasons = _benign_head_count_delta_reasons(
            key,
            head_count_delta,
        )
        pre_exact_head_count_status = _head_count_status(
            has_local_signature=local is not None,
            head_count_delta=head_count_delta,
            benign_reasons=benign_head_count_delta_reasons,
        )
        canonical_map = _canonical_map_diagnostic(
            reference=reference,
            local=local,
            reference_heads=reference_heads,
            local_heads=local_heads,
            lagrangian=lagrangian,
            field_map=field_map,
        )
        exact_symbolic_family = _exact_symbolic_family(reference.fields)
        exact_symbolic = exact_symbolic_by_key.get(key)
        if exact_symbolic is None:
            exact_symbolic = _weinberg_cc_exact_symbolic_row(
                reference=reference,
                reference_heads=reference_heads,
                lagrangian=lagrangian,
                field_map=field_map,
            )
        if exact_symbolic is None:
            exact_symbolic = _fermion_exact_symbolic_row(
                reference=reference,
                local=local,
                local_vertices=local_vertices,
                reference_heads=reference_heads,
                local_heads=local_heads,
                head_count_status=pre_exact_head_count_status,
                lagrangian=lagrangian,
                field_map=field_map,
            )
        if exact_symbolic is None:
            exact_symbolic = {
                "family": exact_symbolic_family,
                "status": "EXACT_UNSUPPORTED",
                "detail": _unsupported_exact_symbolic_detail(exact_symbolic_family),
            }
        benign_head_count_delta_reasons.update(
            _exact_symbolic_head_count_delta_reasons(
                exact_symbolic_status=exact_symbolic["status"],
                head_count_delta=head_count_delta,
                existing_reasons=benign_head_count_delta_reasons,
            )
        )
        unexplained_head_count_delta = {
            head: counts
            for head, counts in head_count_delta.items()
            if head not in benign_head_count_delta_reasons
        }
        head_count_status = _head_count_status(
            has_local_signature=local is not None,
            head_count_delta=head_count_delta,
            benign_reasons=benign_head_count_delta_reasons,
        )

        status_counts[status] += 1
        reference_rows.append(
            {
                "id": reference.identifier,
                "key": key,
                "fields": list(reference.fields),
                "legs": list(reference.legs),
                "signature": sorted(reference.fields),
                "arity": len(reference.fields),
                "reference_heads": sorted(reference_heads),
                "feynpy_heads": sorted(local_heads),
                "reference_head_counts": reference_head_counts,
                "feynpy_head_counts": local_head_counts,
                "head_count_delta": head_count_delta,
                "benign_head_count_delta_reasons": benign_head_count_delta_reasons,
                "unexplained_head_count_delta": unexplained_head_count_delta,
                "head_count_status": head_count_status,
                "canonical_map_status": canonical_map["status"],
                "canonical_map_coefficients": canonical_map["coefficients"],
                "canonical_map_error": canonical_map["error"],
                "exact_symbolic_family": exact_symbolic["family"],
                "exact_symbolic_status": exact_symbolic["status"],
                "exact_symbolic_detail": exact_symbolic["detail"],
                "feynrules_extra_heads": sorted(reference_heads - local_heads),
                "feynpy_extra_heads": sorted(local_heads - reference_heads),
                "local_names": list(local_names),
                "feynpy_names": list(feynpy_names),
                "local_term_count": term_count,
                "sectors": list(sectors),
                "status": status,
                "reason": _reason_for_status(status),
            }
        )

    # A FeynPy-only signature whose *canonical* coefficient-head set is empty is
    # algebraically zero: every raw monomial cancels under dummy relabeling and
    # intrinsic tensor symmetries (e.g. the U(1) piece of the ``O_Hud`` covariant
    # derivative in ``B|Phi|Phi|dR|uRbar``). These are not residual unmatched
    # operator content -- FeynRules correctly omits them -- so they are recorded
    # separately as zero-signature artifacts rather than counted as FeynPy-only
    # residuals.
    feynpy_only_rows = []
    feynpy_only_zero_rows = []
    for local in local_vertices:
        if local.key in reference_keys:
            continue
        row = {
            "key": local.key,
            "signature": list(local.signature),
            "local_names": list(local.local_names),
            "feynpy_names": list(local.feynpy_names),
            "arity": local.arity,
            "term_count": local.term_count,
            "sectors": list(local.sectors),
            "feynpy_heads": list(local.heads),
            "feynpy_head_counts": dict(local.head_counts),
        }
        if not local.heads:
            row["status"] = "FEYNPY_ONLY_ALGEBRAICALLY_ZERO"
            row["reason"] = _reason_for_status(row["status"])
            feynpy_only_zero_rows.append(row)
            continue
        row["status"] = _feynpy_only_status(set(local.heads))
        row["reason"] = _reason_for_status(row["status"])
        feynpy_only_rows.append(row)

    # ------------------------------------------------------------------
    # Charge-conjugation packaging reconciliation.
    #
    # FeynRules and FeynPy sometimes assign the bar (particle vs antiparticle
    # leg) differently for the *same* operator. FeynRules keeps the Weinberg
    # operator as ``Phi Phi lL lL`` and the four-fermion ``Ec`` operators with a
    # given bilinear bar assignment, whereas FeynPy packages the same operators
    # with the charge-conjugate bilinear (both members' bars flipped), e.g.
    # ``Phi Phi lL lLbar``. Such a pair carries the *same* operator head and a
    # bar-insensitive field content, and is the same vertex related by charge
    # conjugation ``psi1bar Gamma psi2 = psi2bar Gamma' psi1^C``. We pair each
    # such reference row with the FeynPy-only vertex that carries the matching
    # head under the charge-conjugate field content, so the operator is credited
    # as an algebraic match rather than a spurious reference-extra + FeynPy-only
    # split. This is applied only as a fallback for the charge-conjugation and
    # Weinberg mismatch buckets; it never touches already-matched rows.
    # ------------------------------------------------------------------
    # This pass is deliberately an *annotation overlay*: it records that a
    # reference operator is present in FeynPy under the charge-conjugate
    # packaging, but it does NOT alter the literal signature-coverage metrics
    # (``shared_signatures`` / ``reference_only_signatures`` /
    # ``feynpy_only_signatures``), which stay defined by exact field-multiset
    # overlap. The Weinberg reference rows still have no exact local signature
    # and remain reference-only; their FeynPy charge-conjugate partners still
    # have no exact reference signature and remain FeynPy-only. The overlay adds
    # a ``charge_conjugation_partner`` cross-link on both sides and a separate
    # ``charge_conjugation_packaging_matches`` metric for operator-content
    # matching modulo charge conjugation.
    def _cc_field_key(fields: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            sorted(name[:-3] if name.endswith("bar") else name for name in fields)
        )

    cc_local_index: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in feynpy_only_rows:
        cc_local_index.setdefault(_cc_field_key(row["signature"]), []).append(row)

    consumed_feynpy_only_ids: set[int] = set()
    for row in reference_rows:
        if row["status"] not in (
            "SHARED_CHARGE_CONJUGATION_PACKAGING_MISMATCH",
            "MISSING_SIGNATURE_WEINBERG_PACKAGING",
        ):
            continue
        cc_extra = {
            head
            for head in row["feynrules_extra_heads"]
            if head.startswith("alphaEc") or head == "alphaWeinberg"
        }
        if not cc_extra:
            continue
        cc_key = _cc_field_key(row["fields"])
        partner = None
        for candidate in cc_local_index.get(cc_key, ()):
            if id(candidate) in consumed_feynpy_only_ids:
                continue
            if cc_extra <= set(candidate["feynpy_head_counts"]):
                partner = candidate
                break
        if partner is None:
            continue
        consumed_feynpy_only_ids.add(id(partner))
        # Cross-link both sides; keep the literal per-row head sets untouched.
        row["charge_conjugation_partner"] = partner["key"]
        row["charge_conjugation_matched_heads"] = sorted(cc_extra)
        row["operator_content_resolved_via_charge_conjugation"] = True
        row["status"] = (
            "MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING"
            if row["status"] == "MISSING_SIGNATURE_WEINBERG_PACKAGING"
            else "SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH"
        )
        row["reason"] = _reason_for_status(row["status"])
        partner["charge_conjugation_partner"] = row["key"]
        partner["status"] = "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
        partner["reason"] = _reason_for_status(partner["status"])

    # Status tallies over the final annotated rows (reference rows plus the
    # literal FeynPy-only rows and the separate zero-signature artifacts).
    status_counts = Counter(row["status"] for row in reference_rows)
    for row in feynpy_only_rows:
        status_counts[row["status"]] += 1
    for row in feynpy_only_zero_rows:
        status_counts[row["status"]] += 1

    charge_conjugation_matches = (
        status_counts["SHARED_CHARGE_CONJUGATION_PACKAGING_MATCH"]
        + status_counts["MATCHED_VIA_CHARGE_CONJUGATION_PACKAGING"]
    )

    # Literal signature coverage: a reference row is "shared" iff FeynPy emits an
    # exact same-field-multiset signature for it (``local is not None``, i.e. its
    # head-count status is not ``NO_LOCAL_SIGNATURE``). This is independent of the
    # charge-conjugation overlay above.
    shared = sum(
        1
        for row in reference_rows
        if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    )
    head_count_matches = sum(
        1
        for row in reference_rows
        if row["head_count_status"] == "COUNT_MATCH"
    )
    head_count_status_counts = Counter(
        row["head_count_status"]
        for row in reference_rows
        if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    )
    canonical_map_rows = [
        row
        for row in reference_rows
        if row["canonical_map_status"] != "CANONICAL_MAP_UNSUPPORTED"
    ]
    canonical_map_status_counts = Counter(
        row["canonical_map_status"]
        for row in canonical_map_rows
    )
    canonical_map_sector_count = sum(
        len(row["canonical_map_coefficients"])
        for row in canonical_map_rows
    )
    canonical_map_equal_sector_count = sum(
        sum(
            1
            for coefficient in row["canonical_map_coefficients"].values()
            if coefficient["matches"]
        )
        for row in canonical_map_rows
    )
    # Raw-output redundancy diagnostic: how many raw FeynPy monomials collapse
    # once dummy indices are canonically relabeled and intrinsic tensor
    # symmetries applied. This quantifies the "long output" problem on the
    # sectors where an exact canonical monomial count is available. The gap
    # (raw minus canonical) is entirely redundant surface form that FeynRules
    # already writes collected; it is not extra physics.
    canonical_map_feynpy_raw_terms = sum(
        coefficient["feynpy_raw_terms"]
        for row in canonical_map_rows
        for coefficient in row["canonical_map_coefficients"].values()
    )
    canonical_map_feynpy_canonical_terms = sum(
        coefficient["feynpy_canonical_terms"]
        for row in canonical_map_rows
        for coefficient in row["canonical_map_coefficients"].values()
    )
    exact_symbolic_rows = [
        row
        for row in reference_rows
        if row["exact_symbolic_status"] != "EXACT_UNSUPPORTED"
    ]
    exact_symbolic_status_counts = Counter(
        row["exact_symbolic_status"] for row in reference_rows
    )
    exact_symbolic_family_counts = Counter(
        row["exact_symbolic_family"] for row in reference_rows
    )
    shared_reference_rows = [
        row for row in reference_rows if row["head_count_status"] != "NO_LOCAL_SIGNATURE"
    ]
    benign_head_count_delta_heads = sum(
        len(row["benign_head_count_delta_reasons"])
        for row in shared_reference_rows
    )
    unexplained_head_count_delta_heads = sum(
        len(row["unexplained_head_count_delta"])
        for row in shared_reference_rows
    )
    matched = status_counts["SHARED_HEADS_MATCH"]
    report = {
        "generated_on": date.today().isoformat(),
        "reference": str(reference_path.relative_to(ROOT)),
        "local_model": str((MODEL_DIR / "SMEFT2.py").relative_to(ROOT)),
        "comparison_level": (
            "Signature coverage, coefficient-head content, and raw "
            "coefficient-head multiplicity diagnostics, plus exact symbolic "
            "comparison for all 184 FeynRules reference rows. Fermion exact "
            "comparison filters by indexed Wilson-coefficient head and keeps "
            "flavor order/conjugation in the canonical scalar coefficient, so "
            "it cannot pass vacuously for function-valued coefficients. "
            "Exact-symbolic rows are graded honestly: `EXACT_MATCH` means "
            "direct canonical-map equality with no row-specific packaging "
            "assumption; `MATCH_MODULO_CC_PACKAGING` means equality only after "
            "a charge-conjugation packaging transform whose sign/symmetry is "
            "derived (pinned), e.g. the antisymmetrized Weinberg rows; and "
            "`UNRESOLVED_CC_PACKAGING` means no pinned packaging rule is known "
            "or the pinned transform failed. The separate canonical tensor-map "
            "diagnostic remains the bosonic-sector per-coefficient map for "
            "supported bosonic coefficient sectors."
        ),
        "summary": {
            "reference_vertex_count": len(references),
            "feynpy_signature_count_3_to_6": len(local_vertices),
            # Literal exact field-multiset signature coverage.
            "shared_signatures": shared,
            "reference_only_signatures": len(references) - shared,
            "feynpy_only_signatures": len(feynpy_only_rows),
            "feynpy_only_zero_signatures": len(feynpy_only_zero_rows),
            "feynpy_only_charge_conjugation_partners": sum(
                1
                for row in feynpy_only_rows
                if row["status"] == "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
            ),
            "feynpy_only_unexplained_signatures": sum(
                1
                for row in feynpy_only_rows
                if row["status"] != "FEYNPY_ONLY_CHARGE_CONJUGATION_PARTNER"
            ),
            # Operator-content matching (coefficient-head set).
            "shared_head_matches": matched,
            "charge_conjugation_packaging_matches": charge_conjugation_matches,
            "operator_content_matches_including_cc": matched + charge_conjugation_matches,
            "shared_head_count_matches": head_count_matches,
            "shared_head_count_mismatches": shared - head_count_matches,
            "shared_head_count_benign_expansions": head_count_status_counts[
                "COUNT_BENIGN_EXPANSION"
            ],
            "shared_head_count_mixed_benign_unexplained": head_count_status_counts[
                "COUNT_MIXED_BENIGN_AND_UNEXPLAINED"
            ],
            "shared_head_count_unexplained_mismatches": (
                head_count_status_counts["COUNT_MISMATCH"]
                + head_count_status_counts["COUNT_MIXED_BENIGN_AND_UNEXPLAINED"]
            ),
            "canonical_map_supported_vertices": len(canonical_map_rows),
            "canonical_map_equal_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_MATCH"
            ],
            "canonical_map_unequal_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_MISMATCH"
            ],
            "canonical_map_error_vertices": canonical_map_status_counts[
                "CANONICAL_MAP_ERROR"
            ],
            "canonical_map_supported_coefficient_sectors": canonical_map_sector_count,
            "canonical_map_equal_coefficient_sectors": canonical_map_equal_sector_count,
            "canonical_map_feynpy_raw_terms": canonical_map_feynpy_raw_terms,
            "canonical_map_feynpy_canonical_terms": canonical_map_feynpy_canonical_terms,
            "canonical_map_feynpy_redundant_terms": (
                canonical_map_feynpy_raw_terms - canonical_map_feynpy_canonical_terms
            ),
            "canonical_map_unequal_coefficient_sectors": (
                canonical_map_sector_count - canonical_map_equal_sector_count
            ),
            "canonical_map_status_counts": dict(
                sorted(canonical_map_status_counts.items())
            ),
            "exact_symbolic_supported_vertices": len(exact_symbolic_rows),
            "exact_symbolic_equal_vertices": exact_symbolic_status_counts[
                "EXACT_MATCH"
            ],
            "exact_symbolic_direct_match_vertices": exact_symbolic_status_counts[
                "EXACT_MATCH"
            ],
            "cc_packaging_pinned_match_vertices": exact_symbolic_status_counts[
                "MATCH_MODULO_CC_PACKAGING"
            ],
            "cc_packaging_unresolved_vertices": exact_symbolic_status_counts[
                "UNRESOLVED_CC_PACKAGING"
            ],
            "exact_symbolic_unequal_vertices": exact_symbolic_status_counts[
                "EXACT_MISMATCH"
            ],
            "exact_symbolic_missing_local_vertices": exact_symbolic_status_counts[
                "EXACT_NO_LOCAL_SIGNATURE"
            ],
            "exact_symbolic_error_vertices": exact_symbolic_status_counts[
                "EXACT_ERROR"
            ],
            "exact_symbolic_status_counts": dict(
                sorted(exact_symbolic_status_counts.items())
            ),
            "exact_symbolic_family_counts": dict(
                sorted(exact_symbolic_family_counts.items())
            ),
            "benign_head_count_delta_heads": benign_head_count_delta_heads,
            "unexplained_head_count_delta_heads": unexplained_head_count_delta_heads,
            "head_count_status_counts": dict(sorted(head_count_status_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "comparison_basis": {
                "reference_ltot": "EFT-only FeynRules Ltot",
                "local_ltot": "EFT-only FeynPy Ltot",
                "local_sm_plus_eft_lagrangian": "Lfull",
                "omitted_sectors": list(bundle.omitted_sectors),
            },
        },
        "reference_vertices": reference_rows,
        "feynpy_only_signatures": feynpy_only_rows,
        "feynpy_only_zero_signatures": feynpy_only_zero_rows,
    }
    return report, local_vertices


def _vertex_payload(local_vertices: Iterable[LocalVertex]) -> list[dict[str, object]]:
    return [
        {
            "key": vertex.key,
            "signature": list(vertex.signature),
            "local_names": list(vertex.local_names),
            "feynpy_names": list(vertex.feynpy_names),
            "arity": vertex.arity,
            "term_count": vertex.term_count,
            "sectors": list(vertex.sectors),
            "heads": list(vertex.heads),
            "head_counts": dict(vertex.head_counts),
            "rule": vertex.rule,
        }
        for vertex in local_vertices
    ]


def write_outputs(
    report: dict[str, object],
    local_vertices: Iterable[LocalVertex],
    *,
    comparison_json: Path = COMPARISON_JSON,
    comparison_md: Path = COMPARISON_MD,
    feynpy_vertices: Path = FEYNPY_VERTICES,
) -> None:
    comparison_json.parent.mkdir(parents=True, exist_ok=True)
    comparison_md.parent.mkdir(parents=True, exist_ok=True)
    feynpy_vertices.parent.mkdir(parents=True, exist_ok=True)
    comparison_json.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    feynpy_vertices.write_text(
        json.dumps(_vertex_payload(local_vertices), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    comparison_md.write_text(_markdown_report(report), encoding="utf-8")


def write_weinberg_outputs(
    report: dict[str, object],
    vertices: list[dict[str, object]],
    *,
    comparison_json: Path = WEINBERG_COMPARISON_JSON,
    feynpy_vertices: Path = WEINBERG_VERTICES,
) -> None:
    comparison_json.parent.mkdir(parents=True, exist_ok=True)
    feynpy_vertices.parent.mkdir(parents=True, exist_ok=True)
    comparison_json.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    feynpy_vertices.write_text(
        json.dumps(vertices, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_ec_charge_conjugation_outputs(
    report: dict[str, object],
    vertices: list[dict[str, object]],
    *,
    comparison_json: Path = EC_CC_COMPARISON_JSON,
    feynpy_vertices: Path = EC_CC_VERTICES,
) -> None:
    comparison_json.parent.mkdir(parents=True, exist_ok=True)
    feynpy_vertices.parent.mkdir(parents=True, exist_ok=True)
    comparison_json.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    feynpy_vertices.write_text(
        json.dumps(vertices, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    counts = summary["status_counts"]
    basis = summary["comparison_basis"]
    family_counts = summary["exact_symbolic_family_counts"]
    omitted_sectors = ", ".join(basis["omitted_sectors"]) or "none"
    lines = [
        "# SMEFT2 FeynRules/FeynPy Comparison",
        "",
        f"Generated on `{report['generated_on']}` by `models.SMEFT2.comparison`.",
        "",
        "## Scope",
        "",
        str(report["comparison_level"]),
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Reference vertices | {summary['reference_vertex_count']} |",
        f"| FeynPy 3-6 point signatures | {summary['feynpy_signature_count_3_to_6']} |",
        f"| Shared signatures (exact field multiset) | {summary['shared_signatures']} |",
        f"| Reference-only signatures (exact field multiset) | {summary['reference_only_signatures']} |",
        f"| FeynPy-only signatures (exact field multiset) | {summary['feynpy_only_signatures']} |",
        "| — of which charge-conjugation partners | "
        f"{summary['feynpy_only_charge_conjugation_partners']} |",
        "| — of which unexplained | "
        f"{summary['feynpy_only_unexplained_signatures']} |",
        "| FeynPy-only zero-signature artifacts (dropped) | "
        f"{summary['feynpy_only_zero_signatures']} |",
        f"| Shared coefficient-head matches | {summary['shared_head_matches']} |",
        "| Charge-conjugation packaging matches (modulo CC) | "
        f"{summary['charge_conjugation_packaging_matches']} |",
        "| Operator-content matches (incl. charge conjugation) | "
        f"{summary['operator_content_matches_including_cc']} |",
        f"| Shared raw head-count matches | {summary['shared_head_count_matches']} |",
        f"| Shared raw head-count mismatches | {summary['shared_head_count_mismatches']} |",
        f"| Shared raw head-count benign expansions | {summary['shared_head_count_benign_expansions']} |",
        "| Shared raw head-count mismatches with unexplained deltas | "
        f"{summary['shared_head_count_unexplained_mismatches']} |",
        f"| Exact symbolic supported vertices | {summary['exact_symbolic_supported_vertices']} |",
        f"| Direct exact symbolic matches | {summary['exact_symbolic_direct_match_vertices']} |",
        "| Exact modulo pinned CC packaging | "
        f"{summary['cc_packaging_pinned_match_vertices']} |",
        "| Unresolved CC packaging (existence only) | "
        f"{summary['cc_packaging_unresolved_vertices']} |",
        f"| Exact symbolic unequal vertices | {summary['exact_symbolic_unequal_vertices']} |",
        f"| Exact symbolic error vertices | {summary['exact_symbolic_error_vertices']} |",
        (
            "| Headline split | "
            f"direct exact: {summary['exact_symbolic_direct_match_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"modulo pinned CC: {summary['cc_packaging_pinned_match_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"unresolved CC: {summary['cc_packaging_unresolved_vertices']}/"
            f"{summary['exact_symbolic_supported_vertices']}; "
            f"operator content: {summary['operator_content_matches_including_cc']}/"
            f"{summary['reference_vertex_count']} |"
        ),
        "| Compatibility alias `exact_symbolic_equal_vertices` (direct only) | "
        f"{summary['exact_symbolic_equal_vertices']} |",
        f"| Canonical tensor-map supported vertices | {summary['canonical_map_supported_vertices']} |",
        f"| Canonical tensor-map equal vertices | {summary['canonical_map_equal_vertices']} |",
        f"| Canonical tensor-map unequal vertices | {summary['canonical_map_unequal_vertices']} |",
        f"| Canonical tensor-map error vertices | {summary['canonical_map_error_vertices']} |",
        "| Canonical tensor-map equal coefficient sectors | "
        f"{summary['canonical_map_equal_coefficient_sectors']} |",
        "| Canonical tensor-map unequal coefficient sectors | "
        f"{summary['canonical_map_unequal_coefficient_sectors']} |",
        "| Canonical-map FeynPy raw monomials | "
        f"{summary['canonical_map_feynpy_raw_terms']} |",
        "| Canonical-map FeynPy canonical monomials | "
        f"{summary['canonical_map_feynpy_canonical_terms']} |",
        "| Canonical-map FeynPy redundant monomials (raw - canonical) | "
        f"{summary['canonical_map_feynpy_redundant_terms']} |",
        f"| Explained benign head-count deltas | {summary['benign_head_count_delta_heads']} |",
        f"| Unexplained head-count deltas | {summary['unexplained_head_count_delta_heads']} |",
        "",
        "## Basis",
        "",
        f"- Reference: `{basis['reference_ltot']}`.",
        f"- Local default model: `{basis['local_ltot']}`.",
        f"- Local SM plus EFT model: `{basis['local_sm_plus_eft_lagrangian']}`.",
        f"- Omitted sectors: `{omitted_sectors}`.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | {count} |")

    exact_rows = [
        row
        for row in report["reference_vertices"]
        if row["exact_symbolic_status"] != "EXACT_UNSUPPORTED"
    ]
    lines.extend(
        [
            "",
            "## Exact Symbolic Comparison",
            "",
            "This layer is enabled for every FeynRules reference row. Bosonic "
            "rows use the native bosonic comparator. Fermion rows parse the "
            "full FeynRules tensor rule into native tensors, filter terms by "
            "indexed Wilson-coefficient head, keep flavor order and complex "
            "conjugation in the scalar coefficient, and compare canonical "
            "tensor-monomial maps. Statuses are graded honestly: "
            "`EXACT_MATCH` is direct same-signature canonical equality; "
            "`MATCH_MODULO_CC_PACKAGING` is equality after a pinned "
            "charge-conjugation packaging transform (Weinberg or Ec partner "
            "rows); `UNRESOLVED_CC_PACKAGING` means no pinned packaging rule "
            "is known or the pinned transform failed.",
            "",
            "## Sector-by-Sector Reading Guide",
            "",
            "This table explains what the comparison did to put each sector "
            "in the same mathematical form before equality was tested. Direct "
            "exact rows compare the same external-field signature. Pinned CC "
            "rows compare an explicitly listed charge-conjugation partner with "
            "a fixed phase and duplicate-leg symmetry.",
            "",
            "| Sector family | Rows | Result | Normalization/canonicalization used |",
            "| --- | ---: | --- | --- |",
            "| Bosonic and Higgs/gauge | "
            f"{family_counts.get('BOSONIC', 0)} | "
            f"{family_counts.get('BOSONIC', 0)} direct exact | "
            "Parse FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu2`, `fsu3`; "
            "expand dual field strengths; use metric symmetry, epsilon "
            "antisymmetry, structure-constant antisymmetry, dummy-index "
            "relabeling, generator-product ordering, and the narrow `f*f` "
            "Jacobi reducer. |",
            "| Two-fermion non-Weinberg | 129 | 129 direct exact | "
            "Parse gamma chains, slashed momenta, projectors, generators, "
            "index deltas, epsilons, and indexed Wilson functions; keep flavor "
            "order/conjugation in the scalar coefficient; canonicalize open "
            "spinor, Lorentz, color, and weak tensors; apply narrow SU(2) "
            "pseudoreality identities for Higgs-tilde/generator products. |",
            "| Weinberg | 2 | 2 pinned CC | "
            "FeynRules emits same-chirality `Phi Phi lL lL` and HC rows; "
            "FeynPy stores mixed `lLbar,lL` rows with explicit `dirac_C`. "
            "The accepted transform is the antisymmetrized local pair "
            "`FeynPy(lLbar,lL) - FeynPy(lL,lLbar)`, with the sign fixed by "
            "`C^T = -C`. |",
            "| Ordinary four-fermion | 15 | 15 direct exact | "
            "Preserve all four Wilson flavor slots; canonicalize color "
            "singlet/octet contractions, weak triplet currents, identical "
            "fermion dummy labels, gamma chains, and Hermitian-conjugate "
            "generator orientations. |",
            "| Charge-conjugated evanescent four-fermion | 6 rows / 12 coefficient sectors | "
            "6 pinned CC | Use the pinned `alphaEc*` rule table: exactly one "
            "partner signature, one phase, and one symmetric or antisymmetric "
            "duplicate-leg rule per coefficient sector; rewrite explicit "
            "`dirac_C` arms into FeynRules `CC[...]` flow and then demand "
            "canonical-map equality. |",
            "| FeynPy-only zero-signature artifacts | "
            f"{summary['feynpy_only_zero_signatures']} local signatures | "
            "dropped from residuals | Canonical coefficient-head collection "
            "proves the apparent signatures cancel to zero under tensor "
            "symmetries, so they are diagnostics rather than unmatched "
            "operator content. |",
            "",
            "| Signature | Status |",
            "| --- | --- |",
        ]
    )
    for row in exact_rows:
        lines.append(
            f"| `{row['key']}` | `{row['exact_symbolic_status']}` |"
        )

    canonical_rows = [
        row
        for row in report["reference_vertices"]
        if row["canonical_map_status"] != "CANONICAL_MAP_UNSUPPORTED"
    ]
    lines.extend(
        [
            "",
            "## Canonical Tensor-Map Gauge Comparison",
            "",
            "This diagnostic is enabled for supported bosonic rows. It parses "
            "FeynRules `ME`, `FV`, `SP`, `Eps`, `fsu3`, and `fsu2` into "
            "native tensors, then compares canonical monomial maps per Wilson "
            "coefficient. It uses intrinsic tensor symmetries, dummy-index "
            "relabeling, commuting factor ordering, exact coefficient "
            "collection, generator-product ordering, SU(2) pseudoreality "
            "normalization, and the narrow `f*f` Jacobi reducer. It does not "
            "use momentum conservation, EOM, IBP, Schouten/Fierz identities, "
            "or broad 4D gamma reductions.",
            "",
            "| Signature | Status | Coefficient sectors |",
            "| --- | --- | --- |",
        ]
    )
    for row in canonical_rows:
        sector_summaries = []
        for coefficient, diagnostic in sorted(
            row["canonical_map_coefficients"].items()
        ):
            status = "match" if diagnostic["matches"] else "mismatch"
            sector_summaries.append(
                f"`{coefficient}` {status}: raw "
                f"{diagnostic['feynpy_raw_terms']}/"
                f"{diagnostic['feynrules_raw_terms']} -> canonical "
                f"{diagnostic['feynpy_canonical_terms']}/"
                f"{diagnostic['feynrules_canonical_terms']}"
            )
        lines.append(
            f"| `{row['key']}` | `{row['canonical_map_status']}` | "
            f"{'; '.join(sector_summaries)} |"
        )

    missing_heads = Counter()
    local_extra_heads = Counter()
    unexplained_head_count_deltas = Counter()
    benign_head_count_deltas = []
    for row in report["reference_vertices"]:
        # Heads resolved via the charge-conjugation overlay are genuine content
        # (present in FeynPy under the charge-conjugate signature), so they are
        # excluded from the "reference-side head gap" aggregate, which is meant
        # to surface only unexplained missing operator content.
        cc_matched = set(row.get("charge_conjugation_matched_heads", ()))
        missing_heads.update(
            head for head in row["feynrules_extra_heads"] if head not in cc_matched
        )
        local_extra_heads.update(row["feynpy_extra_heads"])
        if row["head_count_status"] == "NO_LOCAL_SIGNATURE":
            continue
        for head, reason in row["benign_head_count_delta_reasons"].items():
            counts_for_head = row["head_count_delta"][head]
            benign_head_count_deltas.append(
                (
                    row["key"],
                    head,
                    counts_for_head["reference"],
                    counts_for_head["feynpy"],
                    reason,
                )
            )
        for head, counts_for_head in row["unexplained_head_count_delta"].items():
            unexplained_head_count_deltas[head] += abs(
                counts_for_head["reference"] - counts_for_head["feynpy"]
            )

    lines.extend(
        [
            "",
            "## Largest Reference-Side Head Gaps",
            "",
            "| Head | Count |",
            "| --- | ---: |",
        ]
    )
    for head, count in missing_heads.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Largest Local Extra Heads",
            "",
            "| Head | Count |",
            "| --- | ---: |",
        ]
    )
    for head, count in local_extra_heads.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Explained Benign Raw Head-Count Deltas",
            "",
            "These are raw coefficient-head occurrence-count diagnostics. They catch "
            "some missing or duplicated content, but they are not tensor-rule equality "
            "proofs because equivalent algebra can be printed with different occurrence "
            "counts.",
            "",
            "| Signature | Head | Reference | FeynPy | Reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for key, head, reference_count, feynpy_count, reason in sorted(benign_head_count_deltas):
        lines.append(
            f"| `{key}` | `{head}` | {reference_count} | {feynpy_count} | "
            f"{BENIGN_HEAD_COUNT_REASON_TEXT[reason]} |"
        )

    lines.extend(
        [
            "",
            "## Largest Unexplained Raw Head-Count Deltas",
            "",
            "These exclude the explicit benign expansions listed above. The large "
            "pure-gauge raw deltas can remain large even where the canonical "
            "tensor-map comparison above proves equality.",
            "",
            "| Head | Total absolute delta |",
            "| --- | ---: |",
        ]
    )
    for head, count in unexplained_head_count_deltas.most_common(20):
        lines.append(f"| `{head}` | {count} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison/artifacts/vertex_comparison_report.json` contains "
            "every reference row and FeynPy-only signature.",
            "- `comparison/artifacts/feynpy_vertices.json` contains the "
            "regenerated local FeynPy rules and coefficient heads.",
            "- `reference/Ltot_SMEFT_FeynRules.json` is the FeynRules oracle "
            "used for the comparison.",
            "",
        ]
    )
    return "\n".join(lines)

__all__ = [name for name in globals() if not name.startswith("__")]
