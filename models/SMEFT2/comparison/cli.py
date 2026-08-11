"""Command-line entry point for the SMEFT2 comparison package."""

import sys

from .charge_conjugation import _EC_PARTNER_PACKAGING_RULES
from .reporting import *


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE,
        help="FeynRules JSON reference to compare against.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write files; return nonzero unless operator-content coverage "
            "is complete and every supported row is a direct `EXACT_MATCH` "
            "(strict exact symbolic). Use `--allow-cc-packaging` to also accept "
            "`MATCH_MODULO_EC_CC_CONVENTION` rows and pinned "
            "`MATCH_MODULO_CC_PACKAGING` rows. Unresolved CC packaging rows "
            "never pass `--check`."
        ),
    )
    parser.add_argument(
        "--allow-cc-packaging",
        action="store_true",
        help=(
            "With --check, accept `MATCH_MODULO_EC_CC_CONVENTION` rows (the "
            "single global evanescent charge-conjugation packaging convention) "
            "and pinned `MATCH_MODULO_CC_PACKAGING` rows (Weinberg and pinned "
            "Ec partner rows). Does not accept `UNRESOLVED_CC_PACKAGING` rows."
        ),
    )
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help=(
            "With --check, also require matching raw coefficient-head occurrence "
            "counts for every shared signature."
        ),
    )
    args = parser.parse_args(argv)

    api = sys.modules.get("models.SMEFT2.comparison")
    compare_fn = getattr(api, "compare", compare)
    compare_weinberg_fn = getattr(
        api,
        "compare_reconstructed_weinberg",
        compare_reconstructed_weinberg,
    )
    compare_ec_cc_fn = getattr(
        api,
        "compare_ec_charge_conjugation_reconstruction",
        compare_ec_charge_conjugation_reconstruction,
    )

    report, local_vertices = compare_fn(args.reference)
    weinberg_report, weinberg_vertices = compare_weinberg_fn(
        args.reference
    )
    ec_cc_report, ec_cc_vertices = compare_ec_cc_fn(
        args.reference
    )
    if not args.check:
        write_outputs(report, local_vertices)
        write_weinberg_outputs(weinberg_report, weinberg_vertices)
        write_ec_charge_conjugation_outputs(ec_cc_report, ec_cc_vertices)

    summary = report["summary"]
    weinberg_summary = weinberg_report["summary"]
    ec_cc_summary = ec_cc_report["summary"]
    exact_supported = summary["exact_symbolic_supported_vertices"]
    direct_exact = summary["exact_symbolic_direct_match_vertices"]
    ec_convention = summary["ec_cc_convention_match_vertices"]
    pinned_cc = summary["cc_packaging_pinned_match_vertices"]
    unresolved_cc = summary["cc_packaging_unresolved_vertices"]
    exact_unequal = summary["exact_symbolic_unequal_vertices"]
    exact_missing = summary["exact_symbolic_missing_local_vertices"]
    exact_error = summary["exact_symbolic_error_vertices"]
    exact_accounted = (
        direct_exact
        + ec_convention
        + pinned_cc
        + unresolved_cc
        + exact_unequal
        + exact_missing
        + exact_error
    )
    accepted_exact = direct_exact + (
        ec_convention + pinned_cc if args.allow_cc_packaging else 0
    )
    weinberg_check_failed = (
        weinberg_summary["reference_vertices"] != 2
        or (
            weinberg_summary["direct_matches"]
            != weinberg_summary["reference_vertices"]
        )
        or (
            weinberg_summary["coefficient_matches"]
            != weinberg_summary["coefficient_checks"]
        )
        or weinberg_summary["wrong_sign_matches"]
    )
    ec_cc_check_failed = (
        ec_cc_summary["coefficient_sectors"] != len(_EC_PARTNER_PACKAGING_RULES)
        or ec_cc_summary["exact_matches"] != ec_cc_summary["coefficient_sectors"]
        or ec_cc_summary["wrong_combination_matches"]
    )
    print(
        "SMEFT2 comparison: "
        f"{summary['operator_content_matches_including_cc']}/"
        f"{summary['reference_vertex_count']} "
        "reference vertices match at operator-content level "
        f"({summary['shared_head_matches']} literal-signature head matches + "
        f"{summary['charge_conjugation_packaging_matches']} CC-packaging head "
        "matches); "
        "exact symbolic split="
        f"direct {direct_exact}/{exact_supported}, "
        f"modulo global Ec CC convention {ec_convention}/{exact_supported}, "
        f"modulo pinned CC {pinned_cc}/{exact_supported}, "
        f"unresolved CC {unresolved_cc}/{exact_supported}; "
        f"raw-head-count matches={summary['shared_head_count_matches']}/"
        f"{summary['shared_signatures']}; "
        "bosonic canonical tensor-map matches="
        f"{summary['canonical_map_equal_vertices']}/"
        f"{summary['canonical_map_supported_vertices']} supported bosonic vertices "
        f"({summary['canonical_map_equal_coefficient_sectors']}/"
        f"{summary['canonical_map_supported_coefficient_sectors']} sectors); "
        "Weinberg reconstructed sidecar="
        f"{weinberg_summary['direct_matches']}/"
        f"{weinberg_summary['reference_vertices']} direct, "
        f"{weinberg_summary['coefficient_matches']}/"
        f"{weinberg_summary['coefficient_checks']} coefficient checks, "
        f"wrong-sign matches={weinberg_summary['wrong_sign_matches']}; "
        "EC CC sidecar="
        f"{ec_cc_summary['exact_matches']}/"
        f"{ec_cc_summary['coefficient_sectors']} coefficient sectors, "
        f"wrong-combination matches={ec_cc_summary['wrong_combination_matches']}; "
        f"reference-only={summary['reference_only_signatures']}; "
        f"feynpy-only={summary['feynpy_only_signatures']}."
    )
    if args.check and (
        summary["operator_content_matches_including_cc"]
        != summary["reference_vertex_count"]
        or summary["feynpy_only_unexplained_signatures"]
        or summary["exact_symbolic_supported_vertices"]
        != summary["reference_vertex_count"]
        or exact_accounted != exact_supported
        or accepted_exact != exact_supported
        or exact_unequal
        or exact_missing
        or exact_error
        or unresolved_cc
        or (
            (pinned_cc or ec_convention)
            and not args.allow_cc_packaging
        )
        or summary["canonical_map_unequal_vertices"]
        or summary["canonical_map_error_vertices"]
        or (args.strict_counts and summary["shared_head_count_mismatches"])
        or weinberg_check_failed
        or ec_cc_check_failed
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
