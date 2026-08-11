"""Mutation-sensitivity tests for the SMEFT2 FeynRules comparison.

These mirror ``models/SM/tests/test_feynrules_comparison_sensitivity.py``: each
test corrupts the FeynRules reference in a physically meaningful way and
asserts that the comparison notices. Without them, a comparison that silently
degraded into a tautology would still report a perfect score.

The assertions deliberately accept any non-accepting verdict rather than one
specific status, because different corruptions surface at different layers
(canonical-map inequality, chirality validation, or a parse failure). What
matters is that no corruption is reported as agreement.
"""

import re
from dataclasses import replace

import pytest

from models.SMEFT2 import build_smeft_green_bpreserving
from models.SMEFT2 import comparison as smeft2_comparison
from models.SMEFT2.comparison import (
    GENERIC_PARAMETER_NAMES,
    REFERENCE,
    ChiralityMismatch,
    _comparison_field_map,
    _fermion_exact_symbolic_row,
    _local_vertices,
    _name_key,
    _reference_heads,
    load_feynrules_json,
    validate_smeft2_projector_chirality,
)

ACCEPTING_STATUSES = {
    "EXACT_MATCH",
    "MATCH_MODULO_EC_CC_CONVENTION",
    "MATCH_MODULO_CC_PACKAGING",
}


@pytest.fixture(scope="module")
def context():
    references = load_feynrules_json(REFERENCE)
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    field_map = _comparison_field_map(bundle)
    parameter_names = set(bundle.parameters) | GENERIC_PARAMETER_NAMES
    local_vertices = _local_vertices(parameter_names)
    return {
        "references": {_name_key(r.fields): r for r in references},
        "lagrangian": lagrangian,
        "field_map": field_map,
        "parameter_names": parameter_names,
        "local_vertices": local_vertices,
        "local_by_key": {v.key: v for v in local_vertices},
    }


def _status(context, reference):
    key = _name_key(reference.fields)
    heads = set(_reference_heads(reference, context["parameter_names"]))
    row = _fermion_exact_symbolic_row(
        reference=reference,
        local=context["local_by_key"].get(key),
        local_vertices=context["local_vertices"],
        reference_heads=heads,
        local_heads=heads,
        head_count_status="IGNORED",
        lagrangian=context["lagrangian"],
        field_map=context["field_map"],
    )
    return row["status"] if row else "NONE"


def _assert_detected(context, reference, mutate, label):
    mutated = replace(reference, rule=mutate(reference.rule))
    assert mutated.rule != reference.rule, f"{label}: mutation was a no-op"
    status = _status(context, mutated)
    assert status not in ACCEPTING_STATUSES, (
        f"{label}: corrupted reference was still accepted as {status}"
    )


# A two-fermion dipole row and a four-fermion row, so the suite covers both
# the direct-exact path and the Ec charge-conjugation convention path.
TWO_FERMION_KEY = "B|Phi|eR|lLbar"
FOUR_FERMION_KEY = "eR|eRbar|lL|lLbar"


def test_baseline_rows_are_accepted(context):
    """Guard against the suite passing because every row already fails."""

    for key in (TWO_FERMION_KEY, FOUR_FERMION_KEY):
        reference = context["references"][key]
        assert _status(context, reference) in ACCEPTING_STATUSES


def test_detects_rescaled_wilson_coefficient(context):
    reference = context["references"][FOUR_FERMION_KEY]
    _assert_detected(
        context,
        reference,
        lambda rule: rule.replace("alphaEcle[", "2*alphaEcle[", 1),
        "rescaled Wilson coefficient",
    )


def test_detects_two_fermion_coefficient_rescaling(context):
    reference = context["references"][TWO_FERMION_KEY]
    head = next(
        h
        for h in _reference_heads(reference, context["parameter_names"])
        if h.startswith("alpha")
    )
    _assert_detected(
        context,
        reference,
        lambda rule: rule.replace(f"{head}[", f"3*{head}[", 1),
        "rescaled two-fermion coefficient",
    )


def test_detects_global_sign_flip(context):
    reference = context["references"][TWO_FERMION_KEY]
    _assert_detected(
        context,
        reference,
        lambda rule: f"-({rule})",
        "global sign flip",
    )


def test_detects_chirality_flip(context):
    """The projector drop must not make the comparison chirality-blind."""

    reference = context["references"][FOUR_FERMION_KEY]
    _assert_detected(
        context,
        reference,
        lambda rule: rule.replace("ProjM", "ProjP", 1),
        "chirality flip",
    )


def test_detects_reversed_chirality_flip(context):
    reference = context["references"][FOUR_FERMION_KEY]
    _assert_detected(
        context,
        reference,
        lambda rule: rule.replace("ProjP", "ProjM", 1),
        "reverse chirality flip",
    )


def _swap_first_two_flavor_slots(rule: str) -> str:
    """Transpose the first two generation arguments of the first coefficient."""

    match = re.search(
        r"(alpha[A-Za-z0-9]+)\[\s*"
        r"(Index\[Generation,\s*Ext\[\d+\]\]),\s*"
        r"(Index\[Generation,\s*Ext\[\d+\]\])",
        rule,
    )
    if match is None or match.group(2) == match.group(3):
        return rule
    replacement = f"{match.group(1)}[{match.group(3)}, {match.group(2)}"
    return rule[: match.start()] + replacement + rule[match.end() :]


def test_detects_swapped_flavor_indices(context):
    """Flavor order must survive canonicalization as physical information."""

    reference = context["references"][FOUR_FERMION_KEY]
    _assert_detected(
        context,
        reference,
        _swap_first_two_flavor_slots,
        "swapped flavor indices",
    )


def test_detects_removed_imaginary_unit(context):
    reference = context["references"][TWO_FERMION_KEY]
    if "I*" not in reference.rule:
        pytest.skip("reference row carries no explicit imaginary unit")
    _assert_detected(
        context,
        reference,
        lambda rule: rule.replace("I*", "", 1),
        "removed imaginary unit",
    )


def test_chirality_validation_rejects_flipped_projector():
    """The validator itself must reject a projector that contradicts fields."""

    fields = ("eRbar", "eR", "lLbar", "lL")
    good = (
        "ProjM[Index[Spin, Ext[1]], Index[Spin, Ext[4]]]"
        "*ProjP[Index[Spin, Ext[3]], Index[Spin, Ext[2]]]"
    )
    assert validate_smeft2_projector_chirality(good, fields) == 2

    bad = good.replace("ProjM", "ProjP", 1)
    with pytest.raises(ChiralityMismatch):
        validate_smeft2_projector_chirality(bad, fields)


def test_chirality_validation_rejects_forbidden_chain():
    """A chain that vanishes identically by chirality must be rejected."""

    # lLbar -> lL across one gamma is allowed (same chirality, odd gamma
    # count); lLbar -> eR across one gamma is not.
    fields = ("lLbar", "eR")
    forbidden = (
        "TensDot[Ga[Index[Lorentz, Ext[3]]], ProjP]"
        "[Index[Spin, Ext[1]], Index[Spin, Ext[2]]]"
    )
    with pytest.raises(ChiralityMismatch):
        validate_smeft2_projector_chirality(forbidden, fields)


def test_every_reference_projector_passes_validation():
    """The validator must be satisfied by the unmutated reference export."""

    references = load_feynrules_json(REFERENCE)
    checked = sum(
        validate_smeft2_projector_chirality(reference.rule, reference.fields)
        for reference in references
    )
    # A non-trivial number of projectors must actually have been examined,
    # otherwise the check above would be vacuous.
    assert checked > 2_000


def test_comparison_module_exposes_sensitivity_entry_points():
    """Keep the mutation hooks these tests rely on part of the public surface."""

    for name in (
        "_fermion_exact_symbolic_row",
        "validate_smeft2_projector_chirality",
        "ChiralityMismatch",
    ):
        assert hasattr(smeft2_comparison, name)
