from pathlib import Path

from feynrules.comparison import load_feynrules_json
from theories import build_standard_model
from theories.standard_model_feynrules import standard_model_feynrules_name_aliases


REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "feynrules"
    / "sm"
    / "sm_vertices_FeynRules.json"
)


def test_all_nonzero_standard_model_interaction_signatures_match_sm_fr():
    sm = build_standard_model(
        include_ghosts=True,
        include_gauge_fixing=False,
    )
    aliases = standard_model_feynrules_name_aliases(sm.fields)
    reference_signatures = {
        reference.signature
        for reference in load_feynrules_json(REFERENCE_PATH)
    }

    nonzero_signatures: set[tuple[str, ...]] = set()
    zero_candidates: set[tuple[str, ...]] = set()
    for signature in sm.lagrangian.vertex_signatures(flavor_expand=True):
        if signature.arity not in (3, 4):
            continue
        normalized = tuple(
            sorted(aliases.get(name, name) for name in signature.names)
        )
        rule = sm.lagrangian.feynman_rule(
            *signature.fields,
            simplify=True,
            include_delta=False,
            flavor_expand=True,
        )
        if rule.to_canonical_string() == "0":
            zero_candidates.add(normalized)
        else:
            nonzero_signatures.add(normalized)

    assert len(zero_candidates) == 8, sorted(zero_candidates)
    assert nonzero_signatures == reference_signatures, {
        "feynrules_only": sorted(reference_signatures - nonzero_signatures),
        "feynpy_only": sorted(nonzero_signatures - reference_signatures),
    }
    assert len(nonzero_signatures) == 163
