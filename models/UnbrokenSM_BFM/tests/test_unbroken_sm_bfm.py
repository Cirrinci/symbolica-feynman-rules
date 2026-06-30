from models.UnbrokenSM_BFM import build_unbroken_sm_bfm
from models.UnbrokenSM_BFM.comparison import compare


def test_complete_interaction_signature_set_has_reference_size():
    theory = build_unbroken_sm_bfm()
    interactions = tuple(
        signature
        for signature in theory.lagrangian.vertex_signatures()
        if signature.arity in (3, 4)
    )
    assert len(interactions) == 67
    assert sum(signature.arity == 3 for signature in interactions) == 42
    assert sum(signature.arity == 4 for signature in interactions) == 25


def test_complete_feynrules_comparison():
    result, _vertices = compare()
    assert result.matched == result.total == 67
    assert result.all_match
