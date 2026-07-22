from symbolica import S

from feynrules.comparison import parse_feynrules_gauge_rule
from symbolic.spenso_structures import COLOR_ADJ, WEAK_ADJ


def test_parse_feynrules_gauge_rule_keeps_bare_su2w_deltas_in_weak_adjoint_metric():
    parsed = parse_feynrules_gauge_rule("IndexDelta[aw3, aw4]")
    expected = WEAK_ADJ.g(S("aw3"), S("aw4")).to_expression()
    wrong = COLOR_ADJ.g(S("aw3"), S("aw4")).to_expression()

    assert parsed.to_canonical_string() == expected.to_canonical_string()
    assert parsed.to_canonical_string() != wrong.to_canonical_string()
