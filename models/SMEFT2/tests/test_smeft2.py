from models.SMEFT2 import OMITTED_SECTORS, build_smeft_green_bpreserving


def test_smeft2_supported_subset_builds_and_compiles():
    bundle = build_smeft_green_bpreserving()
    lagrangian = bundle.model.lagrangian()
    signatures = {signature.names for signature in lagrangian.vertex_signatures()}

    assert len(lagrangian.terms) == 798
    assert ("QL.bar", "QL", "B") in signatures
    assert ("Phi.bar", "QL.bar", "UR", "G") in signatures
    assert ("LL.bar", "LR", "DR.bar", "QL") in signatures
    assert ("LL.bar", "LL", "Phi", "Phi") in signatures
    assert ("QL.bar", "LR", "DR.bar", "LL") in signatures
    assert ("B", "Phi.bar", "QL.bar", "B", "UR") in signatures


def test_smeft2_omitted_sector_list_is_explicit():
    assert "LWeinberg" not in OMITTED_SECTORS
    assert "LH4D2[alphaRHDpp]" not in OMITTED_SECTORS
    assert "LEvF2XH" not in OMITTED_SECTORS
    assert "LEv4q" not in OMITTED_SECTORS
    assert "LEvF2HD2" not in OMITTED_SECTORS
    assert "LEvCCRRLL" not in OMITTED_SECTORS
    assert "LF2HD2" in OMITTED_SECTORS
