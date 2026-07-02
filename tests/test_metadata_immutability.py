import pytest
from symbolica import S

from feynpy import COLOR_FUND_INDEX, Field, Parameter
from feynpy.interactions import ExternalLeg


def test_field_quantum_numbers_are_copied_and_read_only():
    source = {"Q": 1}
    field = Field("phi", spin=0, self_conjugate=True, quantum_numbers=source)
    fields = {field}

    source["Q"] = 2

    assert field.quantum_numbers["Q"] == 1
    with pytest.raises(TypeError):
        field.quantum_numbers["Q"] = 2
    assert field in fields


def test_parameter_components_are_copied_and_read_only():
    components = {(1,): S("y1")}
    parameter = Parameter(
        "y",
        indices=(COLOR_FUND_INDEX,),
        components=components,
    )

    components[(1,)] = S("changed")

    assert parameter.components[(1,)] == S("y1")
    with pytest.raises(TypeError):
        parameter.components[(1,)] = S("changed")


def test_occurrence_and_external_leg_labels_cannot_diverge_from_slot_labels():
    field = Field(
        "phi",
        spin=0,
        self_conjugate=True,
        indices=(COLOR_FUND_INDEX,),
    )
    original = S("c1")
    changed = S("c2")
    labels = {COLOR_FUND_INDEX.kind: original}
    occurrence = field.occurrence(labels=labels)
    leg = ExternalLeg(field=field, momentum=S("p"), labels=labels)

    labels[COLOR_FUND_INDEX.kind] = changed

    assert occurrence.labels[COLOR_FUND_INDEX.kind] == original
    assert occurrence.slot_labels.get(0) == original
    assert leg.labels[COLOR_FUND_INDEX.kind] == original
    assert leg.slot_labels.get(0) == original
    with pytest.raises(TypeError):
        occurrence.labels[COLOR_FUND_INDEX.kind] = changed
    with pytest.raises(TypeError):
        leg.labels[COLOR_FUND_INDEX.kind] = changed
