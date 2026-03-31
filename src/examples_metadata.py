"""
Clean metadata-only examples for model_symbolica.py.

This file intentionally uses only the metadata-layer API on the user-facing
side: Field, FieldOccurrence, ExternalLeg, DerivativeAction, InteractionTerm.
The older parallel-list examples from examples_symbolica.py are imported only
inside the regression checks so we can verify that both interfaces agree.
"""

import argparse
from fractions import Fraction

import examples_symbolica as legacy
from model_legacy import vertex_factor as legacy_vertex_factor
from model_schema import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    IndexType,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    DerivativeAction,
    Field,
    InteractionTerm,
    bind_indices,
)
from model_symbolica import (
    S,
    Expression,
    I,
    simplify_deltas,
    vertex_factor,
)
from spenso_structures import (
    gamma_lowered_matrix,
    gamma_matrix,
    gamma5_matrix,
    gauge_generator,
    lorentz_metric,
    simplify_gamma_chain,
)


# ---------------------------------------------------------------------------
# Common symbols
# ---------------------------------------------------------------------------

x = legacy.x
d = legacy.d

p1, p2, p3, p4 = legacy.p1, legacy.p2, legacy.p3, legacy.p4
b1, b2, b3, b4 = legacy.b1, legacy.b2, legacy.b3, legacy.b4

phi0 = legacy.phi0
chi0 = legacy.chi0
phiC0 = legacy.phiC0
phiCdag0 = legacy.phiCdag0
psibar0 = legacy.psibar0
psi0 = legacy.psi0
A0 = legacy.A0
G0 = legacy.G0

mu, nu = legacy.mu, legacy.nu
mu3, mu4 = legacy.mu3, legacy.mu4

lam4 = legacy.lam4
g_sym = legacy.g_sym
lamC = legacy.lamC
yF = legacy.yF
gV = legacy.gV
gS = legacy.gS
gPhiA = legacy.gPhiA
gPhiAA = legacy.gPhiAA
g_psi4 = legacy.g_psi4
gJJ = legacy.gJJ

alpha_s, beta_s = legacy.alpha_s, legacy.beta_s
a_bar, a_psi, b_bar, b_psi = legacy.a_bar, legacy.a_psi, legacy.b_bar, legacy.b_psi
i_psi_bar, i_psi = legacy.i_psi_bar, legacy.i_psi
i1, i2, i3, i4 = legacy.i1, legacy.i2, legacy.i3, legacy.i4
s1, s2, s3, s4 = legacy.s1, legacy.s2, legacy.s3, legacy.s4
i_bar_q, i_psi_q = legacy.i_bar_q, legacy.i_psi_q
c_bar_q, c_psi_q, a_g = legacy.c_bar_q, legacy.c_psi_q, legacy.a_g
c1, c2, a3 = legacy.c1, legacy.c2, legacy.a3


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

PhiField = Field(
    "Phi",
    spin=0,
    self_conjugate=True,
    kind="scalar",
    symbol=phi0,
)
ChiField = Field(
    "Chi",
    spin=0,
    self_conjugate=True,
    kind="scalar",
    symbol=chi0,
)
PhiCField = Field(
    "PhiC",
    spin=0,
    self_conjugate=False,
    kind="scalar",
    symbol=phiC0,
    conjugate_symbol=phiCdag0,
)
PsiField = Field(
    "Psi",
    spin=Fraction(1, 2),
    self_conjugate=False,
    kind="fermion",
    symbol=psi0,
    conjugate_symbol=psibar0,
    indices=(SPINOR_INDEX,),
)
GaugeField = Field(
    "A",
    spin=1,
    self_conjugate=True,
    kind="vector",
    symbol=A0,
    indices=(LORENTZ_INDEX,),
)
QuarkField = Field(
    "q",
    spin=Fraction(1, 2),
    self_conjugate=False,
    kind="fermion",
    symbol=psi0,
    conjugate_symbol=psibar0,
    indices=(SPINOR_INDEX, COLOR_FUND_INDEX),
)
GluonField = Field(
    "G",
    spin=1,
    self_conjugate=True,
    kind="vector",
    symbol=G0,
    indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
)


# ---------------------------------------------------------------------------
# Small metadata helpers
# ---------------------------------------------------------------------------

def _bindings(*pairs):
    pairs = tuple(pair for pair in pairs if pair is not None)
    return bind_indices(*pairs) if pairs else None


def _repeat_occurrences(field_obj, count, **kwargs):
    return tuple(field_obj.occurrence(**kwargs) for _ in range(count))


def _fermion_occ(label, *, field_obj=PsiField, conjugated=False, extra_pairs=()):
    pairs = [pair for pair in extra_pairs]
    if label is not None:
        pairs.insert(0, (SPINOR_INDEX, label))
    return field_obj.occurrence(
        conjugated=conjugated,
        slot_labels=_bindings(*pairs),
    )


def _fermion_leg(
    momentum,
    *,
    spin,
    spinor_label=None,
    species=None,
    conjugated=False,
    field_obj=PsiField,
    extra_pairs=(),
):
    pairs = [pair for pair in extra_pairs]
    if spinor_label is not None:
        pairs.insert(0, (SPINOR_INDEX, spinor_label))
    return field_obj.leg(
        momentum,
        conjugated=conjugated,
        species=species,
        spin=spin,
        slot_labels=_bindings(*pairs),
    )


def _boson_leg(field_obj, momentum, *, species=None, conjugated=False):
    return field_obj.leg(momentum, species=species, conjugated=conjugated)


def _vector_occ(label, *, field_obj=GaugeField, extra_pairs=()):
    pairs = [pair for pair in extra_pairs]
    if label is not None:
        pairs.insert(0, (LORENTZ_INDEX, label))
    return field_obj.occurrence(slot_labels=_bindings(*pairs))


def _vector_leg(momentum, *, label=None, species=None, field_obj=GaugeField, extra_pairs=()):
    pairs = [pair for pair in extra_pairs]
    if label is not None:
        pairs.insert(0, (LORENTZ_INDEX, label))
    return field_obj.leg(momentum, species=species, slot_labels=_bindings(*pairs))


def _vertex(*, interaction, external_legs=None, strip_externals=True, species_map=None):
    expr = vertex_factor(
        interaction=interaction,
        external_legs=external_legs,
        x=x,
        d=d,
        strip_externals=strip_externals,
    )
    if species_map is not None:
        return simplify_deltas(expr, species_map=species_map)
    return simplify_deltas(expr)


def _show(title, expr, *, interaction, external_legs):
    print("=" * 80)
    print(f"  {title}")
    print(f"  alphas = {[occ.species for occ in interaction.fields]}")
    print(f"  betas  = {[leg.species for leg in external_legs]}")
    print(f"  ps     = {[leg.momentum for leg in external_legs]}")
    print()
    print("  Vertex:")
    print(f"  {expr}")
    print()


def _check_same(label, got, expected):
    assert (
        got.expand().to_canonical_string()
        == expected.expand().to_canonical_string()
    ), f"{label} FAILED:\n  got:      {got}\n  expected: {expected}"
    print(f"  {label}: PASS")


# ---------------------------------------------------------------------------
# Metadata-only interactions
# ---------------------------------------------------------------------------

TERM_phi4 = InteractionTerm(
    coupling=lam4,
    fields=_repeat_occurrences(PhiField, 4),
    label="lam4 * phi^4",
)
LEGS_phi4 = (
    _boson_leg(PhiField, p1, species=b1),
    _boson_leg(PhiField, p2, species=b2),
    _boson_leg(PhiField, p3, species=b3),
    _boson_leg(PhiField, p4, species=b4),
)

TERM_phi2chi2 = InteractionTerm(
    coupling=g_sym,
    fields=(
        *_repeat_occurrences(PhiField, 2),
        *_repeat_occurrences(ChiField, 2),
    ),
    label="g * phi^2 * chi^2",
)
LEGS_phi2chi2 = (
    _boson_leg(PhiField, p1, species=b1),
    _boson_leg(PhiField, p2, species=b2),
    _boson_leg(ChiField, p3, species=b3),
    _boson_leg(ChiField, p4, species=b4),
)

TERM_phiCdag_phiC = InteractionTerm(
    coupling=lamC,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
    ),
    label="lamC * phi^dagger * phi",
)
LEGS_phiCdag_phiC = (
    _boson_leg(PhiCField, p1, conjugated=True, species=b1),
    _boson_leg(PhiCField, p2, species=b2),
)

TERM_yukawa = InteractionTerm(
    coupling=yF,
    fields=(
        _fermion_occ(alpha_s, conjugated=True),
        _fermion_occ(alpha_s),
        PhiField.occurrence(),
    ),
    label="yF * psibar * psi * phi",
)
LEGS_yukawa = (
    _fermion_leg(p1, conjugated=True, spin=s1, spinor_label=i1),
    _fermion_leg(p2, spin=s2, spinor_label=i2),
    _boson_leg(PhiField, p3),
)
LEGS_yukawa_matrix = (
    _fermion_leg(p1, conjugated=True, species=b1, spin=s1),
    _fermion_leg(p2, species=b2, spin=s2),
    _boson_leg(PhiField, p3, species=b3),
)

TERM_vec_current = InteractionTerm(
    coupling=gV * gamma_matrix(i_psi_bar, i_psi, mu),
    fields=(
        _fermion_occ(i_psi_bar, conjugated=True),
        _fermion_occ(i_psi),
        _vector_occ(mu),
    ),
    label="gV * psibar gamma^mu psi A_mu",
)

TERM_axial_current = InteractionTerm(
    coupling=gV * gamma_matrix(i_psi_bar, alpha_s, mu) * gamma5_matrix(alpha_s, i_psi),
    fields=(
        _fermion_occ(i_psi_bar, conjugated=True),
        _fermion_occ(i_psi),
        _vector_occ(mu),
    ),
    label="gV * psibar gamma^mu gamma5 psi A_mu",
)

LEGS_vec_current = (
    _fermion_leg(p1, conjugated=True, spin=s1, spinor_label=i1),
    _fermion_leg(p2, spin=s2, spinor_label=i2),
    _vector_leg(p3, label=mu3),
)

TERM_psibar_psi_sq = InteractionTerm(
    coupling=-g_psi4 / Expression.num(2),
    fields=(
        _fermion_occ(alpha_s, conjugated=True),
        _fermion_occ(alpha_s),
        _fermion_occ(beta_s, conjugated=True),
        _fermion_occ(beta_s),
    ),
    label="-(g/2)(psibar psi)^2",
)
LEGS_fermion4 = (
    _fermion_leg(p1, conjugated=True, spin=s1, spinor_label=i1),
    _fermion_leg(p2, spin=s2, spinor_label=i2),
    _fermion_leg(p3, conjugated=True, spin=s3, spinor_label=i3),
    _fermion_leg(p4, spin=s4, spinor_label=i4),
)
LEGS_fermion4_matrix = (
    _fermion_leg(p1, conjugated=True, species=b1, spin=s1),
    _fermion_leg(p2, species=b2, spin=s2),
    _fermion_leg(p3, conjugated=True, species=b3, spin=s3),
    _fermion_leg(p4, species=b4, spin=s4),
)

TERM_current_current = InteractionTerm(
    coupling=gJJ * gamma_matrix(a_bar, a_psi, mu) * gamma_lowered_matrix(b_bar, b_psi, mu),
    fields=(
        _fermion_occ(a_bar, conjugated=True),
        _fermion_occ(a_psi),
        _fermion_occ(b_bar, conjugated=True),
        _fermion_occ(b_psi),
    ),
    label="gJJ * (psibar gamma^mu psi)(psibar gamma_mu psi)",
)

TERM_quark_gluon = InteractionTerm(
    coupling=gS * gamma_matrix(i_bar_q, i_psi_q, mu) * gauge_generator(a_g, c_bar_q, c_psi_q),
    fields=(
        _fermion_occ(
            i_bar_q,
            field_obj=QuarkField,
            conjugated=True,
            extra_pairs=((COLOR_FUND_INDEX, c_bar_q),),
        ),
        _fermion_occ(
            i_psi_q,
            field_obj=QuarkField,
            extra_pairs=((COLOR_FUND_INDEX, c_psi_q),),
        ),
        _vector_occ(
            mu,
            field_obj=GluonField,
            extra_pairs=((COLOR_ADJ_INDEX, a_g),),
        ),
    ),
    label="gS * psibar gamma^mu T^a psi G^a_mu",
)
LEGS_quark_gluon = (
    _fermion_leg(
        p1,
        conjugated=True,
        spin=s1,
        spinor_label=i1,
        field_obj=QuarkField,
        extra_pairs=((COLOR_FUND_INDEX, c1),),
    ),
    _fermion_leg(
        p2,
        spin=s2,
        spinor_label=i2,
        field_obj=QuarkField,
        extra_pairs=((COLOR_FUND_INDEX, c2),),
    ),
    _vector_leg(
        p3,
        label=mu3,
        field_obj=GluonField,
        extra_pairs=((COLOR_ADJ_INDEX, a3),),
    ),
)

TERM_complex_scalar_current_phi = InteractionTerm(
    coupling=gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        _vector_occ(mu),
    ),
    derivatives=(DerivativeAction(target=1, indices=(mu,)),),
    label="gPhiA * A_mu * phi^dagger * d^mu phi",
)
TERM_complex_scalar_current_phidag = InteractionTerm(
    coupling=-gPhiA,
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        _vector_occ(mu),
    ),
    derivatives=(DerivativeAction(target=0, indices=(mu,)),),
    label="-gPhiA * A_mu * d^mu phi^dagger * phi",
)
LEGS_complex_scalar_current = (
    _boson_leg(PhiCField, p1, conjugated=True, species=b1),
    _boson_leg(PhiCField, p2, species=b2),
    _vector_leg(p3, label=mu3, species=b3),
)

TERM_complex_scalar_contact = InteractionTerm(
    coupling=gPhiAA * lorentz_metric(mu, nu),
    fields=(
        PhiCField.occurrence(conjugated=True),
        PhiCField.occurrence(),
        _vector_occ(mu),
        _vector_occ(nu),
    ),
    label="gPhiAA * A_mu A^mu phi^dagger phi",
)
LEGS_complex_scalar_contact = (
    _boson_leg(PhiCField, p1, conjugated=True, species=b1),
    _boson_leg(PhiCField, p2, species=b2),
    _vector_leg(p3, label=mu3, species=b3),
    _vector_leg(p4, label=mu4, species=b4),
)


# ---------------------------------------------------------------------------
# Regression checks against legacy examples_symbolica.py
# ---------------------------------------------------------------------------

def _run_regression_checks():
    sm_phi = {b1: phi0, b2: phi0, b3: phi0, b4: phi0}
    sm_phi_chi = {b1: phi0, b2: phi0, b3: chi0, b4: chi0}
    sm_yukawa = {b1: psibar0, b2: psi0, b3: phi0}
    sm_fermion4 = {b1: psibar0, b2: psi0, b3: psibar0, b4: psi0}
    sm_gauge = {b1: psibar0, b2: psi0, b3: G0}
    sm_scalar_gauge = {b1: phiCdag0, b2: phiC0, b3: A0}
    sm_scalar_contact = {b1: phiCdag0, b2: phiC0, b3: A0, b4: A0}

    _check_same(
        "phi^4",
        _vertex(interaction=TERM_phi4, external_legs=LEGS_phi4, species_map=sm_phi),
        simplify_deltas(legacy_vertex_factor(**legacy.L_phi4, x=x, d=d), species_map=sm_phi),
    )
    _check_same(
        "phi^2 chi^2",
        _vertex(interaction=TERM_phi2chi2, external_legs=LEGS_phi2chi2, species_map=sm_phi_chi),
        simplify_deltas(legacy_vertex_factor(**legacy.L_phi2chi2, x=x, d=d), species_map=sm_phi_chi),
    )
    _check_same(
        "phi^dagger phi",
        _vertex(
            interaction=TERM_phiCdag_phiC,
            external_legs=LEGS_phiCdag_phiC,
            species_map={b1: phiCdag0, b2: phiC0},
        ),
        simplify_deltas(
            legacy_vertex_factor(**legacy.L_phiCdag_phiC, x=x, d=d),
            species_map={b1: phiCdag0, b2: phiC0},
        ),
    )
    _check_same(
        "Yukawa amputated",
        _vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        simplify_deltas(legacy_vertex_factor(**legacy.L_yukawa, x=x, d=d), species_map=sm_yukawa),
    )
    _check_same(
        "Yukawa matrix element",
        _vertex(
            interaction=TERM_yukawa,
            external_legs=LEGS_yukawa_matrix,
            strip_externals=False,
            species_map=sm_yukawa,
        ),
        simplify_deltas(
            legacy_vertex_factor(**legacy.L_yukawa, x=x, d=d, strip_externals=False),
            species_map=sm_yukawa,
        ),
    )
    _check_same(
        "Vector current",
        _vertex(interaction=TERM_vec_current, external_legs=LEGS_vec_current),
        I * gV * gamma_matrix(i1, i2, mu3),
    )
    _check_same(
        "Axial current",
        _vertex(interaction=TERM_axial_current, external_legs=LEGS_vec_current),
        I * gV * gamma_matrix(i1, alpha_s, mu3) * gamma5_matrix(alpha_s, i2),
    )
    _check_same(
        "(psibar psi)^2",
        _vertex(interaction=TERM_psibar_psi_sq, external_legs=LEGS_fermion4),
        simplify_deltas(legacy_vertex_factor(**legacy.L_psibar_psi_sq, x=x, d=d), species_map=sm_fermion4),
    )
    _check_same(
        "(psibar psi)^2 matrix element",
        _vertex(
            interaction=TERM_psibar_psi_sq,
            external_legs=LEGS_fermion4_matrix,
            strip_externals=False,
            species_map=sm_fermion4,
        ),
        simplify_deltas(
            legacy_vertex_factor(**legacy.L_psibar_psi_sq, x=x, d=d, strip_externals=False),
            species_map=sm_fermion4,
        ),
    )
    _check_same(
        "Current-current",
        simplify_gamma_chain(_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)),
        simplify_gamma_chain(
            simplify_deltas(legacy_vertex_factor(**legacy.L_current_current, x=x, d=d), species_map=sm_fermion4)
        ),
    )
    _check_same(
        "Quark-gluon current",
        _vertex(interaction=TERM_quark_gluon, external_legs=LEGS_quark_gluon),
        simplify_deltas(legacy_vertex_factor(**legacy.L_quark_gluon, x=x, d=d), species_map=sm_gauge),
    )
    _check_same(
        "Complex scalar current",
        _vertex(
            interaction=TERM_complex_scalar_current_phi,
            external_legs=LEGS_complex_scalar_current,
            species_map=sm_scalar_gauge,
        )
        + _vertex(
            interaction=TERM_complex_scalar_current_phidag,
            external_legs=LEGS_complex_scalar_current,
            species_map=sm_scalar_gauge,
        ),
        simplify_deltas(
            legacy_vertex_factor(**legacy.L_complex_scalar_current_phi, x=x, d=d)
            + legacy_vertex_factor(**legacy.L_complex_scalar_current_phidag, x=x, d=d),
            species_map=sm_scalar_gauge,
        ),
    )
    _check_same(
        "Complex scalar contact",
        _vertex(
            interaction=TERM_complex_scalar_contact,
            external_legs=LEGS_complex_scalar_contact,
            species_map=sm_scalar_contact,
        ),
        simplify_deltas(
            legacy_vertex_factor(**legacy.L_complex_scalar_contact, x=x, d=d),
            species_map=sm_scalar_contact,
        ),
    )

    expected_qg = I * gS * gamma_matrix(i1, i2, mu3) * gauge_generator(a3, c1, c2)
    direct_concrete_term = InteractionTerm(
        coupling=TERM_quark_gluon.coupling,
        fields=(
            QuarkField.occurrence(
                conjugated=True,
                concrete_index_slots=QuarkField.bound_index_slots(
                    _bindings((SPINOR_INDEX, i_bar_q), (COLOR_FUND_INDEX, c_bar_q)),
                    conjugated=True,
                ),
            ),
            QuarkField.occurrence(
                concrete_index_slots=QuarkField.bound_index_slots(
                    _bindings((SPINOR_INDEX, i_psi_q), (COLOR_FUND_INDEX, c_psi_q)),
                ),
            ),
            GluonField.occurrence(
                concrete_index_slots=GluonField.bound_index_slots(
                    _bindings((LORENTZ_INDEX, mu), (COLOR_ADJ_INDEX, a_g)),
                ),
            ),
        ),
    )
    direct_concrete_legs = (
        QuarkField.leg(
            p1,
            conjugated=True,
            spin=s1,
            concrete_index_slots=QuarkField.bound_index_slots(
                _bindings((SPINOR_INDEX, i1), (COLOR_FUND_INDEX, c1)),
                conjugated=True,
            ),
        ),
        QuarkField.leg(
            p2,
            spin=s2,
            concrete_index_slots=QuarkField.bound_index_slots(
                _bindings((SPINOR_INDEX, i2), (COLOR_FUND_INDEX, c2)),
            ),
        ),
        GluonField.leg(
            p3,
            concrete_index_slots=GluonField.bound_index_slots(
                _bindings((LORENTZ_INDEX, mu3), (COLOR_ADJ_INDEX, a3)),
            ),
        ),
    )
    _check_same(
        "Quark-gluon current [concrete slots]",
        _vertex(interaction=direct_concrete_term, external_legs=direct_concrete_legs),
        expected_qg,
    )

    OrderedIndexField = Field(
        name="OrderedIndexField",
        spin=0,
        self_conjugate=True,
        kind="scalar",
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX, LORENTZ_INDEX),
    )
    ordered_occurrence = OrderedIndexField.occurrence(
        slot_labels=bind_indices(
            (LORENTZ_INDEX, (mu, nu)),
            (COLOR_ADJ_INDEX, a3),
        )
    )
    assert tuple(slot.index_type for slot in ordered_occurrence.index_slots) == (
        LORENTZ_INDEX,
        COLOR_ADJ_INDEX,
        LORENTZ_INDEX,
    )
    assert tuple(slot.label for slot in ordered_occurrence.index_slots) == (mu, a3, nu)
    print("  Ordered concrete index slots: PASS")

    try:
        PhiCField.occurrence(role="psi")
        raise AssertionError("Expected ValueError for scalar field with fermion role")
    except ValueError as exc:
        assert "incompatible" in str(exc), exc
    print("  Invalid scalar role: PASS")

    try:
        GaugeField.occurrence(role="vector_dag")
        raise AssertionError("Expected ValueError for self-conjugate field with dag role")
    except ValueError as exc:
        assert "self-conjugate" in str(exc), exc
    print("  Invalid self-conjugate dag role: PASS")

    try:
        InteractionTerm(
            coupling=yF,
            fields=TERM_yukawa.fields,
            statistics="boson",
        )
        raise AssertionError("Expected ValueError for incompatible explicit interaction statistics")
    except ValueError as exc:
        assert "inconsistent" in str(exc), exc
    print("  Incompatible interaction statistics: PASS")

    AntiColorFundIndex = IndexType(
        "ColorFundBar",
        COLOR_FUND_INDEX.representation,
        kind="color_fund_bar",
        aliases=("color_fund_bar",),
    )
    ColoredScalarField = Field(
        "ColoredScalar",
        spin=0,
        self_conjugate=False,
        kind="scalar",
        indices=(COLOR_FUND_INDEX,),
        conjugate_indices=(AntiColorFundIndex,),
    )
    colored_conjugate = ColoredScalarField.occurrence(
        conjugated=True,
        slot_labels=bind_indices((AntiColorFundIndex, legacy.c3)),
    )
    assert colored_conjugate.index_slots[0].index_type == AntiColorFundIndex
    try:
        ColoredScalarField.occurrence(
            conjugated=True,
            slot_labels=bind_indices((COLOR_FUND_INDEX, legacy.c3)),
        )
        raise AssertionError("Expected ValueError for wrong conjugate index signature")
    except ValueError as exc:
        assert "does not declare slot kinds" in str(exc) or "expects" in str(exc), exc
    print("  Conjugate index signature: PASS")

    print("\nAll metadata regressions agree with the legacy examples.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _run_demo():
    print("\n=== metadata-only: scalar sector ===")
    _show(
        TERM_phi4.label,
        _vertex(interaction=TERM_phi4, external_legs=LEGS_phi4, species_map={b1: phi0, b2: phi0, b3: phi0, b4: phi0}),
        interaction=TERM_phi4,
        external_legs=LEGS_phi4,
    )
    _show(
        TERM_phi2chi2.label,
        _vertex(interaction=TERM_phi2chi2, external_legs=LEGS_phi2chi2, species_map={b1: phi0, b2: phi0, b3: chi0, b4: chi0}),
        interaction=TERM_phi2chi2,
        external_legs=LEGS_phi2chi2,
    )
    _show(
        TERM_phiCdag_phiC.label,
        _vertex(interaction=TERM_phiCdag_phiC, external_legs=LEGS_phiCdag_phiC, species_map={b1: phiCdag0, b2: phiC0}),
        interaction=TERM_phiCdag_phiC,
        external_legs=LEGS_phiCdag_phiC,
    )

    print("\n=== metadata-only: fermion sector ===")
    _show(
        TERM_yukawa.label,
        _vertex(interaction=TERM_yukawa, external_legs=LEGS_yukawa),
        interaction=TERM_yukawa,
        external_legs=LEGS_yukawa,
    )
    _show(
        TERM_vec_current.label,
        _vertex(interaction=TERM_vec_current, external_legs=LEGS_vec_current),
        interaction=TERM_vec_current,
        external_legs=LEGS_vec_current,
    )
    _show(
        TERM_axial_current.label,
        _vertex(interaction=TERM_axial_current, external_legs=LEGS_vec_current),
        interaction=TERM_axial_current,
        external_legs=LEGS_vec_current,
    )
    _show(
        TERM_psibar_psi_sq.label,
        _vertex(interaction=TERM_psibar_psi_sq, external_legs=LEGS_fermion4),
        interaction=TERM_psibar_psi_sq,
        external_legs=LEGS_fermion4,
    )
    _show(
        TERM_current_current.label,
        simplify_gamma_chain(_vertex(interaction=TERM_current_current, external_legs=LEGS_fermion4)),
        interaction=TERM_current_current,
        external_legs=LEGS_fermion4,
    )

    print("\n=== metadata-only: gauge-ready sector ===")
    _show(
        TERM_quark_gluon.label,
        _vertex(interaction=TERM_quark_gluon, external_legs=LEGS_quark_gluon),
        interaction=TERM_quark_gluon,
        external_legs=LEGS_quark_gluon,
    )
    current_expr = _vertex(
        interaction=TERM_complex_scalar_current_phi,
        external_legs=LEGS_complex_scalar_current,
        species_map={b1: phiCdag0, b2: phiC0, b3: A0},
    ) + _vertex(
        interaction=TERM_complex_scalar_current_phidag,
        external_legs=LEGS_complex_scalar_current,
        species_map={b1: phiCdag0, b2: phiC0, b3: A0},
    )
    _show(
        "gPhiA * A_mu * phi^dagger <-> d^mu phi",
        current_expr,
        interaction=TERM_complex_scalar_current_phi,
        external_legs=LEGS_complex_scalar_current,
    )
    _show(
        TERM_complex_scalar_contact.label,
        _vertex(
            interaction=TERM_complex_scalar_contact,
            external_legs=LEGS_complex_scalar_contact,
            species_map={b1: phiCdag0, b2: phiC0, b3: A0, b4: A0},
        ),
        interaction=TERM_complex_scalar_contact,
        external_legs=LEGS_complex_scalar_contact,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clean metadata-only vertex examples.")
    parser.add_argument(
        "--mode",
        choices=("demo", "test", "all"),
        default="all",
        help="Run the metadata demo, the regression tests, or both.",
    )
    args = parser.parse_args()

    if args.mode in ("demo", "all"):
        print("\n" + "=" * 80)
        print("  Metadata Demo")
        print("=" * 80)
        _run_demo()

    if args.mode in ("test", "all"):
        print("\n" + "=" * 80)
        print("  Metadata Regression Checks")
        print("=" * 80)
        _run_regression_checks()
