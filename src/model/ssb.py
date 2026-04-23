"""Focused spontaneous-symmetry-breaking helpers for the electroweak sector.

This module deliberately stays small and explicit.  It does not try to turn
the full framework into a generic symmetry-breaking engine; instead it adds the
minimum electroweak SSB layer needed to:

- declare the standard Higgs doublet with ``Y = 1/2``
- expose the textbook Higgs/Goldstone expansion around the vev
- expose the charged and neutral electroweak mixing relations
- build a broken-phase ``Model`` with explicit local interaction terms for the
  Higgs-sector mass terms, Goldstone-vector mixings, flavor-aware Yukawas,
  physical-basis charged currents, and ``R_xi`` ghost interactions

The resulting broken model reuses the existing ``Model`` / ``InteractionTerm`` /
``Lagrangian`` pipeline directly, so all vertex extraction keeps working
without touching the core compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from symbolica import Expression, S
from symbolica.community.spenso import Representation

from symbolic.spenso_structures import chiral_projector_left, gamma_matrix, lorentz_metric

from .core import Model
from .declared import PartialD
from .interactions import DerivativeAction, InteractionTerm
from .lagrangian import DeclaredLagrangian
from .metadata import Field, IndexType, LORENTZ_INDEX, SPINOR_INDEX, WEAK_ADJ_INDEX, WEAK_FUND_INDEX

_HALF = Expression.num(1) / Expression.num(2)
_INV_SQRT2 = _HALF ** _HALF
FLAVOR_INDEX = IndexType("FlavorFund", Representation.cof(3), "flavor", prefix="f")

#===============================================================
#Pretty-printable linear relations for Higgs expansion and gauge mixing
#===============================================================
@dataclass(frozen=True)
class LinearTerm:
    """One term in an inspectable linear field relation.

    ``item`` can be:
    - ``None`` for a constant term
    - a ``Field`` for a physical field
    - a short string like ``"W1"`` or ``"B"`` for gauge-basis labels
    """

    coefficient: object
    item: object | None = None
    conjugated: bool = False

    def display_name(self) -> str:
        if self.item is None:
            return "1"
        if isinstance(self.item, Field):
            if self.conjugated and not self.item.self_conjugate:
                return f"{self.item.name}.bar"
            return self.item.name
        return str(self.item)

    def __str__(self) -> str:
        if self.item is None:
            return str(self.coefficient)
        return f"{self.coefficient} * {self.display_name()}"


@dataclass(frozen=True)
class LinearRelation:
    """Human-readable linear relation used for Higgs expansion or field mixing."""

    target: str
    terms: tuple[LinearTerm, ...]

    def __str__(self) -> str:
        if not self.terms:
            return f"{self.target} = 0"
        return f"{self.target} = " + " + ".join(str(term) for term in self.terms)

#===============================================================
# Convenience data structures for electroweak SSB inputs and outputs
#===============================================================
@dataclass(frozen=True)
class DiagonalYukawaAssignment:
    """Minimal diagonal Yukawa input for one Dirac fermion. 
        m_f = y_f v / sqrt(2)."""

    fermion: Field
    yukawa: object
    label: str = ""

    def mass(self, vev):
        return self.yukawa * vev * _INV_SQRT2


@dataclass(frozen=True)
class FlavorMatrix:
    """Named two-index flavor object such as ``Ye(i,j)`` or ``VCKM(i,j)``."""

    name: str
    symbol: object = None
    dagger_symbol: object | None = None

    def __post_init__(self):
        if self.symbol is None:
            object.__setattr__(self, "symbol", S(self.name))

    def entry(self, left_label, right_label):
        return self.symbol(left_label, right_label)

    def dagger_entry(self, left_label, right_label):
        if self.dagger_symbol is not None:
            return self.dagger_symbol(left_label, right_label)
        return self.symbol(right_label, left_label)


@dataclass(frozen=True)
class FlavorMatrixYukawaAssignment:
    """Matrix-valued broken-phase Yukawa assignment for one flavored Dirac field."""

    fermion: Field
    matrix: FlavorMatrix | object
    label: str = ""

    def mass_entry(self, left_label, right_label, vev):
        return _flavor_matrix_entry(self.matrix, left_label, right_label) * vev * _INV_SQRT2


@dataclass(frozen=True)
class CKMChargedCurrentAssignment:
    """Left-handed charged-current flavor mixing between one up/down multiplet pair."""

    up_field: Field
    down_field: Field
    matrix: FlavorMatrix | object
    label: str = ""


@dataclass(frozen=True)
class ElectroweakMassSpectrum:
    """Symbolic tree-level masses induced by the broken Higgs-sector terms."""

    mw: object
    mz: object
    photon: object
    mh: object | None = None
    fermions: tuple[tuple[Field, object], ...] = ()


@dataclass(frozen=True)
class BrokenElectroweakFields:
    """Convenience bundle of source-basis and physical electroweak fields."""

    higgs_doublet: Field
    weak_gauge: Field
    hypercharge_gauge: Field
    higgs: Field
    goldstone_neutral: Field
    goldstone_charged: Field
    charged_w: Field
    z_boson: Field
    photon: Field
    ghost_charged: Field | None = None
    ghost_z: Field | None = None
    ghost_photon: Field | None = None
    fermions: tuple[Field, ...] = ()


@dataclass(frozen=True)
class ElectroweakGaugeCouplings:
    """Physical electroweak gauge couplings after neutral mixing."""

    electric_charge: object
    g_ww_a: object
    g_ww_z: object


@dataclass(frozen=True)
class HiggsPotentialData:
    """Broken-phase Higgs-potential parameters and derived self-couplings."""

    quartic: object
    mh: object
    mh_sq: object
    cubic_lagrangian_coefficient: object
    quartic_lagrangian_coefficient: object
    hhh_vertex_coefficient: object
    hhhh_vertex_coefficient: object


@dataclass(frozen=True)
class ElectroweakGaugeFixing:
    """Linear ``R_xi`` gauge-fixing parameters in the physical basis."""

    xi_w: object
    xi_z: object
    xi_a: object


@dataclass(frozen=True)
class BrokenElectroweakSector:
    """Inspectible broken-phase electroweak sector plus its compiled model."""

    model: Model
    fields: BrokenElectroweakFields
    higgs_expansion: tuple[LinearRelation, ...]
    charged_mixing: tuple[LinearRelation, ...]
    neutral_mixing: tuple[LinearRelation, ...]
    masses: ElectroweakMassSpectrum
    gauge_couplings: ElectroweakGaugeCouplings | None = None
    higgs_potential: HiggsPotentialData | None = None
    gauge_fixing: ElectroweakGaugeFixing | None = None
    yukawas: tuple[DiagonalYukawaAssignment, ...] = ()
    matrix_yukawas: tuple[FlavorMatrixYukawaAssignment, ...] = ()
    charged_current_mixings: tuple[CKMChargedCurrentAssignment, ...] = ()


def electroweak_gz(g1, g2):
    """Return the usual neutral-current coupling ``g_Z = sqrt(g1^2 + g2^2)``."""

    return (g1**2 + g2**2) ** _HALF


def electroweak_sin_theta_w(g1, g2):
    """Return the symbolic weak-mixing sine ``s_W``."""

    return g1 / electroweak_gz(g1, g2)


def electroweak_cos_theta_w(g1, g2):
    """Return the symbolic weak-mixing cosine ``c_W``."""

    return g2 / electroweak_gz(g1, g2)


def electroweak_mw(g2, vev):
    """Return the charged weak-boson mass ``M_W = g2 v / 2``."""

    return g2 * vev * _HALF


def electroweak_mz(g1, g2, vev):
    """Return the neutral weak-boson mass ``M_Z = g_Z v / 2``."""

    return electroweak_gz(g1, g2) * vev * _HALF


def electroweak_e(g1, g2):
    """Return the electromagnetic coupling ``e = g1 g2 / sqrt(g1^2 + g2^2)``."""

    return g1 * g2 / electroweak_gz(g1, g2)


def electroweak_gwwz(g1, g2):
    """Return the physical ``W W Z`` coupling ``g2 cos(theta_W)``."""

    return g2 * electroweak_cos_theta_w(g1, g2)


def standard_model_higgs_doublet(
    *,
    name: str = "H",
    symbol=None,
    conjugate_symbol=None,
    hypercharge=None,
) -> Field:
    """Return the standard complex electroweak Higgs doublet.

    The declaration stays fully compatible with the existing gauge-group
    machinery: it is just a non-self-conjugate scalar carrying one
    ``WEAK_FUND_INDEX`` slot plus hypercharge ``Y = 1/2`` by default.
    """

    if symbol is None:
        symbol = S(f"{name}0")
    if conjugate_symbol is None:
        conjugate_symbol = S(f"{name}dag0")
    if hypercharge is None:
        hypercharge = _HALF

    return Field(
        name=name,
        spin=0,
        self_conjugate=False,
        symbol=symbol,
        conjugate_symbol=conjugate_symbol,
        indices=(WEAK_FUND_INDEX,),
        quantum_numbers={"Y": hypercharge},
    )


def electroweak_higgs_vev_expansion(
    *,
    vev,
    higgs: Field,
    goldstone_neutral: Field,
    goldstone_charged: Field,
) -> tuple[LinearRelation, ...]:
    """Return the textbook Higgs-doublet component expansion around the vev."""

    return (
        LinearRelation(
            target="H[1]",
            terms=(LinearTerm(Expression.num(1), goldstone_charged),),
        ),
        LinearRelation(
            target="H[2]",
            terms=(
                LinearTerm(vev * _INV_SQRT2),
                LinearTerm(_INV_SQRT2, higgs),
                LinearTerm(Expression.I * _INV_SQRT2, goldstone_neutral),
            ),
        ),
        LinearRelation(
            target="Hdag[1]",
            terms=(LinearTerm(Expression.num(1), goldstone_charged, conjugated=True),),
        ),
        LinearRelation(
            target="Hdag[2]",
            terms=(
                LinearTerm(vev * _INV_SQRT2),
                LinearTerm(_INV_SQRT2, higgs),
                LinearTerm(-Expression.I * _INV_SQRT2, goldstone_neutral),
            ),
        ),
    )


def electroweak_charged_gauge_mixing() -> tuple[LinearRelation, ...]:
    """Return the charged weak-boson basis change ``W1, W2 -> W+, W-``."""

    return (
        LinearRelation(
            target="W+",
            terms=(
                LinearTerm(_INV_SQRT2, "W1"),
                LinearTerm(-Expression.I * _INV_SQRT2, "W2"),
            ),
        ),
        LinearRelation(
            target="W-",
            terms=(
                LinearTerm(_INV_SQRT2, "W1"),
                LinearTerm(Expression.I * _INV_SQRT2, "W2"),
            ),
        ),
    )


def electroweak_neutral_gauge_mixing(*, g1, g2) -> tuple[LinearRelation, ...]:
    """Return the neutral electroweak mixing ``W3, B -> Z, A``."""

    c_w = electroweak_cos_theta_w(g1, g2)
    s_w = electroweak_sin_theta_w(g1, g2)
    return (
        LinearRelation(
            target="Z",
            terms=(
                LinearTerm(c_w, "W3"),
                LinearTerm(-s_w, "B"),
            ),
        ),
        LinearRelation(
            target="A",
            terms=(
                LinearTerm(s_w, "W3"),
                LinearTerm(c_w, "B"),
            ),
        ),
    )


def _default_field(field: Optional[Field], default: Field) -> Field:
    return default if field is None else field


def _unique_fields(*fields: Field | None) -> tuple[Field, ...]:
    ordered: list[Field] = []
    for field in fields:
        if field is None:
            continue
        if field not in ordered:
            ordered.append(field)
    return tuple(ordered)


def _validate_source_higgs_doublet(field: Field):
    if field.kind != "scalar" or field.self_conjugate:
        raise ValueError("Electroweak SSB expects a non-self-conjugate scalar Higgs doublet.")
    if not field.index_positions(index=WEAK_FUND_INDEX):
        raise ValueError("Electroweak SSB expects the Higgs field to carry a weak-fundamental index.")
    if "Y" not in field.quantum_numbers:
        raise ValueError("Electroweak SSB expects the Higgs field to declare hypercharge 'Y'.")


def _single_index_slot(field: Field, index: IndexType, *, purpose: str) -> int:
    slots = field.index_positions(index=index)
    if len(slots) != 1:
        raise ValueError(
            f"{purpose} expects field {field.name!r} to carry exactly one {index.name!r} slot."
        )
    return slots[0]


def _field_slot_symbol(field: Field, slot: int, stem: str):
    index = field.indices[slot]
    return S(f"{index.prefix}_{field.name}_{stem}_{slot}")


def _flavor_matrix_symbol(matrix):
    if isinstance(matrix, FlavorMatrix):
        return matrix.symbol
    return matrix


def _flavor_matrix_entry(matrix, left_label, right_label):
    if isinstance(matrix, FlavorMatrix):
        return matrix.entry(left_label, right_label)
    return matrix(left_label, right_label)


def _flavor_matrix_dagger_entry(matrix, left_label, right_label):
    if isinstance(matrix, FlavorMatrix):
        return matrix.dagger_entry(left_label, right_label)
    return matrix(right_label, left_label)


def _validate_flavored_fermion(field: Field, *, purpose: str):
    if field.kind != "fermion" or field.self_conjugate:
        raise ValueError(f"{purpose} expects a non-self-conjugate fermion field, got {field.name!r}.")
    _single_index_slot(field, SPINOR_INDEX, purpose=purpose)
    _single_index_slot(field, FLAVOR_INDEX, purpose=purpose)


def _fermion_bilinear_occurrences(
    field: Field,
    *,
    spinor_label,
    stem: str,
    left_slot_overrides: dict[int, object] | None = None,
    right_slot_overrides: dict[int, object] | None = None,
):
    spinor_slot = _single_index_slot(field, SPINOR_INDEX, purpose=f"{field.name} bilinear")
    left_slot_labels = {spinor_slot: spinor_label}
    right_slot_labels = {spinor_slot: spinor_label}

    for slot, _index in enumerate(field.indices):
        if slot == spinor_slot:
            continue
        shared = _field_slot_symbol(field, slot, stem)
        left_slot_labels[slot] = shared
        right_slot_labels[slot] = shared

    if left_slot_overrides:
        left_slot_labels.update(left_slot_overrides)
    if right_slot_overrides:
        right_slot_labels.update(right_slot_overrides)

    return (
        field.occurrence(conjugated=True, labels=field.pack_slot_labels(left_slot_labels)),
        field.occurrence(labels=field.pack_slot_labels(right_slot_labels)),
    )


def _paired_fermion_occurrences(
    left_field: Field,
    right_field: Field,
    *,
    left_spinor_label,
    right_spinor_label,
    stem: str,
    left_slot_overrides: dict[int, object] | None = None,
    right_slot_overrides: dict[int, object] | None = None,
):
    left_spinor_slot = _single_index_slot(left_field, SPINOR_INDEX, purpose=f"{left_field.name}/{right_field.name} current")
    right_spinor_slot = _single_index_slot(right_field, SPINOR_INDEX, purpose=f"{left_field.name}/{right_field.name} current")

    if len(left_field.indices) != len(right_field.indices):
        raise ValueError(
            f"Charged-current pair {left_field.name!r}/{right_field.name!r} must carry aligned index slots."
        )

    left_slot_labels = {left_spinor_slot: left_spinor_label}
    right_slot_labels = {right_spinor_slot: right_spinor_label}

    for slot, (left_index, right_index) in enumerate(zip(left_field.indices, right_field.indices, strict=True)):
        if slot in (left_spinor_slot, right_spinor_slot):
            continue
        if left_index != right_index:
            raise ValueError(
                f"Charged-current pair {left_field.name!r}/{right_field.name!r} has mismatched index slot {slot}."
            )
        shared = S(f"{left_index.prefix}_{left_field.name}_{right_field.name}_{stem}_{slot}")
        left_slot_labels[slot] = shared
        right_slot_labels[slot] = shared

    if left_slot_overrides:
        left_slot_labels.update(left_slot_overrides)
    if right_slot_overrides:
        right_slot_labels.update(right_slot_overrides)

    return (
        left_field.occurrence(conjugated=True, labels=left_field.pack_slot_labels(left_slot_labels)),
        right_field.occurrence(labels=right_field.pack_slot_labels(right_slot_labels)),
    )


def _vector_occurrence(field: Field, lorentz_label, *, conjugated: bool = False):
    return field.occurrence(conjugated=conjugated, labels={LORENTZ_INDEX.kind: lorentz_label})


def _real_vector_kinetic_terms(*, vector: Field, label_prefix: str = "") -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{vector.name}_kin")
    beta = S(f"beta_{vector.name}_kin")
    rho = S(f"rho_{vector.name}_kin")
    rho_left = S(f"rho_left_{vector.name}_kin")
    rho_right = S(f"rho_right_{vector.name}_kin")
    prefix = label_prefix + " " if label_prefix else ""
    fields = (
        _vector_occurrence(vector, alpha),
        _vector_occurrence(vector, beta),
    )
    return (
        InteractionTerm(
            coupling=-_HALF * lorentz_metric(alpha, beta),
            fields=fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho),
                DerivativeAction(target=1, lorentz_index=rho),
            ),
            label=prefix + f"{vector.name}: gauge kinetic bilinear (metric)",
        ),
        InteractionTerm(
            coupling=_HALF * lorentz_metric(rho_left, beta) * lorentz_metric(rho_right, alpha),
            fields=fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=prefix + f"{vector.name}: gauge kinetic bilinear (cross)",
        ),
    )


def _complex_vector_kinetic_terms(*, vector: Field, label_prefix: str = "") -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{vector.name}_kin")
    beta = S(f"beta_{vector.name}_kin")
    rho = S(f"rho_{vector.name}_kin")
    rho_left = S(f"rho_left_{vector.name}_kin")
    rho_right = S(f"rho_right_{vector.name}_kin")
    prefix = label_prefix + " " if label_prefix else ""
    fields = (
        _vector_occurrence(vector, alpha, conjugated=True),
        _vector_occurrence(vector, beta),
    )
    return (
        InteractionTerm(
            coupling=-lorentz_metric(alpha, beta),
            fields=fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho),
                DerivativeAction(target=1, lorentz_index=rho),
            ),
            label=prefix + f"{vector.name}: charged-vector kinetic bilinear (metric)",
        ),
        InteractionTerm(
            coupling=lorentz_metric(rho_left, beta) * lorentz_metric(rho_right, alpha),
            fields=fields,
            derivatives=(
                DerivativeAction(target=0, lorentz_index=rho_left),
                DerivativeAction(target=1, lorentz_index=rho_right),
            ),
            label=prefix + f"{vector.name}: charged-vector kinetic bilinear (cross)",
        ),
    )


def _real_vector_gauge_fixing_term(*, vector: Field, xi, label: str = "") -> InteractionTerm:
    alpha = S(f"alpha_{vector.name}_gf")
    beta = S(f"beta_{vector.name}_gf")
    rho_left = S(f"rho_left_{vector.name}_gf")
    rho_right = S(f"rho_right_{vector.name}_gf")
    return InteractionTerm(
        coupling=(
            -_HALF
            / xi
            * lorentz_metric(alpha, rho_left)
            * lorentz_metric(beta, rho_right)
        ),
        fields=(
            _vector_occurrence(vector, alpha),
            _vector_occurrence(vector, beta),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=rho_left),
            DerivativeAction(target=1, lorentz_index=rho_right),
        ),
        label=label or f"EW R_xi: {vector.name} gauge fixing",
    )


def _complex_vector_gauge_fixing_term(*, vector: Field, xi, label: str = "") -> InteractionTerm:
    alpha = S(f"alpha_{vector.name}_gf")
    beta = S(f"beta_{vector.name}_gf")
    rho_left = S(f"rho_left_{vector.name}_gf")
    rho_right = S(f"rho_right_{vector.name}_gf")
    return InteractionTerm(
        coupling=(
            -Expression.num(1)
            / xi
            * lorentz_metric(alpha, rho_left)
            * lorentz_metric(beta, rho_right)
        ),
        fields=(
            _vector_occurrence(vector, alpha, conjugated=True),
            _vector_occurrence(vector, beta),
        ),
        derivatives=(
            DerivativeAction(target=0, lorentz_index=rho_left),
            DerivativeAction(target=1, lorentz_index=rho_right),
        ),
        label=label or f"EW R_xi: {vector.name} gauge fixing",
    )


def _wwv_cubic_terms(
    *,
    charged_w: Field,
    neutral_vector: Field,
    coupling,
    label: str = "",
) -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{charged_w.name}_{neutral_vector.name}_cubic")
    beta = S(f"beta_{charged_w.name}_{neutral_vector.name}_cubic")
    gamma = S(f"gamma_{charged_w.name}_{neutral_vector.name}_cubic")
    rho_left = S(f"rho_left_{charged_w.name}_{neutral_vector.name}_cubic")
    rho_mid = S(f"rho_mid_{charged_w.name}_{neutral_vector.name}_cubic")
    rho_right = S(f"rho_right_{charged_w.name}_{neutral_vector.name}_cubic")
    fields = (
        _vector_occurrence(charged_w, alpha, conjugated=True),
        _vector_occurrence(charged_w, beta),
        _vector_occurrence(neutral_vector, gamma),
    )
    prefactor = Expression.I * coupling
    vertex_label = label or f"EW gauge: {charged_w.name} {neutral_vector.name} cubic"
    return (
        InteractionTerm(
            coupling=prefactor * lorentz_metric(alpha, beta) * lorentz_metric(rho_left, gamma),
            fields=fields,
            derivatives=(DerivativeAction(target=0, lorentz_index=rho_left),),
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-prefactor * lorentz_metric(gamma, alpha) * lorentz_metric(rho_left, beta),
            fields=fields,
            derivatives=(DerivativeAction(target=0, lorentz_index=rho_left),),
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-prefactor * lorentz_metric(alpha, beta) * lorentz_metric(rho_mid, gamma),
            fields=fields,
            derivatives=(DerivativeAction(target=1, lorentz_index=rho_mid),),
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=prefactor * lorentz_metric(beta, gamma) * lorentz_metric(rho_mid, alpha),
            fields=fields,
            derivatives=(DerivativeAction(target=1, lorentz_index=rho_mid),),
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-prefactor * lorentz_metric(beta, gamma) * lorentz_metric(rho_right, alpha),
            fields=fields,
            derivatives=(DerivativeAction(target=2, lorentz_index=rho_right),),
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=prefactor * lorentz_metric(gamma, alpha) * lorentz_metric(rho_right, beta),
            fields=fields,
            derivatives=(DerivativeAction(target=2, lorentz_index=rho_right),),
            label=vertex_label,
        ),
    )


def _wwv_contact_terms_identical(
    *,
    charged_w: Field,
    neutral_vector: Field,
    coupling,
    label: str = "",
) -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{charged_w.name}_{neutral_vector.name}_quartic")
    beta = S(f"beta_{charged_w.name}_{neutral_vector.name}_quartic")
    gamma = S(f"gamma_{charged_w.name}_{neutral_vector.name}_quartic")
    delta = S(f"delta_{charged_w.name}_{neutral_vector.name}_quartic")
    fields = (
        _vector_occurrence(charged_w, alpha, conjugated=True),
        _vector_occurrence(charged_w, beta),
        _vector_occurrence(neutral_vector, gamma),
        _vector_occurrence(neutral_vector, delta),
    )
    vertex_label = label or f"EW gauge: {charged_w.name} {neutral_vector.name}{neutral_vector.name} quartic"
    return (
        InteractionTerm(
            coupling=coupling * lorentz_metric(alpha, beta) * lorentz_metric(gamma, delta),
            fields=fields,
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-coupling * lorentz_metric(alpha, gamma) * lorentz_metric(beta, delta),
            fields=fields,
            label=vertex_label,
        ),
    )


def _wwv_contact_terms_distinct(
    *,
    charged_w: Field,
    left_neutral: Field,
    right_neutral: Field,
    coupling,
    label: str = "",
) -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{charged_w.name}_{left_neutral.name}_{right_neutral.name}_quartic")
    beta = S(f"beta_{charged_w.name}_{left_neutral.name}_{right_neutral.name}_quartic")
    gamma = S(f"gamma_{charged_w.name}_{left_neutral.name}_{right_neutral.name}_quartic")
    delta = S(f"delta_{charged_w.name}_{left_neutral.name}_{right_neutral.name}_quartic")
    fields = (
        _vector_occurrence(charged_w, alpha, conjugated=True),
        _vector_occurrence(charged_w, beta),
        _vector_occurrence(left_neutral, gamma),
        _vector_occurrence(right_neutral, delta),
    )
    vertex_label = label or f"EW gauge: {charged_w.name} {left_neutral.name} {right_neutral.name} quartic"
    return (
        InteractionTerm(
            coupling=2 * coupling * lorentz_metric(alpha, beta) * lorentz_metric(gamma, delta),
            fields=fields,
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-coupling * lorentz_metric(alpha, gamma) * lorentz_metric(beta, delta),
            fields=fields,
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-coupling * lorentz_metric(alpha, delta) * lorentz_metric(beta, gamma),
            fields=fields,
            label=vertex_label,
        ),
    )


def _four_w_contact_terms(*, charged_w: Field, coupling, label: str = "") -> tuple[InteractionTerm, ...]:
    alpha = S(f"alpha_{charged_w.name}_4W")
    beta = S(f"beta_{charged_w.name}_4W")
    gamma = S(f"gamma_{charged_w.name}_4W")
    delta = S(f"delta_{charged_w.name}_4W")
    fields = (
        _vector_occurrence(charged_w, alpha, conjugated=True),
        _vector_occurrence(charged_w, beta),
        _vector_occurrence(charged_w, gamma, conjugated=True),
        _vector_occurrence(charged_w, delta),
    )
    vertex_label = label or f"EW gauge: {charged_w.name} 4-point"
    return (
        InteractionTerm(
            coupling=_HALF * coupling * lorentz_metric(alpha, beta) * lorentz_metric(gamma, delta),
            fields=fields,
            label=vertex_label,
        ),
        InteractionTerm(
            coupling=-_HALF * coupling * lorentz_metric(alpha, gamma) * lorentz_metric(beta, delta),
            fields=fields,
            label=vertex_label,
        ),
    )


def _ghost_bilinear_term(*, ghost: Field, mass_sq=Expression.num(0), label: str = "") -> tuple[object, ...]:
    kinetic_mu = S(f"mu_{ghost.name}_ghost")
    kinetic_nu = S(f"nu_{ghost.name}_ghost")
    terms: list[object] = [
        InteractionTerm(
            coupling=lorentz_metric(kinetic_mu, kinetic_nu),
            fields=(
                ghost.occurrence(conjugated=True),
                ghost.occurrence(),
            ),
            derivatives=(
                DerivativeAction(target=0, lorentz_index=kinetic_mu),
                DerivativeAction(target=1, lorentz_index=kinetic_nu),
            ),
            label=label or f"EW ghost: {ghost.name} kinetic",
        )
    ]
    if mass_sq != 0:
        terms.append(
            _ghost_mass_term(
                coefficient=-mass_sq,
                ghost=ghost,
                label=label or f"EW ghost: {ghost.name} mass",
            )
        )
    return tuple(terms)


def _build_physical_gauge_terms(
    *,
    g1,
    g2,
    charged_w: Field,
    z_boson: Field,
    photon: Field,
) -> tuple[tuple[InteractionTerm, ...], ElectroweakGaugeCouplings]:
    electric_charge = electroweak_e(g1, g2)
    g_ww_z = electroweak_gwwz(g1, g2)
    terms = (
        *_complex_vector_kinetic_terms(vector=charged_w, label_prefix="EW gauge"),
        *_real_vector_kinetic_terms(vector=z_boson, label_prefix="EW gauge"),
        *_real_vector_kinetic_terms(vector=photon, label_prefix="EW gauge"),
        *_wwv_cubic_terms(charged_w=charged_w, neutral_vector=photon, coupling=electric_charge, label="EW gauge: W W A"),
        *_wwv_cubic_terms(charged_w=charged_w, neutral_vector=z_boson, coupling=g_ww_z, label="EW gauge: W W Z"),
        *_wwv_contact_terms_identical(charged_w=charged_w, neutral_vector=photon, coupling=electric_charge**2, label="EW gauge: W W A A"),
        *_wwv_contact_terms_distinct(charged_w=charged_w, left_neutral=photon, right_neutral=z_boson, coupling=electric_charge * g_ww_z, label="EW gauge: W W A Z"),
        *_wwv_contact_terms_identical(charged_w=charged_w, neutral_vector=z_boson, coupling=g_ww_z**2, label="EW gauge: W W Z Z"),
        *_four_w_contact_terms(charged_w=charged_w, coupling=g2**2, label="EW gauge: W W W W"),
    )
    return terms, ElectroweakGaugeCouplings(
        electric_charge=electric_charge,
        g_ww_a=electric_charge,
        g_ww_z=g_ww_z,
    )


def _build_higgs_potential_terms(
    *,
    quartic,
    vev,
    higgs: Field,
    goldstone_neutral: Field,
    goldstone_charged: Field,
) -> tuple[tuple[object, ...], HiggsPotentialData]:
    mh_sq = 2 * quartic * vev**2
    mh = mh_sq ** _HALF
    terms: tuple[object, ...] = (
        -(quartic * vev**2) * higgs * higgs,
        -(quartic * vev) * higgs * higgs * higgs,
        -(quartic / 4) * higgs * higgs * higgs * higgs,
        -(quartic * vev) * higgs * goldstone_neutral * goldstone_neutral,
        -(2 * quartic * vev) * higgs * goldstone_charged.bar * goldstone_charged,
        -(quartic / 2) * higgs * higgs * goldstone_neutral * goldstone_neutral,
        -quartic * higgs * higgs * goldstone_charged.bar * goldstone_charged,
        -(quartic / 4) * goldstone_neutral * goldstone_neutral * goldstone_neutral * goldstone_neutral,
        -quartic * goldstone_neutral * goldstone_neutral * goldstone_charged.bar * goldstone_charged,
        -quartic * goldstone_charged.bar * goldstone_charged * goldstone_charged.bar * goldstone_charged,
    )
    return terms, HiggsPotentialData(
        quartic=quartic,
        mh=mh,
        mh_sq=mh_sq,
        cubic_lagrangian_coefficient=quartic * vev,
        quartic_lagrangian_coefficient=quartic / 4,
        hhh_vertex_coefficient=-6 * quartic * vev,
        hhhh_vertex_coefficient=-6 * quartic,
    )


def _vector_pair_interaction(
    *,
    coefficient,
    left_vector: Field,
    right_vector: Field,
    scalars: tuple[Field, ...] = (),
    label: str = "",
) -> InteractionTerm:
    mu = S("mu_ssb")
    nu = S("nu_ssb")
    return InteractionTerm(
        coupling=coefficient * lorentz_metric(mu, nu),
        fields=(
            *(scalar.occurrence() for scalar in scalars),
            left_vector.occurrence(conjugated=True, labels={LORENTZ_INDEX.kind: mu})
            if not left_vector.self_conjugate
            else left_vector.occurrence(labels={LORENTZ_INDEX.kind: mu}),
            right_vector.occurrence(labels={LORENTZ_INDEX.kind: nu}),
        ),
        label=label,
    )


def _self_conjugate_vector_pair_interaction(
    *,
    coefficient,
    vector: Field,
    scalars: tuple[Field, ...] = (),
    label: str = "",
) -> InteractionTerm:
    mu = S("mu_ssb")
    nu = S("nu_ssb")
    return InteractionTerm(
        coupling=coefficient * lorentz_metric(mu, nu),
        fields=(
            *(scalar.occurrence() for scalar in scalars),
            vector.occurrence(labels={LORENTZ_INDEX.kind: mu}),
            vector.occurrence(labels={LORENTZ_INDEX.kind: nu}),
        ),
        label=label,
    )


def _scalar_vector_mixing_interaction(
    *,
    coefficient,
    scalar: Field,
    vector: Field,
    scalar_conjugated: bool = False,
    vector_conjugated: bool = False,
    label: str = "",
) -> InteractionTerm:
    mu = S("mu_ssb_mix")
    nu = S("nu_ssb_mix")
    return InteractionTerm(
        coupling=coefficient * lorentz_metric(mu, nu),
        fields=(
            scalar.occurrence(conjugated=scalar_conjugated),
            vector.occurrence(conjugated=vector_conjugated, labels={LORENTZ_INDEX.kind: nu}),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=mu),),
        label=label,
    )


def _fermion_bilinear_term(
    *,
    coefficient,
    fermion: Field,
    scalar: Field | None = None,
    label: str = "",
) -> InteractionTerm:
    spinor = S(f"alpha_{fermion.name}_ssb")
    bar_occurrence, fermion_occurrence = _fermion_bilinear_occurrences(
        fermion,
        spinor_label=spinor,
        stem="ssb",
    )
    fields = [
        bar_occurrence,
        fermion_occurrence,
    ]
    if scalar is not None:
        fields.append(scalar.occurrence())
    return InteractionTerm(
        coupling=coefficient,
        fields=tuple(fields),
        label=label,
    )


def _fermion_matrix_bilinear_term(
    *,
    coefficient,
    matrix,
    fermion: Field,
    scalar: Field | None = None,
    label: str = "",
) -> InteractionTerm:
    _validate_flavored_fermion(fermion, purpose="Matrix Yukawa assignment")
    spinor = S(f"alpha_{fermion.name}_matrix_ssb")
    flavor_slot = _single_index_slot(fermion, FLAVOR_INDEX, purpose=f"{fermion.name} matrix Yukawa")
    left_flavor = S(f"fL_{fermion.name}_matrix")
    right_flavor = S(f"fR_{fermion.name}_matrix")
    bar_occurrence, fermion_occurrence = _fermion_bilinear_occurrences(
        fermion,
        spinor_label=spinor,
        stem="matrix_ssb",
        left_slot_overrides={flavor_slot: left_flavor},
        right_slot_overrides={flavor_slot: right_flavor},
    )
    fields = [
        bar_occurrence,
        fermion_occurrence,
    ]
    if scalar is not None:
        fields.append(scalar.occurrence())
    return InteractionTerm(
        coupling=coefficient * _flavor_matrix_entry(matrix, left_flavor, right_flavor),
        fields=tuple(fields),
        label=label,
    )


def _charged_current_interaction(
    *,
    coefficient,
    matrix,
    left_fermion: Field,
    right_fermion: Field,
    vector: Field,
    vector_conjugated: bool = False,
    dagger_matrix: bool = False,
    label: str = "",
) -> InteractionTerm:
    _validate_flavored_fermion(left_fermion, purpose="Charged-current mixing")
    _validate_flavored_fermion(right_fermion, purpose="Charged-current mixing")
    left_flavor_slot = _single_index_slot(left_fermion, FLAVOR_INDEX, purpose=f"{left_fermion.name}/{right_fermion.name} current")
    right_flavor_slot = _single_index_slot(right_fermion, FLAVOR_INDEX, purpose=f"{left_fermion.name}/{right_fermion.name} current")
    spinor_left = S(f"alpha_{left_fermion.name}_{right_fermion.name}_cc_left")
    spinor_mid = S(f"alpha_{left_fermion.name}_{right_fermion.name}_cc_mid")
    spinor_right = S(f"alpha_{left_fermion.name}_{right_fermion.name}_cc_right")
    left_flavor = S(f"fL_{left_fermion.name}_{right_fermion.name}_cc")
    right_flavor = S(f"fR_{left_fermion.name}_{right_fermion.name}_cc")
    lorentz = S(f"mu_{left_fermion.name}_{right_fermion.name}_{vector.name}_cc")
    bar_occurrence, fermion_occurrence = _paired_fermion_occurrences(
        left_fermion,
        right_fermion,
        left_spinor_label=spinor_left,
        right_spinor_label=spinor_right,
        stem="cc",
        left_slot_overrides={left_flavor_slot: left_flavor},
        right_slot_overrides={right_flavor_slot: right_flavor},
    )
    matrix_entry = (
        _flavor_matrix_dagger_entry(matrix, left_flavor, right_flavor)
        if dagger_matrix
        else _flavor_matrix_entry(matrix, left_flavor, right_flavor)
    )
    return InteractionTerm(
        coupling=(
            coefficient
            * matrix_entry
            * gamma_matrix(spinor_left, spinor_mid, lorentz)
            * chiral_projector_left(spinor_mid, spinor_right)
        ),
        fields=(
            bar_occurrence,
            fermion_occurrence,
            vector.occurrence(conjugated=vector_conjugated, labels={LORENTZ_INDEX.kind: lorentz}),
        ),
        label=label,
    )


def _ghost_scalar_interaction(
    *,
    coefficient,
    antighost: Field,
    ghost: Field,
    scalar: Field,
    scalar_conjugated: bool = False,
    label: str = "",
) -> InteractionTerm:
    fields = [
        antighost.occurrence(conjugated=True),
        ghost.occurrence(),
    ]
    fields.append(scalar.occurrence(conjugated=scalar_conjugated))
    return InteractionTerm(
        coupling=coefficient,
        fields=tuple(fields),
        label=label,
    )


def _ghost_vector_interaction(
    *,
    coefficient,
    antighost: Field,
    vector: Field,
    ghost: Field,
    vector_conjugated: bool = False,
    label: str = "",
) -> InteractionTerm:
    rho = S(f"rho_{antighost.name}_{vector.name}_{ghost.name}_ghost")
    mu = S(f"mu_{antighost.name}_{vector.name}_{ghost.name}_ghost")
    return InteractionTerm(
        coupling=coefficient * lorentz_metric(rho, mu),
        fields=(
            antighost.occurrence(conjugated=True),
            vector.occurrence(conjugated=vector_conjugated, labels={LORENTZ_INDEX.kind: mu}),
            ghost.occurrence(),
        ),
        derivatives=(DerivativeAction(target=0, lorentz_index=rho),),
        label=label,
    )


def _ghost_mass_term(*, ghost: Field, coefficient, scalar: Field | None = None, label: str = "") -> InteractionTerm:
    fields = [
        ghost.occurrence(conjugated=True),
        ghost.occurrence(),
    ]
    if scalar is not None:
        fields.append(scalar.occurrence())
    return InteractionTerm(
        coupling=coefficient,
        fields=tuple(fields),
        label=label,
    )


def _build_physical_ghost_terms(
    *,
    g1,
    g2,
    vev,
    gauge_fixing: ElectroweakGaugeFixing,
    higgs: Field,
    goldstone_neutral: Field,
    goldstone_charged: Field,
    charged_w: Field,
    z_boson: Field,
    photon: Field,
    ghost_charged: Field,
    ghost_z: Field,
    ghost_photon: Field,
) -> tuple[object, ...]:
    mw = electroweak_mw(g2, vev)
    mz = electroweak_mz(g1, g2, vev)
    mw_sq = mw**2
    mz_sq = mz**2
    electric_charge = electroweak_e(g1, g2)
    g_ww_z = electroweak_gwwz(g1, g2)
    charged_goldstone_z = (g2**2 - g1**2) / (2 * electroweak_gz(g1, g2))

    return (
        *_ghost_bilinear_term(
            ghost=ghost_charged,
            mass_sq=gauge_fixing.xi_w * mw_sq,
            label="EW ghost: charged ghost",
        ),
        *_ghost_bilinear_term(
            ghost=ghost_z,
            mass_sq=gauge_fixing.xi_z * mz_sq,
            label="EW ghost: Z ghost",
        ),
        *_ghost_bilinear_term(
            ghost=ghost_photon,
            mass_sq=Expression.num(0),
            label="EW ghost: photon ghost",
        ),
        _ghost_mass_term(
            ghost=ghost_charged,
            scalar=higgs,
            coefficient=-(gauge_fixing.xi_w * mw_sq / vev),
            label="EW ghost: h cWbar cW",
        ),
        _ghost_mass_term(
            ghost=ghost_z,
            scalar=higgs,
            coefficient=-(gauge_fixing.xi_z * mz_sq / vev),
            label="EW ghost: h cZbar cZ",
        ),
        _ghost_vector_interaction(
            coefficient=-Expression.I * electric_charge,
            antighost=ghost_charged,
            vector=photon,
            ghost=ghost_charged,
            label="EW ghost: cWbar A cW",
        ),
        _ghost_vector_interaction(
            coefficient=-Expression.I * g_ww_z,
            antighost=ghost_charged,
            vector=z_boson,
            ghost=ghost_charged,
            label="EW ghost: cWbar Z cW",
        ),
        _ghost_vector_interaction(
            coefficient=Expression.I * electric_charge,
            antighost=ghost_charged,
            vector=charged_w,
            ghost=ghost_photon,
            label="EW ghost: cWbar W+ cA",
        ),
        _ghost_vector_interaction(
            coefficient=Expression.I * g_ww_z,
            antighost=ghost_charged,
            vector=charged_w,
            ghost=ghost_z,
            label="EW ghost: cWbar W+ cZ",
        ),
        _ghost_vector_interaction(
            coefficient=Expression.I * electric_charge,
            antighost=ghost_photon,
            vector=charged_w,
            ghost=ghost_charged,
            vector_conjugated=True,
            label="EW ghost: cAbar W- cW",
        ),
        _ghost_vector_interaction(
            coefficient=Expression.I * g_ww_z,
            antighost=ghost_z,
            vector=charged_w,
            ghost=ghost_charged,
            vector_conjugated=True,
            label="EW ghost: cZbar W- cW",
        ),
        _ghost_scalar_interaction(
            coefficient=-(Expression.I * gauge_fixing.xi_w * mw * g2 * _HALF),
            antighost=ghost_charged,
            ghost=ghost_charged,
            scalar=goldstone_neutral,
            label="EW ghost: cWbar G0 cW",
        ),
        _ghost_scalar_interaction(
            coefficient=-(gauge_fixing.xi_w * mw * electric_charge),
            antighost=ghost_charged,
            ghost=ghost_photon,
            scalar=goldstone_charged,
            label="EW ghost: cWbar G+ cA",
        ),
        _ghost_scalar_interaction(
            coefficient=-(gauge_fixing.xi_w * mw * charged_goldstone_z),
            antighost=ghost_charged,
            ghost=ghost_z,
            scalar=goldstone_charged,
            label="EW ghost: cWbar G+ cZ",
        ),
        _ghost_scalar_interaction(
            coefficient=gauge_fixing.xi_z * mz * g2 * _HALF,
            antighost=ghost_z,
            ghost=ghost_charged,
            scalar=goldstone_charged,
            scalar_conjugated=True,
            label="EW ghost: cZbar G- cW",
        ),
    )


def build_broken_electroweak_sector(
    *,
    g1,
    g2,
    vev,
    include_gauge_sector: bool = False,
    higgs_quartic=None,
    gauge_fixing: ElectroweakGaugeFixing | None = None,
    higgs_doublet: Field | None = None,
    weak_gauge: Field | None = None,
    hypercharge_gauge: Field | None = None,
    higgs: Field | None = None,
    goldstone_neutral: Field | None = None,
    goldstone_charged: Field | None = None,
    charged_w: Field | None = None,
    z_boson: Field | None = None,
    photon: Field | None = None,
    ghost_charged: Field | None = None,
    ghost_z: Field | None = None,
    ghost_photon: Field | None = None,
    yukawas: tuple[DiagonalYukawaAssignment, ...] = (),
    matrix_yukawas: tuple[FlavorMatrixYukawaAssignment, ...] = (),
    charged_current_mixings: tuple[CKMChargedCurrentAssignment, ...] = (),
    name: str = "EW-broken-phase",
) -> BrokenElectroweakSector:
    """Build a compact broken electroweak sector as an explicit local model.

    Scope of the returned model:
    - canonical kinetic terms for ``h``, ``G0``, and ``G+``
    - Higgs-induced ``W`` and ``Z`` mass terms
    - explicit ``hWW`` and ``hZZ`` trilinears
    - Goldstone-vector bilinear mixings from the broken Higgs kinetic term
    - diagonal fermion mass terms plus ``h fbar f`` couplings
    - flavor-matrix Yukawa mass terms plus broken-phase ``h fbar_i f_j`` couplings
    - explicit left-chiral charged currents with optional CKM-like matrices
    - optional physical-basis gauge kinetic / self-interaction terms
    - optional Higgs-potential mass and self-coupling terms
    - optional broken-phase ``R_xi`` gauge-fixing bilinears plus physical-basis ghost terms

    The pure gauge kinetic / self-interaction sector is intentionally left in
    the unbroken compiler unless ``include_gauge_sector=True`` is requested.
    The ghost layer stays explicit and inspectable in the physical basis:
    antighost-charged-ghost couplings to ``A``, ``Z``, ``W`` and the Goldstones
    are assembled directly from the textbook ``R_xi`` gauge-fixing functions.
    """

    source_higgs = _default_field(
        higgs_doublet,
        standard_model_higgs_doublet(),
    )
    _validate_source_higgs_doublet(source_higgs)
    source_weak = _default_field(
        weak_gauge,
        Field(
            "W",
            spin=1,
            self_conjugate=True,
            symbol=S("W0"),
            indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
        ),
    )
    source_hypercharge = _default_field(
        hypercharge_gauge,
        Field(
            "B",
            spin=1,
            self_conjugate=True,
            symbol=S("B0"),
            indices=(LORENTZ_INDEX,),
        ),
    )

    h_field = _default_field(
        higgs,
        Field("h", spin=0, self_conjugate=True, symbol=S("h0")),
    )
    g0_field = _default_field(
        goldstone_neutral,
        Field("G0", spin=0, self_conjugate=True, symbol=S("G00")),
    )
    gp_field = _default_field(
        goldstone_charged,
        Field(
            "Gp",
            spin=0,
            self_conjugate=False,
            symbol=S("Gp0"),
            conjugate_symbol=S("Gm0"),
        ),
    )
    wp_field = _default_field(
        charged_w,
        Field(
            "Wp",
            spin=1,
            self_conjugate=False,
            symbol=S("Wp0"),
            conjugate_symbol=S("Wm0"),
            indices=(LORENTZ_INDEX,),
        ),
    )
    z_field = _default_field(
        z_boson,
        Field("Z", spin=1, self_conjugate=True, symbol=S("Z0"), indices=(LORENTZ_INDEX,)),
    )
    a_field = _default_field(
        photon,
        Field("A", spin=1, self_conjugate=True, symbol=S("A0"), indices=(LORENTZ_INDEX,)),
    )
    gh_wp_field = None
    gh_z_field = None
    gh_a_field = None
    if gauge_fixing is not None:
        gh_wp_field = _default_field(
            ghost_charged,
            Field(
                "ghWp",
                spin=0,
                kind="ghost",
                self_conjugate=False,
                symbol=S("ghWp0"),
                conjugate_symbol=S("ghWm0"),
            ),
        )
        gh_z_field = _default_field(
            ghost_z,
            Field(
                "ghZ",
                spin=0,
                kind="ghost",
                self_conjugate=False,
                symbol=S("ghZ0"),
                conjugate_symbol=S("ghZbar0"),
            ),
        )
        gh_a_field = _default_field(
            ghost_photon,
            Field(
                "ghA",
                spin=0,
                kind="ghost",
                self_conjugate=False,
                symbol=S("ghA0"),
                conjugate_symbol=S("ghAbar0"),
            ),
        )

    mw = electroweak_mw(g2, vev)
    mz = electroweak_mz(g1, g2, vev)
    mw_sq = mw**2
    mz_sq = mz**2
    gauge_couplings = None
    higgs_potential_data = None

    lagrangian_terms: list[object] = [
        _HALF * PartialD(h_field, S("mu")) * PartialD(h_field, S("mu")),
        _HALF * PartialD(g0_field, S("mu")) * PartialD(g0_field, S("mu")),
        PartialD(gp_field.bar, S("mu")) * PartialD(gp_field, S("mu")),
        _vector_pair_interaction(
            coefficient=mw_sq,
            left_vector=wp_field,
            right_vector=wp_field,
            label="EW SSB: W mass term",
        ),
        _self_conjugate_vector_pair_interaction(
            coefficient=mz_sq * _HALF,
            vector=z_field,
            label="EW SSB: Z mass term",
        ),
        _vector_pair_interaction(
            coefficient=g2**2 * vev * _HALF,
            left_vector=wp_field,
            right_vector=wp_field,
            scalars=(h_field,),
            label="EW SSB: h W+ W-",
        ),
        _self_conjugate_vector_pair_interaction(
            coefficient=mz_sq / vev,
            vector=z_field,
            scalars=(h_field,),
            label="EW SSB: h Z Z",
        ),
        _scalar_vector_mixing_interaction(
            coefficient=mw,
            scalar=gp_field,
            vector=wp_field,
            vector_conjugated=True,
            label="EW SSB: G+ / W- mixing",
        ),
        _scalar_vector_mixing_interaction(
            coefficient=mw,
            scalar=gp_field,
            vector=wp_field,
            scalar_conjugated=True,
            label="EW SSB: G- / W+ mixing",
        ),
        _scalar_vector_mixing_interaction(
            coefficient=mz,
            scalar=g0_field,
            vector=z_field,
            label="EW SSB: G0 / Z mixing",
        ),
    ]

    if include_gauge_sector:
        gauge_terms, gauge_couplings = _build_physical_gauge_terms(
            g1=g1,
            g2=g2,
            charged_w=wp_field,
            z_boson=z_field,
            photon=a_field,
        )
        lagrangian_terms.extend(gauge_terms)

    if higgs_quartic is not None:
        potential_terms, higgs_potential_data = _build_higgs_potential_terms(
            quartic=higgs_quartic,
            vev=vev,
            higgs=h_field,
            goldstone_neutral=g0_field,
            goldstone_charged=gp_field,
        )
        lagrangian_terms.extend(potential_terms)

    if gauge_fixing is not None:
        lagrangian_terms.extend((
            _complex_vector_gauge_fixing_term(
                vector=wp_field,
                xi=gauge_fixing.xi_w,
                label="EW R_xi: W gauge fixing",
            ),
            _real_vector_gauge_fixing_term(
                vector=z_field,
                xi=gauge_fixing.xi_z,
                label="EW R_xi: Z gauge fixing",
            ),
            _real_vector_gauge_fixing_term(
                vector=a_field,
                xi=gauge_fixing.xi_a,
                label="EW R_xi: A gauge fixing",
            ),
            _scalar_vector_mixing_interaction(
                coefficient=-mw,
                scalar=gp_field,
                vector=wp_field,
                vector_conjugated=True,
                label="EW R_xi: G+ / W- cancellation",
            ),
            _scalar_vector_mixing_interaction(
                coefficient=-mw,
                scalar=gp_field,
                vector=wp_field,
                scalar_conjugated=True,
                label="EW R_xi: G- / W+ cancellation",
            ),
            _scalar_vector_mixing_interaction(
                coefficient=-mz,
                scalar=g0_field,
                vector=z_field,
                label="EW R_xi: G0 / Z cancellation",
            ),
            -(gauge_fixing.xi_w * mw_sq) * gp_field.bar * gp_field,
            -(gauge_fixing.xi_z * mz_sq / 2) * g0_field * g0_field,
        ))
        lagrangian_terms.extend(
            _build_physical_ghost_terms(
                g1=g1,
                g2=g2,
                vev=vev,
                gauge_fixing=gauge_fixing,
                higgs=h_field,
                goldstone_neutral=g0_field,
                goldstone_charged=gp_field,
                charged_w=wp_field,
                z_boson=z_field,
                photon=a_field,
                ghost_charged=gh_wp_field,
                ghost_z=gh_z_field,
                ghost_photon=gh_a_field,
            )
        )

    fermion_masses: list[tuple[Field, object]] = []
    for assignment in yukawas:
        if assignment.fermion.kind != "fermion":
            raise ValueError(
                f"Diagonal Yukawa assignment requires a fermion field, got {assignment.fermion.kind!r}."
            )
        mass = assignment.mass(vev)
        fermion_masses.append((assignment.fermion, mass))
        lagrangian_terms.append(
            _fermion_bilinear_term(
                coefficient=-mass,
                fermion=assignment.fermion,
                label=assignment.label or f"EW SSB: {assignment.fermion.name} mass term",
            )
        )
        lagrangian_terms.append(
            _fermion_bilinear_term(
                coefficient=-(assignment.yukawa * _INV_SQRT2),
                fermion=assignment.fermion,
                scalar=h_field,
                label=assignment.label or f"EW SSB: h {assignment.fermion.name}bar {assignment.fermion.name}",
            )
        )

    for assignment in matrix_yukawas:
        _validate_flavored_fermion(assignment.fermion, purpose="Matrix Yukawa assignment")
        mass_matrix = _flavor_matrix_symbol(assignment.matrix) * vev * _INV_SQRT2
        fermion_masses.append((assignment.fermion, mass_matrix))
        lagrangian_terms.append(
            _fermion_matrix_bilinear_term(
                coefficient=-(vev * _INV_SQRT2),
                matrix=assignment.matrix,
                fermion=assignment.fermion,
                label=assignment.label or f"EW SSB: matrix mass term for {assignment.fermion.name}",
            )
        )
        lagrangian_terms.append(
            _fermion_matrix_bilinear_term(
                coefficient=-_INV_SQRT2,
                matrix=assignment.matrix,
                fermion=assignment.fermion,
                scalar=h_field,
                label=assignment.label or f"EW SSB: h matrix Yukawa for {assignment.fermion.name}",
            )
        )

    for assignment in charged_current_mixings:
        lagrangian_terms.append(
            _charged_current_interaction(
                coefficient=-(g2 * _INV_SQRT2),
                matrix=assignment.matrix,
                left_fermion=assignment.up_field,
                right_fermion=assignment.down_field,
                vector=wp_field,
                label=assignment.label or f"EW SSB: W+ current {assignment.up_field.name} -> {assignment.down_field.name}",
            )
        )
        lagrangian_terms.append(
            _charged_current_interaction(
                coefficient=-(g2 * _INV_SQRT2),
                matrix=assignment.matrix,
                left_fermion=assignment.down_field,
                right_fermion=assignment.up_field,
                vector=wp_field,
                vector_conjugated=True,
                dagger_matrix=True,
                label=assignment.label or f"EW SSB: W- current {assignment.down_field.name} -> {assignment.up_field.name}",
            )
        )

    physical_fields = _unique_fields(
        h_field,
        g0_field,
        gp_field,
        wp_field,
        z_field,
        a_field,
        gh_wp_field,
        gh_z_field,
        gh_a_field,
        *(assignment.fermion for assignment in yukawas),
        *(assignment.fermion for assignment in matrix_yukawas),
        *(assignment.up_field for assignment in charged_current_mixings),
        *(assignment.down_field for assignment in charged_current_mixings),
    )
    fermion_fields = tuple(field for field in physical_fields if field.kind == "fermion")

    model = Model(
        name=name,
        fields=physical_fields,
        lagrangian_decl=DeclaredLagrangian(source_terms=tuple(lagrangian_terms)),
    )

    return BrokenElectroweakSector(
        model=model,
        fields=BrokenElectroweakFields(
            higgs_doublet=source_higgs,
            weak_gauge=source_weak,
            hypercharge_gauge=source_hypercharge,
            higgs=h_field,
            goldstone_neutral=g0_field,
            goldstone_charged=gp_field,
            charged_w=wp_field,
            z_boson=z_field,
            photon=a_field,
            ghost_charged=gh_wp_field,
            ghost_z=gh_z_field,
            ghost_photon=gh_a_field,
            fermions=fermion_fields,
        ),
        higgs_expansion=electroweak_higgs_vev_expansion(
            vev=vev,
            higgs=h_field,
            goldstone_neutral=g0_field,
            goldstone_charged=gp_field,
        ),
        charged_mixing=electroweak_charged_gauge_mixing(),
        neutral_mixing=electroweak_neutral_gauge_mixing(g1=g1, g2=g2),
        masses=ElectroweakMassSpectrum(
            mw=mw,
            mz=mz,
            photon=Expression.num(0),
            mh=None if higgs_potential_data is None else higgs_potential_data.mh,
            fermions=tuple(fermion_masses),
        ),
        gauge_couplings=gauge_couplings,
        higgs_potential=higgs_potential_data,
        gauge_fixing=gauge_fixing,
        yukawas=tuple(yukawas),
        matrix_yukawas=tuple(matrix_yukawas),
        charged_current_mixings=tuple(charged_current_mixings),
    )
