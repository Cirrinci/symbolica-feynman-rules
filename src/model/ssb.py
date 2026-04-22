"""Focused spontaneous-symmetry-breaking helpers for the electroweak sector.

This module deliberately stays small and explicit.  It does not try to turn
the full framework into a generic symmetry-breaking engine; instead it adds the
minimum electroweak SSB layer needed to:

- declare the standard Higgs doublet with ``Y = 1/2``
- expose the textbook Higgs/Goldstone expansion around the vev
- expose the charged and neutral electroweak mixing relations
- build a broken-phase ``Model`` with explicit local interaction terms for the
  Higgs-sector mass terms, Goldstone-vector mixings, and diagonal Yukawas

The resulting broken model reuses the existing ``Model`` / ``InteractionTerm`` /
``Lagrangian`` pipeline directly, so all vertex extraction keeps working
without touching the core compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from symbolica import Expression, S

from symbolic.spenso_structures import lorentz_metric

from .core import Model
from .declared import PartialD
from .interactions import DerivativeAction, InteractionTerm
from .lagrangian import DeclaredLagrangian
from .metadata import Field, LORENTZ_INDEX, SPINOR_INDEX, WEAK_ADJ_INDEX, WEAK_FUND_INDEX

_HALF = Expression.num(1) / Expression.num(2)
_INV_SQRT2 = _HALF ** _HALF


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


@dataclass(frozen=True)
class DiagonalYukawaAssignment:
    """Minimal diagonal Yukawa input for one Dirac fermion."""

    fermion: Field
    yukawa: object
    label: str = ""

    def mass(self, vev):
        return self.yukawa * vev * _INV_SQRT2


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


def _validate_source_higgs_doublet(field: Field):
    if field.kind != "scalar" or field.self_conjugate:
        raise ValueError("Electroweak SSB expects a non-self-conjugate scalar Higgs doublet.")
    if not field.index_positions(index=WEAK_FUND_INDEX):
        raise ValueError("Electroweak SSB expects the Higgs field to carry a weak-fundamental index.")
    if "Y" not in field.quantum_numbers:
        raise ValueError("Electroweak SSB expects the Higgs field to declare hypercharge 'Y'.")


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
    fields = [
        fermion.occurrence(conjugated=True, labels={SPINOR_INDEX.kind: spinor}),
        fermion.occurrence(labels={SPINOR_INDEX.kind: spinor}),
    ]
    if scalar is not None:
        fields.append(scalar.occurrence())
    return InteractionTerm(
        coupling=coefficient,
        fields=tuple(fields),
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
    name: str = "EW-broken-phase",
) -> BrokenElectroweakSector:
    """Build a compact broken electroweak sector as an explicit local model.

    Scope of the returned model:
    - canonical kinetic terms for ``h``, ``G0``, and ``G+``
    - Higgs-induced ``W`` and ``Z`` mass terms
    - explicit ``hWW`` and ``hZZ`` trilinears
    - Goldstone-vector bilinear mixings from the broken Higgs kinetic term
    - diagonal fermion mass terms plus ``h fbar f`` couplings
    - optional physical-basis gauge kinetic / self-interaction terms
    - optional Higgs-potential mass and self-coupling terms
    - optional broken-phase ``R_xi`` gauge-fixing bilinears plus ghost bilinears

    The pure gauge kinetic / self-interaction sector is intentionally left in
    the unbroken compiler unless ``include_gauge_sector=True`` is requested.
    Broken-phase ghost interactions are currently implemented at the bilinear
    level plus the leading ``h cbar c`` couplings; full charged ghost-gauge and
    ghost-Goldstone interaction completion still belongs to a later step.
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
        lagrangian_terms.extend(_ghost_bilinear_term(
            ghost=gh_wp_field,
            mass_sq=gauge_fixing.xi_w * mw_sq,
            label="EW ghost: charged ghost",
        ))
        lagrangian_terms.extend(_ghost_bilinear_term(
            ghost=gh_z_field,
            mass_sq=gauge_fixing.xi_z * mz_sq,
            label="EW ghost: Z ghost",
        ))
        lagrangian_terms.extend(_ghost_bilinear_term(
            ghost=gh_a_field,
            mass_sq=Expression.num(0),
            label="EW ghost: photon ghost",
        ))
        lagrangian_terms.append(
            _ghost_mass_term(
                ghost=gh_wp_field,
                scalar=h_field,
                coefficient=-(gauge_fixing.xi_w * mw_sq / vev),
                label="EW ghost: h cWbar cW",
            )
        )
        lagrangian_terms.append(
            _ghost_mass_term(
                ghost=gh_z_field,
                scalar=h_field,
                coefficient=-(gauge_fixing.xi_z * mz_sq / vev),
                label="EW ghost: h cZbar cZ",
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

    physical_fields = (
        h_field,
        g0_field,
        gp_field,
        wp_field,
        z_field,
        a_field,
        *(ghost for ghost in (gh_wp_field, gh_z_field, gh_a_field) if ghost is not None),
        *(assignment.fermion for assignment in yukawas),
    )

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
            fermions=tuple(assignment.fermion for assignment in yukawas),
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
    )
