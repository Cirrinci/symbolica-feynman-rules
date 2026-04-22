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
    fermions: tuple[Field, ...] = ()


@dataclass(frozen=True)
class BrokenElectroweakSector:
    """Inspectible broken-phase electroweak sector plus its compiled model."""

    model: Model
    fields: BrokenElectroweakFields
    higgs_expansion: tuple[LinearRelation, ...]
    charged_mixing: tuple[LinearRelation, ...]
    neutral_mixing: tuple[LinearRelation, ...]
    masses: ElectroweakMassSpectrum
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


def build_broken_electroweak_sector(
    *,
    g1,
    g2,
    vev,
    higgs_doublet: Field | None = None,
    weak_gauge: Field | None = None,
    hypercharge_gauge: Field | None = None,
    higgs: Field | None = None,
    goldstone_neutral: Field | None = None,
    goldstone_charged: Field | None = None,
    charged_w: Field | None = None,
    z_boson: Field | None = None,
    photon: Field | None = None,
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

    The pure gauge kinetic / self-interaction sector is intentionally left in
    the unbroken compiler for now; this helper focuses only on the broken
    Higgs/Yukawa layer requested in the current step.
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

    mw = electroweak_mw(g2, vev)
    mz = electroweak_mz(g1, g2, vev)
    mw_sq = mw**2
    mz_sq = mz**2

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
            fermions=tuple(fermion_masses),
        ),
        yukawas=tuple(yukawas),
    )
