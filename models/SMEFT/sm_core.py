"""Unbroken Standard Model foundation for the SMEFT Green basis.

This module builds the renormalizable Standard Model of Appendix D, Eq. (D.1)
of arXiv:2112.10787 in the *unbroken* phase and exposes the shared field,
parameter and gauge-group declarations that every dimension-six operator of the
Green basis is written in.

Conventions follow Appendix D exactly:

* covariant derivative (Eq. D.3)
  ``D_mu = d_mu - i g3 T^A G^A_mu - i g2 (sigma^I/2) W^I_mu - i g1 Y B_mu`` ;
* field strengths (Eqs. D.4-D.6) with ``T^A = lambda^A/2`` and ``sigma^I`` the
  Pauli matrices;
* hypercharges ``Y_q = 1/6``, ``Y_l = -1/2``, ``Y_u = 2/3``, ``Y_d = -1/3``,
  ``Y_e = -1``, ``Y_H = 1/2`` ;
* the conjugate doublet ``Htilde = i sigma2 H^*``.

The fermion fields ``q, l, u, d, e`` are declared as ordinary four-component
Dirac fields.  Chirality is **not** attached to the field object; every operator
inserts the appropriate chiral projector explicitly (see :mod:`.tensors`).  This
keeps ordered gamma-matrix chains intact (no field-transformation
post-processing runs on the operators), which is essential for the evanescent
operators of Tables 4-9.

The renormalizable Lagrangian is provided separately, built through the proven
carrier-field + ``ProjM``/``ProjP`` transformation idiom used by the bundled
``UnbrokenSM_BFM`` model, purely for validation and completeness; the
dimension-six operators do not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from symbolica import Expression, S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    CompiledLagrangian,
    DC,
    DeclaredLagrangian,
    Field,
    FieldTransformation,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    IndexType,
    LORENTZ_INDEX,
    Model,
    Parameter,
    ProjM,
    ProjP,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    flavor_index,
)
from symbolic.spenso_structures import (
    COLOR_ADJ_KIND,
    COLOR_FUND_KIND,
    LORENTZ_KIND,
    SPINOR_KIND,
    WEAK_ADJ_KIND,
    WEAK_FUND_KIND,
    gauge_generator,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I


ONE = Expression.num(1)
TWO = Expression.num(2)
THREE = Expression.num(3)
FOUR = Expression.num(4)
SIX = Expression.num(6)
HALF = ONE / TWO


@dataclass(frozen=True)
class SMEFTIndices:
    generation: IndexType
    spinor: IndexType
    lorentz: IndexType
    weak_fund: IndexType
    weak_adj: IndexType
    colour_fund: IndexType
    colour_adj: IndexType


@dataclass(frozen=True)
class SMEFTParameters:
    g1: Parameter
    g2: Parameter
    g3: Parameter
    lam: Parameter
    mH2: Parameter
    Yu: Parameter
    Yd: Parameter
    Ye: Parameter


@dataclass(frozen=True)
class SMEFTFields:
    # Physical unbroken-phase fields the operators are written in.
    q: Field
    l: Field
    u: Field
    d: Field
    e: Field
    H: Field
    B: Field
    W: Field
    G: Field
    # Full-Dirac carriers used only to build the renormalizable Lagrangian
    # through the ProjM/ProjP transformation.
    Q: Field
    L: Field
    U: Field
    D: Field
    E: Field


@dataclass(frozen=True)
class SMEFTGaugeGroups:
    U1Y: GaugeGroup
    SU2L: GaugeGroup
    SU3C: GaugeGroup


@dataclass(frozen=True)
class SMEFTCore:
    """Shared declarations and helpers for the SMEFT Green basis."""

    name: str
    indices: SMEFTIndices
    parameters: SMEFTParameters
    fields: SMEFTFields
    gauge_groups: SMEFTGaugeGroups
    renormalizable: CompiledLagrangian

    # -- convenience views -------------------------------------------------
    @property
    def all_parameters(self) -> tuple[Parameter, ...]:
        return tuple(self.parameters.__dict__.values())

    @property
    def operator_fields(self) -> tuple[Field, ...]:
        f = self.fields
        return (f.B, f.W, f.G, f.H, f.q, f.l, f.u, f.d, f.e)

    @property
    def group_tuple(self) -> tuple[GaugeGroup, ...]:
        return tuple(self.gauge_groups.__dict__.values())

    def source_model(
        self,
        decl: DeclaredLagrangian,
        *,
        extra_parameters: tuple[Parameter, ...] = (),
    ) -> Model:
        """Return a ``Model`` wrapping ``decl`` in the SMEFT field/gauge context."""

        return Model(
            name=f"{self.name} operator",
            gauge_groups=self.group_tuple,
            fields=self.operator_fields,
            parameters=self.all_parameters + tuple(extra_parameters),
            lagrangian_decl=decl,
        )

    def compile_operator(
        self,
        decl: DeclaredLagrangian,
        *,
        extra_parameters: tuple[Parameter, ...] = (),
    ) -> CompiledLagrangian:
        """Compile one declared dimension-six operator (no field transformation).

        The operators carry explicit chiral projectors, so no ``ProjM``/``ProjP``
        transformation is applied and ordered gamma chains are preserved.
        """

        return self.source_model(decl, extra_parameters=extra_parameters).lagrangian()


# ---------------------------------------------------------------------------
# Occurrence helper
# ---------------------------------------------------------------------------

_KIND_KEYS = {
    "sp": SPINOR_KIND,
    "mu": LORENTZ_KIND,
    "w": WEAK_FUND_KIND,
    "aw": WEAK_ADJ_KIND,
    "c": COLOR_FUND_KIND,
    "ac": COLOR_ADJ_KIND,
}


def occ(field: Field, *, conjugated: bool = False, **labels):
    """Return a labelled field occurrence in the field's declared index order.

    Keyword labels are keyed by short role names:
    ``sp`` (spinor), ``mu`` (Lorentz), ``w`` (weak doublet), ``aw`` (weak
    triplet/adjoint), ``c`` (colour fundamental), ``ac`` (colour adjoint) and
    ``f`` (generation).  Every declared index of ``field`` must receive a label.
    """

    generation_kind = None
    for index in field.indices:
        if index.is_flavor:
            generation_kind = index.kind
            break

    kind_to_label: dict[str, object] = {}
    for key, value in labels.items():
        if key == "f":
            if generation_kind is None:
                raise TypeError(f"Field {field.name!r} carries no generation index.")
            kind_to_label[generation_kind] = value
            continue
        if key not in _KIND_KEYS:
            raise TypeError(f"Unknown occurrence label {key!r} for field {field.name!r}.")
        kind_to_label[_KIND_KEYS[key]] = value

    positional = []
    for index in field.indices:
        if index.kind not in kind_to_label:
            raise TypeError(
                f"Field {field.name!r} index {index.name!r} (kind {index.kind!r}) "
                "was not given a label."
            )
        positional.append(kind_to_label[index.kind])

    base = field.bar if conjugated else field
    return base(*positional)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_sm_core(*, name: str = "SMEFT") -> SMEFTCore:
    """Build the shared SMEFT foundation (fields, parameters, gauge groups).

    Returns an :class:`SMEFTCore` exposing the unbroken-phase fields the Green
    basis is written in, together with the renormalizable Lagrangian (Eq. D.1)
    for validation.
    """

    generation = flavor_index("Generation", 3, prefix="f")
    indices = SMEFTIndices(
        generation=generation,
        spinor=SPINOR_INDEX,
        lorentz=LORENTZ_INDEX,
        weak_fund=WEAK_FUND_INDEX,
        weak_adj=WEAK_ADJ_INDEX,
        colour_fund=COLOR_FUND_INDEX,
        colour_adj=COLOR_ADJ_INDEX,
    )

    parameters = SMEFTParameters(
        g1=Parameter("g1"),
        g2=Parameter("g2"),
        g3=Parameter("g3"),
        lam=Parameter("lam"),
        mH2=Parameter("mH2"),
        Yu=Parameter("Yu", indices=(generation, generation), complex_param=True),
        Yd=Parameter("Yd", indices=(generation, generation), complex_param=True),
        Ye=Parameter("Ye", indices=(generation, generation), complex_param=True),
    )

    fermion = Fraction(1, 2)

    # Physical unbroken fields (operators are written in these).
    q = Field(
        "q",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": ONE / SIX},
    )
    lep = Field(
        "l",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation),
        quantum_numbers={"Y": -HALF},
    )
    up = Field(
        "u",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": TWO / THREE},
    )
    dn = Field(
        "d",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": -(ONE / THREE)},
    )
    er = Field(
        "e",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation),
        quantum_numbers={"Y": -ONE},
    )
    H = Field(
        "H",
        spin=0,
        self_conjugate=False,
        conjugate_symbol=S("Hbar"),
        indices=(WEAK_FUND_INDEX,),
        quantum_numbers={"Y": HALF},
    )
    B = Field("B", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,))
    W = Field(
        "W", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX)
    )
    G = Field(
        "G", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX)
    )

    # Full-Dirac carriers used to declare the renormalizable Lagrangian.
    Q = Field("Q", spin=fermion, self_conjugate=False, indices=q.indices,
              quantum_numbers={"Y": ONE / SIX})
    Lc = Field("L", spin=fermion, self_conjugate=False, indices=lep.indices,
               quantum_numbers={"Y": -HALF})
    Uc = Field("U", spin=fermion, self_conjugate=False, indices=up.indices,
               quantum_numbers={"Y": TWO / THREE})
    Dc = Field("D", spin=fermion, self_conjugate=False, indices=dn.indices,
               quantum_numbers={"Y": -(ONE / THREE)})
    Ec = Field("E", spin=fermion, self_conjugate=False, indices=er.indices,
               quantum_numbers={"Y": -ONE})

    fields = SMEFTFields(
        q=q, l=lep, u=up, d=dn, e=er, H=H, B=B, W=W, G=G,
        Q=Q, L=Lc, U=Uc, D=Dc, E=Ec,
    )

    gauge_groups = SMEFTGaugeGroups(
        U1Y=GaugeGroup(
            "U1Y",
            abelian=True,
            coupling=parameters.g1,
            gauge_boson=B,
            charge="Y",
        ),
        SU2L=GaugeGroup(
            "SU2L",
            abelian=False,
            coupling=parameters.g2,
            gauge_boson=W,
            structure_constant=weak_structure_constant,
            representations=(
                GaugeRepresentation(
                    index=WEAK_FUND_INDEX,
                    generator_builder=weak_gauge_generator,
                    name="doublet",
                ),
            ),
        ),
        SU3C=GaugeGroup(
            "SU3C",
            abelian=False,
            coupling=parameters.g3,
            gauge_boson=G,
            structure_constant=structure_constant,
            representations=(
                GaugeRepresentation(
                    index=COLOR_FUND_INDEX,
                    generator_builder=gauge_generator,
                    name="fundamental",
                ),
            ),
        ),
    )

    renormalizable = _build_renormalizable(
        name, indices, parameters, fields, gauge_groups
    )

    return SMEFTCore(
        name=name,
        indices=indices,
        parameters=parameters,
        fields=fields,
        gauge_groups=gauge_groups,
        renormalizable=renormalizable,
    )


def _build_renormalizable(
    name: str,
    indices: SMEFTIndices,
    parameters: SMEFTParameters,
    fields: SMEFTFields,
    gauge_groups: SMEFTGaugeGroups,
) -> CompiledLagrangian:
    """Compile Eq. (D.1) via carrier fields and a chiral transformation."""

    u1, su2, su3 = gauge_groups.U1Y, gauge_groups.SU2L, gauge_groups.SU3C
    Q, Lc, Uc, Dc, Ec = fields.Q, fields.L, fields.U, fields.D, fields.E
    q, lep, up, dn, er = fields.q, fields.l, fields.u, fields.d, fields.e
    H = fields.H

    mu, nu = S("mu"), S("nu")
    aw, ac = S("aw"), S("ac")
    sp, wi, wj, color = S("sp"), S("wi"), S("wj"), S("color")
    f1, f2 = S("f1"), S("f2")

    from feynpy import FS

    LGauge = (
        -ONE / FOUR * FS(u1, mu, nu) * FS(u1, mu, nu)
        - ONE / FOUR * FS(su2, mu, nu, aw) * FS(su2, mu, nu, aw)
        - ONE / FOUR * FS(su3, mu, nu, ac) * FS(su3, mu, nu, ac)
    )
    LFermions = (
        I * Q.bar * Gamma(mu) * DC(Q, mu)
        + I * Lc.bar * Gamma(mu) * DC(Lc, mu)
        + I * Uc.bar * Gamma(mu) * DC(Uc, mu)
        + I * Dc.bar * Gamma(mu) * DC(Dc, mu)
        + I * Ec.bar * Gamma(mu) * DC(Ec, mu)
    )
    LHiggs = (
        DC(H.bar, mu) * DC(H, mu)
        - parameters.mH2 * H.bar * H
        - parameters.lam * H.bar * H * H.bar * H
    )
    # Yukawa couplings: -[ lbar Ye e H + qbar Yu u Htilde + qbar Yd d H ] + h.c.
    # with Htilde_r = eps_{rs} Hbar_s.
    LYukawa = (
        -parameters.Ye(f1, f2) * Lc.bar(sp, wi, f1) * Ec(sp, f2) * H(wi)
        - parameters.Yd(f1, f2) * Q.bar(sp, wi, f1, color) * Dc(sp, f2, color) * H(wi)
        - parameters.Yu(f1, f2)
        * Q.bar(sp, wi, f1, color)
        * Uc(sp, f2, color)
        * H.bar(wj)
        * weak_eps2(wi, wj)
        - parameters.Ye(f1, f2).conj()
        * H.bar(wi)
        * Ec.bar(sp, f2)
        * Lc(sp, wi, f1)
        - parameters.Yd(f1, f2).conj()
        * H.bar(wi)
        * Dc.bar(sp, f2, color)
        * Q(sp, wi, f1, color)
        - parameters.Yu(f1, f2).conj()
        * weak_eps2(wi, wj)
        * H(wj)
        * Uc.bar(sp, f2, color)
        * Q(sp, wi, f1, color)
    )

    decl = DeclaredLagrangian.from_item(LGauge + LFermions + LHiggs + LYukawa)
    all_parameters = tuple(parameters.__dict__.values())
    source_fields = (fields.B, fields.W, fields.G, H, Q, Lc, Uc, Dc, Ec)
    source = Model(
        name=f"{name} renormalizable source",
        gauge_groups=(u1, su2, su3),
        fields=source_fields,
        parameters=all_parameters,
        lagrangian_decl=decl,
    )
    transformations = (
        FieldTransformation(Q, ProjM * q),
        FieldTransformation(Lc, ProjM * lep),
        FieldTransformation(Uc, ProjP * up),
        FieldTransformation(Dc, ProjP * dn),
        FieldTransformation(Ec, ProjP * er),
    )
    real_symbols = tuple(p.symbol for p in all_parameters if p.is_real)
    return (
        source.lagrangian()
        .transform_fields(*transformations, repeat=False, real_symbols=real_symbols)
        .simplify_parameter_identities()
    )


__all__ = (
    "SMEFTCore",
    "SMEFTFields",
    "SMEFTGaugeGroups",
    "SMEFTIndices",
    "SMEFTParameters",
    "build_sm_core",
    "occ",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "SIX",
    "HALF",
)
