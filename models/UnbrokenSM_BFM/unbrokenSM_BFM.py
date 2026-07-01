"""FeynPy implementation of ``papers/UnbrokenSM_BFM.fr.txt``.

The gauge, fermion, Higgs and Yukawa sectors are declared in the unsplit
gauge basis.  One simultaneous field transformation implements the
FeynRules ``gotoBFM`` replacement.  Background-covariant gauge-fixing and
ghost terms are added in the split basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from symbolica import Expression, S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    LORENTZ_INDEX,
    SPINOR_INDEX,
    CompiledLagrangian,
    DC,
    DeclaredLagrangian,
    Field,
    FS,
    FieldTransformation,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    Model,
    Parameter,
    PartialD,
    ProjM,
    ProjP,
    flavor_index,
)
from symbolic.spenso_structures import (
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
class UnbrokenSMBFMFields:
    B: Field
    BQuantum: Field
    Wi: Field
    WiQuantum: Field
    G: Field
    GQuantum: Field
    ghB: Field
    ghWi: Field
    ghG: Field
    lL: Field
    eR: Field
    qL: Field
    uR: Field
    dR: Field
    Phi: Field
    BTotal: Field
    WiTotal: Field
    GTotal: Field
    LL: Field
    LR: Field
    QL: Field
    UR: Field
    DR: Field


@dataclass(frozen=True)
class UnbrokenSMBFMParameters:
    g1: Parameter
    g2: Parameter
    g3: Parameter
    lam: Parameter
    muH: Parameter
    yl: Parameter
    yu: Parameter
    yd: Parameter


@dataclass(frozen=True)
class UnbrokenSMBFM:
    model: Model
    source_model: Model
    lagrangian: CompiledLagrangian
    fields: UnbrokenSMBFMFields
    parameters: UnbrokenSMBFMParameters
    gauge_groups: tuple[GaugeGroup, ...]
    lagrangians: dict[str, DeclaredLagrangian]
    transformations: tuple[FieldTransformation, ...]


def _compile_piece(
    declaration,
    *,
    name: str,
    groups: tuple[GaugeGroup, ...],
    fields: tuple[Field, ...],
    parameters: tuple[Parameter, ...],
    transformations: tuple[FieldTransformation, ...] = (),
) -> CompiledLagrangian:
    declared = DeclaredLagrangian.from_item(declaration)
    if not declared.source_terms:
        return CompiledLagrangian(parameters=parameters)
    model = Model(
        name=name,
        gauge_groups=groups,
        fields=fields,
        parameters=parameters,
        lagrangian_decl=declared,
    )
    compiled = model.lagrangian()
    if transformations:
        compiled = compiled.transform_fields(
            *transformations,
            repeat=False,
            real_symbols=tuple(p.symbol for p in parameters if p.is_real),
        )
    return compiled.simplify_parameter_identities()


def _bfm_gauge_fixing(background, quantum, coupling, f_builder, *, tag: str):
    """Return ``-1/2 (D_background . quantum)^2`` in explicit local form."""

    mu, nu = S(f"mu_gf_{tag}"), S(f"nu_gf_{tag}")
    a, b, c = S(f"a_gf_{tag}"), S(f"b_gf_{tag}"), S(f"c_gf_{tag}")
    d, e = S(f"d_gf_{tag}"), S(f"e_gf_{tag}")
    divergence = PartialD(quantum(mu, a), mu)
    return (
        -HALF * divergence * PartialD(quantum(nu, a), nu)
        - coupling
        * f_builder(a, b, c)
        * divergence
        * background(nu, b)
        * quantum(nu, c)
        - HALF
        * coupling**2
        * f_builder(a, b, c)
        * f_builder(a, d, e)
        * background(mu, b)
        * quantum(mu, c)
        * background(nu, d)
        * quantum(nu, e)
    )


def _bfm_outer_ghost_terms(
    ghost,
    background,
    quantum,
    coupling,
    f_builder,
    *,
    tag: str,
):
    """Outer background-covariant part of ``-cbar D_bg D_total c``."""

    mu = S(f"mu_gh_{tag}")
    a, b, c = S(f"a_gh_{tag}"), S(f"b_gh_{tag}"), S(f"c_gh_{tag}")
    d, e = S(f"d_gh_{tag}"), S(f"e_gh_{tag}")
    cubic = (
        coupling
        * f_builder(b, a, c)
        * ghost.bar(a)
        * background(mu, b)
        * PartialD(ghost(c), mu)
    )
    quartic_background = (
        -coupling**2
        * f_builder(b, a, c)
        * f_builder(d, c, e)
        * ghost.bar(a)
        * background(mu, b)
        * background(mu, d)
        * ghost(e)
    )
    quartic_quantum = (
        -coupling**2
        * f_builder(b, a, c)
        * f_builder(d, c, e)
        * ghost.bar(a)
        * background(mu, b)
        * quantum(mu, d)
        * ghost(e)
    )
    return cubic + quartic_background + quartic_quantum


def build_unbroken_sm_bfm(*, name: str = "UnbrokenSM_BFM") -> UnbrokenSMBFM:
    """Build the complete unbroken SM background-field-gauge Lagrangian."""

    generation = flavor_index("Generation", 3, prefix="f")
    parameters = UnbrokenSMBFMParameters(
        g1=Parameter("g1"),
        g2=Parameter("g2"),
        g3=Parameter("g3"),
        lam=Parameter("lam"),
        muH=Parameter("muH"),
        yl=Parameter(
            "yl", indices=(generation, generation), complex_param=True
        ),
        yu=Parameter(
            "yu", indices=(generation, generation), complex_param=True
        ),
        yd=Parameter(
            "yd", indices=(generation, generation), complex_param=True
        ),
    )

    B = Field("B", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,))
    Bq = Field(
        "BQuantum", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,)
    )
    W = Field(
        "Wi",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )
    Wq = Field(
        "WiQuantum",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )
    G = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    Gq = Field(
        "GQuantum",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )
    Bt = Field(
        "BTotal", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,)
    )
    Wt = Field(
        "WiTotal",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )
    Gt = Field(
        "GTotal",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
    )

    ghB = Field(
        "ghB",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of=B,
        conjugate_symbol=S("ghBbar"),
        quantum_numbers={"GhostNumber": ONE},
    )
    ghW = Field(
        "ghWi",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(WEAK_ADJ_INDEX,),
        ghost_of=W,
        conjugate_symbol=S("ghWibar"),
        quantum_numbers={"GhostNumber": ONE},
    )
    ghG = Field(
        "ghG",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
        ghost_of=G,
        conjugate_symbol=S("ghGbar"),
        quantum_numbers={"GhostNumber": ONE},
    )
    ghWt = Field(
        "ghWiTotal",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(WEAK_ADJ_INDEX,),
        ghost_of=Wt,
        conjugate_symbol=S("ghWiTotalbar"),
        quantum_numbers={"GhostNumber": ONE},
    )
    ghGt = Field(
        "ghGTotal",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
        ghost_of=Gt,
        conjugate_symbol=S("ghGTotalbar"),
        quantum_numbers={"GhostNumber": ONE},
    )

    fermion = Fraction(1, 2)
    lL = Field(
        "lL",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation),
        quantum_numbers={"Y": -HALF},
    )
    eR = Field(
        "eR",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation),
        quantum_numbers={"Y": -ONE},
    )
    qL = Field(
        "qL",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": ONE / SIX},
    )
    uR = Field(
        "uR",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": TWO / THREE},
    )
    dR = Field(
        "dR",
        spin=fermion,
        self_conjugate=False,
        indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
        quantum_numbers={"Y": -(ONE / THREE)},
    )
    Phi = Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        conjugate_symbol=S("Phibar"),
        indices=(WEAK_FUND_INDEX,),
        mass=parameters.muH.symbol,
        quantum_numbers={"Y": HALF},
    )

    # FeynRules' unphysical fields, used to apply chirality definitions.
    LL = Field(
        "LL",
        spin=fermion,
        self_conjugate=False,
        indices=lL.indices,
        quantum_numbers={"Y": -HALF},
    )
    LR = Field(
        "LR",
        spin=fermion,
        self_conjugate=False,
        indices=eR.indices,
        quantum_numbers={"Y": -ONE},
    )
    QL = Field(
        "QL",
        spin=fermion,
        self_conjugate=False,
        indices=qL.indices,
        quantum_numbers={"Y": ONE / SIX},
    )
    UR = Field(
        "UR",
        spin=fermion,
        self_conjugate=False,
        indices=uR.indices,
        quantum_numbers={"Y": TWO / THREE},
    )
    DR = Field(
        "DR",
        spin=fermion,
        self_conjugate=False,
        indices=dR.indices,
        quantum_numbers={"Y": -(ONE / THREE)},
    )

    fields = UnbrokenSMBFMFields(
        B=B,
        BQuantum=Bq,
        Wi=W,
        WiQuantum=Wq,
        G=G,
        GQuantum=Gq,
        ghB=ghB,
        ghWi=ghW,
        ghG=ghG,
        lL=lL,
        eR=eR,
        qL=qL,
        uR=uR,
        dR=dR,
        Phi=Phi,
        BTotal=Bt,
        WiTotal=Wt,
        GTotal=Gt,
        LL=LL,
        LR=LR,
        QL=QL,
        UR=UR,
        DR=DR,
    )

    groups = (
        GaugeGroup(
            "U1Y",
            abelian=True,
            coupling=parameters.g1,
            gauge_boson=Bt,
            charge="Y",
        ),
        GaugeGroup(
            "SU2L",
            abelian=False,
            coupling=parameters.g2,
            gauge_boson=Wt,
            structure_constant=weak_structure_constant,
            representations=(
                GaugeRepresentation(
                    WEAK_FUND_INDEX,
                    weak_gauge_generator,
                    name="doublet",
                ),
            ),
        ),
        GaugeGroup(
            "SU3C",
            abelian=False,
            coupling=parameters.g3,
            gauge_boson=Gt,
            structure_constant=structure_constant,
            representations=(
                GaugeRepresentation(
                    COLOR_FUND_INDEX,
                    gauge_generator,
                    name="fundamental",
                ),
            ),
        ),
    )
    u1, su2, su3 = groups

    mu, nu = S("mu"), S("nu")
    aw, ac = S("aw"), S("ac")
    sp, wi, wj, color = S("sp"), S("wi"), S("wj"), S("color")
    f1, f2 = S("f1"), S("f2")

    LGauge = (
        -ONE / FOUR * FS(u1, mu, nu) * FS(u1, mu, nu)
        - ONE
        / FOUR
        * FS(su2, mu, nu, aw)
        * FS(su2, mu, nu, aw)
        - ONE
        / FOUR
        * FS(su3, mu, nu, ac)
        * FS(su3, mu, nu, ac)
    )
    LFermions = (
        I * QL.bar * Gamma(mu) * DC(QL, mu)
        + I * LL.bar * Gamma(mu) * DC(LL, mu)
        + I * UR.bar * Gamma(mu) * DC(UR, mu)
        + I * DR.bar * Gamma(mu) * DC(DR, mu)
        + I * LR.bar * Gamma(mu) * DC(LR, mu)
    )
    LHiggs = (
        DC(Phi.bar, mu) * DC(Phi, mu)
        - parameters.muH**2 * Phi.bar * Phi
        - parameters.lam * Phi.bar * Phi * Phi.bar * Phi
    )
    yukawa = (
        -parameters.yd(f1, f2)
        * QL.bar(sp, wi, f1, color)
        * DR(sp, f2, color)
        * Phi(wi)
        - parameters.yl(f1, f2)
        * LL.bar(sp, wi, f1)
        * LR(sp, f2)
        * Phi(wi)
        - parameters.yu(f1, f2)
        * QL.bar(sp, wi, f1, color)
        * UR(sp, f2, color)
        * Phi.bar(wj)
        * weak_eps2(wi, wj)
    )
    LYukawa = (
        yukawa
        - parameters.yd(f1, f2).conj()
        * Phi.bar(wi)
        * DR.bar(sp, f2, color)
        * QL(sp, wi, f1, color)
        - parameters.yl(f1, f2).conj()
        * Phi.bar(wi)
        * LR.bar(sp, f2)
        * LL(sp, wi, f1)
        - parameters.yu(f1, f2).conj()
        * weak_eps2(wi, wj)
        * Phi(wj)
        * UR.bar(sp, f2, color)
        * QL(sp, wi, f1, color)
    )

    transformations = (
        FieldTransformation(Bt, B + Bq),
        FieldTransformation(Wt, W + Wq),
        FieldTransformation(Gt, G + Gq),
        FieldTransformation(LL, ProjM * lL),
        FieldTransformation(LR, ProjP * eR),
        FieldTransformation(QL, ProjM * qL),
        FieldTransformation(UR, ProjP * uR),
        FieldTransformation(DR, ProjP * dR),
    )
    all_parameters = tuple(parameters.__dict__.values())
    source_fields = (Bt, Wt, Gt, LL, LR, QL, UR, DR, Phi)
    core_decl = DeclaredLagrangian.from_item(
        LGauge + LFermions + LHiggs + LYukawa
    )
    source_model = Model(
        name=f"{name} unsplit source",
        gauge_groups=groups,
        fields=source_fields,
        parameters=all_parameters,
        lagrangian_decl=core_decl,
    )
    core = source_model.lagrangian().transform_fields(
        *transformations,
        repeat=False,
        real_symbols=tuple(p.symbol for p in all_parameters if p.is_real),
    ).simplify_parameter_identities()

    # Background-covariant gauge fixing.
    LGaugeFixing = DeclaredLagrangian.from_item(
        -HALF * PartialD(Bq(mu), mu) * PartialD(Bq(nu), nu)
        + _bfm_gauge_fixing(
            W,
            Wq,
            parameters.g2,
            weak_structure_constant,
            tag="w",
        )
        + _bfm_gauge_fixing(
            G,
            Gq,
            parameters.g3,
            structure_constant,
            tag="g",
        )
    )
    gauge_fixing = _compile_piece(
        LGaugeFixing,
        name=f"{name} gauge fixing",
        groups=(),
        fields=(Bq, W, Wq, G, Gq),
        parameters=all_parameters,
    )

    # First derivative in the FeynRules ghost expression: the inner
    # derivative contains background + quantum gauge fields.
    LGhostDirect = DeclaredLagrangian.from_item(
        -ghB.bar * PartialD(PartialD(ghB, mu), mu)
        - ghWt.bar * PartialD(DC(ghWt, mu), mu)
        - ghGt.bar * PartialD(DC(ghGt, mu), mu)
    )
    ghost_transformations = (
        FieldTransformation(Bt, B + Bq),
        FieldTransformation(Wt, W + Wq),
        FieldTransformation(Gt, G + Gq),
        FieldTransformation(ghWt, ghW),
        FieldTransformation(ghGt, ghG),
    )
    ghost_direct = _compile_piece(
        LGhostDirect,
        name=f"{name} direct ghost",
        groups=groups,
        fields=(Bt, Wt, Gt, ghB, ghWt, ghGt),
        parameters=all_parameters,
        transformations=ghost_transformations,
    )
    LGhostOuter = DeclaredLagrangian.from_item(
        _bfm_outer_ghost_terms(
            ghW,
            W,
            Wq,
            parameters.g2,
            weak_structure_constant,
            tag="w",
        )
        + _bfm_outer_ghost_terms(
            ghG,
            G,
            Gq,
            parameters.g3,
            structure_constant,
            tag="g",
        )
    )
    ghost_outer = _compile_piece(
        LGhostOuter,
        name=f"{name} outer ghost",
        groups=(),
        fields=(W, Wq, G, Gq, ghW, ghG),
        parameters=all_parameters,
    )
    LGhost = LGhostDirect + LGhostOuter
    split_fields = (
        B,
        Bq,
        W,
        Wq,
        G,
        Gq,
        ghB,
        ghW,
        ghG,
        lL,
        eR,
        qL,
        uR,
        dR,
        Phi,
    )
    complete = CompiledLagrangian(
        terms=core.terms + gauge_fixing.terms + ghost_direct.terms + ghost_outer.terms,
        parameters=all_parameters,
    )
    model = Model(name=name, fields=split_fields, parameters=all_parameters)
    model._compiled_lagrangian = complete

    return UnbrokenSMBFM(
        model=model,
        source_model=source_model,
        lagrangian=complete,
        fields=fields,
        parameters=parameters,
        gauge_groups=groups,
        lagrangians={
            "LGauge": DeclaredLagrangian.from_item(LGauge),
            "LFermions": DeclaredLagrangian.from_item(LFermions),
            "LHiggs": DeclaredLagrangian.from_item(LHiggs),
            "LYukawa": DeclaredLagrangian.from_item(LYukawa),
            "LGaugeFixing": LGaugeFixing,
            "LGhost": LGhost,
            "LSM": core_decl + LGaugeFixing + LGhost,
        },
        transformations=transformations,
    )


__all__ = ("UnbrokenSMBFM", "build_unbroken_sm_bfm")
