"""Broken-phase Standard Model generated from gauge-basis declarations.

The source Lagrangian follows the non-BFM sectors in FeynRules ``SM.fr``.
Covariant derivatives and field strengths are compiled in the gauge basis,
finite weak indices are expanded, and one simultaneous field-transformation
stage produces the physical basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from symbolica import Expression, S

from symbolic.spenso_structures import (
    gauge_generator,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    DC,
    DeclaredLagrangian,
    Field,
    FS,
    FieldTransformation,
    Gamma,
    GaugeFixing,
    GaugeGroup,
    GaugeRepresentation,
    GhostLagrangian,
    IndexType,
    LORENTZ_INDEX,
    Model,
    Parameter,
    PartialD,
    ProjM,
    ProjP,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    CompiledLagrangian,
    flavor_index,
    rotation,
)

from .SM_support import (
    FOUR as _FOUR,
    HALF as _HALF,
    INV_SQRT2 as _INV_SQRT2,
    ONE as _ONE,
    SIX as _SIX,
    THREE as _THREE,
    TWO as _TWO,
    ckm_components as _ckm_components,
    ckm_dagger_components as _ckm_dagger_components,
    compile_source_piece as _compile_source_piece,
    diagonal_components as _diagonal_components,
    electroweak_rxi_gauge_fixing_lagrangian,
    electroweak_scalar_ghost_lagrangian,
    parameter_value_or_symbol,
    standard_model_weak_tensor_components,
)


@dataclass(frozen=True)
class StandardModelIndices:
    generation: IndexType
    weak_fundamental: IndexType
    weak_adjoint: IndexType
    colour_fundamental: IndexType
    colour_adjoint: IndexType


@dataclass(frozen=True)
class StandardModelGaugeGroups:
    U1Y: GaugeGroup
    SU2L: GaugeGroup
    SU3C: GaugeGroup


@dataclass(frozen=True)
class StandardModelFields:
    # Gauge-basis fields used in source declarations.
    LL: Field
    lR: Field
    QL: Field
    uR: Field
    dR: Field
    Phi: Field
    B: Field
    Wi: Field
    ghB: Field
    ghWi: Field

    # Physical fields.
    vl: Field
    l: Field
    uq: Field
    dq: Field
    H: Field
    G0: Field
    GP: Field
    W: Field
    Z: Field
    A: Field
    G: Field
    ghA: Field
    ghZ: Field
    ghWp: Field
    ghWm: Field
    ghG: Field


@dataclass(frozen=True)
class StandardModelParameters:
    g1: Parameter
    g2: Parameter
    g3: Parameter
    lam: Parameter
    vev: Parameter
    Mvl: Parameter
    Ml: Parameter
    Mu: Parameter
    Md: Parameter
    MW: Parameter
    MZ: Parameter
    MH: Parameter
    sw: Parameter
    cw: Parameter
    ee: Parameter
    xiA: Parameter
    xiZ: Parameter
    xiW: Parameter
    xiG: Parameter
    Yu: Parameter
    YuDag: Parameter
    Yd: Parameter
    YdDag: Parameter
    Ye: Parameter
    YeDag: Parameter
    CKM: Parameter
    CKMDag: Parameter


@dataclass(frozen=True)
class StandardModelLagrangians:
    LGauge: DeclaredLagrangian
    LFermions: DeclaredLagrangian
    LHiggs: DeclaredLagrangian
    LYukawa: DeclaredLagrangian
    LGaugeFixing: DeclaredLagrangian
    LGhost: DeclaredLagrangian
    LSM: DeclaredLagrangian


@dataclass(frozen=True)
class StandardModel:
    model: Model
    source_model: Model
    lagrangian: CompiledLagrangian
    indices: StandardModelIndices
    gauge_groups: StandardModelGaugeGroups
    fields: StandardModelFields
    parameters: StandardModelParameters
    lagrangians: StandardModelLagrangians
    transformations: tuple[FieldTransformation, ...]


def build_standard_model(
    *,
    name: str = "Standard Model",
    include_ghosts: bool = True,
    include_gauge_fixing: bool = False,
    xiA=1,
    xiZ=1,
    xiW=1,
    xiG=1,
) -> StandardModel:
    """Build the broken-phase Standard Model from gauge-basis declarations."""

    # Indices
    generation = flavor_index("Generation", 3, prefix="fl")
    indices = StandardModelIndices(
        generation=generation,
        weak_fundamental=WEAK_FUND_INDEX,
        weak_adjoint=WEAK_ADJ_INDEX,
        colour_fundamental=COLOR_FUND_INDEX,
        colour_adjoint=COLOR_ADJ_INDEX,
    )

    # Parameters
    g1 = Parameter("g1")
    g2 = Parameter("gw")
    g3 = Parameter("gs")
    lam = Parameter("lam")
    vev = Parameter("vev")
    sw = Parameter("sw")
    cw = Parameter("cw")
    ee = Parameter("ee")
    MW = Parameter("MW", value=g2.symbol * vev.symbol / 2)
    MZ = Parameter(
        "MZ",
        value=((g1.symbol**2 + g2.symbol**2) ** _HALF) * vev.symbol / 2,
    )
    MH = Parameter("MH", value=(2 * lam.symbol * vev.symbol**2) ** _HALF)
    ckm_components = _ckm_components()
    parameters = StandardModelParameters(
        g1=g1,
        g2=g2,
        g3=g3,
        lam=lam,
        vev=vev,
        Mvl=Parameter(
            "Mvl",
            indices=(generation,),
            components={
                (1,): Expression.num(0),
                (2,): Expression.num(0),
                (3,): Expression.num(0),
            },
        ),
        Ml=Parameter(
            "Ml",
            indices=(generation,),
            components={
                (1,): _INV_SQRT2 * vev.symbol * S("ye1"),
                (2,): _INV_SQRT2 * vev.symbol * S("ye2"),
                (3,): _INV_SQRT2 * vev.symbol * S("ye3"),
            },
        ),
        Mu=Parameter(
            "Mu",
            indices=(generation,),
            components={
                (1,): _INV_SQRT2 * vev.symbol * S("yu1"),
                (2,): _INV_SQRT2 * vev.symbol * S("yu2"),
                (3,): _INV_SQRT2 * vev.symbol * S("yu3"),
            },
        ),
        Md=Parameter(
            "Md",
            indices=(generation,),
            components={
                (1,): _INV_SQRT2 * vev.symbol * S("yd1"),
                (2,): _INV_SQRT2 * vev.symbol * S("yd2"),
                (3,): _INV_SQRT2 * vev.symbol * S("yd3"),
            },
        ),
        MW=MW,
        MZ=MZ,
        MH=MH,
        sw=sw,
        cw=cw,
        ee=ee,
        xiA=Parameter("xiA", internal=False, value=xiA),
        xiZ=Parameter("xiZ", internal=False, value=xiZ),
        xiW=Parameter("xiW", internal=False, value=xiW),
        xiG=Parameter("xiG", internal=False, value=xiG),
        Yu=Parameter(
            "yu",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("yu"),
        ),
        YuDag=Parameter(
            "YuDag",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("yu"),
        ),
        Yd=Parameter(
            "yd",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("yd"),
        ),
        YdDag=Parameter(
            "YdDag",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("yd"),
        ),
        Ye=Parameter(
            "yl",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("ye"),
        ),
        YeDag=Parameter(
            "YeDag",
            indices=(generation, generation),
            complex_param=True,
            components=_diagonal_components("ye"),
        ),
        CKM=Parameter(
            "CKM",
            indices=(generation, generation),
            complex_param=True,
            components=ckm_components,
            unitary_partner="CKMDag",
        ),
        CKMDag=Parameter(
            "CKMDag",
            indices=(generation, generation),
            complex_param=True,
            components=_ckm_dagger_components(),
            unitary_partner="CKM",
        ),
    )

    # Gauge-basis fields and physical fields
    B = Field("B", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,))
    Wi = Field(
        "Wi",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
    )
    G = Field(
        "G",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
        mass=Expression.num(0),
    )
    ghB = Field(
        "ghB",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of=B,
        conjugate_symbol=S("ghBbar"),
    )
    ghWi = Field(
        "ghWi",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(WEAK_ADJ_INDEX,),
        ghost_of=Wi,
        conjugate_symbol=S("ghWibar"),
    )
    ghG = Field(
        "ghG",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=(COLOR_ADJ_INDEX,),
        ghost_of=G,
        conjugate_symbol=S("ghGbar"),
        mass=Expression.num(0),
        quantum_numbers={"GhostNumber": _ONE},
    )
    xiA_value = parameter_value_or_symbol(parameters.xiA)
    xiZ_value = parameter_value_or_symbol(parameters.xiZ)
    xiW_value = parameter_value_or_symbol(parameters.xiW)
    z_ghost_mass = (
        (xiZ_value * parameters.MZ.symbol**2) ** _HALF
        if include_gauge_fixing
        else parameters.MZ.symbol
    )
    w_ghost_mass = (
        (xiW_value * parameters.MW.symbol**2) ** _HALF
        if include_gauge_fixing
        else parameters.MW.symbol
    )

    W_field = Field(
        "W",
        spin=1,
        self_conjugate=False,
        conjugate_symbol=S("Wbar"),
        indices=(LORENTZ_INDEX,),
        mass=parameters.MW.symbol,
        quantum_numbers={"Q": _ONE},
    )
    Z_field = Field(
        "Z",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX,),
        mass=parameters.MZ.symbol,
        quantum_numbers={"Q": Expression.num(0)},
    )
    A_field = Field(
        "A",
        spin=1,
        self_conjugate=True,
        indices=(LORENTZ_INDEX,),
        mass=Expression.num(0),
        quantum_numbers={"Q": Expression.num(0)},
    )
    H_field = Field(
        "H",
        spin=0,
        self_conjugate=True,
        mass=parameters.MH.symbol,
        quantum_numbers={"Q": Expression.num(0)},
    )
    G0_field = Field(
        "G0",
        spin=0,
        self_conjugate=True,
        mass=z_ghost_mass,
        quantum_numbers={"Q": Expression.num(0)},
        goldstone_of=Z_field,
    )
    GP_field = Field(
        "GP",
        spin=0,
        self_conjugate=False,
        conjugate_symbol=S("GPbar"),
        mass=w_ghost_mass,
        quantum_numbers={"Q": _ONE},
        goldstone_of=W_field,
    )
    ghA_field = Field(
        "ghA",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of=A_field,
        conjugate_symbol=S("ghAbar"),
        mass=Expression.num(0),
        quantum_numbers={"GhostNumber": _ONE},
    )
    ghZ_field = Field(
        "ghZ",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of=Z_field,
        conjugate_symbol=S("ghZbar"),
        mass=z_ghost_mass,
        quantum_numbers={"GhostNumber": _ONE},
    )
    ghWp_field = Field(
        "ghWp",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of=W_field,
        conjugate_symbol=S("ghWpbar"),
        mass=w_ghost_mass,
        quantum_numbers={"GhostNumber": _ONE, "Q": _ONE},
    )
    ghWm_field = Field(
        "ghWm",
        spin=0,
        kind="ghost",
        self_conjugate=False,
        ghost_of="Wbar",
        conjugate_symbol=S("ghWmbar"),
        mass=w_ghost_mass,
        quantum_numbers={"GhostNumber": _ONE, "Q": -_ONE},
    )

    fields = StandardModelFields(
        LL=Field(
            "LL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation),
            quantum_numbers={"Y": -_HALF},
        ),
        lR=Field(
            "lR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            quantum_numbers={"Y": -_ONE},
        ),
        QL=Field(
            "QL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(
                SPINOR_INDEX,
                WEAK_FUND_INDEX,
                generation,
                COLOR_FUND_INDEX,
            ),
            quantum_numbers={"Y": _ONE / _SIX},
        ),
        uR=Field(
            "uR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": _TWO / _THREE},
        ),
        dR=Field(
            "dR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": -(_ONE / _THREE)},
        ),
        Phi=Field(
            "Phi",
            spin=0,
            self_conjugate=False,
            symbol=S("Phi"),
            conjugate_symbol=S("Phibar"),
            indices=(WEAK_FUND_INDEX,),
            quantum_numbers={"Y": _HALF},
        ),
        B=B,
        Wi=Wi,
        ghB=ghB,
        ghWi=ghWi,
        vl=Field(
            "vl",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            mass=parameters.Mvl,
            quantum_numbers={"Q": Expression.num(0), "LeptonNumber": _ONE},
            flavor_index=generation,
            class_members=("ve", "vm", "vt"),
        ),
        l=Field(
            "l",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            mass=parameters.Ml,
            quantum_numbers={"Q": -_ONE, "LeptonNumber": _ONE},
            flavor_index=generation,
            class_members=("e", "mu", "ta"),
        ),
        uq=Field(
            "uq",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            mass=parameters.Mu,
            quantum_numbers={"Q": _TWO / _THREE},
            flavor_index=generation,
            class_members=("u", "c", "t"),
        ),
        dq=Field(
            "dq",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            mass=parameters.Md,
            quantum_numbers={"Q": -(_ONE / _THREE)},
            flavor_index=generation,
            class_members=("d", "s", "b"),
        ),
        H=H_field,
        G0=G0_field,
        GP=GP_field,
        W=W_field,
        Z=Z_field,
        A=A_field,
        G=G,
        ghA=ghA_field,
        ghZ=ghZ_field,
        ghWp=ghWp_field,
        ghWm=ghWm_field,
        ghG=ghG,
    )

    # Gauge groups
    gauge_groups = StandardModelGaugeGroups(
        U1Y=GaugeGroup(
            "U1Y",
            abelian=True,
            coupling=parameters.g1,
            gauge_boson=fields.B,
            ghost_field=fields.ghB,
            charge="Y",
        ),
        SU2L=GaugeGroup(
            "SU2L",
            abelian=False,
            coupling=parameters.g2,
            gauge_boson=fields.Wi,
            ghost_field=fields.ghWi,
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
            gauge_boson=fields.G,
            ghost_field=fields.ghG,
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

    mu, nu = S("mu"), S("nu")
    weak_adj, colour_adj = S("aW"), S("aC")
    spinor, weak_left, weak_right = S("sp"), S("ii"), S("jj")
    colour = S("cc")
    f1, f2, f3 = S("ff1"), S("ff2"), S("ff3")

    # Source Lagrangian in the gauge basis
    LGauge = (
        -_ONE / _FOUR
        * FS(gauge_groups.U1Y, mu, nu)
        * FS(gauge_groups.U1Y, mu, nu)
        - _ONE / _FOUR
        * FS(gauge_groups.SU2L, mu, nu, weak_adj)
        * FS(gauge_groups.SU2L, mu, nu, weak_adj)
        - _ONE / _FOUR
        * FS(gauge_groups.SU3C, mu, nu, colour_adj)
        * FS(gauge_groups.SU3C, mu, nu, colour_adj)
    )
    LFermions = (
        I * fields.QL.bar * Gamma(mu) * DC(fields.QL, mu)
        + I * fields.LL.bar * Gamma(mu) * DC(fields.LL, mu)
        + I * fields.uR.bar * Gamma(mu) * DC(fields.uR, mu)
        + I * fields.dR.bar * Gamma(mu) * DC(fields.dR, mu)
        + I * fields.lR.bar * Gamma(mu) * DC(fields.lR, mu)
    )
    LHiggs = (
        DC(fields.Phi.bar, mu) * DC(fields.Phi, mu)
        + parameters.lam
        * parameters.vev**2
        * fields.Phi.bar
        * fields.Phi
        - parameters.lam
        * fields.Phi.bar
        * fields.Phi
        * fields.Phi.bar
        * fields.Phi
    )
    LYukawa = (
        -parameters.Yd(f2, f3)
        * parameters.CKM(f1, f2)
        * fields.QL.bar(spinor, weak_left, f1, colour)
        * fields.dR(spinor, f3, colour)
        * fields.Phi(weak_left)
        - parameters.Ye(f1, f3)
        * fields.LL.bar(spinor, weak_left, f1)
        * fields.lR(spinor, f3)
        * fields.Phi(weak_left)
        - parameters.Yu(f1, f2)
        * fields.QL.bar(spinor, weak_left, f1, colour)
        * fields.uR(spinor, f2, colour)
        * fields.Phi.bar(weak_right)
        * weak_eps2(weak_left, weak_right)
        - parameters.YdDag(f3, f2)
        * parameters.CKMDag(f2, f1)
        * fields.Phi.bar(weak_left)
        * fields.dR.bar(spinor, f3, colour)
        * fields.QL(spinor, weak_left, f1, colour)
        - parameters.YeDag(f3, f1)
        * fields.Phi.bar(weak_left)
        * fields.lR.bar(spinor, f3)
        * fields.LL(spinor, weak_left, f1)
        - parameters.YuDag(f2, f1)
        * weak_eps2(weak_left, weak_right)
        * fields.Phi(weak_right)
        * fields.uR.bar(spinor, f2, colour)
        * fields.QL(spinor, weak_left, f1, colour)
    )

    # Optional source sectors
    LGaugeFixing = DeclaredLagrangian()
    if include_gauge_fixing:
        LGaugeFixing = DeclaredLagrangian.from_item(
            electroweak_rxi_gauge_fixing_lagrangian(fields, parameters)
            + GaugeFixing(
                gauge_groups.SU3C,
                xi=parameter_value_or_symbol(parameters.xiG),
            )
        )

    LGhost = DeclaredLagrangian()
    if include_ghosts:
        LGhost = DeclaredLagrangian.from_item(
            PartialD(fields.ghB.bar, mu) * PartialD(fields.ghB, mu)
            + GhostLagrangian(gauge_groups.SU2L)
            + GhostLagrangian(gauge_groups.SU3C)
            + electroweak_scalar_ghost_lagrangian(fields, parameters)
        )

    LSM = DeclaredLagrangian.from_item(
        LGauge + LFermions + LHiggs + LYukawa + LGaugeFixing + LGhost
    )
    all_parameters = tuple(parameters.__dict__.values())
    source_fields = (
        fields.LL,
        fields.lR,
        fields.QL,
        fields.uR,
        fields.dR,
        fields.Phi,
        fields.B,
        fields.Wi,
        fields.G,
        fields.ghB,
        fields.ghWi,
        fields.ghG,
    )
    gauge_group_tuple = tuple(gauge_groups.__dict__.values())
    source_model = Model(
        name=f"{name} gauge basis",
        gauge_groups=gauge_group_tuple,
        fields=source_fields,
        parameters=all_parameters,
        lagrangian_decl=LSM,
    )

    # Definitions: gauge basis -> physical basis
    sw_symbol = parameters.sw.symbol
    cw_symbol = parameters.cw.symbol
    vev_symbol = parameters.vev.symbol
    ckm = rotation(parameters.CKM, parameters.CKMDag)

    transformations = (
        FieldTransformation(
            fields.B,
            -sw_symbol * fields.Z + cw_symbol * fields.A,
        ),
        FieldTransformation(
            fields.Wi,
            _INV_SQRT2 * fields.W.bar + _INV_SQRT2 * fields.W,
            components={1: 1},
        ),
        FieldTransformation(
            fields.Wi,
            _INV_SQRT2 / I * fields.W.bar - _INV_SQRT2 / I * fields.W,
            components={1: 2},
        ),
        FieldTransformation(
            fields.Wi,
            cw_symbol * fields.Z + sw_symbol * fields.A,
            components={1: 3},
        ),
        FieldTransformation(fields.Phi, -I * fields.GP, components={0: 1}),
        FieldTransformation(
            fields.Phi,
            vev_symbol * _INV_SQRT2
            + _INV_SQRT2 * fields.H
            + I * _INV_SQRT2 * fields.G0,
            components={0: 2},
        ),
        FieldTransformation(fields.LL, ProjM * fields.vl, components={1: 1}),
        FieldTransformation(fields.LL, ProjM * fields.l, components={1: 2}),
        FieldTransformation(fields.lR, ProjP * fields.l),
        FieldTransformation(fields.QL, ProjM * fields.uq, components={1: 1}),
        FieldTransformation(fields.QL, ckm * ProjM * fields.dq, components={1: 2}),
        FieldTransformation(fields.uR, ProjP * fields.uq),
        FieldTransformation(fields.dR, ProjP * fields.dq),
        FieldTransformation(
            fields.ghB,
            -sw_symbol * fields.ghZ + cw_symbol * fields.ghA,
        ),
        FieldTransformation(
            fields.ghWi,
            _INV_SQRT2 * fields.ghWp + _INV_SQRT2 * fields.ghWm,
            components={0: 1},
        ),
        FieldTransformation(
            fields.ghWi,
            -_INV_SQRT2 / I * fields.ghWp + _INV_SQRT2 / I * fields.ghWm,
            components={0: 2},
        ),
        FieldTransformation(
            fields.ghWi,
            cw_symbol * fields.ghZ + sw_symbol * fields.ghA,
            components={0: 3},
        ),
    )
    transform_real_symbols = (
        parameters.g1,
        parameters.g2,
        parameters.g3,
        parameters.lam,
        parameters.vev,
        parameters.MW,
        parameters.MZ,
        parameters.MH,
        parameters.xiA,
        parameters.xiZ,
        parameters.xiW,
        parameters.xiG,
        parameters.sw,
        parameters.cw,
        parameters.ee,
        parameter_value_or_symbol(parameters.xiA),
        parameter_value_or_symbol(parameters.xiZ),
        parameter_value_or_symbol(parameters.xiW),
        parameter_value_or_symbol(parameters.xiG),
    )
    coupling_substitutions = (
        (parameters.g1.symbol, parameters.ee.symbol / parameters.cw.symbol),
        (parameters.g2.symbol, parameters.ee.symbol / parameters.sw.symbol),
    )

    # Compile once per sector, then keep sector labels on optional pieces.
    broken_core = _compile_source_piece(
        DeclaredLagrangian.from_item(LGauge + LFermions + LHiggs + LYukawa),
        name=name,
        gauge_groups=gauge_group_tuple,
        source_fields=source_fields,
        all_parameters=all_parameters,
        transformations=transformations,
        real_symbols=transform_real_symbols,
        coupling_substitutions=coupling_substitutions,
    )
    broken_gauge_fixing = _compile_source_piece(
        LGaugeFixing,
        name=name,
        gauge_groups=gauge_group_tuple,
        source_fields=source_fields,
        all_parameters=all_parameters,
        transformations=transformations,
        real_symbols=transform_real_symbols,
        coupling_substitutions=coupling_substitutions,
        sector="gauge_fixing",
        origin="StandardModelGaugeFixing",
    )
    broken_ghost = _compile_source_piece(
        LGhost,
        name=name,
        gauge_groups=gauge_group_tuple,
        source_fields=source_fields,
        all_parameters=all_parameters,
        transformations=transformations,
        real_symbols=transform_real_symbols,
        coupling_substitutions=coupling_substitutions,
        sector="ghost",
        origin="StandardModelGhost",
    )
    broken_lagrangian = CompiledLagrangian(
        terms=(
            broken_core.terms
            + broken_gauge_fixing.terms
            + broken_ghost.terms
        ),
        parameters=all_parameters,
    )

    physical_fields = (
        fields.vl,
        fields.l,
        fields.uq,
        fields.dq,
        fields.H,
        fields.G0,
        fields.GP,
        fields.W,
        fields.Z,
        fields.A,
        fields.G,
    )
    if include_ghosts:
        physical_fields += (
            fields.ghA,
            fields.ghZ,
            fields.ghWp,
            fields.ghWm,
            fields.ghG,
        )
    physical_model = Model(
        name=name,
        fields=physical_fields,
        parameters=all_parameters,
    )
    physical_model._compiled_lagrangian = broken_lagrangian

    lagrangians = StandardModelLagrangians(
        LGauge=DeclaredLagrangian.from_item(LGauge),
        LFermions=DeclaredLagrangian.from_item(LFermions),
        LHiggs=DeclaredLagrangian.from_item(LHiggs),
        LYukawa=DeclaredLagrangian.from_item(LYukawa),
        LGaugeFixing=LGaugeFixing,
        LGhost=LGhost,
        LSM=LSM,
    )
    return StandardModel(
        model=physical_model,
        source_model=source_model,
        lagrangian=broken_lagrangian,
        indices=indices,
        gauge_groups=gauge_groups,
        fields=fields,
        parameters=parameters,
        lagrangians=lagrangians,
        transformations=transformations,
    )


_DEFAULT_STANDARD_MODEL: StandardModel | None = None
_PLAYGROUND_NAMESPACE: dict[str, object] | None = None

_INDEX_EXPORTS = tuple(StandardModelIndices.__annotations__)
_GAUGE_GROUP_EXPORTS = tuple(StandardModelGaugeGroups.__annotations__)
_FIELD_EXPORTS = tuple(StandardModelFields.__annotations__)
_PARAMETER_EXPORTS = tuple(StandardModelParameters.__annotations__)
_LAGRANGIAN_EXPORTS = tuple(StandardModelLagrangians.__annotations__)
_PLAYGROUND_ALIAS_EXPORTS = (
    "default_standard_model",
    "sm_model",
    "STANDARD_MODEL",
    "SM_MODEL",
    "SM_SOURCE_MODEL",
    "SM_LAGRANGIAN",
    "SM_INDICES",
    "SM_FIELDS",
    "SM_PARAMETERS",
    "SM_GAUGE_GROUP_NAMESPACE",
    "SM_GAUGE_GROUPS",
    "SM_LAGRANGIANS",
    "SM_TRANSFORMATIONS",
    "L_gauge",
    "L_fermions",
    "L_higgs",
    "L_yukawa",
    "L_gauge_fixing",
    "L_ghost",
    "L_tot",
    "mu",
    "nu",
    "weak_adj",
    "colour_adj",
    "spinor",
    "weak_left",
    "weak_right",
    "colour",
    "f1",
    "f2",
    "f3",
)


def _namespace_dict(namespace) -> dict[str, object]:
    return dict(vars(namespace))


def _standard_model_gauge_groups(sm: StandardModel) -> tuple[GaugeGroup, ...]:
    return tuple(_namespace_dict(sm.gauge_groups).values())


def default_standard_model() -> StandardModel:
    """Return the shared broken-phase Standard Model instance for playground use."""

    global _DEFAULT_STANDARD_MODEL
    if _DEFAULT_STANDARD_MODEL is None:
        _DEFAULT_STANDARD_MODEL = build_standard_model()
    return _DEFAULT_STANDARD_MODEL


def sm_model(lagrangian_decl, *, name: str = "") -> Model:
    """Build a ``Model`` with the default Standard-Model metadata attached."""

    sm = default_standard_model()
    return Model(
        lagrangian_decl,
        name=name,
        gauge_groups=_standard_model_gauge_groups(sm),
        parameters=sm.model.parameters,
    )


def _playground_namespace() -> dict[str, object]:
    global _PLAYGROUND_NAMESPACE
    if _PLAYGROUND_NAMESPACE is not None:
        return _PLAYGROUND_NAMESPACE

    sm = default_standard_model()
    lagrangians = sm.lagrangians
    namespace = {
        "STANDARD_MODEL": sm,
        "SM_MODEL": sm.model,
        "SM_SOURCE_MODEL": sm.source_model,
        "SM_LAGRANGIAN": sm.lagrangian,
        "SM_INDICES": sm.indices,
        "SM_FIELDS": sm.fields,
        "SM_PARAMETERS": sm.model.parameters,
        "SM_GAUGE_GROUP_NAMESPACE": sm.gauge_groups,
        "SM_GAUGE_GROUPS": _standard_model_gauge_groups(sm),
        "SM_LAGRANGIANS": lagrangians,
        "SM_TRANSFORMATIONS": sm.transformations,
        "L_gauge": lagrangians.LGauge,
        "L_fermions": lagrangians.LFermions,
        "L_higgs": lagrangians.LHiggs,
        "L_yukawa": lagrangians.LYukawa,
        "L_gauge_fixing": lagrangians.LGaugeFixing,
        "L_ghost": lagrangians.LGhost,
        "L_tot": lagrangians.LSM,
        "mu": S("mu"),
        "nu": S("nu"),
        "weak_adj": S("aW"),
        "colour_adj": S("aC"),
        "spinor": S("sp"),
        "weak_left": S("ii"),
        "weak_right": S("jj"),
        "colour": S("cc"),
        "f1": S("ff1"),
        "f2": S("ff2"),
        "f3": S("ff3"),
    }
    namespace.update(_namespace_dict(sm.indices))
    namespace.update(_namespace_dict(sm.gauge_groups))
    namespace.update(_namespace_dict(sm.fields))
    namespace.update(_namespace_dict(sm.parameters))
    namespace.update(_namespace_dict(lagrangians))

    _PLAYGROUND_NAMESPACE = namespace
    return namespace


def __getattr__(name: str):
    namespace = _playground_namespace()
    if name in namespace:
        return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = (
    "StandardModel",
    "StandardModelFields",
    "StandardModelGaugeGroups",
    "StandardModelIndices",
    "StandardModelLagrangians",
    "StandardModelParameters",
    "build_standard_model",
    "default_standard_model",
    "sm_model",
    "standard_model_weak_tensor_components",
    *_INDEX_EXPORTS,
    *_GAUGE_GROUP_EXPORTS,
    *_FIELD_EXPORTS,
    *_PARAMETER_EXPORTS,
    *_LAGRANGIAN_EXPORTS,
    *_PLAYGROUND_ALIAS_EXPORTS,
)
