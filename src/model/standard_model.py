"""Broken-phase Standard Model generated from gauge-basis declarations.

The source Lagrangian follows the non-BFM sectors in FeynRules ``SM.fr``.
Covariant derivatives and field strengths are compiled in the gauge basis,
finite weak indices are expanded, and one simultaneous field-transformation
stage produces the physical basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from symbolica import Expression, S

from symbolic.spenso_structures import (
    chiral_projector_left,
    chiral_projector_right,
    gauge_generator,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I

from .core import Model
from .declared import CovD, FieldStrength, Gamma, GhostLagrangian, PartialD
from .lagrangian import CompiledLagrangian, DeclaredLagrangian
from .metadata import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    Field,
    GaugeGroup,
    GaugeRepresentation,
    IndexType,
    LORENTZ_INDEX,
    Parameter,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    flavor_index,
)
from .transformations import (
    FieldTransformation,
    ReplacementTerm,
    TransformationContext,
    replacement,
)

_ONE = Expression.num(1)
_TWO = Expression.num(2)
_THREE = Expression.num(3)
_FOUR = Expression.num(4)
_SIX = Expression.num(6)
_HALF = _ONE / _TWO
_INV_SQRT2 = _HALF**_HALF


def _is_zero(value) -> bool:
    return value == 0 or (
        hasattr(value, "expand")
        and value.expand().to_canonical_string() == "0"
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
    sw: Parameter
    cw: Parameter
    ee: Parameter
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


def _diagonal_components(prefix: str) -> dict[tuple[int, int], object]:
    return {
        (row, column): (
            S(f"{prefix}{row}") if row == column else Expression.num(0)
        )
        for row in range(1, 4)
        for column in range(1, 4)
    }


def _ckm_components(cabibbo) -> dict[tuple[int, int], object]:
    cosine = S("cos")(cabibbo)
    sine = S("sin")(cabibbo)
    return {
        (1, 1): cosine,
        (1, 2): sine,
        (1, 3): Expression.num(0),
        (2, 1): -sine,
        (2, 2): cosine,
        (2, 3): Expression.num(0),
        (3, 1): Expression.num(0),
        (3, 2): Expression.num(0),
        (3, 3): Expression.num(1),
    }


def _transpose_components(
    components: dict[tuple[int, int], object],
) -> dict[tuple[int, int], object]:
    return {
        (column, row): value
        for (row, column), value in components.items()
    }


def _concrete_tensor(builder, *values):
    labels = tuple(S(f"component_{position}") for position in range(len(values)))
    expression = builder(*labels)
    for label, value in zip(labels, values):
        expression = expression.replace(label, Expression.num(value))
    return expression


def _real_conjugate(value, *real_symbols):
    result = value.conj() if hasattr(value, "conj") else value
    for symbol in real_symbols:
        result = result.replace(symbol.conj(), symbol)
    return result


def standard_model_weak_tensor_components() -> dict[object, object]:
    """Return explicit SU(2) tensor components used during weak unfolding."""

    components: dict[object, object] = {}
    pauli_over_two = {
        (1, 1, 1): 0,
        (1, 1, 2): _HALF,
        (1, 2, 1): _HALF,
        (1, 2, 2): 0,
        (2, 1, 1): 0,
        (2, 1, 2): -I * _HALF,
        (2, 2, 1): I * _HALF,
        (2, 2, 2): 0,
        (3, 1, 1): _HALF,
        (3, 1, 2): 0,
        (3, 2, 1): 0,
        (3, 2, 2): -_HALF,
    }
    for labels, value in pauli_over_two.items():
        components[_concrete_tensor(weak_gauge_generator, *labels)] = value

    for left in range(1, 4):
        for middle in range(1, 4):
            for right in range(1, 4):
                if len({left, middle, right}) < 3:
                    value = 0
                else:
                    values = (left, middle, right)
                    inversions = sum(
                        first > second
                        for position, first in enumerate(values)
                        for second in values[position + 1 :]
                    )
                    value = -1 if inversions % 2 else 1
                components[
                    _concrete_tensor(
                        weak_structure_constant,
                        left,
                        middle,
                        right,
                    )
                ] = Expression.num(value)

    for left in range(1, 3):
        for right in range(1, 3):
            value = 1 if (left, right) == (1, 2) else -1 if (left, right) == (2, 1) else 0
            components[
                _concrete_tensor(weak_eps2, left, right)
            ] = Expression.num(value)
    return components


def _electroweak_scalar_ghost_lagrangian(
    fields: StandardModelFields,
    parameters: StandardModelParameters,
):
    """Faddeev-Popov scalar term in the electroweak gauge basis.

    For anti-Hermitian scalar-representation generators ``T_a`` and vacuum
    vector ``phi_0``, the real-component scalar product used by ``SM.fr`` is

        (T_a phi_0)^dagger T_b Phi + (T_b Phi)^dagger T_a phi_0.

    Keeping ``Phi`` unexpanded here lets the ordinary field-transformation
    stage generate the ghost masses and Higgs/Goldstone interactions.
    """

    g1 = parameters.g1.symbol
    g2 = parameters.g2.symbol
    vev = parameters.vev.symbol
    zero = Expression.num(0)
    half = _HALF
    generators = (
        (
            (-I * g1 * half, zero),
            (zero, -I * g1 * half),
        ),
        (
            (zero, -I * g2 * half),
            (-I * g2 * half, zero),
        ),
        (
            (zero, -g2 * half),
            (g2 * half, zero),
        ),
        (
            (-I * g2 * half, zero),
            (zero, I * g2 * half),
        ),
    )
    ghosts = (
        fields.ghB,
        fields.ghWi(Expression.num(1)),
        fields.ghWi(Expression.num(2)),
        fields.ghWi(Expression.num(3)),
    )
    antighosts = (
        fields.ghB.bar,
        fields.ghWi.bar(Expression.num(1)),
        fields.ghWi.bar(Expression.num(2)),
        fields.ghWi.bar(Expression.num(3)),
    )
    vacuum = (zero, vev * _INV_SQRT2)
    vacuum_images = tuple(
        tuple(
            sum(
                (matrix[row][column] * vacuum[column] for column in range(2)),
                zero,
            )
            for row in range(2)
        )
        for matrix in generators
    )

    lagrangian = zero
    real_symbols = (g1, g2, vev)
    for left in range(4):
        for right in range(4):
            for component in range(2):
                phi_coefficient = -sum(
                    (
                        _real_conjugate(
                            vacuum_images[left][row],
                            *real_symbols,
                        )
                        * generators[right][row][component]
                        for row in range(2)
                    ),
                    zero,
                )
                phibar_coefficient = -sum(
                    (
                        _real_conjugate(
                            generators[right][row][component],
                            *real_symbols,
                        )
                        * vacuum_images[left][row]
                        for row in range(2)
                    ),
                    zero,
                )
                if not _is_zero(phi_coefficient):
                    lagrangian += (
                        phi_coefficient
                        * antighosts[left]
                        * ghosts[right]
                        * fields.Phi(Expression.num(component + 1))
                    )
                if not _is_zero(phibar_coefficient):
                    lagrangian += (
                        phibar_coefficient
                        * antighosts[left]
                        * ghosts[right]
                        * fields.Phi.bar(Expression.num(component + 1))
                    )
    return lagrangian


def _source_higgs() -> Field:
    return Field(
        "Phi",
        spin=0,
        self_conjugate=False,
        symbol=S("Phi"),
        conjugate_symbol=S("Phibar"),
        indices=(WEAK_FUND_INDEX,),
        quantum_numbers={"Y": _HALF},
    )


def _ghost(
    name: str,
    *,
    indices=(),
    ghost_of=None,
    symbol=None,
    conjugate_symbol=None,
) -> Field:
    return Field(
        name,
        spin=0,
        kind="ghost",
        self_conjugate=False,
        indices=tuple(indices),
        ghost_of=ghost_of,
        symbol=symbol or S(name),
        conjugate_symbol=conjugate_symbol or S(f"{name}bar"),
    )


def _field_labels_by_index(
    context: TransformationContext,
) -> dict[IndexType, object]:
    return {
        index: context.label(slot)
        for slot, index in enumerate(context.occurrence.field.indices)
    }


def _fermion_builder(
    target: Field,
    *,
    chirality: str,
    mixing: Parameter | None = None,
    conjugated: bool = False,
) -> Callable[[TransformationContext], tuple[ReplacementTerm, ...]]:
    def build(context: TransformationContext) -> tuple[ReplacementTerm, ...]:
        labels = _field_labels_by_index(context)
        source_spinor = labels[SPINOR_INDEX]
        source_generation = next(
            label
            for index, label in labels.items()
            if index.is_flavor
        )
        target_spinor = (
            source_spinor
            if conjugated
            else context.fresh(SPINOR_INDEX, "fermion")
        )
        target_generation = source_generation
        coefficient = _ONE

        if mixing is not None:
            generation_index = next(
                index for index in target.indices if index.is_flavor
            )
            target_generation = context.fresh(generation_index, "mix")
            coefficient *= (
                mixing(target_generation, source_generation)
                if conjugated
                else mixing(source_generation, target_generation)
            )

        target_labels = {}
        for slot, index in enumerate(target.indices):
            if index == SPINOR_INDEX:
                target_labels[slot] = target_spinor
            elif index.is_flavor:
                target_labels[slot] = target_generation
            elif index in labels:
                target_labels[slot] = labels[index]

        if conjugated:
            # In the SM source every fermion appears in a closed bilinear.
            # Keeping the chiral projector on the unconjugated endpoint gives
            # the equivalent bilinear form directly: bar(psi) gamma P psi for
            # kinetic currents and bar(psi) P psi for Yukawa terms.
            projector = _ONE
        elif chirality == "left":
            projector = (
                chiral_projector_left(source_spinor, target_spinor)
            )
        else:
            projector = (
                chiral_projector_right(source_spinor, target_spinor)
            )
        coefficient *= projector
        occurrence = target.occurrence(
            conjugated=conjugated,
            labels=target.pack_slot_labels(target_labels),
        )
        return (replacement(coefficient, occurrence),)

    return build


def _standard_model_transformations(
    fields: StandardModelFields,
    parameters: StandardModelParameters,
) -> tuple[FieldTransformation, ...]:
    sw = parameters.sw.value
    cw = parameters.cw.value
    vev = parameters.vev.symbol

    return (
        FieldTransformation(
            fields.B,
            terms=(replacement(-sw, fields.Z), replacement(cw, fields.A)),
        ),
        FieldTransformation(
            fields.Wi,
            components={1: 1},
            terms=(
                replacement(_INV_SQRT2, fields.W.bar),
                replacement(_INV_SQRT2, fields.W),
            ),
        ),
        FieldTransformation(
            fields.Wi,
            components={1: 2},
            terms=(
                replacement(_INV_SQRT2 / I, fields.W.bar),
                replacement(-_INV_SQRT2 / I, fields.W),
            ),
        ),
        FieldTransformation(
            fields.Wi,
            components={1: 3},
            terms=(replacement(cw, fields.Z), replacement(sw, fields.A)),
        ),
        FieldTransformation(
            fields.Phi,
            components={0: 1},
            terms=(replacement(-I, fields.GP),),
        ),
        FieldTransformation(
            fields.Phi,
            components={0: 2},
            terms=(
                replacement(vev * _INV_SQRT2),
                replacement(_INV_SQRT2, fields.H),
                replacement(I * _INV_SQRT2, fields.G0),
            ),
        ),
        FieldTransformation(
            fields.LL,
            components={1: 1},
            builder=_fermion_builder(fields.vl, chirality="left"),
            conjugate_builder=_fermion_builder(
                fields.vl,
                chirality="left",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.LL,
            components={1: 2},
            builder=_fermion_builder(fields.l, chirality="left"),
            conjugate_builder=_fermion_builder(
                fields.l,
                chirality="left",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.lR,
            builder=_fermion_builder(fields.l, chirality="right"),
            conjugate_builder=_fermion_builder(
                fields.l,
                chirality="right",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.QL,
            components={1: 1},
            builder=_fermion_builder(fields.uq, chirality="left"),
            conjugate_builder=_fermion_builder(
                fields.uq,
                chirality="left",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.QL,
            components={1: 2},
            builder=_fermion_builder(
                fields.dq,
                chirality="left",
                mixing=parameters.CKM,
            ),
            conjugate_builder=_fermion_builder(
                fields.dq,
                chirality="left",
                mixing=parameters.CKMDag,
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.uR,
            builder=_fermion_builder(fields.uq, chirality="right"),
            conjugate_builder=_fermion_builder(
                fields.uq,
                chirality="right",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.dR,
            builder=_fermion_builder(fields.dq, chirality="right"),
            conjugate_builder=_fermion_builder(
                fields.dq,
                chirality="right",
                conjugated=True,
            ),
        ),
        FieldTransformation(
            fields.ghB,
            terms=(replacement(-sw, fields.ghZ), replacement(cw, fields.ghA)),
        ),
        FieldTransformation(
            fields.ghWi,
            components={0: 1},
            terms=(
                replacement(_INV_SQRT2, fields.ghWp),
                replacement(_INV_SQRT2, fields.ghWm),
            ),
        ),
        FieldTransformation(
            fields.ghWi,
            components={0: 2},
            terms=(
                replacement(-_INV_SQRT2 / I, fields.ghWp),
                replacement(_INV_SQRT2 / I, fields.ghWm),
            ),
        ),
        FieldTransformation(
            fields.ghWi,
            components={0: 3},
            terms=(replacement(cw, fields.ghZ), replacement(sw, fields.ghA)),
        ),
    )


def build_standard_model(
    *,
    name: str = "Standard Model",
    include_ghosts: bool = True,
) -> StandardModel:
    """Build the broken-phase Standard Model from gauge-basis declarations."""

    generation = flavor_index("Generation", 3, prefix="fl")
    indices = StandardModelIndices(
        generation=generation,
        weak_fundamental=WEAK_FUND_INDEX,
        weak_adjoint=WEAK_ADJ_INDEX,
        colour_fundamental=COLOR_FUND_INDEX,
        colour_adjoint=COLOR_ADJ_INDEX,
    )

    g1 = Parameter("g1")
    g2 = Parameter("g2")
    g3 = Parameter("g3")
    lam = Parameter("lam")
    vev = Parameter("vev")
    gz = (g1.symbol**2 + g2.symbol**2) ** _HALF
    sw = Parameter("sw", value=g1.symbol / gz)
    cw = Parameter("cw", value=g2.symbol / gz)
    ee = Parameter("ee", value=g1.symbol * g2.symbol / gz)
    cabibbo = S("cabi")
    ckm_components = _ckm_components(cabibbo)
    parameters = StandardModelParameters(
        g1=g1,
        g2=g2,
        g3=g3,
        lam=lam,
        vev=vev,
        sw=sw,
        cw=cw,
        ee=ee,
        Yu=Parameter(
            "Yu",
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
            "Yd",
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
            "Ye",
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
        ),
        CKMDag=Parameter(
            "CKMDag",
            indices=(generation, generation),
            complex_param=True,
            components=_transpose_components(ckm_components),
        ),
    )

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
    )
    ghB = _ghost("ghB", ghost_of=B)
    ghWi = _ghost("ghWi", indices=(WEAK_ADJ_INDEX,), ghost_of=Wi)
    ghG = _ghost("ghG", indices=(COLOR_ADJ_INDEX,), ghost_of=G)

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
        Phi=_source_higgs(),
        B=B,
        Wi=Wi,
        ghB=ghB,
        ghWi=ghWi,
        vl=Field(
            "vl",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            flavor_index=generation,
            class_members=("ve", "vm", "vt"),
        ),
        l=Field(
            "l",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            flavor_index=generation,
            class_members=("e", "mu", "ta"),
        ),
        uq=Field(
            "uq",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            flavor_index=generation,
            class_members=("u", "c", "t"),
        ),
        dq=Field(
            "dq",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            flavor_index=generation,
            class_members=("d", "s", "b"),
        ),
        H=Field("H", spin=0, self_conjugate=True),
        G0=Field("G0", spin=0, self_conjugate=True),
        GP=Field(
            "GP",
            spin=0,
            self_conjugate=False,
            conjugate_symbol=S("GM"),
        ),
        W=Field(
            "W",
            spin=1,
            self_conjugate=False,
            conjugate_symbol=S("Wbar"),
            indices=(LORENTZ_INDEX,),
        ),
        Z=Field("Z", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,)),
        A=Field("A", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,)),
        G=G,
        ghA=_ghost("ghA"),
        ghZ=_ghost("ghZ"),
        ghWp=_ghost("ghWp"),
        ghWm=_ghost("ghWm"),
        ghG=ghG,
    )

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

    LGauge = (
        -_ONE / _FOUR
        * FieldStrength(gauge_groups.U1Y, mu, nu)
        * FieldStrength(gauge_groups.U1Y, mu, nu)
        - _ONE / _FOUR
        * FieldStrength(gauge_groups.SU2L, mu, nu, weak_adj)
        * FieldStrength(gauge_groups.SU2L, mu, nu, weak_adj)
        - _ONE / _FOUR
        * FieldStrength(gauge_groups.SU3C, mu, nu, colour_adj)
        * FieldStrength(gauge_groups.SU3C, mu, nu, colour_adj)
    )
    LFermions = (
        I * fields.QL.bar * Gamma(mu) * CovD(fields.QL, mu)
        + I * fields.LL.bar * Gamma(mu) * CovD(fields.LL, mu)
        + I * fields.uR.bar * Gamma(mu) * CovD(fields.uR, mu)
        + I * fields.dR.bar * Gamma(mu) * CovD(fields.dR, mu)
        + I * fields.lR.bar * Gamma(mu) * CovD(fields.lR, mu)
    )
    LHiggs = (
        CovD(fields.Phi.bar, mu) * CovD(fields.Phi, mu)
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
    LGhost = DeclaredLagrangian()
    if include_ghosts:
        LGhost = DeclaredLagrangian.from_item(
            PartialD(fields.ghB.bar, mu) * PartialD(fields.ghB, mu)
            + GhostLagrangian(gauge_groups.SU2L)
            + GhostLagrangian(gauge_groups.SU3C)
            + _electroweak_scalar_ghost_lagrangian(fields, parameters)
        )

    LSM = DeclaredLagrangian.from_item(
        LGauge + LFermions + LHiggs + LYukawa + LGhost
    )
    all_parameters = tuple(parameters.__dict__.values())
    source_model = Model(
        name=f"{name} gauge basis",
        gauge_groups=tuple(gauge_groups.__dict__.values()),
        fields=(
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
        ),
        parameters=all_parameters,
        lagrangian_decl=LSM,
    )

    transformations = _standard_model_transformations(fields, parameters)
    component_lagrangian = source_model.lagrangian().expand_index_components(
        WEAK_FUND_INDEX,
        WEAK_ADJ_INDEX,
        tensor_components=standard_model_weak_tensor_components(),
    )
    broken_lagrangian = component_lagrangian.transform_fields(
        *transformations,
        repeat=False,
        real_symbols=(
            parameters.g1,
            parameters.g2,
            parameters.g3,
            parameters.lam,
            parameters.vev,
            parameters.sw.value,
            parameters.cw.value,
            parameters.ee.value,
        ),
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


__all__ = (
    "StandardModel",
    "StandardModelFields",
    "StandardModelGaugeGroups",
    "StandardModelIndices",
    "StandardModelLagrangians",
    "StandardModelParameters",
    "build_standard_model",
    "standard_model_weak_tensor_components",
)
