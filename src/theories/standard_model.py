"""Broken-phase Standard Model generated from gauge-basis declarations.

The source Lagrangian follows the non-BFM sectors in FeynRules ``SM.fr``.
Covariant derivatives and field strengths are compiled in the gauge basis,
finite weak indices are expanded, and one simultaneous field-transformation
stage produces the physical basis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    CovD,
    DeclaredLagrangian,
    Field,
    FieldStrength,
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


def _parameter_value_or_symbol(parameter: Parameter):
    return parameter.value if parameter.value is not None else parameter.symbol


def _electroweak_generators_and_vacuum_images(
    parameters: StandardModelParameters,
):
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
    return generators, vacuum_images


def _electroweak_omega_coefficients(
    parameters: StandardModelParameters,
) -> tuple[tuple[tuple[str, int, object], ...], ...]:
    generators, vacuum_images = _electroweak_generators_and_vacuum_images(
        parameters
    )
    del generators
    g1 = parameters.g1.symbol
    g2 = parameters.g2.symbol
    vev = parameters.vev.symbol
    real_symbols = (g1, g2, vev)

    coefficients: list[tuple[tuple[str, int, object], ...]] = []
    for gauge_index in range(4):
        components: list[tuple[str, int, object]] = []
        for component in range(2):
            phi_coefficient = -_real_conjugate(
                vacuum_images[gauge_index][component],
                *real_symbols,
            )
            phibar_coefficient = -vacuum_images[gauge_index][component]
            if not _is_zero(phi_coefficient):
                components.append(("phi", component + 1, phi_coefficient))
            if not _is_zero(phibar_coefficient):
                components.append(("phibar", component + 1, phibar_coefficient))
        coefficients.append(tuple(components))
    return tuple(coefficients)


def _electroweak_xi_matrices(
    parameters: StandardModelParameters,
) -> tuple[dict[tuple[int, int], object], dict[tuple[int, int], object]]:
    sw = _parameter_value_or_symbol(parameters.sw)
    cw = _parameter_value_or_symbol(parameters.cw)
    xiA = _parameter_value_or_symbol(parameters.xiA)
    xiZ = _parameter_value_or_symbol(parameters.xiZ)
    xiW = _parameter_value_or_symbol(parameters.xiW)
    one = Expression.num(1)

    xi_inverse = {
        (0, 0): cw**2 / xiA + sw**2 / xiZ,
        (0, 3): cw * sw * (one / xiA - one / xiZ),
        (1, 1): one / xiW,
        (2, 2): one / xiW,
        (3, 0): cw * sw * (one / xiA - one / xiZ),
        (3, 3): sw**2 / xiA + cw**2 / xiZ,
    }
    xi_matrix = {
        (0, 0): cw**2 * xiA + sw**2 * xiZ,
        (0, 3): cw * sw * (xiA - xiZ),
        (1, 1): xiW,
        (2, 2): xiW,
        (3, 0): cw * sw * (xiA - xiZ),
        (3, 3): sw**2 * xiA + cw**2 * xiZ,
    }
    return xi_inverse, xi_matrix


def _electroweak_gauge_basis_field(
    fields: StandardModelFields,
    *,
    gauge_index: int,
    lorentz_label,
):
    if gauge_index == 0:
        return fields.B(lorentz_label)
    return fields.Wi(lorentz_label, Expression.num(gauge_index))


def _electroweak_scalar_component(
    fields: StandardModelFields,
    *,
    kind: str,
    component: int,
):
    if kind == "phi":
        return fields.Phi(Expression.num(component))
    return fields.Phi.bar(Expression.num(component))


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

    zero = Expression.num(0)
    generators, vacuum_images = _electroweak_generators_and_vacuum_images(
        parameters
    )
    _xi_inverse, xi_matrix = _electroweak_xi_matrices(parameters)
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

    lagrangian = zero
    g1 = parameters.g1.symbol
    g2 = parameters.g2.symbol
    vev = parameters.vev.symbol
    real_symbols = (g1, g2, vev)
    for left in range(4):
        for right in range(4):
            for mixed_left in range(4):
                xi_coefficient = xi_matrix.get((left, mixed_left), zero)
                if _is_zero(xi_coefficient):
                    continue
                for component in range(2):
                    phi_coefficient = -sum(
                        (
                            _real_conjugate(
                                vacuum_images[mixed_left][row],
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
                            * vacuum_images[mixed_left][row]
                            for row in range(2)
                        ),
                        zero,
                    )
                    if not _is_zero(phi_coefficient):
                        lagrangian += (
                            xi_coefficient
                            * phi_coefficient
                            * antighosts[left]
                            * ghosts[right]
                            * fields.Phi(Expression.num(component + 1))
                        )
                    if not _is_zero(phibar_coefficient):
                        lagrangian += (
                            xi_coefficient
                            * phibar_coefficient
                            * antighosts[left]
                            * ghosts[right]
                            * fields.Phi.bar(Expression.num(component + 1))
                        )
    return lagrangian


def _electroweak_rxi_gauge_fixing_lagrangian(
    fields: StandardModelFields,
    parameters: StandardModelParameters,
):
    """Electroweak ``R_xi`` gauge fixing in the gauge basis.

    The gauge-fixing functions are built as

        F_a = del.V_a - Omega_a(Phi),

    with ``Omega_a`` determined by the Higgs vacuum vector and gauge-basis
    generators. Neutral gauge parameters are diagonal in the physical
    ``(A, Z)`` basis and rotated back into the ``(B, W3)`` basis.
    """

    mu = S("mu")
    nu = S("nu")
    zero = Expression.num(0)
    xi_inverse, xi_matrix = _electroweak_xi_matrices(parameters)
    omega_coefficients = _electroweak_omega_coefficients(parameters)

    lagrangian = None

    def add(term):
        nonlocal lagrangian
        lagrangian = term if lagrangian is None else lagrangian + term

    for (left, right), coefficient in xi_inverse.items():
        add(
            -coefficient
            * _HALF
            * PartialD(
                _electroweak_gauge_basis_field(
                    fields,
                    gauge_index=left,
                    lorentz_label=mu,
                ),
                mu,
            )
            * PartialD(
                _electroweak_gauge_basis_field(
                    fields,
                    gauge_index=right,
                    lorentz_label=nu,
                ),
                nu,
            )
        )

    for gauge_index, components in enumerate(omega_coefficients):
        gauge_field = _electroweak_gauge_basis_field(
            fields,
            gauge_index=gauge_index,
            lorentz_label=mu,
        )
        for kind, component, coefficient in components:
            add(
                coefficient
                * PartialD(
                    _electroweak_scalar_component(
                        fields,
                        kind=kind,
                        component=component,
                    ),
                    mu,
                )
                * gauge_field
            )

    for (left, right), coefficient in xi_matrix.items():
        if _is_zero(coefficient):
            continue
        for left_kind, left_component, left_coefficient in omega_coefficients[left]:
            left_scalar = _electroweak_scalar_component(
                fields,
                kind=left_kind,
                component=left_component,
            )
            for (
                right_kind,
                right_component,
                right_coefficient,
            ) in omega_coefficients[right]:
                right_scalar = _electroweak_scalar_component(
                    fields,
                    kind=right_kind,
                    component=right_component,
                )
                add(
                    -coefficient
                    * _HALF
                    * left_coefficient
                    * right_coefficient
                    * left_scalar
                    * right_scalar
                )

    return zero if lagrangian is None else lagrangian


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
    mass=None,
    quantum_numbers=None,
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
        mass=mass,
        quantum_numbers=dict(quantum_numbers or {}),
    )


def _standard_model_transformations(
    fields: StandardModelFields,
    parameters: StandardModelParameters,
) -> tuple[FieldTransformation, ...]:
    sw = parameters.sw.value
    cw = parameters.cw.value
    vev = parameters.vev.symbol
    ckm = rotation(parameters.CKM, parameters.CKMDag)

    return (
        # gauge-boson definitions (FeynRules SM.fr field expressions)
        FieldTransformation(fields.B, -sw * fields.Z + cw * fields.A),
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
            cw * fields.Z + sw * fields.A,
            components={1: 3},
        ),
        # Higgs/Goldstone definitions (the bare vev is a vacuum shift)
        FieldTransformation(fields.Phi, -I * fields.GP, components={0: 1}),
        FieldTransformation(
            fields.Phi,
            vev * _INV_SQRT2 + _INV_SQRT2 * fields.H + I * _INV_SQRT2 * fields.G0,
            components={0: 2},
        ),
        # chiral fermion definitions: source -> Proj * target (SM.fr ProjM/ProjP)
        FieldTransformation(fields.LL, ProjM * fields.vl, components={1: 1}),
        FieldTransformation(fields.LL, ProjM * fields.l, components={1: 2}),
        FieldTransformation(fields.lR, ProjP * fields.l),
        FieldTransformation(fields.QL, ProjM * fields.uq, components={1: 1}),
        FieldTransformation(fields.QL, ckm * ProjM * fields.dq, components={1: 2}),
        FieldTransformation(fields.uR, ProjP * fields.uq),
        FieldTransformation(fields.dR, ProjP * fields.dq),
        # ghost definitions
        FieldTransformation(fields.ghB, -sw * fields.ghZ + cw * fields.ghA),
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
            cw * fields.ghZ + sw * fields.ghA,
            components={0: 3},
        ),
    )


def build_standard_model(
    *,
    name: str = "Standard Model",
    include_ghosts: bool = True,
    include_gauge_fixing: bool = True,
    xiA=1,
    xiZ=1,
    xiW=1,
    xiG=1,
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
    mw_relation = g2.symbol * vev.symbol / 2
    mz_relation = gz * vev.symbol / 2
    mh_relation = (2 * lam.symbol * vev.symbol**2) ** _HALF
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
        MW=Parameter("MW", value=mw_relation),
        MZ=Parameter("MZ", value=mz_relation),
        MH=Parameter("MH", value=mh_relation),
        sw=sw,
        cw=cw,
        ee=ee,
        xiA=Parameter("xiA", internal=False, value=xiA),
        xiZ=Parameter("xiZ", internal=False, value=xiZ),
        xiW=Parameter("xiW", internal=False, value=xiW),
        xiG=Parameter("xiG", internal=False, value=xiG),
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
            unitary_partner="CKMDag",
        ),
        CKMDag=Parameter(
            "CKMDag",
            indices=(generation, generation),
            complex_param=True,
            components=_transpose_components(ckm_components),
            unitary_partner="CKM",
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
    xiA_value = _parameter_value_or_symbol(parameters.xiA)
    xiZ_value = _parameter_value_or_symbol(parameters.xiZ)
    xiW_value = _parameter_value_or_symbol(parameters.xiW)
    z_ghost_mass = (xiZ_value * parameters.MZ.symbol**2) ** _HALF
    w_ghost_mass = (xiW_value * parameters.MW.symbol**2) ** _HALF

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
        conjugate_symbol=S("GM"),
        mass=w_ghost_mass,
        quantum_numbers={"Q": _ONE},
        goldstone_of=W_field,
    )
    ghA_field = _ghost(
        "ghA",
        ghost_of=A_field,
        mass=Expression.num(0),
        quantum_numbers={"GhostNumber": _ONE},
    )
    ghZ_field = _ghost(
        "ghZ",
        ghost_of=Z_field,
        mass=z_ghost_mass,
        quantum_numbers={"GhostNumber": _ONE},
    )
    ghWp_field = _ghost(
        "ghWp",
        ghost_of=W_field,
        mass=w_ghost_mass,
        quantum_numbers={"GhostNumber": _ONE, "Q": _ONE},
    )
    ghWm_field = _ghost(
        "ghWm",
        ghost_of=W_field,
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
    LGaugeFixing = DeclaredLagrangian()
    if include_gauge_fixing:
        LGaugeFixing = DeclaredLagrangian.from_item(
            _electroweak_rxi_gauge_fixing_lagrangian(fields, parameters)
            + GaugeFixing(
                gauge_groups.SU3C,
                xi=_parameter_value_or_symbol(parameters.xiG),
            )
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
    source_model = Model(
        name=f"{name} gauge basis",
        gauge_groups=tuple(gauge_groups.__dict__.values()),
        fields=source_fields,
        parameters=all_parameters,
        lagrangian_decl=LSM,
    )

    transformations = _standard_model_transformations(fields, parameters)
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
        parameters.sw.value,
        parameters.cw.value,
        parameters.ee.value,
        _parameter_value_or_symbol(parameters.xiA),
        _parameter_value_or_symbol(parameters.xiZ),
        _parameter_value_or_symbol(parameters.xiW),
        _parameter_value_or_symbol(parameters.xiG),
    )

    def compile_source_piece(
        lagrangian_decl: DeclaredLagrangian,
        *,
        sector: str | None = None,
        origin: str = "",
    ) -> CompiledLagrangian:
        if not lagrangian_decl.source_terms:
            return CompiledLagrangian(parameters=all_parameters)
        source_piece = Model(
            name=f"{name} source piece",
            gauge_groups=tuple(gauge_groups.__dict__.values()),
            fields=source_fields,
            parameters=all_parameters,
            lagrangian_decl=lagrangian_decl,
        )
        component_lagrangian = source_piece.lagrangian().expand_index_components(
            WEAK_FUND_INDEX,
            WEAK_ADJ_INDEX,
            tensor_components=standard_model_weak_tensor_components(),
        )
        broken_piece = component_lagrangian.transform_fields(
            *transformations,
            repeat=False,
            real_symbols=transform_real_symbols,
        )
        broken_piece = broken_piece.simplify_parameter_identities()
        if sector is None:
            return broken_piece
        return CompiledLagrangian(
            terms=tuple(
                replace(
                    term,
                    sector=sector,
                    origin=origin or term.origin,
                )
                for term in broken_piece.terms
            ),
            parameters=broken_piece.parameters,
        )

    broken_core = compile_source_piece(
        DeclaredLagrangian.from_item(LGauge + LFermions + LHiggs + LYukawa)
    )
    broken_gauge_fixing = compile_source_piece(
        LGaugeFixing,
        sector="gauge_fixing",
        origin="StandardModelGaugeFixing",
    )
    broken_ghost = compile_source_piece(
        LGhost,
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
