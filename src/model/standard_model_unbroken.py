"""Non-BFM unbroken Standard Model builder.

This mirrors the non-BFM content of ``UnbrokenSM_BFM.fr`` while staying within
the current declarative ``Model`` / ``DeclaredLagrangian`` framework:

- gauge groups: ``U1Y``, ``SU2L``, ``SU3C``
- indices: generation, weak fundamental/adjoint, colour fundamental/adjoint
- fields: ``qL``, ``uR``, ``dR``, ``lL``, ``eR``, ``Phi``, ``B``, ``Wi``, ``G``
- parameters: ``g1``, ``g2``, ``g3``, ``lam``, ``muH``, ``Yu``, ``Yd``, ``Ye``,
  plus explicit stand-ins ``YuDag``, ``YdDag``, ``YeDag`` for the Yukawa
  matrices appearing in the hermitian-conjugate terms
- Lagrangian sections: ``LGauge``, ``LFermions``, ``LHiggs``, ``LYukawa``

The generation-carrying fermions are declared as flavor classes with
``class_members`` so the existing ``flavor_expand=True`` machinery can expand
them into concrete generation members, matching the flavor-class examples in
``examples/examples_flavor_expansion.py``.

Explicitly omitted from this builder:

- background/quantum gauge-field splitting
- BFM gauge fixing and ghosts
- electroweak symmetry breaking and vev expansion
- physical ``W/Z/A`` mass eigenstates
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

from .core import Model
from .declared import CovD, FieldStrength, Gamma
from .lagrangian import DeclaredLagrangian
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
from .ssb import standard_model_higgs_doublet

_ONE = Expression.num(1)
_TWO = Expression.num(2)
_THREE = Expression.num(3)
_FOUR = Expression.num(4)
_SIX = Expression.num(6)
_HALF = _ONE / _TWO


@dataclass(frozen=True)
class UnbrokenStandardModelIndices:
    generation: IndexType
    weak_fundamental: IndexType
    weak_adjoint: IndexType
    colour_fundamental: IndexType
    colour_adjoint: IndexType


@dataclass(frozen=True)
class UnbrokenStandardModelGaugeGroups:
    U1Y: GaugeGroup
    SU2L: GaugeGroup
    SU3C: GaugeGroup


@dataclass(frozen=True)
class UnbrokenStandardModelFields:
    qL: Field
    uR: Field
    dR: Field
    lL: Field
    eR: Field
    Phi: Field
    B: Field
    Wi: Field
    G: Field


@dataclass(frozen=True)
class UnbrokenStandardModelParameters:
    g1: Parameter
    g2: Parameter
    g3: Parameter
    lam: Parameter
    muH: Parameter
    Yu: Parameter
    YuDag: Parameter
    Yd: Parameter
    YdDag: Parameter
    Ye: Parameter
    YeDag: Parameter


@dataclass(frozen=True)
class UnbrokenStandardModelLagrangians:
    LGauge: DeclaredLagrangian
    LFermions: DeclaredLagrangian
    LHiggs: DeclaredLagrangian
    LYukawa: DeclaredLagrangian
    LSM: DeclaredLagrangian


@dataclass(frozen=True)
class UnbrokenStandardModel:
    model: Model
    indices: UnbrokenStandardModelIndices
    gauge_groups: UnbrokenStandardModelGaugeGroups
    fields: UnbrokenStandardModelFields
    parameters: UnbrokenStandardModelParameters
    lagrangians: UnbrokenStandardModelLagrangians


def build_unbroken_standard_model(
    *,
    name: str = "SM-unbroken-non-BFM",
) -> UnbrokenStandardModel:
    """Build the non-BFM unbroken Standard Model."""

    generation = flavor_index("Generation", 3, prefix="fl")
    indices = UnbrokenStandardModelIndices(
        generation=generation,
        weak_fundamental=WEAK_FUND_INDEX,
        weak_adjoint=WEAK_ADJ_INDEX,
        colour_fundamental=COLOR_FUND_INDEX,
        colour_adjoint=COLOR_ADJ_INDEX,
    )

    parameters = UnbrokenStandardModelParameters(
        g1=Parameter("g1"),
        g2=Parameter("g2"),
        g3=Parameter("g3"),
        lam=Parameter("lam"),
        muH=Parameter("muH"),
        Yu=Parameter("Yu", indices=(generation, generation), complex_param=True),
        # Until the framework has a symbolic HC/Conjugate operator, the
        # hermitian-conjugate Yukawa terms use explicit stand-in parameters for
        # the matrix entries of Y^\dagger.
        YuDag=Parameter("YuDag", indices=(generation, generation), complex_param=True),
        Yd=Parameter("Yd", indices=(generation, generation), complex_param=True),
        YdDag=Parameter("YdDag", indices=(generation, generation), complex_param=True),
        Ye=Parameter("Ye", indices=(generation, generation), complex_param=True),
        YeDag=Parameter("YeDag", indices=(generation, generation), complex_param=True),
    )

    fields = UnbrokenStandardModelFields(
        qL=Field(
            "qL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("qL0"),
            conjugate_symbol=S("qLbar0"),
            indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": _ONE / _SIX},
            flavor_index=generation,
            class_members=("qL1", "qL2", "qL3"),
        ),
        uR=Field(
            "uR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("uR0"),
            conjugate_symbol=S("uRbar0"),
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": _TWO / _THREE},
            flavor_index=generation,
            class_members=("u", "c", "t"),
        ),
        dR=Field(
            "dR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("dR0"),
            conjugate_symbol=S("dRbar0"),
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": -(_ONE / _THREE)},
            flavor_index=generation,
            class_members=("d", "s", "b"),
        ),
        lL=Field(
            "lL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("lL0"),
            conjugate_symbol=S("lLbar0"),
            indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation),
            quantum_numbers={"Y": -_HALF},
            flavor_index=generation,
            class_members=("lL1", "lL2", "lL3"),
        ),
        eR=Field(
            "eR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            symbol=S("eR0"),
            conjugate_symbol=S("eRbar0"),
            indices=(SPINOR_INDEX, generation),
            quantum_numbers={"Y": -_ONE},
            flavor_index=generation,
            class_members=("e", "mu", "ta"),
        ),
        Phi=standard_model_higgs_doublet(
            name="Phi",
            symbol=S("Phi0"),
            conjugate_symbol=S("Phibar0"),
        ),
        B=Field(
            "B",
            spin=1,
            self_conjugate=True,
            symbol=S("B0"),
            indices=(LORENTZ_INDEX,),
        ),
        Wi=Field(
            "Wi",
            spin=1,
            self_conjugate=True,
            symbol=S("Wi0"),
            indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
        ),
        G=Field(
            "G",
            spin=1,
            self_conjugate=True,
            symbol=S("G0"),
            indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
        ),
    )

    gauge_groups = UnbrokenStandardModelGaugeGroups(
        U1Y=GaugeGroup(
            name="U1Y",
            abelian=True,
            coupling=parameters.g1,
            gauge_boson=fields.B,
            charge="Y",
        ),
        SU2L=GaugeGroup(
            name="SU2L",
            abelian=False,
            coupling=parameters.g2,
            gauge_boson=fields.Wi,
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
            name="SU3C",
            abelian=False,
            coupling=parameters.g3,
            gauge_boson=fields.G,
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

    mu = S("mu")
    nu = S("nu")
    ii = S("ii")
    jj = S("jj")
    sp1 = S("sp1")
    cc = S("cc")
    ff1 = S("ff1")
    ff2 = S("ff2")
    aW = S("aW")
    aC = S("aC")

    LGauge = (
        -(_ONE / _FOUR)
        * FieldStrength(gauge_groups.U1Y, mu, nu)
        * FieldStrength(gauge_groups.U1Y, mu, nu)
        - (_ONE / _FOUR)
        * FieldStrength(gauge_groups.SU2L, mu, nu, aW)
        * FieldStrength(gauge_groups.SU2L, mu, nu, aW)
        - (_ONE / _FOUR)
        * FieldStrength(gauge_groups.SU3C, mu, nu, aC)
        * FieldStrength(gauge_groups.SU3C, mu, nu, aC)
    )

    LFermions = (
        I * fields.qL.bar * Gamma(mu) * CovD(fields.qL, mu)
        + I * fields.lL.bar * Gamma(mu) * CovD(fields.lL, mu)
        + I * fields.uR.bar * Gamma(mu) * CovD(fields.uR, mu)
        + I * fields.dR.bar * Gamma(mu) * CovD(fields.dR, mu)
        + I * fields.eR.bar * Gamma(mu) * CovD(fields.eR, mu)
    )

    LHiggs = (
        CovD(fields.Phi.bar, mu) * CovD(fields.Phi, mu)
        - (parameters.muH ** 2) * fields.Phi.bar * fields.Phi
        - parameters.lam * fields.Phi.bar * fields.Phi * fields.Phi.bar * fields.Phi
    )

    # These `*Dag` parameters are temporary stand-ins for the complex-conjugate
    # Yukawa matrices in the hermitian-conjugate operators until the framework
    # grows a first-class symbolic HC/Conjugate operation.
    LYukawa = (
        -parameters.Yd(ff1, ff2)
        * fields.qL.bar(sp1, ii, ff1, cc)
        * fields.dR(sp1, ff2, cc)
        * fields.Phi(ii)
        - parameters.Ye(ff1, ff2)
        * fields.lL.bar(sp1, ii, ff1)
        * fields.eR(sp1, ff2)
        * fields.Phi(ii)
        - parameters.Yu(ff1, ff2)
        * weak_eps2(ii, jj)
        * fields.qL.bar(sp1, ii, ff1, cc)
        * fields.Phi.bar(jj)
        * fields.uR(sp1, ff2, cc)
        - parameters.YdDag(ff2, ff1)
        * fields.dR.bar(sp1, ff2, cc)
        * fields.Phi.bar(ii)
        * fields.qL(sp1, ii, ff1, cc)
        - parameters.YeDag(ff2, ff1)
        * fields.eR.bar(sp1, ff2)
        * fields.Phi.bar(ii)
        * fields.lL(sp1, ii, ff1)
        - parameters.YuDag(ff2, ff1)
        * weak_eps2(ii, jj)
        * fields.uR.bar(sp1, ff2, cc)
        * fields.Phi(jj)
        * fields.qL(sp1, ii, ff1, cc)
    )

    LSM = LGauge + LFermions + LHiggs + LYukawa
    model = Model(
        name=name,
        gauge_groups=(
            gauge_groups.U1Y,
            gauge_groups.SU2L,
            gauge_groups.SU3C,
        ),
        fields=(
            fields.qL,
            fields.uR,
            fields.dR,
            fields.lL,
            fields.eR,
            fields.Phi,
            fields.B,
            fields.Wi,
            fields.G,
        ),
        parameters=(
            parameters.g1,
            parameters.g2,
            parameters.g3,
            parameters.lam,
            parameters.muH,
            parameters.Yu,
            parameters.YuDag,
            parameters.Yd,
            parameters.YdDag,
            parameters.Ye,
            parameters.YeDag,
        ),
        lagrangian_decl=LSM,
    )

    lagrangians = UnbrokenStandardModelLagrangians(
        LGauge=DeclaredLagrangian.from_item(LGauge),
        LFermions=DeclaredLagrangian.from_item(LFermions),
        LHiggs=DeclaredLagrangian.from_item(LHiggs),
        LYukawa=DeclaredLagrangian.from_item(LYukawa),
        LSM=DeclaredLagrangian.from_item(LSM),
    )

    return UnbrokenStandardModel(
        model=model,
        indices=indices,
        gauge_groups=gauge_groups,
        fields=fields,
        parameters=parameters,
        lagrangians=lagrangians,
    )


__all__ = (
    "UnbrokenStandardModel",
    "UnbrokenStandardModelFields",
    "UnbrokenStandardModelGaugeGroups",
    "UnbrokenStandardModelIndices",
    "UnbrokenStandardModelLagrangians",
    "UnbrokenStandardModelParameters",
    "build_unbroken_standard_model",
)
