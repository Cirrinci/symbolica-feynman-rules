"""Simple unbroken-gauge-basis SMEFT model bundled in ``models/SMEFT2``.

This follows the field names and operator sectors in
``reference/feynrules/SMEFT_Green_Bpreserving.fr`` while staying inside the
current declarative surface of FeynPy. The implementation intentionally avoids
extra framework machinery: define the parameters, gauge groups, fields and the
Lagrangian in a single builder, close to how a user would write it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from symbolica import Expression, S

from feynpy import (
    COLOR_ADJ_INDEX,
    COLOR_FUND_INDEX,
    DC,
    DeclaredLagrangian,
    Field,
    FS,
    Gamma,
    GaugeGroup,
    GaugeRepresentation,
    LORENTZ_INDEX,
    Model,
    Parameter,
    PartialD,
    SPINOR_INDEX,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
    flavor_index,
)
from symbolic.spenso_structures import (
    dirac_charge_conjugation,
    gauge_generator,
    lorentz_levi_civita,
    structure_constant,
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I


ZERO = Expression.num(0)
ONE = Expression.num(1)
TWO = Expression.num(2)
THREE = Expression.num(3)
FOUR = Expression.num(4)
SIX = Expression.num(6)
HALF = ONE / TWO


OMITTED_SECTORS = ()


@dataclass(frozen=True)
class SMEFT2Bundle:
    model: Model
    fields: dict[str, Field]
    parameters: dict[str, Parameter]
    gauge_groups: dict[str, GaugeGroup]
    lagrangians: dict[str, DeclaredLagrangian]
    omitted_sectors: tuple[str, ...] = OMITTED_SECTORS


def _param(name: str, *, indices=(), complex_param: bool = False) -> Parameter:
    return Parameter(name, indices=indices, complex_param=complex_param)


def _sum_supported_lagrangian_blocks(*blocks):
    total = None
    for block in blocks:
        if block is ZERO:
            continue
        total = block if total is None else total + block
    return ZERO if total is None else total


def _field_strength(group, mu, nu, adjoint=None):
    if adjoint is None:
        return FS(group, mu, nu)
    return FS(group, mu, nu, adjoint)


def _covd_fs(group, mu, nu, derivative, adjoint=None):
    fs_factor = _field_strength(group, mu, nu, adjoint)
    if adjoint is None:
        return PartialD(fs_factor, derivative)
    return DC(fs_factor, derivative)


def _dual_fs(group, mu, nu, rho, sigma, adjoint=None):
    return HALF * lorentz_levi_civita(mu, nu, rho, sigma) * _field_strength(
        group, rho, sigma, adjoint
    )


def _dual_covd_fs(group, mu, nu, derivative, rho, sigma, adjoint=None):
    return HALF * lorentz_levi_civita(mu, nu, rho, sigma) * _covd_fs(
        group, rho, sigma, derivative, adjoint
    )


def build_smeft_green_bpreserving(
    *,
    name: str = "SMEFT_Green_Bpreserving",
) -> SMEFT2Bundle:
    """Return the supported unbroken-basis SMEFT model."""

    generation = flavor_index("Generation", 3, prefix="f")

    parameters: dict[str, Parameter] = {
        "g1": _param("g1"),
        "g2": _param("g2"),
        "g3": _param("g3"),
        "muH": _param("muH"),
        "lam": _param("lam"),
        "yl": _param(
            "yl", indices=(generation, generation), complex_param=True
        ),
        "yu": _param(
            "yu", indices=(generation, generation), complex_param=True
        ),
        "yd": _param(
            "yd", indices=(generation, generation), complex_param=True
        ),
    }

    for param_name in (
        "alphaOmuH2",
        "alphaKB",
        "alphaKW",
        "alphaKG",
        "alphaKH",
        "alphaOlambda",
        "alphaO3G",
        "alphaO3Gt",
        "alphaO3W",
        "alphaO3Wt",
        "alphaR2G",
        "alphaR2W",
        "alphaR2B",
        "alphaOHG",
        "alphaOHGt",
        "alphaOHW",
        "alphaOHWt",
        "alphaOHB",
        "alphaOHBt",
        "alphaOHWB",
        "alphaOHWBt",
        "alphaRWDH",
        "alphaRBDH",
        "alphaRDH",
        "alphaOHBox",
        "alphaOHD",
        "alphaRHDp",
        "alphaRHDpp",
        "alphaOH",
    ):
        parameters[param_name] = _param(param_name)

    for param_name in (
        "alphaKq",
        "alphaKl",
        "alphaKu",
        "alphaKd",
        "alphaKe",
        "alphaOlambdad",
        "alphaOlambdae",
        "alphaOlambdau",
        "alphaOuG",
        "alphaOuW",
        "alphaOuB",
        "alphaOdG",
        "alphaOdW",
        "alphaOdB",
        "alphaOeW",
        "alphaOeB",
        "alphaOHq1",
        "alphaRHq1p",
        "alphaRHq1pp",
        "alphaOHq3",
        "alphaRHq3p",
        "alphaRHq3pp",
        "alphaOHu",
        "alphaRHup",
        "alphaRHupp",
        "alphaOHd",
        "alphaRHdp",
        "alphaRHdpp",
        "alphaOHud",
        "alphaOHl1",
        "alphaRHl1p",
        "alphaRHl1pp",
        "alphaOHl3",
        "alphaRHl3p",
        "alphaRHl3pp",
        "alphaOHe",
        "alphaRHep",
        "alphaRHepp",
        "alphaOuH",
        "alphaOdH",
        "alphaOeH",
        "alphaWeinberg",
        "alphaRqD",
        "alphaRuD",
        "alphaRdD",
        "alphaRlD",
        "alphaReD",
        "alphaRuHD1",
        "alphaRuHD2",
        "alphaRuHD3",
        "alphaRuHD4",
        "alphaRdHD1",
        "alphaRdHD2",
        "alphaRdHD3",
        "alphaRdHD4",
        "alphaReHD1",
        "alphaReHD2",
        "alphaReHD3",
        "alphaReHD4",
        "alphaRGq",
        "alphaRGqp",
        "alphaRGqtp",
        "alphaRWq",
        "alphaRWqp",
        "alphaRWqtp",
        "alphaRBq",
        "alphaRBqp",
        "alphaRBqtp",
        "alphaRGu",
        "alphaRGup",
        "alphaRGutp",
        "alphaRBu",
        "alphaRBup",
        "alphaRButp",
        "alphaRGd",
        "alphaRGdp",
        "alphaRGdtp",
        "alphaRBd",
        "alphaRBdp",
        "alphaRBdtp",
        "alphaRWl",
        "alphaRWlp",
        "alphaRWltp",
        "alphaRBl",
        "alphaRBlp",
        "alphaRBltp",
        "alphaRBe",
        "alphaRBep",
        "alphaRBetp",
        "alphaEuG",
        "alphaEuW",
        "alphaEuB",
        "alphaEdG",
        "alphaEdW",
        "alphaEdB",
        "alphaEeW",
        "alphaEeB",
        "alphaEuH",
        "alphaEdH",
        "alphaEeH",
        "alphaEGq",
        "alphaEGqp",
        "alphaEGqtp",
        "alphaEWq",
        "alphaEWqp",
        "alphaEWqtp",
        "alphaEBq",
        "alphaEBqp",
        "alphaEBqtp",
        "alphaEGu",
        "alphaEGup",
        "alphaEGutp",
        "alphaEBu",
        "alphaEBup",
        "alphaEButp",
        "alphaEGd",
        "alphaEGdp",
        "alphaEGdtp",
        "alphaEBd",
        "alphaEBdp",
        "alphaEBdtp",
        "alphaEWl",
        "alphaEWlp",
        "alphaEWltp",
        "alphaEBl",
        "alphaEBlp",
        "alphaEBltp",
        "alphaEBe",
        "alphaEBep",
        "alphaEBetp",
    ):
        parameters[param_name] = _param(
            param_name,
            indices=(generation, generation),
            complex_param=True,
        )

    for param_name in (
        "alphaOqq1",
        "alphaOqq3",
        "alphaOuu",
        "alphaOdd",
        "alphaOud1",
        "alphaOud8",
        "alphaOqu1",
        "alphaOqu8",
        "alphaOqd1",
        "alphaOqd8",
        "alphaOquqd1",
        "alphaOquqd8",
        "alphaOll",
        "alphaOee",
        "alphaOle",
        "alphaOlq1",
        "alphaOlq3",
        "alphaOeu",
        "alphaOed",
        "alphaOqe",
        "alphaOlu",
        "alphaOld",
        "alphaOledq",
        "alphaOlequ1",
        "alphaEqu",
        "alphaEqu8",
        "alphaEqd",
        "alphaEqd8",
        "alphaEqutwo",
        "alphaEqutwo8",
        "alphaEqdtwo",
        "alphaEqdtwo8",
        "alphaEquqdtwo",
        "alphaEquqdtwo8",
        "alphaEuu8",
        "alphaEuuthree",
        "alphaEuuthree8",
        "alphaEdd8",
        "alphaEddthree",
        "alphaEddthree8",
        "alphaEud",
        "alphaEud8",
        "alphaEudthree",
        "alphaEudthree8",
        "alphaEudthreep",
        "alphaEudthree8p",
        "alphaEquthree",
        "alphaEquthree8",
        "alphaEqdthree",
        "alphaEqdthree8",
        "alphaEqq8",
        "alphaEqq38",
        "alphaEqqthree1",
        "alphaEqqthree3",
        "alphaEqqthree8",
        "alphaEqqthree38",
        "alphaEeethree",
        "alphaEll3",
        "alphaEllthree",
        "alphaEllthree3",
        "alphaEle",
        "alphaEletwo",
        "alphaElethree",
        "alphaEeu",
        "alphaEed",
        "alphaEeuthree",
        "alphaEedthree",
        "alphaEeuthreep",
        "alphaEedthreep",
        "alphaElq",
        "alphaElq3",
        "alphaElqthree",
        "alphaElqthree3",
        "alphaElqthreep",
        "alphaElqthree3p",
        "alphaElequtwo",
        "alphaEluqe",
        "alphaEluqetwo",
        "alphaElu",
        "alphaEld",
        "alphaEqe",
        "alphaEledqtwo",
        "alphaElutwo",
        "alphaEldtwo",
        "alphaEqetwo",
        "alphaElqde",
        "alphaEluthree",
        "alphaEldthree",
        "alphaEqethree",
        "alphaElqdethree",
        "alphaEcll",
        "alphaEclltwo",
        "alphaEcqq",
        "alphaEcqqtwo",
        "alphaEcqqp",
        "alphaEcqqptwo",
        "alphaEcql",
        "alphaEcqltwo",
        "alphaEcqlp",
        "alphaEcqlptwo",
        "alphaEcee",
        "alphaEceetwo",
        "alphaEceu",
        "alphaEceutwo",
        "alphaEced",
        "alphaEcedtwo",
        "alphaEcuu",
        "alphaEcuutwo",
        "alphaEcdd",
        "alphaEcddtwo",
        "alphaEcud",
        "alphaEcudtwo",
        "alphaEcudp",
        "alphaEcudptwo",
        "alphaEcle",
        "alphaEcqe",
        "alphaEclu",
        "alphaEcld",
        "alphaEcqu",
        "alphaEcqd",
        "alphaEcqup",
        "alphaEcqdp",
        "alphaEcqedl",
        "alphaEclethree",
        "alphaEcqethree",
        "alphaEcluthree",
        "alphaEcldthree",
        "alphaEcquthree",
        "alphaEcqdthree",
        "alphaEcqupthree",
        "alphaEcqdpthree",
        "alphaEcqedlthree",
        "alphaEcuelq",
        "alphaEcudqq",
        "alphaEcuelqtwo",
        "alphaEcudqqtwo",
    ):
        parameters[param_name] = _param(
            param_name,
            indices=(generation, generation, generation, generation),
            complex_param=True,
        )

    fields: dict[str, Field] = {
        "B": Field("B", spin=1, self_conjugate=True, indices=(LORENTZ_INDEX,)),
        "Wi": Field(
            "Wi",
            spin=1,
            self_conjugate=True,
            indices=(LORENTZ_INDEX, WEAK_ADJ_INDEX),
        ),
        "G": Field(
            "G",
            spin=1,
            self_conjugate=True,
            indices=(LORENTZ_INDEX, COLOR_ADJ_INDEX),
        ),
        "LL": Field(
            "LL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation),
            quantum_numbers={"Y": -HALF},
        ),
        "LR": Field(
            "LR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation),
            quantum_numbers={"Y": -ONE},
        ),
        "QL": Field(
            "QL",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, WEAK_FUND_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": ONE / SIX},
        ),
        "UR": Field(
            "UR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": TWO / THREE},
        ),
        "DR": Field(
            "DR",
            spin=Fraction(1, 2),
            self_conjugate=False,
            indices=(SPINOR_INDEX, generation, COLOR_FUND_INDEX),
            quantum_numbers={"Y": -(ONE / THREE)},
        ),
        "Phi": Field(
            "Phi",
            spin=0,
            self_conjugate=False,
            conjugate_symbol=S("Phibar"),
            indices=(WEAK_FUND_INDEX,),
            quantum_numbers={"Y": HALF},
        ),
    }

    gauge_groups: dict[str, GaugeGroup] = {
        "U1Y": GaugeGroup(
            "U1Y",
            abelian=True,
            coupling=parameters["g1"],
            gauge_boson=fields["B"],
            charge="Y",
        ),
        "SU2L": GaugeGroup(
            "SU2L",
            abelian=False,
            coupling=parameters["g2"],
            gauge_boson=fields["Wi"],
            structure_constant=weak_structure_constant,
            representations=(
                GaugeRepresentation(
                    WEAK_FUND_INDEX,
                    weak_gauge_generator,
                    name="doublet",
                ),
            ),
        ),
        "SU3C": GaugeGroup(
            "SU3C",
            abelian=False,
            coupling=parameters["g3"],
            gauge_boson=fields["G"],
            structure_constant=structure_constant,
            representations=(
                GaugeRepresentation(
                    COLOR_FUND_INDEX,
                    gauge_generator,
                    name="fundamental",
                ),
            ),
        ),
    }

    B = fields["B"]
    Wi = fields["Wi"]
    G = fields["G"]
    LL = fields["LL"]
    LR = fields["LR"]
    QL = fields["QL"]
    UR = fields["UR"]
    DR = fields["DR"]
    Phi = fields["Phi"]

    weak_kind = WEAK_FUND_INDEX.kind
    generation_kind = generation.kind
    color_kind = COLOR_FUND_INDEX.kind

    def ql(*, sp=None, w=None, f=None, c=None, bar=False):
        labels = {}
        if w is not None:
            labels[weak_kind] = w
        if f is not None:
            labels[generation_kind] = f
        if c is not None:
            labels[color_kind] = c
        target = QL.bar if bar else QL
        if sp is None:
            return target(index_labels=labels)
        return target(sp, index_labels=labels)

    def ur(*, sp=None, f=None, c=None, bar=False):
        labels = {}
        if f is not None:
            labels[generation_kind] = f
        if c is not None:
            labels[color_kind] = c
        target = UR.bar if bar else UR
        if sp is None:
            return target(index_labels=labels)
        return target(sp, index_labels=labels)

    def dr(*, sp=None, f=None, c=None, bar=False):
        labels = {}
        if f is not None:
            labels[generation_kind] = f
        if c is not None:
            labels[color_kind] = c
        target = DR.bar if bar else DR
        if sp is None:
            return target(index_labels=labels)
        return target(sp, index_labels=labels)

    def ll(*, sp=None, w=None, f=None, bar=False):
        labels = {}
        if w is not None:
            labels[weak_kind] = w
        if f is not None:
            labels[generation_kind] = f
        target = LL.bar if bar else LL
        if sp is None:
            return target(index_labels=labels)
        return target(sp, index_labels=labels)

    def lr(*, sp=None, f=None, bar=False):
        labels = {}
        if f is not None:
            labels[generation_kind] = f
        target = LR.bar if bar else LR
        if sp is None:
            return target(index_labels=labels)
        return target(sp, index_labels=labels)

    def weak_t(adjoint, left, right):
        return TWO * weak_gauge_generator(adjoint, left, right)

    def phitilde(target, source):
        return weak_eps2(target, source) * Phi.bar(source)

    def phitildebar(target, source):
        return weak_eps2(target, source) * Phi(source)

    def sigma_term(prefactor, left, right, mu, nu):
        return (
            (I / TWO) * prefactor * left * Gamma(mu) * Gamma(nu) * right
            - (I / TWO) * prefactor * left * Gamma(nu) * Gamma(mu) * right
        )

    def sigma_matrix(left, right, mu, nu, middle):
        return (I / TWO) * gamma2(left, right, mu, nu, middle) - (
            I / TWO
        ) * gamma2(left, right, nu, mu, middle)

    def gamma2(left, right, mu, nu, middle):
        return Gamma(left, middle, mu) * Gamma(middle, right, nu)

    def gamma3(left, right, mu, nu, rho, middle1, middle2):
        return (
            Gamma(left, middle1, mu)
            * Gamma(middle1, middle2, nu)
            * Gamma(middle2, right, rho)
        )

    mu, nu, rho, rho2, sigma = (
        S("mu"),
        S("nu"),
        S("rho"),
        S("rho2"),
        S("sigma"),
    )
    w1, w2, w3, w4 = S("w1"), S("w2"), S("w3"), S("w4")
    aW, aW1, aW2, aW3 = S("aW"), S("aW1"), S("aW2"), S("aW3")
    aC, aC1, aC2, aC3 = S("aC"), S("aC1"), S("aC2"), S("aC3")
    f1, f2, f3, f4 = S("f1"), S("f2"), S("f3"), S("f4")
    c1, c2, c3, c4 = S("c1"), S("c2"), S("c3"), S("c4")
    sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10, sp11, sp12 = (
        S("sp1"),
        S("sp2"),
        S("sp3"),
        S("sp4"),
        S("sp5"),
        S("sp6"),
        S("sp7"),
        S("sp8"),
        S("sp9"),
        S("sp10"),
        S("sp11"),
        S("sp12"),
    )

    p = parameters
    g = gauge_groups

    LGauge = (
        -(ONE / FOUR) * FS(g["U1Y"], mu, nu) * FS(g["U1Y"], mu, nu)
        - (ONE / FOUR) * FS(g["SU2L"], mu, nu, aW) * FS(g["SU2L"], mu, nu, aW)
        - (ONE / FOUR) * FS(g["SU3C"], mu, nu, aC) * FS(g["SU3C"], mu, nu, aC)
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
        - p["muH"]**2 * Phi.bar * Phi
        - p["lam"] * Phi.bar * Phi * Phi.bar * Phi
    )

    LYukawa = (
        -p["yd"](f1, f2)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * dr(f=f2, c=c1)
        * Phi(w1)
        - p["yl"](f1, f2) * ll(w=w1, f=f1, bar=True) * lr(f=f2) * Phi(w1)
        - p["yu"](f1, f2)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * ur(f=f2, c=c1)
        * phitilde(w1, w2)
        - p["yd"](f1, f2).conj()
        * Phi.bar(w1)
        * dr(f=f2, c=c1, bar=True)
        * ql(w=w1, f=f1, c=c1)
        - p["yl"](f1, f2).conj()
        * Phi.bar(w1)
        * lr(f=f2, bar=True)
        * ll(w=w1, f=f1)
        - p["yu"](f1, f2).conj()
        * phitildebar(w1, w2)
        * ur(f=f2, c=c1, bar=True)
        * ql(w=w1, f=f1, c=c1)
    )

    L2Higgs = -p["alphaOmuH2"] * Phi.bar * Phi

    L4Gauge = (
        -(p["alphaKB"] / FOUR) * FS(g["U1Y"], mu, nu) * FS(g["U1Y"], mu, nu)
        - (p["alphaKW"] / FOUR)
        * FS(g["SU2L"], mu, nu, aW)
        * FS(g["SU2L"], mu, nu, aW)
        - (p["alphaKG"] / FOUR)
        * FS(g["SU3C"], mu, nu, aC)
        * FS(g["SU3C"], mu, nu, aC)
    )

    L4Fermions = (
        I
        * p["alphaKq"](f1, f2)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(QL, mu)
        + I
        * p["alphaKl"](f1, f2)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * DC(LL, mu)
        + I * p["alphaKu"](f1, f2) * ur(f=f1, c=c1, bar=True) * Gamma(mu) * DC(UR, mu)
        + I * p["alphaKd"](f1, f2) * dr(f=f1, c=c1, bar=True) * Gamma(mu) * DC(DR, mu)
        + I * p["alphaKe"](f1, f2) * lr(f=f1, bar=True) * Gamma(mu) * DC(LR, mu)
    )

    L4Higgs = (
        p["alphaKH"] * DC(Phi.bar, mu) * DC(Phi, mu)
        - p["alphaOlambda"] * Phi.bar * Phi * Phi.bar * Phi
    )

    L4Yukawa = (
        -p["alphaOlambdad"](f1, f2)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * dr(f=f2, c=c1)
        * Phi(w1)
        - p["alphaOlambdae"](f1, f2)
        * ll(w=w1, f=f1, bar=True)
        * lr(f=f2)
        * Phi(w1)
        - p["alphaOlambdau"](f1, f2)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * ur(f=f2, c=c1)
        * phitilde(w1, w2)
        - p["alphaOlambdad"](f1, f2).conj()
        * Phi.bar(w1)
        * dr(f=f2, c=c1, bar=True)
        * ql(w=w1, f=f1, c=c1)
        - p["alphaOlambdae"](f1, f2).conj()
        * Phi.bar(w1)
        * lr(f=f2, bar=True)
        * ll(w=w1, f=f1)
        - p["alphaOlambdau"](f1, f2).conj()
        * phitildebar(w1, w2)
        * ur(f=f2, c=c1, bar=True)
        * ql(w=w1, f=f1, c=c1)
    )

    LWeinberg = (
        p["alphaWeinberg"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ll(sp=sp2, w=w2, f=f2)
        * weak_eps2(w3, w1)
        * weak_eps2(w4, w2)
        * Phi(w4)
        * Phi(w3)
        + p["alphaWeinberg"](f1, f2).conj()
        * Phi.bar(w3)
        * Phi.bar(w4)
        * ll(sp=sp2, w=w2, f=f2, bar=True)
        * dirac_charge_conjugation(sp2, sp1)
        * ll(sp=sp1, w=w1, f=f1)
        * weak_eps2(w3, w1)
        * weak_eps2(w4, w2)
    )

    LX3 = (
        p["alphaO3G"]
        * structure_constant(aC1, aC2, aC3)
        * FS(g["SU3C"], mu, nu, aC1)
        * FS(g["SU3C"], nu, rho, aC2)
        * FS(g["SU3C"], rho, mu, aC3)
        + p["alphaO3Gt"]
        * structure_constant(aC1, aC2, aC3)
        * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
        * FS(g["SU3C"], nu, rho2, aC2)
        * FS(g["SU3C"], rho2, mu, aC3)
        + p["alphaO3W"]
        * weak_structure_constant(aW1, aW2, aW3)
        * FS(g["SU2L"], mu, nu, aW1)
        * FS(g["SU2L"], nu, rho, aW2)
        * FS(g["SU2L"], rho, mu, aW3)
        + p["alphaO3Wt"]
        * weak_structure_constant(aW1, aW2, aW3)
        * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
        * FS(g["SU2L"], nu, rho2, aW2)
        * FS(g["SU2L"], rho2, mu, aW3)
    )

    LX2D2 = (
        -HALF
        * p["alphaR2G"]
        * DC(FS(g["SU3C"], mu, nu, aC1), mu)
        * DC(FS(g["SU3C"], rho, nu, aC1), rho)
        - HALF
        * p["alphaR2W"]
        * DC(FS(g["SU2L"], mu, nu, aW1), mu)
        * DC(FS(g["SU2L"], rho, nu, aW1), rho)
        - HALF
        * p["alphaR2B"]
        * PartialD(FS(g["U1Y"], mu, nu), mu)
        * PartialD(FS(g["U1Y"], rho, nu), rho)
    )

    LX2H2 = (
        p["alphaOHG"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["SU3C"], mu, nu, aC1)
        * FS(g["SU3C"], mu, nu, aC1)
        + p["alphaOHGt"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["SU3C"], mu, nu, aC1)
        * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
        + p["alphaOHW"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["SU2L"], mu, nu, aW1)
        * FS(g["SU2L"], mu, nu, aW1)
        + p["alphaOHWt"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["SU2L"], mu, nu, aW1)
        * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
        + p["alphaOHB"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["U1Y"], mu, nu)
        * FS(g["U1Y"], mu, nu)
        + p["alphaOHBt"]
        * Phi.bar(w1)
        * Phi(w1)
        * FS(g["U1Y"], mu, nu)
        * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
        + p["alphaOHWB"]
        * Phi.bar(w1)
        * Phi(w2)
        * weak_t(aW1, w1, w2)
        * FS(g["SU2L"], mu, nu, aW1)
        * FS(g["U1Y"], mu, nu)
        + p["alphaOHWBt"]
        * Phi.bar(w1)
        * Phi(w2)
        * weak_t(aW1, w1, w2)
        * FS(g["SU2L"], mu, nu, aW1)
        * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
    )

    LH2XD2 = (
        I
        * p["alphaRWDH"]
        * Phi.bar(w1)
        * DC(Phi(w2), mu)
        * weak_t(aW1, w1, w2)
        * DC(FS(g["SU2L"], mu, nu, aW1), nu)
        - I
        * p["alphaRWDH"]
        * DC(Phi.bar(w1), mu)
        * Phi(w2)
        * weak_t(aW1, w1, w2)
        * DC(FS(g["SU2L"], mu, nu, aW1), nu)
        + I
        * p["alphaRBDH"]
        * Phi.bar(w1)
        * DC(Phi(w1), mu)
        * PartialD(FS(g["U1Y"], mu, nu), nu)
        - I
        * p["alphaRBDH"]
        * DC(Phi.bar(w1), mu)
        * Phi(w1)
        * PartialD(FS(g["U1Y"], mu, nu), nu)
    )

    LH2D4 = (
        p["alphaRDH"]
        * DC(DC(Phi.bar, mu), mu)
        * DC(DC(Phi, nu), nu)
    )

    LH4D2 = (
        p["alphaOHBox"]
        * Phi.bar(w1)
        * Phi(w1)
        * PartialD(PartialD(Phi.bar(w2), mu), mu)
        * Phi(w2)
        + TWO
        * p["alphaOHBox"]
        * Phi.bar(w1)
        * Phi(w1)
        * PartialD(Phi.bar(w2), mu)
        * PartialD(Phi(w2), mu)
        + p["alphaOHBox"]
        * Phi.bar(w1)
        * Phi(w1)
        * Phi.bar(w2)
        * PartialD(PartialD(Phi(w2), mu), mu)
        + p["alphaOHD"]
        * DC(Phi.bar, mu)
        * Phi(w1)
        * Phi.bar(w2)
        * DC(Phi, mu)
        + p["alphaRHDp"] * Phi.bar(w1) * Phi(w1) * DC(Phi.bar, mu) * DC(Phi, mu)
        + I
        * p["alphaRHDpp"]
        * Phi.bar(w1)
        * Phi(w1)
        * PartialD(Phi.bar(w2), mu)
        * DC(Phi, mu)
        + I
        * p["alphaRHDpp"]
        * Phi.bar(w1)
        * Phi(w1)
        * Phi.bar(w2)
        * PartialD(DC(Phi, mu), mu)
        - I
        * p["alphaRHDpp"]
        * Phi.bar(w1)
        * Phi(w1)
        * PartialD(DC(Phi.bar, mu), mu)
        * Phi(w2)
        - I
        * p["alphaRHDpp"]
        * Phi.bar(w1)
        * Phi(w1)
        * DC(Phi.bar, mu)
        * PartialD(Phi(w2), mu)
    )

    LH6 = (
        p["alphaOH"]
        * Phi.bar(w1)
        * Phi(w1)
        * Phi.bar(w2)
        * Phi(w2)
        * Phi.bar(w3)
        * Phi(w3)
    )

    LF2XH = (
        sigma_term(
            p["alphaOuG"](f1, f2)
            * FS(g["SU3C"], mu, nu, aC1)
            * gauge_generator(aC1, c1, c2)
            * phitilde(w1, w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOuW"](f1, f2)
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * phitilde(w2, w3),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOuB"](f1, f2)
            * FS(g["U1Y"], mu, nu)
            * phitilde(w1, w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdG"](f1, f2)
            * FS(g["SU3C"], mu, nu, aC1)
            * gauge_generator(aC1, c1, c2)
            * Phi(w1),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdW"](f1, f2)
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * Phi(w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdB"](f1, f2) * FS(g["U1Y"], mu, nu) * Phi(w1),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOeW"](f1, f2)
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * Phi(w2),
            ll(w=w1, f=f1, bar=True),
            lr(f=f2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOeB"](f1, f2) * FS(g["U1Y"], mu, nu) * Phi(w1),
            ll(w=w1, f=f1, bar=True),
            lr(f=f2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOuG"](f1, f2).conj()
            * FS(g["SU3C"], mu, nu, aC1)
            * gauge_generator(aC1, c1, c2)
            * phitildebar(w1, w2),
            ur(f=f2, c=c2, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOuW"](f1, f2).conj()
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * phitildebar(w2, w3),
            ur(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOuB"](f1, f2).conj()
            * FS(g["U1Y"], mu, nu)
            * phitildebar(w1, w2),
            ur(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdG"](f1, f2).conj()
            * FS(g["SU3C"], mu, nu, aC1)
            * gauge_generator(aC1, c1, c2)
            * Phi.bar(w1),
            dr(f=f2, c=c2, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdW"](f1, f2).conj()
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * Phi.bar(w2),
            dr(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOdB"](f1, f2).conj() * FS(g["U1Y"], mu, nu) * Phi.bar(w1),
            dr(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOeW"](f1, f2).conj()
            * FS(g["SU2L"], mu, nu, aW1)
            * weak_t(aW1, w1, w2)
            * Phi.bar(w2),
            lr(f=f2, bar=True),
            ll(w=w1, f=f1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaOeB"](f1, f2).conj() * FS(g["U1Y"], mu, nu) * Phi.bar(w1),
            lr(f=f2, bar=True),
            ll(w=w1, f=f1),
            mu,
            nu,
        )
    )

    LF2D3 = (
        I
        * HALF
        * p["alphaRqD"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ql(sp=sp2, w=w1, f=f2, c=c1), nu), nu), mu)
        + I
        * HALF
        * p["alphaRqD"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ql(sp=sp2, w=w1, f=f2, c=c1), mu), nu), nu)
        + I
        * HALF
        * p["alphaRuD"](f1, f2)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ur(sp=sp2, f=f2, c=c1), nu), nu), mu)
        + I
        * HALF
        * p["alphaRuD"](f1, f2)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ur(sp=sp2, f=f2, c=c1), mu), nu), nu)
        + I
        * HALF
        * p["alphaRdD"](f1, f2)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(dr(sp=sp2, f=f2, c=c1), nu), nu), mu)
        + I
        * HALF
        * p["alphaRdD"](f1, f2)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(dr(sp=sp2, f=f2, c=c1), mu), nu), nu)
        + I
        * HALF
        * p["alphaRlD"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ll(sp=sp2, w=w1, f=f2), nu), nu), mu)
        + I
        * HALF
        * p["alphaRlD"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(ll(sp=sp2, w=w1, f=f2), mu), nu), nu)
        + I
        * HALF
        * p["alphaReD"](f1, f2)
        * lr(sp=sp1, f=f1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(lr(sp=sp2, f=f2), nu), nu), mu)
        + I
        * HALF
        * p["alphaReD"](f1, f2)
        * lr(sp=sp1, f=f1, bar=True)
        * Gamma(mu)
        * DC(DC(DC(lr(sp=sp2, f=f2), mu), nu), nu)
    )

    LF2HD2 = (
        p["alphaRuHD1"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp1, f=f2, c=c1)
        * weak_eps2(w1, w2)
        * DC(DC(Phi.bar(w2), mu), mu)
        + sigma_term(
            p["alphaRuHD2"](f1, f2)
            * weak_eps2(w1, w2)
            * DC(Phi.bar(w2), nu),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            DC(ur(sp=sp2, f=f2, c=c1), mu),
            mu,
            nu,
        )
        + p["alphaRuHD3"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * DC(DC(ur(sp=sp1, f=f2, c=c1), mu), mu)
        * weak_eps2(w1, w2)
        * Phi.bar(w2)
        + p["alphaRuHD4"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * DC(ur(sp=sp1, f=f2, c=c1), mu)
        * weak_eps2(w1, w2)
        * DC(Phi.bar(w2), mu)
        + p["alphaRdHD1"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dr(sp=sp1, f=f2, c=c1)
        * DC(DC(Phi(w1), mu), mu)
        + sigma_term(
            p["alphaRdHD2"](f1, f2) * DC(Phi(w1), nu),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            DC(dr(sp=sp2, f=f2, c=c1), mu),
            mu,
            nu,
        )
        + p["alphaRdHD3"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * DC(DC(dr(sp=sp1, f=f2, c=c1), mu), mu)
        * Phi(w1)
        + p["alphaRdHD4"](f1, f2)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * DC(dr(sp=sp1, f=f2, c=c1), mu)
        * DC(Phi(w1), mu)
        + p["alphaReHD1"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp1, f=f2)
        * DC(DC(Phi(w1), mu), mu)
        + sigma_term(
            p["alphaReHD2"](f1, f2) * DC(Phi(w1), nu),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            DC(lr(sp=sp2, f=f2), mu),
            mu,
            nu,
        )
        + p["alphaReHD3"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * DC(DC(lr(sp=sp1, f=f2), mu), mu)
        * Phi(w1)
        + p["alphaReHD4"](f1, f2)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * DC(lr(sp=sp1, f=f2), mu)
        * DC(Phi(w1), mu)
        + p["alphaRuHD1"](f1, f2).conj()
        * ur(sp=sp1, f=f2, c=c1, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * weak_eps2(w1, w2)
        * DC(DC(Phi(w2), mu), mu)
        + sigma_term(
            p["alphaRuHD2"](f1, f2).conj()
            * weak_eps2(w1, w2)
            * DC(Phi(w2), nu),
            DC(ur(sp=sp1, f=f2, c=c1, bar=True), mu),
            ql(sp=sp2, w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + p["alphaRuHD3"](f1, f2).conj()
        * DC(DC(ur(sp=sp1, f=f2, c=c1, bar=True), mu), mu)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * weak_eps2(w1, w2)
        * Phi(w2)
        + p["alphaRuHD4"](f1, f2).conj()
        * DC(ur(sp=sp1, f=f2, c=c1, bar=True), mu)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * weak_eps2(w1, w2)
        * DC(Phi(w2), mu)
        + p["alphaRdHD1"](f1, f2).conj()
        * dr(sp=sp1, f=f2, c=c1, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * DC(DC(Phi.bar(w1), mu), mu)
        + sigma_term(
            p["alphaRdHD2"](f1, f2).conj() * DC(Phi.bar(w1), nu),
            DC(dr(sp=sp1, f=f2, c=c1, bar=True), mu),
            ql(sp=sp2, w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + p["alphaRdHD3"](f1, f2).conj()
        * DC(DC(dr(sp=sp1, f=f2, c=c1, bar=True), mu), mu)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * Phi.bar(w1)
        + p["alphaRdHD4"](f1, f2).conj()
        * DC(dr(sp=sp1, f=f2, c=c1, bar=True), mu)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * DC(Phi.bar(w1), mu)
        + p["alphaReHD1"](f1, f2).conj()
        * lr(sp=sp1, f=f2, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * DC(DC(Phi.bar(w1), mu), mu)
        + sigma_term(
            p["alphaReHD2"](f1, f2).conj() * DC(Phi.bar(w1), nu),
            DC(lr(sp=sp1, f=f2, bar=True), mu),
            ll(sp=sp2, w=w1, f=f1),
            mu,
            nu,
        )
        + p["alphaReHD3"](f1, f2).conj()
        * DC(DC(lr(sp=sp1, f=f2, bar=True), mu), mu)
        * ll(sp=sp1, w=w1, f=f1)
        * Phi.bar(w1)
        + p["alphaReHD4"](f1, f2).conj()
        * DC(lr(sp=sp1, f=f2, bar=True), mu)
        * ll(sp=sp1, w=w1, f=f1)
        * DC(Phi.bar(w1), mu)
    )

    def f2xd_fs_current(coeff, tensor, left, right, fs_derivative):
        return coeff * tensor * left * Gamma(mu) * right * fs_derivative

    def f2xd_derivative_current(coeff, tensor, left, right, fs_factor):
        return (
            I
            * HALF
            * coeff
            * tensor
            * left
            * Gamma(mu)
            * DC(right, nu)
            * fs_factor
            - I
            * HALF
            * coeff
            * tensor
            * DC(left, nu)
            * Gamma(mu)
            * right
            * fs_factor
        )

    LF2XD = (
        f2xd_fs_current(
            p["alphaRGq"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c2),
            _covd_fs(g["SU3C"], mu, nu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGqp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGqtp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1),
        )
        + f2xd_fs_current(
            p["alphaRWq"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w2, f=f2, c=c1),
            _covd_fs(g["SU2L"], mu, nu, nu, aW1),
        )
        + f2xd_derivative_current(
            p["alphaRWqp"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w2, f=f2, c=c1),
            FS(g["SU2L"], mu, nu, aW1),
        )
        + f2xd_derivative_current(
            p["alphaRWqtp"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w2, f=f2, c=c1),
            _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1),
        )
        + f2xd_fs_current(
            p["alphaRBq"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c1),
            _covd_fs(g["U1Y"], mu, nu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBqp"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBqtp"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp2, w=w1, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho, sigma),
        )
        + f2xd_fs_current(
            p["alphaRGu"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c2),
            _covd_fs(g["SU3C"], mu, nu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGup"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGutp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1),
        )
        + f2xd_fs_current(
            p["alphaRBu"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c1),
            _covd_fs(g["U1Y"], mu, nu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBup"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRButp"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp2, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho, sigma),
        )
        + f2xd_fs_current(
            p["alphaRGd"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c2),
            _covd_fs(g["SU3C"], mu, nu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGdp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + f2xd_derivative_current(
            p["alphaRGdtp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1),
        )
        + f2xd_fs_current(
            p["alphaRBd"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c1),
            _covd_fs(g["U1Y"], mu, nu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBdp"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBdtp"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp2, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho, sigma),
        )
        + f2xd_fs_current(
            p["alphaRWl"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w2, f=f2),
            _covd_fs(g["SU2L"], mu, nu, nu, aW1),
        )
        + f2xd_derivative_current(
            p["alphaRWlp"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w2, f=f2),
            FS(g["SU2L"], mu, nu, aW1),
        )
        + f2xd_derivative_current(
            p["alphaRWltp"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w2, f=f2),
            _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1),
        )
        + f2xd_fs_current(
            p["alphaRBl"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w1, f=f2),
            _covd_fs(g["U1Y"], mu, nu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBlp"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w1, f=f2),
            FS(g["U1Y"], mu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBltp"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp2, w=w1, f=f2),
            _dual_fs(g["U1Y"], mu, nu, rho, sigma),
        )
        + f2xd_fs_current(
            p["alphaRBe"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp2, f=f2),
            _covd_fs(g["U1Y"], mu, nu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBep"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp2, f=f2),
            FS(g["U1Y"], mu, nu),
        )
        + f2xd_derivative_current(
            p["alphaRBetp"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp2, f=f2),
            _dual_fs(g["U1Y"], mu, nu, rho, sigma),
        )
    )

    LF2DH2 = (
        I
        * p["alphaOHq1"](f1, f2)
        * Phi.bar(w1)
        * DC(Phi, mu)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        - I
        * p["alphaOHq1"](f1, f2)
        * DC(Phi.bar, mu)
        * Phi(w1)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + I
        * p["alphaRHq1p"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(QL, mu)
        - I
        * p["alphaRHq1p"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * DC(QL.bar, mu)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + p["alphaRHq1pp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * Phi(w1)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + p["alphaRHq1pp"](f1, f2)
        * Phi.bar(w1)
        * PartialD(Phi(w1), mu)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + I
        * p["alphaOHq3"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * DC(Phi, mu)
        * ql(w=w3, f=f1, c=c1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        - I
        * p["alphaOHq3"](f1, f2)
        * DC(Phi.bar, mu)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ql(w=w3, f=f1, c=c1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + I
        * p["alphaRHq3p"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ql(w=w3, f=f1, c=c1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * DC(QL, mu)
        - I
        * p["alphaRHq3p"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * DC(QL.bar, mu)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + p["alphaRHq3pp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ql(w=w3, f=f1, c=c1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + p["alphaRHq3pp"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * DC(Phi, mu)
        * ql(w=w3, f=f1, c=c1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        + I
        * p["alphaOHu"](f1, f2)
        * Phi.bar(w1)
        * DC(Phi, mu)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        - I
        * p["alphaOHu"](f1, f2)
        * DC(Phi.bar, mu)
        * Phi(w1)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        + I
        * p["alphaRHup"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(UR, mu)
        - I
        * p["alphaRHup"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * DC(UR.bar, mu)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        + p["alphaRHupp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * Phi(w1)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        + p["alphaRHupp"](f1, f2)
        * Phi.bar(w1)
        * PartialD(Phi(w1), mu)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        + I
        * p["alphaOHd"](f1, f2)
        * Phi.bar(w1)
        * DC(Phi, mu)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        - I
        * p["alphaOHd"](f1, f2)
        * DC(Phi.bar, mu)
        * Phi(w1)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        + I
        * p["alphaRHdp"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * DC(DR, mu)
        - I
        * p["alphaRHdp"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * DC(DR.bar, mu)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        + p["alphaRHdpp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * Phi(w1)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        + p["alphaRHdpp"](f1, f2)
        * Phi.bar(w1)
        * PartialD(Phi(w1), mu)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        + I
        * p["alphaOHl1"](f1, f2)
        * Phi.bar(w1)
        * DC(Phi, mu)
        * ll(w=w2, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        - I
        * p["alphaOHl1"](f1, f2)
        * DC(Phi.bar, mu)
        * Phi(w1)
        * ll(w=w2, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + I
        * p["alphaRHl1p"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ll(w=w2, f=f1, bar=True)
        * Gamma(mu)
        * DC(LL, mu)
        - I
        * p["alphaRHl1p"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * DC(LL.bar, mu)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + p["alphaRHl1pp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * Phi(w1)
        * ll(w=w2, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + p["alphaRHl1pp"](f1, f2)
        * Phi.bar(w1)
        * PartialD(Phi(w1), mu)
        * ll(w=w2, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + I
        * p["alphaOHl3"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * DC(Phi, mu)
        * ll(w=w3, f=f1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        - I
        * p["alphaOHl3"](f1, f2)
        * DC(Phi.bar, mu)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ll(w=w3, f=f1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + I
        * p["alphaRHl3p"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ll(w=w3, f=f1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * DC(LL, mu)
        - I
        * p["alphaRHl3p"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * DC(LL.bar, mu)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + p["alphaRHl3pp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * weak_t(aW1, w1, w2)
        * Phi(w2)
        * ll(w=w3, f=f1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + p["alphaRHl3pp"](f1, f2)
        * Phi.bar(w1)
        * weak_t(aW1, w1, w2)
        * DC(Phi, mu)
        * ll(w=w3, f=f1, bar=True)
        * weak_t(aW1, w3, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        + I
        * p["alphaOHe"](f1, f2)
        * Phi.bar(w1)
        * DC(Phi, mu)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        - I
        * p["alphaOHe"](f1, f2)
        * DC(Phi.bar, mu)
        * Phi(w1)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        + I
        * p["alphaRHep"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * DC(LR, mu)
        - I
        * p["alphaRHep"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * DC(LR.bar, mu)
        * Gamma(mu)
        * lr(f=f2)
        + p["alphaRHepp"](f1, f2)
        * PartialD(Phi.bar(w1), mu)
        * Phi(w1)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        + p["alphaRHepp"](f1, f2)
        * Phi.bar(w1)
        * PartialD(Phi(w1), mu)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        + I
        * p["alphaOHud"](f1, f2)
        * phitildebar(w1, w2)
        * DC(Phi, mu)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        - I
        * p["alphaOHud"](f1, f2).conj()
        * DC(Phi.bar, mu)
        * phitilde(w1, w2)
        * dr(f=f2, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f1, c=c1)
    )

    LF2H3 = (
        p["alphaOeH"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ll(w=w2, f=f1, bar=True)
        * lr(f=f2)
        * Phi(w2)
        + p["alphaOuH"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * ur(f=f2, c=c1)
        * phitilde(w2, w3)
        + p["alphaOdH"](f1, f2)
        * Phi.bar(w1)
        * Phi(w1)
        * ql(w=w2, f=f1, c=c1, bar=True)
        * dr(f=f2, c=c1)
        * Phi(w2)
        + p["alphaOeH"](f1, f2).conj()
        * Phi.bar(w1)
        * Phi(w1)
        * Phi.bar(w2)
        * lr(f=f2, bar=True)
        * ll(w=w2, f=f1)
        + p["alphaOuH"](f1, f2).conj()
        * Phi.bar(w1)
        * Phi(w1)
        * phitildebar(w2, w3)
        * ur(f=f2, c=c1, bar=True)
        * ql(w=w2, f=f1, c=c1)
        + p["alphaOdH"](f1, f2).conj()
        * Phi.bar(w1)
        * Phi(w1)
        * Phi.bar(w2)
        * dr(f=f2, c=c1, bar=True)
        * ql(w=w2, f=f1, c=c1)
    )

    L4q = (
        p["alphaOqq1"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c1)
        * ql(w=w2, f=f3, c=c2, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f4, c=c2)
        + p["alphaOqq3"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * weak_t(aW1, w1, w2)
        * Gamma(mu)
        * ql(w=w2, f=f2, c=c1)
        * ql(w=w3, f=f3, c=c2, bar=True)
        * weak_t(aW1, w3, w1)
        * Gamma(mu)
        * ql(w=w1, f=f4, c=c2)
        + p["alphaOuu"](f1, f2, f3, f4)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        * ur(f=f3, c=c2, bar=True)
        * Gamma(mu)
        * ur(f=f4, c=c2)
        + p["alphaOdd"](f1, f2, f3, f4)
        * dr(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f2, c=c1)
        * dr(f=f3, c=c2, bar=True)
        * Gamma(mu)
        * dr(f=f4, c=c2)
        + p["alphaOud1"](f1, f2, f3, f4)
        * ur(f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f2, c=c1)
        * dr(f=f3, c=c2, bar=True)
        * Gamma(mu)
        * dr(f=f4, c=c2)
        + p["alphaOud8"](f1, f2, f3, f4)
        * ur(f=f1, c=c1, bar=True)
        * gauge_generator(aC1, c1, c2)
        * Gamma(mu)
        * ur(f=f2, c=c2)
        * dr(f=f3, c=c3, bar=True)
        * gauge_generator(aC1, c3, c4)
        * Gamma(mu)
        * dr(f=f4, c=c4)
        + p["alphaOqu1"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c1)
        * ur(f=f3, c=c2, bar=True)
        * Gamma(mu)
        * ur(f=f4, c=c2)
        + p["alphaOqu8"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * gauge_generator(aC1, c1, c2)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c2)
        * ur(f=f3, c=c3, bar=True)
        * gauge_generator(aC1, c3, c4)
        * Gamma(mu)
        * ur(f=f4, c=c4)
        + p["alphaOqd1"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c1)
        * dr(f=f3, c=c2, bar=True)
        * Gamma(mu)
        * dr(f=f4, c=c2)
        + p["alphaOqd8"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * gauge_generator(aC1, c1, c2)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c2)
        * dr(f=f3, c=c3, bar=True)
        * gauge_generator(aC1, c3, c4)
        * Gamma(mu)
        * dr(f=f4, c=c4)
        + p["alphaOquqd1"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp1, f=f2, c=c1)
        * ql(sp=sp2, w=w2, f=f3, c=c2, bar=True)
        * dr(sp=sp2, f=f4, c=c2)
        * weak_eps2(w1, w2)
        + p["alphaOquqd8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp1, f=f2, c=c2)
        * ql(sp=sp2, w=w2, f=f3, c=c3, bar=True)
        * dr(sp=sp2, f=f4, c=c4)
        * weak_eps2(w1, w2)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaOquqd1"](f1, f2, f3, f4).conj()
        * ur(sp=sp1, f=f2, c=c1, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * dr(sp=sp2, f=f4, c=c2, bar=True)
        * ql(sp=sp2, w=w2, f=f3, c=c2)
        * weak_eps2(w1, w2)
        + p["alphaOquqd8"](f1, f2, f3, f4).conj()
        * ur(sp=sp1, f=f2, c=c2, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * dr(sp=sp2, f=f4, c=c4, bar=True)
        * ql(sp=sp2, w=w2, f=f3, c=c3)
        * weak_eps2(w1, w2)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
    )

    L4l = (
        p["alphaOll"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w1, f=f2)
        * ll(w=w2, f=f3, bar=True)
        * Gamma(mu)
        * ll(w=w2, f=f4)
        + p["alphaOee"](f1, f2, f3, f4)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        * lr(f=f3, bar=True)
        * Gamma(mu)
        * lr(f=f4)
        + p["alphaOle"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w1, f=f2)
        * lr(f=f3, bar=True)
        * Gamma(mu)
        * lr(f=f4)
    )

    L4lq = (
        p["alphaOlq1"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w1, f=f2)
        * ql(w=w2, f=f3, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w2, f=f4, c=c1)
        + p["alphaOlq3"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * weak_t(aW1, w1, w2)
        * Gamma(mu)
        * ll(w=w2, f=f2)
        * ql(w=w3, f=f3, c=c1, bar=True)
        * weak_t(aW1, w3, w1)
        * Gamma(mu)
        * ql(w=w1, f=f4, c=c1)
        + p["alphaOeu"](f1, f2, f3, f4)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        * ur(f=f3, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f4, c=c1)
        + p["alphaOed"](f1, f2, f3, f4)
        * lr(f=f1, bar=True)
        * Gamma(mu)
        * lr(f=f2)
        * dr(f=f3, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f4, c=c1)
        + p["alphaOqe"](f1, f2, f3, f4)
        * ql(w=w1, f=f1, c=c1, bar=True)
        * Gamma(mu)
        * ql(w=w1, f=f2, c=c1)
        * lr(f=f3, bar=True)
        * Gamma(mu)
        * lr(f=f4)
        + p["alphaOlu"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w1, f=f2)
        * ur(f=f3, c=c1, bar=True)
        * Gamma(mu)
        * ur(f=f4, c=c1)
        + p["alphaOld"](f1, f2, f3, f4)
        * ll(w=w1, f=f1, bar=True)
        * Gamma(mu)
        * ll(w=w1, f=f2)
        * dr(f=f3, c=c1, bar=True)
        * Gamma(mu)
        * dr(f=f4, c=c1)
        + p["alphaOledq"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp1, f=f2)
        * dr(sp=sp2, f=f3, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c1)
        + p["alphaOlequ1"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp1, f=f2)
        * ql(sp=sp2, w=w2, f=f3, c=c1, bar=True)
        * ur(sp=sp2, f=f4, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaOledq"](f1, f2, f3, f4).conj()
        * lr(sp=sp1, f=f2, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * ql(sp=sp2, w=w1, f=f4, c=c1, bar=True)
        * dr(sp=sp2, f=f3, c=c1)
        + p["alphaOlequ1"](f1, f2, f3, f4).conj()
        * lr(sp=sp1, f=f2, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * ur(sp=sp2, f=f4, c=c1, bar=True)
        * ql(sp=sp2, w=w2, f=f3, c=c1)
        * weak_eps2(w1, w2)
    )

    LEvF2XH = (
        sigma_term(
            p["alphaEuG"](f1, f2)
            * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
            * gauge_generator(aC1, c1, c2)
            * phitilde(w1, w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEuW"](f1, f2)
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * phitilde(w2, w3),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEuB"](f1, f2)
            * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
            * phitilde(w1, w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            ur(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdG"](f1, f2)
            * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
            * gauge_generator(aC1, c1, c2)
            * Phi(w1),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdW"](f1, f2)
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * Phi(w2),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdB"](f1, f2) * _dual_fs(g["U1Y"], mu, nu, rho, sigma) * Phi(w1),
            ql(w=w1, f=f1, c=c1, bar=True),
            dr(f=f2, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEeW"](f1, f2)
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * Phi(w2),
            ll(w=w1, f=f1, bar=True),
            lr(f=f2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEeB"](f1, f2) * _dual_fs(g["U1Y"], mu, nu, rho, sigma) * Phi(w1),
            ll(w=w1, f=f1, bar=True),
            lr(f=f2),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEuG"](f1, f2).conj()
            * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
            * gauge_generator(aC1, c1, c2)
            * phitildebar(w1, w2),
            ur(f=f2, c=c2, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEuW"](f1, f2).conj()
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * phitildebar(w2, w3),
            ur(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEuB"](f1, f2).conj()
            * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
            * phitildebar(w1, w2),
            ur(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdG"](f1, f2).conj()
            * _dual_fs(g["SU3C"], mu, nu, rho, sigma, aC1)
            * gauge_generator(aC1, c1, c2)
            * Phi.bar(w1),
            dr(f=f2, c=c2, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdW"](f1, f2).conj()
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * Phi.bar(w2),
            dr(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEdB"](f1, f2).conj()
            * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
            * Phi.bar(w1),
            dr(f=f2, c=c1, bar=True),
            ql(w=w1, f=f1, c=c1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEeW"](f1, f2).conj()
            * _dual_fs(g["SU2L"], mu, nu, rho, sigma, aW1)
            * weak_t(aW1, w1, w2)
            * Phi.bar(w2),
            lr(f=f2, bar=True),
            ll(w=w1, f=f1),
            mu,
            nu,
        )
        + sigma_term(
            p["alphaEeB"](f1, f2).conj()
            * _dual_fs(g["U1Y"], mu, nu, rho, sigma)
            * Phi.bar(w1),
            lr(f=f2, bar=True),
            ll(w=w1, f=f1),
            mu,
            nu,
        )
    )

    # Keep these first covariant derivatives explicit for now: the compact
    # DC(...) form changes the local sigma-chain fermion-pair inference.
    dphitildebar_sigma_terms = (
        weak_eps2(w1, w2) * PartialD(Phi.bar(w2), sigma),
        I
        * HALF
        * p["g1"]
        * weak_eps2(w1, w2)
        * B(sigma)
        * Phi.bar(w2),
        I
        * p["g2"]
        * weak_eps2(w1, w2)
        * Wi(sigma, aW1)
        * Phi.bar(w3)
        * weak_t(aW1, w3, w2),
    )
    dphitilde_sigma_terms = (
        weak_eps2(w1, w2) * PartialD(Phi(w2), sigma),
        -I * HALF * p["g1"] * weak_eps2(w1, w2) * B(sigma) * Phi(w2),
        -I
        * p["g2"]
        * weak_eps2(w1, w2)
        * Wi(sigma, aW1)
        * weak_t(aW1, w2, w3)
        * Phi(w3),
    )
    dur_rho_terms = (
        PartialD(ur(sp=sp2, f=f2, c=c1), rho),
        -I * (TWO / THREE) * p["g1"] * B(rho) * ur(sp=sp2, f=f2, c=c1),
        -I
        * p["g3"]
        * G(rho, aC1)
        * gauge_generator(aC1, c2, c1)
        * ur(sp=sp2, f=f2, c=c2),
    )
    durbar_rho_terms = (
        PartialD(ur(sp=sp1, f=f2, c=c1, bar=True), rho),
        I * (TWO / THREE) * p["g1"] * B(rho) * ur(sp=sp1, f=f2, c=c1, bar=True),
        I
        * p["g3"]
        * G(rho, aC1)
        * ur(sp=sp1, f=f2, c=c2, bar=True)
        * gauge_generator(aC1, c1, c2),
    )
    dphi_sigma_terms = (
        PartialD(Phi(w1), sigma),
        -I * HALF * p["g1"] * B(sigma) * Phi(w1),
        -I * p["g2"] * Wi(sigma, aW1) * weak_t(aW1, w1, w2) * Phi(w2),
    )
    dphibar_sigma_terms = (
        PartialD(Phi.bar(w1), sigma),
        I * HALF * p["g1"] * B(sigma) * Phi.bar(w1),
        I * p["g2"] * Wi(sigma, aW1) * Phi.bar(w2) * weak_t(aW1, w2, w1),
    )
    ddr_rho_terms = (
        PartialD(dr(sp=sp2, f=f2, c=c1), rho),
        I * (ONE / THREE) * p["g1"] * B(rho) * dr(sp=sp2, f=f2, c=c1),
        -I
        * p["g3"]
        * G(rho, aC1)
        * gauge_generator(aC1, c2, c1)
        * dr(sp=sp2, f=f2, c=c2),
    )
    ddrbar_rho_terms = (
        PartialD(dr(sp=sp1, f=f2, c=c1, bar=True), rho),
        -I * (ONE / THREE) * p["g1"] * B(rho) * dr(sp=sp1, f=f2, c=c1, bar=True),
        I
        * p["g3"]
        * G(rho, aC1)
        * dr(sp=sp1, f=f2, c=c2, bar=True)
        * gauge_generator(aC1, c1, c2),
    )
    dlr_rho_terms = (
        PartialD(lr(sp=sp2, f=f2), rho),
        I * p["g1"] * B(rho) * lr(sp=sp2, f=f2),
    )
    dlrbar_rho_terms = (
        PartialD(lr(sp=sp1, f=f2, bar=True), rho),
        -I * p["g1"] * B(rho) * lr(sp=sp1, f=f2, bar=True),
    )

    sigma_chain = (I / TWO) * gamma2(sp1, sp2, mu, nu, sp5) - (I / TWO) * gamma2(
        sp1, sp2, nu, mu, sp5
    )

    LEvF2HD2 = ZERO
    for phitilde_piece in dphitildebar_sigma_terms:
        for ur_piece in dur_rho_terms:
            LEvF2HD2 += (
                p["alphaEuH"](f1, f2)
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phitilde_piece
                * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
                * sigma_chain
                * ur_piece
            )
    for phitilde_piece in dphitilde_sigma_terms:
        for urbar_piece in durbar_rho_terms:
            LEvF2HD2 += (
                p["alphaEuH"](f1, f2).conj()
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phitilde_piece
                * urbar_piece
                * sigma_chain
                * ql(sp=sp2, w=w1, f=f1, c=c1)
            )
    for phi_piece in dphi_sigma_terms:
        for dr_piece in ddr_rho_terms:
            LEvF2HD2 += (
                p["alphaEdH"](f1, f2)
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phi_piece
                * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
                * sigma_chain
                * dr_piece
            )
    for phibar_piece in dphibar_sigma_terms:
        for drbar_piece in ddrbar_rho_terms:
            LEvF2HD2 += (
                p["alphaEdH"](f1, f2).conj()
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phibar_piece
                * drbar_piece
                * sigma_chain
                * ql(sp=sp2, w=w1, f=f1, c=c1)
            )
    for phi_piece in dphi_sigma_terms:
        for lr_piece in dlr_rho_terms:
            LEvF2HD2 += (
                p["alphaEeH"](f1, f2)
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phi_piece
                * ll(sp=sp1, w=w1, f=f1, bar=True)
                * sigma_chain
                * lr_piece
            )
    for phibar_piece in dphibar_sigma_terms:
        for lrbar_piece in dlrbar_rho_terms:
            LEvF2HD2 += (
                p["alphaEeH"](f1, f2).conj()
                * lorentz_levi_civita(mu, nu, rho, sigma)
                * phibar_piece
                * lrbar_piece
                * sigma_chain
                * ll(sp=sp2, w=w1, f=f1)
            )

    sigma_gamma_left = sigma_matrix(sp1, sp2, mu, nu, sp5) * Gamma(sp2, sp3, rho)
    gamma_sigma_right = Gamma(sp1, sp2, rho) * sigma_matrix(
        sp2, sp3, mu, nu, sp6
    )
    mixed_sigma_gamma = sigma_gamma_left + gamma_sigma_right

    def ev_f2xd_fs_current(coeff, tensor, left, right, dual_fs_derivative):
        return coeff * tensor * left * mixed_sigma_gamma * right * dual_fs_derivative

    def ev_f2xd_derivative_current(coeff, tensor, left, right, fs_factor):
        return (
            I
            * coeff
            * tensor
            * left
            * sigma_gamma_left
            * DC(right, rho)
            * fs_factor
            - I
            * coeff
            * tensor
            * DC(left, rho)
            * gamma_sigma_right
            * right
            * fs_factor
        )

    LEvF2XD = (
        ev_f2xd_fs_current(
            p["alphaEGq"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c2),
            _dual_covd_fs(g["SU3C"], mu, nu, rho, rho2, sigma, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGqp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGqtp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho2, sigma, aC1),
        )
        + ev_f2xd_fs_current(
            p["alphaEWq"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w2, f=f2, c=c1),
            _dual_covd_fs(g["SU2L"], mu, nu, rho, rho2, sigma, aW1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEWqp"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w2, f=f2, c=c1),
            FS(g["SU2L"], mu, nu, aW1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEWqtp"](f1, f2),
            weak_t(aW1, w1, w2),
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w2, f=f2, c=c1),
            _dual_fs(g["SU2L"], mu, nu, rho2, sigma, aW1),
        )
        + ev_f2xd_fs_current(
            p["alphaEBq"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c1),
            _dual_covd_fs(g["U1Y"], mu, nu, rho, rho2, sigma),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBqp"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBqtp"](f1, f2),
            ONE,
            ql(sp=sp1, w=w1, f=f1, c=c1, bar=True),
            ql(sp=sp3, w=w1, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho2, sigma),
        )
        + ev_f2xd_fs_current(
            p["alphaEGu"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c2),
            _dual_covd_fs(g["SU3C"], mu, nu, rho, rho2, sigma, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGup"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGutp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho2, sigma, aC1),
        )
        + ev_f2xd_fs_current(
            p["alphaEBu"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c1),
            _dual_covd_fs(g["U1Y"], mu, nu, rho, rho2, sigma),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBup"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + ev_f2xd_derivative_current(
            p["alphaEButp"](f1, f2),
            ONE,
            ur(sp=sp1, f=f1, c=c1, bar=True),
            ur(sp=sp3, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho2, sigma),
        )
        + ev_f2xd_fs_current(
            p["alphaEGd"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c2),
            _dual_covd_fs(g["SU3C"], mu, nu, rho, rho2, sigma, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGdp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c2),
            FS(g["SU3C"], mu, nu, aC1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEGdtp"](f1, f2),
            gauge_generator(aC1, c1, c2),
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c2),
            _dual_fs(g["SU3C"], mu, nu, rho2, sigma, aC1),
        )
        + ev_f2xd_fs_current(
            p["alphaEBd"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c1),
            _dual_covd_fs(g["U1Y"], mu, nu, rho, rho2, sigma),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBdp"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c1),
            FS(g["U1Y"], mu, nu),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBdtp"](f1, f2),
            ONE,
            dr(sp=sp1, f=f1, c=c1, bar=True),
            dr(sp=sp3, f=f2, c=c1),
            _dual_fs(g["U1Y"], mu, nu, rho2, sigma),
        )
        + ev_f2xd_fs_current(
            p["alphaEWl"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w2, f=f2),
            _dual_covd_fs(g["SU2L"], mu, nu, rho, rho2, sigma, aW1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEWlp"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w2, f=f2),
            FS(g["SU2L"], mu, nu, aW1),
        )
        + ev_f2xd_derivative_current(
            p["alphaEWltp"](f1, f2),
            weak_t(aW1, w1, w2),
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w2, f=f2),
            _dual_fs(g["SU2L"], mu, nu, rho2, sigma, aW1),
        )
        + ev_f2xd_fs_current(
            p["alphaEBl"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w1, f=f2),
            _dual_covd_fs(g["U1Y"], mu, nu, rho, rho2, sigma),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBlp"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w1, f=f2),
            FS(g["U1Y"], mu, nu),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBltp"](f1, f2),
            ONE,
            ll(sp=sp1, w=w1, f=f1, bar=True),
            ll(sp=sp3, w=w1, f=f2),
            _dual_fs(g["U1Y"], mu, nu, rho2, sigma),
        )
        + ev_f2xd_fs_current(
            p["alphaEBe"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp3, f=f2),
            _dual_covd_fs(g["U1Y"], mu, nu, rho, rho2, sigma),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBep"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp3, f=f2),
            FS(g["U1Y"], mu, nu),
        )
        + ev_f2xd_derivative_current(
            p["alphaEBetp"](f1, f2),
            ONE,
            lr(sp=sp1, f=f1, bar=True),
            lr(sp=sp3, f=f2),
            _dual_fs(g["U1Y"], mu, nu, rho2, sigma),
        )
    )

    LEv4q = (
        p["alphaEqu"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp1, f=f2, c=c1)
        * ur(sp=sp2, f=f3, c=c2, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c2)
        + p["alphaEqu8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp1, f=f2, c=c2)
        * ur(sp=sp2, f=f3, c=c3, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c4)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqd"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dr(sp=sp1, f=f2, c=c1)
        * dr(sp=sp2, f=f3, c=c2, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c2)
        + p["alphaEqd8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dr(sp=sp1, f=f2, c=c2)
        * dr(sp=sp2, f=f3, c=c3, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c4)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqutwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaEqutwo8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqdtwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaEqdtwo8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEuu8"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c3, bar=True)
        * ur(sp=sp4, f=f4, c=c4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEuuthree"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * ur(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEuuthree8"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c3, bar=True)
        * ur(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEdd8"](f1, f2, f3, f4)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * dr(sp=sp4, f=f4, c=c4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEddthree"](f1, f2, f3, f4)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * dr(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEddthree8"](f1, f2, f3, f4)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * dr(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEud"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * ur(sp=sp4, f=f4, c=c2)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        + p["alphaEud8"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * ur(sp=sp4, f=f4, c=c4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEudthree"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * ur(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEudthree8"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * ur(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEudthreep"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * dr(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEudthree8p"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * dr(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEquthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * ur(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEquthree8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c3, bar=True)
        * ur(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqdthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * dr(sp=sp4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEqdthree8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c3, bar=True)
        * dr(sp=sp4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqq8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c2)
        * ql(sp=sp3, w=w2, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w2, f=f4, c=c4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqq38"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w2, f=f2, c=c2)
        * ql(sp=sp3, w=w3, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w4, f=f4, c=c4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaEqqthree1"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * ql(sp=sp3, w=w2, f=f3, c=c2, bar=True)
        * ql(sp=sp4, w=w2, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEqqthree3"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w2, f=f2, c=c1)
        * ql(sp=sp3, w=w3, f=f3, c=c2, bar=True)
        * ql(sp=sp4, w=w4, f=f4, c=c2)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaEqqthree8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c2)
        * ql(sp=sp3, w=w2, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w2, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEqqthree38"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w2, f=f2, c=c2)
        * ql(sp=sp3, w=w3, f=f3, c=c3, bar=True)
        * ql(sp=sp4, w=w4, f=f4, c=c4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaEquqdtwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ql(sp=sp3, w=w2, f=f3, c=c2, bar=True)
        * dr(sp=sp4, f=f4, c=c2)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEquqdtwo8"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ur(sp=sp2, f=f2, c=c2)
        * ql(sp=sp3, w=w2, f=f3, c=c3, bar=True)
        * dr(sp=sp4, f=f4, c=c4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * weak_eps2(w1, w2)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
        + p["alphaEquqdtwo"](f1, f2, f3, f4).conj()
        * ur(sp=sp2, f=f2, c=c1, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * dr(sp=sp4, f=f4, c=c2, bar=True)
        * ql(sp=sp3, w=w2, f=f3, c=c2)
        * gamma2(sp2, sp1, mu, nu, sp5)
        * gamma2(sp4, sp3, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEquqdtwo8"](f1, f2, f3, f4).conj()
        * ur(sp=sp2, f=f2, c=c2, bar=True)
        * ql(sp=sp1, w=w1, f=f1, c=c1)
        * dr(sp=sp4, f=f4, c=c4, bar=True)
        * ql(sp=sp3, w=w2, f=f3, c=c3)
        * gamma2(sp2, sp1, mu, nu, sp5)
        * gamma2(sp4, sp3, mu, nu, sp6)
        * weak_eps2(w1, w2)
        * gauge_generator(aC1, c1, c2)
        * gauge_generator(aC1, c3, c4)
    )

    LEv4l = (
        p["alphaEeethree"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEll3"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w2, f=f2)
        * ll(sp=sp3, w=w3, f=f3, bar=True)
        * ll(sp=sp4, w=w4, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaEllthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w1, f=f2)
        * ll(sp=sp3, w=w2, f=f3, bar=True)
        * ll(sp=sp4, w=w2, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEllthree3"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w2, f=f2)
        * ll(sp=sp3, w=w3, f=f3, bar=True)
        * ll(sp=sp4, w=w4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaEle"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp1, f=f2)
        * lr(sp=sp2, f=f3, bar=True)
        * ll(sp=sp2, w=w1, f=f4)
        + p["alphaEletwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * ll(sp=sp4, w=w1, f=f4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaElethree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w1, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
    )

    LEv4lq = (
        p["alphaEeu"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        + p["alphaEed"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        + p["alphaEeuthree"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEedthree"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEeuthreep"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * ur(sp=sp4, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEedthreep"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * dr(sp=sp4, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaElq"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w2, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        + p["alphaElq3"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w2, f=f2, c=c1)
        * ql(sp=sp3, w=w3, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w4, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaElqthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w2, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaElqthree3"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w2, f=f2, c=c1)
        * ql(sp=sp3, w=w3, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaElqthreep"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w1, f=f2)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaElqthree3p"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w2, f=f2)
        * ql(sp=sp3, w=w3, f=f3, c=c1, bar=True)
        * ql(sp=sp4, w=w4, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        * weak_t(aW1, w1, w2)
        * weak_t(aW1, w3, w4)
        + p["alphaElu"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ur(sp=sp1, f=f2, c=c1)
        * ur(sp=sp2, f=f3, c=c1, bar=True)
        * ll(sp=sp2, w=w1, f=f4)
        + p["alphaEld"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dr(sp=sp1, f=f2, c=c1)
        * dr(sp=sp2, f=f3, c=c1, bar=True)
        * ll(sp=sp2, w=w1, f=f4)
        + p["alphaEqe"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * lr(sp=sp1, f=f2)
        * lr(sp=sp2, f=f3, bar=True)
        * ql(sp=sp2, w=w1, f=f4, c=c1)
        + p["alphaElutwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w1, f=f4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaEldtwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * ll(sp=sp4, w=w1, f=f4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaEqetwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaEluthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w1, f=f2)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * ur(sp=sp4, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEldthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ll(sp=sp2, w=w1, f=f2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * dr(sp=sp4, f=f4, c=c1)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaEqethree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * lr(sp=sp3, f=f3, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaElequtwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * ur(sp=sp4, f=f4, c=c1)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEluqe"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ur(sp=sp1, f=f2, c=c1)
        * ql(sp=sp2, w=w2, f=f3, c=c1, bar=True)
        * lr(sp=sp2, f=f4)
        * weak_eps2(w1, w2)
        + p["alphaEluqetwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ur(sp=sp2, f=f2, c=c1)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEledqtwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * lr(sp=sp2, f=f2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        * gamma2(sp1, sp2, mu, nu, sp5)
        * gamma2(sp3, sp4, mu, nu, sp6)
        + p["alphaElqde"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * Gamma(sp1, sp2, mu)
        * Gamma(sp3, sp4, mu)
        + p["alphaElqdethree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * ql(sp=sp2, w=w1, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * lr(sp=sp4, f=f4)
        * gamma3(sp1, sp2, mu, nu, rho, sp5, sp6)
        * gamma3(sp3, sp4, mu, nu, rho, sp7, sp8)
        + p["alphaElequtwo"](f1, f2, f3, f4).conj()
        * lr(sp=sp2, f=f2, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * ur(sp=sp4, f=f4, c=c1, bar=True)
        * ql(sp=sp3, w=w2, f=f3, c=c1)
        * gamma2(sp2, sp1, mu, nu, sp5)
        * gamma2(sp4, sp3, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEluqe"](f1, f2, f3, f4).conj()
        * ur(sp=sp1, f=f2, c=c1, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * lr(sp=sp2, f=f4, bar=True)
        * ql(sp=sp2, w=w2, f=f3, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEluqetwo"](f1, f2, f3, f4).conj()
        * ur(sp=sp2, f=f2, c=c1, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * lr(sp=sp4, f=f4, bar=True)
        * ql(sp=sp3, w=w2, f=f3, c=c1)
        * gamma2(sp2, sp1, mu, nu, sp5)
        * gamma2(sp4, sp3, mu, nu, sp6)
        * weak_eps2(w1, w2)
        + p["alphaEledqtwo"](f1, f2, f3, f4).conj()
        * lr(sp=sp2, f=f2, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * ql(sp=sp4, w=w1, f=f4, c=c1, bar=True)
        * dr(sp=sp3, f=f3, c=c1)
        * gamma2(sp2, sp1, mu, nu, sp5)
        * gamma2(sp4, sp3, mu, nu, sp6)
        + p["alphaElqde"](f1, f2, f3, f4).conj()
        * ql(sp=sp2, w=w1, f=f2, c=c1, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * lr(sp=sp4, f=f4, bar=True)
        * dr(sp=sp3, f=f3, c=c1)
        * Gamma(sp2, sp1, mu)
        * Gamma(sp4, sp3, mu)
        + p["alphaElqdethree"](f1, f2, f3, f4).conj()
        * ql(sp=sp2, w=w1, f=f2, c=c1, bar=True)
        * ll(sp=sp1, w=w1, f=f1)
        * lr(sp=sp4, f=f4, bar=True)
        * dr(sp=sp3, f=f3, c=c1)
        * gamma3(sp2, sp1, mu, nu, rho, sp5, sp6)
        * gamma3(sp4, sp3, mu, nu, rho, sp7, sp8)
    )

    LEvCCLLLL = (
        p["alphaEcll"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ll(sp=sp2, w=w2, f=f2)
        * ll(sp=sp3, w=w2, f=f3, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEclltwo"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ll(sp=sp4, w=w2, f=f2)
        * ll(sp=sp5, w=w2, f=f3, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ll(sp=sp8, w=w1, f=f4)
        + p["alphaEcqq"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ql(sp=sp2, w=w2, f=f2, c=c2)
        * ql(sp=sp3, w=w2, f=f3, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqqtwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ql(sp=sp4, w=w2, f=f2, c=c2)
        * ql(sp=sp5, w=w2, f=f3, c=c2, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ql(sp=sp8, w=w1, f=f4, c=c1)
        + p["alphaEcqqp"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ql(sp=sp2, w=w2, f=f2, c=c2)
        * ql(sp=sp3, w=w2, f=f3, c=c1, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        + p["alphaEcqqptwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ql(sp=sp4, w=w2, f=f2, c=c2)
        * ql(sp=sp5, w=w2, f=f3, c=c1, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ql(sp=sp8, w=w1, f=f4, c=c2)
        + p["alphaEcql"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ll(sp=sp2, w=w2, f=f2)
        * ll(sp=sp3, w=w2, f=f3, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqltwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ll(sp=sp4, w=w2, f=f2)
        * ll(sp=sp5, w=w2, f=f3, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ql(sp=sp8, w=w1, f=f4, c=c1)
        + p["alphaEcqlp"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ll(sp=sp2, w=w2, f=f2)
        * ll(sp=sp3, w=w1, f=f3, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        + p["alphaEcqlptwo"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ll(sp=sp4, w=w2, f=f2)
        * ll(sp=sp5, w=w1, f=f3, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ql(sp=sp8, w=w2, f=f4, c=c1)
    )

    LEvCCRRRR = (
        p["alphaEcee"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * lr(sp=sp4, f=f4)
        + p["alphaEceetwo"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * lr(sp=sp4, f=f2)
        * lr(sp=sp5, f=f3, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * lr(sp=sp8, f=f4)
        + p["alphaEceu"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * lr(sp=sp4, f=f4)
        + p["alphaEceutwo"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ur(sp=sp4, f=f2, c=c1)
        * ur(sp=sp5, f=f3, c=c1, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * lr(sp=sp8, f=f4)
        + p["alphaEced"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * lr(sp=sp4, f=f4)
        + p["alphaEcedtwo"](f1, f2, f3, f4)
        * lr(sp=sp1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * dr(sp=sp4, f=f2, c=c1)
        * dr(sp=sp5, f=f3, c=c1, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * lr(sp=sp8, f=f4)
        + p["alphaEcuu"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ur(sp=sp4, f=f4, c=c1)
        + p["alphaEcuutwo"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * ur(sp=sp4, f=f2, c=c2)
        * ur(sp=sp5, f=f3, c=c2, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ur(sp=sp8, f=f4, c=c1)
        + p["alphaEcdd"](f1, f2, f3, f4)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * dr(sp=sp4, f=f4, c=c1)
        + p["alphaEcddtwo"](f1, f2, f3, f4)
        * dr(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * dr(sp=sp4, f=f2, c=c2)
        * dr(sp=sp5, f=f3, c=c2, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * dr(sp=sp8, f=f4, c=c1)
        + p["alphaEcud"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ur(sp=sp4, f=f4, c=c1)
        + p["alphaEcudtwo"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * dr(sp=sp4, f=f2, c=c2)
        * dr(sp=sp5, f=f3, c=c2, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ur(sp=sp8, f=f4, c=c1)
        + p["alphaEcudp"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ur(sp=sp4, f=f4, c=c2)
        + p["alphaEcudptwo"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * gamma2(sp2, sp4, mu, nu, sp3)
        * dr(sp=sp4, f=f2, c=c2)
        * dr(sp=sp5, f=f3, c=c1, bar=True)
        * gamma2(sp5, sp7, mu, nu, sp6)
        * dirac_charge_conjugation(sp7, sp8)
        * ur(sp=sp8, f=f4, c=c2)
    )

    LEvCCLRRL = (
        p["alphaEcle"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcqe"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEclu"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcld"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcqu"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqd"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqup"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        + p["alphaEcqdp"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        + p["alphaEcqedl"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * lr(sp=sp2, f=f2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcqedl"](f1, f2, f3, f4).conj()
        * ll(sp=sp1, w=w1, f=f4, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * Gamma(sp5, sp2, mu)
        * dr(sp=sp2, f=f3, c=c1)
        * lr(sp=sp3, f=f2, bar=True)
        * Gamma(sp3, sp6, mu)
        * dirac_charge_conjugation(sp6, sp4)
        * ql(sp=sp4, w=w1, f=f1, c=c1)
        + p["alphaEclethree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcqethree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * lr(sp=sp2, f=f2)
        * lr(sp=sp3, f=f3, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcluthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * ur(sp=sp2, f=f2, c=c1)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcldthree"](f1, f2, f3, f4)
        * ll(sp=sp1, w=w1, f=f1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * dr(sp=sp2, f=f2, c=c1)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcquthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c2, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqdthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c2, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c1)
        + p["alphaEcqupthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * ur(sp=sp2, f=f2, c=c2)
        * ur(sp=sp3, f=f3, c=c1, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        + p["alphaEcqdpthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * dr(sp=sp2, f=f2, c=c2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f4, c=c2)
        + p["alphaEcqedlthree"](f1, f2, f3, f4)
        * ql(sp=sp1, w=w1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * lr(sp=sp2, f=f2)
        * dr(sp=sp3, f=f3, c=c1, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ll(sp=sp4, w=w1, f=f4)
        + p["alphaEcqedlthree"](f1, f2, f3, f4).conj()
        * ll(sp=sp1, w=w1, f=f4, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma3(sp5, sp2, mu, nu, rho, sp6, sp7)
        * dr(sp=sp2, f=f3, c=c1)
        * lr(sp=sp3, f=f2, bar=True)
        * gamma3(sp3, sp8, mu, nu, rho, sp9, sp10)
        * dirac_charge_conjugation(sp8, sp4)
        * ql(sp=sp4, w=w1, f=f1, c=c1)
    )

    LEvCCRRLL = (
        p["alphaEcuelq"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * lr(sp=sp2, f=f2)
        * ll(sp=sp3, w=w1, f=f3, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcudqq"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * dr(sp=sp2, f=f2, c=c2)
        * ql(sp=sp3, w=w1, f=f3, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcuelqtwo"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma2(sp5, sp2, mu, nu, sp6)
        * lr(sp=sp2, f=f2)
        * ll(sp=sp3, w=w1, f=f3, bar=True)
        * gamma2(sp3, sp7, mu, nu, sp8)
        * dirac_charge_conjugation(sp7, sp4)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcudqqtwo"](f1, f2, f3, f4)
        * ur(sp=sp1, f=f1, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp5)
        * gamma2(sp5, sp2, mu, nu, sp6)
        * dr(sp=sp2, f=f2, c=c2)
        * ql(sp=sp3, w=w1, f=f3, c=c2, bar=True)
        * gamma2(sp3, sp7, mu, nu, sp8)
        * dirac_charge_conjugation(sp7, sp4)
        * ql(sp=sp4, w=w2, f=f4, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcuelq"](f1, f2, f3, f4).conj()
        * ql(sp=sp1, w=w1, f=f4, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ll(sp=sp2, w=w2, f=f3)
        * lr(sp=sp3, f=f2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ur(sp=sp4, f=f1, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcudqq"](f1, f2, f3, f4).conj()
        * ql(sp=sp1, w=w1, f=f4, c=c1, bar=True)
        * dirac_charge_conjugation(sp1, sp2)
        * ql(sp=sp2, w=w2, f=f3, c=c2)
        * dr(sp=sp3, f=f2, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp4)
        * ur(sp=sp4, f=f1, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcuelqtwo"](f1, f2, f3, f4).conj()
        * ql(sp=sp1, w=w1, f=f4, c=c1, bar=True)
        * gamma2(sp1, sp5, mu, nu, sp6)
        * dirac_charge_conjugation(sp5, sp2)
        * ll(sp=sp2, w=w2, f=f3)
        * lr(sp=sp3, f=f2, bar=True)
        * dirac_charge_conjugation(sp3, sp7)
        * gamma2(sp7, sp4, mu, nu, sp8)
        * ur(sp=sp4, f=f1, c=c1)
        * weak_eps2(w1, w2)
        + p["alphaEcudqqtwo"](f1, f2, f3, f4).conj()
        * ql(sp=sp1, w=w1, f=f4, c=c1, bar=True)
        * gamma2(sp1, sp5, mu, nu, sp6)
        * dirac_charge_conjugation(sp5, sp2)
        * ql(sp=sp2, w=w2, f=f3, c=c2)
        * dr(sp=sp3, f=f2, c=c2, bar=True)
        * dirac_charge_conjugation(sp3, sp7)
        * gamma2(sp7, sp4, mu, nu, sp8)
        * ur(sp=sp4, f=f1, c=c1)
        * weak_eps2(w1, w2)
    )

    LSM = LGauge + LFermions + LHiggs + LYukawa
    Ltot = _sum_supported_lagrangian_blocks(
        L2Higgs,
        L4Gauge,
        L4Fermions,
        L4Higgs,
        L4Yukawa,
        LWeinberg,
        LX3,
        LX2D2,
        LX2H2,
        LH2XD2,
        LH2D4,
        LH4D2,
        LH6,
        LF2D3,
        LF2HD2,
        LF2XH,
        LF2XD,
        LF2DH2,
        LF2H3,
        L4q,
        L4l,
        L4lq,
        LEvF2XH,
        LEvF2HD2,
        LEvF2XD,
        LEv4q,
        LEv4l,
        LEv4lq,
        LEvCCLLLL,
        LEvCCRRRR,
        LEvCCLRRL,
        LEvCCRRLL,
    )
    Lfull = LSM + Ltot

    lagrangians = {
        "LGauge": DeclaredLagrangian.from_item(LGauge),
        "LFermions": DeclaredLagrangian.from_item(LFermions),
        "LHiggs": DeclaredLagrangian.from_item(LHiggs),
        "LYukawa": DeclaredLagrangian.from_item(LYukawa),
        "LSM": DeclaredLagrangian.from_item(LSM),
        "L2Higgs": DeclaredLagrangian.from_item(L2Higgs),
        "L4Gauge": DeclaredLagrangian.from_item(L4Gauge),
        "L4Fermions": DeclaredLagrangian.from_item(L4Fermions),
        "L4Higgs": DeclaredLagrangian.from_item(L4Higgs),
        "L4Yukawa": DeclaredLagrangian.from_item(L4Yukawa),
        "LWeinberg": DeclaredLagrangian.from_item(LWeinberg),
        "LX3": DeclaredLagrangian.from_item(LX3),
        "LX2D2": DeclaredLagrangian.from_item(LX2D2),
        "LX2H2": DeclaredLagrangian.from_item(LX2H2),
        "LH2XD2": DeclaredLagrangian.from_item(LH2XD2),
        "LH2D4": DeclaredLagrangian.from_item(LH2D4),
        "LH4D2": DeclaredLagrangian.from_item(LH4D2),
        "LH6": DeclaredLagrangian.from_item(LH6),
        "LF2D3": DeclaredLagrangian.from_item(LF2D3),
        "LF2HD2": DeclaredLagrangian.from_item(LF2HD2),
        "LF2XH": DeclaredLagrangian.from_item(LF2XH),
        "LF2XD": DeclaredLagrangian.from_item(LF2XD),
        "LF2DH2": DeclaredLagrangian.from_item(LF2DH2),
        "LF2H3": DeclaredLagrangian.from_item(LF2H3),
        "L4q": DeclaredLagrangian.from_item(L4q),
        "L4l": DeclaredLagrangian.from_item(L4l),
        "L4lq": DeclaredLagrangian.from_item(L4lq),
        "LEvF2XH": DeclaredLagrangian.from_item(LEvF2XH),
        "LEvF2HD2": DeclaredLagrangian.from_item(LEvF2HD2),
        "LEvF2XD": DeclaredLagrangian.from_item(LEvF2XD),
        "LEv4q": DeclaredLagrangian.from_item(LEv4q),
        "LEv4l": DeclaredLagrangian.from_item(LEv4l),
        "LEv4lq": DeclaredLagrangian.from_item(LEv4lq),
        "LEvCCLLLL": DeclaredLagrangian.from_item(LEvCCLLLL),
        "LEvCCRRRR": DeclaredLagrangian.from_item(LEvCCRRRR),
        "LEvCCLRRL": DeclaredLagrangian.from_item(LEvCCLRRL),
        "LEvCCRRLL": DeclaredLagrangian.from_item(LEvCCRRLL),
        "Ltot": DeclaredLagrangian.from_item(Ltot),
        "Lfull": DeclaredLagrangian.from_item(Lfull),
    }

    model = Model(
        name=name,
        gauge_groups=tuple(gauge_groups.values()),
        fields=tuple(fields.values()),
        parameters=tuple(parameters.values()),
        lagrangian_decl=lagrangians["Ltot"],
    )

    return SMEFT2Bundle(
        model=model,
        fields=fields,
        parameters=parameters,
        gauge_groups=gauge_groups,
        lagrangians=lagrangians,
    )


__all__ = ("OMITTED_SECTORS", "SMEFT2Bundle", "build_smeft_green_bpreserving")
