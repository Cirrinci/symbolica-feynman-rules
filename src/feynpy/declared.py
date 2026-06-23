"""Declarative mini-language factors and builders."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import Callable

from .metadata import (
    COLOR_FUND_KIND,
    SPINOR_INDEX,
    ConjugateField,
    Field,
    IndexType,
    Parameter,
    gamma_matrix,
    gauge_generator,
)
from symbolic.spenso_structures import (
    chiral_projector_left,
    chiral_projector_right,
)
# ---------------------------------------------------------------------------
# Declarative Lagrangian factors  (CovD / Gamma / FieldStrength DSL)
# ---------------------------------------------------------------------------


def _declared_source_terms_from_item(item):
    from .lowering import _declared_source_terms_from_item as impl

    return impl(item)


class _DeclaredFactorMixin:
    def __mul__(self, other):
        return _DeclaredMonomial.from_factor(self).__mul__(other)

    def __rmul__(self, other):
        return _DeclaredMonomial.from_factor(self).__rmul__(other)

    def __add__(self, other):
        return _DeclaredMonomial.from_factor(self).__add__(other)

    def __radd__(self, other):
        return _DeclaredMonomial.from_factor(self).__radd__(other)

    def __sub__(self, other):
        return _DeclaredMonomial.from_factor(self).__sub__(other)

    def __rsub__(self, other):
        return _DeclaredMonomial.from_factor(self).__rsub__(other)


@dataclass(frozen=True)
class _FieldFactor(_DeclaredFactorMixin):
    field: Field
    conjugated: bool = False
    labels: dict = dataclass_field(default_factory=dict)

    def __str__(self):
        if self.conjugated and not self.field.self_conjugate:
            return f"{self.field.name}.bar"
        return self.field.name


@dataclass(frozen=True)
class CovariantDerivativeFactor(_DeclaredFactorMixin):
    field: Field
    lorentz_index: object
    conjugated: bool = False

    @property
    def bar(self) -> "CovariantDerivativeFactor":
        if self.field.self_conjugate:
            return self
        return CovariantDerivativeFactor(
            field=self.field,
            lorentz_index=self.lorentz_index,
            conjugated=not self.conjugated,
        )

    def __str__(self):
        base = f"{self.field.name}.bar" if self.conjugated and not self.field.self_conjugate else self.field.name
        return f"CovD({base}, {self.lorentz_index})"


@dataclass(frozen=True)
class DifferentiatedCovariantFactor(_DeclaredFactorMixin):
    covariant_factor: CovariantDerivativeFactor
    lorentz_indices: tuple[object, ...]

    def __str__(self):
        rendered = str(self.covariant_factor)
        for lorentz_index in self.lorentz_indices:
            rendered = f"PartialD({rendered}, {lorentz_index})"
        return rendered


@dataclass(frozen=True)
class PartialDerivativeFactor(_DeclaredFactorMixin):
    field: Field
    lorentz_indices: tuple[object, ...]
    conjugated: bool = False
    labels: dict = dataclass_field(default_factory=dict)

    @property
    def bar(self) -> "PartialDerivativeFactor":
        if self.field.self_conjugate:
            return self
        return PartialDerivativeFactor(
            field=self.field,
            lorentz_indices=self.lorentz_indices,
            conjugated=not self.conjugated,
            labels=self.labels,
        )

    def __str__(self):
        base = f"{self.field.name}.bar" if self.conjugated and not self.field.self_conjugate else self.field.name
        indices = ", ".join(str(index) for index in self.lorentz_indices)
        return f"PartialD({base}, {indices})"


@dataclass(frozen=True)
class GammaFactor(_DeclaredFactorMixin):
    lorentz_index: object

    def __str__(self):
        return f"Gamma({self.lorentz_index})"


@dataclass(frozen=True)
class Gamma5Factor(_DeclaredFactorMixin):
    def __str__(self):
        return "Gamma5()"


@dataclass(frozen=True)
class _MatrixFactor(_DeclaredFactorMixin):
    """A matrix acting on one index family inside a replacement expression.

    The factor multiplies a single target field on one of its indices, e.g. a
    chiral projector on the spinor index (``ProjM * l``) or a flavor rotation on
    the generation index (``rotation(CKM, CKMDag) * dq``). ``build`` and
    ``conjugate_build`` are ``(left, right) -> Expression`` callables; the
    conjugate is used (with swapped legs) for ``field.bar`` occurrences.
    """

    name: str
    index: IndexType
    build: Callable
    conjugate_build: Callable

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class MetricFactor(_DeclaredFactorMixin):
    left_index: object
    right_index: object

    def __str__(self):
        return f"Metric({self.left_index}, {self.right_index})"


@dataclass(frozen=True)
class GeneratorFactor(_DeclaredFactorMixin):
    adjoint_index: object
    generator_builder: object = gauge_generator
    index_kind: str = COLOR_FUND_KIND

    def __str__(self):
        return f"T({self.adjoint_index})"


@dataclass(frozen=True)
class StructureConstantFactor(_DeclaredFactorMixin):
    left_index: object
    middle_index: object
    right_index: object

    def __str__(self):
        return (
            f"StructureConstant({self.left_index}, {self.middle_index}, "
            f"{self.right_index})"
        )


@dataclass(frozen=True)
class FieldStrengthFactor(_DeclaredFactorMixin):
    gauge_group: object
    left_index: object
    right_index: object
    adjoint_index: object = None

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        if self.adjoint_index is not None:
            return (
                f"FieldStrength({group_name}, {self.left_index}, "
                f"{self.right_index}, {self.adjoint_index})"
            )
        return f"FieldStrength({group_name}, {self.left_index}, {self.right_index})"


@dataclass(frozen=True)
class GaugeFixingDeclaration:
    gauge_group: object
    xi: object = 1
    coefficient: object = 1
    label: str = ""

    def __add__(self, other):
        from .lagrangian import DeclaredLagrangian

        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian

        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

    def __sub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__sub__(other)

    def __rsub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__rsub__(other)

    def __mul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=self.coefficient * other)
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=other * self.coefficient)
        return NotImplemented

    def __neg__(self):
        return replace(self, coefficient=-self.coefficient)

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        body = f"GaugeFixing({group_name}, xi={self.xi})"
        if self.coefficient == 1:
            return body
        return f"{self.coefficient} * {body}"


@dataclass(frozen=True)
class GhostLagrangianDeclaration:
    gauge_group: object
    coefficient: object = 1
    label: str = ""

    def __add__(self, other):
        from .lagrangian import DeclaredLagrangian

        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian

        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

    def __sub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__sub__(other)

    def __rsub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__rsub__(other)

    def __mul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=self.coefficient * other)
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return replace(self, coefficient=other * self.coefficient)
        return NotImplemented

    def __neg__(self):
        return replace(self, coefficient=-self.coefficient)

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        body = f"GhostLagrangian({group_name})"
        if self.coefficient == 1:
            return body
        return f"{self.coefficient} * {body}"


def _is_decl_scalar(value) -> bool:
    from .interactions import FieldOccurrence, InteractionTerm
    from .lagrangian import (
        CompiledLagrangian,
        ComplexScalarKineticTerm,
        DeclaredLagrangian,
        DiracKineticTerm,
        GaugeFixingTerm,
        GhostTerm,
    )

    return not isinstance(
        value,
        (
            Field,
            ConjugateField,
            FieldOccurrence,
            _FieldFactor,
            CovariantDerivativeFactor,
            DifferentiatedCovariantFactor,
            PartialDerivativeFactor,
            GammaFactor,
            Gamma5Factor,
            _MatrixFactor,
            MetricFactor,
            GeneratorFactor,
            StructureConstantFactor,
            FieldStrengthFactor,
            _DeclaredMonomial,
            DeclaredLagrangian,
            InteractionTerm,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
            CompiledLagrangian,
        ),
    )


def _coerce_decl_factor(value):
    from .interactions import FieldOccurrence

    if isinstance(value, Field):
        return _FieldFactor(value)
    if isinstance(value, ConjugateField):
        return _FieldFactor(value.field, conjugated=True)
    if isinstance(value, FieldOccurrence):
        return _FieldFactor(
            value.field,
            conjugated=value.conjugated,
            labels=value.labels,
        )
    if isinstance(
        value,
        (
            _FieldFactor,
            CovariantDerivativeFactor,
            PartialDerivativeFactor,
            GammaFactor,
            Gamma5Factor,
            _MatrixFactor,
            MetricFactor,
            GeneratorFactor,
            StructureConstantFactor,
            FieldStrengthFactor,
        ),
    ):
        return value
    return None


@dataclass(frozen=True)
class _DeclaredMonomial:
    coefficient: object = 1
    factors: tuple[object, ...] = ()

    @classmethod
    def from_factor(cls, factor) -> "_DeclaredMonomial":
        return cls(coefficient=1, factors=(factor,))

    def __mul__(self, other):
        if isinstance(other, _DeclaredMonomial):
            return _DeclaredMonomial(
                coefficient=self.coefficient * other.coefficient,
                factors=self.factors + other.factors,
            )
        factor = _coerce_decl_factor(other)
        if factor is not None:
            return _DeclaredMonomial(
                coefficient=self.coefficient,
                factors=self.factors + (factor,),
            )
        if _is_decl_scalar(other):
            return _DeclaredMonomial(
                coefficient=self.coefficient * other,
                factors=self.factors,
            )
        return NotImplemented

    def __rmul__(self, other):
        if _is_decl_scalar(other):
            return _DeclaredMonomial(
                coefficient=other * self.coefficient,
                factors=self.factors,
            )
        factor = _coerce_decl_factor(other)
        if factor is not None:
            return _DeclaredMonomial(
                coefficient=self.coefficient,
                factors=(factor,) + self.factors,
            )
        return NotImplemented

    def __neg__(self):
        return _DeclaredMonomial(coefficient=-self.coefficient, factors=self.factors)

    def __add__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__add__(other)

    def __radd__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__radd__(other)

    def __sub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__sub__(other)

    def __rsub__(self, other):
        from .lagrangian import DeclaredLagrangian

        return DeclaredLagrangian.from_item(self).__rsub__(other)

    def __str__(self):
        pieces = [str(factor) for factor in self.factors]
        if self.coefficient != 1 or not pieces:
            pieces = [str(self.coefficient)] + pieces
        return " * ".join(pieces)


def CovD(field, lorentz_index, *, conjugated=False) -> CovariantDerivativeFactor:
    """Declarative covariant derivative factor for ``DeclaredLagrangian``.

    Accepts ``Field``, ``Field.bar``, or ``(Field, bool)`` and can be used in
    expressions such as ``I * Psi.bar * Gamma(mu) * CovD(Psi, mu)``.

    ``CovD(...)`` is metadata-dependent: it belongs in a ``Model(...)``
    declaration compiled through ``model.lagrangian()``.
    """
    from .interactions import _parse_field_arg

    field_obj, parsed_conjugated = _parse_field_arg(field)
    return CovariantDerivativeFactor(
        field=field_obj,
        lorentz_index=lorentz_index,
        conjugated=bool(parsed_conjugated or conjugated),
    )


def PartialD(field, lorentz_index, *, labels=None, conjugated=False) -> PartialDerivativeFactor:
    """Declarative partial derivative factor for local derivative monomials.

    Accepts ``Field``, ``Field.bar``, ``FieldOccurrence``, ``(Field, bool)``,
    another ``PartialD(...)`` factor to build higher derivatives, or
    ``CovD(...)`` to differentiate an already-declared covariant factor.
    """
    from .interactions import FieldOccurrence
    from .interactions import _parse_field_arg

    if isinstance(field, DifferentiatedCovariantFactor):
        if labels is not None or conjugated:
            raise TypeError(
                "Nested PartialD(CovD(...)) already carries field labels and conjugation."
            )
        return DifferentiatedCovariantFactor(
            covariant_factor=field.covariant_factor,
            lorentz_indices=field.lorentz_indices + (lorentz_index,),
        )
    if isinstance(field, CovariantDerivativeFactor):
        if labels is not None or conjugated:
            raise TypeError(
                "Pass labels/conjugation either through CovD(...) or through PartialD(...), "
                "not both."
            )
        return DifferentiatedCovariantFactor(
            covariant_factor=field,
            lorentz_indices=(lorentz_index,),
        )
    if isinstance(field, PartialDerivativeFactor):
        if labels is not None or conjugated:
            raise TypeError(
                "Nested PartialD(...) already carries field labels and conjugation."
            )
        return PartialDerivativeFactor(
            field=field.field,
            lorentz_indices=field.lorentz_indices + (lorentz_index,),
            conjugated=field.conjugated,
            labels=field.labels,
        )
    if isinstance(field, FieldOccurrence):
        if labels is not None or conjugated:
            raise TypeError(
                "Pass labels/conjugation either through FieldOccurrence(...) "
                "or through PartialD(...), not both."
            )
        return PartialDerivativeFactor(
            field=field.field,
            lorentz_indices=(lorentz_index,),
            conjugated=field.conjugated,
            labels=field.labels,
        )
    field_obj, parsed_conjugated = _parse_field_arg(field)
    return PartialDerivativeFactor(
        field=field_obj,
        lorentz_indices=(lorentz_index,),
        conjugated=bool(parsed_conjugated or conjugated),
        labels=labels or {},
    )


def Gamma(*indices):
    """Gamma-matrix helper for declarative and fully explicit local operators.

    Supported forms:
    - ``Gamma(mu)``: declarative chain placeholder
    - ``Gamma(i, j, mu)``: fully explicit raw tensor ``gamma(i, j, mu)``
    """
    if len(indices) == 1:
        return GammaFactor(lorentz_index=indices[0])
    if len(indices) == 3:
        left_spinor, right_spinor, lorentz_index = indices
        return gamma_matrix(left_spinor, right_spinor, lorentz_index)
    raise TypeError(
        "Gamma(...) expects either 1 index (Gamma(mu)) or 3 indices "
        "(Gamma(i, j, mu))."
    )


def Gamma5() -> Gamma5Factor:
    """Declarative gamma5 placeholder for local spinor chains."""
    return Gamma5Factor()


def Metric(left_index, right_index) -> MetricFactor:
    """Declarative Lorentz metric placeholder for local tensor monomials."""
    return MetricFactor(left_index=left_index, right_index=right_index)


#: Left chiral projector ``P_L`` as a replacement-expression matrix factor.
ProjM = _MatrixFactor(
    name="ProjM",
    index=SPINOR_INDEX,
    build=chiral_projector_left,
    conjugate_build=chiral_projector_right,
)

#: Right chiral projector ``P_R`` as a replacement-expression matrix factor.
ProjP = _MatrixFactor(
    name="ProjP",
    index=SPINOR_INDEX,
    build=chiral_projector_right,
    conjugate_build=chiral_projector_left,
)


def rotation(forward: Parameter, dagger: Parameter) -> _MatrixFactor:
    """Flavor-rotation matrix factor for replacement expressions.

    ``forward`` and ``dagger`` are two-index unitary partner parameters such as
    a CKM matrix and its conjugate. The factor rotates the target field on the
    parameter's (flavor) index, e.g. ``rotation(CKM, CKMDag) * ProjM * dq``.
    """
    if not isinstance(forward, Parameter) or not isinstance(dagger, Parameter):
        raise TypeError("rotation(...) expects two Parameter matrices.")
    if len(forward.indices) != 2 or len(dagger.indices) != 2:
        raise TypeError(
            f"rotation matrix {forward.name!r} must carry exactly two indices."
        )
    return _MatrixFactor(
        name=forward.name,
        index=forward.indices[0],
        build=lambda left, right: forward(left, right),
        conjugate_build=lambda left, right: dagger(left, right),
    )


def T(*indices):
    """Generator helper for declarative and fully explicit local operators.

    Supported forms:
    - ``T(a)``: declarative chain placeholder
    - ``T(a, i, j)``: fully explicit raw tensor ``t(a, i, j)``
    """
    if len(indices) == 1:
        return GeneratorFactor(adjoint_index=indices[0])
    if len(indices) == 3:
        adjoint_index, left_index, right_index = indices
        return gauge_generator(adjoint_index, left_index, right_index)
    raise TypeError(
        "T(...) expects either 1 index (T(a)) or 3 indices (T(a, i, j))."
    )


def StructureConstant(left_index, middle_index, right_index) -> StructureConstantFactor:
    """Local structure-constant placeholder for already-expanded monomials.

    ``StructureConstant(...)`` is a local tensor helper, not a fully generic
    group object. It does not determine which gauge group is meant, does not
    validate normalization conventions, and does not infer adjoint slots from
    model metadata.

    Use it for explicit local tensor terms only. For gauge-aware Yang-Mills or
    covariant constructions, prefer ``Model(..., lagrangian_decl=...)`` with a
    declared ``GaugeGroup``.
    """
    return StructureConstantFactor(
        left_index=left_index,
        middle_index=middle_index,
        right_index=right_index,
    )


def FieldStrength(gauge_group, left_index, right_index, adjoint_index=None) -> FieldStrengthFactor:
    """Declarative field-strength placeholder for ``DeclaredLagrangian``.

    ``FieldStrength(...)`` participates in metadata-dependent gauge-sector
    lowering and should be declared through a ``Model(...)`` compiled with
    ``model.lagrangian()``.

    For non-abelian gauge groups the adjoint index is mandatory, e.g.
    ``FieldStrength(SU3, mu, nu, a)``; the compiler expands it into
    ``d_mu A^a_nu - d_nu A^a_mu + g f^{abc} A^b_mu A^c_nu``. Abelian groups
    must omit it (``FieldStrength(U1, mu, nu)``) and expand into
    ``d_mu A_nu - d_nu A_mu``.
    """
    return FieldStrengthFactor(
        gauge_group=gauge_group,
        left_index=left_index,
        right_index=right_index,
        adjoint_index=adjoint_index,
    )


def GaugeFixing(gauge_group, *, xi=1, coefficient=1, label="") -> GaugeFixingDeclaration:
    """Declarative ordinary gauge-fixing wrapper for ``DeclaredLagrangian``."""
    return GaugeFixingDeclaration(
        gauge_group=gauge_group,
        xi=xi,
        coefficient=coefficient,
        label=label,
    )


def GhostLagrangian(gauge_group, *, coefficient=1, label="") -> GhostLagrangianDeclaration:
    """Declarative Faddeev-Popov ghost-sector wrapper for ``DeclaredLagrangian``."""
    return GhostLagrangianDeclaration(
        gauge_group=gauge_group,
        coefficient=coefficient,
        label=label,
    )
