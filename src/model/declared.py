"""Declarative mini-language factors and builders."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .metadata import ConjugateField, Field
# ---------------------------------------------------------------------------
# Declarative Lagrangian factors  (CovD / Gamma / FieldStrength DSL)
# ---------------------------------------------------------------------------


class _DeclaredFactorMixin:
    def __mul__(self, other):
        return _DeclaredMonomial.from_factor(self).__mul__(other)

    def __rmul__(self, other):
        return _DeclaredMonomial.from_factor(self).__rmul__(other)

    def __add__(self, other):
        return _DeclaredMonomial.from_factor(self).__add__(other)

    def __radd__(self, other):
        return _DeclaredMonomial.from_factor(self).__radd__(other)


@dataclass(frozen=True)
class _FieldFactor(_DeclaredFactorMixin):
    field: Field
    conjugated: bool = False

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
class PartialDerivativeFactor(_DeclaredFactorMixin):
    field: Field
    lorentz_indices: tuple[object, ...]
    conjugated: bool = False

    @property
    def bar(self) -> "PartialDerivativeFactor":
        if self.field.self_conjugate:
            return self
        return PartialDerivativeFactor(
            field=self.field,
            lorentz_indices=self.lorentz_indices,
            conjugated=not self.conjugated,
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
class MetricFactor(_DeclaredFactorMixin):
    left_index: object
    right_index: object

    def __str__(self):
        return f"Metric({self.left_index}, {self.right_index})"


@dataclass(frozen=True)
class GeneratorFactor(_DeclaredFactorMixin):
    adjoint_index: object

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

    def __str__(self):
        group_name = getattr(self.gauge_group, "name", self.gauge_group)
        return f"FieldStrength({group_name}, {self.left_index}, {self.right_index})"


@dataclass(frozen=True)
class GaugeFixingDeclaration:
    gauge_group: object
    xi: object = 1
    coefficient: object = 1
    label: str = ""

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

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
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=(self,) + terms)

    def __radd__(self, other):
        if other == 0:
            return DeclaredLagrangian(source_terms=(self,))
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + (self,))

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
    from .interactions import InteractionTerm
    from .lagrangian import (
        CompiledLagrangian,
        ComplexScalarKineticTerm,
        DeclaredLagrangian,
        DiracKineticTerm,
        GaugeFixingTerm,
        GaugeKineticTerm,
        GhostTerm,
    )

    return not isinstance(
        value,
        (
            Field,
            ConjugateField,
            _FieldFactor,
            CovariantDerivativeFactor,
            PartialDerivativeFactor,
            GammaFactor,
            Gamma5Factor,
            MetricFactor,
            GeneratorFactor,
            StructureConstantFactor,
            FieldStrengthFactor,
            _DeclaredMonomial,
            DeclaredLagrangian,
            InteractionTerm,
            DiracKineticTerm,
            ComplexScalarKineticTerm,
            GaugeKineticTerm,
            GaugeFixingDeclaration,
            GaugeFixingTerm,
            GhostLagrangianDeclaration,
            GhostTerm,
            CompiledLagrangian,
        ),
    )


def _coerce_decl_factor(value):
    if isinstance(value, Field):
        return _FieldFactor(value)
    if isinstance(value, ConjugateField):
        return _FieldFactor(value.field, conjugated=True)
    if isinstance(
        value,
        (
            _FieldFactor,
            CovariantDerivativeFactor,
            PartialDerivativeFactor,
            GammaFactor,
            Gamma5Factor,
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

    def __str__(self):
        pieces = [str(factor) for factor in self.factors]
        if self.coefficient != 1 or not pieces:
            pieces = [str(self.coefficient)] + pieces
        return " * ".join(pieces)


def CovD(field, lorentz_index) -> CovariantDerivativeFactor:
    """Declarative covariant derivative factor for ``DeclaredLagrangian``.

    Accepts ``Field``, ``Field.bar``, or ``(Field, bool)`` and can be used in
    expressions such as ``I * Psi.bar * Gamma(mu) * CovD(Psi, mu)``.
    """
    from .interactions import _parse_field_arg

    field_obj, conjugated = _parse_field_arg(field)
    return CovariantDerivativeFactor(
        field=field_obj,
        lorentz_index=lorentz_index,
        conjugated=conjugated,
    )


def PartialD(field, lorentz_index) -> PartialDerivativeFactor:
    """Declarative partial derivative factor for local derivative monomials.

    Accepts ``Field``, ``Field.bar``, ``(Field, bool)``, or another
    ``PartialD(...)`` factor to build higher derivatives.
    """
    from .interactions import _parse_field_arg

    if isinstance(field, PartialDerivativeFactor):
        return PartialDerivativeFactor(
            field=field.field,
            lorentz_indices=field.lorentz_indices + (lorentz_index,),
            conjugated=field.conjugated,
        )
    field_obj, conjugated = _parse_field_arg(field)
    return PartialDerivativeFactor(
        field=field_obj,
        lorentz_indices=(lorentz_index,),
        conjugated=conjugated,
    )


def Gamma(lorentz_index) -> GammaFactor:
    """Declarative gamma-matrix placeholder for ``DeclaredLagrangian``."""
    return GammaFactor(lorentz_index=lorentz_index)


def Gamma5() -> Gamma5Factor:
    """Declarative gamma5 placeholder for local spinor chains."""
    return Gamma5Factor()


def Metric(left_index, right_index) -> MetricFactor:
    """Declarative Lorentz metric placeholder for local tensor monomials."""
    return MetricFactor(left_index=left_index, right_index=right_index)


def T(adjoint_index) -> GeneratorFactor:
    """Declarative fundamental-representation generator placeholder."""
    return GeneratorFactor(adjoint_index=adjoint_index)


def StructureConstant(left_index, middle_index, right_index) -> StructureConstantFactor:
    """Declarative structure-constant placeholder for local tensor monomials."""
    return StructureConstantFactor(
        left_index=left_index,
        middle_index=middle_index,
        right_index=right_index,
    )


def FieldStrength(gauge_group, left_index, right_index) -> FieldStrengthFactor:
    """Declarative field-strength placeholder for ``DeclaredLagrangian``."""
    return FieldStrengthFactor(
        gauge_group=gauge_group,
        left_index=left_index,
        right_index=right_index,
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
