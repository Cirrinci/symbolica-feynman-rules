"""Operator registry for the SMEFT Green basis.

Every Appendix D operator is registered as an :class:`Operator` carrying its
paper name, sector, type (physical / redundant / evanescent), source table,
Wilson-coefficient flavour arity, implementation status and a builder.  The
registry makes it possible to

* inspect one operator in isolation (``op.structure(core)``),
* build one operator with its Wilson coefficient (``op.term(core)``),
* compile one operator to Feynman rules (``op.feynman_rule(core, ...)``),
* assemble a whole sector or the complete basis (see :func:`operators_in`),

without ever expanding the entire model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from symbolica import Expression, S

from feynpy import CompiledLagrangian, DeclaredLagrangian, Parameter
from feynpy.transformation_postprocess import _hermitian_conjugate_term

from .tensors import Poly
from .wilson import SYMMETRY_NOTES, make_wilson_coefficient


SECTORS = ("bosonic", "two_fermion", "four_fermion")
TYPES = ("physical", "redundant", "evanescent")
STATUS_VALUES = ("implemented", "partial", "blocked")

_FLAVOUR_LABELS = ("i", "j", "k", "l", "m", "n")

_NON_HERMITIAN_OPERATORS = frozenset(
    {
        # Table 2: explicitly ``+ h.c.`` sectors and the non-Hermitian ``OHud``.
        "OHud",
        "OuG",
        "OuW",
        "OuB",
        "OdG",
        "OdW",
        "OdB",
        "OeW",
        "OeB",
        "OuH",
        "OdH",
        "OeH",
        "RuHD1",
        "RuHD2",
        "RuHD3",
        "RuHD4",
        "RdHD1",
        "RdHD2",
        "RdHD3",
        "RdHD4",
        "ReHD1",
        "ReHD2",
        "ReHD3",
        "ReHD4",
        # Table 3: LR scalar/tensor structures.
        "Oquqd1",
        "Oquqd8",
        "Oledq",
        "Olequ1",
        "Olequ3",
        # Table 4: explicitly ``+ h.c.`` evanescent dipoles / psi2HD2.
        "EuG",
        "EuW",
        "EuB",
        "EdG",
        "EdW",
        "EdB",
        "EeW",
        "EeB",
        "EuH",
        "EdH",
        "EeH",
        # Tables 5-6: LR-LR epsilon / ledq topologies are not self-conjugate.
        "E2quqd",
        "E2_8quqd",
        "E2ledq",
        "E2lequ",
        "Eluqe",
        "E2luqe",
    }
)


BuilderResult = "Poly | DeclaredLagrangian"


@dataclass(frozen=True)
class Operator:
    """One Green-basis operator."""

    name: str
    label: str
    sector: str
    otype: str
    table: int
    n_flavour: int
    builder: Callable[..., object]
    status: str = "implemented"
    note: str = ""

    def __post_init__(self):
        if self.sector not in SECTORS:
            raise ValueError(f"{self.name}: unknown sector {self.sector!r}.")
        if self.otype not in TYPES:
            raise ValueError(f"{self.name}: unknown type {self.otype!r}.")
        if self.status not in STATUS_VALUES:
            raise ValueError(f"{self.name}: unknown status {self.status!r}.")

    # -- Wilson coefficient / flavour ------------------------------------
    def flavour_labels(self) -> tuple[Expression, ...]:
        if self.n_flavour > len(_FLAVOUR_LABELS):
            raise ValueError(f"{self.name}: too many flavour indices.")
        return tuple(S(_FLAVOUR_LABELS[k]) for k in range(self.n_flavour))

    def wilson_coefficient(self, core) -> Parameter:
        return make_wilson_coefficient(
            self.name,
            self.otype,
            self.n_flavour,
            generation=core.indices.generation,
        )

    def symmetry_note(self) -> str:
        return SYMMETRY_NOTES.get(self.name, "")

    def is_non_hermitian(self) -> bool:
        return self.name in _NON_HERMITIAN_OPERATORS

    # -- construction ----------------------------------------------------
    def _declared(self, core, coeff_expr) -> DeclaredLagrangian:
        flav = self.flavour_labels()
        result = self.builder(core, coeff_expr, flav)
        if isinstance(result, Poly):
            return result.declared()
        if isinstance(result, DeclaredLagrangian):
            return result
        return DeclaredLagrangian.from_item(result)

    def structure(self, core) -> DeclaredLagrangian:
        """Return the bare operator (Wilson coefficient set to one)."""
        return self._declared(core, Expression.num(1))

    def term(self, core, *, coeff=None) -> DeclaredLagrangian:
        """Return ``C_i O_i`` as a declared Lagrangian."""
        if coeff is None:
            wc = self.wilson_coefficient(core)
            coeff = wc(*self.flavour_labels())
        return self._declared(core, coeff)

    def lagrangian(self, core, *, coeff=None) -> CompiledLagrangian:
        """Compile this single operator to a :class:`CompiledLagrangian`."""
        extra = () if coeff is not None else (self.wilson_coefficient(core),)
        compiled = core.compile_operator(
            self.term(core, coeff=coeff),
            extra_parameters=extra,
        )
        if self.is_non_hermitian():
            compiled = _with_hermitian_conjugate(compiled)
        return compiled

    def feynman_rule(self, core, *fields, coeff=None, **kwargs):
        kwargs.setdefault("simplify_gamma", False)
        return self.lagrangian(core, coeff=coeff).feynman_rule(*fields, **kwargs)

    def canonical_dimensions(self, core) -> set:
        """Return the set of mass dimensions of the operator's monomials.

        Wilson coefficients and gauge couplings are dimensionless in the
        convention of :mod:`.wilson`, so every monomial of a genuine dimension-six
        operator must have mass dimension 6.
        """
        from fractions import Fraction

        structure = self.structure(core)
        return {term_mass_dimension(term) for term in structure.source_terms}


def _field_mass_dimension(field) -> "Fraction":
    from fractions import Fraction

    spin = str(Fraction(field.spin))
    if spin == "0":
        return Fraction(1)
    if spin == "1/2":
        return Fraction(3, 2)
    if spin == "1":
        return Fraction(1)
    raise ValueError(f"Cannot assign a mass dimension to spin {field.spin!r}.")


def term_mass_dimension(monomial):
    """Mass dimension of one declared monomial (couplings taken dimensionless)."""
    from fractions import Fraction

    from feynpy.declared import (
        CovariantDerivativeFactor,
        DifferentiatedCovariantFactor,
        FieldStrengthFactor,
        PartialDerivativeFactor,
        _FieldFactor,
    )

    total = Fraction(0)
    for factor in monomial.factors:
        if isinstance(factor, _FieldFactor):
            total += _field_mass_dimension(factor.field)
        elif isinstance(factor, CovariantDerivativeFactor):
            total += _field_mass_dimension(factor.field) + 1
        elif isinstance(factor, DifferentiatedCovariantFactor):
            total += (
                _field_mass_dimension(factor.covariant_factor.field)
                + 1
                + len(factor.lorentz_indices)
            )
        elif isinstance(factor, PartialDerivativeFactor):
            total += _field_mass_dimension(factor.field) + len(factor.lorentz_indices)
        elif isinstance(factor, FieldStrengthFactor):
            total += Fraction(2)
        # gamma / gamma5 / metric / generator / structure-constant / projector
        # factors and scalar coefficients are dimensionless.
    return total


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Operator] = {}


def _with_hermitian_conjugate(compiled: CompiledLagrangian) -> CompiledLagrangian:
    """Return ``compiled + compiled^dagger`` using the compiled-term conjugator."""

    real_symbols = tuple(
        parameter.symbol
        for parameter in compiled.parameters
        if getattr(parameter, "is_real", False)
    )
    conjugate = CompiledLagrangian(
        terms=tuple(
            _hermitian_conjugate_term(
                term,
                real_symbols=real_symbols,
                parameters=compiled.parameters,
            )
            for term in compiled.terms
        ),
        parameters=compiled.parameters,
    )
    return compiled + conjugate


def register(operator: Operator) -> Operator:
    if operator.name in _REGISTRY:
        raise ValueError(f"Operator {operator.name!r} already registered.")
    _REGISTRY[operator.name] = operator
    return operator


def operator(
    name: str,
    label: str,
    sector: str,
    otype: str,
    table: int,
    *,
    n_flavour: int = 0,
    status: str = "implemented",
    note: str = "",
):
    """Decorator registering the decorated function as an operator builder."""

    def decorate(builder: Callable[..., object]) -> Operator:
        return register(
            Operator(
                name=name,
                label=label,
                sector=sector,
                otype=otype,
                table=table,
                n_flavour=n_flavour,
                builder=builder,
                status=status,
                note=note,
            )
        )

    return decorate


def all_operators() -> tuple[Operator, ...]:
    _ensure_loaded()
    return tuple(_REGISTRY.values())


def get_operator(name: str) -> Operator:
    _ensure_loaded()
    return _REGISTRY[name]


def operators_in(
    *,
    sector: Optional[str] = None,
    otype: Optional[str] = None,
    table: Optional[int] = None,
    status: Optional[str] = None,
) -> tuple[Operator, ...]:
    """Return registered operators filtered by sector / type / table / status."""

    _ensure_loaded()
    result = []
    for op in _REGISTRY.values():
        if sector is not None and op.sector != sector:
            continue
        if otype is not None and op.otype != otype:
            continue
        if table is not None and op.table != table:
            continue
        if status is not None and op.status != status:
            continue
        result.append(op)
    return tuple(result)


_LOADED = False


def _ensure_loaded() -> None:
    """Import the operator modules so their registrations run exactly once."""

    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import operators_bosonic  # noqa: F401
    from . import operators_two_fermion  # noqa: F401
    from . import operators_four_fermion  # noqa: F401
    from . import operators_evanescent  # noqa: F401


__all__ = (
    "Operator",
    "SECTORS",
    "TYPES",
    "STATUS_VALUES",
    "register",
    "operator",
    "all_operators",
    "get_operator",
    "operators_in",
)
