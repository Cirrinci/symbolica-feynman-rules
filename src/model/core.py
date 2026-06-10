"""Top-level model container."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

from .declared import (
    CovariantDerivativeFactor,
    DifferentiatedCovariantFactor,
    FieldStrengthFactor,
    GaugeFixingDeclaration,
    GhostLagrangianDeclaration,
    PartialDerivativeFactor,
    _DeclaredMonomial,
    _FieldFactor,
)
from .lagrangian import (
    CompiledLagrangian,
    ComplexScalarKineticTerm,
    DeclaredLagrangian,
    DiracKineticTerm,
    FlavorExpandOption,
    GaugeFixingTerm,
    GhostTerm,
)
from .lowering import (
    _analyze_declared_source_term,
    _unsupported_declared_source_term_error,
)
from .metadata import Field, GaugeGroup, Parameter
from .validation import ValidationIssue, ValidationReport, validate_model


def _append_unique_identity(items: list[object], value):
    if value is None:
        return
    if any(existing is value for existing in items):
        return
    items.append(value)


def _append_gauge_group(gauge_groups: list[GaugeGroup], gauge_group):
    if isinstance(gauge_group, GaugeGroup):
        _append_unique_identity(gauge_groups, gauge_group)


def _looks_like_lagrangian_decl(item) -> bool:
    try:
        DeclaredLagrangian.from_item(item)
    except TypeError:
        return False
    return True


_MISSING = object()


def _assign_model_argument(values: dict[str, object], slot: str, value):
    if values[slot] is not _MISSING:
        raise TypeError(f"Model() got multiple values for argument '{slot}'")
    values[slot] = value


_GAUGE_GROUP_DECLARATION_TERM_TYPES = (
    GaugeFixingTerm,
    GhostTerm,
    GaugeFixingDeclaration,
    GhostLagrangianDeclaration,
)


def _collect_declared_term_metadata(term, *, fields: list[Field], gauge_groups: list[GaugeGroup]):
    if isinstance(term, (DiracKineticTerm, ComplexScalarKineticTerm)):
        _append_gauge_group(gauge_groups, term.gauge_group)
        if isinstance(term.field, Field):
            _append_unique_identity(fields, term.field)
        return

    if isinstance(term, _GAUGE_GROUP_DECLARATION_TERM_TYPES):
        _append_gauge_group(gauge_groups, term.gauge_group)
        return

    if isinstance(term, _DeclaredMonomial):
        for factor in term.factors:
            if isinstance(factor, (_FieldFactor, CovariantDerivativeFactor, PartialDerivativeFactor)):
                _append_unique_identity(fields, factor.field)
                continue
            if isinstance(factor, DifferentiatedCovariantFactor):
                _append_unique_identity(fields, factor.covariant_factor.field)
                continue
            if isinstance(factor, FieldStrengthFactor):
                _append_gauge_group(gauge_groups, factor.gauge_group)


def _infer_model_metadata(
    *,
    explicit_fields: tuple[Field, ...],
    explicit_gauge_groups: tuple[GaugeGroup, ...],
    source_terms: tuple[object, ...],
) -> tuple[tuple[Field, ...], tuple[GaugeGroup, ...]]:
    fields: list[Field] = list(explicit_fields)
    gauge_groups: list[GaugeGroup] = list(explicit_gauge_groups)

    for term in source_terms:
        _collect_declared_term_metadata(term, fields=fields, gauge_groups=gauge_groups)

    for gauge_group in tuple(gauge_groups):
        if isinstance(gauge_group.gauge_boson, Field):
            _append_unique_identity(fields, gauge_group.gauge_boson)
        if isinstance(gauge_group.ghost_field, Field):
            _append_unique_identity(fields, gauge_group.ghost_field)

    for field in tuple(fields):
        for member in getattr(field, "class_members", ()):
            _append_unique_identity(fields, member)

    return tuple(fields), tuple(gauge_groups)


@dataclass(init=False)
class Model:
    """Top-level FeynRules-style model container.

    The recommended workflow is:

    1. declare indices (``flavor_index``, ``COLOR_FUND_INDEX``, ...)
    2. declare gauge representations and gauge groups
    3. declare fields with ``Field(...)`` (or convenience helpers like
       ``dirac_field`` / ``scalar_field``; pass ``class_members=(...)`` and
       ``flavor_index=...`` for FeynRules-like flavor-class declarations)
    4. declare parameters with ``Parameter(...)``
    5. build the model with ``Model(gauge_groups=..., fields=...,
       parameters=..., lagrangian_decl=...)`` or the shorthand
       ``Model(declared_term, name=..., gauge_groups=..., fields=...,
       parameters=...)``
    6. extract Feynman rules with ``model.feynman_rule(...)`` /
       ``model.vertex_signatures(...)``.
    """

    name: str = ""
    gauge_groups: tuple[GaugeGroup, ...] = ()
    fields: tuple[Field, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    lagrangian_decl: Optional[DeclaredLagrangian] = None
    _compiled_lagrangian: Optional[CompiledLagrangian] = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *args,
        name=_MISSING,
        gauge_groups=_MISSING,
        fields=_MISSING,
        parameters=_MISSING,
        lagrangian_decl=_MISSING,
    ):
        values = {
            "name": _MISSING,
            "gauge_groups": _MISSING,
            "fields": _MISSING,
            "parameters": _MISSING,
            "lagrangian_decl": _MISSING,
        }

        if (
            args
            and lagrangian_decl is _MISSING
            and not isinstance(args[0], str)
            and _looks_like_lagrangian_decl(args[0])
        ):
            if len(args) > 5:
                raise TypeError(f"Model() takes at most 5 positional arguments but {len(args)} were given")
            _assign_model_argument(values, "lagrangian_decl", args[0])
            for slot, value in zip(("name", "gauge_groups", "fields", "parameters"), args[1:]):
                _assign_model_argument(values, slot, value)
        else:
            if len(args) > 5:
                raise TypeError(f"Model() takes at most 5 positional arguments but {len(args)} were given")
            for slot, value in zip(
                ("name", "gauge_groups", "fields", "parameters", "lagrangian_decl"),
                args,
            ):
                _assign_model_argument(values, slot, value)

            if args and not isinstance(args[0], str) and values["lagrangian_decl"] is _MISSING:
                raise TypeError(
                    "Model() accepts only modern source declarations as the first positional "
                    "argument. Compiled InteractionTerm inputs are internal-only; use "
                    "CompiledLagrangian(terms=...) for already-lowered terms."
                )

        for slot, value in (
            ("name", name),
            ("gauge_groups", gauge_groups),
            ("fields", fields),
            ("parameters", parameters),
            ("lagrangian_decl", lagrangian_decl),
        ):
            if value is not _MISSING:
                _assign_model_argument(values, slot, value)

        self.name = "" if values["name"] is _MISSING else values["name"]
        self.gauge_groups = () if values["gauge_groups"] is _MISSING else values["gauge_groups"]
        self.fields = () if values["fields"] is _MISSING else values["fields"]
        self.parameters = () if values["parameters"] is _MISSING else values["parameters"]
        self.lagrangian_decl = None if values["lagrangian_decl"] is _MISSING else values["lagrangian_decl"]
        self._compiled_lagrangian = None

        self.__post_init__()

    def __post_init__(self):
        if (
            self.lagrangian_decl is None
            and not isinstance(self.name, str)
            and _looks_like_lagrangian_decl(self.name)
        ):
            self.lagrangian_decl = self.name
            self.name = ""

        if not isinstance(self.name, str):
            raise TypeError(
                "Model.name must be a string. For source declarations, pass a modern "
                "declarative expression to Model(...). Compiled InteractionTerm inputs "
                "are internal-only; use CompiledLagrangian(terms=...)."
            )

        if self.lagrangian_decl is None:
            self.lagrangian_decl = DeclaredLagrangian()
        elif not isinstance(self.lagrangian_decl, DeclaredLagrangian):
            self.lagrangian_decl = DeclaredLagrangian.from_item(self.lagrangian_decl)

        inferred_fields, inferred_gauge_groups = _infer_model_metadata(
            explicit_fields=self.fields,
            explicit_gauge_groups=self.gauge_groups,
            source_terms=self.lagrangian_decl.source_terms,
        )
        self.fields = inferred_fields
        self.gauge_groups = inferred_gauge_groups

        for term in self.lagrangian_decl.source_terms:
            if _analyze_declared_source_term(term, parameters=self.parameters) is None:
                raise _unsupported_declared_source_term_error()

    def find_field(self, target) -> Optional[Field]:
        """Resolve a field by object identity, declaration name, or symbol."""
        if isinstance(target, Field):
            for field in self.fields:
                if field is target:
                    return field
            return None
        if target is None:
            return None

        target_text = str(target)
        for field in self.fields:
            if field.name == target_text:
                return field
            if str(field.symbol) == target_text:
                return field
            if field.conjugate_symbol is not None and str(field.conjugate_symbol) == target_text:
                return field
        return None

    def find_gauge_group(self, target) -> Optional[GaugeGroup]:
        """Resolve a gauge group by object identity or declaration name."""
        if isinstance(target, GaugeGroup):
            for gauge_group in self.gauge_groups:
                if gauge_group is target:
                    return gauge_group
            return None
        if target is None:
            return None

        target_text = str(target)
        for gauge_group in self.gauge_groups:
            if gauge_group.name == target_text:
                return gauge_group
        return None

    def find_parameter(self, target) -> Optional[Parameter]:
        """Resolve a parameter by object identity, declaration name, or symbol."""
        if isinstance(target, Parameter):
            for parameter in self.parameters:
                if parameter is target:
                    return parameter
            return None
        if target is None:
            return None

        if hasattr(target, "to_canonical_string"):
            target_text = target.to_canonical_string()
        else:
            target_text = str(target)

        for parameter in self.parameters:
            if parameter.name == target_text:
                return parameter
            symbol = parameter.symbol
            if hasattr(symbol, "to_canonical_string"):
                symbol_text = symbol.to_canonical_string()
            else:
                symbol_text = str(symbol)
            if symbol_text == target_text:
                return parameter
        return None

    def gauge_boson_field(self, gauge_group: GaugeGroup) -> Field:
        """Resolve the gauge boson field declared for a gauge group."""
        if isinstance(gauge_group.gauge_boson, Field):
            field = self.find_field(gauge_group.gauge_boson)
            if field is None:
                raise ValueError(
                    f"Gauge group {gauge_group.name!r} requires gauge boson "
                    f"{gauge_group.gauge_boson.name!r} to be declared in model.fields."
                )
            return field

        field = self.find_field(gauge_group.gauge_boson)
        if field is None:
            raise ValueError(
                f"Could not resolve gauge boson {gauge_group.gauge_boson!r} "
                f"for gauge group {gauge_group.name!r}."
            )
        return field

    def validate(self) -> ValidationReport:
        """Return structured model diagnostics without changing compilation behavior."""
        return validate_model(self)

    def lagrangian(self) -> CompiledLagrangian:
        """Compile the declared Lagrangian into a ``CompiledLagrangian``.

        The result is cached: subsequent calls (and the direct
        ``feynman_rule`` / ``vertex_signatures`` methods)
        reuse the same compiled object.
        """
        if self._compiled_lagrangian is None:
            from compiler.gauge import _compile_declared_lagrangian_terms

            self._compiled_lagrangian = CompiledLagrangian(
                terms=_compile_declared_lagrangian_terms(self),
                parameters=self.parameters,
            )
        return self._compiled_lagrangian

    def feynman_rule(
        self,
        *fields,
        momenta=None,
        arity=None,
        select=None,
        simplify: bool = True,
        key_format: str = "names",
        include_delta: bool = False,
        strip_externals: bool = True,
        simplify_gamma: bool = False,
        flavor_expand: FlavorExpandOption = False,
    ):
        """Extract one Feynman rule or a grouped zero-argument rule mapping."""
        return self.lagrangian().feynman_rule(
            *fields,
            momenta=momenta,
            arity=arity,
            select=select,
            simplify=simplify,
            key_format=key_format,
            include_delta=include_delta,
            strip_externals=strip_externals,
            simplify_gamma=simplify_gamma,
            flavor_expand=flavor_expand,
        )

    def apply_operator(
        self,
        operator,
        *,
        flavor_expand: FlavorExpandOption = False,
        max_generated_terms: Optional[int] = None,
    ) -> CompiledLagrangian:
        """Apply one runtime operator to the compiled Lagrangian."""

        return self.lagrangian().apply_operator(
            operator,
            flavor_expand=flavor_expand,
            max_generated_terms=max_generated_terms,
        )

    def to_symbolica(
        self,
        *,
        flavor_expand: FlavorExpandOption = False,
        derivative_style: str = "partiald",
        coordinate_map=None,
    ):
        """Render the compiled Lagrangian as one Symbolica expression.

        This is a convenience forwarder to ``model.lagrangian().to_symbolica()``
        so the top-level ``Model`` API matches the compiled-Lagrangian API.
        """
        return self.lagrangian().to_symbolica(
            flavor_expand=flavor_expand,
            derivative_style=derivative_style,
            coordinate_map=coordinate_map,
        )

    def vertex_signatures(
        self,
        *,
        arity=None,
        signature=None,
        contains_fields=None,
        sector=None,
        flavor_expand: FlavorExpandOption = False,
    ):
        """Enumerate available compiled vertex signatures."""
        return self.lagrangian().vertex_signatures(
            arity=arity,
            signature=signature,
            contains_fields=contains_fields,
            sector=sector,
            flavor_expand=flavor_expand,
        )
