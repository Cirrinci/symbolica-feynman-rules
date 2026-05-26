"""Compiled, standalone, and declarative Lagrangian containers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Iterable, Optional, Union

from symbolica import Expression, S

from .interactions import (
    ExternalLeg,
    InteractionTerm,
    _auto_leg_labels,
    _field_match_key,
    _parse_field_arg,
    _term_matches_fields,
)
from .metadata import IndexType


FlavorExpandOption = Union[bool, IndexType, Iterable[IndexType]]
_EXPAND_ALL_FLAVOR_INDICES = object()


def _normalize_interaction_terms_input(terms):
    from .lowering import _normalize_interaction_terms_input as impl

    return impl(terms)


def _normalize_lagrangian_source_terms(items):
    from .lowering import _normalize_lagrangian_source_terms as impl

    return impl(items)


def _lower_standalone_lagrangian_source_term(term):
    from .lowering import _lower_standalone_lagrangian_source_term as impl

    return impl(term)


def _standalone_lagrangian_source_terms_from_item(item):
    from .lowering import _standalone_lagrangian_source_terms_from_item as impl

    return impl(item)


def _declared_source_terms_from_item(item):
    from .lowering import _declared_source_terms_from_item as impl

    return impl(item)


def _standalone_lagrangian_context_error():
    from .lowering import _standalone_lagrangian_context_error as impl

    return impl()


def _compiled_lagrangian_context_error():
    from .lowering import _compiled_lagrangian_context_error as impl

    return impl()


def _merge_unique_metadata(left, right):
    merged = list(left)
    for item in right:
        if any(existing is item for existing in merged):
            continue
        merged.append(item)
    return tuple(merged)


def _scaled_terms(
    terms: Iterable[InteractionTerm],
    factor,
) -> tuple[InteractionTerm, ...]:
    return tuple(
        replace(term, coupling=term.coupling * factor)
        for term in terms
    )


def _operator_parity(operator) -> int:
    parity = getattr(operator, "parity", None)
    if parity not in (0, 1):
        raise TypeError(
            "operator must expose parity 0 or 1; got "
            f"{type(operator).__name__}."
        )
    return parity


def _normalize_flavor_expand_option(flavor_expand):
    if flavor_expand is _EXPAND_ALL_FLAVOR_INDICES:
        return flavor_expand
    if isinstance(flavor_expand, bool):
        return _EXPAND_ALL_FLAVOR_INDICES if flavor_expand else False
    if isinstance(flavor_expand, IndexType):
        indices = (flavor_expand,)
    else:
        try:
            indices = tuple(flavor_expand)
        except TypeError as exc:
            raise TypeError(
                "`flavor_expand` must be a boolean, one flavor index, or an iterable of flavor indices."
            ) from exc

    normalized: list[IndexType] = []
    for index in indices:
        if not isinstance(index, IndexType):
            raise TypeError(
                "`flavor_expand` iterable entries must be IndexType instances."
            )
        if not index.is_flavor:
            raise ValueError(
                f"`flavor_expand` only accepts flavor indices; got {index.name!r}."
            )
        if index not in normalized:
            normalized.append(index)
    return tuple(normalized)


def _flavor_expand_cache_key(flavor_expand) -> tuple[object, ...]:
    """Return a hashable cache key for one normalized ``flavor_expand`` option."""

    if flavor_expand is False:
        return ("none",)
    if flavor_expand is _EXPAND_ALL_FLAVOR_INDICES:
        return ("all",)

    selected_identifiers = sorted(
        (
            index.name,
            index.kind,
            index.dimension,
            index.is_flavor,
            index.prefix,
        )
        for index in flavor_expand
    )
    return (
        "selected",
        tuple(selected_identifiers),
    )


def _field_arg_from_occurrence(occurrence):
    if occurrence.conjugated and not occurrence.field.self_conjugate:
        return occurrence.field.bar
    return occurrence.field


def _field_arg_name(field_arg):
    if hasattr(field_arg, "field"):
        return f"{field_arg.field.name}.bar"
    return field_arg.name


def _vertex_name_tuple(vertex_fields):
    return tuple(_field_arg_name(field_arg) for field_arg in vertex_fields)


def _parsed_field_arg_name(field_obj, conjugated: bool) -> str:
    if conjugated and not field_obj.self_conjugate:
        return f"{field_obj.name}.bar"
    return field_obj.name


def _term_vertex_key(term: InteractionTerm):
    return tuple(
        sorted(
            (
                _field_match_key(occ.field, occ.conjugated)
                for occ in term.fields
            ),
            key=repr,
        )
    )


def _term_vertex_fields(term: InteractionTerm):
    return tuple(_field_arg_from_occurrence(occ) for occ in term.fields)


def _normalize_field_filter_args(field_args, *, parameter_name: str) -> Optional[tuple[tuple[object, bool], ...]]:
    if field_args is None:
        return None
    try:
        return (_parse_field_arg(field_args),)
    except TypeError:
        pass

    try:
        items = tuple(field_args)
    except TypeError as exc:
        raise TypeError(
            f"`{parameter_name}` must be a field argument or an iterable of field arguments."
        ) from exc

    return tuple(_parse_field_arg(item) for item in items)


def _parsed_field_counter(parsed_fields) -> Counter:
    return Counter(_field_match_key(field, conjugated) for field, conjugated in parsed_fields)


def _counter_contains(counter: Counter, required: Counter) -> bool:
    return all(counter.get(key, 0) >= count for key, count in required.items())


def _vertex_signature_sort_key(field_args):
    parsed = tuple(_parse_field_arg(field_arg) for field_arg in field_args)
    return (
        len(field_args),
        _vertex_name_tuple(field_args),
        tuple(repr(_field_match_key(field, conjugated)) for field, conjugated in parsed),
    )


def _format_signature_names(signature_names: tuple[str, ...]) -> str:
    return ", ".join(signature_names)


# Sector tags exposed by ``vertex_signatures(...)`` and ``vertex_report(...)``.
# These are intentionally narrow: we only classify terms in ways that can be
# established directly from compiled metadata (field kinds, gauge-fixing label
# marker). Everything else degrades to ``"unknown"`` instead of guessing.
KNOWN_VERTEX_SECTORS: tuple[str, ...] = (
    "matter",
    "pure_gauge",
    "gauge_fixing",
    "ghost",
    "unknown",
)


def _classify_term_sector(term) -> str:
    """Return a conservative sector tag for one compiled ``InteractionTerm``.

    Classification rules, in order:
    1. Explicit ``term.sector`` provenance when present.
    2. Any field with ``kind == 'ghost'`` -> ``ghost``.
    3. All field occurrences are spin-1 self-conjugate vectors -> ``pure_gauge``.
    4. At least one matter (scalar or fermion) field -> ``matter``.
    5. Otherwise -> ``unknown``.

    This intentionally avoids label-text heuristics. It only uses explicit
    provenance or information reliably present on compiled outputs from
    ``compiler.gauge`` and on directly-constructed ``InteractionTerm``
    instances built from declared ``Field`` metadata.
    """

    explicit_sector = getattr(term, "sector", "")
    if explicit_sector in KNOWN_VERTEX_SECTORS:
        return explicit_sector

    fields = tuple(occ.field for occ in term.fields)
    if not fields:
        return "unknown"
    if any(getattr(field_obj, "kind", None) == "ghost" for field_obj in fields):
        return "ghost"
    if all(
        getattr(field_obj, "kind", None) == "vector"
        and getattr(field_obj, "self_conjugate", False)
        for field_obj in fields
    ):
        return "pure_gauge"
    if any(
        getattr(field_obj, "kind", None) in ("scalar", "fermion") for field_obj in fields
    ):
        return "matter"
    return "unknown"


@dataclass(frozen=True)
class VertexSignature:
    """One grouped interaction signature available in a compiled Lagrangian.

    ``term_count`` counts the compiled terms contributing to this signature in
    the current query context. For unfiltered listings this is the full grouped
    count. For ``sector=...`` listings it is the sector-local grouped count.
    """

    fields: tuple[object, ...]
    names: tuple[str, ...]
    arity: int
    term_count: int
    sectors: tuple[str, ...] = ()


@dataclass(frozen=True)
class VertexReport:
    """Structured, filterable report of available compiled interaction signatures.

    ``total_terms`` / ``total_signatures`` always describe the full compiled
    Lagrangian. ``signatures`` and ``matched_terms`` describe only the current
    filtered view.
    """

    signatures: tuple[VertexSignature, ...]
    total_terms: int
    total_signatures: int

    @property
    def matched_signatures(self) -> int:
        return len(self.signatures)

    @property
    def matched_terms(self) -> int:
        return sum(signature.term_count for signature in self.signatures)


@dataclass(init=False)
class CompiledLagrangian:
    """Collection of compiled ``InteractionTerm`` objects."""

    terms: tuple[InteractionTerm, ...] = ()
    parameters: tuple[object, ...] = ()
    _expanded_terms_cache: dict[tuple[object, ...], tuple[InteractionTerm, ...]] = dataclass_field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ("terms", "parameters") and hasattr(self, "_expanded_terms_cache"):
            object.__setattr__(self, "_expanded_terms_cache", {})

    def __init__(self, terms=(), parameters=()):
        self.terms = _normalize_interaction_terms_input(terms)
        self.parameters = tuple(parameters)
        self._expanded_terms_cache = {}

    @classmethod
    def from_item(cls, item) -> "CompiledLagrangian":
        if isinstance(item, CompiledLagrangian):
            return cls(terms=item.terms, parameters=item.parameters)
        return cls(terms=item)

    def __add__(self, other):
        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(
                terms=self.terms + (other,),
                parameters=self.parameters,
            )
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(
                terms=self.terms + other.terms,
                parameters=_merge_unique_metadata(self.parameters, other.parameters),
            )
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_compiled_lagrangian_context_error())
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(
                terms=(other,) + self.terms,
                parameters=self.parameters,
            )
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(
                terms=other.terms + self.terms,
                parameters=_merge_unique_metadata(other.parameters, self.parameters),
            )
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_compiled_lagrangian_context_error())
        return NotImplemented

    def __str__(self):
        if not self.terms:
            return "0"
        return " + ".join(str(term) for term in self.terms)

    def __getattr__(self, name):
        """Forward missing attributes/methods to the default Symbolica export.

        This lets a compiled Lagrangian behave like its Symbolica expression
        view for read-only algebraic manipulations such as ``expand()``,
        ``collect(...)``, ``match(...)``, or ``coefficient(...)`` without
        forcing callers to spell ``to_symbolica()`` first.

        The forwarded view always uses the default export options. If a caller
        needs non-default ``flavor_expand`` or ``derivative_style`` settings,
        they should call ``to_symbolica(...)`` explicitly.
        """

        expression = self.to_symbolica()
        if not hasattr(expression, name):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}."
            )
        return getattr(expression, name)

    def __dir__(self):
        """Expose Symbolica-expression methods in interactive completion."""

        return sorted(
            set(super().__dir__()) | set(dir(Expression))
        )

    def _expanded_terms(self, *, flavor_expand: FlavorExpandOption = False) -> tuple[InteractionTerm, ...]:
        flavor_expand = _normalize_flavor_expand_option(flavor_expand)
        if not flavor_expand:
            return self.terms

        cache_key = _flavor_expand_cache_key(flavor_expand)
        cached = self._expanded_terms_cache.get(cache_key)
        if cached is not None:
            return cached

        from .flavor import expand_flavor_terms

        selected_indices = (
            None
            if flavor_expand is _EXPAND_ALL_FLAVOR_INDICES
            else tuple(flavor_expand)
        )
        expanded_terms = expand_flavor_terms(
            self.terms,
            parameters=self.parameters,
            selected_indices=selected_indices,
        )
        self._expanded_terms_cache[cache_key] = expanded_terms
        return expanded_terms

    def _vertex_field_tuples(self, *, flavor_expand: FlavorExpandOption = False):
        vertices = {}
        for term in self._expanded_terms(flavor_expand=flavor_expand):
            key = _term_vertex_key(term)
            if key not in vertices:
                vertices[key] = _term_vertex_fields(term)
        return tuple(vertices.values())

    def _vertex_signature_entries(self, *, sector=None, flavor_expand: FlavorExpandOption = False):
        grouped = {}
        for term in self._expanded_terms(flavor_expand=flavor_expand):
            term_sector = _classify_term_sector(term)
            if sector is not None and term_sector != sector:
                continue
            key = _term_vertex_key(term)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = {
                    "fields": _term_vertex_fields(term),
                    "term_count": 1,
                    "sectors": {term_sector},
                }
            else:
                entry["term_count"] += 1
                entry["sectors"].add(term_sector)

        entries = []
        for entry in grouped.values():
            fields = entry["fields"]
            sectors = tuple(sorted(entry["sectors"]))
            entries.append(
                VertexSignature(
                    fields=fields,
                    names=_vertex_name_tuple(fields),
                    arity=len(fields),
                    term_count=entry["term_count"],
                    sectors=sectors,
                )
            )
        return tuple(sorted(entries, key=lambda entry: _vertex_signature_sort_key(entry.fields)))

    def _no_matching_interaction_terms_error(
        self,
        parsed_fields,
        *,
        max_signatures: int = 8,
        flavor_expand: FlavorExpandOption = False,
    ) -> ValueError:
        requested = ", ".join(
            _parsed_field_arg_name(field_obj, conjugated)
            for field_obj, conjugated in parsed_fields
        )
        available_signatures = self.vertex_signatures(flavor_expand=flavor_expand)

        lines = [f"No matching interaction terms for: {requested}.", "Available signatures:"]
        if not available_signatures:
            lines.append("  - (none)")
            return ValueError("\n".join(lines))

        displayed_signatures = available_signatures[:max_signatures]
        for signature in displayed_signatures:
            lines.append(f"  - {_format_signature_names(signature.names)}")

        omitted = len(available_signatures) - len(displayed_signatures)
        if omitted > 0:
            lines.append(f"  - ... (+ {omitted} more)")

        return ValueError("\n".join(lines))

    def vertex_signatures(
        self,
        *,
        arity=None,
        signature=None,
        contains_fields=None,
        sector=None,
        flavor_expand: FlavorExpandOption = False,
    ):
        """Enumerate grouped compiled interaction signatures without extracting vertices.

        Filters:
        - ``arity=...`` keeps only vertices with that many legs.
        - ``signature=(...)`` matches one exact field-content signature using the same
          field-argument syntax accepted by ``feynman_rule(...)``.
        - ``contains_fields=...`` keeps signatures containing at least the requested
          field multiset.
        - ``sector=...`` keeps only terms in the requested sector before
          regrouping them into signatures. Supported sectors are listed in
          ``KNOWN_VERTEX_SECTORS``: ``'matter'``, ``'pure_gauge'``,
          ``'gauge_fixing'``, ``'ghost'``, ``'unknown'``. Sector classification
          is intentionally narrow (see ``_classify_term_sector``); ambiguous
          terms degrade to ``'unknown'`` rather than guessing. In particular,
          ``VertexSignature.term_count`` in a sector-filtered query is the
          sector-local grouped count, not the aggregate grouped count across all
          sectors for the same field signature.
        """

        if sector is not None and sector not in KNOWN_VERTEX_SECTORS:
            raise ValueError(
                f"Unknown sector {sector!r}; expected one of "
                f"{KNOWN_VERTEX_SECTORS}."
            )

        exact_signature = _normalize_field_filter_args(signature, parameter_name="signature")
        contains_signature = _normalize_field_filter_args(
            contains_fields,
            parameter_name="contains_fields",
        )
        exact_counter = (
            _parsed_field_counter(exact_signature) if exact_signature is not None else None
        )
        contains_counter = (
            _parsed_field_counter(contains_signature)
            if contains_signature is not None
            else None
        )

        signatures = []
        for vertex_signature in self._vertex_signature_entries(
            sector=sector,
            flavor_expand=flavor_expand,
        ):
            if arity is not None and vertex_signature.arity != arity:
                continue

            parsed_fields = tuple(_parse_field_arg(field_arg) for field_arg in vertex_signature.fields)
            parsed_counter = _parsed_field_counter(parsed_fields)

            if exact_counter is not None and parsed_counter != exact_counter:
                continue
            if contains_counter is not None and not _counter_contains(parsed_counter, contains_counter):
                continue

            signatures.append(vertex_signature)

        return tuple(signatures)

    def vertex_report(
        self,
        *,
        arity=None,
        signature=None,
        contains_fields=None,
        sector=None,
        flavor_expand: FlavorExpandOption = False,
    ) -> VertexReport:
        """Return a structured summary of available compiled interaction signatures.

        ``total_terms`` and ``total_signatures`` refer to the full compiled
        Lagrangian. Filtered views affect ``signatures``, ``matched_signatures``,
        and ``matched_terms`` only.
        """

        all_signatures = self._vertex_signature_entries(flavor_expand=flavor_expand)
        filtered = self.vertex_signatures(
            arity=arity,
            signature=signature,
            contains_fields=contains_fields,
            sector=sector,
            flavor_expand=flavor_expand,
        )
        return VertexReport(
            signatures=filtered,
            total_terms=len(self.terms),
            total_signatures=len(all_signatures),
        )

    def validate(self):
        """Return structured diagnostics inferred from compiled interaction terms."""
        from .validation import validate_compiled_lagrangian

        return validate_compiled_lagrangian(self)

    def apply_operator(
        self,
        operator,
        *,
        flavor_expand: FlavorExpandOption = False,
        max_generated_terms: Optional[int] = None,
    ) -> "CompiledLagrangian":
        """Apply one runtime operator to every compiled interaction term.

        Returns a new ``CompiledLagrangian`` whose terms are the graded
        Leibniz expansion of ``operator`` over the current terms when
        ``operator`` is a ``FieldOperator``, or the direct whole-term
        rewrite when it is a ``TermOperator``. The action is implemented
        at the ``InteractionTerm`` level (see ``lagrangian.operator_action``);
        the authoritative ordered representation is preserved.

        ``flavor_expand`` mirrors ``feynman_rule``: pass
        ``True`` (or a specific flavor index) to act on the
        flavor-expanded view of the terms rather than the flavor-generic
        ones. Parameters are forwarded unchanged.
        """

        from lagrangian.operator_action import apply_operator as apply_runtime_operator

        source_terms = self._expanded_terms(flavor_expand=flavor_expand)
        return CompiledLagrangian(
            terms=apply_runtime_operator(
                source_terms,
                operator,
                max_generated_terms=max_generated_terms,
            ),
            parameters=self.parameters,
        )

    def apply_operators(
        self,
        *operators,
        flavor_expand: FlavorExpandOption = False,
        max_generated_terms: Optional[int] = None,
    ) -> "CompiledLagrangian":
        """Apply several runtime operators in strict left-to-right order."""

        current = self if not flavor_expand else CompiledLagrangian(
            terms=self._expanded_terms(flavor_expand=flavor_expand),
            parameters=self.parameters,
        )
        for operator in operators:
            current = current.apply_operator(
                operator,
                max_generated_terms=max_generated_terms,
            )
        return current

    def operator_bracket(
        self,
        left,
        right,
        *,
        graded: bool = True,
        flavor_expand: FlavorExpandOption = False,
        max_generated_terms: Optional[int] = None,
    ) -> "CompiledLagrangian":
        """Return ``left(right(L)) - (-1)^(|left||right|) right(left(L))``."""

        left_after_right = self.apply_operators(
            right,
            left,
            flavor_expand=flavor_expand,
            max_generated_terms=max_generated_terms,
        )
        right_after_left = self.apply_operators(
            left,
            right,
            flavor_expand=flavor_expand,
            max_generated_terms=max_generated_terms,
        )
        sign = 1
        if graded:
            sign = 1 if (_operator_parity(left) * _operator_parity(right)) % 2 == 0 else -1
        return CompiledLagrangian(
            terms=left_after_right.terms + _scaled_terms(right_after_left.terms, -sign),
            parameters=_merge_unique_metadata(left_after_right.parameters, right_after_left.parameters),
        )

    def operator_anticommutator(
        self,
        left,
        right,
        *,
        flavor_expand: FlavorExpandOption = False,
        max_generated_terms: Optional[int] = None,
    ) -> "CompiledLagrangian":
        """Return ``left(right(L)) + (-1)^(|left||right|) right(left(L))``."""

        left_after_right = self.apply_operators(
            right,
            left,
            flavor_expand=flavor_expand,
            max_generated_terms=max_generated_terms,
        )
        right_after_left = self.apply_operators(
            left,
            right,
            flavor_expand=flavor_expand,
            max_generated_terms=max_generated_terms,
        )
        sign = 1 if (_operator_parity(left) * _operator_parity(right)) % 2 == 0 else -1
        return CompiledLagrangian(
            terms=left_after_right.terms + _scaled_terms(right_after_left.terms, sign),
            parameters=_merge_unique_metadata(left_after_right.parameters, right_after_left.parameters),
        )

    def ibp_normal_form(self) -> "CompiledLagrangian":
        """Return the scalar IBP normal form of the compiled Lagrangian."""

        from lagrangian.ibp import ibp_normal_form

        return ibp_normal_form(self)

    def to_symbolica(
        self,
        *,
        flavor_expand: FlavorExpandOption = False,
        derivative_style: str = "partiald",
        coordinate_map=None,
    ):
        """Render the compiled Lagrangian as a single Symbolica expression.

        Display / simplification only -- Symbolica multiplication is
        commutative, so fermion / ghost product ordering is not preserved.
        Use the ordered ``terms`` tuple for ordering-sensitive operations.

        ``flavor_expand`` mirrors ``feynman_rule``: pass
        ``True`` (or a specific flavor index, or an iterable of flavor
        indices) to export the flavor-expanded view of the terms rather
        than the flavor-generic ones.
        """

        from lagrangian.symbolica_export import lagrangian_to_symbolica

        return lagrangian_to_symbolica(
            self,
            flavor_expand=flavor_expand,
            derivative_style=derivative_style,
            coordinate_map=coordinate_map,
        )

    def pattern_matches(
        self,
        pattern,
        *,
        flavor_expand: FlavorExpandOption = False,
        derivative_style: str = "partiald",
        coordinate_map=None,
        expand: bool = False,
        deduplicate: bool = True,
    ):
        """Enumerate top-level wildcard matches in the Symbolica export."""

        from lagrangian.symbolica_export import pattern_matches

        return pattern_matches(
            self.to_symbolica(
                flavor_expand=flavor_expand,
                derivative_style=derivative_style,
                coordinate_map=coordinate_map,
            ),
            pattern,
            expand=expand,
            deduplicate=deduplicate,
        )

    def pattern_coefficient(
        self,
        pattern,
        *,
        flavor_expand: FlavorExpandOption = False,
        derivative_style: str = "partiald",
        coordinate_map=None,
        expand: bool = False,
        deduplicate: bool = True,
    ):
        """Return the summed residual coefficient of a wildcard pattern."""

        from lagrangian.symbolica_export import pattern_coefficient

        return pattern_coefficient(
            self.to_symbolica(
                flavor_expand=flavor_expand,
                derivative_style=derivative_style,
                coordinate_map=coordinate_map,
            ),
            pattern,
            expand=expand,
            deduplicate=deduplicate,
        )

    def feynman_rule(
        self,
        *fields,
        momenta=None,
        arity=None,
        select=None,
        simplify=True,
        key_format="names",
        include_delta: bool = False,
        strip_externals: bool = True,
        simplify_gamma: bool = False,
        flavor_expand: FlavorExpandOption = False,
    ):
        """Compute Feynman vertex rules.

        With explicit fields, return the vertex rule for that field content.
        With no fields, return a mapping from available vertex signatures to
        their corresponding vertex rules. By default, zero-argument keys are
        tuples of readable field names. Use ``key_format="fields"`` to return
        tuples of ``Field`` / ``Field.bar`` objects instead.

        Zero-argument extraction also supports:
        - ``arity=...`` to keep only vertices with that many legs.
        - ``select=[(...), ...]`` to restrict extraction to exact field tuples.
        """

        from symbolic.vertex_engine import simplify_vertex, vertex_factor

        if key_format not in ("names", "fields"):
            raise ValueError("key_format must be either 'names' or 'fields'.")
        flavor_expand = _normalize_flavor_expand_option(flavor_expand)

        parsed = [_parse_field_arg(f) for f in fields]
        n = len(parsed)
        if n == 0:
            if momenta is not None:
                raise ValueError("`momenta=` is only supported for explicit vertex extraction.")
            if select is None:
                vertex_fields_list = self._vertex_field_tuples(flavor_expand=flavor_expand)
            else:
                vertex_fields_list = tuple(tuple(vertex_fields) for vertex_fields in select)

            if arity is not None:
                vertex_fields_list = tuple(
                    vertex_fields
                    for vertex_fields in vertex_fields_list
                    if len(vertex_fields) == arity
                )

            rules_by_field = {
                tuple(vertex_fields): self.feynman_rule(
                    *vertex_fields,
                    simplify=simplify,
                    include_delta=include_delta,
                    strip_externals=strip_externals,
                    simplify_gamma=simplify_gamma,
                    flavor_expand=flavor_expand,
                )
                for vertex_fields in vertex_fields_list
            }
            if key_format == "fields":
                return rules_by_field

            rules_by_name = {}
            for vertex_fields, expression in rules_by_field.items():
                name_key = _vertex_name_tuple(vertex_fields)
                if name_key in rules_by_name:
                    raise ValueError(
                        "Name-keyed vertex signatures are ambiguous; "
                        "use key_format='fields'."
                    )
                rules_by_name[name_key] = expression
            return rules_by_name

        if arity is not None:
            raise ValueError("`arity=` is only supported for zero-argument vertex extraction.")
        if select is not None:
            raise ValueError("`select=` is only supported for zero-argument vertex extraction.")

        if momenta is None:
            momenta_list = [S(f"q{k + 1}") for k in range(n)]
        else:
            momenta_list = list(momenta)
        if len(momenta_list) != n:
            raise ValueError(f"Expected {n} momenta, got {len(momenta_list)}.")

        idx_counter = [1]
        legs = []
        for k, (fld, conj) in enumerate(parsed):
            labels = _auto_leg_labels(fld, idx_counter)
            legs.append(
                ExternalLeg(
                    field=fld,
                    momentum=momenta_list[k],
                    conjugated=conj,
                    labels=labels,
                )
            )

        matching = [
            term
            for term in self._expanded_terms(flavor_expand=flavor_expand)
            if _term_matches_fields(term, parsed)
        ]
        if not matching:
            raise self._no_matching_interaction_terms_error(
                parsed,
                flavor_expand=flavor_expand,
            )

        x = S("x_")
        d = S("d")
        total = Expression.num(0)
        for term in matching:
            total += vertex_factor(
                interaction=term,
                external_legs=legs,
                x=x,
                d=d,
                strip_externals=strip_externals,
                include_delta=include_delta,
            )

        if simplify:
            species_map = None
            unique_species = list(dict.fromkeys(
                fld.species_for(conj) for fld, conj in parsed
            ))
            if len(unique_species) > 1:
                species_map = {sp: sp for sp in unique_species}
            total = simplify_vertex(
                total,
                species_map=species_map,
                external_legs=legs,
                simplify_gamma=simplify_gamma,
            )

        return total


@dataclass(init=False)
class Lagrangian(CompiledLagrangian):
    """User-facing extraction object for local, already-expanded terms.

    Use ``Lagrangian(...)`` when the interaction is already written as a local
    operator built directly from fields and local placeholders such as
    ``PartialD(...)``, ``Gamma(...)``, ``Metric(...)``, ``T(...)``, and
    ``StructureConstant(...)``.

    ``Lagrangian(...)`` does not look up gauge-group metadata and does not
    perform gauge-sector compilation. Metadata-dependent source declarations
    such as ``CovD(...)``, ``FieldStrength(...)``, ``GaugeFixing(...)``, and
    ``GhostLagrangian(...)`` belong in ``Model(..., lagrangian_decl=...)`` and
    should be compiled through ``model.lagrangian()``.
    """

    source_terms: tuple[object, ...] = ()

    def __init__(self, *items, terms=None, lagrangian_decl=None):
        self._expanded_terms_cache = {}
        if terms is not None and (items or lagrangian_decl is not None):
            raise TypeError(
                "Lagrangian accepts either `terms=` or declarative input, not both."
            )

        if lagrangian_decl is not None:
            if items:
                raise TypeError(
                    "Pass declarative input either positionally or via "
                    "`lagrangian_decl=`, not both."
                )
            items = (lagrangian_decl,)

        if terms is not None:
            normalized_terms = _normalize_interaction_terms_input(terms)
            self.terms = normalized_terms
            self.source_terms = normalized_terms
            return

        if not items:
            self.terms = ()
            self.source_terms = ()
            return

        source_terms = _normalize_lagrangian_source_terms(items)
        self.terms = tuple(
            _lower_standalone_lagrangian_source_term(term)
            for term in source_terms
        )
        self.source_terms = source_terms

    @classmethod
    def from_item(cls, item) -> "Lagrangian":
        return cls(item)

    def __add__(self, other):
        terms = _standalone_lagrangian_source_terms_from_item(other)
        if terms is not None:
            return Lagrangian(*self.source_terms, *terms)
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_standalone_lagrangian_context_error())
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        terms = _standalone_lagrangian_source_terms_from_item(other)
        if terms is not None:
            return Lagrangian(*terms, *self.source_terms)
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_standalone_lagrangian_context_error())
        return NotImplemented

    def __str__(self):
        if not self.source_terms:
            return "0"
        return " + ".join(str(term) for term in self.source_terms)


@dataclass(frozen=True)
class DiracKineticTerm:
    """Model-level declaration for ``psibar i gamma^mu D_mu psi``."""

    field: object
    gauge_group: object = None
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


@dataclass(frozen=True)
class ComplexScalarKineticTerm:
    """Model-level declaration for ``(D_mu phi)^dagger (D^mu phi)``."""

    field: object
    gauge_group: object = None
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


@dataclass(frozen=True)
class GaugeKineticTerm:
    """Model-level declaration for ``-1/4 F_{mu nu} F^{mu nu}``."""

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


@dataclass(frozen=True)
class GaugeFixingTerm:
    """Model-level declaration for ``-(1/2 xi) (partial.A)^2``."""

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


@dataclass(frozen=True)
class GhostTerm:
    """Model-level declaration for the ordinary Faddeev-Popov ghost sector."""

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


@dataclass(frozen=True)
class DeclaredLagrangian:
    """Source-level declaration container for ``Model(lagrangian_decl=...)``."""

    source_terms: tuple[object, ...] = ()

    @classmethod
    def from_item(cls, item) -> "DeclaredLagrangian":
        terms = _declared_source_terms_from_item(item)
        if terms is None:
            raise TypeError(
                f"Cannot build DeclaredLagrangian from {type(item).__name__}"
            )
        return cls(source_terms=terms)

    def __add__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=self.source_terms + terms)

    def __radd__(self, other):
        if other == 0:
            return self
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(source_terms=terms + self.source_terms)

    def __sub__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(
            source_terms=self.source_terms + tuple(-term for term in terms)
        )

    def __rsub__(self, other):
        terms = _declared_source_terms_from_item(other)
        if terms is None:
            return NotImplemented
        return DeclaredLagrangian(
            source_terms=terms + tuple(-term for term in self.source_terms)
        )

    def __str__(self):
        if not self.source_terms:
            return "0"
        return " + ".join(str(term) for term in self.source_terms)

CovariantTerm = Union[DiracKineticTerm, ComplexScalarKineticTerm]
