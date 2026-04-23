"""Compiled, standalone, and declarative Lagrangian containers."""

from __future__ import annotations

from dataclasses import dataclass

from symbolica import Expression, S

from .interactions import (
    ExternalLeg,
    InteractionTerm,
    _auto_leg_labels,
    _field_match_key,
    _parse_field_arg,
    _term_matches_fields,
)


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


@dataclass(init=False)
class CompiledLagrangian:
    """Collection of compiled ``InteractionTerm`` objects."""

    terms: tuple[InteractionTerm, ...] = ()

    def __init__(self, terms=()):
        self.terms = _normalize_interaction_terms_input(terms)

    @classmethod
    def from_item(cls, item) -> "CompiledLagrangian":
        return cls(terms=item)

    def __add__(self, other):
        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(terms=self.terms + (other,))
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(terms=self.terms + other.terms)
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_compiled_lagrangian_context_error())
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, InteractionTerm):
            return CompiledLagrangian(terms=(other,) + self.terms)
        if isinstance(other, CompiledLagrangian):
            return CompiledLagrangian(terms=other.terms + self.terms)
        if _declared_source_terms_from_item(other) is not None:
            raise ValueError(_compiled_lagrangian_context_error())
        return NotImplemented

    def __str__(self):
        if not self.terms:
            return "0"
        return " + ".join(str(term) for term in self.terms)

    def _vertex_field_tuples(self):
        vertices = {}
        for term in self.terms:
            key = _term_vertex_key(term)
            if key not in vertices:
                vertices[key] = _term_vertex_fields(term)
        return tuple(vertices.values())

    def feynman_rule(self, *fields, momenta=None, simplify=True, key_format="names"):
        """Compute Feynman vertex rules.

        With explicit fields, return the vertex rule for that field content.
        With no fields, return a mapping from available vertex signatures to
        their corresponding vertex rules. By default, zero-argument keys are
        tuples of readable field names. Use ``key_format="fields"`` to return
        tuples of ``Field`` / ``Field.bar`` objects instead.
        """

        from symbolic.vertex_engine import simplify_vertex, vertex_factor

        if key_format not in ("names", "fields"):
            raise ValueError("key_format must be either 'names' or 'fields'.")

        parsed = [_parse_field_arg(f) for f in fields]
        n = len(parsed)
        if n == 0:
            if momenta is not None:
                raise ValueError("`momenta=` is only supported for explicit vertex extraction.")
            rules_by_field = {
                vertex_fields: self.feynman_rule(*vertex_fields, simplify=simplify)
                for vertex_fields in self._vertex_field_tuples()
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

        matching = [term for term in self.terms if _term_matches_fields(term, parsed)]
        if not matching:
            desc = ", ".join(
                f"{fld.name}{'bar' if conj else ''}" for fld, conj in parsed
            )
            raise ValueError(f"No matching interaction terms for: {desc}")

        x = S("x_")
        d = S("d")
        total = Expression.num(0)
        for term in matching:
            total += vertex_factor(
                interaction=term,
                external_legs=legs,
                x=x,
                d=d,
                strip_externals=True,
                include_delta=True,
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
            )

        return total


@dataclass(init=False)
class Lagrangian(CompiledLagrangian):
    """User-facing extraction object with a convenience local source front door.

    ``Lagrangian(...)`` accepts metadata-free local declarations directly.
    Declarations that require model metadata still belong in
    ``Model(lagrangian_decl=...)``.
    """

    source_terms: tuple[object, ...] = ()

    def __init__(self, *items, terms=None, lagrangian_decl=None):
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

    def __str__(self):
        if not self.source_terms:
            return "0"
        return " + ".join(str(term) for term in self.source_terms)

CovariantTerm = DiracKineticTerm | ComplexScalarKineticTerm
