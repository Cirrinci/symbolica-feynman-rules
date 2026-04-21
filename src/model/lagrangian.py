"""User-facing Lagrangian and source-term classes."""

from __future__ import annotations

from dataclasses import dataclass

from symbolica import Expression, S

from .interactions import (
    ExternalLeg,
    InteractionTerm,
    _auto_leg_labels,
    _parse_field_arg,
    _standalone_lagrangian_context_error,
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

@dataclass(init=False)
class Lagrangian:
    """Collection of interaction terms with a single ``feynman_rule()`` entry point.

    Mirrors the FeynRules workflow: declare Lagrangian pieces, compose them
    with ``+``, then extract vertex factors by specifying external fields.

    Standalone ``Lagrangian(...)`` accepts local declarative monomials directly,
    so users do not need to build ``InteractionTerm`` objects by hand for
    operators such as ``lam * Phi * Phi * Phi * Phi`` or
    ``g * PartialD(Phi, mu) * PartialD(Phi, mu) * Phi * Phi``.

    Source forms that require model metadata, such as ``CovD(...)``,
    ``FieldStrength(...)``, ``GaugeFixing(...)``, or ``GhostLagrangian(...)``,
    still belong in ``Model(lagrangian_decl=...)`` and are compiled there.

    Example::

        L = Lagrangian(lam * Phi * Phi * Phi * Phi)
        vertex = L.feynman_rule(Phi.bar, Phi, A)
    """
    terms: tuple[InteractionTerm, ...] = ()
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

    def feynman_rule(self, *fields, momenta=None, simplify=True):
        """Compute the Feynman vertex rule for the given external fields.

        Conventions:
        - leg order = argument order
        - momenta default to q1, q2, q3, ...
        - open indices are labeled i1, i2, i3, ... sequentially across legs

        Parameters
        ----------
        *fields : Field, tuple[Field, bool], or ConjugateField
            External fields in leg order.  Use ``field.bar`` or ``(field, True)``
            for conjugated fields (e.g. ``Phi.bar``).
        momenta : list of expressions, optional
            Override the default q1, q2, ... momentum assignment.  Each entry
            can be an algebraic expression (e.g. ``p3 - p6``).
        simplify : bool
            If True (default), apply ``simplify_vertex`` to the result.

        Returns
        -------
        Expression
            The summed, stripped Feynman vertex factor with ``(2 pi)^d Delta``
            momentum conservation.
        """
        from symbolic.vertex_engine import simplify_vertex, vertex_factor
        from symbolica import Expression

        parsed = [_parse_field_arg(f) for f in fields]
        n = len(parsed)
        if n == 0:
            raise ValueError("At least one external field is required.")

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
            legs.append(ExternalLeg(
                field=fld,
                momentum=momenta_list[k],
                conjugated=conj,
                labels=labels,
            ))

        matching = [t for t in self.terms if _term_matches_fields(t, parsed)]
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
            total = simplify_vertex(total, species_map=species_map, external_legs=legs)

        return total


# ---------------------------------------------------------------------------
# Convention-fixed kinetic terms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiracKineticTerm:
    """Model-level declaration for ``psibar i gamma^mu D_mu psi``.

    The current compiler expands only the gauge-interaction part of this term.
    If ``gauge_group`` is omitted, the compiler infers the unique applicable
    gauge group from the model metadata.
    """
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
    """Model-level declaration for ``(D_mu phi)^dagger (D^mu phi)``.

    The current compiler expands only the gauge-interaction part of this term.
    If ``gauge_group`` is omitted, the compiler infers the unique applicable
    gauge group from the model metadata.
    """
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
    """Model-level declaration for ``-1/4 F_{mu nu} F^{mu nu}``.

    ``gauge_group`` is required because the gauge field and non-abelian
    structure constants are properties of the group declaration, not of a
    separate matter field.
    """
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
    """Model-level declaration for ``-(1/2 xi) (partial.A)^2``.

    This covers the ordinary unbroken linear covariant gauge-fixing term for one
    declared gauge group. ``xi`` is the usual gauge-fixing parameter.
    """
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
    """Model-level declaration for the ordinary Faddeev-Popov ghost sector.

    The current implementation covers the unbroken non-abelian linear-covariant
    gauge case. The corresponding ghost field is resolved from the parent gauge
    group's ``ghost_field`` metadata.
    """
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
    """User-facing declarative Lagrangian built from fields and covariant derivatives.

    This is a source-level declaration container. Terms are preserved in the
    original FeynRules-style form (`CovD(...)`, `FieldStrength(...)`,
    `GaugeFixing(...)`, `GhostLagrangian(...)`, etc.) and compiled one by one
    when `Model.lagrangian()` is built. Canonical declarative ``CovD(...)``
    monomials compile to the full operator: free bilinear partial term plus
    gauge-interaction pieces.
    """
    source_terms: tuple[object, ...] = ()

    @classmethod
    def from_item(cls, item) -> "DeclaredLagrangian":
        terms = _declared_source_terms_from_item(item)
        if terms is None:
            raise TypeError(f"Cannot build DeclaredLagrangian from {type(item).__name__}")
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


CovariantTerm = (
    DiracKineticTerm
    | ComplexScalarKineticTerm
)
