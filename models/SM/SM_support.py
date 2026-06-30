"""Private support routines for the packaged Standard Model builder."""

from __future__ import annotations

from dataclasses import replace

from symbolica import Expression, S

from feynpy import (
    CompiledLagrangian,
    Model,
    PartialD,
    WEAK_ADJ_INDEX,
    WEAK_FUND_INDEX,
)
from symbolic.spenso_structures import (
    weak_eps2,
    weak_gauge_generator,
    weak_structure_constant,
)
from symbolic.vertex_engine import I

ONE = Expression.num(1)
TWO = Expression.num(2)
THREE = Expression.num(3)
FOUR = Expression.num(4)
SIX = Expression.num(6)
HALF = ONE / TWO
INV_SQRT2 = HALF**HALF


def is_zero(value) -> bool:
    return value == 0 or (
        hasattr(value, "expand")
        and value.expand().to_canonical_string() == "0"
    )


def diagonal_components(prefix: str) -> dict[tuple[int, int], object]:
    return {
        (row, column): (
            S(f"{prefix}{row}") if row == column else Expression.num(0)
        )
        for row in range(1, 4)
        for column in range(1, 4)
    }


def ckm_components() -> dict[tuple[int, int], object]:
    return {
        (row, column): S(f"CKM{row}{column}")
        for row in range(1, 4)
        for column in range(1, 4)
    }


def ckm_dagger_components() -> dict[tuple[int, int], object]:
    return {
        (column, row): S(f"CKMConj{row}{column}")
        for row in range(1, 4)
        for column in range(1, 4)
    }


def concrete_tensor(builder, *values):
    labels = tuple(S(f"component_{position}") for position in range(len(values)))
    expression = builder(*labels)
    for label, value in zip(labels, values):
        expression = expression.replace(label, Expression.num(value))
    return expression


def real_conjugate(value, *real_symbols):
    result = value.conj() if hasattr(value, "conj") else value
    for symbol in real_symbols:
        result = result.replace(symbol.conj(), symbol)
    return result


def parameter_value_or_symbol(parameter):
    return parameter.value if parameter.value is not None else parameter.symbol


def apply_parameter_substitutions(
    lagrangian: CompiledLagrangian,
    substitutions: tuple[tuple[object, object], ...],
) -> CompiledLagrangian:
    if not substitutions:
        return lagrangian

    terms = []
    for term in lagrangian.terms:
        coupling = term.coupling
        for symbol, definition in substitutions:
            coupling = coupling.replace(symbol, definition)
        terms.append(replace(term, coupling=coupling.cancel().expand()))
    return CompiledLagrangian(
        terms=tuple(terms),
        parameters=lagrangian.parameters,
    )


def electroweak_generators_and_vacuum_images(parameters):
    g1 = parameters.g1.symbol
    g2 = parameters.g2.symbol
    vev = parameters.vev.symbol
    zero = Expression.num(0)

    generators = (
        (
            (-I * g1 * HALF, zero),
            (zero, -I * g1 * HALF),
        ),
        (
            (zero, -I * g2 * HALF),
            (-I * g2 * HALF, zero),
        ),
        (
            (zero, -g2 * HALF),
            (g2 * HALF, zero),
        ),
        (
            (-I * g2 * HALF, zero),
            (zero, I * g2 * HALF),
        ),
    )
    vacuum = (zero, vev * INV_SQRT2)
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


def electroweak_omega_coefficients(parameters):
    _generators, vacuum_images = electroweak_generators_and_vacuum_images(
        parameters
    )
    g1 = parameters.g1.symbol
    g2 = parameters.g2.symbol
    vev = parameters.vev.symbol
    real_symbols = (g1, g2, vev)

    coefficients: list[tuple[tuple[str, int, object], ...]] = []
    for gauge_index in range(4):
        components: list[tuple[str, int, object]] = []
        for component in range(2):
            phi_coefficient = -real_conjugate(
                vacuum_images[gauge_index][component],
                *real_symbols,
            )
            phibar_coefficient = -vacuum_images[gauge_index][component]
            if not is_zero(phi_coefficient):
                components.append(("phi", component + 1, phi_coefficient))
            if not is_zero(phibar_coefficient):
                components.append(("phibar", component + 1, phibar_coefficient))
        coefficients.append(tuple(components))
    return tuple(coefficients)


def electroweak_xi_matrices(parameters):
    sw = parameters.sw.symbol
    cw = parameters.cw.symbol
    xiA = parameter_value_or_symbol(parameters.xiA)
    xiZ = parameter_value_or_symbol(parameters.xiZ)
    xiW = parameter_value_or_symbol(parameters.xiW)

    xi_inverse = {
        (0, 0): cw**2 / xiA + sw**2 / xiZ,
        (0, 3): cw * sw * (ONE / xiA - ONE / xiZ),
        (1, 1): ONE / xiW,
        (2, 2): ONE / xiW,
        (3, 0): cw * sw * (ONE / xiA - ONE / xiZ),
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


def electroweak_gauge_basis_field(fields, *, gauge_index: int, lorentz_label):
    if gauge_index == 0:
        return fields.B(lorentz_label)
    return fields.Wi(lorentz_label, Expression.num(gauge_index))


def electroweak_scalar_component(fields, *, kind: str, component: int):
    if kind == "phi":
        return fields.Phi(Expression.num(component))
    return fields.Phi.bar(Expression.num(component))


def standard_model_weak_tensor_components() -> dict[object, object]:
    """Return explicit SU(2) tensor components used during weak unfolding."""

    components: dict[object, object] = {}
    pauli_over_two = {
        (1, 1, 1): 0,
        (1, 1, 2): HALF,
        (1, 2, 1): HALF,
        (1, 2, 2): 0,
        (2, 1, 1): 0,
        (2, 1, 2): -I * HALF,
        (2, 2, 1): I * HALF,
        (2, 2, 2): 0,
        (3, 1, 1): HALF,
        (3, 1, 2): 0,
        (3, 2, 1): 0,
        (3, 2, 2): -HALF,
    }
    for labels, value in pauli_over_two.items():
        components[concrete_tensor(weak_gauge_generator, *labels)] = value

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
                    concrete_tensor(
                        weak_structure_constant,
                        left,
                        middle,
                        right,
                    )
                ] = Expression.num(value)

    for left in range(1, 3):
        for right in range(1, 3):
            value = (
                1
                if (left, right) == (1, 2)
                else -1
                if (left, right) == (2, 1)
                else 0
            )
            components[concrete_tensor(weak_eps2, left, right)] = (
                Expression.num(value)
            )
    return components


def electroweak_scalar_ghost_lagrangian(fields, parameters):
    """Faddeev-Popov scalar term in the electroweak gauge basis."""

    zero = Expression.num(0)
    generators, vacuum_images = electroweak_generators_and_vacuum_images(
        parameters
    )
    _xi_inverse, xi_matrix = electroweak_xi_matrices(parameters)
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
                if is_zero(xi_coefficient):
                    continue
                for component in range(2):
                    phi_coefficient = -sum(
                        (
                            real_conjugate(
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
                            real_conjugate(
                                generators[right][row][component],
                                *real_symbols,
                            )
                            * vacuum_images[mixed_left][row]
                            for row in range(2)
                        ),
                        zero,
                    )
                    if not is_zero(phi_coefficient):
                        lagrangian += (
                            xi_coefficient
                            * phi_coefficient
                            * antighosts[left]
                            * ghosts[right]
                            * fields.Phi(Expression.num(component + 1))
                        )
                    if not is_zero(phibar_coefficient):
                        lagrangian += (
                            xi_coefficient
                            * phibar_coefficient
                            * antighosts[left]
                            * ghosts[right]
                            * fields.Phi.bar(Expression.num(component + 1))
                        )
    return lagrangian


def electroweak_rxi_gauge_fixing_lagrangian(fields, parameters):
    """Electroweak R_xi gauge fixing in the gauge basis."""

    mu = S("mu")
    nu = S("nu")
    zero = Expression.num(0)
    xi_inverse, xi_matrix = electroweak_xi_matrices(parameters)
    omega_coefficients = electroweak_omega_coefficients(parameters)

    lagrangian = None

    def add(term):
        nonlocal lagrangian
        lagrangian = term if lagrangian is None else lagrangian + term

    for (left, right), coefficient in xi_inverse.items():
        add(
            -coefficient
            * HALF
            * PartialD(
                electroweak_gauge_basis_field(
                    fields,
                    gauge_index=left,
                    lorentz_label=mu,
                ),
                mu,
            )
            * PartialD(
                electroweak_gauge_basis_field(
                    fields,
                    gauge_index=right,
                    lorentz_label=nu,
                ),
                nu,
            )
        )

    for gauge_index, components in enumerate(omega_coefficients):
        gauge_field = electroweak_gauge_basis_field(
            fields,
            gauge_index=gauge_index,
            lorentz_label=mu,
        )
        for kind, component, coefficient in components:
            add(
                coefficient
                * PartialD(
                    electroweak_scalar_component(
                        fields,
                        kind=kind,
                        component=component,
                    ),
                    mu,
                )
                * gauge_field
            )

    for (left, right), coefficient in xi_matrix.items():
        if is_zero(coefficient):
            continue
        for left_kind, left_component, left_coefficient in omega_coefficients[left]:
            left_scalar = electroweak_scalar_component(
                fields,
                kind=left_kind,
                component=left_component,
            )
            for (
                right_kind,
                right_component,
                right_coefficient,
            ) in omega_coefficients[right]:
                right_scalar = electroweak_scalar_component(
                    fields,
                    kind=right_kind,
                    component=right_component,
                )
                add(
                    -coefficient
                    * HALF
                    * left_coefficient
                    * right_coefficient
                    * left_scalar
                    * right_scalar
                )

    return zero if lagrangian is None else lagrangian


def compile_source_piece(
    lagrangian_decl,
    *,
    name: str,
    gauge_groups: tuple,
    source_fields: tuple,
    all_parameters: tuple,
    transformations: tuple,
    real_symbols: tuple,
    coupling_substitutions: tuple,
    sector: str | None = None,
    origin: str = "",
) -> CompiledLagrangian:
    if not lagrangian_decl.source_terms:
        return CompiledLagrangian(parameters=all_parameters)

    source_piece = Model(
        name=f"{name} source piece",
        gauge_groups=gauge_groups,
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
        real_symbols=real_symbols,
    )
    broken_piece = broken_piece.simplify_parameter_identities()
    broken_piece = apply_parameter_substitutions(
        broken_piece,
        coupling_substitutions,
    )
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


__all__ = (
    "FOUR",
    "HALF",
    "INV_SQRT2",
    "ONE",
    "SIX",
    "THREE",
    "TWO",
    "ckm_components",
    "ckm_dagger_components",
    "compile_source_piece",
    "diagonal_components",
    "electroweak_rxi_gauge_fixing_lagrangian",
    "electroweak_scalar_ghost_lagrangian",
    "parameter_value_or_symbol",
    "standard_model_weak_tensor_components",
)
