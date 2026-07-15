"""Assembly entry point for the SMEFT dimension-six Green basis.

:func:`build_smeft` builds the shared unbroken Standard Model foundation and
compiles a chosen set of dimension-six operator sectors into a single
:class:`CompiledLagrangian`, so a user can work with a whole sector (or the
complete basis) without hand-expanding it.  Every operator remains individually
reachable through the registry (:func:`~.registry.get_operator`,
:func:`~.registry.operators_in`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from feynpy import CompiledLagrangian

from .registry import Operator, operators_in
from .sm_core import SMEFTCore, build_sm_core
from .tensors import Poly


@dataclass(frozen=True)
class SMEFT:
    """A compiled selection of the Green basis together with its foundation."""

    core: SMEFTCore
    operators: tuple[Operator, ...]
    lagrangian: CompiledLagrangian

    @property
    def renormalizable(self) -> CompiledLagrangian:
        return self.core.renormalizable


def select_operators(
    *,
    sectors: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    tables: Optional[Sequence[int]] = None,
    include_blocked: bool = False,
) -> tuple[Operator, ...]:
    """Return the registered operators matching the given filters.

    ``sectors`` / ``types`` / ``tables`` are inclusive filters (``None`` means
    "any").  Operators whose status is ``"blocked"`` (the charge-conjugation
    C-chains of Tables 8-9) are excluded unless ``include_blocked`` is set.
    """

    result: list[Operator] = []
    for op in operators_in():
        if sectors is not None and op.sector not in sectors:
            continue
        if types is not None and op.otype not in types:
            continue
        if tables is not None and op.table not in tables:
            continue
        if op.status == "blocked" and not include_blocked:
            continue
        result.append(op)
    return tuple(result)


def build_smeft(
    *,
    core: Optional[SMEFTCore] = None,
    sectors: Optional[Sequence[str]] = None,
    types: Optional[Sequence[str]] = None,
    tables: Optional[Sequence[int]] = None,
) -> SMEFT:
    """Assemble and compile a chosen part of the Green basis.

    By default every implementable operator (all sectors, all types) is
    compiled.  Restrict the selection with ``sectors`` (``"bosonic"``,
    ``"two_fermion"``, ``"four_fermion"``), ``types`` (``"physical"``,
    ``"redundant"``, ``"evanescent"``) and/or ``tables`` (1-9).  Each operator
    is multiplied by its Wilson coefficient (see :mod:`.wilson`); the overall
    ``1/Lambda^2`` is left implicit.
    """

    core = core or build_sm_core()
    operators = select_operators(sectors=sectors, types=types, tables=tables)

    # Compile operator-by-operator so non-Hermitian sectors can add their
    # Hermitian conjugates at the compiled level.
    lagrangian = CompiledLagrangian()
    for op in operators:
        lagrangian = lagrangian + op.lagrangian(core)
    return SMEFT(core=core, operators=operators, lagrangian=lagrangian)


__all__ = ("SMEFT", "select_operators", "build_smeft")
