"""Wilson coefficients for the SMEFT Green basis.

Normalization convention
-------------------------
Each dimension-six operator ``O_i`` enters the Lagrangian as

    L_SMEFT = L_SM + sum_i  C_i O_i ,

with one Wilson coefficient ``C_i`` per operator.  Following Appendix D of
arXiv:2112.10787 we name the coefficients

* ``alpha_*`` for physical (Warsaw-basis) operators ``O`` ,
* ``beta_*``  for redundant operators ``R`` ,
* ``gamma_*`` for evanescent operators ``E`` .

The overall dimensionful factor ``1/Lambda^2`` is left implicit inside ``C_i``
(equivalently ``Lambda`` is set to one); this is the standard matchmakereft
convention and keeps every coefficient dimensionless in the code.

Flavour indices.  An operator acting on ``n`` fermion currents carries an
``n``-index coefficient ``C_i^{ijkl}`` whose generation labels are contracted,
in order, with the fermion fields of the operator.  Coefficients are declared
``complex`` by default; the physical reality/Hermiticity relations implied by
each operator (for example ``C_{Hq}^{(1)}`` Hermitian, ``C_{qq}^{(1)}`` with the
``ij<->kl`` symmetry of a current-current structure) are recorded as metadata
in :data:`SYMMETRY_NOTES` rather than imposed on the symbol, so that the
generated Feynman rules remain fully general.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from feynpy import IndexType, Parameter


_PREFIX = {
    "physical": "alpha",
    "redundant": "beta",
    "evanescent": "gamma",
}


@dataclass(frozen=True)
class WilsonSpec:
    """Metadata describing one operator's Wilson coefficient."""

    name: str
    otype: str
    n_flavour: int
    complex_param: bool = True
    symmetry: str = ""


def coefficient_symbol_name(operator_name: str, otype: str) -> str:
    prefix = _PREFIX[otype]
    return f"{prefix}_{operator_name}"


def make_wilson_coefficient(
    operator_name: str,
    otype: str,
    n_flavour: int,
    *,
    generation: IndexType,
    complex_param: bool = True,
) -> Parameter:
    """Build the Wilson-coefficient :class:`Parameter` for one operator."""

    if otype not in _PREFIX:
        raise ValueError(f"Unknown operator type {otype!r}.")
    name = coefficient_symbol_name(operator_name, otype)
    indices = (generation,) * n_flavour
    return Parameter(
        name,
        indices=indices,
        complex_param=complex_param and n_flavour > 0,
    )


# Human-readable Hermiticity / flavour-symmetry notes (not enforced on the
# symbol; used for documentation and for the checklist).
SYMMETRY_NOTES: dict[str, str] = {
    "OHq1": "Hermitian: C = C^dagger",
    "OHq3": "Hermitian: C = C^dagger",
    "OHu": "Hermitian: C = C^dagger",
    "OHd": "Hermitian: C = C^dagger",
    "OHl1": "Hermitian: C = C^dagger",
    "OHl3": "Hermitian: C = C^dagger",
    "OHe": "Hermitian: C = C^dagger",
    "Oqq1": "C^{ijkl} = C^{klij}",
    "Oqq3": "C^{ijkl} = C^{klij}",
    "Ouu": "C^{ijkl} = C^{klij}",
    "Odd": "C^{ijkl} = C^{klij}",
    "Oll": "C^{ijkl} = C^{klij}",
    "Oee": "C^{ijkl} = C^{klij}",
}


__all__ = (
    "WilsonSpec",
    "SYMMETRY_NOTES",
    "coefficient_symbol_name",
    "make_wilson_coefficient",
)
