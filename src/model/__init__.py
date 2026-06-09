"""FeynRules-style model package (metadata, declarations, compilation, vertices).

The public surface is intentionally small. The recommended workflow is:

1. declare indices (``flavor_index(...)``, ``COLOR_FUND_INDEX``, ...)
2. declare gauge representations and ``GaugeGroup(...)``
3. declare fields with ``Field(...)`` (or ``dirac_field`` / ``scalar_field``;
   pass ``class_members=(...)`` and ``flavor_index=...`` for FeynRules-like
   flavor-class declarations)
4. declare parameters with ``Parameter(...)``
5. build the model with ``Model(gauge_groups=..., fields=..., parameters=...,
   lagrangian_decl=...)``
6. extract Feynman rules with ``model.feynman_rule(...)`` /
   ``model.vertex_signatures(...)``.

Declarative ``lagrangian_decl`` factors live in ``declared.py``:
``CovD``, ``PartialD``, ``Gamma``, ``Gamma5``, ``Metric``, ``T``,
``StructureConstant``, ``FieldStrength``, ``GaugeFixing``, ``GhostLagrangian``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

# ---- metadata ------------------------------------------------------------
from .metadata import (
    COLOR_ADJ_INDEX,
    COLOR_ADJ_KIND,
    COLOR_FUND_INDEX,
    COLOR_FUND_KIND,
    ConjugateField,
    Field,
    FieldRole,
    GaugeGroup,
    GaugeRepresentation,
    GhostField,
    IndexType,
    LORENTZ_INDEX,
    LORENTZ_KIND,
    Parameter,
    ParameterAssumptions,
    ROLE_GHOST,
    ROLE_GHOST_DAG,
    ROLE_PSI,
    ROLE_PSIBAR,
    ROLE_SCALAR,
    ROLE_SCALAR_DAG,
    ROLE_VECTOR,
    SPINOR_INDEX,
    SPINOR_KIND,
    Statistics,
    WEAK_ADJ_INDEX,
    WEAK_ADJ_KIND,
    WEAK_FUND_INDEX,
    WEAK_FUND_KIND,
    dirac_field,
    flavor_index,
    scalar_field,
)

# ---- declarative factors -------------------------------------------------
from .declared import (
    CovariantDerivativeFactor,
    CovD,
    DifferentiatedCovariantFactor,
    FieldStrength,
    FieldStrengthFactor,
    Gamma,
    Gamma5,
    GammaFactor,
    GaugeFixing,
    GaugeFixingDeclaration,
    GeneratorFactor,
    GhostLagrangian,
    GhostLagrangianDeclaration,
    Metric,
    MetricFactor,
    PartialD,
    PartialDerivativeFactor,
    StructureConstant,
    StructureConstantFactor,
    T,
)

# ---- interactions / compiled terms ---------------------------------------
from .interactions import (
    DerivativeRef,
    DiracBilinear,
    DerivativeAction,
    ExternalLeg,
    FieldOccurrence,
    IndexBinding,
    InteractionTerm,
    SlotLabels,
    SlotRef,
)

# ---- lagrangian containers (compiled + declared) -------------------------
from .lagrangian import (
    CompiledLagrangian,
    DeclaredLagrangian,
    KNOWN_VERTEX_SECTORS,
    VertexReport,
    VertexSignature,
)

# ---- top-level model -----------------------------------------------------
from .core import Model
from .validation import ValidationIssue, ValidationReport

# ---- unbroken Standard Model helper --------------------------------------
_SM_EXPORTS = frozenset(
    {
        "UnbrokenStandardModel",
        "UnbrokenStandardModelFields",
        "UnbrokenStandardModelGaugeGroups",
        "UnbrokenStandardModelIndices",
        "UnbrokenStandardModelLagrangians",
        "UnbrokenStandardModelParameters",
        "build_unbroken_standard_model",
    }
)

if TYPE_CHECKING:
    from .standard_model_unbroken import (
        UnbrokenStandardModel,
        UnbrokenStandardModelFields,
        UnbrokenStandardModelGaugeGroups,
        UnbrokenStandardModelIndices,
        UnbrokenStandardModelLagrangians,
        UnbrokenStandardModelParameters,
        build_unbroken_standard_model,
    )

# ---- SSB helpers (kept for the existing electroweak workflow) ------------
from .ssb import *  # noqa: F401,F403

# ---- internal symbols still used across the codebase ---------------------
from .declared import _DeclaredMonomial
from .lowering import (
    _expand_field_strengths_in_monomial,
    _match_covariant_monomial,
)


def __getattr__(name: str):
    if name in _SM_EXPORTS:
        module = import_module(".standard_model_unbroken", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
