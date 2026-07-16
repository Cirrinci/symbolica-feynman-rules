"""Public FeynPy toolkit API.

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

Concrete theories are intentionally not part of this toolkit package. They
live in the sibling ``theories`` package, which plays the role that model
files play in FeynRules.

Declarative ``lagrangian_decl`` factors live in ``declared.py``:
``DC``, ``PartialD``, ``Gamma``, ``Gamma5``, ``Metric``, ``T``,
``StructureConstant``, ``FS``, ``GaugeFixing``, ``GhostLagrangian``.
The descriptive aliases ``CovD`` and ``FieldStrength`` remain available.
Replacement-expression matrix factors (``ProjM``, ``ProjP``, ``rotation(...)``)
let ``FieldTransformation(source, expr)`` carry projectors and flavor rotations.
"""

from __future__ import annotations

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
    CovariantDerivativeOperatorFactor,
    CovD,
    DC,
    DifferentiatedCovariantFactor,
    DifferentiatedOperatorFactor,
    FS,
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
    ProjM,
    ProjP,
    StructureConstant,
    StructureConstantFactor,
    T,
    rotation,
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
from .transformations import (
    CyclicTransformationError,
    FieldTransformation,
    ReplacementTerm,
    TransformationContext,
    apply_field_transformations,
    expand_index_components,
    replacement,
)
from .transformation_postprocess import (
    canonical_compiled_expression,
    canonicalize_transformed_terms,
    compiled_is_hermitian,
    equivalent_canonical_compiled,
    find_source_basis_occurrences,
    validate_compiled_index_multiplicities,
)
from .display import (
    clean_text,
    format_rule,
    pretty_expression,
    show_model,
    show_result,
)

# ---- internal symbols still used across the codebase ---------------------
from .declared import _DeclaredMonomial
from .lowering import (
    _expand_field_strengths_in_monomial,
    _match_covariant_monomial,
)
