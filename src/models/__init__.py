"""Concrete model definitions built on top of the FeynPy toolkit.

This package plays the role that model files play in FeynRules: the reusable
engine lives under ``model``, while concrete theories live here.
"""

from __future__ import annotations

from .standard_model import (
    StandardModel,
    StandardModelFields,
    StandardModelGaugeGroups,
    StandardModelIndices,
    StandardModelLagrangians,
    StandardModelParameters,
    build_standard_model,
    standard_model_weak_tensor_components,
)
