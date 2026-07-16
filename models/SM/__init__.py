"""Standard Model implementation and FeynRules comparison support."""

from . import SM as _sm_module

StandardModel = _sm_module.StandardModel
StandardModelFields = _sm_module.StandardModelFields
StandardModelGaugeGroups = _sm_module.StandardModelGaugeGroups
StandardModelIndices = _sm_module.StandardModelIndices
StandardModelLagrangians = _sm_module.StandardModelLagrangians
StandardModelParameters = _sm_module.StandardModelParameters
build_standard_model = _sm_module.build_standard_model
default_standard_model = _sm_module.default_standard_model
sm_model = _sm_module.sm_model
standard_model_weak_tensor_components = _sm_module.standard_model_weak_tensor_components

__all__ = _sm_module.__all__


def __getattr__(name: str):
    return getattr(_sm_module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
