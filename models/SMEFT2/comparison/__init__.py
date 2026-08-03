"""SMEFT2 FeynRules/FeynPy comparison package.

The package is split by comparison layer while re-exporting the prior
`models.SMEFT2.comparison` API used by tests and notebooks. Run it with
`python -m models.SMEFT2.comparison`.
"""

from .base import *
from .canonical import *
from .charge_conjugation import *
from .vertices import *
from .exact import *
from .sidecars import *
from .reporting import *
from .cli import main

__all__ = [name for name in globals() if not name.startswith("__")]
