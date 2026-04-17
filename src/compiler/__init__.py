"""Compiler package exports.

`compiler.gauge` owns minimal gauge-structure compilation.
`compiler.covariant` owns covariant-derivative and covariant-source expansion APIs.
"""

from .covariant import compile_covariant_terms, with_compiled_covariant_terms
from .gauge import compile_minimal_gauge_interactions, with_minimal_gauge_interactions

__all__ = [
    "compile_covariant_terms",
    "with_compiled_covariant_terms",
    "compile_minimal_gauge_interactions",
    "with_minimal_gauge_interactions",
]
