"""Covariant-sector compiler API.

This module is the canonical import surface for covariant-derivative and
covariant-source compilation entry points.

Implementation currently lives in ``compiler.gauge``; this facade keeps the
separation explicit for callers while preserving behaviour.
"""

from __future__ import annotations

from .gauge import (
    compile_complex_scalar_gauge_terms,
    compile_complex_scalar_kinetic_term,
    compile_covariant_terms,
    compile_dirac_kinetic_term,
    compile_fermion_gauge_current,
    compile_gauge_fixing_term,
    compile_gauge_kinetic_term,
    compile_ghost_term,
    compile_mixed_complex_scalar_contact_terms,
    expand_cov_der,
    with_compiled_covariant_terms,
)

__all__ = [
    "compile_complex_scalar_gauge_terms",
    "compile_complex_scalar_kinetic_term",
    "compile_covariant_terms",
    "compile_dirac_kinetic_term",
    "compile_fermion_gauge_current",
    "compile_gauge_fixing_term",
    "compile_gauge_kinetic_term",
    "compile_ghost_term",
    "compile_mixed_complex_scalar_contact_terms",
    "expand_cov_der",
    "with_compiled_covariant_terms",
]
