# Output Policy and Canonical Forms

Purpose: define how to read raw vs compact vertex output.

## Rule 1: raw output is authoritative

Compiler raw output is the source of truth for correctness and tests.

## Rule 2: compact output is a readability transform

Compact textbook-looking expressions are accepted as equivalent if they are algebraic rewrites of the same raw result under the active conventions.

## Common compact targets

- scalar QED current:
  `i e q (p_out - p_in)^mu`
- gauge bilinear:
  `i (g^{mu nu} p^2 - p^mu p^nu)`
- non-abelian gauge bilinear:
  `i delta^{ab} (g^{mu nu} p^2 - p^mu p^nu)`
- gauge-fixed inverse propagator:
  `i [g^{mu nu} p^2 - (1 - 1/xi) p^mu p^nu]`
- ghost bilinear:
  `-i delta^{ab} p^2`

## Review labels

When curating outputs, use one of:

- Correct
- Correct, needs simplification
- Correct rejection

This keeps physics validation separate from display/canonicalization quality.
