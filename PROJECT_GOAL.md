## Project Goal

Implement a Python-based analogue of FeynRules using Symbolica and Spenso.

### Core objective

Build a reusable symbolic pipeline that starts from a field-theory model
definition and derives Feynman interaction vertices in Python, using:

- Symbolica for symbolic expressions, pattern matching, simplification, and
  rule-based rewriting.
- Spenso for tensor structures, indices, representations, and spinor/Lorentz
  bookkeeping.

### Target capabilities

- Define fields, parameters, indices, and representations in Python.
- Encode interaction terms in a Lagrangian.
- Select interaction monomials relevant to a chosen process/order.
- Perform Wick-contraction style combinatorics for external legs.
- Handle derivative interactions as momentum factors.
- Track tensor, Lorentz, and spinor indices with Spenso objects.
- Simplify Kronecker deltas, metrics, and symbolic index structures.
- Extract final vertex factors, including momentum-conservation structure.
- Support at least scalar theories first, then fermions, then gauge structures.

### Near-term implementation strategy

- Keep notebook experiments as exploration only.
- Move stable logic into importable Python modules under `code/`.
- Maintain runnable examples and regression checks alongside the core engine.
- Prefer Symbolica-native expressions over string-based algebra.
- Use Spenso representations and tensor objects whenever index structure matters.

### Working interpretation for future sessions

When continuing this project, optimize for turning the current prototype into a
clean, tested, extensible "FeynRules with Symbolica + Spenso" library in
Python, not just isolated notebook demos.

### Session handoff

This section is the short reminder for the next session so nobody has to
reconstruct the current state from the code again.

What we now believe is the correct physics direction:

- final fermion vertices should be amputated open-index objects
- matrix elements with `UF/UbarF` kept explicitly are useful diagnostics, but
  they are not the final Feynman rules
- multi-fermion operators need explicit spinor-contraction information; a bare
  product like `psi * psibar * psi * psibar` is underspecified

What was fixed in the current cleanup:

- scalar fermion bilinears now amputate to open spinor metrics instead of being
  stripped to scalars
- `-(g/2)(psibar psi)^2` now gives the expected direct-minus-exchange
  four-fermion structure
- explicit open spinor labels carried inside coupling tensors now follow the
  contraction permutation and remap to the external leg spinor indices
- runnable regressions now cover `psibar gamma^mu psi A_mu` and a gamma-current
  four-fermion operator
- underspecified four-fermion products are rejected instead of misleadingly
  returning `0`

What is still missing before the fermion engine feels truly solid:

- normalization and symmetry-factor conventions for fermion operators are not
  centralized yet
- ambiguous encodings that mix repeated dummy spinor labels with explicit
  tensor endpoints need stronger validation
- support is still limited to the currently exercised bilinear/current-current
  patterns, not general multi-fermion tensor structures

Recommended first target for the next session:

- centralize normalization and symmetry-factor conventions for fermion
  operators
- then add stronger validation for ambiguous fermion encodings before widening
  the supported operator classes
