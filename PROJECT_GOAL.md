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
