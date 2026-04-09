# Frozen Compiler Conventions

This note is the main reference for the sign and normalization choices used by
the active Symbolica/Spenso compiler path.

## Scope

These conventions apply to:

- `vertex_factor(...)` in `src/model_symbolica.py`
- the convention-fixed physical compiler in `src/gauge_compiler.py`
- the regression checks that validate those paths

The minimal gauge compiler is a structural helper. The physical compiler is the
physics-facing path whose conventions are frozen here.

## Fourier and Vertex Rules

- plane waves use `exp(-i p.x)`
- all external momenta are treated as incoming
- derivatives map to `partial_mu -> -i p_mu`
- `vertex_factor(...)` multiplies the stripped coefficient by the universal
  overall `+i`
- with `include_delta=True`, the phase becomes `(2*pi)^d Delta(sum p)`
- with `strip_externals=True`, external `U`, `UF`, and `UbarF` factors are
  amputated from the displayed result

## Matter Covariant Derivative

The convention-fixed matter compiler uses:

`D_mu = partial_mu + i g A_mu`

Consequences:

- `psibar i gamma^mu D_mu psi` compiles to a `-i g` current vertex
- `(D_mu phi)^dagger (D^mu phi)` compiles to:
  - current terms proportional to `+i (p_out - p_in)`
  - two-gauge contact terms with the same convention set

For multi-group matter fields, one kinetic term expands into one contribution
per active gauge group, plus the ordered mixed-group scalar contact terms.

## Pure-Gauge Field Strength

The convention-fixed pure-gauge compiler uses:

- abelian:
  `F_{mu nu} = partial_mu A_nu - partial_nu A_mu`
- non-abelian:
  `F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu`

Consequences:

- the abelian and non-abelian gauge bilinears follow from the same derivative
  convention above
- the Yang-Mills cubic vertex is real in the stripped raw output
- the Yang-Mills quartic vertex keeps an explicit overall `i`

## Raw vs Compact Output

For the pure-gauge sector, the compiler's raw output may keep derivative-index
metrics explicit. Compact textbook-looking forms used in displays or tests are
readability rewrites of the same convention-fixed result, not alternative
conventions.
