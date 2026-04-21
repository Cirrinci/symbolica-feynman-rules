# Frozen Compiler Conventions

Purpose: single source of truth for signs and normalization in compiled vertices.

## Scope

Applies to:

- `src/compiler/gauge.py`
- `src/model/lowering.py`
- `src/model/interactions.py`
- tests that validate compiled/vertex output

## Fourier and momentum rules

- all external momenta are incoming
- Fourier convention uses `exp(-i p.x)`
- derivatives map as `partial_mu -> -i p_mu`
- full vertices keep overall momentum-conservation delta unless explicitly stripped by API options

## Vertex extraction rules

- `vertex_factor(...)` multiplies the stripped coefficient by the universal overall `+i`
- with `include_delta=True`, the phase becomes `(2*pi)^d Delta(sum p)`
- with `strip_externals=True`, external `U`, `UF`, and `UbarF` factors are amputated from the displayed result

## Matter covariant derivative convention

`D_mu = partial_mu + i g A_mu`

Consequences in covered cases:

- fermion current from `i psibar gamma^mu D_mu psi` carries `-i g`
- complex-scalar current from `(D phi)^dagger (D phi)` carries `+i (p_out - p_in)`
- scalar two-gauge contact follows the same sign family

## Non-abelian field strength convention

`F^a_{mu nu} = partial_mu A^a_nu - partial_nu A^a_mu - g f^{abc} A^b_mu A^c_nu`

Consequences:

- abelian and non-abelian gauge bilinears follow the same derivative convention above
- Yang-Mills cubic raw output is real after stripping the universal extraction factor
- Yang-Mills 3-point and 4-point structures follow standard convention for this sign choice
- Yang-Mills quartic raw output keeps an explicit overall `i`

## Ordinary gauge fixing and ghosts

- gauge fixing: `L_gf = -(1 / (2 xi)) (partial.A)^2`
- ghosts (integrated form):
  `L_gh = -cbar partial.D c = (partial cbar)(partial c) - g f (partial cbar) A c`

Consequences in covered cases:

- raw gauge-fixing bilinears may stay symmetrized and unsimplified before readability rewrites
- ghost bilinear is proportional to `-i delta_adj p_bar.p_ghost`
- ghost-gauge raw vertex is real and carries the antighost momentum

## Readability policy

Raw compiler output may be overexpanded. Compact textbook forms are readability rewrites of the same conventions, not alternative definitions.
