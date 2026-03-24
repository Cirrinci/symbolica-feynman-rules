# Deep Dive: `phi phi psibar psi` Vertex Extraction in `model_symbolica.py`

This note traces your exact pipeline for

```python
V_base = simplify_deltas(vertex_factor(**L, x=x, d=d), species_map=species_map)
```

with a fermion-containing interaction of the form

\[
\mathcal{L}_{\text{int}} = g \, \phi \, \phi \, \bar\psi \, \psi
\]

and code-style data

```python
L = dict(
    coupling=g,
    alphas=[phi0, phi0, psibar0, psi0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["scalar", "scalar", "psibar", "psi"],
    leg_roles=["scalar", "scalar", "psibar", "psi"],
)
```

---

## 1) Start from the Lagrangian (physics object)

Your interaction monomial is

\[
g \, \phi_{\alpha_1}(x)\phi_{\alpha_2}(x)\bar\psi_{\alpha_3}(x)\psi_{\alpha_4}(x)
\]

In your list representation:

- slot `0` -> first `phi` (`alpha_1 = phi0`)
- slot `1` -> second `phi` (`alpha_2 = phi0`)
- slot `2` -> `psibar` (`alpha_3 = psibar0`)
- slot `3` -> `psi` (`alpha_4 = psi0`)

The algorithm then sums over all Wick contractions between these field slots and external legs.

---

## 2) Big picture of the call chain

1. `vertex_factor(**L, x=x, d=d)`  
2. Inside it, `contract_to_full_expression(...)` constructs the contraction sum.
3. Back in `vertex_factor`, multiply by coupling, do momentum-delta replacement, strip external factors, multiply by `i`.
4. `simplify_deltas(..., species_map=...)` projects symbolic species deltas to your selected external content.

---

## 3) What `vertex_factor` does before/after the loop

`vertex_factor` is a wrapper around the contraction engine:

1. `contracted = contract_to_full_expression(...)`
2. `full = coupling * contracted`
3. Replace `exp(-i (sum p).x)` with `(2*pi)^d * Delta(sum p)` if `include_delta=True`
4. If `strip_externals=True`, replace:
   - `U(...) -> 1`
   - `UF(...), UbarF(...) -> 1` (unless using spinor-delta mode)
   - residual exponentials `exp(-i q.x) -> 1`
5. Return `I * full`

So the heavy combinatorics/sign logic is all in `contract_to_full_expression`.

---

## 4) `contract_to_full_expression`: detailed execution

## 4.1 Input checks and mode setup

The function first checks:

- lengths: `len(alphas) == len(betas) == len(ps) = n`
- derivative indices/targets consistency
- role-array consistency (`field_roles`, `leg_roles`)
- fermion mode requires role arrays (to avoid ambiguous signs/contractions)

Then it prepares:

- `n = 4`
- loop over `permutations(range(n))` -> 24 candidate bijections
- `total = 0`

Interpretation of a permutation `perm`:

\[
i \mapsto j = \text{perm}[i]
\]

meaning: **field slot `i` contracts with external leg `j`**.

---

## 4.2 Per-permutation procedure

For each `perm`:

1. `term = 1`
2. Check role compatibility for each slot pair `(i, perm[i])`  
   If any mismatch, discard permutation (`continue`).
3. If fermion statistics:
   - find fermion slots (`[2, 3]` here)
   - compute inversion parity of assigned fermion legs
   - multiply `term` by sign `(+1/-1)`
4. Multiply derivative factors if present (none in this basic example)
5. For each slot `i`:
   - multiply by `delta(alphas[i], betas[j])`
   - multiply by external factor by role:
     - scalar -> `U(betas[j], ps[j])`
     - `psibar` -> `UbarF(...)`
     - `psi` -> `UF(...)`
6. Multiply by common `exp(-i p_sum.x)` where `p_sum` is sum of assigned external momenta.
7. Add term into `total`.

---

## 5) Explicit permutation example 1: `(0132)`

Interpret `(0132)` as tuple `(0,1,3,2)`.

Mapping:

- slot 0 (`scalar`) -> leg 0 (`scalar`) : compatible
- slot 1 (`scalar`) -> leg 1 (`scalar`) : compatible
- slot 2 (`psibar`) -> leg 3 (`psi`) : **incompatible**
- slot 3 (`psi`) -> leg 2 (`psibar`) : incompatible as well

Because your `factor_leg_compatible(...)` enforces `field_roles[i] == leg_roles[j]`, this permutation is rejected at validity check stage.

### Consequence

- no fermion sign is even computed for this term (practically irrelevant after rejection)
- no delta/U factors are built
- no exponential is appended
- contribution to `total` is exactly zero by skipping the term

So in your current configuration with explicit role arrays, `(0132)` is a **dead contraction**.

---

## 6) Explicit permutation example 2: `(3021)`

Interpret `(3021)` as tuple `(3,0,2,1)`.

Mapping:

- slot 0 (`scalar`) -> leg 3 (`psi`) : **incompatible immediately**

The loop breaks at first mismatch and this permutation is discarded.

### Consequence

Same as above: this term never enters algebraic multiplication.

---

## 7) Why those examples die: role-filter physics meaning

You told the code:

- the Lagrangian has exactly one `psibar` slot and one `psi` slot
- external legs also carry exactly one `psibar` and one `psi` in fixed role positions

So physically, contractions that attach `psibar` field to `psi` external state are forbidden by construction.  
This implements the same idea as selecting only physically allowed operator-state contractions before summing.

---

## 8) A valid permutation for contrast: `(1023)`

Tuple `(1,0,2,3)`:

- slot 0 scalar -> leg 1 scalar (ok)
- slot 1 scalar -> leg 0 scalar (ok)
- slot 2 psibar -> leg 2 psibar (ok)
- slot 3 psi -> leg 3 psi (ok)

This survives.

### 8.1 Fermion sign on this term

Fermion slots are `[2,3]`. Assigned fermion legs are `[2,3]`.
Inversion count of `[2,3]` is 0 -> sign `+1`.

### 8.2 Term factors built by loop

\[
\delta(\phi0,b2)\,U(b2,p2)\;
\delta(\phi0,b1)\,U(b1,p1)\;
\delta(psibar0,b3)\,\bar U_F(b3,p3)\;
\delta(psi0,b4)\,U_F(b4,p4)\;
e^{-i(p1+p2+p3+p4)\cdot x}
\]

(Ordering in product follows loop multiplication order; mathematically commutative in symbolic product context.)

---

## 9) Which permutations survive overall in this setup

With fixed `leg_roles=["scalar","scalar","psibar","psi"]`, only permutations that:

- map scalar slots `{0,1}` into scalar legs `{0,1}`
- map psibar slot `2` to leg `2`
- map psi slot `3` to leg `3`

survive. Hence only:

- `(0,1,2,3)`
- `(1,0,2,3)`

Both have fermion sign `+1`; they differ only by scalar swap -> gives factor `2!` after species projection.

---

## 10) Momentum handling and derivatives

Even without derivatives, each surviving term gets:

\[
e^{-i(\sum p_j)\cdot x}
\]

Then in `vertex_factor`, this is replaced by

\[
(2\pi)^d \Delta\!\left(\sum p_j\right)
\]

If derivatives exist, each derivative acting on slot `tgt` contributes:

\[
i\partial_\mu \to (-i) p_{\text{perm}[tgt],\mu}
\]

so derivative momentum assignment is permutation-dependent.

---

## 11) What `simplify_deltas` finally does

Given a map like

```python
species_map = {b1: phi0, b2: phi0, b3: psibar0, b4: psi0}
```

it enforces:

- matching `delta(species, beta)` -> `1`
- non-matching `delta(other, same_beta)` -> `0`
- trivial `delta(a,a)` -> `1`

This collapses symbolic channel sums into your selected external-species channel.

---

## 12) Final conceptual summary

For `phi phi psibar psi`, your code computes:

1. all bijective slot-to-leg assignments,
2. keeps only role-compatible assignments,
3. adds fermionic sign from permutation parity on fermion slots,
4. multiplies contraction building blocks (species deltas + external factors),
5. sums all surviving terms,
6. performs x-integration replacement and amputation,
7. projects species via `simplify_deltas`.

In your specific role-constrained setup, `(0132)` and `(3021)` are explicitly discarded at compatibility check, while `(0123)` and `(1023)` survive.

