# SMEFT Green-basis coverage checklist

Reference: **Appendix D of arXiv:2112.10787** (Tables 1-9).

Every **registered** operator is written in the fully explicit style (raw
Spenso tensors + declarative `DC`/`FS`), with explicit chiral projectors so
ordered gamma chains survive (compiled with `simplify_gamma=False`). No EOM /
IBP / Fierz / 4D reduction is applied anywhere. Each registered operator is
individually reachable via the registry (`get_operator`, `operators_in`) and
can be inspected (`op.structure(core)`), compiled (`op.lagrangian(core)`) and
turned into Feynman rules (`op.feynman_rule(core, ...)`).

This file no longer claims full Appendix D completeness: the Tables 8-9
charge-conjugation sector is only partially represented in the registry.

## Summary

| Table | Sector        | Type       | Count | Status                    |
|-------|---------------|------------|-------|---------------------------|
| 1     | bosonic       | physical   | 15    | implemented               |
| 1     | bosonic       | redundant  | 8     | implemented               |
| 2     | two-fermion   | physical   | 19    | implemented               |
| 2     | two-fermion   | redundant  | 61    | implemented               |
| 3     | four-fermion  | physical   | 25    | implemented               |
| 4     | two-fermion   | evanescent | 41    | implemented               |
| 5     | four-fermion  | evanescent | 32    | implemented               |
| 6     | four-fermion  | evanescent | 25    | implemented               |
| 7     | four-fermion  | evanescent | 5     | implemented               |
| 8     | four-fermion  | evanescent | 18    | 5 representative entries registered as blocked; 13 unsupported |
| 9     | four-fermion  | evanescent | 19    | 3 representative entries registered as blocked; 16 unsupported |

**Totals:** Appendix D contains 268 operators. The current registry contains 239
entries: 231 compile to Feynman rules at canonical mass dimension 6, 8
charge-conjugation operators are registered as declared-only blocked
representatives, and the remaining 29 charge-conjugation operators from Tables
8-9 are not yet registered.

## Conventions (Appendix D)

* Covariant derivative (Eq. D.3):
  `D_mu = d_mu - i g3 T^A G^A_mu - i g2 (sigma^I/2) W^I_mu - i g1 Y B_mu`.
* Field strengths Eqs. D.4-D.6; `T^A = lambda^A/2`, `sigma^I` Pauli matrices.
* Hypercharges `Y_q=1/6, Y_l=-1/2, Y_u=2/3, Y_d=-1/3, Y_e=-1, Y_H=1/2`.
* Conjugate doublet `Htilde_r = eps_{rs} H*_s`; dual field strength
  `Xtilde_{mu nu} = 1/2 eps_{mu nu ab} X^{ab}`.
* Each operator carries a symbolic dimensionless Wilson coefficient (with the
  correct flavour arity); the overall `1/Lambda^2` is implicit (`wilson.py`).

## Table 1 -- bosonic

* Physical (15): `O3G, O3Gtilde, O3W, O3Wtilde` (X^3); `OHG, OHGtilde, OHW,
  OHWtilde, OHB, OHBtilde, OHWB, OHWBtilde` (X^2 H^2); `OH` (H^6); `OHbox, OHD`
  (H^4 D^2).
* Redundant (8): `RDH, RpHD, RppHD` (H^2 X D^2 / iterated D), `R2G, R2W, R2B`
  (D X D X), `RWDH, RBDH` (D X . H i<->D H).  Iterated covariant derivatives and
  the covariant derivative of a field strength are handled by expanding them
  explicitly (Leibniz) into bare-field + field-strength monomials, so **no
  engine change was needed**.

## Table 2 -- two-fermion physical (19) + redundant (61)

* Physical: `psi2DH2` currents (`OHq1, OHq3, OHu, OHd, OHl1, OHl3, OHe, OHud`),
  `psi2XH` dipoles (`OuG, OuW, OuB, OdG, OdW, OdB, OeW, OeB`), `psi2H3`
  (`OuH, OdH, OeH`).
* Redundant:
  * `psi2D3` (5): `RqD, RuD, RdD, RlD, ReD` -- `i/2 psibar {D^2,/D} psi`
    (three covariant derivatives, expanded explicitly).
  * `psi2XD` (30): `R{G,W,B}f`, `R'{G,W,B}f`, `R'{Gtilde,Wtilde,Btilde}f` for the
    relevant fermions -- currents times `D^nu X_{mu nu}`, and `i<->D` currents
    times `X_{mu nu}` / `Xtilde_{mu nu}`.
  * `psi2HD2` (12): `R{u,d,e}HD1..4` -- scalar/tensor bilinears with fermion
    covariant derivatives times `D^2 H`, `D H`, `H`.
  * `psi2DH2` redundant (14): `R'^{(1)}, R''^{(1)}, R'^{(3)}, R''^{(3)}` for `Hq,
    Hl` and `R', R''` for `Hu, Hd, He`.

## Table 3 -- four-fermion physical (25)

Four-quark, four-lepton and semileptonic current-current products, colour-octet
(`T^A x T^A`), isospin-triplet (`sigma^I x sigma^I`) and `eps_{rs}` scalar/tensor
LR operators (`Oquqd1/8, Oledq, Olequ1/3`).

## Table 4 -- two-fermion evanescent (41)

* `psi2XH` (8): the dipole operators with the **dual** field strength
  (`EuG, EuW, EuB, EdG, EdW, EdB, EeW, EeB`).
* `psi2XD` (30): `E{G,W,B}f` = `psibar Gamma (sigma^{mu nu} gamma^rho + gamma^rho
  sigma^{mu nu}) psi . D_rho Xtilde`, and `E'{...}f`, `E'{...tilde}f` =
  `i psibar (Gamma sigma^{mu nu} /D - <-/D sigma^{mu nu} Gamma) psi . X` for the
  relevant fermions.
* `psi2HD2` (3): `EuH, EdH, EeH` = `psibar sigma^{mu nu} D_rho psi . D_sigma
  H(tilde) . eps^{mu nu rho sigma}`.

Ordered chains `sigma^{mu nu} gamma^rho` are preserved verbatim.

## Tables 5-7 -- four-fermion evanescent, no charge conjugation (62)

Products of two Dirac bilinears with ordered gamma chains
`gamma^{mu1...mun} = gamma^{mu1} ... gamma^{mun}` (n = 0, 1, 2, 3), shared Lorentz
indices, colour-singlet/octet and isospin-singlet/triplet variants and `eps_{rs}`
structures.  Topologies: `direct` (LL RR, RR RR, LL LL), `crossed` (LR RL),
`eps` (LR LR), and the `ledq` semileptonic topology.  All 62 compile at
dimension 6 with the triple-gamma chains intact (verified in
`tests/test_evanescent.py::test_ordered_triple_gamma_chain_preserved`).

## Tables 8-9 -- charge-conjugation four-fermion evanescent (BLOCKED)

Representative operators registered (`status="blocked"`): `Ecuu, Ecdd, Ecqq,
Ec2uu, Ec2dd` (Table 8, quarks) and `Ecee, Ecll, Ec2ee` (Table 9, leptons),
spanning the scalar / vector / tensor C-chain structures.  Their declared
structure is inspectable (`op.structure(core)` builds the `psi^T C Gamma psi`
chains verbatim), but compilation is blocked.

Appendix D contains **37** such operators in total: **18** in Table 8 and
**19** in Table 9. Only the 8 representative entries listed above are currently
registered. The completeness claim "the complete dimension-six SMEFT Green basis
has been implemented" is therefore false as written.

**Exact blocker.**  The engine's local fermion-flow lowering
(`_ordered_local_dirac_bilinears` / `_unsupported_local_fermion_ordering_error`
in `src/feynpy/lowering.py`) requires every closed Dirac chain to have exactly
**one conjugated and one unconjugated** endpoint.  A charge-conjugation bilinear
`(psi^c-bar Gamma chi) = psi^T C Gamma chi` (or `(psibar Gamma chi^c)`) joins two
endpoints of the **same** conjugation through the charge-conjugation matrix `C`,
which the validator rejects.

**Why not patched here.**  A correct extension is *not* minimal: it must teach
the fermion-flow lowering to treat `C` as a chain connector, choose a consistent
fermion-flow orientation for the C-joined chain, and emit the correct relative
signs (`C gamma^mu C^{-1} = -gamma^{mu T}`, antisymmetry of `C`).  This touches
the core vertex-orientation logic and its sign bookkeeping, so per the plan it is
left as a documented, self-contained follow-up rather than a rushed change.  The
other two engine-change candidates flagged in the plan -- iterated covariant
derivatives and the covariant derivative of a field strength -- turned out to be
representable without any engine edit (explicit Leibniz expansion), so no `src/`
change was required for them.

## Engine changes

**None.**  The entire implementable basis (231 operators, Tables 1-7) is built on
the existing engine.  The only genuinely unrepresentable structures are the
charge-conjugation chains of Tables 8-9, documented above.
