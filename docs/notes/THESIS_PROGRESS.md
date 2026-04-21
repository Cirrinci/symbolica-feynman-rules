## Thesis Progress

Working title: Toward a Python implementation of FeynRules using Symbolica and Spenso.

## Current thesis position

The project is beyond proof of concept for the ordinary gauge-theory baseline. The key result is that Symbolica handles the combinatoric/symbolic core while Spenso provides a viable tensor/index layer for realistic vertex structures.

## Milestone status

- symbolic contraction engine: done
- model/declaration layer: usable and actively refined
- declarative Lagrangian front end: done (primary path)
- covariant matter compiler: done for covered cases
- pure-gauge Yang-Mills compiler: done for covered cases
- ordinary gauge fixing + ghosts: done for covered cases
- SU(2)L example + tests: done
- BFM split and BFM sectors: not started

## Current architecture snapshot

- symbolic engine and tensor utilities: `src/symbolic/*`
- model/declaration and lowering: `src/model/*`
- declarative helper layer: `src/lagrangian/*`
- gauge/covariant compiler: `src/compiler/gauge.py`
- runnable examples: `examples/*`
- regression suite: `tests/*`

## Main open risks

- fast gauge/lowering refactors may outpace targeted regression coverage
- a portion of behavioral checks still sits in examples rather than tests
- API ergonomics for whole-Lagrangian extraction still incomplete

## Next thesis milestone

Deliver first BFM-ready declaration/compiler layer (background/quantum split) on top of the already working ordinary gauge-fixed baseline, with tests first.
