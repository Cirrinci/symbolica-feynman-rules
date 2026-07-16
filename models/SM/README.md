# Standard Model

This directory is the complete Standard Model vertical slice for FeynPy.

- `SM.py` — packaged gauge-basis Standard Model implementation.
- `SM_support.py` — Standard-Model-specific finite-index, gauge-fixing, ghost,
  and compilation helpers.
- `feynrules_comparison.py` — Standard-Model-specific routing and field-name
  alignment for the generic FeynRules comparator.
- `notebooks/SM_feynpy.ipynb` — literate `SM.fr`-style implementation.
- `notebooks/SM_comparison.ipynb` — complete 163-vertex symbolic comparison.
- `reference/feynrules/` — tracked FeynRules `SM.fr` JSON oracle.
- `tests/` — Standard Model and oracle regression tests.
- `docs/SM_FR_COMPARISON_REVIEW.md` — scope and reliability review.
- `playground.py` — runnable Standard-Model playground with concise imports
  and formatted vertex display.

Reusable framework code remains under `src/feynpy/`. Generic FeynRules parsing
and canonical comparison remain under `src/feynrules/`.

The public model import is:

```python
from models.SM import build_standard_model
```

For a minimal playground-style workflow, the package now also re-exports the
default Standard-Model fields, parameters, gauge groups, and source sectors:

```python
from feynpy import *
from models.SM import *

L = sm_model(L_gauge)

L2 = sm_model(
    -Yd(f2, f3)
    * CKM(f1, f2)
    * QL.bar(spinor, weak_left, f1, colour)
    * dR(spinor, f3, colour)
    * Phi(weak_left)
)
```

The ready-to-edit playground file is [playground.py](/Users/rems/Documents/MScThesis/models/SM/playground.py).
