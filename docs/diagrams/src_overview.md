# Code Structure Overview

This diagram shows the main FeynPy layers, what each layer owns, and how
information moves between them.

```mermaid
flowchart TB
    subgraph Models["Model packages and user-facing entry points"]
        M1["models/SM<br/>full Standard Model slice"]
        M2["models/SMEFT<br/>Green-basis SMEFT slice"]
        M3["models/UnbrokenSM_BFM<br/>reference model slice"]
        M4["notebooks/<br/>usage walkthroughs"]
    end

    subgraph Public["Public declaration layer: src/feynpy"]
        P1["metadata.py<br/>Field, Parameter, IndexType, GaugeGroup"]
        P2["declared.py<br/>declarative DSL factors"]
        P3["core.py<br/>Model API and orchestration"]
        P4["lagrangian.py<br/>DeclaredLagrangian and CompiledLagrangian"]
        P5["transformations.py<br/>basis changes on compiled terms"]
        P6["validation.py<br/>model and IR checks"]
    end

    subgraph Compile["Compilation layer"]
        C1["feynpy.lowering<br/>source-term analysis and routing"]
        C2["compiler.gauge<br/>CovD, FS, ghost, gauge-fixing compilation"]
        C3["lagrangian.lowering and lagrangian.operators<br/>local lowering helpers and tensor blocks"]
        C4["lagrangian.operator_action and ibp<br/>post-compilation operators"]
    end

    subgraph Symbolic["Symbolic backend"]
        S1["symbolic.spenso_structures<br/>tensor vocabulary and index structures"]
        S2["symbolic.vertex_engine<br/>vertex contraction engine"]
        S3["symbolic.vertex_postprocessing<br/>output cleanup"]
        S4["symbolic.tensor_canonicalization<br/>canonical tensor forms"]
    end

    subgraph Outputs["Outputs and comparison"]
        O1["CompiledLagrangian<br/>InteractionTerm IR"]
        O2["feynman_rule(...)<br/>vertex extraction"]
        O3["to_symbolica()<br/>expression export"]
        O4["feynrules.comparison<br/>reference-data validation"]
        O5["tests/<br/>regression and physics checks"]
    end

    M1 --> P3
    M2 --> P3
    M3 --> P3
    M4 --> P3

    P1 --> P2
    P1 --> P3
    P2 --> P3
    P3 --> P4
    P3 --> C1
    P3 --> P6

    S1 --> P1
    S1 --> C2
    S1 --> C3
    C3 --> C2
    C1 --> C2
    C2 --> O1

    O1 --> P5
    O1 --> C4
    O1 --> S2
    O1 --> O3
    O1 --> O4

    S2 --> S3
    S3 --> S4
    S4 --> O2

    O5 --> P6
    O5 --> O4
```

## Reading

- `src/feynpy` is the public model layer.
- `src/compiler` turns structured declarations into plain interaction terms.
- `src/symbolic` is the downstream symbolic backend used for contraction and
  canonicalization.
- `models/*` are complete physics applications built on top of the reusable
  engine.
