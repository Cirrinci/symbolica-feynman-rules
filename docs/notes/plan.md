---
name: Notes cleanup and commit-driven alignment
overview: Keep docs/notes concise, non-duplicative, and aligned with the current compiler architecture and latest commits.
todos:
  - id: commit-analysis
    content: summarize latest commit chain into concrete architecture changes
    status: completed
  - id: deduplicate-notes
    content: assign one clear purpose per note file and remove repeated status blocks
    status: completed
  - id: docs-sync
    content: align terminology with current modular src layout and tests
    status: completed
isProject: false
---

# Maintenance Rule

- `RESEARCH_LOG.md`: chronology and commit analysis only.
- `ROADMAP.md`: future plan only.
- `THESIS_PROGRESS.md`: milestone snapshot only.
- topic notes (`CONVENTIONS`, `COVARIANT_*`, `DECLARATIVE_*`, `LAGRANGIAN_*`, `FEYNRULES_*`, `Output`) should avoid repeating global project status.

## Current cleanup goals

1. Keep each note useful on its own, not just short.
2. Remove repeated status summaries, but keep implementation-shaping details where decisions are made.
3. Prefer moving duplicated content into one canonical note rather than deleting it everywhere.

## Note map

- `CONVENTIONS.md`: frozen sign and normalization choices used by compiler and tests.
- `COVARIANT_DERIVATIVE_GENERALIZATION.md`: representation-driven covariant expansion rules.
- `DECLARATIVE_LAGRANGIAN_TRANSITION.md`: migration state for `lagrangian_decl`.
- `LAGRANGIAN_API_NEXT_STEPS.md`: concrete user-facing API backlog and implementation direction.
- `ROADMAP.md`: forward-looking delivery order.
- `RESEARCH_LOG.md`: dated technical history and commit-window analysis.
