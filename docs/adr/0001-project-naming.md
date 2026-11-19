# ADR 0001: Project naming — Sagwa, not AuditTrail

**Status**: Accepted (2026-08-18)

## Context

`PRD.md` and `PLAN.md` refer to the project as "AuditTrail" throughout — CLI examples (`audittrail run ...`), the repo-structure diagram, glossary, etc. The actual folder on disk (sibling to `../ringo`) is named `Sagwa`.

## Decision

Keep `Sagwa` as the folder name, Python package name, and CLI command. `PRD.md`/`PLAN.md` are edited in place (`AuditTrail` → `Sagwa`, `audittrail` → `sagwa`) rather than renaming the directory to match the docs.

## Consequences

- Package import path is `sagwa`, entry point is `sagwa run|diff|gate`.
- `../ringo`'s relative path (used throughout PLAN.md's integration boundary discussion) is unaffected — it's a sibling of this folder regardless of this folder's name.
