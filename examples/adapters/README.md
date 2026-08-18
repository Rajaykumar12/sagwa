# Example adapters

Sagwa evaluates *any* ML pipeline through one contract:
[`TargetAdapter`](../../sagwa/adapters/base.py) —

```python
class TargetAdapter(Protocol):
    name: str
    def run(self, case_input: str) -> AdapterResult: ...
```

`AdapterResult` is `{answer, context, latency_ms, tokens, cost_usd}`. Zero
required changes to your project's own codebase (PRD FR-3a) — and, just as
importantly, zero required changes to Sagwa's own source either.

## Writing your own adapter

1. Implement `TargetAdapter` in a class anywhere on your own filesystem —
   it doesn't need to live in this repo, and it isn't registered anywhere in
   `sagwa/`.
2. Point Sagwa at it:
   ```bash
   sagwa run --target your_module.path:YourAdapterClass --dataset your_golden_set.jsonl
   ```
   The module path is resolved the same way `python -m` would resolve it
   from your current working directory.
3. Your adapter owns its own configuration. If it needs an API key, a repo
   path, or anything else, read it from the environment (or accept
   constructor args) inside your own `__init__` — Sagwa's core never needs
   to know those exist. See `ringo_adapter.py`'s use of `RINGO_REPO_PATH`
   for the pattern.
4. Optional attributes Sagwa's CLI will pick up if present, via duck-typing
   (neither is required):
   - `self.repo_path` — if your target is its own git-versioned repo, expose
     this and Sagwa pins its git SHA into the run record (PRD FR-7a).
   - `self.model_label` — if there's no single static model id for your
     target (e.g. it routes per-query across models/tiers), set a
     human-readable string here instead of leaving the run record's `model`
     field as the default `"n/a"`.

## `ringo_adapter.py`

The reference example: an adapter for [ringo](../../../ringo), the RAG chat
app used as this project's own real-world validation target (see
`docs/PRD.md`). It's a genuine worked example, not a toy — read it for the
in-process-call pattern (vs. calling over HTTP) documented in its module
docstring and [docs/adr/0003-ringo-adapter-in-process.md](../../docs/adr/0003-ringo-adapter-in-process.md).

Try it (from this repo's root, with `RINGO_REPO_PATH` set and a live ringo
instance with documents indexed):

```bash
sagwa run --target examples.adapters.ringo_adapter:RingoAdapter --dataset golden_sets/example.jsonl
```
