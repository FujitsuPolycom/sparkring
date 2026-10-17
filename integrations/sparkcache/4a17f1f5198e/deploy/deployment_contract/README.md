# Deployment contract

**Status: implemented.** This package contains the model-neutral code that
turns a Docker inspection record into a repeatable SparkCache launch.

A profile supplies model, topology, quantization, cache identity, and accepted
source policy. The shared package does not guess or relax those choices.

## Modules

| Module | Purpose |
|---|---|
| `command.py` | Normalize vLLM commands and encode deterministic JSON. |
| `inspection.py` | Read one Docker inspection record and its environment. |
| `ports.py` | Validate numeric port ranges. |
| `source.py` | Hash files and calculate line-ending-independent source identities. |
| `patches.py` | Apply Git patches with exact input and output checks. |
| `receipts.py` | Create and verify overlay inventories and generated-file hashes. |
| `container.py` | Build and execute deterministic Docker commands. |
| `semantic.py` | Run deterministic miss, restart/restore, and continued-output checks. |

Compatibility imports remain available from existing profile modules. Receipt
schema identifiers and encoded receipt bytes are unchanged.

## Semantic checks

The response reader keeps final `content` separate from the combined assistant
body. The combined body contains `reasoning`, `reasoning_content`, and
`content`, in that order.

A length-limited response or an empty assistant body returns an `INCONCLUSIVE`
report. It is not treated as a successful or failed semantic comparison.

Exact expected-answer checks use final `content`. Existing references without
the optional `assistant_body` field remain readable.

## Test

```bash
python -m pytest deploy/deployment_contract -q
python -m pytest sparkcache deploy -q
python -m ruff check sparkcache deploy
```
