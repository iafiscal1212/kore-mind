# kore-mind

Persistent memory + emergent identity engine for any LLM.

**One file = one mind.** SQLite-based. Zero config. Runtime-agnostic.

## Install

```bash
pip install kore-mind
```

## Usage

```python
from kore_mind import Mind

mind = Mind("agent.db")

# Register experiences
mind.experience("User works on complexity theory proofs")
mind.experience("User prefers direct, concise answers")

# Recall relevant memories
memories = mind.recall("proof techniques")

# Reflect: decay old memories, consolidate, update identity
identity = mind.reflect()
print(identity.summary)

# Forget: explicit pruning
mind.forget(threshold=0.1)
```

## Core concepts

- **Memory has a lifecycle**: salience decays over time. Unused memories fade. Accessed memories strengthen.
- **Identity is emergent**: not configured, but computed from accumulated memories.
- **reflect()** is the key operation: decay + consolidation + identity update.

## API (5 methods)

| Method | Description |
|--------|-------------|
| `experience(text)` | Something happened. Record it. |
| `recall(query)` | What's relevant now? |
| `reflect(fn)` | Consolidate. Decay. Evolve. |
| `identity()` | Who am I now? |
| `forget(threshold)` | Explicit pruning. |

## License

MIT
