# kore-mind

Persistent memory + emergent identity engine for any LLM.

**One file = one mind.** SQLite-based. Zero config. Zero external dependencies. Runtime-agnostic.

Part of [**kore-stack**](https://github.com/iafiscal1212/kore-stack) — the complete cognitive middleware for LLMs. `pip install kore-stack` for the full stack, or install individually:

## Install

```bash
pip install kore-mind          # just the memory engine
pip install kore-stack         # full stack: mind + bridge + SC routing
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

## API

| Method | Description |
|--------|-------------|
| `experience(text)` | Something happened. Record it. |
| `recall(query)` | What's relevant now? |
| `reflect(fn)` | Consolidate. Decay. Evolve. |
| `identity()` | Who am I now? |
| `forget(threshold)` | Explicit pruning. |
| `scoped(source)` | Filtered view per user. Same DB. |
| `traces()` | Query operation traces. |

## Semantic Search (v0.3)

Built-in embedding providers — semantic recall works with one line:

```python
from kore_mind import Mind, numpy_embed

# Zero-dependency option (numpy only, no external service)
mind = Mind("agent.db", embed_fn=numpy_embed())

mind.experience("me gusta el café por la mañana")
mind.experience("Python es un lenguaje de programación")

# Finds "café" even searching for "bebidas calientes"
results = mind.recall("bebidas calientes")
```

Three providers available:

```python
from kore_mind.embeddings import numpy_embed, ollama_embed, openai_embed

# 1. numpy_embed — zero dependencies, deterministic, fast
mind = Mind("agent.db", embed_fn=numpy_embed())

# 2. ollama_embed — local Ollama server (falls back to numpy if unavailable)
mind = Mind("agent.db", embed_fn=ollama_embed())

# 3. openai_embed — cloud, max quality (requires API key)
mind = Mind("agent.db", embed_fn=openai_embed(api_key="sk-..."))
```

## v0.2 Features

### Per-user filtering

Each user gets their own "mind" — same database, different context.

```python
# Option 1: default source
mind = Mind("agent.db", default_source="carlos")
mind.experience("Likes Python")  # automatically tagged to carlos
mind.recall("Python")            # only carlos's memories

# Option 2: scoped view
alice = mind.scoped("alice")
alice.experience("Prefers Rust")
alice.recall()  # only alice's memories
```

### Observability

Full tracing of every operation. Zero overhead when disabled (default).

```python
mind = Mind("agent.db", enable_traces=True)

mind.experience("Something happened")
mind.recall("what happened")

# Query traces
traces = mind.traces(operation="recall")
for t in traces:
    print(f"{t.operation} took {t.duration_ms:.1f}ms")

# Filter by source
traces = mind.traces(source="carlos", limit=50)
```

### Smart Cache (storage layer)

Hash-based cache with TTL, per-user isolation, and hit counting. Used by [kore-bridge](https://github.com/iafiscal1212/kore-bridge) for token savings.

```python
from kore_mind.models import CacheEntry

entry = CacheEntry(
    query="What is P vs NP?",
    response="It's an open problem...",
    query_hash="a1b2c3d4",
    source="carlos",
    ttl=3600.0,
)
mind._storage.save_cache_entry(entry)
found = mind._storage.find_cache_by_hash("a1b2c3d4", source="carlos")
```

### Rate Limiting (storage layer)

Query logging with temporal window counting. Used by [kore-bridge](https://github.com/iafiscal1212/kore-bridge) for cognitive rate limiting.

## Models

| Model | Description |
|-------|-------------|
| `Memory` | A memory with lifecycle (salience, decay, tags, embedding) |
| `Identity` | Emergent identity (traits, summary, relationships) |
| `MemoryType` | episodic, semantic, procedural |
| `Trace` | Operation trace (operation, duration, source, metadata) |
| `CacheEntry` | Cache entry (query, response, hash, TTL, hit count) |

## Backward compatibility

All new parameters have defaults that preserve v0.1 behavior:

```python
# This works exactly the same as v0.1
mind = Mind("agent.db")
mind.experience("fact")
mind.recall("query")
```

## Part of kore-stack

| Package | What it does |
|---------|-------------|
| **kore-mind** (this) | Memory, identity, traces, cache storage |
| [kore-bridge](https://github.com/iafiscal1212/kore-bridge) | LLM integration, cache logic, rate limiting, A/B testing |
| [sc-router](https://github.com/iafiscal1212/sc-router) | Query routing by Selector Complexity theory |
| [**kore-stack**](https://github.com/iafiscal1212/kore-stack) | All of the above, one install: `pip install kore-stack` |

## License

MIT
