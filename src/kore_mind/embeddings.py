"""Optional embeddings support. Cosine similarity for semantic recall."""

from __future__ import annotations

from typing import Callable

from kore_mind.models import Memory

# Type alias: takes text, returns embedding bytes
EmbedFn = Callable[[str], bytes]


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two embedding byte vectors."""
    import numpy as np
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def semantic_search(query_embedding: bytes, memories: list[Memory],
                    limit: int = 10) -> list[tuple[float, Memory]]:
    """Rank memories by cosine similarity to query embedding.

    Returns list of (similarity, memory) sorted by similarity descending.
    Only includes memories that have embeddings.
    """
    scored = []
    for mem in memories:
        if mem.embedding is None:
            continue
        sim = cosine_similarity(query_embedding, mem.embedding)
        scored.append((sim, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def find_similar_pairs(memories: list[Memory],
                       threshold: float = 0.85) -> list[tuple[Memory, Memory, float]]:
    """Find pairs of memories with similarity above threshold.

    Used by consolidation in reflect(). O(n²) but only on alive memories.
    Returns list of (mem_a, mem_b, similarity).
    """
    embedded = [m for m in memories if m.embedding is not None]
    pairs = []
    seen = set()

    for i, a in enumerate(embedded):
        for b in embedded[i + 1:]:
            sim = cosine_similarity(a.embedding, b.embedding)
            if sim >= threshold and (a.id, b.id) not in seen:
                pairs.append((a, b, sim))
                seen.add((a.id, b.id))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
