"""Memory consolidation. Similar memories merge into stronger ones."""

from __future__ import annotations

import time
from difflib import SequenceMatcher

from kore_mind.models import Memory, MemoryType


def text_similarity(a: str, b: str) -> float:
    """Simple text similarity using SequenceMatcher (stdlib, no deps)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar_text(memories: list[Memory],
                      threshold: float = 0.7) -> list[tuple[Memory, Memory, float]]:
    """Find similar memory pairs by text content. Fallback when no embeddings."""
    pairs = []
    for i, a in enumerate(memories):
        for b in memories[i + 1:]:
            sim = text_similarity(a.content, b.content)
            if sim >= threshold:
                pairs.append((a, b, sim))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def merge_memories(a: Memory, b: Memory, merged_content: str | None = None) -> Memory:
    """Merge two memories into one. The stronger survives, enriched.

    - Takes the highest salience
    - Sums access counts
    - Unions tags
    - Keeps the earlier created_at
    - Content: merged_content if provided, else the more salient one's content
    """
    # Keep the more salient as base
    if a.salience >= b.salience:
        base, other = a, b
    else:
        base, other = b, a

    content = merged_content or f"{base.content} | {other.content}"

    merged_tags = list(set(base.tags + other.tags))

    return Memory(
        content=content,
        type=base.type,
        salience=min(1.0, max(a.salience, b.salience) + 0.1),  # consolidation bonus
        created_at=min(a.created_at, b.created_at),
        last_accessed=time.time(),
        access_count=a.access_count + b.access_count,
        source=base.source or other.source,
        tags=merged_tags,
        embedding=base.embedding,  # keep the base's embedding
        id=base.id,  # reuse base id (other gets deleted)
    )


def consolidate(memories: list[Memory], threshold: float = 0.7,
                use_embeddings: bool = False,
                merge_fn=None) -> tuple[list[Memory], list[str]]:
    """Consolidate similar memories.

    Args:
        memories: list of alive memories
        threshold: similarity threshold for merging
        use_embeddings: if True and embeddings available, use cosine similarity
        merge_fn: optional custom merge function(mem_a, mem_b) -> Memory

    Returns:
        (consolidated_memories, deleted_ids)
    """
    if len(memories) < 2:
        return memories, []

    # Find similar pairs
    if use_embeddings:
        has_embeddings = any(m.embedding is not None for m in memories)
        if has_embeddings:
            from kore_mind.embeddings import find_similar_pairs
            pairs = find_similar_pairs(memories, threshold=max(threshold, 0.80))
        else:
            pairs = find_similar_text(memories, threshold=threshold)
    else:
        pairs = find_similar_text(memories, threshold=threshold)

    if not pairs:
        return memories, []

    # Merge pairs (greedy: highest similarity first, each memory merges at most once)
    merged_ids = set()
    deleted_ids = []
    new_memories = []

    for a, b, sim in pairs:
        if a.id in merged_ids or b.id in merged_ids:
            continue

        if merge_fn is not None:
            merged = merge_fn(a, b)
        else:
            merged = merge_memories(a, b)

        new_memories.append(merged)
        merged_ids.add(a.id)
        merged_ids.add(b.id)

        # The one whose id is NOT kept gets deleted
        deleted_id = b.id if merged.id == a.id else a.id
        deleted_ids.append(deleted_id)

    # Keep unmerged memories
    for mem in memories:
        if mem.id not in merged_ids:
            new_memories.append(mem)

    return new_memories, deleted_ids
