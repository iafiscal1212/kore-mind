"""Optional embeddings support. Cosine similarity for semantic recall.

Built-in providers (v0.3):
    numpy_embed(dims)   — zero-dependency hashing vectorizer
    ollama_embed(model)  — local Ollama server, falls back to numpy
    openai_embed(api_key) — OpenAI API embeddings
"""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.error
import urllib.request
from typing import Callable

from kore_mind.models import Memory

# Type alias: takes text, returns embedding bytes
EmbedFn = Callable[[str], bytes]


# ── Built-in providers ──────────────────────────────────────────────────


def numpy_embed(dims: int = 256) -> EmbedFn:
    """Hashing vectorizer: tokenize → hash each token to an index → normalized TF vector.

    Zero external dependencies beyond numpy (already optional dep).
    Deterministic, fast, captures word overlap.
    """
    import numpy as np

    def _embed(text: str) -> bytes:
        vec = np.zeros(dims, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tobytes()
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % dims
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tobytes()

    return _embed


def ollama_embed(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> EmbedFn:
    """Ollama embedding provider. Falls back to numpy_embed if Ollama is unavailable.

    Uses POST /api/embed (stdlib urllib only — zero new dependencies).
    """
    import numpy as np

    fallback = numpy_embed()

    def _embed(text: str) -> bytes:
        try:
            payload = json.dumps({"model": model, "input": text}).encode()
            req = urllib.request.Request(
                f"{base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            embedding = data["embeddings"][0]
            vec = np.array(embedding, dtype=np.float32)
            return vec.tobytes()
        except (urllib.error.URLError, OSError, KeyError, IndexError):
            return fallback(text)

    return _embed


def openai_embed(
    api_key: str,
    model: str = "text-embedding-3-small",
) -> EmbedFn:
    """OpenAI embedding provider.

    Uses urllib.request (stdlib only — zero new dependencies).
    Requires an API key.
    """
    import numpy as np

    def _embed(text: str) -> bytes:
        payload = json.dumps({"model": model, "input": text}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        embedding = data["data"][0]["embedding"]
        vec = np.array(embedding, dtype=np.float32)
        return vec.tobytes()

    return _embed


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
