"""Optional embeddings support. Cosine similarity for semantic recall.

Built-in providers (v0.3.1):
    numpy_embed(dims)   — zero-dependency hashing vectorizer
    ollama_embed(model)  — local Ollama server, falls back to numpy
    openai_embed(api_key) — OpenAI API embeddings
"""

from __future__ import annotations

import collections
import hashlib
import http.client
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
import warnings
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
    cache_size: int = 512,
) -> EmbedFn:
    """Ollama embedding provider. Falls back to numpy_embed if Ollama is unavailable.

    v0.3.1 improvements:
    - Connection reuse via http.client.HTTPConnection (keep-alive)
    - LRU cache (OrderedDict, configurable size)
    - Batch embedding via .batch(texts) — single HTTP call
    - RuntimeWarning on fallback (dimension mismatch: numpy=256d, Ollama=768d)
    """
    import numpy as np

    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434

    fallback = numpy_embed()
    cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()
    conn_holder: list[http.client.HTTPConnection | None] = [None]
    warned_fallback = [False]

    def _get_conn() -> http.client.HTTPConnection:
        """Return existing connection or create a new one."""
        if conn_holder[0] is None:
            conn_holder[0] = http.client.HTTPConnection(host, port, timeout=10)
        return conn_holder[0]

    def _post(payload: bytes) -> dict:
        """POST to /api/embed with 1 automatic retry on broken connection."""
        for attempt in range(2):
            try:
                conn = _get_conn()
                conn.request(
                    "POST", "/api/embed", body=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                return json.loads(resp.read())
            except (http.client.HTTPException, OSError):
                conn_holder[0] = None  # force reconnect
                if attempt == 1:
                    raise
        raise OSError("unreachable")

    def _cache_put(text: str, value: bytes) -> None:
        cache[text] = value
        cache.move_to_end(text)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def _fallback_with_warning(text: str) -> bytes:
        if not warned_fallback[0]:
            warnings.warn(
                "Ollama unavailable — falling back to numpy_embed. "
                "Warning: numpy vectors (256d) are incompatible with "
                "Ollama vectors (768d). Do not mix them.",
                RuntimeWarning,
                stacklevel=3,
            )
            warned_fallback[0] = True
        return fallback(text)

    def _embed(text: str) -> bytes:
        # Cache hit
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        # Cache miss — call Ollama
        try:
            payload = json.dumps({"model": model, "input": text}).encode()
            data = _post(payload)
            embedding = data["embeddings"][0]
            vec = np.array(embedding, dtype=np.float32)
            result = vec.tobytes()
            _cache_put(text, result)
            return result
        except (http.client.HTTPException, OSError, KeyError, IndexError):
            result = _fallback_with_warning(text)
            _cache_put(text, result)
            return result

    def _batch(texts: list[str]) -> list[bytes]:
        """Embed multiple texts in a single HTTP call. Uses cache for known texts."""
        results: list[bytes | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            if text in cache:
                cache.move_to_end(text)
                results[i] = cache[text]
            else:
                to_fetch.append((i, text))

        if to_fetch:
            fetch_texts = [t for _, t in to_fetch]
            try:
                payload = json.dumps({"model": model, "input": fetch_texts}).encode()
                data = _post(payload)
                embeddings = data["embeddings"]
                for j, (idx, text) in enumerate(to_fetch):
                    vec = np.array(embeddings[j], dtype=np.float32)
                    result = vec.tobytes()
                    _cache_put(text, result)
                    results[idx] = result
            except (http.client.HTTPException, OSError, KeyError, IndexError):
                for idx, text in to_fetch:
                    result = _fallback_with_warning(text)
                    _cache_put(text, result)
                    results[idx] = result

        return results  # type: ignore[return-value]

    _embed.batch = _batch  # type: ignore[attr-defined]
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
