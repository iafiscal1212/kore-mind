"""Optional embeddings support. Cosine similarity for semantic recall.

Built-in providers (v0.5.0):
    numpy_embed(dims)        — zero-dependency hashing vectorizer
    ollama_embed(model)      — local Ollama server, falls back to numpy
    ollama_embed_async()     — native async variant of ollama_embed
    openai_embed(api_key)    — OpenAI API embeddings

v0.5.0 improvements:
    1. Retry with exponential backoff + jitter (sync and async)
    2. Native async HTTP via asyncio.open_connection (no thread pool)
    3. Retryable HTTP errors: 5xx, 429, OSError, HTTPException
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import http.client
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import TYPE_CHECKING, Callable

from kore_mind.models import Memory

if TYPE_CHECKING:
    from kore_mind.storage import Storage

# Type alias: takes text, returns embedding bytes
EmbedFn = Callable[[str], bytes]


# ── Helpers ────────────────────────────────────────────────────────────


class _RetryableHTTPError(Exception):
    """HTTP status that should trigger a retry (5xx, 429)."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}")


def _text_hash(text: str) -> str:
    """SHA-256 truncated to 16 hex chars — collision-safe for cache keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _to_float32(data: bytes) -> "np.ndarray":  # noqa: F821
    """Decode embedding bytes, auto-detecting float32 or float16."""
    import numpy as np
    n32 = len(data) / 4
    if n32 == int(n32) and n32 > 0:
        return np.frombuffer(data, dtype=np.float32)
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)


def _quantize(data: bytes) -> bytes:
    """Compress float32 embedding to float16 (half the size)."""
    import numpy as np
    vec = np.frombuffer(data, dtype=np.float32)
    return vec.astype(np.float16).tobytes()


# ── Built-in providers ──────────────────────────────────────────────────


def numpy_embed(dims: int = 256) -> EmbedFn:
    """Hashing vectorizer: tokenize -> hash each token to an index -> normalized TF vector.

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
    storage: Storage | None = None,
    quantize: bool = False,
    max_retries: int = 3,
    retry_base_delay: float = 0.5,
    retry_max_delay: float = 10.0,
) -> EmbedFn:
    """Ollama embedding provider. Falls back to numpy_embed if Ollama is unavailable.

    v0.5.0 improvements:
    - Retry with exponential backoff + jitter on 5xx/429/OSError
    - Native async HTTP via asyncio.open_connection (async_call, async_batch)
    - L1 in-memory LRU cache + L2 persistent SQLite cache (via storage)
    - Float16 quantization (quantize=True halves embedding size)
    - stream_batch() / astream_batch() for incremental processing
    - Connection reuse via http.client.HTTPConnection (keep-alive, sync)
    - Batch embedding via .batch(texts) -- single HTTP call
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
        """POST to /api/embed with exponential backoff retry."""
        import random
        import time

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                conn = _get_conn()
                conn.request(
                    "POST", "/api/embed", body=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                body = resp.read()
                if resp.status >= 500 or resp.status == 429:
                    raise _RetryableHTTPError(resp.status, body)
                return json.loads(body)
            except (http.client.HTTPException, OSError, _RetryableHTTPError) as exc:
                conn_holder[0] = None  # force reconnect
                last_exc = exc
                if attempt < max_retries:
                    delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
                    delay *= random.uniform(0.5, 1.5)  # jitter
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def _async_post(payload: bytes) -> dict:
        """POST to /api/embed via native asyncio.open_connection."""
        import random

        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                reader, writer = await asyncio.open_connection(host, port)
                try:
                    request = (
                        f"POST /api/embed HTTP/1.1\r\n"
                        f"Host: {host}:{port}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(payload)}\r\n"
                        f"Connection: close\r\n"
                        f"\r\n"
                    ).encode() + payload
                    writer.write(request)
                    await writer.drain()

                    # Parse response status line
                    status_line = await asyncio.wait_for(
                        reader.readline(), timeout=30
                    )
                    status = int(status_line.split(b" ")[1])

                    # Parse headers
                    content_length = 0
                    while True:
                        line = await reader.readline()
                        if line == b"\r\n":
                            break
                        if line.lower().startswith(b"content-length:"):
                            content_length = int(line.split(b":")[1].strip())

                    # Read body
                    body = await asyncio.wait_for(
                        reader.readexactly(content_length), timeout=30
                    )

                    if status >= 500 or status == 429:
                        raise _RetryableHTTPError(status, body)
                    return json.loads(body)
                finally:
                    writer.close()
                    await writer.wait_closed()
            except (OSError, asyncio.TimeoutError, _RetryableHTTPError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = min(retry_base_delay * (2 ** attempt), retry_max_delay)
                    delay *= random.uniform(0.5, 1.5)
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _cache_put(text: str, value: bytes) -> None:
        cache[text] = value
        cache.move_to_end(text)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def _l2_get(text: str) -> bytes | None:
        """Check L2 (SQLite) cache. Returns bytes or None."""
        if storage is None:
            return None
        return storage.load_embedding_cache(_text_hash(text), model)

    def _l2_put(text: str, value: bytes) -> None:
        """Write to L2 (SQLite) cache."""
        if storage is None:
            return
        storage.save_embedding_cache(_text_hash(text), model, value)

    def _maybe_quantize(data: bytes) -> bytes:
        return _quantize(data) if quantize else data

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
        # L1 cache hit
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        # L2 cache hit
        l2 = _l2_get(text)
        if l2 is not None:
            _cache_put(text, l2)
            return l2
        # Cache miss — call Ollama
        try:
            payload = json.dumps({"model": model, "input": text}).encode()
            data = _post(payload)
            embedding = data["embeddings"][0]
            vec = np.array(embedding, dtype=np.float32)
            result = _maybe_quantize(vec.tobytes())
            _cache_put(text, result)
            _l2_put(text, result)
            return result
        except (
            http.client.HTTPException, OSError, _RetryableHTTPError,
            KeyError, IndexError,
        ):
            result = _fallback_with_warning(text)
            _cache_put(text, result)
            return result

    def _batch(texts: list[str]) -> list[bytes]:
        """Embed multiple texts in a single HTTP call. Uses L1+L2 cache."""
        results: list[bytes | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            # L1
            if text in cache:
                cache.move_to_end(text)
                results[i] = cache[text]
                continue
            # L2
            l2 = _l2_get(text)
            if l2 is not None:
                _cache_put(text, l2)
                results[i] = l2
                continue
            to_fetch.append((i, text))

        if to_fetch:
            fetch_texts = [t for _, t in to_fetch]
            try:
                payload = json.dumps({"model": model, "input": fetch_texts}).encode()
                data = _post(payload)
                embeddings = data["embeddings"]
                for j, (idx, text) in enumerate(to_fetch):
                    vec = np.array(embeddings[j], dtype=np.float32)
                    result = _maybe_quantize(vec.tobytes())
                    _cache_put(text, result)
                    _l2_put(text, result)
                    results[idx] = result
            except (
                http.client.HTTPException, OSError, _RetryableHTTPError,
                KeyError, IndexError,
            ):
                for idx, text in to_fetch:
                    result = _fallback_with_warning(text)
                    _cache_put(text, result)
                    results[idx] = result

        return results  # type: ignore[return-value]

    # ── Async-native methods ──────────────────────────────────────────

    async def _async_embed(text: str) -> bytes:
        """Async-native single embed. Cache checks are sync (fast), HTTP is async."""
        # L1 cache hit
        if text in cache:
            cache.move_to_end(text)
            return cache[text]
        # L2 cache hit
        l2 = _l2_get(text)
        if l2 is not None:
            _cache_put(text, l2)
            return l2
        # Cache miss — async HTTP
        try:
            payload = json.dumps({"model": model, "input": text}).encode()
            data = await _async_post(payload)
            embedding = data["embeddings"][0]
            vec = np.array(embedding, dtype=np.float32)
            result = _maybe_quantize(vec.tobytes())
            _cache_put(text, result)
            _l2_put(text, result)
            return result
        except (
            OSError, asyncio.TimeoutError, _RetryableHTTPError,
            KeyError, IndexError,
        ):
            result = _fallback_with_warning(text)
            _cache_put(text, result)
            return result

    async def _async_batch(texts: list[str]) -> list[bytes]:
        """Async-native batch embed. Cache checks sync, HTTP async."""
        results: list[bytes | None] = [None] * len(texts)
        to_fetch: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            # L1
            if text in cache:
                cache.move_to_end(text)
                results[i] = cache[text]
                continue
            # L2
            l2 = _l2_get(text)
            if l2 is not None:
                _cache_put(text, l2)
                results[i] = l2
                continue
            to_fetch.append((i, text))

        if to_fetch:
            fetch_texts = [t for _, t in to_fetch]
            try:
                payload = json.dumps({"model": model, "input": fetch_texts}).encode()
                data = await _async_post(payload)
                embeddings = data["embeddings"]
                for j, (idx, text) in enumerate(to_fetch):
                    vec = np.array(embeddings[j], dtype=np.float32)
                    result = _maybe_quantize(vec.tobytes())
                    _cache_put(text, result)
                    _l2_put(text, result)
                    results[idx] = result
            except (
                OSError, asyncio.TimeoutError, _RetryableHTTPError,
                KeyError, IndexError,
            ):
                for idx, text in to_fetch:
                    result = _fallback_with_warning(text)
                    _cache_put(text, result)
                    results[idx] = result

        return results  # type: ignore[return-value]

    # ── Sync generators ───────────────────────────────────────────────

    def _stream_batch(texts: list[str], chunk_size: int = 8):
        """Yield list[bytes] as each mini-batch completes (sync generator)."""
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            yield _batch(chunk)

    async def _astream_batch(texts: list[str], chunk_size: int = 8,
                             concurrency: int = 2):
        """Yield list[bytes] per chunk, with N chunks in-flight (async generator).

        v0.5.0: uses native async HTTP instead of thread pool.
        """
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        sem = asyncio.Semaphore(concurrency)
        queue: asyncio.Queue = asyncio.Queue()

        async def _process(idx: int, chunk: list[str]) -> None:
            async with sem:
                result = await _async_batch(chunk)
                await queue.put((idx, result))

        tasks = [asyncio.create_task(_process(i, c)) for i, c in enumerate(chunks)]

        buffer: dict[int, list[bytes]] = {}
        next_idx = 0
        for _ in range(len(chunks)):
            idx, result = await queue.get()
            buffer[idx] = result
            while next_idx in buffer:
                yield buffer.pop(next_idx)
                next_idx += 1

        # Ensure all tasks are done
        await asyncio.gather(*tasks)

    # ── Attach methods to _embed ──────────────────────────────────────

    _embed.batch = _batch  # type: ignore[attr-defined]
    _embed.stream_batch = _stream_batch  # type: ignore[attr-defined]
    _embed.astream_batch = _astream_batch  # type: ignore[attr-defined]
    _embed.async_call = _async_embed  # type: ignore[attr-defined]
    _embed.async_batch = _async_batch  # type: ignore[attr-defined]
    return _embed


def ollama_embed_async(
    **kwargs,
) -> Callable:
    """Native async variant of ollama_embed. Returns an async embed function.

    v0.5.0: uses asyncio.open_connection for true async HTTP (no thread pool).

    Usage:
        embed = ollama_embed_async(model="nomic-embed-text")
        result = await embed("hello world")
        results = await embed.batch(["a", "b", "c"])
    """
    sync_embed = ollama_embed(**kwargs)

    async def _embed(text: str) -> bytes:
        return await sync_embed.async_call(text)

    async def _batch(texts: list[str]) -> list[bytes]:
        return await sync_embed.async_batch(texts)

    _embed.batch = _batch  # type: ignore[attr-defined]
    _embed.stream_batch = sync_embed.stream_batch  # type: ignore[attr-defined]
    _embed.astream_batch = sync_embed.astream_batch  # type: ignore[attr-defined]
    return _embed


def openai_embed(
    api_key: str,
    model: str = "text-embedding-3-small",
    storage: Storage | None = None,
    quantize: bool = False,
) -> EmbedFn:
    """OpenAI embedding provider.

    Uses urllib.request (stdlib only -- zero new dependencies).
    Requires an API key.

    v0.4.0: optional L2 cache (storage) and float16 quantization.
    """
    import numpy as np

    l1_cache: collections.OrderedDict[str, bytes] = collections.OrderedDict()

    def _embed(text: str) -> bytes:
        # L1
        if text in l1_cache:
            l1_cache.move_to_end(text)
            return l1_cache[text]
        # L2
        if storage is not None:
            l2 = storage.load_embedding_cache(_text_hash(text), model)
            if l2 is not None:
                l1_cache[text] = l2
                return l2
        # HTTP
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
        result = vec.tobytes()
        if quantize:
            result = _quantize(result)
        l1_cache[text] = result
        if storage is not None:
            storage.save_embedding_cache(_text_hash(text), model, result)
        return result

    return _embed


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two embedding byte vectors.

    Auto-detects float32 vs float16 encoding (handles mixed dtypes).
    Raises ValueError on dimension mismatch.
    """
    import numpy as np

    # Try float32 for both first (most common case)
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)

    if va.shape != vb.shape:
        # One or both might be float16 — try all combinations
        va_16 = np.frombuffer(a, dtype=np.float16).astype(np.float32)
        vb_16 = np.frombuffer(b, dtype=np.float16).astype(np.float32)

        if va.shape == vb_16.shape:
            vb = vb_16
        elif va_16.shape == vb.shape:
            va = va_16
        elif va_16.shape == vb_16.shape:
            va, vb = va_16, vb_16
        else:
            raise ValueError(
                f"Embedding dimension mismatch: {va.shape[0]}d vs {vb.shape[0]}d. "
                f"Do not mix providers (numpy=256d, Ollama=768d)."
            )

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

    Used by consolidation in reflect(). O(n^2) but only on alive memories.
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
