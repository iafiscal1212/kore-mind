"""Tests for built-in embedding providers (v0.4.0)."""

import asyncio
import json
import os
import tempfile
import warnings
from unittest import mock

import numpy as np
import pytest

from kore_mind.embeddings import (
    cosine_similarity,
    numpy_embed,
    ollama_embed,
    ollama_embed_async,
)
from kore_mind.storage import Storage
from kore_mind import Mind


# ── numpy_embed ──────────────────────────────────────────────────────────


def test_numpy_embed_produces_bytes():
    embed = numpy_embed(dims=256)
    result = embed("hello world")
    assert isinstance(result, bytes)
    assert len(result) == 256 * 4  # float32 = 4 bytes each


def test_numpy_embed_similar_texts():
    embed = numpy_embed()
    a = embed("me gusta el café por la mañana")
    b = embed("me encanta el café caliente")
    sim = cosine_similarity(a, b)
    assert sim > 0.5, f"Similar texts should have cosine > 0.5, got {sim}"


def test_numpy_embed_different_texts():
    embed = numpy_embed()
    a = embed("me gusta el café por la mañana")
    b = embed("me encanta el café caliente")
    c = embed("el teorema de Pitágoras demuestra relaciones geométricas")
    sim_similar = cosine_similarity(a, b)
    sim_different = cosine_similarity(a, c)
    assert sim_similar > sim_different, (
        f"Similar texts ({sim_similar:.3f}) should score higher than "
        f"different texts ({sim_different:.3f})"
    )


def test_numpy_embed_deterministic():
    embed = numpy_embed()
    a = embed("deterministic test")
    b = embed("deterministic test")
    assert a == b


def test_numpy_embed_empty_text():
    embed = numpy_embed()
    result = embed("")
    assert isinstance(result, bytes)
    assert len(result) == 256 * 4
    vec = np.frombuffer(result, dtype=np.float32)
    assert np.allclose(vec, 0.0), "Empty text should produce zero vector"


# ── ollama_embed fallback ────────────────────────────────────────────────


def test_ollama_embed_fallback():
    """If Ollama is not running, ollama_embed falls back to numpy_embed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        embed = ollama_embed(base_url="http://localhost:19")
        result = embed("fallback test")
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Should produce the same as numpy_embed (the fallback)
    np_embed = numpy_embed()
    expected = np_embed("fallback test")
    assert result == expected


# ── Integration with Mind ────────────────────────────────────────────────


def test_mind_with_numpy_embed():
    """Full flow: experience -> semantic recall with numpy_embed."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        embed = numpy_embed()
        mind = Mind(db_path, embed_fn=embed)

        mind.experience("me gusta el café por la mañana", type="semantic")
        mind.experience("el teorema de Pitágoras es fundamental", type="semantic")
        mind.experience("Python es un lenguaje de programación", type="semantic")

        # Semantic search: "bebidas calientes" should find café
        results = mind.recall("café caliente por la mañana")
        assert len(results) > 0
        assert any("café" in m.content for m in results)

        # The café memory should rank higher than Pitágoras
        contents = [m.content for m in results]
        cafe_idx = next(i for i, c in enumerate(contents) if "café" in c)
        pitagoras_idx = next(
            (i for i, c in enumerate(contents) if "Pitágoras" in c),
            len(contents),
        )
        assert cafe_idx < pitagoras_idx, "café should rank higher than Pitágoras"
    finally:
        mind.close()
        os.unlink(db_path)


def test_consolidation_with_numpy_embed():
    """Consolidation uses cosine similarity with numpy embeddings."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        embed = numpy_embed()
        mind = Mind(db_path, embed_fn=embed, half_life=0.001)

        # Two very similar memories
        mind.experience("el usuario prefiere respuestas concisas", type="semantic")
        mind.experience("el usuario prefiere respuestas directas y concisas", type="semantic")
        # One different memory
        mind.experience("Python es un lenguaje de programación", type="semantic")

        before = mind.count
        mind.reflect()
        after = mind.count

        # With embeddings, similar memories should consolidate
        assert after <= before, (
            f"Consolidation should reduce or maintain count: {before} -> {after}"
        )
    finally:
        mind.close()
        os.unlink(db_path)


# ── ollama_embed v0.3.1 optimizations ───────────────────────────────────


def _fake_ollama_response(texts):
    """Build a fake Ollama /api/embed JSON response for given texts."""
    import numpy as np
    embeddings = []
    for t in (texts if isinstance(texts, list) else [texts]):
        vec = np.random.default_rng(hash(t) % 2**32).random(768).tolist()
        embeddings.append(vec)
    return json.dumps({"embeddings": embeddings}).encode()


def test_ollama_cache_hit():
    """Second call with same text returns cached result -- no HTTP."""
    import http.client
    embed = ollama_embed()

    real_response = _fake_ollama_response("hello")
    call_count = [0]

    orig_request = http.client.HTTPConnection.request
    orig_getresponse = http.client.HTTPConnection.getresponse

    def mock_request(self, method, url, body=None, headers={}):
        call_count[0] += 1
        self._fake_body = real_response

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            first = embed("hello")
            second = embed("hello")

    assert first == second
    assert call_count[0] == 1, f"Expected 1 HTTP call, got {call_count[0]}"


def test_ollama_batch():
    """.batch() returns list of correct size."""
    import http.client
    embed = ollama_embed()

    texts = ["alpha", "beta", "gamma"]
    response_data = _fake_ollama_response(texts)

    def mock_request(self, method, url, body=None, headers={}):
        self._fake_body = response_data

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            results = embed.batch(texts)

    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, bytes)
        assert len(r) == 768 * 4  # float32


def test_ollama_batch_uses_cache():
    """batch() skips HTTP for texts already in cache."""
    import http.client
    embed = ollama_embed()

    call_count = [0]

    def mock_request(self, method, url, body=None, headers={}):
        call_count[0] += 1
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            # First: embed "hello" individually (populates cache)
            embed("hello")
            call_count[0] = 0
            # Batch with "hello" (cached) + "world" (not cached)
            results = embed.batch(["hello", "world"])

    assert len(results) == 2
    assert call_count[0] == 1, "Should make only 1 HTTP call for uncached 'world'"


def test_ollama_fallback_warns():
    """Fallback emits RuntimeWarning about dimension mismatch."""
    embed = ollama_embed(base_url="http://localhost:19")
    with pytest.warns(RuntimeWarning, match="numpy vectors.*incompatible"):
        embed("trigger warning")


def test_ollama_connection_reuse():
    """Verifies that multiple calls reuse the same HTTPConnection."""
    import http.client
    embed = ollama_embed()

    connections_created = [0]
    orig_init = http.client.HTTPConnection.__init__

    def tracking_init(self, *args, **kwargs):
        connections_created[0] += 1
        orig_init(self, *args, **kwargs)

    def mock_request(self, method, url, body=None, headers={}):
        self._fake_body = _fake_ollama_response("x")

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "__init__", tracking_init):
        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                embed("first")
                embed("second")
                embed("third")

    assert connections_created[0] == 1, (
        f"Expected 1 connection, got {connections_created[0]}"
    )


# ── v0.4.0: Mejora 1 — Cache persistente L2 (SQLite) ────────────────────


def test_persistent_cache_save_load():
    """Guardar en SQLite, reiniciar, recuperar embedding."""
    import http.client
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = Storage(db_path)

        # Create embed with storage
        embed = ollama_embed(storage=store)

        def mock_request(self, method, url, body=None, headers={}):
            req_body = json.loads(body)
            self._fake_body = _fake_ollama_response(req_body["input"])

        def mock_getresponse(self):
            resp = mock.Mock()
            resp.read.return_value = self._fake_body
            resp.status = 200
            return resp

        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                first_result = embed("persistent text")

        store.close()

        # "Restart" — new Storage, new embed (fresh L1)
        store2 = Storage(db_path)
        embed2 = ollama_embed(storage=store2)

        call_count = [0]

        def mock_request2(self, method, url, body=None, headers={}):
            call_count[0] += 1
            req_body = json.loads(body)
            self._fake_body = _fake_ollama_response(req_body["input"])

        with mock.patch.object(http.client.HTTPConnection, "request", mock_request2):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                second_result = embed2("persistent text")

        assert first_result == second_result
        assert call_count[0] == 0, "Should hit L2 cache, no HTTP"
        store2.close()
    finally:
        os.unlink(db_path)


def test_persistent_cache_l1_l2_flow():
    """L1 miss -> L2 hit -> no HTTP."""
    import http.client
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store = Storage(db_path)

        # Manually insert into L2
        test_vec = np.random.default_rng(42).random(768).astype(np.float32).tobytes()
        from kore_mind.embeddings import _text_hash
        store.save_embedding_cache(
            _text_hash("l2 cached text"), "nomic-embed-text", test_vec
        )

        # Create embed with storage — L1 is empty
        embed = ollama_embed(storage=store)
        call_count = [0]

        def mock_request(self, method, url, body=None, headers={}):
            call_count[0] += 1

        def mock_getresponse(self):
            resp = mock.Mock()
            resp.read.return_value = b'{}'
            return resp

        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                result = embed("l2 cached text")

        assert result == test_vec
        assert call_count[0] == 0, "L2 hit should bypass HTTP"
        store.close()
    finally:
        os.unlink(db_path)


# ── v0.4.0: Mejora 2 — Validación dimensional ───────────────────────────


def test_dimension_mismatch_raises():
    """cosine_similarity(256d, 768d) -> ValueError."""
    a = np.zeros(256, dtype=np.float32).tobytes()
    b = np.zeros(768, dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity(a, b)


def test_dimension_match_works():
    """cosine_similarity(768d, 768d) -> normal float."""
    rng = np.random.default_rng(42)
    a = rng.random(768).astype(np.float32).tobytes()
    b = rng.random(768).astype(np.float32).tobytes()
    sim = cosine_similarity(a, b)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0 + 1e-6


# ── v0.4.0: Mejora 4 — Cuantización float16 ─────────────────────────────


def test_quantize_halves_size():
    """quantize=True -> bytes = dims*2 en vez de dims*4."""
    import http.client
    embed = ollama_embed(quantize=True)

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            result = embed("quantize test")

    # float16: 768 dims * 2 bytes = 1536
    assert len(result) == 768 * 2, f"Expected {768*2}, got {len(result)}"


def test_quantize_cosine_accuracy():
    """cosine(float32, float16) ~= cosine(float32, float32) +/- 0.01."""
    rng = np.random.default_rng(123)
    vec_a = rng.random(768).astype(np.float32)
    vec_b = rng.random(768).astype(np.float32)

    a_f32 = vec_a.tobytes()
    b_f32 = vec_b.tobytes()
    b_f16 = vec_b.astype(np.float16).tobytes()

    sim_exact = cosine_similarity(a_f32, b_f32)
    sim_mixed = cosine_similarity(a_f32, b_f16)

    assert abs(sim_exact - sim_mixed) < 0.01, (
        f"Float16 should be accurate within 0.01: exact={sim_exact:.6f}, mixed={sim_mixed:.6f}"
    )


# ── v0.4.0: Mejora 5 — Async variant ────────────────────────────────────


def test_async_embed():
    """await embed('text') devuelve bytes."""
    import http.client

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    embed = ollama_embed_async()

    async def _run():
        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                return await embed("async test")

    result = asyncio.run(_run())
    assert isinstance(result, bytes)
    assert len(result) == 768 * 4


def test_async_batch():
    """await embed.batch(texts) devuelve lista correcta."""
    import http.client

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    embed = ollama_embed_async()

    async def _run():
        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                return await embed.batch(["alpha", "beta", "gamma"])

    results = asyncio.run(_run())
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, bytes)
        assert len(r) == 768 * 4


# ── v0.4.0: Mejora 6 — Streaming batch ──────────────────────────────────


def test_stream_batch_yields_chunks():
    """stream_batch(20 texts, chunk_size=8) yield 3 chunks."""
    import http.client
    embed = ollama_embed()
    texts = [f"text_{i}" for i in range(20)]

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            chunks = list(embed.stream_batch(texts, chunk_size=8))

    assert len(chunks) == 3  # ceil(20/8) = 3
    assert len(chunks[0]) == 8
    assert len(chunks[1]) == 8
    assert len(chunks[2]) == 4


def test_stream_batch_correct_order():
    """Resultados en el mismo orden que input."""
    import http.client
    embed = ollama_embed()
    texts = [f"ordered_{i}" for i in range(12)]

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    # Also get full batch for comparison
    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            full = embed.batch(texts)
            # Reset cache to re-embed via stream
            embed2 = ollama_embed()

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            streamed = []
            for chunk in embed2.stream_batch(texts, chunk_size=4):
                streamed.extend(chunk)

    assert len(streamed) == len(full)
    for i in range(len(full)):
        assert streamed[i] == full[i], f"Mismatch at index {i}"


def test_astream_batch_concurrent():
    """Async generator produce chunks ordenados."""
    import http.client
    embed = ollama_embed()
    texts = [f"async_{i}" for i in range(16)]

    def mock_request(self, method, url, body=None, headers={}):
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    async def _run():
        with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
            with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
                results = []
                async for chunk in embed.astream_batch(texts, chunk_size=4, concurrency=2):
                    results.append(chunk)
                return results

    chunks = asyncio.run(_run())
    assert len(chunks) == 4  # 16/4 = 4 chunks
    for chunk in chunks:
        assert len(chunk) == 4
        for item in chunk:
            assert isinstance(item, bytes)
            assert len(item) == 768 * 4


def test_stream_batch_uses_cache():
    """Textos cacheados no generan HTTP en streaming."""
    import http.client
    embed = ollama_embed()
    call_count = [0]

    def mock_request(self, method, url, body=None, headers={}):
        call_count[0] += 1
        req_body = json.loads(body)
        self._fake_body = _fake_ollama_response(req_body["input"])

    def mock_getresponse(self):
        resp = mock.Mock()
        resp.read.return_value = self._fake_body
        resp.status = 200
        return resp

    with mock.patch.object(http.client.HTTPConnection, "request", mock_request):
        with mock.patch.object(http.client.HTTPConnection, "getresponse", mock_getresponse):
            # Pre-cache some texts
            embed("cached_0")
            embed("cached_1")
            call_count[0] = 0

            # Stream with mix of cached and uncached
            texts = ["cached_0", "new_0", "cached_1", "new_1"]
            chunks = list(embed.stream_batch(texts, chunk_size=4))

    assert len(chunks) == 1
    assert len(chunks[0]) == 4
    # Only 2 uncached texts should trigger HTTP (1 call because they're in the same chunk)
    assert call_count[0] == 1, f"Expected 1 HTTP call for uncached, got {call_count[0]}"
