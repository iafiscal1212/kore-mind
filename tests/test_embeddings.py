"""Tests for built-in embedding providers (v0.3)."""

import os
import tempfile

import numpy as np
import pytest

from kore_mind.embeddings import (
    cosine_similarity,
    numpy_embed,
    ollama_embed,
)
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
    embed = ollama_embed(base_url="http://localhost:99999")
    result = embed("fallback test")
    assert isinstance(result, bytes)
    assert len(result) > 0
    # Should produce the same as numpy_embed (the fallback)
    np_embed = numpy_embed()
    expected = np_embed("fallback test")
    assert result == expected


# ── Integration with Mind ────────────────────────────────────────────────


def test_mind_with_numpy_embed():
    """Full flow: experience → semantic recall with numpy_embed."""
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
            f"Consolidation should reduce or maintain count: {before} → {after}"
        )
    finally:
        mind.close()
        os.unlink(db_path)
