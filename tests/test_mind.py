"""Tests for kore-mind core."""

import os
import tempfile
import time

import pytest

from kore_mind import Mind, Memory, Identity
from kore_mind.models import MemoryType
from kore_mind.decay import compute_decay
from kore_mind.consolidate import text_similarity, merge_memories, consolidate


@pytest.fixture
def mind():
    """Mind temporal que se limpia al terminar."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path)
    yield m
    m.close()
    os.unlink(path)


# ── Experience ─────────────────────────────────────────────────────────


class TestExperience:
    def test_basic(self, mind):
        mem = mind.experience("User likes math")
        assert isinstance(mem, Memory)
        assert mem.content == "User likes math"
        assert mem.salience == 1.0
        assert mind.count == 1

    def test_with_metadata(self, mind):
        mem = mind.experience(
            "Prefers short answers",
            type="semantic",
            source="carlos",
            tags=["preference", "style"],
        )
        assert mem.type == MemoryType.SEMANTIC
        assert mem.source == "carlos"
        assert "style" in mem.tags

    def test_multiple(self, mind):
        mind.experience("Fact 1")
        mind.experience("Fact 2")
        mind.experience("Fact 3")
        assert mind.count == 3


# ── Recall ─────────────────────────────────────────────────────────────


class TestRecall:
    def test_recall_all(self, mind):
        mind.experience("Python is great")
        mind.experience("Rust is fast")
        memories = mind.recall()
        assert len(memories) == 2

    def test_recall_with_query(self, mind):
        mind.experience("User works on P vs NP")
        mind.experience("User likes coffee")
        memories = mind.recall("NP")
        assert len(memories) == 2
        assert "NP" in memories[0].content

    def test_recall_reinforces(self, mind):
        mem = mind.experience("Important fact")
        original_count = mem.access_count
        mind.recall("Important")
        refreshed = mind.recall("Important")
        assert refreshed[0].access_count > original_count

    def test_recall_respects_salience(self, mind):
        mind.experience("High salience", salience=1.0)
        mind.experience("Low salience", salience=0.01)
        memories = mind.recall(min_salience=0.5)
        assert len(memories) == 1


# ── Decay ──────────────────────────────────────────────────────────────


class TestDecay:
    def test_decay_reduces_salience(self):
        mem = Memory(content="Old memory", salience=1.0)
        future = time.time() + 30 * 24 * 3600
        decayed = compute_decay(mem, now=future)
        assert decayed < mem.salience

    def test_accessed_memories_decay_slower(self):
        mem_unused = Memory(content="Unused", salience=1.0)
        mem_used = Memory(content="Used", salience=1.0, access_count=10)
        future = time.time() + 14 * 24 * 3600
        decay_unused = compute_decay(mem_unused, now=future)
        decay_used = compute_decay(mem_used, now=future)
        assert decay_used > decay_unused

    def test_zero_elapsed_no_decay(self):
        mem = Memory(content="Fresh", salience=0.8)
        decayed = compute_decay(mem, now=mem.last_accessed)
        assert decayed == 0.8


# ── Consolidation ──────────────────────────────────────────────────────


class TestConsolidation:
    def test_text_similarity(self):
        assert text_similarity("user likes python", "user likes python") == 1.0
        assert text_similarity("user likes python", "user likes java") > 0.5
        assert text_similarity("hello", "completely different") < 0.3

    def test_merge_memories(self):
        a = Memory(content="User likes Python", salience=0.8,
                   access_count=5, tags=["python"])
        b = Memory(content="User likes Python programming", salience=0.6,
                   access_count=3, tags=["programming"])
        merged = merge_memories(a, b)
        assert merged.salience == min(1.0, 0.8 + 0.1)  # consolidation bonus
        assert merged.access_count == 8
        assert "python" in merged.tags
        assert "programming" in merged.tags

    def test_consolidate_similar(self):
        mems = [
            Memory(content="User prefers Python for data science", tags=["python"]),
            Memory(content="User prefers Python for data analysis", tags=["python"]),
            Memory(content="User likes coffee in the morning", tags=["coffee"]),
        ]
        consolidated, deleted = consolidate(mems, threshold=0.6)
        # The two Python memories should merge
        assert len(deleted) == 1
        assert len(consolidated) == 2

    def test_consolidate_nothing_similar(self):
        mems = [
            Memory(content="Python is great"),
            Memory(content="The weather is sunny"),
        ]
        consolidated, deleted = consolidate(mems, threshold=0.7)
        assert len(deleted) == 0
        assert len(consolidated) == 2

    def test_consolidate_empty(self):
        consolidated, deleted = consolidate([], threshold=0.7)
        assert consolidated == []
        assert deleted == []


# ── Reflect ────────────────────────────────────────────────────────────


class TestReflect:
    def test_reflect_returns_identity(self, mind):
        mind.experience("User studies complexity theory", tags=["math", "cs"])
        mind.experience("User codes in Python", tags=["python", "coding"])
        identity = mind.reflect()
        assert isinstance(identity, Identity)
        assert identity.summary != ""

    def test_reflect_with_custom_summarizer(self, mind):
        mind.experience("Test memory")

        def custom_summarizer(memories):
            return Identity(summary=f"Custom: {len(memories)} mems")

        identity = mind.reflect(summarizer=custom_summarizer)
        assert "Custom:" in identity.summary

    def test_reflect_cleans_dead_memories(self, mind):
        mem = mind.experience("Dying memory", salience=0.005)
        assert mind.count == 1
        mind.reflect()
        assert mind.count == 0

    def test_reflect_consolidates(self, mind):
        mind.experience("User works with Python daily")
        mind.experience("User works with Python every day")
        mind.experience("User enjoys hiking on weekends")
        assert mind.count == 3
        mind.reflect()
        # The two Python memories should consolidate
        assert mind.count == 2


# ── Embeddings ─────────────────────────────────────────────────────────


class TestEmbeddings:
    @pytest.fixture
    def mind_with_embeddings(self):
        """Mind with a simple hash-based pseudo-embedder for testing."""
        import hashlib
        import struct

        def simple_embed(text: str) -> bytes:
            """Deterministic pseudo-embedding for testing. NOT semantic."""
            h = hashlib.sha256(text.encode()).digest()
            # Convert to 8 float32 values
            values = struct.unpack('8f', h)
            return struct.pack(f'{len(values)}f', *values)

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        m = Mind(path, embed_fn=simple_embed)
        yield m
        m.close()
        os.unlink(path)

    def test_experience_creates_embedding(self, mind_with_embeddings):
        mem = mind_with_embeddings.experience("Test content")
        assert mem.embedding is not None
        assert len(mem.embedding) > 0

    def test_embedding_persists(self, mind_with_embeddings):
        mind_with_embeddings.experience("Persistent embedding test")
        memories = mind_with_embeddings.recall()
        assert memories[0].embedding is not None

    def test_no_embedding_without_fn(self, mind):
        mem = mind.experience("No embedding")
        assert mem.embedding is None


# ── Forget ─────────────────────────────────────────────────────────────


class TestForget:
    def test_forget_removes_low_salience(self, mind):
        mind.experience("Strong", salience=1.0)
        mind.experience("Weak", salience=0.05)
        forgotten = mind.forget(threshold=0.1)
        assert forgotten == 1
        assert mind.count == 1

    def test_forget_nothing(self, mind):
        mind.experience("Strong", salience=1.0)
        forgotten = mind.forget(threshold=0.01)
        assert forgotten == 0


# ── Identity ───────────────────────────────────────────────────────────


class TestIdentity:
    def test_identity_empty(self, mind):
        identity = mind.identity()
        assert isinstance(identity, Identity)

    def test_identity_persists_after_reflect(self, mind):
        mind.experience("Focus on AI safety", tags=["ai", "safety"])
        mind.reflect()
        identity = mind.identity()
        assert identity.summary != ""
        assert identity.traits


# ── Context Manager ────────────────────────────────────────────────────


class TestContextManager:
    def test_with_statement(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with Mind(path) as m:
            m.experience("test")
            assert m.count == 1
        os.unlink(path)
