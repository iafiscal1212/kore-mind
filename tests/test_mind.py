"""Tests for kore-mind core."""

import os
import tempfile
import time

import pytest

from kore_mind import Mind, Memory, Identity
from kore_mind.models import MemoryType
from kore_mind.decay import compute_decay


@pytest.fixture
def mind():
    """Mind temporal que se limpia al terminar."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path)
    yield m
    m.close()
    os.unlink(path)


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
        assert "NP" in memories[0].content  # más relevante primero

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


class TestDecay:
    def test_decay_reduces_salience(self):
        mem = Memory(content="Old memory", salience=1.0)
        # Simular que pasó mucho tiempo
        future = time.time() + 30 * 24 * 3600  # 30 días
        decayed = compute_decay(mem, now=future)
        assert decayed < mem.salience

    def test_accessed_memories_decay_slower(self):
        mem_unused = Memory(content="Unused", salience=1.0)
        mem_used = Memory(content="Used", salience=1.0, access_count=10)
        future = time.time() + 14 * 24 * 3600  # 14 días
        decay_unused = compute_decay(mem_unused, now=future)
        decay_used = compute_decay(mem_used, now=future)
        assert decay_used > decay_unused

    def test_zero_elapsed_no_decay(self):
        mem = Memory(content="Fresh", salience=0.8)
        decayed = compute_decay(mem, now=mem.last_accessed)
        assert decayed == 0.8


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
        # Crear recuerdo con salience ya muy baja
        mem = mind.experience("Dying memory", salience=0.005)
        assert mind.count == 1
        mind.reflect()
        # Con salience 0.005 y death_threshold 0.01, debería morir
        assert mind.count == 0


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


class TestIdentity:
    def test_identity_empty(self, mind):
        identity = mind.identity()
        assert isinstance(identity, Identity)

    def test_identity_persists_after_reflect(self, mind):
        mind.experience("Focus on AI safety", tags=["ai", "safety"])
        mind.reflect()
        identity = mind.identity()
        assert identity.summary != ""
        assert identity.traits  # algo debería haber emergido


class TestContextManager:
    def test_with_statement(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with Mind(path) as m:
            m.experience("test")
            assert m.count == 1
        os.unlink(path)
