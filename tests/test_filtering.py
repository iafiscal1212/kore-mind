"""Tests for per-user filtering (v0.2)."""

import os
import tempfile

import pytest

from kore_mind import Mind


@pytest.fixture
def mind():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path)
    yield m
    m.close()
    os.unlink(path)


class TestSourceFiltering:
    def test_memories_by_source(self, mind):
        mind.experience("Fact for alice", source="alice")
        mind.experience("Fact for bob", source="bob")
        mind.experience("Fact for alice 2", source="alice")

        alice_mems = mind._storage.memories_by_source("alice")
        assert len(alice_mems) == 2
        assert all(m.source == "alice" for m in alice_mems)

    def test_top_memories_source_filter(self, mind):
        mind.experience("Alice fact", source="alice")
        mind.experience("Bob fact", source="bob")

        top = mind._storage.top_memories(limit=10, source="alice")
        assert len(top) == 1
        assert top[0].source == "alice"

    def test_recall_with_source(self, mind):
        mind.experience("Python for alice", source="alice")
        mind.experience("Python for bob", source="bob")

        result = mind.recall("Python", source="alice")
        assert len(result) == 1
        assert result[0].source == "alice"

    def test_recall_without_source_returns_all(self, mind):
        mind.experience("Fact A", source="alice")
        mind.experience("Fact B", source="bob")

        # source="" means no filter (default_source is "")
        result = mind.recall()
        assert len(result) == 2


class TestDefaultSource:
    def test_default_source_on_experience(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        m = Mind(path, default_source="carlos")
        m.experience("Test fact")
        mems = m.recall(source="carlos")
        assert len(mems) == 1
        assert mems[0].source == "carlos"
        m.close()
        os.unlink(path)

    def test_explicit_source_overrides_default(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        m = Mind(path, default_source="carlos")
        m.experience("Override test", source="ana")
        mems = m._storage.memories_by_source("ana")
        assert len(mems) == 1
        m.close()
        os.unlink(path)


class TestScoped:
    def test_scoped_creates_filtered_view(self, mind):
        mind.experience("Global fact", source="global")
        mind.experience("Alice fact", source="alice")

        alice_mind = mind.scoped("alice")
        result = alice_mind.recall()
        assert len(result) == 1
        assert result[0].source == "alice"

    def test_scoped_experience_uses_source(self, mind):
        alice = mind.scoped("alice")
        alice.experience("Alice remembers this")

        mems = mind._storage.memories_by_source("alice")
        assert len(mems) == 1

    def test_scoped_shares_db(self, mind):
        mind.experience("Shared fact", source="shared")
        alice = mind.scoped("alice")
        alice.experience("Alice's fact")

        # Both are in the same DB
        assert mind.count == 2
