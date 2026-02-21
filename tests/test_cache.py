"""Tests for cache storage (v0.2)."""

import os
import tempfile
import time

import pytest

from kore_mind.models import CacheEntry
from kore_mind.storage import Storage


@pytest.fixture
def storage():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Storage(path)
    yield s
    s.close()
    os.unlink(path)


class TestCacheStorage:
    def test_save_and_find(self, storage):
        entry = CacheEntry(
            query="hello world",
            response="Hello!",
            query_hash="abc123",
            source="user1",
        )
        storage.save_cache_entry(entry)

        found = storage.find_cache_by_hash("abc123", source="user1")
        assert found is not None
        assert found.query == "hello world"
        assert found.response == "Hello!"

    def test_find_nonexistent(self, storage):
        found = storage.find_cache_by_hash("nonexistent")
        assert found is None

    def test_cache_hit_increments(self, storage):
        entry = CacheEntry(
            query="test",
            response="response",
            query_hash="hash1",
        )
        storage.save_cache_entry(entry)

        storage.cache_hit(entry.id)
        storage.cache_hit(entry.id)

        found = storage.find_cache_by_hash("hash1")
        assert found.hit_count == 2

    def test_delete_expired(self, storage):
        old = CacheEntry(
            query="old",
            response="old response",
            query_hash="old_hash",
            created_at=time.time() - 7200,
            ttl=3600,
        )
        fresh = CacheEntry(
            query="fresh",
            response="fresh response",
            query_hash="fresh_hash",
            ttl=3600,
        )
        storage.save_cache_entry(old)
        storage.save_cache_entry(fresh)

        deleted = storage.delete_expired_cache()
        assert deleted == 1

        assert storage.find_cache_by_hash("old_hash") is None
        assert storage.find_cache_by_hash("fresh_hash") is not None

    def test_source_filter(self, storage):
        entry_a = CacheEntry(
            query="q", response="a", query_hash="same",
            source="alice",
        )
        entry_b = CacheEntry(
            query="q", response="b", query_hash="same",
            source="bob",
        )
        storage.save_cache_entry(entry_a)
        storage.save_cache_entry(entry_b)

        found = storage.find_cache_by_hash("same", source="alice")
        assert found.response == "a"

        found = storage.find_cache_by_hash("same", source="bob")
        assert found.response == "b"
