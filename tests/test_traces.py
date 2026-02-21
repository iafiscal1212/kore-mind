"""Tests for observability / traces (v0.2)."""

import os
import tempfile

import pytest

from kore_mind import Mind
from kore_mind.models import Trace


@pytest.fixture
def mind():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path, enable_traces=True)
    yield m
    m.close()
    os.unlink(path)


@pytest.fixture
def mind_no_traces():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    m = Mind(path, enable_traces=False)
    yield m
    m.close()
    os.unlink(path)


class TestTraces:
    def test_experience_creates_trace(self, mind):
        mind.experience("Test content")
        traces = mind.traces(operation="experience")
        assert len(traces) == 1
        assert traces[0].operation == "experience"
        assert traces[0].duration_ms is not None

    def test_recall_creates_trace(self, mind):
        mind.experience("Something")
        mind.recall("Something")
        traces = mind.traces(operation="recall")
        assert len(traces) == 1

    def test_reflect_creates_trace(self, mind):
        mind.experience("Memory")
        mind.reflect()
        traces = mind.traces(operation="reflect")
        assert len(traces) == 1

    def test_forget_creates_trace(self, mind):
        mind.experience("Weak", salience=0.05)
        mind.forget(threshold=0.1)
        traces = mind.traces(operation="forget")
        assert len(traces) == 1

    def test_no_traces_when_disabled(self, mind_no_traces):
        mind_no_traces.experience("Test")
        mind_no_traces.recall()
        traces = mind_no_traces.traces()
        assert len(traces) == 0

    def test_trace_filter_by_source(self, mind):
        mind.experience("Alice fact", source="alice")
        mind.experience("Bob fact", source="bob")
        traces = mind.traces(source="alice")
        assert len(traces) == 1

    def test_trace_has_duration(self, mind):
        mind.experience("Timed operation")
        traces = mind.traces()
        assert traces[0].duration_ms >= 0

    def test_trace_limit(self, mind):
        for i in range(10):
            mind.experience(f"Fact {i}")
        traces = mind.traces(limit=5)
        assert len(traces) == 5

    def test_zero_overhead_by_default(self):
        """Verify enable_traces=False is the default."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        m = Mind(path)
        assert m._enable_traces is False
        m.close()
        os.unlink(path)
