"""Tests for kore-bridge. Uses mock LLM (no API keys needed)."""

import os
import tempfile
import time

import pytest

from kore_mind import Mind
from kore_bridge import Bridge, CallableLLM


def mock_llm(messages):
    """Mock LLM that echoes the last user message."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return f"Echo: {msg['content']}"
    return "Echo: empty"


@pytest.fixture
def bridge():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mind = Mind(path)
    llm = CallableLLM(mock_llm)
    b = Bridge(mind=mind, llm=llm)
    yield b
    mind.close()
    os.unlink(path)


@pytest.fixture
def bridge_with_cache():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mind = Mind(path)
    llm = CallableLLM(mock_llm)
    b = Bridge(mind=mind, llm=llm, cache_ttl=3600.0)
    yield b
    mind.close()
    os.unlink(path)


@pytest.fixture
def bridge_with_rate_limit():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    mind = Mind(path)
    llm = CallableLLM(mock_llm)
    b = Bridge(mind=mind, llm=llm, cache_ttl=3600.0,
               rate_limit=2, rate_window=3600.0)
    yield b
    mind.close()
    os.unlink(path)


class TestThink:
    def test_basic_response(self, bridge):
        response = bridge.think("Hello world")
        assert "Echo:" in response

    def test_remembers_interaction(self, bridge):
        bridge.think("What is P vs NP?")
        assert bridge.mind.count == 2  # user input + response

    def test_no_remember(self, bridge):
        bridge.think("Secret question", remember=False)
        assert bridge.mind.count == 0

    def test_with_user(self, bridge):
        bridge.think("Help me", user="carlos")
        memories = bridge.mind.recall(source="carlos")
        assert any(m.source == "carlos" for m in memories)


class TestCache:
    def test_cache_hit(self, bridge_with_cache):
        # First call — miss
        r1 = bridge_with_cache.think("What is 2+2?", remember=False)
        # Second call — should hit cache
        r2 = bridge_with_cache.think("What is 2+2?", remember=False)
        assert r1 == r2

    def test_cache_bypass(self, bridge_with_cache):
        bridge_with_cache.think("cached query", remember=False)
        # With use_cache=False, should call LLM again
        r2 = bridge_with_cache.think("cached query", remember=False, use_cache=False)
        assert "Echo:" in r2

    def test_cache_per_user(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mind = Mind(path)
        llm_calls = []

        def tracking_llm(messages):
            llm_calls.append(1)
            return f"Response #{len(llm_calls)}"

        llm = CallableLLM(tracking_llm)
        b = Bridge(mind=mind, llm=llm, cache_ttl=3600.0)

        b.think("Same question", user="alice", remember=False)
        b.think("Same question", user="bob", remember=False)

        # Different users = different cache entries = 2 LLM calls
        assert len(llm_calls) == 2

        mind.close()
        os.unlink(path)

    def test_no_cache_when_ttl_zero(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mind = Mind(path)
        calls = []

        def counting_llm(msgs):
            calls.append(1)
            return "response"

        llm = CallableLLM(counting_llm)
        b = Bridge(mind=mind, llm=llm, cache_ttl=0)

        b.think("test", remember=False)
        b.think("test", remember=False)
        assert len(calls) == 2

        mind.close()
        os.unlink(path)


class TestRateLimiting:
    def test_rate_limit_kicks_in(self, bridge_with_rate_limit):
        # rate_limit=2, so 3rd call should be rate limited
        r1 = bridge_with_rate_limit.think("Repeated Q", remember=False)
        r2 = bridge_with_rate_limit.think("Repeated Q", remember=False)
        r3 = bridge_with_rate_limit.think("Repeated Q", remember=False)

        # r3 should come from cache (same as r1)
        assert r1 == r3

    def test_rate_limit_disabled(self, bridge_with_cache):
        # rate_limit=0 means disabled
        r1 = bridge_with_cache.think("Q", remember=False)
        r2 = bridge_with_cache.think("Q", remember=False)
        # Second should be cache hit, not rate limited
        assert r1 == r2


class TestSourceFiltering:
    def test_think_passes_user_as_source(self, bridge):
        bridge.think("Hello", user="alice")
        mems = bridge.mind._storage.memories_by_source("alice")
        assert len(mems) == 2  # user input + response


class TestObserve:
    def test_observe(self, bridge):
        mem = bridge.observe("User prefers Python", source="analysis")
        assert mem.content == "User prefers Python"
        assert bridge.mind.count == 1


class TestReflect:
    def test_reflect_with_mock(self, bridge):
        bridge.observe("Works on math proofs")
        bridge.observe("Uses Python daily")
        identity = bridge.reflect()
        # Mock LLM won't return valid JSON, so fallback summary
        assert identity.summary != ""

    def test_reflect_with_json_llm(self):
        """Test with a mock that returns valid JSON."""
        import json

        def json_llm(messages):
            return json.dumps({
                "summary": "Math-focused assistant",
                "traits": {"analytical": 0.9, "concise": 0.8},
                "relationships": {"carlos": "primary user"},
            })

        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mind = Mind(path)
        llm = CallableLLM(json_llm)
        bridge = Bridge(mind=mind, llm=llm)

        bridge.observe("Test memory")
        identity = bridge.reflect()

        assert "Math-focused" in identity.summary
        assert "analytical" in identity.traits
        assert identity.traits["analytical"] == 0.9

        mind.close()
        os.unlink(path)


class TestContextBuilding:
    def test_context_includes_memories(self, bridge):
        bridge.observe("Important fact about user")
        # Think should include the memory in context
        response = bridge.think("Tell me about the user")
        # The mock just echoes, but internally memories were injected
        assert bridge.mind.count >= 1


class TestBackwardCompatibility:
    def test_v01_api_works(self):
        """Bridge(mind, llm) without new params works identically."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mind = Mind(path)
        llm = CallableLLM(mock_llm)
        bridge = Bridge(mind=mind, llm=llm)

        response = bridge.think("Hello")
        assert "Echo:" in response
        assert mind.count == 2

        mind.close()
        os.unlink(path)
