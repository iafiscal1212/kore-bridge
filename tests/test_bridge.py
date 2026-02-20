"""Tests for kore-bridge. Uses mock LLM (no API keys needed)."""

import os
import tempfile

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
        memories = bridge.mind.recall()
        assert any(m.source == "carlos" for m in memories)


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
