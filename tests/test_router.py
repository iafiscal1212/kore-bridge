"""Tests for LLM routing (v0.2)."""

import pytest

from kore_bridge import CallableLLM, RouterProvider


def make_provider(prefix: str) -> CallableLLM:
    """Create a mock provider that prefixes responses."""
    return CallableLLM(
        lambda msgs, p=prefix: f"{p}: {msgs[-1]['content']}"
    )


class TestRouterProvider:
    def test_routes_short_to_fast(self):
        router = RouterProvider(
            providers={
                "fast": make_provider("FAST"),
                "quality": make_provider("QUALITY"),
            }
        )
        result = router.complete([{"role": "user", "content": "Hi"}])
        assert result.startswith("FAST:")
        assert router.last_route == "fast"

    def test_routes_long_to_quality(self):
        router = RouterProvider(
            providers={
                "fast": make_provider("FAST"),
                "quality": make_provider("QUALITY"),
            }
        )
        long_msg = "x" * 600
        result = router.complete([{"role": "user", "content": long_msg}])
        assert result.startswith("QUALITY:")
        assert router.last_route == "quality"

    def test_custom_route_fn(self):
        router = RouterProvider(
            providers={
                "a": make_provider("A"),
                "b": make_provider("B"),
            },
            route_fn=lambda msgs: "b",  # always route to b
        )
        result = router.complete([{"role": "user", "content": "test"}])
        assert result.startswith("B:")
        assert router.last_route == "b"

    def test_summarize_goes_to_quality(self):
        router = RouterProvider(
            providers={
                "fast": make_provider("FAST"),
                "quality": make_provider("QUALITY"),
            }
        )
        result = router.summarize("some text", "summarize this")
        assert result.startswith("QUALITY:")
        assert router.last_route == "quality"

    def test_fallback_to_default(self):
        router = RouterProvider(
            providers={"only": make_provider("ONLY")},
            route_fn=lambda msgs: "nonexistent",
        )
        result = router.complete([{"role": "user", "content": "test"}])
        assert result.startswith("ONLY:")

    def test_empty_providers_raises(self):
        with pytest.raises(ValueError):
            RouterProvider(providers={})

    def test_last_route_initially_none(self):
        router = RouterProvider(providers={"a": make_provider("A")})
        assert router.last_route is None
