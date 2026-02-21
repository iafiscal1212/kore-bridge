"""Tests for LLM routing (v0.2)."""

import pytest

from kore_bridge import CallableLLM, RouterProvider, SCRouterProvider


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


class TestSCRouterProvider:
    """Tests for SC-based routing. Requires sc-router installed."""

    @pytest.fixture
    def catalog(self):
        try:
            from sc_router import ToolCatalog, Tool
        except ImportError:
            pytest.skip("sc-router not installed")
        cat = ToolCatalog()
        cat.register(Tool(
            name="calculator",
            description="Perform arithmetic calculations",
            input_types={"expression"},
            output_types={"number"},
            capability_tags={"math", "calculate", "arithmetic"},
        ))
        cat.register(Tool(
            name="search",
            description="Search the web for information",
            input_types={"query"},
            output_types={"results"},
            capability_tags={"search", "web", "find", "lookup"},
        ))
        return cat

    def test_sc_routes_simple_query(self, catalog):
        router = SCRouterProvider(
            providers={
                "fast": make_provider("FAST"),
                "quality": make_provider("QUALITY"),
            },
            catalog=catalog,
        )
        result = router.complete([{"role": "user", "content": "What is 2+2?"}])
        assert router.last_sc_level is not None
        assert router.last_sc_level in (0, 1, 2, 3)
        assert router.last_route in ("fast", "quality")

    def test_sc_custom_mapping(self, catalog):
        custom = {0: "cheap", 1: "cheap", 2: "expensive", 3: "expensive"}
        router = SCRouterProvider(
            providers={
                "cheap": make_provider("CHEAP"),
                "expensive": make_provider("EXPENSIVE"),
            },
            catalog=catalog,
            sc_mapping=custom,
        )
        router.complete([{"role": "user", "content": "Hi"}])
        assert router.last_route in ("cheap", "expensive")

    def test_sc_classification_available(self, catalog):
        router = SCRouterProvider(
            providers={"fast": make_provider("F"), "quality": make_provider("Q")},
            catalog=catalog,
        )
        router.complete([{"role": "user", "content": "Calculate 5*3"}])
        assert router.last_classification is not None

    def test_sc_summarize_goes_to_quality(self, catalog):
        router = SCRouterProvider(
            providers={
                "fast": make_provider("FAST"),
                "quality": make_provider("QUALITY"),
            },
            catalog=catalog,
        )
        result = router.summarize("text", "summarize")
        assert result.startswith("QUALITY:")
